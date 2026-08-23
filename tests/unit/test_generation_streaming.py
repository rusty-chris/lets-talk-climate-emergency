"""Streaming contract: Anthropic stream events -> SSE sequence
(issue #12) — RED.

Unit tier, pure: `answer_stream_to_sse` is a translation function over an
event ITERABLE — no network, no adapter. The fake stream here uses the
transport event shapes spike-03 observed live (text_delta +
citations_delta content_block_delta events, usage on message_delta); the
replay-pinned variant at the bottom re-runs the ordering against a
GENUINE recorded stream once a recording session lands (the #162 rule:
hand-authored replay fixtures are never an acceptable substitute).
"""

from __future__ import annotations

import pytest

from rag.generation import (
    GenerationConfig,
    answer_stream_to_sse,
    build_generation_request,
    build_response_footer,
    stream_grounded_answer,
)
from rag.provider import (
    AnthropicAdapter,
    FakeAdapter,
    ProviderContractError,
    RawProviderResponse,
    RecordingAdapter,
    ReplayAdapter,
    canonical_request_hash,
)
from rag.retrieval import HonestRefusal
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    citation_event,
    make_refusal,
    make_retrieved,
    transport_stream_events,
)
from tests.conftest import FIXTURES_ROOT

REPLAY_FIXTURES_DIR = FIXTURES_ROOT / "replay"


def _sse(events=None):
    return list(
        answer_stream_to_sse(
            events if events is not None else transport_stream_events(),
            corpus_vintage=CORPUS_VINTAGE,
        )
    )


def test_sse_event_sequence_is_text_citation_usage_footer():
    """The pinned envelope for one short cited answer: text deltas in
    transport order, the citation event where it arrived, then usage,
    then the footer — nothing else, nothing reordered."""
    events = _sse()
    assert [e["event"] for e in events] == ["text", "text", "citation", "usage", "footer"]


def test_text_deltas_preserve_transport_order_and_content():
    events = _sse()
    texts = [e["data"]["text"] for e in events if e["event"] == "text"]
    assert texts == [
        "The basin has very likely warmed ",
        "by one point nine degrees.",
    ]


def test_citation_events_carry_resolvable_document_indices_in_order():
    """Citations stream as first-class events, in transport order, each
    carrying the document_index the UI resolves against the passages
    panel (§3.6 chip highlighting)."""
    stream = transport_stream_events()
    # Two citations, interleaved mid-stream: 0 after the first text delta,
    # 2 at the recorded position.
    stream.insert(3, citation_event(document_index=2))
    events = _sse(stream)
    citation_indices = [e["data"]["document_index"] for e in events if e["event"] == "citation"]
    assert citation_indices == [2, 0]
    for e in events:
        if e["event"] == "citation":
            assert e["data"]["cited_text"]


def test_usage_event_carries_cache_read_metadata():
    """The cache smoke check's observable (`cache_read_input_tokens`)
    flows through to the SSE usage event, so the service layer (#22) can
    log cache effectiveness per response."""
    events = _sse()
    (usage_event,) = [e for e in events if e["event"] == "usage"]
    assert usage_event["data"]["cache_read_input_tokens"] == 4200
    assert usage_event["data"]["output_tokens"] == 42


def test_footer_event_is_last_and_carries_corpus_vintage():
    """§3.5/§10: the footer (verification note + corpus vintage) is
    appended AFTER the stream finishes — always the final event."""
    events = _sse()
    footer = events[-1]
    assert footer["event"] == "footer"
    assert footer["data"]["text"] == build_response_footer(CORPUS_VINTAGE)
    assert CORPUS_VINTAGE in footer["data"]["text"]


def test_stream_without_citations_still_closes_with_usage_and_footer():
    """An uncited stream (it can happen — entailment is measured, not
    guaranteed) still yields a well-formed close: usage then footer."""
    stream = [
        e for e in transport_stream_events() if e.get("delta", {}).get("type") != "citations_delta"
    ]
    events = _sse(stream)
    assert [e["event"] for e in events] == ["text", "text", "usage", "footer"]


# ---------------------------------------------------------------------------
# The streaming seam: ProviderAdapter.generate_stream (finding #183)
# ---------------------------------------------------------------------------

QUESTION = "How much has the invented basin warmed?"


def _built_request(count: int = 3) -> dict:
    return build_generation_request(make_retrieved(count), QUESTION, config=GenerationConfig())


class TestGenerateStreamSeam:
    def test_adapter_generate_stream_yields_transport_events_in_order(self):
        """Finding #183: the seam exposes a real event-stream method.
        FakeAdapter serves the programmed sequence in order, and the call
        is recorded like any other seam call."""
        programmed = transport_stream_events()
        fake = FakeAdapter()
        fake.queue("generate_stream", programmed)
        request = _built_request()
        events = list(fake.generate_stream(**request))
        assert events == programmed
        calls = fake.calls_to("generate_stream")
        assert len(calls) == 1
        assert calls[0].payload["documents"] == list(request["documents"])

    def test_generate_stream_validates_request_with_zero_transport_touches(self):
        """The §3.4 backstop runs on the streaming path identically: a
        violating request raises at the seam BEFORE any event is served —
        the FakeAdapter call log is the instrument (finding #62 held open
        on the streaming path)."""
        fake = FakeAdapter()
        fake.queue("generate_stream", transport_stream_events())
        request = _built_request()
        corrupt = dict(request)
        corrupt["config"] = {**request["config"], "output_schema": {"type": "object"}}
        with pytest.raises(ProviderContractError):
            fake.generate_stream(**corrupt)
        assert fake.calls == []

    def test_generate_stream_replay_serves_raw_provider_response_events(self, tmp_path):
        """A RawProviderResponse fixture's `events` replay through the
        stream seam — the envelope built for exactly this (its own
        docstring). Round-trip: RecordingAdapter tees the inner stream
        into the fixture; ReplayAdapter serves the same events."""
        programmed = transport_stream_events()
        inner = FakeAdapter()
        inner.queue("generate_stream", programmed)
        recorder = RecordingAdapter(
            inner,
            tmp_path,
            env={"CLIMATE_CHAT_RECORD": "1", "ANTHROPIC_API_KEY": "SYNTHETIC-NOT-A-KEY"},
        )
        request = _built_request()
        recorded_events = list(recorder.generate_stream(**request))
        assert recorded_events == programmed

        replayed = list(ReplayAdapter(tmp_path).generate_stream(**request))
        assert replayed == programmed

    def test_fake_adapter_serves_raw_provider_response_directly(self):
        """FakeAdapter also accepts a programmed RawProviderResponse and
        serves its event sequence — the same envelope the replay path
        consumes, so tests can share fixtures across both."""
        raw = RawProviderResponse(payload={}, events=tuple(transport_stream_events()))
        fake = FakeAdapter()
        fake.queue("generate_stream", raw)
        assert list(fake.generate_stream(**_built_request())) == transport_stream_events()

    def test_anthropic_adapter_generate_stream_yields_sdk_event_dumps(self):
        """The live adapter's streaming path: same validated request
        builder, SDK events yielded as their model_dump() mappings, in
        SDK order — and the folded `generate` is implemented over the
        SAME stream, so the two paths cannot drift."""

        class _FakeSDKEvent:
            def __init__(self, payload):
                self._payload = payload

            def model_dump(self):
                return dict(self._payload)

        class _FakeMessages:
            def __init__(self, events, log):
                self._events = events
                self._log = log

            def create(self, **kwargs):
                self._log.append(kwargs)
                assert kwargs.get("stream") is True
                return iter(_FakeSDKEvent(e) for e in self._events)

        class _FakeSDKClient:
            def __init__(self, events):
                self.create_calls: list = []
                self.messages = _FakeMessages(events, self.create_calls)

        client = _FakeSDKClient(transport_stream_events())
        adapter = AnthropicAdapter(client=client)
        request = _built_request()
        events = list(adapter.generate_stream(**request))
        assert events == transport_stream_events()
        assert len(client.create_calls) == 1
        # The request on the wire is the validated pure builder's output.
        assert client.create_calls[0]["model"] == request["config"]["model"]

        # The folded convenience folds the same event vocabulary.
        client2 = _FakeSDKClient(transport_stream_events())
        answer = AnthropicAdapter(client=client2).generate(**request)
        assert answer.text == "The basin has very likely warmed by one point nine degrees."
        assert answer.citations and answer.citations[0].document_index == 0

    def test_anthropic_adapter_generate_stream_validates_before_transport(self):
        """A §3.4-violating request raises the contract error on the
        streaming path with the transport untouched."""

        class _ExplodingClient:
            def __getattr__(self, name):
                raise AssertionError("transport touched on a contract-violating request")

        adapter = AnthropicAdapter(client=_ExplodingClient())
        request = _built_request()
        corrupt = dict(request)
        corrupt["documents"] = list(request["documents"]) * 5
        with pytest.raises(ProviderContractError):
            adapter.generate_stream(**corrupt)

    def test_anthropic_adapter_keyless_generate_stream_raises_before_network(self, monkeypatch):
        from rag.provider import AnthropicKeyMissingError

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        adapter = AnthropicAdapter()
        with pytest.raises(AnthropicKeyMissingError):
            adapter.generate_stream(**_built_request())


class TestStreamGroundedAnswer:
    """The thin streamed twin of generate_grounded_answer — the surface
    #22's SSE endpoint consumes (finding #183)."""

    def test_streams_sse_events_from_one_generate_stream_call(self):
        fake = FakeAdapter()
        fake.queue("generate_stream", transport_stream_events())
        retrieved = make_retrieved(3)
        events = list(
            stream_grounded_answer(
                fake,
                retrieved,
                QUESTION,
                config=GenerationConfig(),
                corpus_vintage=CORPUS_VINTAGE,
            )
        )
        assert [e["event"] for e in events] == ["text", "text", "citation", "usage", "footer"]
        calls = fake.calls_to("generate_stream")
        assert len(calls) == 1
        expected = build_generation_request(retrieved, QUESTION, config=GenerationConfig())
        assert calls[0].payload["documents"] == list(expected["documents"])
        assert calls[0].payload["system"] == list(expected["system"])

    def test_honest_refusal_bypasses_the_stream_entirely(self):
        """§3.5 on the streaming path too: a refusal makes ZERO adapter
        calls and comes back unchanged."""
        fake = FakeAdapter()  # unprogrammed: ANY call would raise
        refusal = make_refusal()
        result = stream_grounded_answer(
            fake,
            refusal,
            QUESTION,
            config=GenerationConfig(),
            corpus_vintage=CORPUS_VINTAGE,
        )
        assert isinstance(result, HonestRefusal)
        assert result == refusal
        assert fake.calls == []

    def test_contract_violation_raises_before_the_adapter_is_touched(self):
        from rag.generation import GenerationContractError
        from rag.retrieval import GENERATION_TOP_K

        fake = FakeAdapter()
        with pytest.raises(GenerationContractError):
            stream_grounded_answer(
                fake,
                make_retrieved(GENERATION_TOP_K + 1),
                QUESTION,
                config=GenerationConfig(),
                corpus_vintage=CORPUS_VINTAGE,
            )
        assert fake.calls == []


# ---------------------------------------------------------------------------
# Replay pin (issue #12 TDD plan item 7) — awaiting a recording session
# ---------------------------------------------------------------------------


def _generation_stream_recorded() -> bool:
    """True when a genuine recorded generate_stream exchange exists for
    the canonical fixture request. Guarded: a hand-authored fixture is
    never a substitute for a recording (#67 rule / #162 pattern)."""
    try:
        request = build_generation_request(
            make_retrieved(3),
            QUESTION,
            config=GenerationConfig(),
        )
    except NotImplementedError:
        return False
    fixture = REPLAY_FIXTURES_DIR / f"{canonical_request_hash('generate_stream', request)}.json"
    return fixture.is_file()


@pytest.mark.skipif(
    not _generation_stream_recorded(),
    reason="awaiting a generation recording session — the #162 pattern; "
    "record via CLIMATE_CHAT_RECORD=1 with a live key",
)
def test_streaming_citation_events_ordered_against_recorded_stream():
    """TDD plan item 7, made honest by finding #183: the recorded
    transport stream replays through the SSE translation — text and
    citation events in recorded transport order, the recorded usage on
    the usage event, the footer last."""
    request = build_generation_request(
        make_retrieved(3),
        QUESTION,
        config=GenerationConfig(),
    )
    adapter = ReplayAdapter(REPLAY_FIXTURES_DIR)
    recorded = list(adapter.generate_stream(**request))
    sse = list(answer_stream_to_sse(recorded, corpus_vintage=CORPUS_VINTAGE))

    # Transport-order pins, derived from the recording itself.
    recorded_texts = [
        e["delta"]["text"]
        for e in recorded
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    recorded_citation_indices = [
        e["delta"]["citation"]["document_index"]
        for e in recorded
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "citations_delta"
    ]
    assert recorded_citation_indices, "recorded exchange carries no citations"

    assert [e["data"]["text"] for e in sse if e["event"] == "text"] == recorded_texts
    assert [
        e["data"]["document_index"] for e in sse if e["event"] == "citation"
    ] == recorded_citation_indices
    (usage_event,) = [e for e in sse if e["event"] == "usage"]
    assert usage_event["data"]["output_tokens"] > 0
    assert sse[-1]["event"] == "footer"
