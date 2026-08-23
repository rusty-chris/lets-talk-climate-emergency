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
)
from rag.provider import ReplayAdapter, canonical_request_hash
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    citation_event,
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
# Replay pin (issue #12 TDD plan item 7) — awaiting a recording session
# ---------------------------------------------------------------------------


def _generation_replay_recorded() -> bool:
    """True when a genuine recorded generate exchange exists for the
    canonical fixture request. Guarded: at red the builder itself is
    unimplemented, and a hand-authored fixture is never a substitute for
    a recording (#67 rule / #162 pattern)."""
    try:
        request = build_generation_request(
            make_retrieved(3),
            "How much has the invented basin warmed?",
            config=GenerationConfig(),
        )
    except NotImplementedError:
        return False
    return (REPLAY_FIXTURES_DIR / f"{canonical_request_hash('generate', request)}.json").is_file()


@pytest.mark.skipif(
    not _generation_replay_recorded(),
    reason="awaiting a generation recording session — the #162 pattern; "
    "record via CLIMATE_CHAT_RECORD=1 with a live key",
)
def test_streaming_citation_events_ordered_against_recorded_stream():
    """TDD plan item 7: the recorded transport stream replays through the
    SSE translation with text and citation events in recorded order and
    the recorded usage on the usage event."""
    request = build_generation_request(
        make_retrieved(3),
        "How much has the invented basin warmed?",
        config=GenerationConfig(),
    )
    adapter = ReplayAdapter(REPLAY_FIXTURES_DIR)
    answer = adapter.generate(**request)
    assert answer.citations, "recorded exchange carries no citations"
    assert answer.usage is not None
