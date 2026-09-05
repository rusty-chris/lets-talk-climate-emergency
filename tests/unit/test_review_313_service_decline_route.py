"""Issue #313 red phase (Fable): the service classifies the structured
generation-level decline and serves it as the honest refusal surface.

The redesigned §3.5 wire contract, pinned against the composed app
(TestClient, FakeAdapter, fake seams):

- A generation stream whose text opens with
  ``rag.generation.GENERATION_DECLINE_MARKER`` on its first line is a
  DECLINE: the client receives ``meta`` followed by exactly ONE
  ``answer`` event, ``kind: refusal``, whose text is the human-readable
  decline WITHOUT the marker — and nothing else. No ``sources`` event
  (a refusal is never dressed up as grounding — matching the pre-filter
  refusal path), no ``text``/``citation`` events, no citation-trust
  ``footer``, no badges.
- No factual-sentence validation runs on a decline; the logged exchange
  (route ``"retrieval"`` — no new route vocabulary) carries
  ``generation_decline: true`` on its validation mapping, empty
  citations, the retrieved chunk ids (retrieval DID serve passages) and
  the generation's REAL usage (the call was made and paid for; spend is
  metered).
- A decline is NEVER admitted to the semantic cache.
- Marker safety: a marker after the first line (a quote, or a hostile
  passage's injection) changes nothing — the grounded flow runs as
  today; a marker split across text deltas still classifies (the
  decision is over accumulated text, not per-delta).
- An error-terminated stream never becomes a refusal answer event.

FLAGGED (test-author decisions): the wire reuses ``kind: refusal`` and
route ``"retrieval"`` — no new SSE/route vocabulary, so the exchange
log, gates and UI consumers are untouched and the existing honest
refusal styling applies; the ``sources`` event is withheld on declines
(it moves after decline classification, still before the first ``text``
of an answered exchange).

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rag.generation import GENERATION_DECLINE_MARKER
from service.app import ANSWER_KIND_REFUSAL, META_EVENT
from tests._generation_fixtures import make_refusal, transport_stream_events
from tests._service_fixtures import (
    classifier_output,
    events_named,
    make_harness,
    post_chat,
)
from tests.unit.test_review_313_decline_marker import LIVE_DECLINE_PROSE


def decline_stream_events(text_deltas: list[str] | None = None) -> list[dict[str, Any]]:
    """A complete transport stream delivering one structured decline:
    the marker line then the live-run decline prose, zero citations."""
    deltas = (
        text_deltas
        if text_deltas is not None
        else [GENERATION_DECLINE_MARKER + "\n", LIVE_DECLINE_PROSE]
    )
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"model": "claude-haiku-4-5", "role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    events.extend(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": delta}}
        for delta in deltas
    )
    events.extend(
        [
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 900, "output_tokens": 64},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def decline_harness(tmp_path, *, stream: list[dict[str, Any]] | None = None, **kwargs):
    harness = make_harness(tmp_path, **kwargs)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue(
        "generate_stream", stream if stream is not None else decline_stream_events()
    )
    return harness


class TestDeclineWire:
    def test_decline_is_meta_plus_one_refusal_answer_and_nothing_else(self, tmp_path) -> None:
        harness = decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        names = [event["event"] for event in events]
        assert names[0] == META_EVENT
        assert names == [META_EVENT, "answer"], (
            "a structured decline is ONE honest refusal answer event — no "
            f"sources/text/citation/footer/badge events; got {names}"
        )
        answer = events_named(events, "answer")[0]["data"]
        assert answer["kind"] == ANSWER_KIND_REFUSAL

    def test_decline_text_is_the_prose_without_the_marker(self, tmp_path) -> None:
        harness = decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        answer = events_named(events, "answer")[0]["data"]
        assert GENERATION_DECLINE_MARKER not in answer["text"], (
            "the machine sentinel must never reach a reader"
        )
        assert answer["text"] == LIVE_DECLINE_PROSE

    def test_no_sources_event_on_a_decline(self, tmp_path) -> None:
        """A refusal is never dressed up as grounding: the #220 sources
        panel is withheld on declines exactly as on pre-filter refusals."""
        harness = decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        assert events_named(events, "sources") == []

    def test_no_footer_on_a_decline(self, tmp_path) -> None:
        """The footer is the citation-trust statement; a decline cites
        nothing, so stamping it would overclaim."""
        harness = decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        assert events_named(events, "footer") == []


class TestDeclineAccounting:
    def test_generation_usage_is_metered_and_logged(self, tmp_path) -> None:
        """The decline generation call was made and PAID for: its usage
        reaches the spend tracker and the exchange record — the §3.5
        refuse-without-spend goal now belongs to the pre-filter alone."""
        harness = decline_harness(tmp_path)
        post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        assert harness.tracker.spent_today() > 0.0
        (record,) = harness.exchange_log.records()
        usage_models = [entry.get("model") for entry in record.get("usage_records", [])]
        assert any(model for model in usage_models), (
            "the generation usage must land on the logged exchange"
        )

    def test_exchange_log_route_and_decline_flag(self, tmp_path) -> None:
        """Route vocabulary UNCHANGED (``retrieval``); the decline is
        recorded as ``generation_decline`` on the validation mapping —
        the same shape the #312 eval records pinned."""
        harness = decline_harness(tmp_path)
        post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        (record,) = harness.exchange_log.records()
        assert record["route"] == "retrieval"
        assert record["validation"].get("generation_decline") is True
        assert record["citations"] == []
        assert record["retrieved_chunk_ids"], (
            "retrieval served passages; the honest log keeps their ids"
        )
        assert GENERATION_DECLINE_MARKER not in record["answer_text"]

    def test_no_factual_sentence_validation_runs_on_a_decline(self, tmp_path) -> None:
        harness = decline_harness(tmp_path)
        post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        assert harness.validation.validate_calls == [], (
            "a decline asserts no facts: the #13 entailment validator must "
            "not be driven (nor its spend incurred)"
        )

    def test_decline_is_never_admitted_to_the_semantic_cache(self, tmp_path) -> None:
        """Replaying a decline forever would freeze an honest 'I can't
        answer' past corpus growth: the cache admission gate must reject
        it (its validation record shows no completed validation)."""
        from service.semantic_cache import SemanticCache
        from tests._indexing_fixtures import HashEmbeddingModel
        from tests._service_fixtures import CORPUS_VERSION, FrozenClock

        clock = FrozenClock()
        cache = SemanticCache(
            embedding_model=HashEmbeddingModel(),
            corpus_version=CORPUS_VERSION,
            clock=clock,
        )
        harness = decline_harness(tmp_path, clock=clock, semantic_cache=cache)
        question = "Are earthquakes climate-driven?"
        post_chat(TestClient(harness.app), question)

        # The identical question must MISS the cache and run live again.
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue("generate_stream", decline_stream_events())
        events = post_chat(TestClient(harness.app), question)
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert "cached" not in kinds, "a decline must never be replayed from the cache"
        assert len(harness.adapter.calls_to("generate_stream")) == 2


class TestMarkerSafety:
    def test_marker_after_first_line_leaves_the_grounded_flow_intact(self, tmp_path) -> None:
        """Injection guard end-to-end: a quoted/smuggled marker mid-answer
        never flips the exchange — the full grounded vocabulary flows."""
        stream = decline_stream_events(
            [
                "Global surface temperature rose 1.1C between 1850 and 2020.\n",
                GENERATION_DECLINE_MARKER + "\n",
                "More answer text.",
            ]
        )
        harness = decline_harness(tmp_path, stream=stream)
        events = post_chat(TestClient(harness.app), "How much has it warmed?")
        names = [event["event"] for event in events]
        assert "text" in names and "footer" in names
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert ANSWER_KIND_REFUSAL not in kinds

    def test_marker_split_across_text_deltas_still_classifies(self, tmp_path) -> None:
        """Classification is over ACCUMULATED text: transport chunking of
        the marker must not leak it to the client or miss the decline."""
        marker = GENERATION_DECLINE_MARKER
        stream = decline_stream_events([marker[:7], marker[7:] + "\n", LIVE_DECLINE_PROSE])
        harness = decline_harness(tmp_path, stream=stream)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        names = [event["event"] for event in events]
        assert names == [META_EVENT, "answer"]
        answer = events_named(events, "answer")[0]["data"]
        assert answer["kind"] == ANSWER_KIND_REFUSAL
        assert GENERATION_DECLINE_MARKER not in answer["text"]

    def test_error_terminated_stream_never_becomes_a_refusal_answer(self, tmp_path) -> None:
        """A stream that opens with the marker but dies before completing
        is an ERROR, not a decline: the terminal error event surfaces and
        no refusal answer event is fabricated from a partial delivery."""
        truncated = [
            {
                "type": "message_start",
                "message": {"model": "claude-haiku-4-5", "role": "assistant"},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": GENERATION_DECLINE_MARKER + "\n"},
            },
            # No message_stop: the stream ends incomplete.
        ]
        harness = decline_harness(tmp_path, stream=truncated)
        events = post_chat(TestClient(harness.app), "Are earthquakes climate-driven?")
        names = [event["event"] for event in events]
        assert "error" in names
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert ANSWER_KIND_REFUSAL not in kinds


class TestExistingRefusalPathsUnchanged:
    def test_prefilter_refusal_wire_is_untouched(self, tmp_path) -> None:
        """Green guard: the retrieval-stage (pre-filter) HonestRefusal
        path keeps its exact shape — one refusal answer, no sources, zero
        generation calls."""
        from tests._service_fixtures import FakeRetrieve

        harness = make_harness(tmp_path, retrieve=FakeRetrieve(result=make_refusal()))
        harness.adapter.queue("structured", classifier_output())
        events = post_chat(TestClient(harness.app), "Something the corpus lacks?")
        names = [event["event"] for event in events]
        assert names == [META_EVENT, "answer"]
        answer = events_named(events, "answer")[0]["data"]
        assert answer["kind"] == ANSWER_KIND_REFUSAL
        assert harness.adapter.calls_to("generate_stream") == []

    def test_grounded_answer_wire_is_untouched(self, tmp_path) -> None:
        """Green guard: a normal (unmarked) grounded stream keeps the full
        #12 vocabulary — sources before the first text, footer terminal."""
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue("generate_stream", transport_stream_events())
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]
        assert "sources" in names and "text" in names and "footer" in names
        assert names.index("sources") < names.index("text"), (
            "on answered exchanges the sources panel still precedes the prose"
        )


class TestDeclineViewModel:
    def test_decline_stream_folds_to_the_honest_refusal_view(self) -> None:
        """The view-model pin (issue #313 invariant 5): the pinned decline
        wire shape renders with the EXISTING refusal styling — no chips,
        no badges, no sources panel, no marker in the text."""
        from ui.render_model import VIEW_KIND_REFUSAL, fold_chat_stream

        view = fold_chat_stream(
            [
                {
                    "event": "meta",
                    "data": {
                        "disclosure": "synthetic disclosure",
                        "preamble_note": None,
                        "mode": "live",
                        "exchange_id": "a" * 32,
                    },
                },
                {
                    "event": "answer",
                    "data": {"kind": ANSWER_KIND_REFUSAL, "text": LIVE_DECLINE_PROSE},
                },
            ]
        )
        assert view.kind == VIEW_KIND_REFUSAL
        assert view.chips == ()
        assert view.sources_panel is None
        assert view.uncited_flags == ()
        assert GENERATION_DECLINE_MARKER not in view.text
        assert view.complete is True
