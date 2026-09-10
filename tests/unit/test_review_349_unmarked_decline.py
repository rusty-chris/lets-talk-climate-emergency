"""Issue #349 red phase (Fable, run-4 blocker class): the UNMARKED decline.

Release run 4 (journals in the release-run-4 branch's data directory):
qa-sev-03 ("Wouldn't a couple of degrees warmer just mean nicer summers?")
emitted decline prose — "The passages supplied don't address …" — WITHOUT
the authoritative #313 ``[[NO-ANSWER-DECLINE]]`` marker and with ZERO
citations. Because the marker is the authoritative signal, the exchange
counted as ANSWERED: it violated the ≥1-entailed-citation invariant
(failing the citation_invariants gate), dodged refusal accounting, and —
had it happened in production — would have been a cacheable "answer".
0/3 reproducible on probe (answered 3/3, retrieval healthy at 0.84): a
generation sampling flake in marker discipline, i.e. a permanent hazard
class, not a one-off. Two prongs, per the issue:

1. PROMPT HARDENING — the marker instruction's prominence and
   unconditionality in the committed generation system prompt: the
   decline rule must state the CONVERSE (decline prose without the marker
   is itself a violation) and the hard-limits summary must carry the
   marker rule, so marker discipline is a stated hard limit rather than a
   mid-document detail.
2. A FAIL-CLOSED PROSE-SHAPE FALLBACK — category-independent (unlike the
   #312 heuristic, which is scoped to no_answer golds): an answered
   exchange with ZERO citations whose text matches the decline shape
   classifies as a decline everywhere the marker would — the service SSE
   route (refusal answer, no sources, no cache admission, honest logging)
   and the eval runner (``generation_decline`` on the validation record,
   feeding refusal accounting and the citation gates' decline exemption).

FLAGGED CONTRACT (test-author decisions):

- The shape seam is ONE pure function,
  ``rag.generation.matches_decline_prose_shape(answer_text) -> bool``,
  shared by the service and the harness (the #313 lesson: one constant,
  one classifier, no drift). The shape is an OPENING-SENTENCE property —
  the supplied-passages subject plus a negated answer/address/cover verb —
  so factual openings, the prompt's partial-support openings, and
  decline-ish boundary sentences later in an answered text never match.
- PRECEDENCE: the #313 marker stays PRIMARY and the #312 no_answer
  heuristic stays intact; the shape fallback fires ONLY on answered
  exchanges with zero citations. An exchange that carries citations is
  NEVER reclassified, whatever its text says.
- The SSE wire for a fallback decline mirrors the #313 marked-decline
  wire exactly ([meta, answer kind=refusal]; no sources/text/footer/
  badges; never cached; usage metered and logged). The decision needs
  completed text + zero citations, so a decline-shaped OPENING buffers
  until a citation event resolves it as grounded or clean completion
  confirms the decline — a cited answer that merely opens with a
  boundary sentence still streams as a grounded exchange.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient

from evals.gates import GATE_PASSED, citation_invariants_gate
from evals.harness import AnswerPathDeps, authoritative_refusal, run_answer_path
from rag.citation_validator import ValidationOutcome, segment_answer_sentences
from rag.generation import (
    GENERATION_DECLINE_MARKER,
    SYSTEM_PROMPT_PATH,
    classify_generation_decline,
    matches_decline_prose_shape,
)
from rag.provider import AnswerWithCitations, Citation, FakeAdapter
from rag.retrieval import RerankedPassage, RetrievedPassages
from service.app import ANSWER_KIND_REFUSAL, META_EVENT
from tests._eval_harness_fixtures import (
    production_passage_payload,
    transport_stream_for_answer,
)
from tests._service_fixtures import classifier_output, events_named, make_harness, post_chat
from tests.unit.test_review_313_decline_marker import LIVE_DECLINE_PROSE

ARM_MODEL = "claude-haiku-4-5"

#: The VERBATIM run-4 qa-sev-03 answer text (claude-haiku-4-5 answers
#: journal, release run 4) — the unmarked decline that slipped through as
#: an answered exchange. Pinned character-for-character: this exact text
#: is what the fallback must catch.
RUN4_QA_SEV_03_TEXT = (
    "The passages supplied don't address what a couple of degrees of warming "
    "would mean for summers or daily life. They document that warming is "
    "happening and at what rate, but they don't cover the impacts on seasonal "
    "weather, agricultural productivity, human health, extreme weather "
    "frequency, or livability — the things that would determine whether "
    'warmer temperatures felt "nicer" or harmful.\n\n'
    "The source library likely has material on climate impacts and what "
    "different warming levels mean for people and ecosystems. That's where "
    "you'd find the answer to how a couple of degrees would actually affect you."
)

#: A cited grounded answer (control): decline-free text that must never be
#: reclassified by anything in this suite.
CITED_ANSWER_TEXT = "The basin has very likely warmed by one point nine degrees."

#: The prompt's partial-support openings are ANSWERS (Rule 5's second
#: shape) — the fallback must never turn honest partial support into a
#: refusal.
PARTIAL_SUPPORT_OPENING = (
    "The passages answer the first half, and the finding is stark: regional "
    "temperatures have very likely risen by 1.9°C. They don't address "
    "projections to 2100, so I can't answer that from what I have."
)


# ---------------------------------------------------------------------------
# Prong 1 — prompt hardening: prominence + unconditionality anchors.
# Characterisation anchors, never phrasing (house rule).
# ---------------------------------------------------------------------------


def _prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


class TestPromptDeclineRuleProminence:
    def test_hard_limits_summary_carries_the_marker_rule(self) -> None:
        """PROMINENCE: the 'What you never do' hard-limits summary is the
        prompt's last word on non-negotiables, and run 4 proved marker
        discipline belongs in it — the summary must carry the marker (the
        never-decline-without-it rule), not leave it a mid-document
        Rule-5 detail a sampling flake can drop."""
        text = _prompt()
        summary_at = text.lower().find("## what you never do")
        assert summary_at != -1, "the hard-limits summary section must exist"
        assert GENERATION_DECLINE_MARKER in text[summary_at:], (
            "the hard-limits summary must state the decline-marker rule: run 4's "
            "qa-sev-03 dropped the marker from otherwise-honest decline prose, and "
            "the summary is where the prompt binds its non-negotiables"
        )

    def test_prompt_states_the_unmarked_decline_is_a_violation(self) -> None:
        """UNCONDITIONALITY: the prompt currently says how to mark a
        decline but never states the CONVERSE — that decline prose
        without the marker is itself a violation. Run 4's flake is
        exactly the unstated case: the model wrote the honest decline
        and skipped the machine line. The rule must be stated from the
        never-without side too."""
        assert re.search(
            r"unmarked|without\s+the\s+marker|without\s+\[\[NO-ANSWER-DECLINE\]\]",
            _prompt(),
            re.IGNORECASE,
        ), (
            "the prompt must state that a decline WITHOUT the marker is a "
            "violation (the converse rule), not only how a marked decline looks"
        )


# ---------------------------------------------------------------------------
# The pure shape predicate: one seam, shared by service and harness.
# ---------------------------------------------------------------------------


class TestDeclineProseShape:
    def test_run4_verbatim_unmarked_decline_matches(self) -> None:
        assert matches_decline_prose_shape(RUN4_QA_SEV_03_TEXT) is True

    def test_the_313_live_decline_prose_matches(self) -> None:
        """The other run-observed decline opening ('The passages I was
        given don't answer that question.') is the same shape."""
        assert matches_decline_prose_shape(LIVE_DECLINE_PROSE) is True

    def test_cited_answer_text_never_matches(self) -> None:
        assert matches_decline_prose_shape(CITED_ANSWER_TEXT) is False

    def test_partial_support_opening_never_matches(self) -> None:
        """Rule 5's partial-support shape is an ANSWER: 'The passages
        answer the first half …' must never classify as a decline."""
        assert matches_decline_prose_shape(PARTIAL_SUPPORT_OPENING) is False

    def test_decline_shaped_sentence_later_in_an_answer_never_matches(self) -> None:
        """The shape is an OPENING property: an answered text whose
        honest boundary sentence arrives later stays an answer."""
        text = (
            "Global surface temperature rose 1.1C between 1850 and 2020.\n\n"
            "The passages supplied don't address projections to 2100."
        )
        assert matches_decline_prose_shape(text) is False

    def test_empty_and_whitespace_only_never_match(self) -> None:
        for text in ("", "   \n\n  "):
            assert matches_decline_prose_shape(text) is False

    def test_marker_stays_primary_for_marked_declines(self) -> None:
        """PRECEDENCE: a marked decline is classified by the MARKER
        (classify_generation_decline), which strips it for display — the
        shape predicate is the fallback, not a replacement."""
        marked = GENERATION_DECLINE_MARKER + "\n" + LIVE_DECLINE_PROSE
        classified = classify_generation_decline(marked)
        assert classified.is_decline is True
        assert classified.display_text == LIVE_DECLINE_PROSE


# ---------------------------------------------------------------------------
# Prong 2a — the eval runner: shape fallback on zero-citation answered
# exchanges, category-independent; #312/#313 precedence intact.
# ---------------------------------------------------------------------------

CLASSIFICATION_IN_SCOPE = {"scope": "in_scope", "rewritten_query": "synthetic rewritten query"}

PASSAGES = RetrievedPassages(
    passages=(
        RerankedPassage(
            chunk_id="syn_doc:0001",
            rerank_score=0.9,
            clears_threshold=True,
            payload=production_passage_payload("syn_doc:0001"),
        ),
    )
)

ANSWERABLE_ITEM = {
    "id": "qa-sev-03",
    "category": "severity",
    "question": "Wouldn't a couple of degrees warmer just mean nicer summers?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

NO_ANSWER_ITEM = {
    "id": "qa-na-g-05",
    "category": "no_answer",
    "question": "Are earthquakes getting stronger because of climate change?",
    "expected_behaviour": "refusal",
    "subset": "gate",
    "expected_route": "retrieval_refusal",
}


def _segmenting_validator(grounded, sse_events):
    return ValidationOutcome(
        validated=True,
        sentences=segment_answer_sentences(sse_events),
        verdicts=(),
    )


def _run_single_item(item, answer: AnswerWithCitations):
    adapter = FakeAdapter(
        generate_stream_results=[transport_stream_for_answer(answer)],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_segmenting_validator,
    )
    (result,) = run_answer_path([item], deps, arm_model=ARM_MODEL, mode="fake")
    return result


def _uncited(text: str) -> AnswerWithCitations:
    return AnswerWithCitations(
        text=text, citations=(), usage={"input_tokens": 4188, "output_tokens": 118}
    )


class TestRunnerShapeFallback:
    def test_run4_unmarked_decline_on_answerable_item_is_flagged(self) -> None:
        """THE run-4 defect, category-independent: qa-sev-03 is an
        ANSWERABLE gold (category 'severity', not no_answer), so the #312
        heuristic never fires — the shape fallback must. The exchange is
        honest refusal evidence, not a citation-less answer."""
        result = _run_single_item(ANSWERABLE_ITEM, _uncited(RUN4_QA_SEV_03_TEXT))
        assert result.refused is False, "the pre-filter did not fire; generation declined"
        assert result.citations == ()
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True, (
            "an answered, ZERO-citation exchange whose text matches the decline "
            "shape must classify as a generation-level decline whatever the "
            "gold's category (issue #349) — run 4 counted this exact text as an "
            "answered exchange and broke the citation invariant"
        )
        assert authoritative_refusal(result) is True, (
            "the fallback decline must reach refusal accounting through the ONE "
            "#313 predicate, exactly as a marked decline would"
        )

    def test_flagged_fallback_decline_is_exempt_from_citation_invariants(self) -> None:
        """Run 4's gate-level symptom, closed: the flagged exchange is
        EXEMPT from the ≥1-entailed-citation invariant (visible in the
        evidence, excluded from the failure arithmetic) — the refusal
        gates are its scorers, exactly as for a marked decline."""
        result = _run_single_item(ANSWERABLE_ITEM, _uncited(RUN4_QA_SEV_03_TEXT))
        assert result.validation is not None
        gate = citation_invariants_gate(
            [{"item_id": result.item_id, **dict(result.validation)}],
            citation_events=[],
        )
        assert gate.status == GATE_PASSED, (
            "the run-4 unmarked decline failed citation_invariants as a "
            "citation-less ANSWER; once shape-classified as a decline it must be "
            "exempt-and-visible, like every other generation decline"
        )

    def test_shape_fallback_fires_on_no_answer_golds_too(self) -> None:
        """Category-independence, the other direction: on a no_answer
        gold the #312 heuristic already fires for ANY uncited answer —
        the shape fallback must agree, never fight it (marker OR
        heuristic OR shape; one flag)."""
        result = _run_single_item(NO_ANSWER_ITEM, _uncited(RUN4_QA_SEV_03_TEXT))
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True

    def test_cited_exchange_is_never_reclassified_by_shape(self) -> None:
        """FAIL-CLOSED BOUND: the fallback fires ONLY on zero-citation
        exchanges. The same decline-shaped opening WITH a citation is an
        answered, partially-supported exchange — reclassifying it would
        hide a support failure inside the decline exemption."""
        cited = AnswerWithCitations(
            text=RUN4_QA_SEV_03_TEXT,
            citations=(
                Citation(
                    cited_text="synthetic passage", document_index=0, document_title="Syn doc"
                ),
            ),
            usage={"input_tokens": 4188, "output_tokens": 118},
        )
        result = _run_single_item(ANSWERABLE_ITEM, cited)
        assert result.citations, "the control must actually carry its citation"
        assert result.validation is not None
        assert not result.validation.get("generation_decline"), (
            "an exchange that carries citations is NEVER reclassified by the "
            "prose-shape fallback, whatever its text says"
        )

    def test_marked_decline_still_flags_marker_primacy(self) -> None:
        """PRECEDENCE (green guard): the #313 marker keeps working
        unchanged — the fallback is additive."""
        marked = GENERATION_DECLINE_MARKER + "\n\n" + LIVE_DECLINE_PROSE
        result = _run_single_item(ANSWERABLE_ITEM, _uncited(marked))
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True

    def test_non_decline_uncited_answer_stays_a_support_failure(self) -> None:
        """The #312 pin survives verbatim: an uncited FACTUAL answer on
        an answerable item matches no shape and stays pooled fail-closed
        — the fallback must not widen into 'any uncited answer'."""
        result = _run_single_item(ANSWERABLE_ITEM, _uncited(CITED_ANSWER_TEXT))
        assert result.validation is not None
        assert not result.validation.get("generation_decline")


# ---------------------------------------------------------------------------
# Prong 2b — the service SSE seam: the fallback decline takes the exact
# #313 decline wire (route, accounting, cache exclusion).
# ---------------------------------------------------------------------------


def _stream_events(
    text_deltas: list[str], *, citation: bool = False, complete: bool = True
) -> list[dict[str, Any]]:
    """A transport stream delivering the given text deltas (optionally
    with one citation attached), mirroring the #313 suite's shape."""
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"model": ARM_MODEL, "role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    events.extend(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": delta}}
        for delta in text_deltas
    )
    if citation:
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "char_location",
                        "cited_text": "synthetic passage",
                        "document_index": 0,
                        "document_title": "Syn doc",
                        "start_char_index": 0,
                        "end_char_index": 17,
                    },
                },
            }
        )
    if complete:
        events.extend(
            [
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 4188, "output_tokens": 118},
                },
                {"type": "message_stop"},
            ]
        )
    return events


def _unmarked_decline_harness(tmp_path, **kwargs):
    harness = make_harness(tmp_path, **kwargs)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue(
        "generate_stream",
        _stream_events([RUN4_QA_SEV_03_TEXT[:120], RUN4_QA_SEV_03_TEXT[120:]]),
    )
    return harness


QUESTION = "Wouldn't a couple of degrees warmer just mean nicer summers?"


class TestServiceShapeFallbackWire:
    def test_unmarked_decline_is_meta_plus_one_refusal_answer(self, tmp_path) -> None:
        """The run-4 text over the wire: the completed, zero-citation,
        decline-shaped exchange takes the #313 decline wire — meta then
        ONE refusal answer, and nothing else (no sources, no text/
        citation deltas, no citation-trust footer, no badges)."""
        harness = _unmarked_decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), QUESTION)
        names = [event["event"] for event in events]
        assert names[0] == META_EVENT
        assert names == [META_EVENT, "answer"], (
            f"an unmarked decline must be served exactly like a marked one — got events {names}"
        )
        answer = events_named(events, "answer")[0]["data"]
        assert answer["kind"] == ANSWER_KIND_REFUSAL
        assert answer["text"] == RUN4_QA_SEV_03_TEXT, (
            "no marker to strip: the honest prose is delivered unchanged"
        )

    def test_no_sources_event_dresses_up_the_fallback_decline(self, tmp_path) -> None:
        harness = _unmarked_decline_harness(tmp_path)
        events = post_chat(TestClient(harness.app), QUESTION)
        assert events_named(events, "sources") == [], (
            "a refusal is never dressed up as grounding (the #313/#220 rule "
            "applies to the fallback decline too)"
        )

    def test_fallback_decline_accounting_matches_the_marked_decline(self, tmp_path) -> None:
        """Honest accounting: route stays 'retrieval'; the validation
        mapping carries generation_decline; citations empty; the paid
        generation usage lands on the record; no factual-sentence
        validation is driven for a decline."""
        harness = _unmarked_decline_harness(tmp_path)
        post_chat(TestClient(harness.app), QUESTION)
        (record,) = harness.exchange_log.records()
        assert record["route"] == "retrieval"
        assert record["validation"].get("generation_decline") is True, (
            "run 4 logged this exchange as an ANSWER; the shape fallback must "
            "log it as the decline it is"
        )
        assert record["citations"] == []
        assert record["retrieved_chunk_ids"], "retrieval DID serve passages; keep their ids"
        usage_models = [entry.get("model") for entry in record.get("usage_records", [])]
        assert any(usage_models), "the decline generation call was paid for: meter it"
        assert harness.validation.validate_calls == [], (
            "a decline asserts no facts: the #13 entailment validator must not run"
        )

    def test_fallback_decline_is_never_admitted_to_the_semantic_cache(self, tmp_path) -> None:
        """Run 4's worst production shadow: an unmarked decline that
        counts as a clean answered exchange is CACHEABLE — a permanently
        replayed 'answer' with zero citations. The fallback must exclude
        it exactly as the marked decline is excluded."""
        from service.semantic_cache import SemanticCache
        from tests._indexing_fixtures import HashEmbeddingModel
        from tests._service_fixtures import CORPUS_VERSION, FrozenClock

        clock = FrozenClock()
        cache = SemanticCache(
            embedding_model=HashEmbeddingModel(),
            corpus_version=CORPUS_VERSION,
            clock=clock,
        )
        harness = _unmarked_decline_harness(tmp_path, clock=clock, semantic_cache=cache)
        post_chat(TestClient(harness.app), QUESTION)

        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue(
            "generate_stream",
            _stream_events([RUN4_QA_SEV_03_TEXT]),
        )
        events = post_chat(TestClient(harness.app), QUESTION)
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert "cached" not in kinds, "a decline must never be replayed from the cache"
        assert len(harness.adapter.calls_to("generate_stream")) == 2


class TestServiceCitedExchangesNeverReclassified:
    def test_decline_shaped_opening_with_a_citation_streams_grounded(self, tmp_path) -> None:
        """GREEN GUARD for the implementation: a cited answer that merely
        OPENS with a boundary sentence resolves as grounded once its
        citation arrives — full grounded vocabulary, sources included,
        no refusal answer. The fallback's zero-citation bound is what
        keeps honest partial answers streaming."""
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue(
            "generate_stream",
            _stream_events(
                [
                    "The passages supplied don't address the second half of your question. ",
                    "The first half they do: the basin has very likely warmed by 1.9 degrees.",
                ],
                citation=True,
            ),
        )
        events = post_chat(TestClient(harness.app), "How warm, and how warm by 2100?")
        names = [event["event"] for event in events]
        assert "sources" in names
        assert "text" in names and "footer" in names
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert ANSWER_KIND_REFUSAL not in kinds

    def test_ordinary_cited_answer_wire_is_untouched(self, tmp_path) -> None:
        """GREEN GUARD: a factual cited answer streams exactly as today."""
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue(
            "generate_stream",
            _stream_events([CITED_ANSWER_TEXT], citation=True),
        )
        events = post_chat(TestClient(harness.app), "How warm is the basin?")
        names = [event["event"] for event in events]
        assert "sources" in names and "text" in names and "footer" in names
        kinds = [event["data"].get("kind") for event in events_named(events, "answer")]
        assert ANSWER_KIND_REFUSAL not in kinds
