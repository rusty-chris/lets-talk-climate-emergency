"""Issue #313 red phase (Fable): the refusal/false-refusal release gates
measure the AUTHORITATIVE signal — pre-filter refusal OR structured
generation-level decline — not the demoted threshold alone.

The live run scored refusal 10/20 because the gate counted only
retrieval-stage refusals while all 10 slipped items declined honestly at
generation. Pinned here:

- ``evals.harness.authoritative_refusal``: the ONE predicate — True for
  a pre-filter/canned refusal (``result.refused``) OR a validation
  record carrying ``generation_decline``; False for clean answers and
  truncations.
- ``build_gate_battery``: the refusal gate PASSES a run whose no-answer
  gate items all declined at generation (the live shape, previously
  10/20 = failed); the false-refusal gate COUNTS a structured decline
  on an answerable item (an authoritative refusal is a refusal wherever
  it fires).
- The runner (``_drive_answer_item`` via ``run_answer_path``): the
  structured marker sets ``generation_decline`` CATEGORY-INDEPENDENTLY
  (a marked decline on an answerable gold is a false refusal, measured);
  the #312 zero-citation heuristic survives as the FALLBACK for
  unmarked declines on no_answer golds (precedence: marker OR
  heuristic); a marker after the first line sets nothing.

Gold vocabulary: UNTOUCHED — synthetic golds only, and ``expected_route:
retrieval_refusal`` keeps its name (its semantics now read "refused
honestly by the retrieval-route pipeline, at either stage").

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from typing import Any

from evals import harness
from evals.gates import GATE_FAILED, GATE_PASSED
from evals.harness import (
    AnswerPathDeps,
    GoldSets,
    ItemResult,
    authoritative_refusal,
    build_gate_battery,
    load_and_validate_gold,
    run_answer_path,
)
from rag.citation_validator import ValidationOutcome, segment_answer_sentences
from rag.generation import GENERATION_DECLINE_MARKER
from rag.provider import AnswerWithCitations, FakeAdapter
from rag.retrieval import RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    production_passage_payload,
    transport_stream_for_answer,
    write_synthetic_gold,
)
from tests.unit.test_review_313_decline_marker import LIVE_DECLINE_PROSE

ARM_MODEL = "claude-haiku-4-5"

MARKED_DECLINE_TEXT = GENERATION_DECLINE_MARKER + "\n" + LIVE_DECLINE_PROSE


def _result(**overrides: Any) -> ItemResult:
    values: dict[str, Any] = dict(
        item_id="syn-item",
        arm_model=ARM_MODEL,
        route="retrieval",
        refused=False,
        answer_text="A synthetic answer.",
    )
    values.update(overrides)
    return ItemResult(**values)


# ---------------------------------------------------------------------------
# The predicate.
# ---------------------------------------------------------------------------


class TestAuthoritativeRefusal:
    def test_prefilter_refusal_counts(self) -> None:
        assert authoritative_refusal(_result(refused=True, answer_text="template refusal")) is True

    def test_structured_decline_counts(self) -> None:
        result = _result(
            answer_text=LIVE_DECLINE_PROSE,
            validation={"generation_decline": True},
        )
        assert authoritative_refusal(result) is True

    def test_clean_answer_does_not_count(self) -> None:
        result = _result(validation={"validated": True, "supported": 2, "factual": 2})
        assert authoritative_refusal(result) is False

    def test_answer_without_validation_does_not_count(self) -> None:
        assert authoritative_refusal(_result(validation=None)) is False

    def test_truncated_answer_does_not_count(self) -> None:
        """A truncation is a FAILURE, not a refusal — counting it would
        let a max_tokens wall inflate the refusal gate."""
        result = _result(
            truncated=True,
            validation={"validated": False, "supported": 0, "factual": 3},
        )
        assert authoritative_refusal(result) is False


# ---------------------------------------------------------------------------
# The battery: the live 10/20 shape now passes; declines on answerable
# items now fail the false-refusal gate.
# ---------------------------------------------------------------------------


def _synthetic_gold(tmp_path) -> GoldSets:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _chart_records() -> dict[str, list[dict[str, Any]]]:
    return {"spec": [], "refusal": []}


def _answer_results(
    gold: GoldSets,
    *,
    decline_ids: set[str] = frozenset(),
    prefilter_refused_ids: set[str] = frozenset(),
) -> list[ItemResult]:
    """A run over the synthetic gold: ``prefilter_refused_ids`` refuse at
    retrieval, ``decline_ids`` answer with a structured decline
    (refused=False + generation_decline), everything else answers clean."""
    results: list[ItemResult] = []
    for item in gold.qa_items:
        item_id = item["id"]
        if item_id in prefilter_refused_ids:
            results.append(_result(item_id=item_id, refused=True))
        elif item_id in decline_ids:
            results.append(
                _result(
                    item_id=item_id,
                    answer_text=LIVE_DECLINE_PROSE,
                    validation={"generation_decline": True},
                )
            )
        elif item.get("category") == "no_answer":
            # Canned/unhandled no-answer items refuse at their own stage.
            results.append(_result(item_id=item_id, refused=True))
        else:
            results.append(
                _result(
                    item_id=item_id,
                    validation={"validated": True, "supported": 2, "factual": 2},
                )
            )
    return results


def _gate(battery, name: str):
    matches = [gate for gate in battery if gate.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} gate"
    return matches[0]


class TestRefusalGateMeasuresTheAuthoritativeSignal:
    def test_all_generation_declines_pass_the_refusal_gate(self, tmp_path) -> None:
        """THE live shape: every gate no-answer item slips the (demoted)
        pre-filter and declines at generation. Under #313 that is 20/20
        authoritative refusals — the gate PASSES (it scored 10/20 live
        because only ``refused`` counted)."""
        gold = _synthetic_gold(tmp_path)
        results = _answer_results(gold, decline_ids=set(gold.refusal_gate_ids))
        battery = build_gate_battery(gold, results, _chart_records())
        refusal = _gate(battery, "refusal")
        assert refusal.status == GATE_PASSED
        assert refusal.numerator == len(gold.refusal_gate_ids)

    def test_mixed_prefilter_and_decline_both_count(self, tmp_path) -> None:
        gold = _synthetic_gold(tmp_path)
        gate_ids = list(gold.refusal_gate_ids)
        half = len(gate_ids) // 2
        results = _answer_results(
            gold,
            prefilter_refused_ids=set(gate_ids[:half]),
            decline_ids=set(gate_ids[half:]),
        )
        battery = build_gate_battery(gold, results, _chart_records())
        refusal = _gate(battery, "refusal")
        assert refusal.status == GATE_PASSED
        assert refusal.numerator == len(gate_ids)

    def test_an_unrefused_undeclined_item_still_fails_the_gate(self, tmp_path) -> None:
        """The gate is not vacuous: a gate item that ANSWERED (no marker,
        no heuristic decline, citations delivered) is a genuine miss."""
        gold = _synthetic_gold(tmp_path)
        gate_ids = list(gold.refusal_gate_ids)
        answered_anyway = gate_ids[:3]
        results = []
        for result in _answer_results(gold, decline_ids=set(gate_ids[3:])):
            if result.item_id in answered_anyway:
                results.append(
                    _result(
                        item_id=result.item_id,
                        validation={"validated": True, "supported": 1, "factual": 1},
                    )
                )
            else:
                results.append(result)
        battery = build_gate_battery(gold, results, _chart_records())
        refusal = _gate(battery, "refusal")
        assert refusal.numerator == len(gate_ids) - 3
        assert refusal.status == GATE_FAILED  # 17/20 = 85% <= 90%


class TestFalseRefusalCountsDeclines:
    def test_structured_decline_on_answerable_item_is_a_false_refusal(self, tmp_path) -> None:
        """An authoritative refusal is a refusal WHEREVER it fires: a
        model declining an answerable gold must fail the <5% gate exactly
        as a pre-filter refusal would — otherwise the redesign creates a
        free path to over-refusal."""
        gold = _synthetic_gold(tmp_path)
        answerable_ids = [
            item["id"] for item in gold.qa_items if item.get("category") != "no_answer"
        ]
        declined = {answerable_ids[0]}
        results = _answer_results(gold, decline_ids=declined)
        battery = build_gate_battery(gold, results, _chart_records())
        false_refusal = _gate(battery, "false_refusal")
        assert false_refusal.numerator == 1, (
            "the generation-level decline on an answerable item must be counted as a false refusal"
        )

    def test_clean_run_has_zero_false_refusals(self, tmp_path) -> None:
        gold = _synthetic_gold(tmp_path)
        results = _answer_results(gold)
        battery = build_gate_battery(gold, results, _chart_records())
        false_refusal = _gate(battery, "false_refusal")
        assert false_refusal.numerator == 0
        assert false_refusal.status == GATE_PASSED


# ---------------------------------------------------------------------------
# The runner: marker detection, category-independent; #312 heuristic as
# fallback; injection safety.
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

NO_ANSWER_ITEM = {
    "id": "qa-na-g-05",
    "category": "no_answer",
    "question": "Are earthquakes getting stronger because of climate change?",
    "expected_behaviour": "refusal",
    "subset": "gate",
    "expected_route": "retrieval_refusal",
}

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}


def _segmenting_validator(grounded, sse_events):
    return ValidationOutcome(
        validated=True,
        sentences=segment_answer_sentences(sse_events),
        verdicts=(),
    )


def _run_single_item(item, answer_text: str):
    adapter = FakeAdapter(
        generate_stream_results=[
            transport_stream_for_answer(
                AnswerWithCitations(
                    text=answer_text,
                    citations=(),
                    usage={"input_tokens": 2400, "output_tokens": 96},
                )
            )
        ],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_segmenting_validator,
    )
    (result,) = run_answer_path([item], deps, arm_model=ARM_MODEL, mode="fake")
    return result


class TestRunnerMarkerDetection:
    def test_marked_decline_on_no_answer_item_flags_and_counts(self) -> None:
        result = _run_single_item(NO_ANSWER_ITEM, MARKED_DECLINE_TEXT)
        assert result.refused is False, "the pre-filter did not fire; the marker did"
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True
        assert authoritative_refusal(result) is True

    def test_marked_decline_on_answerable_item_flags_too(self) -> None:
        """The structured marker is CATEGORY-INDEPENDENT — unlike the #312
        heuristic. A marked decline on an answerable gold must be flagged
        so the false-refusal gate can count it (the redesign must not
        create an unmeasured over-refusal channel)."""
        result = _run_single_item(ANSWERABLE_ITEM, MARKED_DECLINE_TEXT)
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True
        assert authoritative_refusal(result) is True

    def test_unmarked_zero_citation_decline_on_no_answer_still_flags(self) -> None:
        """Precedence pin: the #312 heuristic survives as the FALLBACK for
        unmarked declines on no_answer golds — an older-style decline is
        still measured (marker OR heuristic, never marker-only)."""
        result = _run_single_item(NO_ANSWER_ITEM, LIVE_DECLINE_PROSE)
        assert result.validation is not None
        assert result.validation.get("generation_decline") is True

    def test_marker_after_first_line_never_flags(self) -> None:
        """Injection guard in the harness too: a quoted/smuggled marker in
        an otherwise-cited answer must not flip the item into a refusal."""
        text = (
            "Global surface temperature rose 1.1C between 1850 and 2020.\n"
            + GENERATION_DECLINE_MARKER
        )
        result = _run_single_item(ANSWERABLE_ITEM, text)
        assert result.validation is not None
        assert not result.validation.get("generation_decline")
        assert authoritative_refusal(result) is False


class TestBatteryStillSingleSourced:
    def test_the_predicate_is_what_the_battery_uses(self, tmp_path) -> None:
        """Anti-drift: the battery's refusal evidence must agree with
        ``authoritative_refusal`` item-for-item — one predicate, no
        second spelling (the #303 lesson applied to #313)."""
        gold = _synthetic_gold(tmp_path)
        gate_ids = list(gold.refusal_gate_ids)
        results = _answer_results(
            gold,
            prefilter_refused_ids=set(gate_ids[:5]),
            decline_ids=set(gate_ids[5:15]),
        )
        by_id = {result.item_id: result for result in results}
        battery = build_gate_battery(gold, results, _chart_records())
        refusal = _gate(battery, "refusal")
        for entry in refusal.evidence:
            item_id = entry["item_id"]
            assert entry["refused"] == authoritative_refusal(by_id[item_id]), (
                f"gate evidence for {item_id} disagrees with authoritative_refusal"
            )


def test_module_marker_constants_are_shared() -> None:
    """The harness detector and the service classifier must read the SAME
    marker constant — pinned by import identity, not string copies."""
    import rag.generation as generation

    assert harness is not None  # imported at top: the runner module exists
    assert generation.GENERATION_DECLINE_MARKER == GENERATION_DECLINE_MARKER
