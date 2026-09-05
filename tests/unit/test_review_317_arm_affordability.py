"""Review #317 red phase (Fable): arm affordability projection +
deterministic max_tokens truncations.

The live release run's Sonnet arm DNF'd at 5/94 items after burning
~$0.83 that a pre-flight projection would have refused for $0: the
Haiku arm's measured token geometry was already in hand, and priced at
Sonnet rates it showed the arm could never fit the remaining session
budget. Separately, qa-sp-06 hit the production 1024-token output limit
on Sonnet, was refused by the transcript-fidelity guard WITHOUT being
journalled, and therefore re-ran into the identical deterministic
truncation on every resume — unbounded double spend, its real usage
never captured (the ledger row was marked ESTIMATED).

Pinned here (issue #317's three required tests):

1. Before a non-first arm spends anything, its FULL cost is projected
   from the reference arm's journalled actuals priced at the arm's own
   rates (evals/pricing.py); when projection + spent would cross the
   cap, the WHOLE arm is refused BEFORE any billed call and recorded in
   the results payload as an arm verdict (FLAGGED pins:
   ``arm_verdict == "dnf-unaffordable"`` with a ``reason`` naming
   affordability) — never silently dropped, never run item-by-item into
   the wall.
2. A stream stopped for ``stop_reason: max_tokens`` is journalled as a
   SCOREABLE failed item: ``refused=False``, ``truncated=True``
   (FLAGGED field-name pin), real usage captured from the stream's
   ``message_delta`` (which arrives before the terminal error event),
   validation degraded fail-closed (its factual sentences pool with
   zero supported), and resume does NOT re-run it.
3. The ledger carries the truncated attempt's EXACT usage exactly once
   across runs — never an ESTIMATED row, never twice.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import harness, ledger, pricing
from evals.harness import (
    AnswerPathDeps,
    GoldSets,
    RunJournal,
    load_and_validate_gold,
    run_answer_path,
)
from rag.provider import FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    production_passage_payload,
    write_synthetic_gold,
)

HAIKU_ARM = "claude-haiku-4-5"
SONNET_ARM = "claude-sonnet-5"

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

REFUSAL = HonestRefusal(
    refusal_text="The synthetic corpus does not cover this.",
    covered_topics=("synthetic warming",),
    top_score=0.05,
    threshold=0.4,
)

FIRST_SENTENCE = "The synthetic planet warmed 1.1 degrees."


def _cited_stream(input_tokens: int, output_tokens: int) -> list[dict[str, Any]]:
    """One complete cited answer whose transport reports the given usage."""
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": input_tokens}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": FIRST_SENTENCE}},
        {
            "type": "content_block_delta",
            "delta": {
                "type": "citations_delta",
                "citation": {"cited_text": "synthetic passage", "document_index": 0},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]


#: The deterministic max_tokens truncation: transport-complete (the API
#: finished the turn), answer cut mid-claim, usage reported in
#: message_delta BEFORE answer_stream_to_sse's terminal error event.
TRUNCATED_MAX_TOKENS_STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": FIRST_SENTENCE}},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens"},
        "usage": {"output_tokens": 1024},
    },
    {"type": "message_stop"},
]


def _synthetic_gold(tmp_path: Path) -> GoldSets:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _deps_factory(gold: GoldSets, streams_by_arm: dict[str, list[list[dict[str, Any]]]]):
    """Per-arm programmed adapters, recorded so the tests can assert an
    arm was refused before ANY billed call."""
    adapters: dict[str, FakeAdapter] = {}

    def deps_factory(arm_model: str) -> AnswerPathDeps:
        adapter = FakeAdapter(
            generate_stream_results=list(streams_by_arm[arm_model]),
            structured_results=[
                {"scope": "in_scope", "rewritten_query": item["question"]} for item in gold.qa_items
            ],
        )
        adapters[arm_model] = adapter

        def retrieve(decision):
            query = decision.retrieval_query or ""
            if "unanswerable" in query or "out-of-scope" in query:
                return REFUSAL
            return PASSAGES

        return AnswerPathDeps(adapter=adapter, retrieve=retrieve)

    return adapters, deps_factory


def _generation_rows(ledger_path: Path) -> list[dict[str, str]]:
    return [row for row in ledger.read_rows(ledger_path) if "generation" in row.get("activity", "")]


# ---------------------------------------------------------------------------
# 1. The arm-affordability projection refuses a doomed arm for $0.
# ---------------------------------------------------------------------------


def test_unaffordable_arm_is_refused_before_any_billed_call(tmp_path: Path):
    """The Haiku arm's measured geometry (3 answered items at 1M in /
    200K out each ⇒ $3.00 batched) projects the Sonnet arm at ~$9.00 —
    remaining budget under the $9.00 cap cannot fit it, even though the
    static planned-calls estimator (~$0.06) sails through. The Sonnet
    arm must be refused BEFORE any adapter call or batch submission and
    recorded as a DNF-unaffordable arm verdict, with the Haiku arm's
    results intact."""
    gold = _synthetic_gold(tmp_path)
    ledger_path = tmp_path / "spend-ledger.csv"
    big = _cited_stream(1_000_000, 200_000)
    small = _cited_stream(120, 80)
    adapters, deps_factory = _deps_factory(
        gold,
        {HAIKU_ARM: [big, big, big], SONNET_ARM: [small, small, small]},
    )

    batch_client = FakeBatchClient()
    payload = harness.run_release_eval(
        gold,
        arm_models=(HAIKU_ARM, SONNET_ARM),
        deps_factory=deps_factory,
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=batch_client,
        mode="recording",
        ledger_path=ledger_path,
        journal_dir=tmp_path / "journals",
        session_id="review-317-affordability",
    )

    # Refused BEFORE any billed call: no adapter traffic, no judge batch,
    # no ledger row for the Sonnet arm.
    sonnet_adapter = adapters.get(SONNET_ARM)
    assert sonnet_adapter is None or sonnet_adapter.calls == [], (
        "the DNF-unaffordable refusal must fire before the arm's first "
        "adapter call — the live Sonnet arm burned ~$0.83 for 5/94 items"
    )
    assert len(batch_client.create_calls) == 1, (
        "only the Haiku arm's judge batch may exist; an unaffordable arm submits nothing"
    )
    generation_rows = _generation_rows(ledger_path)
    assert [row["model"] for row in generation_rows] == [HAIKU_ARM], (
        "no generation spend row may exist for the refused arm"
    )
    haiku_generation_cost = pricing.estimate_cost_usd(
        HAIKU_ARM, input_tokens=3_000_000, output_tokens=600_000, mode="batch"
    )
    assert float(generation_rows[0]["cost_usd"]) == pytest.approx(haiku_generation_cost)

    # Recorded, never silently dropped: the payload carries the arm verdict.
    entries = {entry["model"]: entry for entry in payload["arms"]}
    assert HAIKU_ARM in entries and entries[HAIKU_ARM].get("gates"), (
        "the reference arm's measured results must stand"
    )
    sonnet_entry = entries.get(SONNET_ARM)
    assert sonnet_entry is not None, (
        "the refused arm must appear in the results payload (DNF is a "
        "reported outcome, not a disappearance)"
    )
    assert sonnet_entry.get("arm_verdict") == "dnf-unaffordable"
    assert "afford" in (sonnet_entry.get("reason") or "").lower(), (
        "the arm verdict must say WHY: the projection from measured "
        "geometry cannot fit the remaining budget"
    )
    assert not sonnet_entry.get("gates"), "an unrun arm has no gate outcomes to report"


# ---------------------------------------------------------------------------
# 2. Deterministic truncation: journalled scoreable, fail-closed, never
#    re-run.
# ---------------------------------------------------------------------------

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic planet warming",
}


def _poisoned_validator(grounded, sse_events):
    raise AssertionError("a truncated delivery must never be validated")


def test_max_tokens_truncation_is_journalled_as_scoreable_failed_item(tmp_path: Path):
    """A max_tokens-stopped stream is deterministic: re-running it is
    pure double spend. The item must come back as a journalled, scored
    FAILURE — refused=False, truncated=True, real usage captured from
    message_delta, validation degraded fail-closed — instead of the
    unjournalled HarnessError that re-ran qa-sp-06 into the identical
    wall on every live resume."""
    from evals.gates import citation_support_gate

    adapter = FakeAdapter(
        generate_stream_results=[TRUNCATED_MAX_TOKENS_STREAM],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_poisoned_validator,
    )
    journal = RunJournal(tmp_path / "answers.jsonl")
    (result,) = run_answer_path(
        [ANSWERABLE_ITEM], deps, arm_model=HAIKU_ARM, mode="fake", journal=journal
    )

    assert result.refused is False
    assert getattr(result, "truncated", None) is True, (
        "the ItemResult must record the deterministic truncation"
    )
    assert result.answer_text == FIRST_SENTENCE
    assert result.usage == {"input_tokens": 120, "output_tokens": 1024}, (
        "message_delta delivers the REAL usage before the terminal error "
        "event — an ESTIMATED ledger row is never necessary"
    )

    # Fail-closed for the citation gate: the delivered factual sentence
    # pools with zero supported (#239's ratified degraded arithmetic).
    validation = result.validation
    assert validation is not None
    assert validation.get("validated") is False
    assert validation.get("supported") == 0
    assert validation.get("factual") == 1
    assert "truncat" in (validation.get("degraded_reason") or "").lower()
    gate = citation_support_gate([{"item_id": result.item_id, **dict(validation)}])
    assert (gate.numerator, gate.denominator) == (0, 1)

    # Journalled as done: resume makes ZERO adapter calls and returns the
    # same scored failure.
    assert journal.completed_item_ids() == frozenset({"syn-sp-01"})
    resume_adapter = FakeAdapter()
    resume_deps = AnswerPathDeps(
        adapter=resume_adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_poisoned_validator,
    )
    resumed = run_answer_path(
        [ANSWERABLE_ITEM], resume_deps, arm_model=HAIKU_ARM, mode="fake", journal=journal
    )
    assert resume_adapter.calls == [], (
        "a deterministic truncation re-runs into the identical wall: resume "
        "must skip it (the live run re-paid qa-sp-06 twice)"
    )
    assert resumed == (result,)


# ---------------------------------------------------------------------------
# 3. The ledger: exact truncated usage, once, never an ESTIMATE.
# ---------------------------------------------------------------------------


def test_truncated_usage_is_ledgered_exactly_once_and_never_estimated(tmp_path: Path):
    """One arm, three answered items, the first truncated at max_tokens:
    the generation spend rows carry the EXACT summed usage including the
    truncated attempt (360 in / 1184 out), no row is marked as an
    estimate, and a resumed run adds nothing."""
    gold = _synthetic_gold(tmp_path)
    ledger_path = tmp_path / "spend-ledger.csv"
    normal = _cited_stream(120, 80)
    adapters, deps_factory = _deps_factory(
        gold, {HAIKU_ARM: [TRUNCATED_MAX_TOKENS_STREAM, normal, normal]}
    )

    def run_once() -> Any:
        return harness.run_release_eval(
            gold,
            arm_models=(HAIKU_ARM,),
            deps_factory=deps_factory,
            plan_chart=lambda request: {
                "kind": "spec",
                "spec": {"chart_id": "syn-chart", "chart_type": "line"},
            },
            batch_client=FakeBatchClient(),
            mode="recording",
            ledger_path=ledger_path,
            journal_dir=tmp_path / "journals",
            session_id="review-317-truncation-ledger",
        )

    run_once()
    generation_rows = _generation_rows(ledger_path)
    assert generation_rows, "the generation segment must be ledgered"
    assert sum(int(row["input_tokens"]) for row in generation_rows) == 360
    assert sum(int(row["output_tokens"]) for row in generation_rows) == 1184, (
        "the truncated attempt's 1024 output tokens are real spend, captured "
        "from message_delta — not an estimate, not dropped"
    )
    for row in ledger.read_rows(ledger_path):
        assert "estimat" not in ",".join(str(value) for value in row.values()).lower(), (
            f"no ledger row may carry ESTIMATED usage (live run's "
            f"'release-eval-generation-truncated-attempts' row): {row}"
        )

    run_once()  # resume: everything journalled, nothing re-run
    resumed_rows = _generation_rows(ledger_path)
    assert sum(int(row["input_tokens"]) for row in resumed_rows) == 360
    assert sum(int(row["output_tokens"]) for row in resumed_rows) == 1184, (
        "resume must not re-pay (or re-ledger) the deterministic truncation"
    )
