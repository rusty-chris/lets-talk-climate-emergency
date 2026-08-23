"""Issue #21 red phase: the answer-path runner contract.

The runner drives the REAL pipeline (classify → route → retrieve →
cited generation) through the injectable provider seam — FakeAdapter
here, ReplayAdapter for recorded shapes, Recording/live only via
explicit opt-in + a passing budget pre-flight. One gold item → one
ItemResult carrying transcript + citations + route classification;
runs are deterministic and resumable via the journal.

No test here (or anywhere under pytest) makes a live call
(IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.harness import (
    AnswerPathDeps,
    BudgetExceededError,
    BudgetPreflight,
    ChartItemResult,
    HarnessError,
    ItemResult,
    LiveRunRefusedError,
    RunJournal,
    run_answer_path,
    run_chart_path,
)
from rag.provider import AnswerWithCitations, Citation, FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

REFUSAL_ITEM = {
    "id": "syn-na-g-01",
    "category": "no_answer",
    "question": "Synthetic unanswerable question?",
    "expected_behaviour": "refusal",
    "subset": "gate",
    "expected_route": "retrieval_refusal",
}

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic planet warming",
}

ANSWER = AnswerWithCitations(
    text="The synthetic planet warmed 1.1 degrees. [1]",
    citations=(
        Citation(cited_text="synthetic passage", document_index=0, document_title="Syn doc"),
    ),
)

PASSAGES = RetrievedPassages(
    passages=(
        RerankedPassage(
            chunk_id="syn_doc:0001",
            rerank_score=0.9,
            clears_threshold=True,
            payload={
                "chunk_id": "syn_doc:0001",
                "source_type": "report",
                "text": "synthetic passage",
            },
        ),
    )
)

REFUSAL = HonestRefusal(
    refusal_text="The synthetic corpus does not cover this.",
    covered_topics=("synthetic warming",),
    top_score=0.05,
    threshold=0.4,
)


def _answerable_deps() -> tuple[FakeAdapter, AnswerPathDeps]:
    adapter = FakeAdapter(generate_results=[ANSWER], structured_results=[CLASSIFICATION_IN_SCOPE])
    return adapter, AnswerPathDeps(adapter=adapter, retrieve=lambda decision: PASSAGES)


def test_answerable_item_produces_transcript_citations_and_route():
    """One answerable gold item drives exactly one classify (structured)
    and one cited generation (generate) through the seam, and yields an
    ItemResult with route, answer, citations and retrieved chunk ids."""
    adapter, deps = _answerable_deps()
    results = run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    assert len(adapter.calls_to("structured")) == 1
    assert len(adapter.calls_to("generate")) == 1
    (result,) = results
    assert isinstance(result, ItemResult)
    assert result.item_id == "syn-sp-01"
    assert result.arm_model == "claude-haiku-4-5"
    assert result.route == "retrieval"
    assert result.refused is False
    assert result.answer_text == ANSWER.text
    assert result.citations and result.citations[0]["cited_text"] == "synthetic passage"
    assert result.retrieved_chunk_ids == ("syn_doc:0001",)
    assert result.transcript  # the judge-facing evidence trail


def test_refusal_item_makes_zero_generation_calls():
    """An HonestRefusal from retrieval short-circuits generation: the
    refusal is recorded (refused=True) with no generate call — the
    refusal gate's raw outcome."""
    adapter = FakeAdapter(structured_results=[CLASSIFICATION_IN_SCOPE])
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: REFUSAL)
    (result,) = run_answer_path([REFUSAL_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    assert adapter.calls_to("generate") == []
    assert result.refused is True
    assert result.answer_text == REFUSAL.refusal_text


def test_runner_is_deterministic_run_to_run():
    """Identical programme + identical gold ⇒ identical results
    (deterministic metrics reproducible run-to-run — issue #21
    acceptance criterion)."""
    _, deps_a = _answerable_deps()
    _, deps_b = _answerable_deps()
    first = run_answer_path([ANSWERABLE_ITEM], deps_a, arm_model="claude-haiku-4-5", mode="fake")
    second = run_answer_path([ANSWERABLE_ITEM], deps_b, arm_model="claude-haiku-4-5", mode="fake")
    assert first == second


def test_journal_makes_runs_resumable(tmp_path: Path):
    """Items already journalled are skipped with ZERO adapter calls;
    fresh items run and are journalled; the returned results cover
    both (journalled + fresh)."""
    journal_path = tmp_path / "run-journal.jsonl"
    journal = RunJournal(journal_path)
    prior = ItemResult(
        item_id="syn-na-g-01",
        arm_model="claude-haiku-4-5",
        route="retrieval",
        refused=True,
        answer_text=REFUSAL.refusal_text,
    )
    journal.record(prior)
    assert journal.completed_item_ids() == frozenset({"syn-na-g-01"})

    adapter, deps = _answerable_deps()
    results = run_answer_path(
        [REFUSAL_ITEM, ANSWERABLE_ITEM],
        deps,
        arm_model="claude-haiku-4-5",
        mode="fake",
        journal=journal,
    )
    # Only the fresh item touched the adapter: one classify + one generate.
    assert len(adapter.calls_to("structured")) == 1
    assert len(adapter.calls_to("generate")) == 1
    assert {result.item_id for result in results} == {"syn-na-g-01", "syn-sp-01"}
    # And the fresh result was journalled for the next resume.
    assert journal.completed_item_ids() == frozenset({"syn-na-g-01", "syn-sp-01"})


def test_live_modes_require_explicit_opt_in_and_preflight():
    """recording/live modes refuse without a pre-flight, and refuse on a
    failing one — BEFORE any adapter call (fail-closed, DESIGN §9)."""
    adapter = FakeAdapter()
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: PASSAGES)
    with pytest.raises(LiveRunRefusedError):
        run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="live")
    refused = BudgetPreflight(estimated_cost_usd=5.0, cumulative_usd=8.9, allowed=False)
    with pytest.raises(BudgetExceededError):
        run_answer_path(
            [ANSWERABLE_ITEM],
            deps,
            arm_model="claude-haiku-4-5",
            mode="recording",
            preflight=refused,
        )
    assert adapter.calls == []  # refusal fired before the seam was touched


def test_unknown_mode_refused():
    adapter = FakeAdapter()
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: PASSAGES)
    with pytest.raises(HarnessError) as excinfo:
        run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="yolo")
    assert "yolo" in str(excinfo.value)
    assert adapter.calls == []


def test_chart_path_skips_blocked_items_visibly():
    """The flagship (blocked on #23/#117) is returned as
    skipped_blocked with its reason — never silently dropped, never a
    pass; unblocked items go through the injected planner seam."""
    items = [
        {"id": "syn-chart-01", "request": "Plot the synthetic series", "expected": "spec"},
        {
            "id": "syn-chart-flagship",
            "request": "The synthetic flagship request",
            "expected": "refusal",
            "blocked_on": "issue-23-licence-confirmations",
            "blocked_reason": "synthetic stand-in for the #117 fixture embargo",
        },
    ]
    planned_requests: list[str] = []

    def plan_chart(request: str) -> dict[str, object]:
        planned_requests.append(request)
        return {"kind": "spec", "spec": {"chart_id": "syn-chart"}}

    results = run_chart_path(items, plan_chart)
    assert planned_requests == ["Plot the synthetic series"]
    by_id = {result.item_id: result for result in results}
    assert isinstance(by_id["syn-chart-01"], ChartItemResult)
    assert by_id["syn-chart-01"].outcome == "spec"
    flagship = by_id["syn-chart-flagship"]
    assert flagship.outcome == "skipped_blocked"
    assert flagship.blocked_reason is not None and "#117" in flagship.blocked_reason
