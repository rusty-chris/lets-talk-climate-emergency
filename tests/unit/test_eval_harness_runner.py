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

import json
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
from rag.generation import (
    GENERATION_MAX_TOKENS_DEFAULT,
    TONE_INSTRUCTION,
    GenerationConfig,
    GenerationContractError,
    build_generation_request,
    load_system_prompt,
)
from rag.provider import AnswerWithCitations, Citation, FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    production_passage_payload,
    transport_stream_for_answer,
)

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

#: The streamed delivery of ANSWER: the runner drives the production
#: streamed seam (adapter.generate_stream) so validation is fed the true
#: SSE transcript (#303 ratification note 6).
ANSWER_STREAM = transport_stream_for_answer(ANSWER)

# Production-shape payload (issue #234): the eval fixtures carry the exact
# stored payload shape build_index writes, so the PRODUCTION request builder
# (rag.generation.build_generation_request) works against them — the builder
# is never weakened, and the eval never forks a thinner document shape.
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


def _answerable_deps() -> tuple[FakeAdapter, AnswerPathDeps]:
    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    return adapter, AnswerPathDeps(adapter=adapter, retrieve=lambda decision: PASSAGES)


def test_answerable_item_produces_transcript_citations_and_route():
    """One answerable gold item drives exactly one classify (structured)
    and one streamed cited generation (generate_stream) through the
    seam, and yields an ItemResult with route, answer, citations and
    retrieved chunk ids."""
    adapter, deps = _answerable_deps()
    results = run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    assert len(adapter.calls_to("structured")) == 1
    assert len(adapter.calls_to("generate_stream")) == 1
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
    assert adapter.calls_to("generate_stream") == []
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
    # Only the fresh item touched the adapter: one classify + one stream.
    assert len(adapter.calls_to("structured")) == 1
    assert len(adapter.calls_to("generate_stream")) == 1
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


# ---------------------------------------------------------------------------
# review-21 #234 (blocker): the eval answer path measures the PRODUCTION
# request shape — it routes through rag.generation.build_generation_request,
# never a hand-rolled parallel builder.
# ---------------------------------------------------------------------------


def test_eval_generation_request_is_built_by_production_builder():
    """The generate call the eval makes IS build_generation_request's
    output, field for field: the committed system prompt on the system
    channel, the ORIGINAL user question (not the classifier rewrite),
    titled document blocks whose context header carries source_type and
    consensus_position, and citations enabled — otherwise every
    judge-based gate scores a prompt production never sends (#234)."""
    adapter, deps = _answerable_deps()
    run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    (call,) = adapter.calls_to("generate_stream")

    expected = build_generation_request(
        PASSAGES,
        ANSWERABLE_ITEM["question"],
        config=GenerationConfig(model="claude-haiku-4-5", max_tokens=GENERATION_MAX_TOKENS_DEFAULT),
    )
    assert call.payload == expected

    # The load-bearing dimensions, asserted by name so a failure reads well:
    assert call.payload["system"][0]["text"] == load_system_prompt()
    user_text = call.payload["messages"][0]["content"][0]["text"]
    assert user_text == "How warm is the synthetic planet?", (
        "production sends the ORIGINAL question; the eval must not substitute "
        "the classifier's rewritten retrieval query"
    )
    (document,) = call.payload["documents"]
    payload = PASSAGES.passages[0].payload
    assert document["title"] == payload["citation_metadata"]["attribution_text"]
    assert "source_type: report" in document["context"]
    assert "consensus_position: assessed" in document["context"]
    assert document["citations"] == {"enabled": True}


def test_eval_adversarial_item_carries_tone_instruction():
    """A tone-flagged retrieval result adds the TONE_INSTRUCTION volatile
    block after the static system prefix — exactly as production does for
    adversarial-routed queries (#234); the adversarial-rubric gate scores
    the prompted model, not an unprompted one."""
    tone_passages = RetrievedPassages(passages=PASSAGES.passages, tone_flag=True)
    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: tone_passages)
    run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    (call,) = adapter.calls_to("generate_stream")
    system = call.payload["system"]
    assert system[0]["text"] == load_system_prompt()
    assert system[1]["text"] == TONE_INSTRUCTION


def test_eval_generation_refuses_more_than_top_k_documents():
    """The §3.4 ≤8-document bound is enforced by the production builder
    BEFORE any generate call exists — nine passages raise
    GenerationContractError with zero generation calls (#234)."""
    nine = RetrievedPassages(
        passages=tuple(
            RerankedPassage(
                chunk_id=f"syn_doc:{index:04d}",
                rerank_score=0.9 - index * 0.01,
                clears_threshold=True,
                payload=production_passage_payload(f"syn_doc:{index:04d}"),
            )
            for index in range(9)
        )
    )
    adapter = FakeAdapter(structured_results=[CLASSIFICATION_IN_SCOPE])
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: nine)
    with pytest.raises(GenerationContractError):
        run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")
    assert adapter.calls_to("generate_stream") == []


def test_eval_arm_models_use_explicit_no_op_budget_guard():
    """Non-default bake-off arms go through the builder's best-mode
    policy — GenerationConfig(best_mode_enabled=True) with the harness's
    NAMED no-op budget guard — never around it (#234, pre-endorsed
    re-alignment path)."""
    from evals import harness

    guard = getattr(harness, "eval_no_budget_guard", None)
    assert guard is not None, (
        "evals.harness.eval_no_budget_guard: the eval's explicit, named no-op "
        "budget guard for non-default arms (the tier that genuinely wants no "
        "budget behaviour passes a guard whose name says so — ADR-015)"
    )
    assert guard("claude-opus-4-8") is None  # a no-op: never refuses

    # Opus escalation arm: the recorded request equals the builder's output
    # under best mode + the no-op guard.
    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: PASSAGES)
    run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-opus-4-8", mode="fake")
    (call,) = adapter.calls_to("generate_stream")
    expected = build_generation_request(
        PASSAGES,
        ANSWERABLE_ITEM["question"],
        config=GenerationConfig(
            model="claude-opus-4-8",
            max_tokens=GENERATION_MAX_TOKENS_DEFAULT,
            best_mode_enabled=True,
            budget_guard=guard,
        ),
    )
    assert call.payload == expected

    # Sonnet bake-off arm likewise routes through the builder: system
    # channel present, arm model kept. (Flagged: claude-sonnet-5 is not in
    # rag.generation.ALLOWED_GENERATION_MODEL_FAMILIES today — the bake-off
    # requires the builder to accept it as a gated non-default family.)
    sonnet_adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    sonnet_deps = AnswerPathDeps(adapter=sonnet_adapter, retrieve=lambda decision: PASSAGES)
    run_answer_path([ANSWERABLE_ITEM], sonnet_deps, arm_model="claude-sonnet-5", mode="fake")
    (sonnet_call,) = sonnet_adapter.calls_to("generate_stream")
    assert sonnet_call.payload["config"]["model"] == "claude-sonnet-5"
    assert sonnet_call.payload["system"][0]["text"] == load_system_prompt()


# ---------------------------------------------------------------------------
# review-21 #235 (blocker): journal resume is arm-aware — a shared journal
# can never return one arm's answers as another's.
# ---------------------------------------------------------------------------


def test_journal_resume_refuses_foreign_arm_results(tmp_path: Path):
    """Resuming a journal recorded under a different arm_model raises a
    HarnessError naming BOTH model ids, with zero adapter calls — the
    fail-closed alternative to silently judging Haiku's answers as
    Sonnet's (#235)."""
    journal = RunJournal(tmp_path / "journal.jsonl")
    journal.record(
        ItemResult(
            item_id="syn-sp-01",
            arm_model="claude-haiku-4-5",
            route="retrieval",
            answer_text="A haiku-arm answer. [1]",
        )
    )
    adapter, deps = _answerable_deps()
    with pytest.raises(HarnessError) as excinfo:
        run_answer_path(
            [ANSWERABLE_ITEM],
            deps,
            arm_model="claude-sonnet-5",
            mode="fake",
            journal=journal,
        )
    message = str(excinfo.value)
    assert "claude-haiku-4-5" in message
    assert "claude-sonnet-5" in message
    assert adapter.calls == []


def test_journal_resume_same_arm_still_skips(tmp_path: Path):
    """The arm check must not over-refuse: a journal recorded under the
    SAME arm still resumes with zero adapter calls (#235)."""
    journal = RunJournal(tmp_path / "journal.jsonl")
    prior = ItemResult(
        item_id="syn-sp-01",
        arm_model="claude-haiku-4-5",
        route="retrieval",
        answer_text="A haiku-arm answer. [1]",
    )
    journal.record(prior)
    adapter, deps = _answerable_deps()
    results = run_answer_path(
        [ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake", journal=journal
    )
    assert adapter.calls == []
    assert results == (prior,)


# ---------------------------------------------------------------------------
# review-21 #237 (major): the chart path journals results and returns
# completed items on resume — resumable in BOTH directions.
# ---------------------------------------------------------------------------

_CHART_ITEMS = (
    {"id": "syn-chart-01", "request": "Plot the synthetic series", "expected": "spec"},
    {"id": "syn-chart-02", "request": "Plot the other synthetic series", "expected": "spec"},
)


def _recording_planner(calls: list[str]):
    def plan_chart(request: str) -> dict[str, object]:
        calls.append(request)
        return {"kind": "spec", "spec": {"chart_id": "syn-chart"}}

    return plan_chart


def test_chart_path_journals_results_as_they_complete(tmp_path: Path):
    """A journalled chart run writes one record per completed item, and a
    second run over the same journal re-plans NOTHING (#237: a crash
    mid-run must not re-pay every planner call)."""
    journal = RunJournal(tmp_path / "chart-journal.jsonl")
    calls: list[str] = []
    first = run_chart_path(list(_CHART_ITEMS), _recording_planner(calls), journal=journal)
    assert len(first) == 2
    assert journal.path.is_file(), "chart results must be journalled as they complete"
    recorded_lines = [
        line for line in journal.path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(recorded_lines) == 2

    calls.clear()
    second = run_chart_path(list(_CHART_ITEMS), _recording_planner(calls), journal=journal)
    assert calls == []
    assert {result.item_id for result in second} == {"syn-chart-01", "syn-chart-02"}


def test_chart_path_resume_returns_completed_results(tmp_path: Path):
    """Journalled chart items appear in the RETURN VALUE on resume —
    skipping them out of the results starves the gate denominator
    (#237's second failure direction)."""
    journal = RunJournal(tmp_path / "chart-journal.jsonl")
    completed = ChartItemResult(
        item_id="syn-chart-01",
        outcome="spec",
        planned={"kind": "spec", "spec": {"chart_id": "syn-chart"}},
    )
    journal.record(completed)

    calls: list[str] = []
    results = run_chart_path(list(_CHART_ITEMS), _recording_planner(calls), journal=journal)
    by_id = {result.item_id: result for result in results}
    assert set(by_id) == {"syn-chart-01", "syn-chart-02"}
    assert by_id["syn-chart-01"] == completed
    assert calls == ["Plot the other synthetic series"]


def test_chart_journal_records_round_trip_chart_item_result(tmp_path: Path):
    """A ChartItemResult — including skipped_blocked with its reason —
    survives record → load with equality (#237)."""
    journal = RunJournal(tmp_path / "chart-journal.jsonl")
    result = ChartItemResult(
        item_id="syn-chart-flagship",
        outcome="skipped_blocked",
        blocked_reason="synthetic stand-in for the #117 fixture embargo",
    )
    journal.record(result)
    (loaded,) = journal.load_results()
    assert isinstance(loaded, ChartItemResult)
    assert loaded == result


# ---------------------------------------------------------------------------
# review-21 #243 (major): ItemResult keeps the generation document set's
# per-document source_type — the voices-separation gate's evidence.
# ---------------------------------------------------------------------------


def test_item_result_records_document_source_types(tmp_path: Path):
    """The ItemResult carries a ``documents`` field of
    {chunk_id, source_type} mappings for the generation document set —
    and it round-trips through the journal — so DESIGN §6.2's structural
    voices gate is computable from run records (#243)."""
    passages = RetrievedPassages(
        passages=(
            RerankedPassage(
                chunk_id="syn_doc:0001",
                rerank_score=0.9,
                clears_threshold=True,
                payload=production_passage_payload("syn_doc:0001"),
            ),
            RerankedPassage(
                chunk_id="voices_doc:0001",
                rerank_score=0.8,
                clears_threshold=True,
                payload=production_passage_payload("voices_doc:0001", source_type="voices"),
            ),
        )
    )
    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(adapter=adapter, retrieve=lambda decision: passages)
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model="claude-haiku-4-5", mode="fake")

    documents = result.documents
    assert [(entry["chunk_id"], entry["source_type"]) for entry in documents] == [
        ("syn_doc:0001", "report"),
        ("voices_doc:0001", "voices"),
    ]

    journal = RunJournal(tmp_path / "journal.jsonl")
    journal.record(result)
    (loaded,) = journal.load_results()
    assert loaded == result
    assert loaded.documents == result.documents


# ---------------------------------------------------------------------------
# review-21 #246 (minor): journal resume survives a crash-truncated tail,
# refuses mid-file corruption loudly, and maps schema drift to HarnessError.
# ---------------------------------------------------------------------------


def _journalled_pair(journal: RunJournal) -> tuple[ItemResult, ItemResult]:
    first = ItemResult(item_id="syn-sp-01", arm_model="claude-haiku-4-5", route="retrieval")
    second = ItemResult(item_id="syn-sp-02", arm_model="claude-haiku-4-5", route="retrieval")
    journal.record(first)
    journal.record(second)
    return first, second


def test_journal_tolerates_truncated_final_line(tmp_path: Path):
    """A run killed mid-record leaves a truncated FINAL line: resume
    loads every complete record, drops the fragment (the item safely
    re-runs) and surfaces the corruption as a warning — never a raw
    JSONDecodeError exactly when resume is needed (#246)."""
    journal = RunJournal(tmp_path / "journal.jsonl")
    first, second = _journalled_pair(journal)
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": "syn-sp-03", "arm_mo')  # crash mid-write, no newline
    with pytest.warns(UserWarning, match="truncated"):
        results = journal.load_results()
    assert results == (first, second)


def test_journal_refuses_mid_file_corruption(tmp_path: Path):
    """Corruption BEFORE the final line is not a safe tail-drop — silent
    skipping would silently lose paid results. load_results raises
    HarnessError naming the corrupt line number (#246)."""
    journal = RunJournal(tmp_path / "journal.jsonl")
    journal.record(ItemResult(item_id="syn-sp-01", arm_model="claude-haiku-4-5", route="retrieval"))
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": "syn-sp-02", "arm_mo\n')  # corrupt, NOT final
    journal.record(ItemResult(item_id="syn-sp-03", arm_model="claude-haiku-4-5", route="retrieval"))
    with pytest.raises(HarnessError) as excinfo:
        journal.load_results()
    assert "2" in str(excinfo.value)  # names the offending line number


def test_journal_unknown_field_becomes_loud_harness_error(tmp_path: Path):
    """Schema drift (a journal line with a field ItemResult does not
    know) surfaces as a HarnessError naming the field — never a bare
    TypeError from the dataclass constructor (#246)."""
    journal = RunJournal(tmp_path / "journal.jsonl")
    line = json.dumps(
        {
            "item_id": "syn-sp-01",
            "arm_model": "claude-haiku-4-5",
            "route": "retrieval",
            "mystery_field": 1,
        }
    )
    journal.path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(HarnessError) as excinfo:
        journal.load_results()
    assert "mystery_field" in str(excinfo.value)
