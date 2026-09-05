"""Review #316 red phase (Fable): verdict-bearing state must be
journalled — a resumed run reuses paid-for judge results at $0, and a
post-hoc diagnosis can attribute citation failures from artifacts.

The live release run (2026-09-04/05) lost $0.34 — ~20% of its whole
$1.71 spend — resubmitting a judge batch whose 155 results were already
paid for and retrievable by id: the batch id and collected verdicts
lived only in process memory, so the crash between "batch ended" and
"results folded" made the resume start over. The same gap made the
citation_support diagnosis inferential: the answer journal keeps only
``{validated, supported, factual}`` and discards the per-pair verdicts
and the SSE transcript.

Pinned here (issue #316's three required tests):

1. Judge-journal contract: after ``submit_judge_batch``, the batch id +
   request custom_ids are journalled BEFORE the first poll, in
   ``<journal_dir>/judges.jsonl`` (FLAGGED filename pin — the issue's
   own proposal), with #246 corruption semantics (tolerate a truncated
   tail with a warning; refuse interior corruption loudly, naming the
   line).
2. Journal contract: each answered ItemResult carries the delivered SSE
   transcript (``sse_transcript`` — FLAGGED field-name pin) and the
   validator's per-pair verdicts
   ``{pair_index, sentence_index, document_index, supported}`` on the
   validation record — enough to recompute ``{supported, factual}``
   offline.
3. Resume: a journal carrying the collected verdicts makes the resumed
   run submit ZERO new batches (the $0.34 duplicate becomes impossible);
   a journal carrying only the submission (kill-after-submit) makes the
   resumed run collect by batch id — ``results()`` is free for 29 days —
   still with zero ``create`` calls, and the ledger unchanged.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import harness, ledger
from evals.harness import (
    AnswerPathDeps,
    GoldSets,
    HarnessError,
    RunJournal,
    load_and_validate_gold,
    run_answer_path,
)
from rag.citation_validator import (
    PairVerdict,
    ValidationOutcome,
    segment_answer_sentences,
)
from rag.provider import FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    production_passage_payload,
    succeeded_batch_result,
    write_synthetic_gold,
)

ARM_MODEL = "claude-haiku-4-5"
SEVERITY_CUSTOM_ID = f"{ARM_MODEL}__severity_fidelity__syn-sev-01"

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
SECOND_SENTENCE = " It is very likely to continue."

#: Two cited sentences, citations interleaved in transport arrival order —
#: the true-SSE shape the #303 ratification pinned.
INTERLEAVED_STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": FIRST_SENTENCE}},
    {
        "type": "content_block_delta",
        "delta": {
            "type": "citations_delta",
            "citation": {"cited_text": "synthetic passage", "document_index": 0},
        },
    },
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": SECOND_SENTENCE}},
    {
        "type": "content_block_delta",
        "delta": {
            "type": "citations_delta",
            "citation": {"cited_text": "synthetic passage", "document_index": 0},
        },
    },
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 80}},
    {"type": "message_stop"},
]


def _entailing_validator(grounded: Any, sse_events: Any) -> ValidationOutcome:
    """A genuine-shaped validator over the REAL production segmentation:
    one supported PairVerdict per (cited factual sentence, document)."""
    sentences = segment_answer_sentences(sse_events)
    verdicts = []
    pair_index = 0
    for sentence in sentences:
        if not sentence.factual:
            continue
        for document_index in sentence.document_indices:
            verdicts.append(
                PairVerdict(
                    pair_index=pair_index,
                    sentence_index=sentence.index,
                    document_index=document_index,
                    supported=True,
                )
            )
            pair_index += 1
    return ValidationOutcome(validated=True, sentences=sentences, verdicts=tuple(verdicts))


def _synthetic_gold(tmp_path: Path) -> GoldSets:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _deps_factory(gold: GoldSets):
    def deps_factory(arm_model: str) -> AnswerPathDeps:
        adapter = FakeAdapter(
            generate_stream_results=[INTERLEAVED_STREAM] * len(gold.qa_items),
            structured_results=[
                {"scope": "in_scope", "rewritten_query": item["question"]} for item in gold.qa_items
            ],
        )

        def retrieve(decision):
            query = decision.retrieval_query or ""
            if "unanswerable" in query or "out-of-scope" in query:
                return REFUSAL
            return PASSAGES

        return AnswerPathDeps(
            adapter=adapter, retrieve=retrieve, validate_exchange=_entailing_validator
        )

    return deps_factory


def _run_release_eval(
    gold: GoldSets,
    tmp_path: Path,
    batch_client: FakeBatchClient,
    *,
    mode: str = "fake",
    session_id: str = "review-316",
):
    return harness.run_release_eval(
        gold,
        arm_models=(ARM_MODEL,),
        deps_factory=_deps_factory(gold),
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=batch_client,
        mode=mode,
        ledger_path=tmp_path / "spend-ledger.csv",
        journal_dir=tmp_path / "journals",
        session_id=session_id,
    )


def _severity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    (arm,) = payload["arms"]
    (gate,) = [gate for gate in arm["gates"] if gate["name"] == "severity"]
    return gate


def _scored_severity_client() -> FakeBatchClient:
    return FakeBatchClient(
        results=[succeeded_batch_result(SEVERITY_CUSTOM_ID, json.dumps({"judged_lead": "serious"}))]
    )


def _judges_journal_path(tmp_path: Path) -> Path:
    return tmp_path / "journals" / "judges.jsonl"


# ---------------------------------------------------------------------------
# 1. Submission is journalled BEFORE the first poll.
# ---------------------------------------------------------------------------


class _JournalSnapshotClient(FakeBatchClient):
    """Snapshots the judges-journal file at the FIRST retrieve() — the
    submission record must already be durable by then, or a crash during
    polling loses the batch id exactly like the live run did."""

    def __init__(self, judges_journal_path: Path, results: Any = ()) -> None:
        super().__init__(results)
        self.judges_journal_path = Path(judges_journal_path)
        self.journal_text_at_first_poll: str | None = None

    def retrieve(self, batch_id: str):
        if self.journal_text_at_first_poll is None:
            self.journal_text_at_first_poll = (
                self.judges_journal_path.read_text(encoding="utf-8")
                if self.judges_journal_path.is_file()
                else ""
            )
        return super().retrieve(batch_id)


def test_judge_batch_id_and_custom_ids_are_journalled_before_polling(tmp_path: Path):
    gold = _synthetic_gold(tmp_path)
    client = _JournalSnapshotClient(_judges_journal_path(tmp_path))
    client.program_results(
        [succeeded_batch_result(SEVERITY_CUSTOM_ID, json.dumps({"judged_lead": "serious"}))]
    )
    _run_release_eval(gold, tmp_path, client)

    text = client.journal_text_at_first_poll
    assert text, (
        "the judge batch id + custom_ids must be journalled BEFORE the first "
        "poll (a crash while polling must not orphan the paid batch)"
    )
    assert "synbatch_001" in text
    (submitted,) = client.create_calls
    for entry in submitted:
        assert entry["custom_id"] in text, (
            f"submitted custom_id {entry['custom_id']!r} missing from the "
            "journalled request set — resume cannot verify batch identity"
        )


# ---------------------------------------------------------------------------
# 2. ItemResult journals the SSE transcript + per-pair verdicts.
# ---------------------------------------------------------------------------

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}


def test_item_result_journals_transcript_and_pair_verdicts(tmp_path: Path):
    adapter = FakeAdapter(
        generate_stream_results=[INTERLEAVED_STREAM],
        structured_results=[{"scope": "in_scope", "rewritten_query": "synthetic"}],
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_entailing_validator,
    )
    journal = RunJournal(tmp_path / "answers.jsonl")
    (result,) = run_answer_path(
        [ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, mode="fake", journal=journal
    )

    transcript = getattr(result, "sse_transcript", None)
    assert transcript, (
        "the answered ItemResult must carry the delivered SSE transcript — "
        "without it no citation failure is attributable from artifacts (#316)"
    )
    assert [event.get("event") for event in transcript] == [
        "text",
        "citation",
        "text",
        "citation",
        "usage",
        "footer",
    ]

    assert result.validation is not None
    verdicts = result.validation.get("verdicts")
    assert verdicts == [
        {"pair_index": 0, "sentence_index": 0, "document_index": 0, "supported": True},
        {"pair_index": 1, "sentence_index": 1, "document_index": 0, "supported": True},
    ], "the validator's per-pair verdicts must ride the validation record"

    # The journalled record round-trips losslessly...
    (loaded,) = journal.load_results()
    assert loaded == result

    # ...and is sufficient to recompute {supported, factual} OFFLINE.
    sentences = segment_answer_sentences(loaded.sse_transcript)
    factual_indices = [sentence.index for sentence in sentences if sentence.factual]
    assert len(factual_indices) == loaded.validation["factual"]
    supported_indices = {
        verdict["sentence_index"]
        for verdict in loaded.validation["verdicts"]
        if verdict["supported"]
    }
    recomputed_supported = sum(1 for index in factual_indices if index in supported_indices)
    assert recomputed_supported == loaded.validation["supported"]


# ---------------------------------------------------------------------------
# 3. Resume reuses the paid batch: zero new submissions, ledger unchanged.
# ---------------------------------------------------------------------------


def test_resume_reuses_journalled_verdicts_with_zero_new_batches(tmp_path: Path):
    """Run 1 completes (batch submitted, verdicts collected + journalled).
    Run 2's batch client has NOTHING programmed: the verdicts must come
    from the journal — zero ``create`` calls, identical severity gate,
    ledger cumulative unchanged. This is the pin that makes the live
    run's $0.34 duplicate batch impossible."""
    gold = _synthetic_gold(tmp_path)
    ledger_path = tmp_path / "spend-ledger.csv"

    payload_1 = _run_release_eval(
        gold, tmp_path, _scored_severity_client(), mode="recording", session_id="resume-0usd"
    )
    severity_1 = _severity_payload(payload_1)
    assert severity_1["status"] == "passed"
    assert (severity_1["numerator"], severity_1["denominator"]) == (1, 1)
    cumulative_after_run_1 = ledger.cumulative_usd(ledger_path)
    assert cumulative_after_run_1 > 0

    unprogrammed_client = FakeBatchClient(results=[])
    payload_2 = _run_release_eval(
        gold, tmp_path, unprogrammed_client, mode="recording", session_id="resume-0usd"
    )
    assert unprogrammed_client.create_calls == [], (
        "the journal carries the collected verdicts: a resumed run must "
        "submit ZERO new batches (the $0.34 duplicate spend of the live run)"
    )
    severity_2 = _severity_payload(payload_2)
    assert (severity_2["status"], severity_2["numerator"], severity_2["denominator"]) == (
        "passed",
        1,
        1,
    ), "resumed verdicts must reproduce the same scored gate"
    assert ledger.cumulative_usd(ledger_path) == pytest.approx(cumulative_after_run_1), (
        "a fully-resumed run spends $0"
    )


class _CrashAfterSubmitClient(FakeBatchClient):
    """Dies on the first poll — the run is killed between submission and
    collection, the exact live-run crash window."""

    def retrieve(self, batch_id: str):
        raise RuntimeError("simulated kill between judge-batch submission and collection")


def test_kill_after_submit_resumes_by_batch_id_without_resubmitting(tmp_path: Path):
    """Run 1 dies after ``create`` (batch id journalled, no verdicts).
    Run 2 must collect the SAME batch via ``batches.results(batch_id)``
    (free — results stay retrievable for 29 days) and never ``create``:
    the paid request set is reused, the gate scores, the ledger holds."""
    gold = _synthetic_gold(tmp_path)
    ledger_path = tmp_path / "spend-ledger.csv"

    crashing_client = _CrashAfterSubmitClient()
    with pytest.raises(RuntimeError, match="simulated kill"):
        _run_release_eval(
            gold, tmp_path, crashing_client, mode="recording", session_id="resume-by-id"
        )
    assert len(crashing_client.create_calls) == 1, "run 1 paid for exactly one batch"
    cumulative_after_crash = ledger.cumulative_usd(ledger_path)

    resumed_client = _scored_severity_client()
    payload = _run_release_eval(
        gold, tmp_path, resumed_client, mode="recording", session_id="resume-by-id"
    )
    assert resumed_client.create_calls == [], (
        "the journalled submission names the batch id: resume collects by id, "
        "it never re-creates the same 155-request batch (the live $0.34 loss)"
    )
    assert "results" in resumed_client.ops, "resume must read results(batch_id)"
    severity = _severity_payload(payload)
    assert (severity["status"], severity["numerator"], severity["denominator"]) == (
        "passed",
        1,
        1,
    )
    assert ledger.cumulative_usd(ledger_path) == pytest.approx(cumulative_after_crash), (
        "collecting an already-paid batch by id costs $0"
    )


# ---------------------------------------------------------------------------
# 4. Judges-journal corruption follows the #246 conventions.
# ---------------------------------------------------------------------------


def test_judges_journal_tolerates_truncated_final_line(tmp_path: Path):
    """A run killed mid-write leaves a truncated tail: the resumed run
    surfaces it as a warning and still completes — never a raw crash
    exactly when resume is the feature (#246 semantics)."""
    gold = _synthetic_gold(tmp_path)
    _run_release_eval(gold, tmp_path, _scored_severity_client())
    judges_path = _judges_journal_path(tmp_path)
    assert judges_path.is_file(), "run 1 must have journalled its judge batch"

    with judges_path.open("a", encoding="utf-8") as handle:
        handle.write('{"batch_id": "synbatch_001", "custo')  # killed mid-record

    with pytest.warns(UserWarning, match="truncated"):
        payload = _run_release_eval(gold, tmp_path, _scored_severity_client())
    severity = _severity_payload(payload)
    assert (severity["numerator"], severity["denominator"]) == (1, 1)


def test_judges_journal_refuses_interior_corruption_loudly(tmp_path: Path):
    """Corruption BEFORE the final line is never silently skipped —
    dropping an interior record silently drops paid verdicts. The
    resumed run raises HarnessError naming the corrupt line (#246)."""
    gold = _synthetic_gold(tmp_path)
    _run_release_eval(gold, tmp_path, _scored_severity_client())
    judges_path = _judges_journal_path(tmp_path)
    assert judges_path.is_file(), "run 1 must have journalled its judge batch"

    intact = judges_path.read_text(encoding="utf-8")
    judges_path.write_text('{"batch_id": "synbatch_001", "custo\n' + intact, encoding="utf-8")

    with pytest.raises(HarnessError, match="line 1"):
        _run_release_eval(gold, tmp_path, _scored_severity_client())
