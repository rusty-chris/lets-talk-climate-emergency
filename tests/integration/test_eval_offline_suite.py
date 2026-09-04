"""Issue #21 TDD plan item 11 (green phase): the one-command offline suite.

The acceptance criterion's single command — ``scripts/run_evals.py`` in
``--offline`` mode — must run the WHOLE release-eval suite end-to-end on
the Fake/Replay provider seam against gold sets: gold validation, the
answer + chart paths, every release gate, the bake-off selection, and
both results artefacts (results.json + RESULTS.md). It touches no
network and needs no LLM key (IMPLEMENTATION.md §4.4), so it is an
integration-tier test that actually RUNS in CI rather than skipping
(review finding #32 — a skip must never green a CI tier).

With the owner severity audit complete (2026-09-04) and every offline
feed simulated-passing (classifier summary, refusal declines, judged
severity), the suite's verdict is PASSED and the command exits 0 —
meaning "harness plumbing green", never a release decision: outcomes
are labelled offline-simulated end to end, the release decision comes
only from the live run, and the non-published default out-dir plus the
boot gate keep an offline file from masquerading as a release
artefact. Fail-closed behaviour stays pinned separately: a missing
severity feed degrades the gate to BLOCKED
(tests/unit/test_review_303_gate_wiring.py) and the pending-audit
BLOCKED path is pinned in tests/unit/test_eval_gates.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._eval_harness_fixtures import write_synthetic_gold

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_offline_single_command_runs_whole_suite(tmp_path: Path) -> None:
    """One command drives the full suite offline against synthetic gold
    and writes both artefacts; with the owner audit complete and every
    offline feed simulated-passing the verdict is PASSED and the command
    exits 0 (harness plumbing green — honestly labelled
    offline-simulated, never a release decision) — no network, no key."""
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "out"

    # A hermetic environment: CI convention set, and NO Anthropic key on
    # the process — the offline suite must never reach for the network.
    env = dict(os.environ)
    env["CI"] = "true"
    env.pop("ANTHROPIC_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_evals.py"),
            "--offline",
            "--qa-gold",
            str(qa_path),
            "--charts-gold",
            str(charts_path),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
    )

    # Harness plumbing green: every simulated feed passes, so the
    # offline verdict is PASSED and the release build exits 0. The
    # stdout banner keeps the run honestly labelled as simulated.
    assert result.returncode == 0, result.stderr
    assert "OFFLINE / SIMULATED" in result.stdout
    assert "PASSED" in result.stdout

    json_path = out_dir / "results.json"
    md_path = out_dir / "RESULTS.md"
    assert json_path.is_file(), "the machine-readable results file must be written"
    assert md_path.is_file(), "the human RESULTS.md summary must be written"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["release_verdict"] == "passed"
    # The offline-simulated mode label is the honesty contract (#242):
    # these are simulated outcomes, never a measured release result.
    assert payload["mode"] == "offline-simulated"
    assert payload["selected_model"] == "claude-haiku-4-5"
    (arm,) = payload["arms"]
    gate_names = {gate["name"] for gate in arm["gates"]}
    # The full gate battery ran: refusal, severity (scored), charts, voices.
    assert {"refusal", "severity", "chart_spec", "voices_separation"} <= gate_names
    severity = next(gate for gate in arm["gates"] if gate["name"] == "severity")
    # With the owner audit complete the severity gate SCORES the
    # simulated judged records (exact-match per gold label) — a scored
    # simulated gate row, never a blocked placeholder and never a
    # vacuous 0/0.
    assert severity["status"] == "passed"
    assert severity["denominator"] >= 1
    assert severity["numerator"] == severity["denominator"]

    rendered = md_path.read_text(encoding="utf-8")
    assert "Release verdict: PASSED" in rendered
    # The simulated banner renders in the human summary too.
    assert "OFFLINE / SIMULATED RESULTS" in rendered
    # The blocked flagship is visible, never silently dropped.
    assert "Skipped-visibly:" in rendered
    assert "syn-chart-flagship" in rendered


def test_offline_suite_reports_deterministic_metrics(tmp_path: Path) -> None:
    """review-21 #242: the acceptance criterion says one command runs
    the deterministic metrics — recall@8/MRR/nDCG (implemented in
    evals/metrics.py), the chart faithfulness and chart refusal gates,
    and the calibrated-term proxy must actually be INVOKED by the
    runner and appear in the payload, not merely exist as tested pure
    functions."""
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    env["CI"] = "true"
    env.pop("ANTHROPIC_API_KEY", None)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_evals.py"),
            "--offline",
            "--qa-gold",
            str(qa_path),
            "--charts-gold",
            str(charts_path),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
    )

    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    (arm,) = payload["arms"]

    metrics = arm.get("retrieval_metrics")
    assert metrics is not None, (
        "recall/MRR/nDCG are implemented in evals/metrics.py but must be computed "
        "by the runner from the run's retrieved_chunk_ids vs gold_chunk_ids (#242)"
    )
    for key in ("recall_at_8", "mrr", "ndcg_at_8"):
        assert key in metrics, f"retrieval_metrics must carry {key}"
        assert 0.0 <= float(metrics[key]) <= 1.0

    gate_names = {gate["name"] for gate in arm["gates"]}
    assert {"chart_faithfulness", "chart_refusal"} <= gate_names, (
        "the chart faithfulness (vs committed fixtures) and chart refusal gates "
        "must run in the offline suite (#242)"
    )

    # The calibrated-term proxy (#244's fixed metric) surfaces in the payload.
    assert "calibrated_term" in json.dumps(arm), (
        "the calibrated-term proxy must be invoked and reported (#242/#244)"
    )


def test_release_orchestrator_runs_fake_mode_end_to_end(tmp_path: Path) -> None:
    """review-21 #236: the SAME orchestrator behind --live/--record runs
    end-to-end in fake mode — arms, chart path, judge batching (against
    the batch-client double), gates and the results payload — so the
    tested path IS the shipped live path with only the seams swapped.
    Fake mode spends $0 and never touches the ledger."""
    from evals import harness
    from evals.harness import AnswerPathDeps, load_and_validate_gold
    from rag.provider import AnswerWithCitations, Citation, FakeAdapter
    from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
    from tests._eval_harness_fixtures import FakeBatchClient, production_passage_payload

    orchestrator = getattr(harness, "run_release_eval", None)
    assert orchestrator is not None, (
        "evals.harness.run_release_eval must exist and be the one orchestration "
        "path shared by --offline and --live (#236)"
    )

    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    gold = load_and_validate_gold(qa_path, charts_path)

    passages = RetrievedPassages(
        passages=(
            RerankedPassage(
                chunk_id="syn_doc:0001",
                rerank_score=0.9,
                clears_threshold=True,
                payload=production_passage_payload("syn_doc:0001"),
            ),
        )
    )
    refusal = HonestRefusal(
        refusal_text="The synthetic corpus does not cover this.",
        covered_topics=("synthetic warming",),
        top_score=0.05,
        threshold=0.4,
    )
    answer = AnswerWithCitations(
        text="The synthetic planet warmed 1.1 degrees. [1]",
        citations=(Citation(cited_text="synthetic passage", document_index=0),),
    )

    def deps_factory(arm_model: str) -> AnswerPathDeps:
        item_count = len(gold.qa_items)
        adapter = FakeAdapter(
            generate_results=[answer] * item_count,
            structured_results=[
                {"scope": "in_scope", "rewritten_query": item["question"]} for item in gold.qa_items
            ],
        )

        def retrieve(decision):
            query = decision.retrieval_query or ""
            return refusal if "unanswerable" in query else passages

        return AnswerPathDeps(adapter=adapter, retrieve=retrieve)

    ledger_path = tmp_path / "spend-ledger.csv"
    payload = orchestrator(
        gold,
        arm_models=("claude-haiku-4-5", "claude-sonnet-5"),
        deps_factory=deps_factory,
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=FakeBatchClient(),
        mode="fake",
        ledger_path=ledger_path,
        journal_dir=tmp_path / "journals",
        session_id="syn-fake-release",
    )

    assert payload["release_verdict"] in ("passed", "failed", "blocked")
    assert [arm["model"] for arm in payload["arms"]] == [
        "claude-haiku-4-5",
        "claude-sonnet-5",
    ]
    for arm in payload["arms"]:
        assert arm["gates"], "each arm carries its full gate battery"
    assert not ledger_path.exists(), "fake mode costs $0 and must never touch the ledger"
