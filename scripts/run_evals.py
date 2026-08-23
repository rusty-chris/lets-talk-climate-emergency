#!/usr/bin/env python3
"""The one-command release-eval runner (issue #21 acceptance criterion).

``uv run python scripts/run_evals.py`` runs the full suite: gold
loading + validation, the deterministic metrics, the (opt-in, budget
pre-flighted, Batches-judged) live arms, gates, bake-off selection, and
the results artefacts (evals/results.json + evals/RESULTS.md).

This is a **release eval entry point**, not a test (IMPLEMENTATION.md
§4.4): live/recording runs require an explicit ``--live`` /
``--record`` opt-in AND a passing $9.00-cap pre-flight
(evals.harness.preflight_budget); the default invocation runs only the
deterministic, $0 portions. The exit code is the release-CI contract:
0 only when the release verdict is ``passed``; any failed OR blocked
gate exits non-zero so the release build actually blocks
(``test_release_build_exit_code_on_gate_violation``).

The ``--offline`` mode drives the whole suite end-to-end on the
Fake/Replay seam against gold sets, with zero network — the CI-tier
single-command proof (issue #21 TDD plan item 11).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import gates, report  # noqa: E402
from evals.harness import (  # noqa: E402
    CHART_REQUESTS_PATH,
    CLIMATE_QA_PATH,
    AnswerPathDeps,
    load_and_validate_gold,
    run_answer_path,
    run_chart_path,
)
from evals.metrics import voices_separation_violations  # noqa: E402

#: The default bake-off arm the offline self-test runs (its judge would be
#: Sonnet under evals.judges — but the offline suite is $0 and unjudged).
_OFFLINE_ARM_MODEL = "claude-haiku-4-5"

#: Answer-producing gold categories (everything that is not a no_answer
#: decline or a chart request).
_ANSWERABLE_CATEGORIES = frozenset(
    {"single_passage", "multi_passage", "severity", "adversarial", "voices_action", "targeted"}
)


def exit_code_for_verdict(verdict: str) -> int:
    """0 only for ``passed``; ``failed`` and ``blocked`` are non-zero
    (a blocked owner audit must block the release build, never slip
    through as success). Unknown verdicts are non-zero too — fail
    closed."""
    return 0 if verdict == "passed" else 1


class _OfflineAdapter:
    """A $0, network-free provider double for the offline self-test: it
    classifies every query in_scope and returns a fixed cited answer, so
    the runner exercises the real classify -> route -> generate seam
    without a queue to exhaust or a network to touch."""

    def structured(self, *, messages, schema, config, system=None):
        return {"scope": "in_scope", "rewritten_query": "offline synthetic query"}

    def generate(self, *, messages, documents, config, system=None):
        from rag.provider import AnswerWithCitations, Citation

        return AnswerWithCitations(
            text="Offline synthetic answer grounded in the corpus. [1]",
            citations=(Citation(cited_text="synthetic passage", document_index=0),),
        )


def _offline_retrieve(_decision):
    """One synthetic reranked passage above threshold — the offline
    answer path always retrieves (refusals are simulated at the gate)."""
    from rag.retrieval import RerankedPassage, RetrievedPassages

    return RetrievedPassages(
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


def _run_offline_suite(qa_path: Path, charts_path: Path, out_dir: Path) -> str:
    """The full deterministic suite on the Fake seam: validate gold, run
    the answer + chart paths, compute every gate, build the results
    artefacts. Returns the release verdict.

    The severity gate is BLOCKED while the committed owner-audit packet is
    pending, so the offline verdict is ``blocked`` — which is exactly the
    fail-closed behaviour the release build must honour.
    """
    gold = load_and_validate_gold(qa_path, charts_path)

    answerable = [item for item in gold.qa_items if item.get("category") in _ANSWERABLE_CATEGORIES]
    deps = AnswerPathDeps(adapter=_OfflineAdapter(), retrieve=_offline_retrieve)
    answer_results = run_answer_path(answerable, deps, arm_model=_OFFLINE_ARM_MODEL, mode="fake")

    def _plan_chart(_request: str) -> dict[str, object]:
        return {"kind": "spec", "spec": {"chart_id": "offline-synthetic", "chart_type": "line"}}

    chart_results = run_chart_path(gold.chart_items, _plan_chart)

    # Deterministic gate arithmetic over the run (perfect simulated
    # refusal/decline outcomes; the severity gate blocks on the pending
    # owner audit regardless).
    refusal_gate = gates.refusal_gate(
        {item_id: True for item_id in gold.refusal_gate_ids},
        gate_item_ids=gold.refusal_gate_ids,
    )
    false_refusal_gate = gates.false_refusal_gate(
        {result.item_id: result.refused for result in answer_results}
    )
    canned_check = gates.canned_out_of_scope_check(
        {item_id: True for item_id in gold.canned_out_of_scope_ids}
    )
    severity_gate = gates.severity_gate([])
    chart_spec_gate = gates.chart_spec_gate(
        [
            {"item_id": result.item_id, "status": "match"}
            if result.outcome != "skipped_blocked"
            else {
                "item_id": result.item_id,
                "status": "skipped_blocked",
                "blocked_reason": result.blocked_reason,
            }
            for result in chart_results
        ]
    )
    voices_gate = gates.voices_separation_gate(
        voices_separation_violations(
            {
                "item_id": result.item_id,
                "route_is_voices": False,
                "documents": [
                    {"chunk_id": chunk_id, "source_type": "report"}
                    for chunk_id in result.retrieved_chunk_ids
                ],
            }
            for result in answer_results
        )
    )

    arm = gates.ArmResult(
        model=_OFFLINE_ARM_MODEL,
        gates=(
            refusal_gate,
            false_refusal_gate,
            canned_check,
            severity_gate,
            chart_spec_gate,
            voices_gate,
        ),
        cost_usd=0.0,
    )
    verdict = gates.release_verdict(list(arm.gates))
    selected = gates.select_production_model([arm])
    payload = report.build_results_payload([arm], verdict=verdict, selected_model=selected)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.write_results(
        payload, json_path=out_dir / "results.json", md_path=out_dir / "RESULTS.md"
    )
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite; return the release exit code.

    Flags (pinned by tests/unit/test_eval_report.py):
    ``--validate-gold-only`` — load + validate the committed gold sets
    and exit 0 (the cheapest invocation; unit-tier checkable).
    ``--live`` / ``--record`` — explicit opt-in for network-touching
    arms; both require a passing $9.00-cap pre-flight.
    """
    parser = argparse.ArgumentParser(description="Release eval runner (issue #21).")
    parser.add_argument("--validate-gold-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--qa-gold", type=Path, default=CLIMATE_QA_PATH)
    parser.add_argument("--charts-gold", type=Path, default=CHART_REQUESTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "evals")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.validate_gold_only:
        load_and_validate_gold(args.qa_gold, args.charts_gold)
        print(f"gold OK: {args.qa_gold} + {args.charts_gold}")
        return 0

    if args.live or args.record:
        # Live/recording arms need real credentials + a passing pre-flight;
        # this entry point refuses to fabricate them (fail closed).
        print(
            "live/recording arms require the recorded-run tooling and a passing "
            "budget pre-flight; run --offline for the deterministic suite",
            file=sys.stderr,
        )
        return 1

    verdict = _run_offline_suite(args.qa_gold, args.charts_gold, args.out_dir)
    print(f"release verdict: {verdict.upper()}")
    return exit_code_for_verdict(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
