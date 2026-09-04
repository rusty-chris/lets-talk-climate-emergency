#!/usr/bin/env python3
"""The one-command release-eval runner (issue #21 acceptance criterion).

``uv run python scripts/run_evals.py --offline`` runs the full
deterministic suite on the Fake/Replay seam against gold sets: gold
loading + validation, the answer + chart paths, the deterministic
metrics (recall@8/MRR/nDCG, the calibrated-term proxy), every release
gate, the bake-off selection, and the results artefacts (results.json +
RESULTS.md). It touches no network and needs no LLM key.

This is a **release eval entry point**, not a test (IMPLEMENTATION.md
§4.4). Two honesty rules the review-21 batch pins (findings #242/#236):

- The offline outcomes are labelled ``offline-simulated`` end to end
  (the payload's ``mode`` + a banner in RESULTS.md), and the default
  invocation NEVER writes the published ``evals/results.json`` /
  ``evals/RESULTS.md`` — publishing there is an explicit ``--out-dir``.
- ``--live`` / ``--record`` require an ``ANTHROPIC_API_KEY`` and the
  recorded-run tooling; without the key this entry point refuses,
  naming the missing credential — it never fabricates a run.

The exit code is the release-CI contract: 0 only when the release
verdict is ``passed``; any failed OR blocked gate exits non-zero so the
release build actually blocks.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import gates, report  # noqa: E402
from evals.harness import (  # noqa: E402
    CHART_REQUESTS_PATH,
    CLIMATE_QA_PATH,
    AnswerPathDeps,
    ItemResult,
    build_gate_battery,
    chart_gate_records,
    compute_calibrated_term_rate,
    compute_chart_faithfulness_records,
    compute_retrieval_metrics,
    load_and_validate_gold,
    run_answer_path,
    run_chart_path,
)

#: The default bake-off arm the offline self-test runs (its judge would be
#: Sonnet under evals.judges — but the offline suite is $0 and unjudged).
_OFFLINE_ARM_MODEL = "claude-haiku-4-5"

#: The visible label distinguishing simulated offline gate outcomes from
#: measured release results (finding #242).
_OFFLINE_MODE_LABEL = "offline-simulated"

#: Answer-producing gold categories (everything that is not a no_answer
#: decline or a chart request).
_ANSWERABLE_CATEGORIES = frozenset(
    {"single_passage", "multi_passage", "severity", "adversarial", "voices_action", "targeted"}
)

#: The simulated classifier-accuracy summary the offline suite feeds the
#: route_accuracy gate (issue #303): there is no live classifier offline,
#: so the summary is simulated as PASSING and honestly labelled by the
#: payload's offline-simulated mode banner — never a measured result.
_OFFLINE_CLASSIFIER_SUMMARY = {
    "release_gate_passes": True,
    "per_class": {"in_scope": 1.0, "unsafe": 1.0},
}


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
            text="Offline synthetic answer, very likely grounded in the corpus. [1]",
            citations=(Citation(cited_text="synthetic passage", document_index=0),),
        )


def _offline_passage_payload() -> dict[str, Any]:
    """One stored chunk payload in the PRODUCTION shape
    build_generation_request requires (finding #234): the offline suite
    drives the real builder, so its retrieved passage cannot be a thinner
    forked shape."""
    return {
        "chunk_id": "syn_doc:0001",
        "doc_id": "syn_doc",
        "section_path": ["1 Synthetic section"],
        "context_header": "Synthetic Assessment (offline) > 1 Synthetic section",
        "body": "synthetic passage",
        "token_count": 12,
        "confidence_markers": ["very likely"],
        "block_types": ["text"],
        "consensus_position": "assessed",
        "source_type": "report",
        "citation_metadata": {
            "licence": "CC-BY-4.0",
            "attribution_text": "Synthetic Assessment Cycle 1 (offline)",
            "canonical_url": "https://example.invalid/offline-assessment",
            "permitted_context": "public-noncommercial",
        },
        "parse_backend": "docling",
        "degraded_fallback": False,
        "needs_hand_review": False,
        "content_hash": "offline-synthetic-hash",
    }


def _offline_retrieve(_decision):
    """One synthetic reranked passage above threshold in the production
    payload shape — the offline answer path always retrieves (refusals
    are simulated at the gate)."""
    from rag.retrieval import RerankedPassage, RetrievedPassages

    return RetrievedPassages(
        passages=(
            RerankedPassage(
                chunk_id="syn_doc:0001",
                rerank_score=0.9,
                clears_threshold=True,
                payload=_offline_passage_payload(),
            ),
        )
    )


def _gold_driven_planner(chart_items: Sequence[Mapping[str, Any]]) -> Callable[[str], Any]:
    """The default offline planner: it replays each gold chart item's
    EXPECTED behaviour (spec items -> the gold spec; refusal items -> a
    refusal payload) keyed by the request phrasing. The gate inputs are
    then DERIVED by comparing this output to gold (finding #242), so a
    real regression (an injected wrong planner) still fails the gate —
    the simulation is honestly labelled, never a hardwired 'match'."""
    by_request = {item["request"]: item for item in chart_items}

    def plan(request: str) -> dict[str, Any]:
        item = by_request.get(request, {})
        if item.get("expected") == "refusal":
            return {"kind": "refusal", **(item.get("refusal") or {})}
        return {"kind": "spec", "spec": item.get("spec", {})}

    return plan


def _offline_validate_exchange(result: Any, sse_events: Sequence[Mapping[str, Any]]) -> Any:
    """The deterministic offline citation-support validator (issue #303):
    there is no live entailment judge offline, so this segments the
    delivered transcript with the production sentence rule and marks every
    cited factual sentence supported — a simulated, honestly-labelled
    all-supported outcome that EXERCISES the citation_support gate on the
    offline path (never a blocked-forever placeholder). Its verdict is
    labelled offline-simulated by the payload mode banner, exactly like the
    refusal/canned simulations."""
    from rag.citation_validator import (
        PairVerdict,
        ValidationOutcome,
        build_entailment_pairs,
        segment_answer_sentences,
    )

    sentences = segment_answer_sentences(sse_events)
    pairs = build_entailment_pairs(sentences, result.cited_passages)
    verdicts = tuple(
        PairVerdict(
            pair_index=pair.pair_index,
            sentence_index=pair.sentence_index,
            document_index=pair.document_index,
            supported=True,
        )
        for pair in pairs
    )
    factual = sum(1 for sentence in sentences if sentence.factual)
    return ValidationOutcome(
        validated=True,
        sentences=sentences,
        verdicts=verdicts,
        support_rate=1.0 if factual else None,
        model=_OFFLINE_MODE_LABEL,
    )


def _simulated_decline_results(gold: Any) -> list[ItemResult]:
    """The offline suite's SIMULATED refusal/canned outcomes as
    ItemResults the shared ``build_gate_battery`` consumes (issue #303):
    there is no live classifier offline, so every no_answer gold item is
    simulated declining — retrieval-refusal items refuse, canned
    out-of-scope items route ``canned``. Honestly labelled offline
    (the payload's offline-simulated banner), never a measured result;
    the shared builder derives the refusal/false-refusal/canned gates
    from these exactly as the live path derives them from real runs."""
    results: list[ItemResult] = []
    for item in gold.qa_items:
        if item.get("category") != "no_answer":
            continue
        is_canned = item.get("expected_route") == "canned_out_of_scope"
        results.append(
            ItemResult(
                item_id=item["id"],
                arm_model=_OFFLINE_ARM_MODEL,
                route="canned" if is_canned else "retrieval",
                refused=True,
            )
        )
    return results


def run_offline_suite(
    qa_path: Path,
    charts_path: Path,
    out_dir: Path,
    *,
    plan_chart: Callable[[str], Any] | None = None,
) -> str:
    """The full deterministic suite on the Fake seam: validate gold, run
    the answer + chart paths, compute the deterministic metrics and every
    gate, and write the results artefacts (labelled offline-simulated).
    Returns the release verdict.

    ``plan_chart`` is injectable (finding #242): the chart gate inputs are
    DERIVED from its ACTUAL output vs gold, so a wrong planner fails the
    chart gate. The refusal/canned decline outcomes are simulated (there
    is no live classifier offline) but honestly labelled; the severity
    gate is BLOCKED while the owner-audit packet is pending, so the
    offline verdict is ``blocked`` — the fail-closed release behaviour.
    """
    gold = load_and_validate_gold(qa_path, charts_path)
    gold_by_id = {item["id"]: item for item in gold.qa_items}

    answerable = [item for item in gold.qa_items if item.get("category") in _ANSWERABLE_CATEGORIES]
    # The answered exchanges drive the REAL validate_exchange seam with the
    # deterministic offline validator, so the citation_support gate is
    # exercised (simulated, labelled) — the same seam production drives.
    deps = AnswerPathDeps(
        adapter=_OfflineAdapter(),
        retrieve=_offline_retrieve,
        validate_exchange=_offline_validate_exchange,
    )
    answer_results = run_answer_path(answerable, deps, arm_model=_OFFLINE_ARM_MODEL, mode="fake")

    planner = plan_chart if plan_chart is not None else _gold_driven_planner(gold.chart_items)
    chart_results = run_chart_path(gold.chart_items, planner)
    chart_records = chart_gate_records(gold.chart_items, chart_results)

    # THE ONE shared battery builder (issue #303): identical membership on
    # the offline and live paths. refusal/canned declines and the classifier
    # summary are simulated offline (no live classifier) and honestly
    # labelled; chart gates, faithfulness and citation_support are derived
    # from the actual run records; severity blocks on the pending owner audit.
    battery = build_gate_battery(
        gold,
        [*answer_results, *_simulated_decline_results(gold)],
        chart_records,
        chart_faithfulness_records=compute_chart_faithfulness_records(),
        classifier_summary=_OFFLINE_CLASSIFIER_SUMMARY,
    )

    arm = gates.ArmResult(model=_OFFLINE_ARM_MODEL, gates=tuple(battery), cost_usd=0.0)
    verdict = gates.release_verdict(list(arm.gates))
    selected = gates.select_production_model([arm])
    payload = report.build_results_payload(
        [arm], verdict=verdict, selected_model=selected, mode=_OFFLINE_MODE_LABEL
    )
    # Wire the deterministic metrics into the arm payload (finding #242).
    payload["arms"][0]["retrieval_metrics"] = compute_retrieval_metrics(answer_results, gold_by_id)
    payload["arms"][0]["calibrated_term_preserved_rate"] = compute_calibrated_term_rate(
        answer_results
    )

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
    ``--offline`` — the deterministic $0 suite (the default action).
    ``--live`` / ``--record`` — explicit opt-in for network-touching
    arms; both require an ANTHROPIC_API_KEY and a passing $9.00-cap
    pre-flight, and this entry point names the missing credential rather
    than fabricating a run.
    """
    parser = argparse.ArgumentParser(description="Release eval runner (issue #21).")
    parser.add_argument("--validate-gold-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--qa-gold", type=Path, default=CLIMATE_QA_PATH)
    parser.add_argument("--charts-gold", type=Path, default=CHART_REQUESTS_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write results.json + RESULTS.md; default is a scratch dir "
        "(publishing to evals/ is an explicit opt-in, finding #242)",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.validate_gold_only:
        load_and_validate_gold(args.qa_gold, args.charts_gold)
        print(f"gold OK: {args.qa_gold} + {args.charts_gold}")
        return 0

    if args.live or args.record:
        # Live/recording arms spend real money and need real credentials +
        # the recorded-run tooling; refuse fail-closed, naming the missing
        # credential (findings #236/#242) — never a fabricated run.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "live/recording release arms require ANTHROPIC_API_KEY in the "
                "environment (absent) plus the recorded-run tooling and a passing "
                "budget pre-flight — refusing to fabricate a run. Run --offline for "
                "the deterministic $0 suite.",
                file=sys.stderr,
            )
            return 1
        print(
            "live/recording release arms are driven by the recorded-run tooling, "
            "not this entry point; run --offline for the deterministic suite.",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="eval-offline-"))
    verdict = run_offline_suite(args.qa_gold, args.charts_gold, out_dir)
    print(f"OFFLINE / SIMULATED release verdict: {verdict.upper()} (artefacts in {out_dir})")
    return exit_code_for_verdict(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
