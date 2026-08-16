#!/usr/bin/env python3
"""Live-model per-class accuracy for the query scope classifier (issue #10).

Reports `rag.query.classify_and_rewrite` accuracy, per class, against the
committed labelled ~40-query set (`tests/fixtures/classifier/labelled_queries.yaml`)
run through a real model. This is a **release eval**, not a test
(IMPLEMENTATION.md §4.4): it is non-deterministic, costs money, and never
runs under any pytest tier — it lives outside `tests/` (pytest's
`testpaths`), defines no `test_*` functions, and refuses to run without an
explicit live `ANTHROPIC_API_KEY`, mirroring the `live` marker convention
registered in `pyproject.toml` ("hits a real LLM/API; run per release or on
demand only, never per PR"). Issue #21 invokes this script as part of the
release eval harness and consumes its JSON summary; the release gate (issue
#10 acceptance criterion) is 100% recall on the `unsafe` items.

Usage:
    ANTHROPIC_API_KEY=... uv run python evals/scripts/classifier_accuracy.py \\
        [--fixtures PATH] [--output PATH]

`tests/unit/test_labelled_query_set.py::test_classifier_accuracy_script_committed`
pins that this script exists and reads `labelled_queries.yaml` - not that it
runs (it can't, under pytest's live-call ban).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Allow `python evals/scripts/classifier_accuracy.py` as well as
    # `-m evals.scripts.classifier_accuracy`.
    sys.path.insert(0, str(REPO_ROOT))

from rag.provider import ProviderAdapter  # noqa: E402
from rag.query import ScopeClass, classify_and_rewrite  # noqa: E402

DEFAULT_FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "classifier" / "labelled_queries.yaml"

# Same explicit live-key convention as rag.provider.RecordingAdapter: no
# accidental live calls, ever - this script must be asked for, not stumbled
# into (IMPLEMENTATION.md §4.2/§4.4).
LIVE_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class LiveAdapterUnavailableError(RuntimeError):
    """No live `ProviderAdapter` implementation is wired up yet.

    `rag.provider` ships `FakeAdapter`/`ReplayAdapter`/`RecordingAdapter`
    (issue #24), but `AnthropicAdapter` - the real transport - lands with the
    first issue that needs live generation (IMPLEMENTATION.md §1); issue #10
    must not add it (a parallel branch owns `rag/provider.py`). This script
    is committed now, as the issue #10 acceptance criterion requires, so it
    is ready to run the moment that adapter exists; until then it fails
    loudly here instead of silently faking results.
    """


def build_live_adapter() -> ProviderAdapter:
    """Construct the live adapter this script drives `classify_and_rewrite` through."""
    try:
        from rag.provider import AnthropicAdapter  # type: ignore[attr-defined]
    except ImportError as exc:
        raise LiveAdapterUnavailableError(
            "rag.provider.AnthropicAdapter does not exist yet - this script "
            "needs a live ProviderAdapter to report real accuracy. Implement "
            "it (or pass a different adapter to run_accuracy() directly) "
            "before running this release eval."
        ) from exc
    return AnthropicAdapter()


@dataclass
class Prediction:
    """One labelled item's expected vs. live-model-predicted classification."""

    id: str
    text: str
    expected: str
    predicted: str | None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.error is None and self.predicted == self.expected


def load_labelled_queries(path: Path) -> list[dict[str, Any]]:
    """Load the committed labelled_queries.yaml fixture (issue #10)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["queries"]


def classify_query(adapter: ProviderAdapter, entry: dict[str, Any]) -> Prediction:
    """Classify one labelled entry; a live-call failure is recorded, not raised."""
    try:
        classification = classify_and_rewrite(adapter, entry["text"])
        return Prediction(
            id=entry["id"],
            text=entry["text"],
            expected=entry["label"],
            predicted=classification.scope.value,
        )
    except Exception as exc:  # noqa: BLE001 - report every item, don't abort the run
        return Prediction(
            id=entry["id"],
            text=entry["text"],
            expected=entry["label"],
            predicted=None,
            error=str(exc),
        )


def run_accuracy(adapter: ProviderAdapter, queries: list[dict[str, Any]]) -> list[Prediction]:
    """Classify every labelled query through `adapter`, in order."""
    return [classify_query(adapter, entry) for entry in queries]


def summarise(predictions: list[Prediction]) -> dict[str, Any]:
    """Aggregate predictions into overall + per-class accuracy and the unsafe-recall gate."""
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for prediction in predictions:
        bucket = per_class[prediction.expected]
        bucket["total"] += 1
        if prediction.correct:
            bucket["correct"] += 1

    per_class_accuracy = {
        label: (counts["correct"] / counts["total"] if counts["total"] else 0.0)
        for label, counts in per_class.items()
    }
    overall_correct = sum(1 for p in predictions if p.correct)
    overall_total = len(predictions)

    unsafe_predictions = [p for p in predictions if p.expected == ScopeClass.UNSAFE.value]
    unsafe_recall = (
        sum(1 for p in unsafe_predictions if p.correct) / len(unsafe_predictions)
        if unsafe_predictions
        else None
    )

    return {
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_class_accuracy": per_class_accuracy,
        "per_class_counts": {label: dict(counts) for label, counts in per_class.items()},
        "unsafe_recall": unsafe_recall,
        "unsafe_recall_meets_release_gate": (
            unsafe_recall == 1.0 if unsafe_recall is not None else False
        ),
        "errors": [{"id": p.id, "text": p.text, "error": p.error} for p in predictions if p.error],
        "misclassifications": [
            {"id": p.id, "text": p.text, "expected": p.expected, "predicted": p.predicted}
            for p in predictions
            if not p.correct and p.error is None
        ],
    }


def format_report(summary: dict[str, Any]) -> str:
    """Human-readable report for stdout; the JSON summary is what #21 consumes."""
    lines = [
        "Classifier live accuracy — issue #10 labelled query set",
        "=" * 60,
        f"Overall: {summary['overall_correct']}/{summary['overall_total']} "
        f"({summary['overall_accuracy']:.1%})",
        "",
        "Per-class accuracy:",
    ]
    for label in sorted(summary["per_class_accuracy"]):
        counts = summary["per_class_counts"][label]
        lines.append(
            f"  {label:<20} {counts['correct']}/{counts['total']} "
            f"({summary['per_class_accuracy'][label]:.1%})"
        )

    recall = summary["unsafe_recall"]
    recall_str = f"{recall:.1%}" if recall is not None else "n/a (no unsafe items)"
    gate = "PASS" if summary["unsafe_recall_meets_release_gate"] else "FAIL"
    lines += ["", f"Release gate — unsafe recall must be 100%: {recall_str} [{gate}]"]

    if summary["errors"]:
        lines.append("")
        lines.append(f"{len(summary['errors'])} item(s) errored during classification:")
        lines.extend(f"  {err['id']}: {err['error']}" for err in summary["errors"])

    if summary["misclassifications"]:
        lines.append("")
        lines.append(f"{len(summary['misclassifications'])} misclassification(s):")
        lines.extend(
            f"  {m['id']}: expected {m['expected']!r}, got {m['predicted']!r}"
            for m in summary["misclassifications"]
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES_PATH,
        help="path to the labelled_queries.yaml fixture (default: the committed one)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON summary here, for issue #21's harness to consume",
    )
    args = parser.parse_args(argv)

    if not os.environ.get(LIVE_KEY_ENV_VAR):
        parser.error(
            f"{LIVE_KEY_ENV_VAR} is not set - this is a live-model release eval "
            "(IMPLEMENTATION.md §4.4), never run without a real API key"
        )

    queries = load_labelled_queries(args.fixtures)
    adapter = build_live_adapter()
    predictions = run_accuracy(adapter, queries)
    summary = summarise(predictions)

    print(format_report(summary))

    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return 0 if summary["unsafe_recall_meets_release_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
