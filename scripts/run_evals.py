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

Red phase: contract pinned, behaviour raises NotImplementedError.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def exit_code_for_verdict(verdict: str) -> int:
    """0 only for ``passed``; ``failed`` and ``blocked`` are non-zero
    (a blocked owner audit must block the release build, never slip
    through as success). Unknown verdicts are non-zero too — fail
    closed."""
    raise NotImplementedError("issue #21 green phase")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite; return the release exit code.

    Flags (pinned by tests/unit/test_eval_report.py):
    ``--validate-gold-only`` — load + validate the committed gold sets
    and exit 0 (the cheapest invocation; unit-tier checkable).
    ``--live`` / ``--record`` — explicit opt-in for network-touching
    arms; both require a passing $9.00-cap pre-flight.
    """
    raise NotImplementedError("issue #21 green phase")


if __name__ == "__main__":
    raise SystemExit(main())
