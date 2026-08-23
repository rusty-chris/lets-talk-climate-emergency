"""Issue #21 TDD plan item 11 (green phase): the one-command offline suite.

The acceptance criterion's single command — ``scripts/run_evals.py`` in
``--offline`` mode — must run the WHOLE release-eval suite end-to-end on
the Fake/Replay provider seam against gold sets: gold validation, the
answer + chart paths, every release gate, the bake-off selection, and
both results artefacts (results.json + RESULTS.md). It touches no
network and needs no LLM key (IMPLEMENTATION.md §4.4), so it is an
integration-tier test that actually RUNS in CI rather than skipping
(review finding #32 — a skip must never green a CI tier).

The committed owner severity-audit packet still says pending, so the
suite's verdict is BLOCKED and the command exits non-zero — the
fail-closed release behaviour, proven by one command.
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
    and writes both artefacts; the pending owner audit makes the verdict
    BLOCKED, so the release build exits non-zero — no network, no key."""
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

    # BLOCKED (owner audit pending) exits non-zero — the release blocks.
    assert result.returncode != 0, result.stderr
    assert "BLOCKED" in result.stdout

    json_path = out_dir / "results.json"
    md_path = out_dir / "RESULTS.md"
    assert json_path.is_file(), "the machine-readable results file must be written"
    assert md_path.is_file(), "the human RESULTS.md summary must be written"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["release_verdict"] == "blocked"
    assert payload["selected_model"] is None
    (arm,) = payload["arms"]
    gate_names = {gate["name"] for gate in arm["gates"]}
    # The full gate battery ran: refusal, severity (blocked), charts, voices.
    assert {"refusal", "severity", "chart_spec", "voices_separation"} <= gate_names
    severity = next(gate for gate in arm["gates"] if gate["name"] == "severity")
    assert severity["status"] == "blocked"

    rendered = md_path.read_text(encoding="utf-8")
    assert "Release verdict: BLOCKED" in rendered
    # The blocked flagship is visible, never silently dropped.
    assert "Skipped-visibly:" in rendered
    assert "syn-chart-flagship" in rendered
