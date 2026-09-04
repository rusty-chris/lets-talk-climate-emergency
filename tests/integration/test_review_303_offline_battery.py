"""Issue #303 red phase (Fable): the one-command offline suite reports
the ENLARGED battery honestly.

After the #303 wiring (one shared battery builder; citation_support and
route_accuracy in the release contract), the single acceptance-criterion
command — ``scripts/run_evals.py --offline`` — must surface the enlarged
battery end-to-end: both new gates appear in results.json AND render as
rows in RESULTS.md, BLOCKED items render as BLOCKED, and the verdict
stays fail-closed (the pending owner severity audit keeps it BLOCKED,
exiting non-zero). No network, no key, synthetic gold only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._eval_harness_fixtures import write_synthetic_gold

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_offline(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "out"
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
    return result, out_dir


def test_offline_command_reports_the_enlarged_battery(tmp_path: Path) -> None:
    """One command, full contract: citation_support and route_accuracy
    join the battery in results.json and render as gate rows in
    RESULTS.md; the verdict stays BLOCKED (owner audit pending) and the
    release build exits non-zero."""
    result, out_dir = _run_offline(tmp_path)

    assert result.returncode != 0, result.stderr
    assert "BLOCKED" in result.stdout

    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["release_verdict"] == "blocked"
    (arm,) = payload["arms"]
    gate_names = {gate["name"] for gate in arm["gates"]}
    assert {"citation_support", "route_accuracy"} <= gate_names, (
        f"the offline battery {sorted(gate_names)} must carry the #303-wired "
        "citation_support and route_accuracy gates — the shared builder is the "
        "release contract, on every path"
    )

    # The offline suite simulates its verdict inputs honestly (the
    # offline-simulated banner is pinned elsewhere), so both wired gates
    # are exercised — never blocked-forever placeholders offline.
    citation = next(gate for gate in arm["gates"] if gate["name"] == "citation_support")
    assert citation["status"] == "passed", (
        "the offline suite must drive the validate_exchange seam with its "
        "deterministic offline validator so the citation gate is exercised "
        f"(simulated, labelled): got {citation['status']!r}"
    )
    assert citation.get("denominator"), "pooled factual sentences must be counted"
    route = next(gate for gate in arm["gates"] if gate["name"] == "route_accuracy")
    assert route["status"] == "passed", (
        "the offline suite must feed route_accuracy a simulated classifier "
        f"summary (labelled offline-simulated): got {route['status']!r}"
    )

    rendered = (out_dir / "RESULTS.md").read_text(encoding="utf-8")
    assert "| citation_support |" in rendered, "the new gate must render as a RESULTS.md row"
    assert "| route_accuracy |" in rendered, "the new gate must render as a RESULTS.md row"
    # Fail-closed rendering: the pending owner audit still renders BLOCKED.
    assert "| severity | BLOCKED |" in rendered
    assert "Release verdict: BLOCKED" in rendered
