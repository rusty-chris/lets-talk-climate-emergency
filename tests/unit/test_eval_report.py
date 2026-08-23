"""Issue #21 red phase: results artefacts + the release exit code.

One run produces a machine-readable results file (results.json) and the
human RESULTS.md summary (the /about contract with #19, golden-pinned).
Gate pass/fail comes with per-item evidence links; the release verdict
is the conjunction, with pending-owner-audit reported as BLOCKED — and
the runner's exit code makes any non-passed verdict fail the release
build.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from evals.gates import GATE_BLOCKED, GATE_PASSED, ArmResult, GateResult
from evals.report import build_results_payload, render_results_md, write_results

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "evals" / "RESULTS_golden.md"

GOLDEN_PAYLOAD = {
    "schema_version": 1,
    "generated_on": "2026-08-23",
    "release_verdict": "blocked",
    "selected_model": None,
    "arms": [
        {
            "model": "claude-haiku-4-5",
            "cost_usd": 0.36,
            "gates": [
                {
                    "name": "refusal",
                    "status": "passed",
                    "numerator": 19,
                    "denominator": 20,
                    "threshold": 0.9,
                    "evidence": [{"item_id": "qa-na-g-01", "refused": False}],
                },
                {
                    "name": "severity",
                    "status": "blocked",
                    "reason": "owner severity audit pending (finding #197)",
                    "evidence": [],
                },
                {
                    "name": "chart_spec",
                    "status": "passed",
                    "numerator": 2,
                    "denominator": 2,
                    "evidence": [
                        {
                            "item_id": ("chart-15-flagship-spec-validation-refusal-of-commitment"),
                            "status": "skipped_blocked",
                            "blocked_reason": "fixtures embargoed until issue #23 (#117)",
                        }
                    ],
                },
            ],
        }
    ],
}


def test_results_md_golden_format():
    """The RESULTS.md rendering is byte-stable against the committed
    golden (the /about contract with #19): verdict spelled out (BLOCKED
    is never rendered as a pass), per-gate rows with evidence links
    into results.json, and the skipped-visibly flagship listed."""
    rendered = render_results_md(GOLDEN_PAYLOAD)
    assert rendered == GOLDEN_PATH.read_text(encoding="utf-8")
    assert "BLOCKED" in rendered
    assert "Release verdict: BLOCKED" in rendered
    assert "chart-15-flagship-spec-validation-refusal-of-commitment" in rendered
    assert "results.json#arms/claude-haiku-4-5/refusal" in rendered


def test_build_results_payload_is_machine_readable():
    arm = ArmResult(
        model="claude-haiku-4-5",
        gates=(
            GateResult(
                name="refusal",
                status=GATE_PASSED,
                numerator=19,
                denominator=20,
                threshold=0.9,
                evidence=({"item_id": "qa-na-g-01", "refused": False},),
            ),
            GateResult(name="severity", status=GATE_BLOCKED, reason="owner audit pending"),
        ),
        cost_usd=0.36,
    )
    payload = build_results_payload([arm], verdict="blocked", selected_model=None)
    # JSON-serialisable end to end — the machine-readable contract.
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["release_verdict"] == "blocked"
    assert round_tripped["selected_model"] is None
    (arm_payload,) = round_tripped["arms"]
    assert arm_payload["model"] == "claude-haiku-4-5"
    gate_names = [gate["name"] for gate in arm_payload["gates"]]
    assert gate_names == ["refusal", "severity"]
    refusal = arm_payload["gates"][0]
    assert refusal["evidence"][0]["item_id"] == "qa-na-g-01"


def test_write_results_writes_both_artefacts(tmp_path: Path):
    json_path = tmp_path / "results.json"
    md_path = tmp_path / "RESULTS.md"
    write_results(GOLDEN_PAYLOAD, json_path=json_path, md_path=md_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == GOLDEN_PAYLOAD
    assert md_path.read_text(encoding="utf-8") == GOLDEN_PATH.read_text(encoding="utf-8")


def _load_run_evals_module():
    path = REPO_ROOT / "scripts" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_build_exit_code_on_gate_violation():
    """Issue #21 TDD plan item 9: any gate violation fails the release
    build — and BLOCKED (pending owner audit) blocks it too; only a
    clean 'passed' verdict exits 0. Unknown verdicts fail closed."""
    run_evals = _load_run_evals_module()
    assert run_evals.exit_code_for_verdict("passed") == 0
    assert run_evals.exit_code_for_verdict("failed") != 0
    assert run_evals.exit_code_for_verdict("blocked") != 0
    assert run_evals.exit_code_for_verdict("mystery") != 0


def test_single_command_entrypoint_validates_gold_cheaply():
    """The acceptance criterion's one command is scripts/run_evals.py;
    its cheapest mode (--validate-gold-only) loads + validates the
    committed gold sets and exits 0 without running anything else —
    unit-tier checkable. The full offline suite is pinned at
    integration tier (issue #21 TDD plan item 11, green phase)."""
    run_evals = _load_run_evals_module()
    exit_code = run_evals.main(["--validate-gold-only"])
    assert exit_code == 0
