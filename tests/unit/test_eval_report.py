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


# ---------------------------------------------------------------------------
# review-21 #242 (major): the default invocation never overwrites the
# published results location with simulated outcomes; offline runs are
# honestly derived and visibly labelled.
# ---------------------------------------------------------------------------


def test_default_out_dir_is_not_the_published_results_location(tmp_path: Path):
    """A bare offline invocation (no --out-dir) must NOT write
    evals/results.json / evals/RESULTS.md — the artefacts /about
    consumes (#19). Publishing there is an explicit opt-in, never the
    default (#242)."""
    from tests._eval_harness_fixtures import write_synthetic_gold

    run_evals = _load_run_evals_module()
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    published_paths = (
        REPO_ROOT / "evals" / "results.json",
        REPO_ROOT / "evals" / "RESULTS.md",
    )
    before = {path: (path.read_bytes() if path.exists() else None) for path in published_paths}
    try:
        run_evals.main(["--offline", "--qa-gold", str(qa_path), "--charts-gold", str(charts_path)])
        for path in published_paths:
            after = path.read_bytes() if path.exists() else None
            assert after == before[path], (
                f"the default invocation must never write the published artefact {path}"
            )
    finally:
        for path, content in before.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def test_offline_suite_gates_reflect_actual_run_outcomes(tmp_path: Path):
    """The chart gate input derives from COMPARING the planner's output
    to gold (match / mismatch / refused_with_nearest), never a
    hardwired 'match': a deliberately wrong planned spec yields a
    FAILED chart_spec gate, and a refusal-expected item is never a spec
    match (#242)."""
    from tests._eval_harness_fixtures import write_synthetic_gold

    run_evals = _load_run_evals_module()
    suite = getattr(run_evals, "run_offline_suite", None)
    assert suite is not None, (
        "scripts/run_evals.py must expose run_offline_suite(qa_path, charts_path, "
        "out_dir, *, plan_chart=...) so the offline gate inputs are derived from an "
        "injectable planner's ACTUAL output, not fabricated constants (#242)"
    )

    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "out"

    def wrong_planner(_request: str) -> dict[str, object]:
        return {"kind": "spec", "spec": {"chart_id": "deliberately-wrong", "chart_type": "bar"}}

    suite(qa_path, charts_path, out_dir, plan_chart=wrong_planner)

    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    (arm,) = payload["arms"]
    chart_gate = next(gate for gate in arm["gates"] if gate["name"] == "chart_spec")
    assert chart_gate["status"] == "failed", (
        "a planner returning a wrong spec must fail the chart gate — a constant "
        "'match' fabricates a pass"
    )
    statuses = {entry.get("item_id"): entry.get("status") for entry in chart_gate["evidence"]}
    assert statuses.get("syn-chart-01") == "mismatch"
    assert statuses.get("syn-chart-02") != "match", (
        "a refusal-expected chart item answered with a spec is never a spec match"
    )


def test_offline_results_are_labelled_simulated(tmp_path: Path):
    """Offline outcomes must be indistinguishable from measured results
    NOWHERE: the payload carries mode 'offline-simulated' and RESULTS.md
    renders a visible banner (#242)."""
    from tests._eval_harness_fixtures import write_synthetic_gold

    run_evals = _load_run_evals_module()
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "out"
    run_evals.main(
        [
            "--offline",
            "--qa-gold",
            str(qa_path),
            "--charts-gold",
            str(charts_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert payload.get("mode") == "offline-simulated"

    rendered = (out_dir / "RESULTS.md").read_text(encoding="utf-8")
    banner = rendered.upper()
    assert "OFFLINE" in banner and "SIMULATED" in banner


def test_live_flag_refuses_naming_missing_credentials(monkeypatch, capsys):
    """--live without ANTHROPIC_API_KEY refuses with a non-zero exit and
    an honest message naming the missing credential — never a silent
    fabricated run and never a misleading pointer (#236/#242)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run_evals = _load_run_evals_module()
    exit_code = run_evals.main(["--live"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in (captured.err + captured.out)
