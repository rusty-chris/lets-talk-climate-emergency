"""Release-eval reporting (issue #21): one machine-readable results
file + the human RESULTS.md summary (consumed by /about, issue #19).

Contract:
- ``build_results_payload`` produces a plain-JSON-serialisable mapping:
  per-arm gate results with per-item evidence, the bake-off selection,
  costs, and the release verdict (the conjunction of gates — with
  pending-owner-audit reported as BLOCKED, never passed).
- ``render_results_md`` renders the payload to the RESULTS.md format
  pinned by ``test_results_md_golden_format`` (the /about contract with
  #19): a verdict line (PASSED / FAILED / **BLOCKED** spelled out), a
  per-arm gate table, and per-item evidence links back into the
  machine-readable file.
- ``write_results`` writes both artefacts atomically side by side.

Red phase: contracts pinned, behaviour raises NotImplementedError.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON_PATH = REPO_ROOT / "evals" / "results.json"
RESULTS_MD_PATH = REPO_ROOT / "evals" / "RESULTS.md"

RESULTS_SCHEMA_VERSION = 1

_EM_DASH = "—"
_BLANK = "—"  # empty cell


def _gate_to_dict(gate: Any) -> dict[str, Any]:
    """One GateResult as a JSON-serialisable mapping (evidence a list)."""
    return {
        "name": gate.name,
        "status": gate.status,
        "numerator": gate.numerator,
        "denominator": gate.denominator,
        "threshold": gate.threshold,
        "reason": gate.reason,
        "evidence": [dict(entry) for entry in gate.evidence],
    }


def build_results_payload(
    arms: Sequence[Any],
    *,
    verdict: str,
    selected_model: str | None,
    gold_coverage: Mapping[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """The machine-readable results mapping (JSON-serialisable).

    Must carry: schema version, per-arm gates (name, status, n/d,
    threshold, per-item evidence with item ids), skipped-visibly items
    (the #23/#117 flagship) with reasons, costs per arm, the selection
    (or the no-model-passed escalation record), and the verdict.
    """
    payload: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_on": date.today().isoformat(),
        "release_verdict": verdict,
        "selected_model": selected_model,
        "arms": [
            {
                "model": arm.model,
                "cost_usd": arm.cost_usd,
                "gates": [_gate_to_dict(gate) for gate in arm.gates],
                # Review #317: a DNF-unaffordable arm reports its verdict +
                # reason (never silently dropped); ordinary arms omit them.
                **({"arm_verdict": arm.arm_verdict} if getattr(arm, "arm_verdict", None) else {}),
                **({"reason": arm.reason} if getattr(arm, "reason", None) else {}),
            }
            for arm in arms
        ],
    }
    if gold_coverage is not None:
        payload["gold_coverage"] = dict(gold_coverage)
    if mode is not None:
        # Report honesty (finding #242): simulated runs are labelled as
        # such — never rendered indistinguishably from measured results.
        payload["mode"] = mode
    return payload


def _score_cell(gate: Mapping[str, Any]) -> str:
    numerator = gate.get("numerator")
    denominator = gate.get("denominator")
    if numerator is None or denominator is None:
        return _BLANK
    return f"{numerator}/{denominator}"


def _threshold_cell(gate: Mapping[str, Any]) -> str:
    threshold = gate.get("threshold")
    return _BLANK if threshold is None else f"{threshold}"


def render_results_md(payload: Mapping[str, Any]) -> str:
    """RESULTS.md text for one payload — byte-stable for a given
    payload (golden-pinned). BLOCKED gates render as BLOCKED, never as
    a pass; every gate row links its evidence anchor in results.json."""
    lines: list[str] = [
        "# Release eval results",
        "",
        f"Generated: {payload['generated_on']} (results schema v{payload['schema_version']})",
        "",
        f"## Release verdict: {payload['release_verdict'].upper()}",
        "",
        f"Production model: {payload['selected_model'] or 'NONE SELECTED'}",
        "",
    ]
    if payload.get("mode") == "offline-simulated":
        # A visible banner (finding #242): these gate outcomes are
        # simulated on the deterministic offline seam, not measured
        # against the live model.
        lines.insert(
            4,
            "> OFFLINE / SIMULATED RESULTS — gate outcomes are simulated on the "
            "deterministic offline seam, not measured against the live model.",
        )
        lines.insert(5, "")
    skipped: list[tuple[str, Any, Any]] = []
    for arm in payload["arms"]:
        model = arm["model"]
        lines.append(f"## Arm: {model} {_EM_DASH} run cost ${arm['cost_usd']:.2f}")
        lines.append("")
        lines.append("| gate | status | score | threshold | evidence |")
        lines.append("|---|---|---|---|---|")
        for gate in arm["gates"]:
            name = gate["name"]
            status = gate["status"].upper()
            if gate["status"] == "blocked":
                evidence_cell = gate.get("reason") or _BLANK
            else:
                evidence_cell = f"[results.json](results.json#arms/{model}/{name})"
            lines.append(
                f"| {name} | {status} | {_score_cell(gate)} | "
                f"{_threshold_cell(gate)} | {evidence_cell} |"
            )
            for entry in gate.get("evidence", ()):
                if isinstance(entry, Mapping) and entry.get("status") == "skipped_blocked":
                    skipped.append((name, entry.get("item_id"), entry.get("blocked_reason")))
        lines.append("")
    lines.append("Skipped-visibly:")
    lines.append("")
    for gate_name, item_id, reason in skipped:
        lines.append(f"- {gate_name} / {item_id} {_EM_DASH} {reason}")
    return "\n".join(lines) + "\n"


def write_results(
    payload: Mapping[str, Any],
    *,
    json_path: Path = RESULTS_JSON_PATH,
    md_path: Path = RESULTS_MD_PATH,
) -> None:
    """Write results.json and RESULTS.md for one release run."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_results_md(payload), encoding="utf-8")
