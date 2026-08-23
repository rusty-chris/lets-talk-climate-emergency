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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON_PATH = REPO_ROOT / "evals" / "results.json"
RESULTS_MD_PATH = REPO_ROOT / "evals" / "RESULTS.md"


def build_results_payload(
    arms: Sequence[Any],
    *,
    verdict: str,
    selected_model: str | None,
    gold_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The machine-readable results mapping (JSON-serialisable).

    Must carry: schema version, per-arm gates (name, status, n/d,
    threshold, per-item evidence with item ids), skipped-visibly items
    (the #23/#117 flagship) with reasons, costs per arm, the selection
    (or the no-model-passed escalation record), and the verdict.
    """
    raise NotImplementedError("issue #21 green phase")


def render_results_md(payload: Mapping[str, Any]) -> str:
    """RESULTS.md text for one payload — byte-stable for a given
    payload (golden-pinned). BLOCKED gates render as BLOCKED, never as
    a pass; every gate row links its evidence anchor in results.json."""
    raise NotImplementedError("issue #21 green phase")


def write_results(
    payload: Mapping[str, Any],
    *,
    json_path: Path = RESULTS_JSON_PATH,
    md_path: Path = RESULTS_MD_PATH,
) -> None:
    """Write results.json and RESULTS.md for one release run."""
    raise NotImplementedError("issue #21 green phase")
