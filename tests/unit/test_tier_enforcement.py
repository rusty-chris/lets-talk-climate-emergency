"""Meta-tests enforcing test-tier membership (review finding #31).

The tier convention (pyproject `addopts`/`markers` + the
tests/unit|integration|smoke directory layout) was previously enforced by
nothing: an unmarked test under tests/integration/ ran in the unit tier,
a unit test mismarked `integration` silently never ran anywhere, and
`-m integration` did not exclude `live` tests. These tests make the
convention load-bearing:

- every collected item runs in exactly the tier its directory dictates;
- an unmarked file dropped into tests/integration/ or tests/smoke/ is
  auto-marked by that directory's conftest (forgotten markers cannot
  leak Docker work into the unit tier, nor vanish from their own tier);
- the CI integration/smoke jobs exclude `live`, so a mismarked live test
  can never make real API calls in a per-PR job.

Unit tier: subprocess collect-only runs, no Docker, a few seconds.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Selects every test regardless of markers (overrides the addopts -m).
SELECT_ALL = "integration or not integration"


def _collect(marker_expr: str | None, *paths: str) -> set[str]:
    """Return the node ids `pytest --collect-only` yields for a marker expression.

    `marker_expr=None` means the default run (the addopts in
    pyproject.toml apply, i.e. the unit tier).
    """
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    if marker_expr is not None:
        cmd += ["-m", marker_expr]
    cmd += list(paths)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    # 5 == no tests collected, legitimate for an empty selection.
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(("=", "<"))
    }


def _tier_dir(node_id: str) -> str | None:
    for tier in ("unit", "integration", "smoke"):
        if node_id.startswith(f"tests/{tier}/"):
            return tier
    return None


def test_collected_items_match_directory_tier() -> None:
    """Every test in the tree is selected by exactly the tier its directory dictates.

    Catches both failure modes finding #31 verified empirically:
    an unmarked test in tests/integration/ running in the unit tier
    (hole 1), and a unit test mismarked `integration`/`smoke`/`live`
    silently deselected from every CI tier (hole 2).
    """
    all_items = _collect(SELECT_ALL)
    assert all_items, "collected nothing at all — the collect helper is broken"

    selections = {
        "unit": _collect(None),
        "integration": _collect("integration"),
        "smoke": _collect("smoke"),
    }

    # No tier selection may reach outside its directory.
    for tier, items in selections.items():
        strays = {i for i in items if _tier_dir(i) != tier}
        assert not strays, f"items selected in the {tier} tier from outside tests/{tier}/: {strays}"

    # Every item must run in the tier its directory dictates.
    for item in all_items:
        tier = _tier_dir(item)
        assert tier is not None, f"test outside the tier directories: {item}"
        assert item in selections[tier], (
            f"{item} is not selected by its directory's tier ({tier}) — "
            "a mismarked or unmarked test that would silently run in the "
            "wrong tier or never run at all"
        )


@pytest.mark.parametrize("tier", ["integration", "smoke"])
def test_unmarked_test_in_tier_directory_is_auto_marked(tier: str) -> None:
    """A file with no pytestmark under tests/<tier>/ still belongs to that tier.

    Before the per-directory conftests existed, a forgotten marker meant
    the test ran in the unit tier (violating its no-Docker contract) and
    never ran in its own tier. The directory conftest must auto-apply
    the tier marker to make that failure mode impossible.
    """
    probe = REPO_ROOT / "tests" / tier / f"test_tier_probe_{uuid.uuid4().hex}.py"
    probe.write_text("def test_probe() -> None:\n    assert True\n")
    try:
        rel = str(probe.relative_to(REPO_ROOT))
        in_tier = _collect(tier, rel)
        assert in_tier, f"unmarked probe in tests/{tier}/ was not selected by -m {tier}"
        in_unit = _collect(None, rel)
        assert not in_unit, f"unmarked probe in tests/{tier}/ leaked into the unit tier: {in_unit}"
    finally:
        probe.unlink()


def test_ci_excludes_live_from_integration_and_smoke_jobs() -> None:
    """The CI integration/smoke jobs must never select `live`-marked tests.

    `-m integration` alone selects a test marked both `integration` and
    `live` (verified in finding #31, hole 3) — real LLM/API calls, cost
    and key exposure inside a per-PR job, which IMPLEMENTATION.md §4
    forbids. The workflow must run `-m "<tier> and not live"`.
    """
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    for job, expr in [("integration", "integration and not live"), ("smoke", "smoke and not live")]:
        steps = workflow["jobs"][job]["steps"]
        pytest_runs = [s["run"] for s in steps if "pytest" in s.get("run", "")]
        assert pytest_runs, f"CI job {job!r} has no pytest step"
        assert any(f'-m "{expr}"' in run for run in pytest_runs), (
            f'CI job {job!r} must run pytest -m "{expr}"; got: {pytest_runs}'
        )
