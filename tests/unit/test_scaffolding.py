"""Meta-tests for the repo scaffolding (issue #1).

These assert the harness itself works: every top-level package imports
cleanly, and the integration/smoke/live pytest markers described in
IMPLEMENTATION.md §3 are registered and enforced under --strict-markers.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

TOP_LEVEL_PACKAGES = ["ingestion", "rag", "charts", "ui", "evals", "service"]


@pytest.mark.parametrize("package_name", TOP_LEVEL_PACKAGES)
def test_package_imports(package_name: str) -> None:
    """Every top-level package (issue #1 acceptance criterion) imports cleanly."""
    module = importlib.import_module(package_name)
    assert module is not None


def test_pytest_markers_registered() -> None:
    """integration/smoke/live markers are declared in pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    ini_options = pyproject["tool"]["pytest"]["ini_options"]
    registered = {line.split(":", 1)[0].strip() for line in ini_options["markers"]}
    assert {"integration", "smoke", "live"} <= registered
    assert "--strict-markers" in ini_options["addopts"]


def test_unknown_marker_errors_under_strict_markers(tmp_path: Path) -> None:
    """An unregistered marker fails collection, proving --strict-markers is enforced."""
    bad_test = tmp_path / "test_bad_marker.py"
    bad_test.write_text(
        "import pytest\n\n"
        "@pytest.mark.this_marker_is_not_registered\n"
        "def test_noop():\n"
        "    assert True\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            "--collect-only",
            str(bad_test),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "this_marker_is_not_registered" in result.stdout + result.stderr
