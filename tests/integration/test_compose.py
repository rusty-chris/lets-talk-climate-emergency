"""Integration test: docker-compose.yml is valid and names the stub services.

Issue #1 acceptance criterion: `docker compose config` parses and names the
`api`, `qdrant`, `ui` services. Requires the Docker CLI with the `compose`
plugin; skipped (not failed) when Docker is unavailable in the current
environment (IMPLEMENTATION.md §3 puts Docker-dependent checks at the
integration tier, never in unit).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "compose", "version"], capture_output=True)
    return result.returncode == 0


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="Docker (with the compose plugin) is not available in this environment",
)
def test_compose_config_valid() -> None:
    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert {"api", "qdrant", "ui"} <= set(config["services"].keys())
