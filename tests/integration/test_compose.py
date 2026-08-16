"""Integration test: docker-compose.yml is valid and names the stub services.

Issue #1 acceptance criterion: `docker compose config` parses and names the
`api`, `qdrant`, `ui` services. Requires the Docker CLI with the `compose`
plugin; skipped when Docker is unavailable locally, failed when it is
unavailable in CI (review finding #32 — a skip must not green a CI tier).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from tests._docker import require_docker

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compose_config_valid() -> None:
    require_docker()
    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert {"api", "qdrant", "ui"} <= set(config["services"].keys())
