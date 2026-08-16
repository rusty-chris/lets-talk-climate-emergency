"""Smoke test: `docker compose up` brings each stub service to a responding
health endpoint (issue #1 acceptance criterion `test_stub_healthchecks_respond`).

Skipped (not failed) when Docker is unavailable in the current environment.
Always tears the stack back down, even on failure.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]

HEALTH_ENDPOINTS = {
    "api": "http://localhost:8000/health",
    "qdrant": "http://localhost:6333/healthz",
    "ui": "http://localhost:8501/_stcore/health",
}

STARTUP_TIMEOUT_S = 120
POLL_INTERVAL_S = 2


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "compose", "version"], capture_output=True)
    return result.returncode == 0


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="Docker (with the compose plugin) is not available in this environment",
)
def test_stub_healthchecks_respond() -> None:
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
    )
    try:
        pending = set(HEALTH_ENDPOINTS)
        last_errors: dict[str, str] = {}
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                try:
                    response = httpx.get(HEALTH_ENDPOINTS[name], timeout=2)
                    if response.status_code == 200:
                        pending.discard(name)
                    else:
                        last_errors[name] = f"HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_errors[name] = str(exc)
            if pending:
                time.sleep(POLL_INTERVAL_S)
        assert not pending, f"services never became healthy: {pending}; last errors: {last_errors}"
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False)
