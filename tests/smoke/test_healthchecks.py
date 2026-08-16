"""Smoke tests: `docker compose up` brings each stub service to a responding
health endpoint (issue #1 acceptance criterion `test_stub_healthchecks_respond`),
and Docker's own healthchecks report every service healthy (review finding
#33 — otherwise the compose `healthcheck:` blocks are unverified decoration
that nothing consumes and nothing observes).

Skipped when Docker is unavailable locally, failed when it is unavailable
in CI (review finding #32 — a skip must not green a CI tier). Always tears
the stack back down, even on failure.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from tests._docker import require_docker

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]

HEALTH_ENDPOINTS = {
    "api": "http://localhost:8000/health",
    "qdrant": "http://localhost:6333/healthz",
    "ui": "http://localhost:8501/_stcore/health",
}

STARTUP_TIMEOUT_S = 120
POLL_INTERVAL_S = 2


def test_stub_healthchecks_respond() -> None:
    require_docker()
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


def _compose_health_by_service() -> dict[str, str]:
    """Map service name -> Docker health status from `docker compose ps`."""
    result = subprocess.run(
        ["docker", "compose", "ps", "--format", "json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    # Compose v2 emits one JSON object per line.
    health: dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        health[record["Service"]] = record.get("Health", "")
    return health


def test_compose_reports_services_healthy() -> None:
    """Docker's own healthchecks converge to `healthy` for every service.

    The host-side endpoint test above never looks at Docker's health
    status, so a healthcheck could be permanently unhealthy (wrong URL,
    missing binary in the image, readiness-vs-liveness confusion) while
    every test stays green — and #10/#22 will build startup ordering and
    monitoring on these checks assuming they are load-bearing. This test
    makes all three `healthcheck:` blocks verified config: it fails
    whenever a healthcheck itself is broken, even if the service happens
    to answer from the host.
    """
    require_docker()
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
    )
    try:
        expected = set(HEALTH_ENDPOINTS)
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        health: dict[str, str] = {}
        while time.monotonic() < deadline:
            health = _compose_health_by_service()
            if {name: status for name, status in health.items() if name in expected} == {
                name: "healthy" for name in expected
            }:
                break
            time.sleep(POLL_INTERVAL_S)
        unhealthy = {name: health.get(name, "<absent>") for name in expected}
        assert all(status == "healthy" for status in unhealthy.values()), (
            f"docker compose never reported all services healthy: {unhealthy}"
        )
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False)
