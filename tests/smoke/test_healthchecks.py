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
import re
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


def _health_check_log(service: str) -> str:
    """Docker's healthcheck attempt log for a service — why it is unhealthy.

    Without this, an unhealthy service in CI is undiagnosable from the
    assertion message alone (learned the hard way when the qdrant probe
    failed only inside the container).
    """
    ps = subprocess.run(
        ["docker", "compose", "ps", "-a", "-q", service],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    container_id = ps.stdout.strip().splitlines()[0] if ps.stdout.strip() else ""
    if not container_id:
        return "<no container>"
    inspect = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State.Health}}", container_id],
        capture_output=True,
        text=True,
    )
    return inspect.stdout.strip() or inspect.stderr.strip()


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
        statuses = {name: health.get(name, "<absent>") for name in expected}
        if any(status != "healthy" for status in statuses.values()):
            details = {
                name: _health_check_log(name)
                for name, status in statuses.items()
                if status != "healthy"
            }
            raise AssertionError(
                f"docker compose never reported all services healthy: {statuses}; "
                f"healthcheck logs: {details}"
            )
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False)


#: An access-log-shaped line: `<ipv4>:<port> - "` (uvicorn's default
#: `%(client_addr)s - "%(request_line)s"` format). Startup lines like
#: `Uvicorn running on http://0.0.0.0:8000` do not match (no ` - "`).
_CLIENT_ADDR_LINE_RE = re.compile(r'\d{1,3}(?:\.\d{1,3}){3}:\d+ - "')

#: A header a misconfigured proxy-trusting server would log as the
#: client address (TEST-NET-3 — never a real peer on this stack).
_PROBE_FORWARDED_IP = "203.0.113.77"


def test_access_log_contains_no_raw_client_ip() -> None:
    """Issue #212 (DESIGN §9): the composed api's log output carries no
    raw client IP for any handled request.

    The app's own logging is already IP-free (unit-pinned); this drives
    the REAL server in the composed stack — where uvicorn's default
    access log, not application code, is what leaks `client_addr` — and
    scans `docker compose logs api` for (a) the forwarded-IP probe
    value, (b) any access-log-shaped `<ip>:<port> - "` line, and (c) any
    logged request line at all (the ratified #212 mechanism is access
    logging OFF, not reformatted).
    """
    require_docker()
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
    )
    try:
        api_health = HEALTH_ENDPOINTS["api"]
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        last_error = ""
        while time.monotonic() < deadline:
            try:
                if httpx.get(api_health, timeout=2).status_code == 200:
                    break
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(POLL_INTERVAL_S)
        else:
            raise AssertionError(f"api never became healthy: {last_error}")

        # Traffic whose handling would produce access-log lines: the
        # health/static surfaces, plus a chat POST carrying a forwarded
        # IP (the dev stack has no live key, so /chat may error — the
        # request still reached the server, which is all this needs).
        httpx.get(api_health, headers={"x-forwarded-for": _PROBE_FORWARDED_IP}, timeout=5)
        httpx.get("http://localhost:8000/about", timeout=5)
        try:
            httpx.post(
                "http://localhost:8000/chat",
                json={"question": "a synthetic smoke question"},
                headers={"x-forwarded-for": _PROBE_FORWARDED_IP},
                timeout=10,
            )
        except httpx.HTTPError:
            pass

        logs_result = subprocess.run(
            ["docker", "compose", "logs", "api"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        logs = logs_result.stdout + logs_result.stderr
        assert _PROBE_FORWARDED_IP not in logs, (
            "the forwarded client IP appeared in the api container logs "
            "(issue #212: raw IPs must never be logged)"
        )
        offending = [line for line in logs.splitlines() if _CLIENT_ADDR_LINE_RE.search(line)]
        assert not offending, (
            f"access-log-shaped lines carrying client addresses found in the "
            f"api container logs (issue #212): {offending[:5]}"
        )
        request_lines = [
            line
            for line in logs.splitlines()
            if '"GET /health' in line or '"GET /about' in line or '"POST /chat' in line
        ]
        assert not request_lines, (
            f"per-request access-log lines found — access logging must be "
            f"disabled (ratified #212 mechanism): {request_lines[:5]}"
        )
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False)
