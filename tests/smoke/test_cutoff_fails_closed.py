"""The DESIGN §10 release GATE, smoke tier (issue #22, ADR-015).

``test_cutoff_fails_closed_end_to_end``: the COMPOSED stack (`docker
compose up`), a simulated budget breach (daily cap 0 — spend == cap is a
breach, pinned at the unit tier), and the read-only behaviour verified
FROM OUTSIDE over real HTTP: chat answers with the dated paused
response and zero LLM traffic is possible (there is no real API key in
the environment — a single leaked call would fail loudly, not spend),
while /health, /about and the permalink route stay up.

Environment plumbing contract for the compose stack (the implementer
extends docker-compose.yml — extend, don't rewrite): the api service
passes through the ``CLIMATE_CHAT_*`` variables and ``ANTHROPIC_API_KEY``
from the host environment, with dev-safe defaults for the plain
``docker compose up`` of tests/smoke/test_healthchecks.py (the committed
synthetic dev cache at /app/service/dev_starter_cache serves as the
default starter cache).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from service.starter_cache import STARTER_QUESTIONS
from tests._docker import require_docker
from tests._service_fixtures import parse_sse

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
API = "http://localhost:8000"
STARTUP_TIMEOUT_S = 120
POLL_INTERVAL_S = 2

#: The simulated breach: cap 0 (spend 0 >= cap 0 — breached from the
#: first request; the boundary semantics are pinned at the unit tier).
BREACH_ENV = {
    "CLIMATE_CHAT_DAILY_BUDGET_USD": "0",
    "CLIMATE_CHAT_OPUS_SUBCAP_USD": "0",
    "CLIMATE_CHAT_CORPUS_VERSION": "corpus-smoke-0000",
    "CLIMATE_CHAT_CORPUS_VINTAGE": "2026-08-01",
    "CLIMATE_CHAT_SITE_URL": "http://localhost:8000",
    "CLIMATE_CHAT_QDRANT_URL": "http://qdrant:6333",
    "CLIMATE_CHAT_STARTER_CACHE_DIR": "/app/service/dev_starter_cache",
    "CLIMATE_CHAT_LOG_DIR": "/tmp/climate-chat-logs",
    # Presence-checked only; NOT a real key — any live call the paused
    # stack tried to make would fail authentication loudly.
    "ANTHROPIC_API_KEY": "smoke-placeholder-not-a-real-key",
}


def _wait_healthy(deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{API}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError("api service never became healthy under the breach environment")


def test_cutoff_fails_closed_end_to_end() -> None:
    require_docker()
    env = {**os.environ, **BREACH_ENV}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    try:
        _wait_healthy(time.monotonic() + STARTUP_TIMEOUT_S)

        # 1. Chat fails closed to the dated paused response — 200 + SSE,
        #    never an error page.
        response = httpx.post(
            f"{API}/chat",
            json={"question": "Why is the planet warming so fast?"},
            timeout=10,
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        answers = [event for event in events if event["event"] == "answer"]
        assert len(answers) == 1
        data = answers[0]["data"]
        assert data["kind"] == "paused"
        assert "paused for today" in data["text"]
        meta = [event for event in events if event["event"] == "meta"]
        assert meta and meta[0]["data"]["mode"] == "paused"

        # 2. A starter-topic question serves the cached (dev) answer.
        cached = httpx.post(f"{API}/chat", json={"question": STARTER_QUESTIONS[0]}, timeout=10)
        cached_events = parse_sse(cached.text)
        cached_answers = [event for event in cached_events if event["event"] == "answer"]
        assert cached_answers[0]["data"]["kind"] == "cached_starter"
        assert cached_answers[0]["data"]["generated_on"] == "2026-08-15"

        # 3. Chart generation refuses too (generation is generation).
        chart = httpx.post(
            f"{API}/chat",
            json={"question": "Show me a chart of CO2 since 1959"},
            timeout=10,
        )
        chart_answers = [event for event in parse_sse(chart.text) if event["event"] == "answer"]
        assert chart_answers[0]["data"]["kind"] == "paused"

        # 4. The read-only surfaces stay up from outside.
        assert httpx.get(f"{API}/health", timeout=5).json() == {"status": "ok"}
        for path in ("/about", "/privacy", "/sources", "/voices"):
            surface = httpx.get(f"{API}{path}", timeout=5)
            assert surface.status_code == 200, f"{path} must serve while paused"

        # 5. The permalink route is alive (unknown hash → clean 404,
        #    never 500/503 — the route itself needs no LLM and no index).
        assert httpx.get(f"{API}/chart/{'0' * 64}", timeout=5).status_code == 404

        # 6. Nothing personal leaked into the api logs for the paused
        #    exchange: the container's log output names no client IP.
        logs = subprocess.run(
            ["docker", "compose", "logs", "api"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        for line in logs.stdout.splitlines():
            if "paused" in line.lower() and line.strip().startswith("{"):
                record = json.loads(line[line.index("{") :])
                assert "ip" not in record and "client" not in record
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False, env=env)
