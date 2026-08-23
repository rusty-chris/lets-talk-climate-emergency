"""Starter-question acceptance, smoke tier (issue #18, deferred from the RED wave).

The RED wave pinned the UI's pure core with synthetic events; this is the
one end-to-end test that proves those models fold a REAL service SSE
response. It drives the click-equivalent path exactly as the Streamlit
shell does — ``starter_submission`` -> the httpx ``ChatTransport`` ->
``stream_chat_events`` -> ``fold_chat_stream`` — against the COMPOSED
stack in its paused (read-only) state, and asserts the dated cached
starter answer renders through the ui pure modules with its citations.

Composition decision (flagged in the RED-phase notes, decided here):
docker-compose already runs a ``ui`` Streamlit service whose Docker
healthcheck the healthcheck smoke test covers. Rendering assertions,
though, want the folded view model, not scraped Streamlit HTML — so this
test runs the ui pure modules IN-PROCESS against the ``api`` service's
real SSE endpoint (the same modules the shell composes), rather than
scripting a browser against the ``ui`` container. No new compose service
is needed: the acceptance criterion is "the cached starter answer
renders through the real SSE path", and the pure fold IS that path.

Paused, like tests/smoke/test_cutoff_fails_closed.py: daily cap 0 forces
the read-only state, the committed synthetic dev cache serves the answer,
and there is no real API key — a single leaked live call would fail
loudly, never spend.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from service.starter_cache import STARTER_QUESTIONS
from tests._docker import require_docker
from ui.render_model import fold_chat_stream
from ui.sse_client import stream_chat_events
from ui.starter import starter_submission
from ui.transport import http_chat_transport

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
API = "http://localhost:8000"
STARTUP_TIMEOUT_S = 120
POLL_INTERVAL_S = 2

#: The paused (read-only) stack: cap 0 is a breach from the first request;
#: the committed synthetic dev cache is the default starter cache.
PAUSED_ENV = {
    "CLIMATE_CHAT_DAILY_BUDGET_USD": "0",
    "CLIMATE_CHAT_OPUS_SUBCAP_USD": "0",
    "CLIMATE_CHAT_CORPUS_VERSION": "corpus-smoke-0000",
    "CLIMATE_CHAT_CORPUS_VINTAGE": "2026-08-01",
    "CLIMATE_CHAT_SITE_URL": "http://localhost:8000",
    "CLIMATE_CHAT_QDRANT_URL": "http://qdrant:6333",
    "CLIMATE_CHAT_STARTER_CACHE_DIR": "/app/service/dev_starter_cache",
    "CLIMATE_CHAT_LOG_DIR": "/tmp/climate-chat-logs",
    # Presence-checked only; NOT a real key.
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
    raise AssertionError("api service never became healthy under the paused environment")


def test_starter_questions_end_to_end() -> None:
    require_docker()
    env = {**os.environ, **PAUSED_ENV}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    try:
        _wait_healthy(time.monotonic() + STARTUP_TIMEOUT_S)

        # The click-equivalent: a starter topic tap IS an immediate,
        # verbatim submission — routed through the real transport and the
        # ui pure fold, exactly as the Streamlit shell composes them.
        question = STARTER_QUESTIONS[0]
        submission = starter_submission(question)
        assert submission.question == question
        assert submission.history == ()

        transport = http_chat_transport(API)
        events = list(stream_chat_events(transport, submission.question, submission.history))
        view = fold_chat_stream(events)

        # The paused stack served the dated cached starter answer, folded
        # through the ui pure modules.
        assert view.mode == "paused"
        assert view.kind == "cached_starter"
        assert view.generated_on == "2026-08-15"
        assert view.text, "the cached starter answer text must render"
        assert view.complete is True
        assert view.footer_text, "the cached answer carries its dated footer"

        # Its citations became chips (the synthetic dev cache carries one
        # citation per starter answer) and a deduped source list — the
        # real cached-starter surface, end to end.
        assert view.chips, "cached starter citations must render as chips"
        first = view.chips[0]
        assert first.chunk_id and first.attribution and first.quote
        assert view.sources and view.sources[0].chunk_id == first.chunk_id
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False, env=env)
