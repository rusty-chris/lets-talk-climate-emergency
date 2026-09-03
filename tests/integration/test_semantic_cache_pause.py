"""The cached response serves during a budget pause (issue #57, integration).

The issue-named ``test_cached_response_serves_during_budget_pause``
tier: uvicorn serves the composed fake-seam app on a loopback socket
and the UI's REAL transports drive the whole loop — one live grounded
exchange warms the semantic cache, the daily budget is then breached
(the fail-closed PAUSED state), and the SAME question comes back as the
clearly-marked cached answer with zero further adapter calls. No
Docker, no LLM key, no network beyond the loopback socket: the embedder
is the deterministic LOCAL hash fake.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from service.budget import ServiceMode
from service.semantic_cache import SemanticCache
from tests._generation_fixtures import transport_stream_events
from tests._indexing_fixtures import HashEmbeddingModel
from tests._service_fixtures import (
    CORPUS_VERSION,
    FrozenClock,
    classifier_output,
    make_harness,
)
from ui.render_model import (
    VIEW_KIND_CACHED,
    VIEW_KIND_GROUNDED,
    feedback_widget_model,
    fold_chat_stream,
)
from ui.sse_client import stream_chat_events
from ui.transport import http_chat_transport

pytestmark = pytest.mark.integration

SERVER_STARTUP_TIMEOUT_S = 15

QUESTION = "Why are scientists calling this an emergency?"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(harness):
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app=harness.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn never started"
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server, thread


def test_cached_response_serves_during_budget_pause(tmp_path) -> None:
    clock = FrozenClock()
    cache = SemanticCache(
        embedding_model=HashEmbeddingModel(),
        corpus_version=CORPUS_VERSION,
        clock=clock,
    )
    harness = make_harness(tmp_path, clock=clock, semantic_cache=cache)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", transport_stream_events())
    base_url, server, thread = _serve(harness)
    try:
        transport = http_chat_transport(base_url)

        # 1. One live grounded exchange warms the cache over the real wire.
        live_view = fold_chat_stream(list(stream_chat_events(transport, QUESTION)))
        assert live_view.kind == VIEW_KIND_GROUNDED
        assert live_view.complete is True
        calls_after_warm = len(harness.adapter.calls)

        # 2. Breach the daily cap: the tracker flips to the fail-closed
        #    read-only state (spend >= cap is a breach).
        harness.tracker.record_usage(
            "claude-opus-4-8",
            {"input_tokens": 100_000_000, "output_tokens": 10_000_000},
        )
        assert harness.tracker.mode() is ServiceMode.PAUSED

        # 3. The same question now serves from the semantic cache: $0,
        #    zero adapter calls, clearly marked as cached and dated with
        #    the ORIGINAL answer's date — never presented as fresh.
        cached_view = fold_chat_stream(list(stream_chat_events(transport, QUESTION)))
        assert cached_view.kind == VIEW_KIND_CACHED
        assert cached_view.mode == "paused"
        assert cached_view.complete is True
        assert cached_view.text == live_view.text
        assert cached_view.chips == live_view.chips
        assert cached_view.sources_panel == live_view.sources_panel
        assert cached_view.generated_on == clock.now.date().isoformat()
        assert len(harness.adapter.calls) == calls_after_warm, (
            "a paused-mode cache hit makes zero adapter calls — the pause "
            "stays a hard no-spend state"
        )

        # 4. The paused serving is its own feedback-able exchange (#56).
        widget = feedback_widget_model(cached_view)
        assert widget is not None
        assert widget.exchange_id != live_view.exchange_id
        serving = harness.exchange_log.records()[-1]
        assert serving["exchange_id"] == widget.exchange_id
        assert serving["route"] == "cached"
        assert serving["cached_from"] == live_view.exchange_id
    finally:
        server.should_exit = True
        thread.join(timeout=10)
