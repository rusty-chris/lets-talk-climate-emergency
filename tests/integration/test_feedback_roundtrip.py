"""The thumbs widget posts a real feedback event (issue #56, integration).

The issue-named ``test_feedback_widget_posts_event`` tier: uvicorn
serves the composed #22 fake-seam app on a loopback socket and the UI's
REAL transports drive the whole loop — stream one exchange through
``http_chat_transport``, fold it, build the widget from the pure model,
post its verdict through ``http_feedback_transport`` — and the verdict
must land on the exchange's log record. No Docker, no LLM key, no
network beyond the loopback socket.

Also pinned here, where the real socket exists: the feedback transport's
never-raise contract — a 404 (unknown exchange) and a dead service both
come back as an honest False (the shell renders the unrecorded state),
never an exception on the public page.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import pytest
import uvicorn

from service.exchange_log import FEEDBACK_DOWN
from tests._service_fixtures import classifier_output, make_harness
from ui.render_model import feedback_widget_model, fold_chat_stream
from ui.sse_client import stream_chat_events
from ui.transport import http_chat_transport, http_feedback_transport

pytestmark = pytest.mark.integration

SERVER_STARTUP_TIMEOUT_S = 15


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _RunningService:
    def __init__(self, harness, port: int) -> None:
        self.harness = harness
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"


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
    return _RunningService(harness, port), server, thread


def test_feedback_widget_posts_event(tmp_path) -> None:
    """UI transport in, UI transport out: the widget built from a REAL
    streamed exchange posts its verdict and the verdict lands on the
    exchange's log record — the full #56 loop over a real socket."""
    harness = make_harness(tmp_path)
    harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
    running, server, thread = _serve(harness)
    try:
        chat = http_chat_transport(running.base_url)
        events = list(stream_chat_events(chat, "a question about the invented basin"))
        view = fold_chat_stream(events)
        assert view.complete is True

        widget = feedback_widget_model(view)
        assert widget is not None, "a completed answer over the real wire must be rateable"

        feedback = http_feedback_transport(running.base_url)
        assert feedback(widget.exchange_id, widget.down_verdict) is True

        records = harness.exchange_log.records()
        assert len(records) == 1
        assert records[0]["exchange_id"] == widget.exchange_id
        assert records[0]["feedback"] == {"verdict": FEEDBACK_DOWN}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_unknown_exchange_comes_back_as_false_not_an_exception(tmp_path) -> None:
    harness = make_harness(tmp_path)
    running, server, thread = _serve(harness)
    try:
        feedback = http_feedback_transport(running.base_url)
        assert feedback(uuid.uuid4().hex, FEEDBACK_DOWN) is False, (
            "the uniform 404 is an honest unrecorded outcome, never a crash"
        )
        assert harness.exchange_log.records() == []
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_dead_service_comes_back_as_false_not_an_exception() -> None:
    """A feedback click against a down/unreachable api must degrade to
    the honest unrecorded state — no traceback ever reaches the page."""
    dead_port = _free_port()  # bound momentarily, then released: nothing listens
    feedback = http_feedback_transport(f"http://127.0.0.1:{dead_port}")
    assert feedback(uuid.uuid4().hex, FEEDBACK_DOWN) is False
