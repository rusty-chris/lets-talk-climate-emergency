"""Route-level client-disconnect metering (issue #211, integration tier).

The unit tier pins the generator-finalization contract by closing the
``_chat_events`` generator directly (starlette's TestClient consumes a
streaming body eagerly, so a route-level abort cannot be simulated
in-process). This test drives the REAL socket path: uvicorn serves the
composed app in-process, a genuine HTTP client reads the first SSE bytes
of ``POST /chat`` and drops the connection, and the charged generation
usage must still reach the SpendTracker and the exchange log — the
fail-closed daily cap (DESIGN §10 GATE) must not be bypassable by the
cheapest client behaviour there is.

No Docker, no network beyond a loopback socket, no API key: every seam
is the #22 fake harness; only the HTTP server/client pair is real.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from rag.generation import GENERATION_MODEL_DEFAULT
from tests._generation_fixtures import transport_stream_events
from tests._service_fixtures import (
    classifier_output,
    make_harness,
    stream_usage,
    usage_cost,
)

pytestmark = pytest.mark.integration

SERVER_STARTUP_TIMEOUT_S = 15
FINALIZATION_TIMEOUT_S = 15

#: Gap between transport events. A list-backed fake stream is written to
#: the socket buffer faster than any client can abort, so the disconnect
#: would land only after the generator completed and the test would
#: prove nothing; pacing the transport keeps the stream in flight long
#: enough for the abort to interrupt the generator mid-yield — the exact
#: production shape (the provider streams slower than a close).
TRANSPORT_EVENT_GAP_S = 0.1


class _PacedEvents(list):
    """A transport-event sequence whose iteration is paced (see above).

    A ``list`` subclass so the FakeAdapter's response-shape validation
    (a SEQUENCE of transport event mappings, finding #183) still holds.
    """

    def __iter__(self):
        for event in super().__iter__():
            time.sleep(TRANSPORT_EVENT_GAP_S)
            yield event


def _paced_transport_events() -> _PacedEvents:
    return _PacedEvents(transport_stream_events())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_route_level_disconnect_still_records_spend_and_logs(tmp_path) -> None:
    harness = make_harness(tmp_path)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", _paced_transport_events())

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
    try:
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn never started"
            time.sleep(0.05)

        # Read the stream only until the first text event has been
        # delivered, then abandon the connection mid-SSE (closing the
        # stream context aborts the response and drops the socket).
        received = b""
        with httpx.Client(timeout=10) as client:
            with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/chat",
                json={"question": "Why is the basin warming?"},
            ) as response:
                assert response.status_code == 200
                for chunk in response.iter_raw():
                    received += chunk
                    if b"event: text" in received:
                        break

        # The disconnected exchange's charged usage must still be
        # metered and the exchange logged (generator finalization); poll
        # because the server notices the disconnect asynchronously.
        expected = usage_cost(GENERATION_MODEL_DEFAULT, stream_usage())
        deadline = time.monotonic() + FINALIZATION_TIMEOUT_S
        while time.monotonic() < deadline:
            if harness.tracker.spent_today() > 0 and harness.exchange_log.records():
                break
            time.sleep(0.05)
        assert harness.tracker.spent_today() == pytest.approx(expected), (
            "the disconnected request's generation usage never reached the "
            "spend tracker — unmetered LLM spend (issue #211)"
        )
        records = harness.exchange_log.records()
        assert len(records) == 1
        assert records[0]["route"] == "retrieval"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
