"""The UI's SSE consumption seam (issue #18, DESIGN §7.2 over the #22 wire).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_ui_sse_client.py`` pins the contract.

The service speaks exactly one wire format — ``service.app.format_sse_event``
writes ``event: <name>\\ndata: <compact one-line JSON>\\n\\n`` frames — and
this module is its pure inverse plus the ONE seam through which the
Streamlit shell is allowed to reach the network:

- :func:`parse_sse_stream` is pure text -> event dicts. It consumes an
  iterable of wire LINES (newline-terminated or bare), buffers fields
  until the blank frame separator, and yields ``{"event": <name>,
  "data": <parsed JSON>}`` mappings — the same in-memory shape the
  service's generators produce, so the whole render model downstream is
  testable against synthetic event lists with no wire round-trip at all.
  SSE comment lines (leading ``:``, keep-alives) are ignored. A frame
  with a missing/blank event name, absent ``data:`` field, or
  non-JSON data raises :class:`SseProtocolError` naming the offending
  frame — the UI never guesses at a half-parsed event.
- :class:`ChatTransport` is the injectable transport protocol: called
  with ``(question, history)``, it yields the raw SSE lines of one
  ``POST /chat`` response. The real implementation (httpx streaming,
  built in the shell's composition root) is the ONLY network code in
  the UI; every test injects a synthetic transport. Framework decision
  (flagged for ratification in the red-phase notes): Streamlit consumes
  the stream via a plain synchronous httpx iterator fed through this
  parser into ``st.write_stream`` — no threads, no websockets.
- :func:`stream_chat_events` composes the two: it calls the transport
  with the question and history VERBATIM (no rewriting, no trimming —
  the service owns query processing) and yields parsed event dicts
  incrementally (streaming is the product surface; buffering the whole
  response before yielding would defeat it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol

__all__ = [
    "SseProtocolError",
    "TransportError",
    "ChatTransport",
    "parse_sse_stream",
    "stream_chat_events",
]


class SseProtocolError(Exception):
    """A wire frame violated the service's SSE format (never guessed at)."""


class TransportError(Exception):
    """The one transport-failure type the shell sees (review finding #224).

    RED-phase contract stub (review-18 fix wave): ``ui.transport`` must
    translate every httpx exception at the seam into THIS type — a
    connect failure, the service's own 429, a mid-stream disconnect, a
    read timeout — so neither the shell nor the pure core ever imports
    or names httpx specifics, and no raw traceback text ever reaches a
    rendered page. The message is a short human-honest description,
    never an exception repr or traceback.
    """


class ChatTransport(Protocol):
    """The injectable ``POST /chat`` seam: (question, history) -> raw SSE lines.

    The single place the UI touches the network. Implementations yield
    the response body's lines as they arrive; tests substitute a
    generator over synthetic wire text.
    """

    def __call__(
        self, question: str, history: Sequence[Mapping[str, Any]]
    ) -> Iterator[str]: ...  # pragma: no cover - protocol signature


def _frame(event_name: str | None, data_field: str | None) -> dict[str, Any]:
    """Validate one buffered frame's fields and parse it into an event dict."""
    if not event_name:
        raise SseProtocolError(f"SSE frame has no event name (data={data_field!r})")
    if data_field is None:
        raise SseProtocolError(f"SSE frame {event_name!r} has no data field")
    try:
        data = json.loads(data_field)
    except json.JSONDecodeError as exc:
        raise SseProtocolError(
            f"SSE frame {event_name!r} carries non-JSON data: {data_field!r}"
        ) from exc
    return {"event": event_name, "data": data}


def parse_sse_stream(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Pure inverse of ``service.app.format_sse_event`` (see module docs)."""
    event_name: str | None = None
    data_field: str | None = None
    have_field = False
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            # Blank line = frame separator. Emit the buffered frame; an
            # empty separator with nothing buffered (the writer's trailing
            # newline, keep-alive gaps) is simply skipped.
            if have_field:
                yield _frame(event_name, data_field)
            event_name = None
            data_field = None
            have_field = False
            continue
        if line.startswith(":"):
            # SSE comment / keep-alive line — ignored.
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value.strip()
            have_field = True
        elif field == "data":
            data_field = value
            have_field = True
        # Unknown fields (id/retry) are not part of this wire; ignore them.
    if have_field:
        # A final frame not terminated by a blank line (defensive; the
        # service always closes frames with a blank line).
        yield _frame(event_name, data_field)


def stream_chat_events(
    transport: ChatTransport,
    question: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> Iterator[dict[str, Any]]:
    """Call ``transport(question, history)`` and yield parsed events incrementally."""
    # Question and history pass VERBATIM (the service owns query processing);
    # yielding from the parser over the live line iterator keeps consumption
    # incremental — streaming is the product surface.
    yield from parse_sse_stream(transport(question, history))
