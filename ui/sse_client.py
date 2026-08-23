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

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol

__all__ = [
    "SseProtocolError",
    "ChatTransport",
    "parse_sse_stream",
    "stream_chat_events",
]


class SseProtocolError(Exception):
    """A wire frame violated the service's SSE format (never guessed at)."""


class ChatTransport(Protocol):
    """The injectable ``POST /chat`` seam: (question, history) -> raw SSE lines.

    The single place the UI touches the network. Implementations yield
    the response body's lines as they arrive; tests substitute a
    generator over synthetic wire text.
    """

    def __call__(
        self, question: str, history: Sequence[Mapping[str, Any]]
    ) -> Iterator[str]: ...  # pragma: no cover - protocol signature


def parse_sse_stream(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Pure inverse of ``service.app.format_sse_event`` (see module docs)."""
    raise NotImplementedError("issue #18 red phase: parse_sse_stream is not implemented yet")


def stream_chat_events(
    transport: ChatTransport,
    question: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> Iterator[dict[str, Any]]:
    """Call ``transport(question, history)`` and yield parsed events incrementally."""
    raise NotImplementedError("issue #18 red phase: stream_chat_events is not implemented yet")
