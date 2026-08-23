"""The real ``ChatTransport``: the UI's ONLY network code (issue #18).

This is shell, not pure core — it is the httpx implementation of the
:class:`ui.sse_client.ChatTransport` seam, kept out of the pure modules
(the shell-hygiene suite forbids ``httpx`` in ``ui/presenters`` and its
siblings) so the whole answer-view fold stays testable with synthetic
transports and no socket is ever opened in a unit test.

:func:`http_chat_transport` binds a base URL (and an optional timeout)
into a transport callable. Called with ``(question, history)`` it opens
one streaming ``POST /chat`` and yields the response body's lines as they
arrive — exactly the incremental wire ``ui.sse_client.parse_sse_stream``
consumes. The read timeout is generous: a grounded answer streams token
by token over many seconds, and cutting it short would truncate the
product surface.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from ui.sse_client import ChatTransport

__all__ = ["http_chat_transport"]

#: Long enough for a full grounded answer to stream token by token; the
#: connect phase is short, the read phase deliberately patient.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def http_chat_transport(
    base_url: str, *, timeout: httpx.Timeout | float | None = None
) -> ChatTransport:
    """Bind ``base_url`` into a streaming ``POST /chat`` transport callable."""
    request_timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

    def transport(question: str, history: Sequence[Mapping[str, Any]]) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat",
            json={"question": question, "history": [dict(turn) for turn in history]},
            timeout=request_timeout,
        ) as response:
            response.raise_for_status()
            yield from response.iter_lines()

    return transport
