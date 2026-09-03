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

from ui.sse_client import ChatTransport, FeedbackTransport, TransportError

__all__ = ["http_chat_transport", "http_feedback_transport"]

#: A human-honest transport-failure message for the shell to fold into an
#: honest view — never an exception repr, never httpx internals, never a
#: traceback (finding #224). Rate limiting (the service's own 429) gets its
#: own calmer line; every other transport failure shares the generic one.
_RATE_LIMITED_MESSAGE = "Too many questions right now — please try again in a minute."
_INTERRUPTED_MESSAGE = "The connection to the answer service was interrupted. Please try again."

#: Long enough for a full grounded answer to stream token by token; the
#: connect phase is short, the read phase deliberately patient. The read
#: value is per-gap (between received chunks), not total, and must survive
#: a cold-start deploy's FIRST grounded query: the api builds its retrieval
#: models lazily (bge-m3 + reranker, service.main._LazyRetrieval), so the
#: gap between the meta event and the first text token can run to minutes
#: once per process — cutting it short would hand the first visitor after
#: every deploy a transport error instead of an answer.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


def http_chat_transport(
    base_url: str, *, timeout: httpx.Timeout | float | None = None
) -> ChatTransport:
    """Bind ``base_url`` into a streaming ``POST /chat`` transport callable."""
    request_timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

    def transport(question: str, history: Sequence[Mapping[str, Any]]) -> Iterator[str]:
        # Every httpx failure mode is translated to TransportError HERE, at
        # the seam (finding #224): connect refused, the service's own 429, a
        # mid-stream disconnect (RemoteProtocolError from iter_lines), a read
        # timeout. Neither the shell nor the pure core ever imports or names
        # httpx specifics, and no raw traceback text ever reaches a rendered
        # page. The message is short human-honest prose, never an exception
        # repr.
        try:
            with httpx.stream(
                "POST",
                f"{base_url.rstrip('/')}/chat",
                json={"question": question, "history": [dict(turn) for turn in history]},
                timeout=request_timeout,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    message = (
                        _RATE_LIMITED_MESSAGE
                        if exc.response.status_code == 429
                        else _INTERRUPTED_MESSAGE
                    )
                    raise TransportError(message) from exc
                # iter_lines can raise mid-stream (an api restart) — that
                # exception surfaces here as the generator is consumed.
                yield from response.iter_lines()
        except httpx.HTTPError as exc:
            raise TransportError(_INTERRUPTED_MESSAGE) from exc

    return transport


#: The feedback POST is a single tiny JSON body: a short, uniform timeout
#: (no streaming read phase — a slow feedback endpoint must not hang the
#: page the way a slow ANSWER is allowed to).
_FEEDBACK_TIMEOUT = httpx.Timeout(10.0)


def http_feedback_transport(
    base_url: str, *, timeout: httpx.Timeout | float | None = None
) -> FeedbackTransport:
    """Bind ``base_url`` into a ``POST /feedback`` transport callable.

    RED-phase contract stub (issue #56); the failing suites in
    ``tests/unit/test_ui_feedback.py`` (structural) and
    ``tests/integration/test_feedback_roundtrip.py`` (real socket) pin
    the contract — the :class:`ui.sse_client.FeedbackTransport` seam
    made real: posts ``{"exchange_id": ..., "verdict": ...}`` to
    ``<base_url>/feedback``; returns True IFF the service answered 204;
    returns False on EVERY other outcome (the uniform 404, a 429, any
    ``httpx`` failure) — never raises to the shell, never fakes a
    recording.
    """
    del timeout  # bound in the implementation; the stub only pins the shape

    def transport(exchange_id: str, verdict: str) -> bool:
        raise NotImplementedError(
            "#56 red phase: the feedback transport is pinned in "
            "tests/integration/test_feedback_roundtrip.py"
        )

    return transport
