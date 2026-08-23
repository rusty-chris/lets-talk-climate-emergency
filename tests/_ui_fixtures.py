"""Synthetic SSE event/wire builders for the issue #18 UI suites.

The UI is tested with NO live network anywhere: these helpers fabricate
the service's parsed event dicts (the exact #22 vocabulary —
``service.app``'s meta/answer/chart events over the #12
text/citation/usage/footer/error and #13 badge/validation_degraded
extensions) and, where a test needs wire text, render them through the
service's own ``format_sse_event`` so the parser is pinned against the
real writer, not a hand-rolled imitation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from service.app import format_sse_event
from service.exchange_log import LOGGING_DISCLOSURE

__all__ = [
    "DISCLOSURE",
    "meta_event",
    "text_event",
    "citation_event",
    "usage_event",
    "footer_event",
    "error_event",
    "badge_event",
    "validation_degraded_event",
    "answer_event",
    "chart_event",
    "wire_lines",
]

DISCLOSURE = LOGGING_DISCLOSURE


def meta_event(
    mode: str = "live", preamble_note: str | None = None, disclosure: str = DISCLOSURE
) -> dict[str, Any]:
    return {
        "event": "meta",
        "data": {"disclosure": disclosure, "preamble_note": preamble_note, "mode": mode},
    }


def text_event(text: str) -> dict[str, Any]:
    return {"event": "text", "data": {"text": text}}


def citation_event(
    document_index: int,
    chunk_id: str,
    cited_text: str,
    document_title: str | None = None,
    *,
    clears_threshold: bool = True,
    degraded_fallback: bool = False,
    needs_hand_review: bool = False,
) -> dict[str, Any]:
    """One resolved streaming citation, shaped like ``answer_stream_to_sse``'s."""
    return {
        "event": "citation",
        "data": {
            "type": "char_location",
            "cited_text": cited_text,
            "document_index": document_index,
            "document_title": document_title,
            "chunk_id": chunk_id,
            "clears_threshold": clears_threshold,
            "degraded_fallback": degraded_fallback,
            "needs_hand_review": needs_hand_review,
        },
    }


def usage_event(input_tokens: int = 900, output_tokens: int = 120) -> dict[str, Any]:
    return {
        "event": "usage",
        "data": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def footer_event(
    text: str = "Every citation links to source text. Sources as of 2026-08.",
) -> dict[str, Any]:
    return {"event": "footer", "data": {"text": text}}


def error_event(
    error_type: str = "incomplete_stream",
    message: str = "The answer stream ended before completion; this response is truncated.",
) -> dict[str, Any]:
    return {"event": "error", "data": {"type": error_type, "message": message}}


def badge_event(
    sentence_index: int, document_index: int | None, reason: str = "entailment_failed"
) -> dict[str, Any]:
    return {
        "event": "badge",
        "data": {
            "sentence_index": sentence_index,
            "document_index": document_index,
            "reason": reason,
        },
    }


def validation_degraded_event(reason: str = "validator transport error") -> dict[str, Any]:
    return {"event": "validation_degraded", "data": {"reason": reason}}


def answer_event(kind: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"event": "answer", "data": {"kind": kind, "text": text, **extra}}


def chart_event(
    spec_hash: str = "cafe0123beef",
    alt_text: str = "Line chart: CO2 and temperature over the last 10,000 years.",
) -> dict[str, Any]:
    return {
        "event": "chart",
        "data": {
            "spec_hash": spec_hash,
            "permalink": f"/chart/{spec_hash}",
            "alt_text": alt_text,
        },
    }


def wire_lines(events: Sequence[Mapping[str, Any]]) -> Iterator[str]:
    """The events as SSE wire LINES, written by the service's own formatter."""
    for event in events:
        yield from format_sse_event(event).split("\n")
