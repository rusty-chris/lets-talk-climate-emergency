"""Issue #18 RED — the UI's SSE parsing + injectable chat transport seam.

Pins ``ui.sse_client`` against the service's OWN wire writer
(``service.app.format_sse_event``): parsing is its exact inverse, the
transport is the single injectable network seam (tests never open a
socket), and consumption is incremental — streaming is the product
surface.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from tests._ui_fixtures import (
    citation_event,
    footer_event,
    meta_event,
    text_event,
    usage_event,
    wire_lines,
)
from ui.sse_client import SseProtocolError, parse_sse_stream, stream_chat_events


class TestParseSseStream:
    def test_round_trips_the_service_wire_format(self) -> None:
        """The parser is the exact inverse of service.app.format_sse_event."""
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C. "),
            citation_event(0, "chunk-a", "observed warming of 1.2C", "Meridian Assessment"),
            usage_event(),
            footer_event(),
        ]
        parsed = list(parse_sse_stream(wire_lines(events)))
        assert parsed == events

    def test_preserves_non_ascii_text_verbatim(self) -> None:
        """The wire is ensure_ascii=False; degree signs and CO₂ survive."""
        events = [meta_event(), text_event("What happens at 1.5°C? CO₂ keeps rising…")]
        parsed = list(parse_sse_stream(wire_lines(events)))
        assert parsed[1]["data"]["text"] == "What happens at 1.5°C? CO₂ keeps rising…"

    def test_ignores_sse_comment_keepalive_lines(self) -> None:
        lines = [": keep-alive", *wire_lines([meta_event()]), ": keep-alive", ""]
        parsed = list(parse_sse_stream(lines))
        assert parsed == [meta_event()]

    def test_frame_without_event_name_raises(self) -> None:
        lines = ['data: {"kind":"canned","text":"hi"}', "", ""]
        with pytest.raises(SseProtocolError):
            list(parse_sse_stream(lines))

    def test_frame_without_data_field_raises(self) -> None:
        lines = ["event: text", "", ""]
        with pytest.raises(SseProtocolError):
            list(parse_sse_stream(lines))

    def test_frame_with_non_json_data_raises(self) -> None:
        lines = ["event: text", "data: not-json{", "", ""]
        with pytest.raises(SseProtocolError):
            list(parse_sse_stream(lines))


class RecordingTransport:
    """A synthetic ChatTransport: records its call, yields canned wire lines."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = list(lines)
        self.calls: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
        self.yielded = 0

    def __call__(self, question: str, history: Sequence[Mapping[str, Any]]) -> Iterator[str]:
        self.calls.append((question, tuple(history)))
        for line in self.lines:
            self.yielded += 1
            yield line


class TestStreamChatEvents:
    def test_uses_only_the_injected_transport(self) -> None:
        """Events come from the transport; question and history pass verbatim."""
        events = [meta_event(), text_event("Yes."), footer_event()]
        transport = RecordingTransport(list(wire_lines(events)))
        history = [{"role": "user", "content": "Is it warming?"}]

        received = list(stream_chat_events(transport, "How fast?", history))

        assert received == events
        assert transport.calls == [("How fast?", (history[0],))]

    def test_yields_events_incrementally_not_after_buffering_everything(self) -> None:
        """Streaming is the product surface: the first event must arrive
        before the transport has been drained to its final line."""
        events = [meta_event(), text_event("First tokens"), text_event("more"), footer_event()]
        lines = list(wire_lines(events))
        transport = RecordingTransport(lines)

        stream = stream_chat_events(transport, "How fast?", [])
        first = next(iter(stream))

        assert first == meta_event()
        assert transport.yielded < len(lines), (
            "the whole response was buffered before the first event was yielded"
        )
