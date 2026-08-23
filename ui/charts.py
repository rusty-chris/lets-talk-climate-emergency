"""The inline chart-answer view (issue #18, DESIGN §7.2 / §3.7 surface).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_ui_charts.py`` pins the contract.

A live chart answer arrives as ONE service ``chart`` SSE event —
``{"spec_hash", "permalink", "alt_text"}`` — and the permalink routes
are the #22 contract: ``GET /chart/<hash>`` (Vega-Lite JSON + alt
text), ``GET /chart/<hash>.csv`` (attribution-headed data), ``GET
/chart/<hash>.svg``. :func:`chart_view_from_event` maps that event to
everything the shell draws:

- ``permalink`` — the copyable permanent link (``base_url`` prefixed
  when the shell knows the public origin; relative otherwise);
- ``csv_href`` / ``svg_href`` — the "View data & sources" and
  "Download SVG" targets, derived by suffixing the SAME permalink
  (never rebuilt from parts, so they cannot drift from it);
- ``embed_snippet`` — copy-paste HTML that embeds the SVG permalink
  WITH the alt text (an embed without alt text would strip
  accessibility on someone else's page);
- ``alt_text`` — passed through VERBATIM from the service (the
  renderer's §3.7 alt text is the accessibility surface; the UI never
  drops, truncates or rewrites it). A missing or blank ``alt_text``
  raises :class:`ChartAccessibilityError` — a chart the UI cannot
  describe is a bug upstream, never a silently inaccessible image.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ChartAccessibilityError",
    "ChartView",
    "chart_view_from_event",
]


class ChartAccessibilityError(Exception):
    """A chart event arrived without usable alt text (never rendered mute)."""


@dataclass(frozen=True)
class ChartView:
    """Everything the shell renders for one inline chart answer."""

    spec_hash: str
    permalink: str
    alt_text: str
    csv_href: str
    svg_href: str
    embed_snippet: str


def chart_view_from_event(data: Mapping[str, Any], *, base_url: str = "") -> ChartView:
    """Map one service ``chart`` event's data to a :class:`ChartView`."""
    raise NotImplementedError("issue #18 red phase: chart_view_from_event is not implemented yet")
