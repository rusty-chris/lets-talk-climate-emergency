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

import html
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
    alt_text = data.get("alt_text")
    if alt_text is None or not str(alt_text).strip():
        raise ChartAccessibilityError(
            "chart event carries no alt text — a chart the UI cannot describe "
            "is an upstream bug, never a silently inaccessible image"
        )

    spec_hash = data.get("spec_hash", "")
    # The ONE permalink; the .csv/.svg targets are suffixes of it, never
    # rebuilt from parts, so they cannot drift from the permalink the
    # service minted. ``base_url`` prefixes it when the shell knows the
    # public origin (relative otherwise).
    permalink = f"{base_url}{data.get('permalink') or f'/chart/{spec_hash}'}"
    csv_href = f"{permalink}.csv"
    svg_href = f"{permalink}.svg"
    # The embed snippet is REAL HTML the user pastes onto their own page
    # (finding #227): alt_text is planner-authored free text (quotes, ``&``
    # and ``<`` are all legal, maxLength 200) and svg_href can carry a
    # query-string ``&`` — both are HTML-attribute-escaped (quote=True) so a
    # quote can never break out of the attribute (broken markup / an
    # attribute-breakout footgun). ``ChartView.alt_text`` itself stays
    # verbatim: only the attribute encoding changes.
    embed_snippet = (
        f'<img src="{html.escape(svg_href, quote=True)}" '
        f'alt="{html.escape(str(alt_text), quote=True)}" />'
    )
    return ChartView(
        spec_hash=spec_hash,
        permalink=permalink,
        alt_text=alt_text,
        csv_href=csv_href,
        svg_href=svg_href,
        embed_snippet=embed_snippet,
    )
