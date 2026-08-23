"""Issue #18 RED — the inline chart-answer view (DESIGN §7.2 chart surface).

Pins ``ui.charts.chart_view_from_event`` over the service's ``chart``
SSE event: permalink + derived .csv/.svg download links, a copyable
embed snippet, and alt text passed through VERBATIM — a chart the UI
cannot describe raises, never renders mute (accessibility is not
optional chrome).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from tests._ui_fixtures import chart_event
from ui.charts import ChartAccessibilityError, chart_view_from_event

ALT = "Line chart: CO2 and temperature over the last 10,000 years."


class TestChartView:
    def test_permalink_and_download_links_derive_from_the_same_permalink(self) -> None:
        view = chart_view_from_event(chart_event("cafe0123beef", ALT)["data"])
        assert view.spec_hash == "cafe0123beef"
        assert view.permalink == "/chart/cafe0123beef"
        assert view.csv_href == "/chart/cafe0123beef.csv"
        assert view.svg_href == "/chart/cafe0123beef.svg"

    def test_base_url_prefixes_every_link_absolutely(self) -> None:
        view = chart_view_from_event(
            chart_event("cafe0123beef", ALT)["data"],
            base_url="https://letstalkclimateemergency.example",
        )
        assert view.permalink == "https://letstalkclimateemergency.example/chart/cafe0123beef"
        assert view.csv_href == "https://letstalkclimateemergency.example/chart/cafe0123beef.csv"
        assert view.svg_href == "https://letstalkclimateemergency.example/chart/cafe0123beef.svg"

    def test_alt_text_passes_through_verbatim(self) -> None:
        view = chart_view_from_event(chart_event("cafe0123beef", ALT)["data"])
        assert view.alt_text == ALT

    def test_embed_snippet_carries_the_svg_permalink_and_the_alt_text(self) -> None:
        """An embed without alt text strips accessibility on someone
        else's page; the snippet must carry both."""
        view = chart_view_from_event(
            chart_event("cafe0123beef", ALT)["data"],
            base_url="https://letstalkclimateemergency.example",
        )
        assert (
            "https://letstalkclimateemergency.example/chart/cafe0123beef.svg" in view.embed_snippet
        )
        assert ALT in view.embed_snippet


class TestEmbedSnippetEscaping:
    """Review finding #227 RED — the embed snippet is real HTML, escape it.

    ``alt_text`` derives from the PLANNER-AUTHORED ChartSpec title/series
    labels (LLM free strings, maxLength 200 — quotes, ``&`` and ``<`` are
    all legal) and is interpolated raw into an HTML attribute users paste
    into their own pages: a quote terminates the attribute (broken HTML,
    truncated accessibility text) and is an attribute-breakout injection
    footgun on the project's flagship share surface.
    """

    HOSTILE_ALT = 'Warming: "the blade" & <inset>'

    def test_embed_snippet_escapes_html_metacharacters_in_alt_text(self) -> None:
        view = chart_view_from_event(chart_event("cafe0123beef", self.HOSTILE_ALT)["data"])
        snippet = view.embed_snippet

        assert "&quot;" in snippet
        assert "&amp;" in snippet
        assert "&lt;" in snippet

        # No raw double quote may survive INSIDE the alt attribute value:
        # the first one would terminate the attribute on the user's page.
        alt_attribute = re.search(r'alt="([^"]*)"', snippet)
        assert alt_attribute is not None, "the snippet must carry a well-formed alt attribute"

        # Round trip: parsing the snippet recovers the original alt text
        # exactly — escaping must be lossless, never truncating.
        recovered: list[str | None] = []

        class AltRecovery(HTMLParser):
            def handle_startendtag(self, tag: str, attrs: list) -> None:
                self.handle_starttag(tag, attrs)

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag == "img":
                    recovered.append(dict(attrs).get("alt"))

        AltRecovery().feed(snippet)
        assert recovered == [self.HOSTILE_ALT]

    def test_embed_snippet_escapes_the_href_too(self) -> None:
        """A base_url with a query string (& is legal there) must render
        &amp; inside the src attribute."""
        view = chart_view_from_event(
            chart_event("cafe0123beef", ALT)["data"],
            base_url="https://site.example/embed?theme=dark&lang=en",
        )
        assert "&amp;" in view.embed_snippet
        assert "theme=dark&lang=en" not in view.embed_snippet

    def test_alt_text_field_remains_verbatim_while_snippet_is_escaped(self) -> None:
        """Regression pin (already true today, kept load-bearing through
        the escaping change): ``view.alt_text`` is the caption surface and
        stays VERBATIM — only the HTML attribute encoding changes."""
        view = chart_view_from_event(chart_event("cafe0123beef", self.HOSTILE_ALT)["data"])
        assert view.alt_text == self.HOSTILE_ALT


class TestChartAccessibility:
    def test_missing_alt_text_raises_never_renders_a_mute_chart(self) -> None:
        data = dict(chart_event("cafe0123beef", ALT)["data"])
        del data["alt_text"]
        with pytest.raises(ChartAccessibilityError):
            chart_view_from_event(data)

    def test_blank_alt_text_raises_too(self) -> None:
        data = dict(chart_event("cafe0123beef", "   ")["data"])
        with pytest.raises(ChartAccessibilityError):
            chart_view_from_event(data)
