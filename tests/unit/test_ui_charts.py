"""Issue #18 RED — the inline chart-answer view (DESIGN §7.2 chart surface).

Pins ``ui.charts.chart_view_from_event`` over the service's ``chart``
SSE event: permalink + derived .csv/.svg download links, a copyable
embed snippet, and alt text passed through VERBATIM — a chart the UI
cannot describe raises, never renders mute (accessibility is not
optional chrome).
"""

from __future__ import annotations

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
