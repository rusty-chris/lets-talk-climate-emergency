"""Issue #19 RED — the service serves the built transparency pages.

App-level pins (TestClient over the composed app, all adapters faked):

- when ``ServiceDeps.transparency`` (a ``TransparencyPages``) is
  provided, ``/about /privacy /sources /voices`` serve EXACTLY its
  html, with ``text/html``, in BOTH modes — the paused state keeps the
  credibility furniture up, like /health (DESIGN §9 read-only-not-dark);
- serving the pages is free and anonymous: zero adapter calls, nothing
  appended to the exchange log, and never rate-limited;
- route parity with the UI footer: ``ui.footer`` links these routes and
  duplicates the ADR-018/§4.11 strings — the constants are pinned EQUAL
  (the wire-vocabulary parity pattern of test_ui_shell_hygiene.py;
  ``ui.footer`` is pure/streamlit-free, enforced there, so importing it
  here drags no UI framework into the service suite).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from service.budget import ServiceMode
from service.transparency import TransparencyPages
from tests._service_fixtures import make_harness

SENTINEL_PAGES = TransparencyPages(
    about_html="<html><body>SENTINEL-ABOUT (synthetic)</body></html>",
    privacy_html="<html><body>SENTINEL-PRIVACY (synthetic)</body></html>",
    sources_html="<html><body>SENTINEL-SOURCES (synthetic)</body></html>",
    voices_html="<html><body>SENTINEL-VOICES (synthetic)</body></html>",
)


def transparency_client(tmp_path, *, paused: bool = False) -> TestClient:
    harness = make_harness(tmp_path, transparency=SENTINEL_PAGES)
    if paused:
        harness.tracker.record_usage(
            "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
        )
        assert harness.tracker.mode() is ServiceMode.PAUSED
    return TestClient(harness.app)


class TestRoutesServeTheBuiltPages:
    @pytest.mark.parametrize("route", sorted(SENTINEL_PAGES.as_route_map()))
    def test_route_serves_the_provided_page_live(self, tmp_path, route: str) -> None:
        response = transparency_client(tmp_path).get(route)
        assert response.status_code == 200
        assert response.text == SENTINEL_PAGES.as_route_map()[route], (
            f"{route} did not serve the injected TransparencyPages html"
        )

    @pytest.mark.parametrize("route", sorted(SENTINEL_PAGES.as_route_map()))
    def test_route_serves_the_provided_page_paused(self, tmp_path, route: str) -> None:
        """Paused is read-only, not dark: the transparency furniture
        stays up when the LLM budget is gone."""
        response = transparency_client(tmp_path, paused=True).get(route)
        assert response.status_code == 200
        assert response.text == SENTINEL_PAGES.as_route_map()[route]

    @pytest.mark.parametrize("route", sorted(SENTINEL_PAGES.as_route_map()))
    def test_route_content_type_is_html(self, tmp_path, route: str) -> None:
        response = transparency_client(tmp_path).get(route)
        assert response.headers["content-type"].startswith("text/html")


class TestServingIsFreeAndAnonymous:
    def test_page_views_make_no_adapter_calls_and_log_nothing(self, tmp_path) -> None:
        """A page view is not an exchange: no LLM calls, no exchange-log
        record, no content telemetry of any kind."""
        harness = make_harness(tmp_path, transparency=SENTINEL_PAGES)
        client = TestClient(harness.app)
        for route in SENTINEL_PAGES.as_route_map():
            assert client.get(route).status_code == 200
        assert harness.adapter.calls == []
        assert harness.exchange_log.records() == []

    def test_pages_are_never_rate_limited(self, tmp_path) -> None:
        """Like /health: the read-only furniture serves every visitor —
        the per-IP limit protects the LLM spend, not static html. A
        1-request/minute limit would 429 the second GET if the static
        routes consulted the limiter."""
        from tests._service_fixtures import make_config

        tight = make_harness(
            tmp_path,
            transparency=SENTINEL_PAGES,
            config=make_config(rate_limit_per_minute=1),
        )
        client = TestClient(tight.app)
        for _ in range(3):
            for route in SENTINEL_PAGES.as_route_map():
                assert client.get(route).status_code == 200


class TestFooterRouteParity:
    """The #18 footer links and the #19 service routes cannot drift."""

    def test_footer_routes_equal_service_transparency_routes(self) -> None:
        import ui.footer
        from service.transparency import TRANSPARENCY_ROUTES

        assert TRANSPARENCY_ROUTES == ui.footer.TRANSPARENCY_ROUTES

    def test_every_footer_route_is_served_by_the_app(self, tmp_path) -> None:
        import ui.footer

        harness = make_harness(tmp_path, transparency=SENTINEL_PAGES)
        served = {route.path for route in harness.app.routes}
        missing = set(ui.footer.TRANSPARENCY_ROUTES) - served
        assert not missing, f"footer links routes the service does not serve: {missing}"

    def test_adr018_strings_match_the_ui_footer(self) -> None:
        """The service pages and the UI footer render the same ADR-018
        pair and §4.11 disclaimer — duplicated constants, parity-pinned
        (the service image never imports the UI package)."""
        import service.transparency as transparency
        import ui.footer

        assert transparency.STEWARD_CREDIT_TEXT == ui.footer.STEWARD_CREDIT_TEXT
        assert transparency.NONCOMMERCIAL_NOTE == ui.footer.NONCOMMERCIAL_NOTE
        assert transparency.NON_AFFILIATION_DISCLAIMER == ui.footer.NON_AFFILIATION_DISCLAIMER
