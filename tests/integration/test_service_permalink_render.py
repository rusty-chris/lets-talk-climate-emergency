"""Chart permalinks through the real service + real renderer (issue #22).

The issue-named integration test: ``/chart/<spec-hash>`` re-renders
#17's real artefact path (validate_spec_for_render → build_vega_lite →
vl-convert SVG) from the STORED spec — no fetch, pinned synthetic
fixture data — through the composed FastAPI app.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import pytest
from fastapi.testclient import TestClient

from charts.render import render_chart
from charts.spec import spec_hash
from service.app import create_app
from tests._chart_render_fixtures import SITE_URL, frames, line_spec, render_manifest
from tests._service_fixtures import make_harness

pytestmark = pytest.mark.integration


@pytest.fixture()
def real_render_harness(tmp_path):
    """The unit harness with the REAL #17 renderer bound as the dep."""
    harness = make_harness(tmp_path)
    manifest = render_manifest()
    bound = partial(
        render_chart,
        frames=frames(),
        manifest=manifest,
        site_url=SITE_URL,
    )
    deps = replace(harness.deps, render_chart=lambda spec: bound(spec))
    app = create_app(harness.config, deps)
    harness.app = app
    return harness


def test_chart_permalink_route_serves_stored_spec(real_render_harness, tmp_path) -> None:
    """The GATE-adjacent issue test: a stored spec's permalink serves the
    fully-rendered artefact — vega-lite JSON, alt text, attribution CSV."""
    harness = real_render_harness
    spec = line_spec()
    stored_hash = harness.spec_store.put(spec)
    assert stored_hash == spec_hash(spec)
    client = TestClient(harness.app)

    response = client.get(f"/chart/{stored_hash}")
    assert response.status_code == 200
    body = response.json()
    assert body["spec_hash"] == stored_hash
    assert body["alt_text"]
    # The rendered vega-lite carries the caption strip (attribution is
    # part of the artefact — DESIGN §3.7).
    assert "Meridian Climate Service (fictional)" in str(body["vega_lite"])

    csv_response = client.get(f"/chart/{stored_hash}.csv")
    assert csv_response.status_code == 200
    assert csv_response.text.startswith("#")
    assert "Meridian Climate Service (fictional)" in csv_response.text


def test_chart_permalink_svg_export(real_render_harness) -> None:
    """The vl-convert edge through the route: real SVG bytes."""
    harness = real_render_harness
    stored_hash = harness.spec_store.put(line_spec())
    response = TestClient(harness.app).get(f"/chart/{stored_hash}.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text


def test_unknown_permalink_404s_through_the_real_service(real_render_harness) -> None:
    client = TestClient(real_render_harness.app)
    assert client.get(f"/chart/{'f' * 64}").status_code == 404
