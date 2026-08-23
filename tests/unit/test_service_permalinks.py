"""Chart permalink routes, unit tier (issue #22, DESIGN §3.7).

Pins ``service.chart_store.ChartSpecStore`` and the ``/chart/<hash>``
routes over a fake renderer: re-render from the STORED spec, 404 on
unknown/malformed hashes, no path traversal, no fetch. The
through-the-real-renderer variant is
``tests/integration/test_service_permalink_render.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from charts.spec import spec_hash
from service.chart_store import ChartSpecStore
from tests._service_fixtures import SYNTHETIC_SPEC, make_harness


class TestChartSpecStore:
    def test_put_get_roundtrip_keyed_by_canonical_hash(self, tmp_path) -> None:
        store = ChartSpecStore(tmp_path / "specs")
        stored_hash = store.put(SYNTHETIC_SPEC)
        assert stored_hash == spec_hash(SYNTHETIC_SPEC)
        assert store.get(stored_hash) == SYNTHETIC_SPEC

    def test_put_is_idempotent(self, tmp_path) -> None:
        store = ChartSpecStore(tmp_path / "specs")
        assert store.put(SYNTHETIC_SPEC) == store.put(SYNTHETIC_SPEC)

    def test_get_unknown_hash_returns_none(self, tmp_path) -> None:
        store = ChartSpecStore(tmp_path / "specs")
        assert store.get("0" * 64) is None

    def test_get_refuses_non_hash_input_without_touching_paths(self, tmp_path) -> None:
        """The hash arrives from a URL: traversal shapes never reach the
        filesystem."""
        store = ChartSpecStore(tmp_path / "specs")
        store.put(SYNTHETIC_SPEC)
        for hostile in ("../../etc/passwd", "..%2f..%2fetc", "specs", "ABC" * 22, ""):
            assert store.get(hostile) is None


class TestPermalinkRoutes:
    def test_stored_spec_re_renders_on_request(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        stored_hash = harness.spec_store.put(SYNTHETIC_SPEC)
        client = TestClient(harness.app)

        response = client.get(f"/chart/{stored_hash}")
        assert response.status_code == 200
        body = response.json()
        assert body["spec_hash"] == stored_hash
        assert body["alt_text"]
        assert body["vega_lite"]
        # The response was RE-RENDERED from the stored spec.
        assert harness.renderer.calls == [SYNTHETIC_SPEC]

    def test_csv_download_carries_attribution_header(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        stored_hash = harness.spec_store.put(SYNTHETIC_SPEC)
        response = TestClient(harness.app).get(f"/chart/{stored_hash}.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert response.text.startswith("#")  # attribution comment lines first

    def test_unknown_hash_is_404(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        assert TestClient(harness.app).get(f"/chart/{'0' * 64}").status_code == 404
        assert harness.renderer.calls == []  # nothing rendered for a miss

    def test_malformed_hash_is_404_not_500(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        client = TestClient(harness.app)
        for hostile in ("not-a-hash", "..%2f..%2fsecret", "A" * 64):
            assert client.get(f"/chart/{hostile}").status_code == 404

    def test_permalink_makes_no_llm_call(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        stored_hash = harness.spec_store.put(SYNTHETIC_SPEC)
        TestClient(harness.app).get(f"/chart/{stored_hash}")
        assert harness.adapter.calls == []
