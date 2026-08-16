"""Unit test for the stub FastAPI app's /health route.

Uses FastAPI's in-process TestClient (no network, no Docker) — the
Docker/HTTP-level check that the same endpoint responds once containerised
is `tests/smoke/test_healthchecks.py`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from service.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
