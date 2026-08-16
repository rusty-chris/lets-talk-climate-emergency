"""Stub FastAPI application backing the `api` docker-compose service.

This is scaffolding only (issue #1): a single `/health` endpoint so
`docker compose up` brings the `api` service to a responding health check.
The real routes (chat, chart planning, permalinks, budget/rate-limit
middleware) land in issues #14, #15, #22 per IMPLEMENTATION.md §1.
"""

from fastapi import FastAPI

app = FastAPI(title="Let's Talk About the Climate Emergency — API (stub)")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/health endpoint used by docker-compose and CI smoke tests."""
    return {"status": "ok"}
