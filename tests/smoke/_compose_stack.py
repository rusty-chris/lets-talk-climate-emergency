"""Shared compose-stack lifecycle for the smoke tier (review finding #231).

One context manager owns `docker compose up/down` with a given
environment overlay, so the paused-stack suite and the replay-stack
suite (which need DIFFERENT api environments and therefore must never
run concurrently — they share host ports) manage the stack identically.
Module-scoped fixtures in each test module keep the two stacks
sequential: pytest tears down one module's fixtures before the next
module's come up.

CI convention (finding #32 / tests/_docker.py): a missing Docker skips
locally and fails in CI (GitHub Actions sets CI=true), so the smoke tier
can never silently become a green no-op.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import httpx

from tests._docker import require_docker

REPO_ROOT = Path(__file__).resolve().parents[2]
API = "http://localhost:8000"
STARTUP_TIMEOUT_S = 120
POLL_INTERVAL_S = 2


def wait_healthy(deadline: float, *, context: str) -> None:
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{API}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"api service never became healthy under the {context} environment")


@contextlib.contextmanager
def compose_stack(env_overlay: Mapping[str, str], *, context: str) -> Iterator[dict[str, str]]:
    """`docker compose up` with the overlay applied; always composes down."""
    require_docker()
    env = {**os.environ, **env_overlay}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    try:
        wait_healthy(time.monotonic() + STARTUP_TIMEOUT_S, context=context)
        yield env
    finally:
        subprocess.run(["docker", "compose", "down", "-v"], cwd=REPO_ROOT, check=False, env=env)
