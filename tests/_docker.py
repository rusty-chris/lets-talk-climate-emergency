"""Shared Docker availability guard for the integration and smoke tiers.

Review finding #32: a bare skipif on Docker availability turns "the tier
tested nothing" into a green CI job. In CI (GitHub Actions always sets
CI=true) a missing Docker is an infrastructure failure and must be red;
on a dev machine a skip remains the right behaviour.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest


def docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "compose", "version"], capture_output=True)
    return result.returncode == 0


def require_docker() -> None:
    """Skip locally when Docker is missing; fail in CI.

    Call at the top of every Docker-dependent test (or fixture) instead
    of a skipif decorator.
    """
    if docker_compose_available():
        return
    if os.environ.get("CI"):
        pytest.fail(
            "Docker required in CI but unavailable — the integration/smoke "
            "tier would otherwise silently no-op with a green job"
        )
    pytest.skip("Docker (with the compose plugin) is not available in this environment")
