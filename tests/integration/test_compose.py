"""Integration test: docker-compose.yml is valid and names the stub services.

Issue #1 acceptance criterion: `docker compose config` parses and names the
`api`, `qdrant`, `ui` services. Requires the Docker CLI with the `compose`
plugin; skipped when Docker is unavailable locally, failed when it is
unavailable in CI (review finding #32 — a skip must not green a CI tier).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from tests._docker import require_docker

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose_config() -> dict:
    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return yaml.safe_load(result.stdout)


def test_compose_config_valid() -> None:
    require_docker()
    config = _compose_config()
    assert {"api", "qdrant", "ui"} <= set(config["services"].keys())


def test_published_ports_bound_to_loopback() -> None:
    """Every published port binds to 127.0.0.1, not 0.0.0.0 (finding #36).

    Unprefixed compose port bindings publish on all interfaces, and on
    Linux Docker's iptables rules typically bypass host firewalls — so
    `docker compose up` on shared Wi-Fi exposes the unauthenticated
    Qdrant HTTP/gRPC API (and the stub api/ui) to the whole network.
    Local dev needs nothing beyond loopback; deployment (#22) overrides
    deliberately for whatever ingress it designs.
    """
    require_docker()
    config = _compose_config()
    violations = []
    for service_name, service in config["services"].items():
        for port in service.get("ports", []):
            host_ip = port.get("host_ip")
            if host_ip != "127.0.0.1":
                violations.append(f"{service_name}: {port} (host_ip={host_ip!r})")
    assert not violations, (
        f"published ports not bound to loopback (unprefixed bindings mean 0.0.0.0): {violations}"
    )
