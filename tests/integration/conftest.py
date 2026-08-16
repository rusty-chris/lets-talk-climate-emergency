"""Auto-apply the `integration` marker to everything in this directory.

Tier membership is dictated by directory (IMPLEMENTATION.md §3); a
forgotten `pytestmark` must not silently move a Docker-dependent test
into the unit tier or drop it from its own tier (review finding #31).
Enforced by tests/unit/test_tier_enforcement.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TIER_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path is not None and item.path.is_relative_to(_TIER_DIR):
            item.add_marker(pytest.mark.integration)
