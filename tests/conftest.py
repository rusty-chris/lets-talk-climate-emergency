"""Shared pytest fixtures: adapter injection + fixture-corpus paths (issue #24).

Every LLM-dependent issue (#10, #12, #13, #16, #21) injects its adapter and
locates the synthetic fixture corpus through these fixtures rather than
improvising its own paths or doubles (IMPLEMENTATION.md §4–5). Tier
membership is unaffected — the per-directory conftests under
tests/integration/ and tests/smoke/ keep auto-marking their trees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.provider import FakeAdapter, ReplayAdapter

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    """An unprogrammed FakeAdapter: records calls; any call raises until programmed.

    Tests that need responses program their own instance (or call
    `queue(method, *results)` on this one); tests asserting zero-call
    invariants (unsafe path, paused service) use it as-is.
    """
    return FakeAdapter()


@pytest.fixture
def replay_adapter(replay_fixtures_dir: Path) -> ReplayAdapter:
    """A ReplayAdapter over the checked-in fixtures in tests/fixtures/replay/."""
    return ReplayAdapter(replay_fixtures_dir)


@pytest.fixture(scope="session")
def fixture_corpus_dir() -> Path:
    """tests/fixtures/corpus/ — the synthetic document corpus (issue #24)."""
    return FIXTURES_ROOT / "corpus"


@pytest.fixture(scope="session")
def fixture_manifest_path(fixture_corpus_dir: Path) -> Path:
    """The synthetic corpus manifest covering every permitted_context value."""
    return fixture_corpus_dir / "manifest.yaml"


@pytest.fixture(scope="session")
def chart_fixtures_dir() -> Path:
    """tests/fixtures/charts/ — tiny synthetic CSVs for chart-transform tests."""
    return FIXTURES_ROOT / "charts"


@pytest.fixture(scope="session")
def replay_fixtures_dir() -> Path:
    """tests/fixtures/replay/ — recorded (scrubbed) LLM response fixtures."""
    return FIXTURES_ROOT / "replay"
