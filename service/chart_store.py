"""The chart-spec permalink store (issue #22, DESIGN §3.7 permalinks).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_service_permalinks.py`` and
``tests/unit/test_service_chat_pipeline.py`` pin the contract.

``/chart/<spec-hash>`` re-renders from the STORED spec + pinned dataset
version (DESIGN §3.7): a permalink is a ~1 KB JSON spec addressed by
``charts.spec.spec_hash``, never a stored image and never a live fetch.
Specs are stored when the chart pipeline produces them (and at release
time for the flagship charts), so permalinks keep serving in the paused
read-only state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["ChartSpecStore"]


class ChartSpecStore:
    """Filesystem-backed spec store, keyed by canonical spec hash."""

    def __init__(self, dir_path: Path) -> None:
        self.dir_path = Path(dir_path)

    def put(self, spec: Mapping[str, Any]) -> str:
        """Store ``spec`` under its ``charts.spec.spec_hash``; return the
        hash. Idempotent — the hash is content-derived, so re-storing an
        identical spec is a no-op with the same hash."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def get(self, spec_hash: str) -> dict[str, Any] | None:
        """The stored spec for ``spec_hash``, or None when unknown.

        ``spec_hash`` is untrusted URL input: anything that is not a
        64-char lowercase hex string returns None without touching the
        filesystem (no path traversal through the permalink route).
        """
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")
