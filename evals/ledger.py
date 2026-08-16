"""The committed spend ledger, evals/spend-ledger.csv (dev-cost-plan M8).

RED phase contract stub (finding #92): `tests/unit/test_spend_ledger.py`
pins the append/read behaviour; the green commit implements it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEDGER_COLUMNS: list[str] = []


def read_rows(path: Path) -> list[dict[str, str]]:
    """All ledger rows (comment lines skipped), in file order."""
    raise NotImplementedError("finding #92: implemented in the green commit")


def append_row(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Append one M8 row, recomputing cumulative_usd; returns the full row."""
    raise NotImplementedError("finding #92: implemented in the green commit")
