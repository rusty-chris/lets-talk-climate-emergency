"""The committed spend ledger, evals/spend-ledger.csv (dev-cost-plan M8).

One row per API-touching session or batch, appended by the recording tooling
and the eval harness and committed in the same PR as the results it accounts
for. `cumulative_usd` is recomputed on every append and enforced monotonic;
the budget pre-flight (M8) reads the last row's cumulative to refuse runs
past the $9.00 threshold. Arithmetic pinned by
`tests/unit/test_spend_ledger.py`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEDGER_COLUMNS = [
    "date",
    "session_id",
    "activity",
    "issue",
    "model",
    "mode",
    "calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "cost_usd",
    "cumulative_usd",
    "notes",
]

_LEDGER_COMMENT = "# evals/spend-ledger.csv - one row per API-touching session or batch (M8)"

# M8: refuse to start any live/batch run at/past this cumulative spend
# unless CLIMATE_CHAT_BUDGET_OVERRIDE=1 (mirrors DESIGN 9's fail-closed
# budget philosophy).
BUDGET_REFUSAL_THRESHOLD_USD = 9.00
BUDGET_OVERRIDE_ENV_FLAG = "CLIMATE_CHAT_BUDGET_OVERRIDE"


def read_rows(path: Path) -> list[dict[str, str]]:
    """All ledger rows (comment lines skipped), in file order."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line[:1] != "#" and line
    ]
    return list(csv.DictReader(lines))


def cumulative_usd(path: Path) -> float:
    """The ledger's current cumulative spend (0.0 for a fresh/empty ledger)."""
    rows = read_rows(path)
    return float(rows[-1]["cumulative_usd"]) if rows else 0.0


def append_row(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Append one M8 row, recomputing cumulative_usd; returns the full row.

    `cost_usd` must be non-negative — cumulative spend is monotonic by
    construction, never by hope.
    """
    path = Path(path)
    cost = float(row["cost_usd"])
    if cost < 0:
        raise ValueError(f"cost_usd must be non-negative, got {cost!r} (ledger monotonicity)")
    unknown = set(row) - set(LEDGER_COLUMNS)
    if unknown:
        raise ValueError(f"unknown ledger column(s) {sorted(unknown)}; M8 columns are fixed")

    previous = cumulative_usd(path)
    complete: dict[str, Any] = {column: row.get(column, "") for column in LEDGER_COLUMNS}
    complete["cost_usd"] = f"{cost:.6f}"
    complete["cumulative_usd"] = round(previous + cost, 6)

    is_new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        if is_new:
            handle.write(_LEDGER_COMMENT + "\n")
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(complete)
    return complete
