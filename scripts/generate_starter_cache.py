"""RED-phase contract stub for the resumable starter-cache generator.

Behaviour raises ``NotImplementedError``; the failing suite in
``tests/unit/test_generate_starter_cache.py`` pins the contract. The GREEN
implementation follows in the next commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

CARRIED_SPEND_FILENAME = "carried_spend.json"


class SpendCapReached(RuntimeError):
    """Raised when a call would cross the pre-call line or the hard cap."""


class SpendMeter:
    """Cross-run fail-closed spend meter (contract stub)."""

    def __init__(self, ledger_path, *, hard_cap_usd, pre_call_line_usd, cost_fn=None) -> None:
        raise NotImplementedError

    def check(self, label: str) -> None:
        raise NotImplementedError

    def record(self, segment: str, model: str, usage: Mapping | None) -> None:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError


def entry_is_valid(saved_entry: Mapping, question: str) -> tuple[bool, str]:
    raise NotImplementedError


def generate_starter_cache(
    questions, *, entries_dir: Path, out_cache_dir: Path, meter, answer_fn, generated_on: str
) -> dict:
    raise NotImplementedError
