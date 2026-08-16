"""Single source of truth for Anthropic API pricing (dev-cost-plan M8).

RED phase contract stub (finding #92): `tests/unit/test_spend_ledger.py`
pins the hand-computed arithmetic; the green commit implements it.
"""

from __future__ import annotations


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    mode: str = "live",
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """USD cost of a call/run from its token usage (per-MTok table, M8)."""
    raise NotImplementedError("finding #92: implemented in the green commit")
