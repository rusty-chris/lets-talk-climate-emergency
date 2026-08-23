"""Single source of truth for Anthropic API pricing (dev-cost-plan M8).

reviews/dev-cost-plan-2026-08.md, pricing basis: per-MTok input/output rates
Haiku 4.5 $1/$5, Sonnet $3/$15, Opus $5/$25; Batches API 50% off all token
usage; prompt-cache reads ~0.1x and writes 1.25x the input rate. Every
ledger row's `cost_usd` must be computed here — one table, unit-tested with
hand-computed values (`tests/unit/test_spend_ledger.py`), like the rest of
the eval-harness arithmetic (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

# Matched by model-family prefix so a point release (claude-haiku-4-6) does
# not silently price at $0; genuinely unknown models are rejected loudly.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku": (1.0, 5.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (5.0, 25.0),
}

BATCH_MULTIPLIER = 0.5
CACHE_READ_INPUT_MULTIPLIER = 0.1
CACHE_WRITE_INPUT_MULTIPLIER = 1.25

#: The pricing vocabulary: the two spend modes a ledger row can carry.
#: Public (finding #238): callers price/ledger against this, never the
#: private membership set.
MODES = frozenset({"live", "batch"})
_MODES = MODES


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    mode: str = "live",
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """USD cost of a call/run from its token usage.

    `input_tokens`/`output_tokens` follow the API's `usage` semantics:
    cache read/creation tokens are reported separately and priced at their
    multipliers of the input rate. `mode="batch"` halves everything.
    """
    rates = next(
        (rates for prefix, rates in PRICES_PER_MTOK.items() if model.startswith(prefix)),
        None,
    )
    if rates is None:
        raise ValueError(
            f"unknown model {model!r}: add its family to evals/pricing.py "
            "PRICES_PER_MTOK - the ledger must never record a silent $0 row"
        )
    rate_in, rate_out = rates
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}: expected one of {sorted(_MODES)}")

    cost = (
        input_tokens * rate_in
        + output_tokens * rate_out
        + cache_read_tokens * rate_in * CACHE_READ_INPUT_MULTIPLIER
        + cache_creation_tokens * rate_in * CACHE_WRITE_INPUT_MULTIPLIER
    ) / 1_000_000
    if mode == "batch":
        cost *= BATCH_MULTIPLIER
    return cost
