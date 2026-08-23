"""Server-side daily budget tracking and the fail-closed cut-off (ADR-015).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_service_budget.py`` and
``tests/unit/test_service_read_only.py`` pin the contract.

DESIGN §9 / ADR-015 as amended: a hard server-side daily spend cap; on
breach the service fails CLOSED to the read-only state (cached starter
answers + flagship charts + static surfaces still served; chat and chart
generation refuse with the dated "paused for today" response — never an
error page, never a dark site). Opus "best" mode sits behind a LOWER
sub-cap inside the daily cap, wired through the #186-hardened
``GenerationConfig.budget_guard`` seam in ``rag.generation``.

Contract points the red suite pins:

- **Accumulation is atomic.** Concurrent ``record_usage`` calls never
  lose spend (a lost increment is a hole in the fail-closed invariant).
- **Costs come from usage records via the single pricing source.**
  ``record_usage`` converts an adapter-reported usage mapping (Anthropic
  key names, cache metadata included) to USD with
  ``evals.pricing.estimate_cost_usd`` — never a second pricing table.
- **Breach flips the state machine to read-only, not to errors.**
  ``mode()`` is :data:`ServiceMode.PAUSED` at/over the cap, LIVE below.
- **Fail closed on tracker failure.** If spend state cannot be read,
  the answer is PAUSED — every failure of the tracking mechanism
  degrades toward *not spending* (ADR-015).
- **Daily reset at midnight UTC** (injected clock; never wall time).
- **The Opus sub-cap is independent**: the guard refuses gated-model
  traffic at the sub-cap while default-model traffic continues under
  the daily cap; snapshot ids gate by family (finding #186).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from enum import StrEnum

__all__ = [
    "ServiceMode",
    "BudgetPausedError",
    "OpusSubCapExceededError",
    "SpendTracker",
    "paused_response_text",
]


class ServiceMode(StrEnum):
    """The budget state machine's two states (ADR-015 as amended)."""

    LIVE = "live"
    PAUSED = "paused"


class BudgetPausedError(Exception):
    """The daily cap is breached (or unknowable): no LLM spend may occur."""


class OpusSubCapExceededError(BudgetPausedError):
    """The Opus daily sub-cap is spent; gated-model traffic must not run.

    Subclasses :class:`BudgetPausedError` so "no gated spend" callers can
    catch one type; the service distinguishes them to fall back to the
    default model instead of pausing the whole exchange.
    """


class SpendTracker:
    """Server-side atomic daily spend accumulator + the budget state machine.

    ``clock`` is injected (returns an aware UTC ``datetime``); the day
    key is the UTC date — "resets at midnight" means midnight UTC.
    ``spend_reader`` is the optional persistence-read seam: when provided
    it is consulted by ``spent_today``/``mode`` and ANY exception it
    raises makes ``mode()`` PAUSED (fail closed), never LIVE and never a
    propagated crash on the request path.
    """

    def __init__(
        self,
        *,
        daily_budget_usd: float,
        opus_subcap_usd: float,
        clock: Callable[[], datetime],
        spend_reader: Callable[[date], Mapping[str, float]] | None = None,
    ) -> None:
        self.daily_budget_usd = daily_budget_usd
        self.opus_subcap_usd = opus_subcap_usd
        self._clock = clock
        self._spend_reader = spend_reader

    def record_usage(self, model: str, usage: Mapping[str, int], *, mode: str = "live") -> float:
        """Record one adapter-reported usage mapping; return the USD cost added.

        Accepts the seam's usage key names (``input_tokens``,
        ``output_tokens``, ``cache_read_input_tokens``,
        ``cache_creation_input_tokens``); cost is computed by
        ``evals.pricing.estimate_cost_usd`` and accumulated atomically
        under today's UTC day key, with gated-family (opus) spend also
        accumulated separately for the sub-cap. ``None``/absent keys
        count as zero. Thread-safe: concurrent calls never lose spend.
        """
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def spent_today(self) -> float:
        """Total USD recorded under today's UTC day key (0.0 for a fresh day)."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def opus_spent_today(self) -> float:
        """USD recorded today for gated (non-default-family) models only."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def mode(self) -> ServiceMode:
        """The state machine: PAUSED at/over the daily cap or on tracker
        failure; LIVE otherwise. Never raises on the request path."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def budget_guard(self, model_id: str) -> None:
        """The #186 ``GenerationConfig.budget_guard`` hook (fail-closed).

        Called with the EXACT model id before a gated-model request is
        built. Raises :class:`OpusSubCapExceededError` when today's
        gated-family spend has reached ``opus_subcap_usd``, and
        :class:`BudgetPausedError` when the service is PAUSED (defence
        in depth — the endpoint should have refused already). Returns
        None when the spend may proceed.
        """
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")


#: How long the pause lasts: until the next UTC midnight (documented for
#: the response template below).
PAUSE_RESETS_AFTER = timedelta(days=1)


def paused_response_text(on_date: date) -> str:
    """Pure template: the dated "paused for today" chat/chart response.

    Must contain the phrase "paused for today", the ISO date itself
    (``on_date.isoformat()`` — the "clearly dated" ADR-015 requirement),
    and honest UX copy pointing at what still works: the cached starter
    answers, charts, and the /about and sources pages. Fixed template —
    interpolates NOTHING user- or model-derived except the date.
    """
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")
