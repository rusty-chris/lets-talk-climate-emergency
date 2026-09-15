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

import json
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

from evals.pricing import estimate_cost_usd
from service.atomic_write import atomic_write_text

#: The per-day spend journal filename under ``state_dir`` (#217).
SPEND_STATE_FILENAME = "spend-state.json"

#: The weekly cap window: today plus the trailing 6 UTC days.
WEEKLY_WINDOW_DAYS = 7
#: Per-day rows retained in the journal — one more than the weekly window so
#: the rolling sum is always complete; older rows are pruned on each write.
STATE_RETENTION_DAYS = WEEKLY_WINDOW_DAYS + 1

#: The default (ungated) generation family. Anything OUTSIDE it is gated
#: "best" mode and spends the Opus sub-cap (finding #186: matched by
#: family prefix, so a dated snapshot counts the same as the family id).
_DEFAULT_MODEL_FAMILY_PREFIX = "claude-haiku"

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
        weekly_budget_usd: float | None = None,
        spend_reader: Callable[[date], Mapping[str, float]] | None = None,
        state_dir: Path | None = None,
    ) -> None:
        # ``state_dir`` (#217 contract, pinned RED by
        # tests/unit/test_service_budget.py::TestSpendStatePersistence):
        # when set, the tracker journals each UTC day's accumulated
        # spend (total + gated) to a small state file under this
        # directory on EVERY record_usage, and a fresh tracker over the
        # same directory reads the current day back before serving — so
        # a restart (crash-loop, redeploy) can never re-spend the daily
        # cap. A corrupt/unreadable journal makes mode() PAUSED (the
        # ADR-015 unreadable-state rule); a new UTC day starts clean.
        self.daily_budget_usd = daily_budget_usd
        self.opus_subcap_usd = opus_subcap_usd
        #: Optional rolling 7-day cap. None ⇒ daily-only (backwards compatible).
        self.weekly_budget_usd = weekly_budget_usd
        self._clock = clock
        self._spend_reader = spend_reader
        self._state_dir = Path(state_dir) if state_dir is not None else None
        # Per-UTC-day accumulators; the lock makes concurrent record_usage
        # calls lose no spend (the fail-closed invariant is only as good as
        # the accumulator under it).
        self._lock = threading.Lock()
        self._spend_by_day: dict[date, float] = {}
        self._opus_spend_by_day: dict[date, float] = {}
        # An unreadable/corrupt journal fails closed: mode() reports PAUSED
        # rather than un-pausing on a spend state it cannot trust (#217).
        self._state_error = False
        if self._state_dir is not None:
            self._load_state()

    def _today(self) -> date:
        """Today's UTC date — the day key ('resets at midnight' = midnight UTC)."""
        return self._clock().astimezone(UTC).date()

    def _state_path(self) -> Path:
        assert self._state_dir is not None  # guarded by callers
        return self._state_dir / SPEND_STATE_FILENAME

    def _recent_cutoff(self) -> date:
        """The oldest UTC day still retained (today − STATE_RETENTION_DAYS + 1)."""
        return date.fromordinal(self._today().toordinal() - (STATE_RETENTION_DAYS - 1))

    def _load_state(self) -> None:
        """Read the journalled per-day spend back at startup (#217, extended).

        The journal now retains a rolling window of recent UTC days (for the
        weekly cap), not just today, so a restart re-spends neither the daily
        nor the weekly cap. Days older than the retention window are ignored (a
        new day starts clean). A single-day legacy journal ({"day","total",
        "opus"}) is still read. An unreadable/corrupt journal sets
        ``_state_error`` so ``mode()`` fails closed to PAUSED.
        """
        path = self._state_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "days" in data:  # current multi-day format
                rows = {
                    date.fromisoformat(day): (float(row["total"]), float(row.get("opus", 0.0)))
                    for day, row in dict(data["days"]).items()
                }
            else:  # legacy single-day journal
                rows = {
                    date.fromisoformat(data["day"]): (
                        float(data["total"]),
                        float(data.get("opus", 0.0)),
                    )
                }
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._state_error = True
            return
        cutoff = self._recent_cutoff()
        for day, (total, opus) in rows.items():
            if day >= cutoff:  # ignore anything older than the retention window
                self._spend_by_day[day] = total
                self._opus_spend_by_day[day] = opus

    def _write_state(self, day: date) -> None:
        """Journal the retained per-day spend atomically (temp file + rename).

        Called under ``_lock`` on every record so spend is on disk immediately
        — a crash (not just a clean shutdown) leaves the journal current. Writes
        the whole retained window (today back STATE_RETENTION_DAYS) and prunes
        older rows, so the weekly rolling sum survives a restart.
        """
        if self._state_dir is None:
            return
        cutoff = self._recent_cutoff()
        # Prune in-memory accumulators to the retained window so they don't grow.
        for stale in [d for d in self._spend_by_day if d < cutoff]:
            self._spend_by_day.pop(stale, None)
            self._opus_spend_by_day.pop(stale, None)
        days = {
            d.isoformat(): {
                "total": self._spend_by_day.get(d, 0.0),
                "opus": self._opus_spend_by_day.get(d, 0.0),
            }
            for d in sorted(self._spend_by_day)
            if d >= cutoff
        }
        # fsync-backed atomic replace (finding #302): the #217 journal's whole
        # point is crash-loop durability, so the spend rows are flushed to disk
        # before the rename.
        atomic_write_text(self._state_path(), json.dumps({"days": days}))

    def record_usage(self, model: str, usage: Mapping[str, int]) -> float:
        """Record one adapter-reported usage mapping; return the USD cost added.

        Accepts the seam's usage key names (``input_tokens``,
        ``output_tokens``, ``cache_read_input_tokens``,
        ``cache_creation_input_tokens``); cost is computed by
        ``evals.pricing.estimate_cost_usd`` and accumulated atomically
        under today's UTC day key, with gated-family (opus) spend also
        accumulated separately for the sub-cap. ``None``/absent keys
        count as zero. Thread-safe: concurrent calls never lose spend.
        """
        # Price first (outside the lock): an unknown model raises here —
        # a loud refusal, never a silent $0 row — before any state moves.
        cost = estimate_cost_usd(
            model,
            input_tokens=_usage_int(usage, "input_tokens"),
            output_tokens=_usage_int(usage, "output_tokens"),
            cache_read_tokens=_usage_int(usage, "cache_read_input_tokens"),
            cache_creation_tokens=_usage_int(usage, "cache_creation_input_tokens"),
        )
        day = self._today()
        gated = not model.startswith(_DEFAULT_MODEL_FAMILY_PREFIX)
        with self._lock:
            self._spend_by_day[day] = self._spend_by_day.get(day, 0.0) + cost
            if gated:
                self._opus_spend_by_day[day] = self._opus_spend_by_day.get(day, 0.0) + cost
            # Journal on EVERY record (under the lock, so the file matches
            # the accumulator): spend must survive a crash-loop, not only a
            # clean shutdown (#217).
            self._write_state(day)
        return cost

    def spent_today(self) -> float:
        """Total USD recorded under today's UTC day key (0.0 for a fresh day)."""
        if self._spend_reader is not None:
            # The persistence-read seam is authoritative when installed; any
            # exception it raises propagates to mode(), which fails closed.
            return float(self._spend_reader(self._today()).get("total", 0.0))
        with self._lock:
            return self._spend_by_day.get(self._today(), 0.0)

    def opus_spent_today(self) -> float:
        """USD recorded today for gated (non-default-family) models only."""
        with self._lock:
            return self._opus_spend_by_day.get(self._today(), 0.0)

    def spent_this_week(self) -> float:
        """Total USD over the trailing WEEKLY_WINDOW_DAYS UTC days (incl. today).

        The rolling weekly window: today and the six prior UTC days. Reads the
        in-memory accumulators (seeded from the journal at startup), so a
        restart does not reset the week."""
        today = self._today()
        oldest = date.fromordinal(today.toordinal() - (WEEKLY_WINDOW_DAYS - 1))
        with self._lock:
            return sum(cost for day, cost in self._spend_by_day.items() if oldest <= day <= today)

    def mode(self) -> ServiceMode:
        """The state machine: PAUSED at/over the daily cap, at/over the weekly
        cap (when set), or on tracker failure; LIVE otherwise. Never raises on
        the request path."""
        if self._state_error:
            # A corrupt/unreadable spend journal is an unknowable spend
            # state: fail closed (ADR-015), never un-pause on it.
            return ServiceMode.PAUSED
        try:
            spent = self.spent_today()
            # spend == cap is a breach (fail-closed boundary, ratified #22.6).
            if spent >= self.daily_budget_usd:
                return ServiceMode.PAUSED
            if (
                self.weekly_budget_usd is not None
                and self.spent_this_week() >= self.weekly_budget_usd
            ):
                return ServiceMode.PAUSED
        except Exception:
            # ADR-015: every failure of the tracking mechanism degrades
            # toward NOT spending — unreadable spend state pauses.
            return ServiceMode.PAUSED
        return ServiceMode.LIVE

    def snapshot(self) -> dict[str, float | str | None]:
        """A read-only spend snapshot for the operator display (never raises).

        Returns the current mode plus today's and the week's spend against
        their caps. On any tracker failure it reports PAUSED with unknown
        (None) spend rather than raising onto the request path."""
        try:
            spent_today = self.spent_today()
            spent_week = self.spent_this_week()
        except Exception:
            return {
                "mode": ServiceMode.PAUSED.value,
                "spent_today_usd": None,
                "daily_cap_usd": self.daily_budget_usd,
                "spent_week_usd": None,
                "weekly_cap_usd": self.weekly_budget_usd,
            }
        return {
            "mode": self.mode().value,
            "spent_today_usd": spent_today,
            "daily_cap_usd": self.daily_budget_usd,
            "spent_week_usd": spent_week,
            "weekly_cap_usd": self.weekly_budget_usd,
        }

    def budget_guard(self, model_id: str) -> None:
        """The #186 ``GenerationConfig.budget_guard`` hook (fail-closed).

        Called with the EXACT model id before a gated-model request is
        built. Raises :class:`OpusSubCapExceededError` when today's
        gated-family spend has reached ``opus_subcap_usd``, and
        :class:`BudgetPausedError` when the service is PAUSED (defence
        in depth — the endpoint should have refused already). Returns
        None when the spend may proceed.
        """
        # Defence in depth: a breached daily cap refuses gated traffic
        # outright (the endpoint should already have paused).
        if self.mode() is ServiceMode.PAUSED:
            raise BudgetPausedError("the daily spend cap is breached — no LLM spend may occur")
        # Gated-family (opus) traffic is refused independently at its lower
        # sub-cap while default-model traffic keeps flowing under the cap.
        if not model_id.startswith(_DEFAULT_MODEL_FAMILY_PREFIX):
            if self.opus_spent_today() >= self.opus_subcap_usd:
                raise OpusSubCapExceededError(
                    f"the Opus daily sub-cap (${self.opus_subcap_usd}) is spent; "
                    "gated-model traffic must fall back to the default model"
                )


def _usage_int(usage: Mapping[str, int], key: str) -> int:
    """A usage-mapping token count as an int; ``None``/absent counts as zero."""
    value = usage.get(key)
    return int(value) if value else 0


def paused_response_text(on_date: date) -> str:
    """Pure template: the dated "paused for today" chat/chart response.

    Must contain the phrase "paused for today", the ISO date itself
    (``on_date.isoformat()`` — the "clearly dated" ADR-015 requirement),
    and honest UX copy pointing at what still works: the cached starter
    answers, charts, and the /about and sources pages. Fixed template —
    interpolates NOTHING user- or model-derived except the date.
    """
    return (
        f"This briefing has paused for today ({on_date.isoformat()}) because it "
        "reached its daily running-cost cap. It is not down — the daily budget "
        "resets at midnight UTC, when live answers return. In the meantime you "
        "can still read the cached starter answers on the home page, view every "
        "chart, and browse the /about and sources pages, all served from the "
        "clearly-dated cached briefing."
    )
