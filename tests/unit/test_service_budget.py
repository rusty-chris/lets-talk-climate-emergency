"""The hard daily budget cut-off, unit tier (issue #22, ADR-015, DESIGN §9).

Pins ``service.budget.SpendTracker``: atomic accumulation from
adapter-reported usage records priced by the single pricing source, the
breach → read-only state machine (never errors), fail-closed behaviour
on tracker failure, the UTC-midnight daily reset, the independent Opus
sub-cap, and the guard's composition with the #186-hardened
``rag.generation`` best-mode gate.
"""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime

import pytest

from evals.pricing import estimate_cost_usd
from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    GenerationConfig,
    build_generation_request,
)
from service.budget import (
    BudgetPausedError,
    OpusSubCapExceededError,
    ServiceMode,
    SpendTracker,
    paused_response_text,
)
from tests._generation_fixtures import make_retrieved
from tests._service_fixtures import FrozenClock

HAIKU = "claude-haiku-4-5"


def make_tracker(
    *,
    daily: float = 1.0,
    subcap: float = 0.25,
    clock: FrozenClock | None = None,
) -> tuple[SpendTracker, FrozenClock]:
    clock = clock or FrozenClock()
    tracker = SpendTracker(daily_budget_usd=daily, opus_subcap_usd=subcap, clock=clock)
    return tracker, clock


def test_spend_tracker_accumulates_atomically() -> None:
    """Concurrent increments never lose spend (the fail-closed invariant
    is only as good as the accumulator under it)."""
    tracker, _ = make_tracker(daily=1_000_000.0)
    usage = {"input_tokens": 1_000, "output_tokens": 500}
    per_call = estimate_cost_usd(HAIKU, input_tokens=1_000, output_tokens=500)
    threads_n, calls_per_thread = 8, 50

    def worker() -> None:
        for _ in range(calls_per_thread):
            tracker.record_usage(HAIKU, usage)

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tracker.spent_today() == pytest.approx(per_call * threads_n * calls_per_thread)


def test_costs_come_from_usage_records_via_the_single_pricing_source() -> None:
    """record_usage prices the seam's usage keys (cache metadata included)
    exactly as evals.pricing.estimate_cost_usd does — one pricing table."""
    tracker, _ = make_tracker()
    usage = {
        "input_tokens": 900,
        "output_tokens": 42,
        "cache_read_input_tokens": 4200,
        "cache_creation_input_tokens": 100,
    }
    recorded = tracker.record_usage(HAIKU, usage)
    expected = estimate_cost_usd(
        HAIKU,
        input_tokens=900,
        output_tokens=42,
        cache_read_tokens=4200,
        cache_creation_tokens=100,
    )
    assert recorded == pytest.approx(expected)
    assert tracker.spent_today() == pytest.approx(expected)


def test_unknown_model_refuses_loudly_never_a_silent_zero_row() -> None:
    tracker, _ = make_tracker()
    with pytest.raises(ValueError):
        tracker.record_usage("mystery-model-9", {"input_tokens": 10, "output_tokens": 1})
    assert tracker.spent_today() == pytest.approx(0.0)


def test_breach_switches_to_read_only() -> None:
    """Crossing the cap flips the state machine to PAUSED — not to errors:
    record_usage itself never raises for the breach."""
    tracker, _ = make_tracker(daily=0.001)
    assert tracker.mode() is ServiceMode.LIVE
    # One large generation pushes past the cap; recording must not raise.
    tracker.record_usage(HAIKU, {"input_tokens": 500_000, "output_tokens": 100_000})
    assert tracker.spent_today() > tracker.daily_budget_usd
    assert tracker.mode() is ServiceMode.PAUSED


def test_reaching_the_cap_exactly_is_a_breach() -> None:
    """Fail closed at the boundary: spend == cap pauses (never one more
    query 'because we are only at the cap')."""
    per_call = estimate_cost_usd(HAIKU, input_tokens=1_000_000, output_tokens=0)
    tracker, _ = make_tracker(daily=per_call)
    tracker.record_usage(HAIKU, {"input_tokens": 1_000_000, "output_tokens": 0})
    assert tracker.mode() is ServiceMode.PAUSED


def test_tracker_failure_fails_closed_to_paused() -> None:
    """ADR-015: every failure of the tracking mechanism degrades toward
    NOT spending. Unreadable spend state → PAUSED, never LIVE and never
    a crash on the request path."""

    def broken_reader(day: date) -> dict[str, float]:
        raise OSError("synthetic spend-state read failure")

    clock = FrozenClock()
    tracker = SpendTracker(
        daily_budget_usd=1.0,
        opus_subcap_usd=0.25,
        clock=clock,
        spend_reader=broken_reader,
    )
    assert tracker.mode() is ServiceMode.PAUSED


def test_daily_reset_at_midnight() -> None:
    """The day key is the UTC date: spend vanishes at midnight UTC, and
    the paused state ends with the day (injected clock, never wall time)."""
    tracker, clock = make_tracker(daily=0.001)
    tracker.record_usage(HAIKU, {"input_tokens": 500_000, "output_tokens": 100_000})
    assert tracker.mode() is ServiceMode.PAUSED

    # 23:59:59 on the same UTC day: still paused.
    clock.now = datetime(2026, 8, 20, 23, 59, 59, tzinfo=UTC)
    assert tracker.mode() is ServiceMode.PAUSED
    assert tracker.spent_today() > 0

    # One second past midnight UTC: a fresh day, a fresh budget.
    clock.now = datetime(2026, 8, 21, 0, 0, 1, tzinfo=UTC)
    assert tracker.spent_today() == pytest.approx(0.0)
    assert tracker.mode() is ServiceMode.LIVE


def test_opus_subcap_enforced_independently() -> None:
    """Opus is blocked at its sub-cap while Haiku continues under the
    daily cap — the expensive mode is structurally incapable of eating
    the whole day's budget."""
    tracker, _ = make_tracker(daily=100.0, subcap=0.05)
    # Spend the Opus sub-cap (Opus pricing: $5/$25 per MTok).
    tracker.record_usage(OPUS_BEST_MODEL, {"input_tokens": 8_000, "output_tokens": 1_000})
    assert tracker.opus_spent_today() >= tracker.opus_subcap_usd
    assert tracker.spent_today() < tracker.daily_budget_usd

    with pytest.raises(OpusSubCapExceededError):
        tracker.budget_guard(OPUS_BEST_MODEL)
    # The service is still LIVE and default-model traffic still flows.
    assert tracker.mode() is ServiceMode.LIVE
    tracker.record_usage(HAIKU, {"input_tokens": 1_000, "output_tokens": 100})


def test_haiku_spend_does_not_consume_the_opus_subcap() -> None:
    tracker, _ = make_tracker(daily=100.0, subcap=0.05)
    tracker.record_usage(HAIKU, {"input_tokens": 2_000_000, "output_tokens": 200_000})
    assert tracker.opus_spent_today() == pytest.approx(0.0)
    tracker.budget_guard(OPUS_BEST_MODEL)  # must not raise


def test_guard_refuses_everything_when_paused() -> None:
    """Defence in depth: once the daily cap is breached, the guard refuses
    gated traffic outright (BudgetPausedError, the broader type)."""
    tracker, _ = make_tracker(daily=0.001, subcap=0.0005)
    tracker.record_usage(HAIKU, {"input_tokens": 500_000, "output_tokens": 100_000})
    with pytest.raises(BudgetPausedError):
        tracker.budget_guard(OPUS_BEST_MODEL)


def test_opus_snapshot_ids_count_toward_the_subcap_by_family() -> None:
    """Finding #186 carried into accounting: a dated snapshot of the
    gated family spends the same sub-cap as the family id."""
    tracker, _ = make_tracker(daily=100.0, subcap=0.05)
    tracker.record_usage(
        f"{OPUS_BEST_MODEL}-20260115", {"input_tokens": 8_000, "output_tokens": 1_000}
    )
    assert tracker.opus_spent_today() > 0
    with pytest.raises(OpusSubCapExceededError):
        tracker.budget_guard(f"{OPUS_BEST_MODEL}-20260115")


class TestGuardComposesWithTheGenerationGate:
    """The tracker's guard IS the #186 ``GenerationConfig.budget_guard``:
    the hardened gate calls it with the exact model id before any request
    exists, and its refusal prevents the request from being built."""

    def test_guard_invoked_with_exact_model_id_through_the_gate(self) -> None:
        tracker, _ = make_tracker(daily=100.0, subcap=1.0)
        seen: list[str] = []

        def spy_guard(model_id: str) -> None:
            seen.append(model_id)
            tracker.budget_guard(model_id)

        config = GenerationConfig(
            model=f"{OPUS_BEST_MODEL}-20260115",
            best_mode_enabled=True,
            budget_guard=spy_guard,
        )
        build_generation_request(make_retrieved(), "a question", config=config)
        assert seen == [f"{OPUS_BEST_MODEL}-20260115"]

    def test_subcap_refusal_stops_the_request_before_it_exists(self) -> None:
        tracker, _ = make_tracker(daily=100.0, subcap=0.05)
        tracker.record_usage(OPUS_BEST_MODEL, {"input_tokens": 8_000, "output_tokens": 1_000})
        config = GenerationConfig(
            model=OPUS_BEST_MODEL,
            best_mode_enabled=True,
            budget_guard=tracker.budget_guard,
        )
        with pytest.raises(OpusSubCapExceededError):
            build_generation_request(make_retrieved(), "a question", config=config)

    def test_default_model_traffic_never_consults_the_guard(self) -> None:
        calls: list[str] = []

        def recording_guard(model_id: str) -> None:
            calls.append(model_id)

        config = GenerationConfig(model=GENERATION_MODEL_DEFAULT, budget_guard=recording_guard)
        build_generation_request(make_retrieved(), "a question", config=config)
        assert calls == []


class TestPausedResponseText:
    def test_paused_response_is_dated_and_honest(self) -> None:
        text = paused_response_text(date(2026, 8, 20))
        assert "paused for today" in text
        assert "2026-08-20" in text
        # Honest UX copy: the reader is told what still works.
        assert "starter" in text.lower() or "cached" in text.lower()

    def test_paused_response_interpolates_only_the_date(self) -> None:
        one = paused_response_text(date(2026, 8, 20))
        two = paused_response_text(date(2026, 8, 21))
        assert one.replace("2026-08-20", "X") == two.replace("2026-08-21", "X")
