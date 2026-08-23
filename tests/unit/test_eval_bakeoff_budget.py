"""Issue #21 red phase: bake-off selection + cost discipline.

The bake-off (issue #21 orchestrator comments): Haiku and Sonnet arms,
cheapest-passing-wins over per-arm gate results + ledger costs; the
Opus arm is escalation-only (ratified: NO top-up — everything must
pre-flight within the remaining budget under the $9.00 cap).

Cost discipline (dev-cost-plan M8): every live/recording run pre-flights
its estimate through evals/pricing.py against the cap BEFORE starting;
live runs append to evals/spend-ledger.csv via evals/ledger.py; replay
runs cost $0 and never touch the ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import ledger
from evals.gates import (
    GATE_BLOCKED,
    GATE_FAILED,
    GATE_PASSED,
    ArmResult,
    GateResult,
    arm_passes,
    opus_escalation_allowed,
    select_production_model,
)
from evals.harness import BudgetPreflight, preflight_budget, record_run_spend


def _arm(model: str, cost: float, *statuses: str) -> ArmResult:
    gates = tuple(
        GateResult(name=f"gate-{index}", status=status) for index, status in enumerate(statuses)
    )
    return ArmResult(model=model, gates=gates, cost_usd=cost)


def test_cheapest_passing_model_wins():
    haiku = _arm("claude-haiku-4-5", 0.36, GATE_PASSED, GATE_PASSED)
    sonnet = _arm("claude-sonnet-5", 0.79, GATE_PASSED, GATE_PASSED)
    assert arm_passes(haiku) and arm_passes(sonnet)
    assert select_production_model([sonnet, haiku]) == "claude-haiku-4-5"

    # Haiku fails a gate → the pricier passing arm wins.
    haiku_failing = _arm("claude-haiku-4-5", 0.36, GATE_PASSED, GATE_FAILED)
    assert not arm_passes(haiku_failing)
    assert select_production_model([haiku_failing, sonnet]) == "claude-sonnet-5"


def test_blocked_gates_never_count_as_passing_an_arm():
    """A blocked gate (pending owner audit) is not a pass: the arm does
    not pass, and cannot be selected."""
    blocked = _arm("claude-haiku-4-5", 0.36, GATE_PASSED, GATE_BLOCKED)
    assert not arm_passes(blocked)
    assert select_production_model([blocked]) is None


def test_no_passing_model_selects_none_for_escalation():
    """When no arm passes cleanly the harness escalates with the numbers
    instead of picking one (issue #21 orchestrator comment)."""
    arms = [
        _arm("claude-haiku-4-5", 0.36, GATE_FAILED),
        _arm("claude-sonnet-5", 0.79, GATE_FAILED),
    ]
    assert select_production_model(arms) is None


def test_opus_arm_is_escalation_only_with_no_top_up():
    """Opus runs only after both cheaper arms failed, and only when its
    pre-flight fits the REMAINING budget — a failing pre-flight is a
    refusal, never a top-up."""
    failing_arms = [
        _arm("claude-haiku-4-5", 0.36, GATE_FAILED),
        _arm("claude-sonnet-5", 0.79, GATE_FAILED),
    ]
    passing_arms = [
        _arm("claude-haiku-4-5", 0.36, GATE_PASSED),
        _arm("claude-sonnet-5", 0.79, GATE_FAILED),
    ]
    ok_preflight = BudgetPreflight(estimated_cost_usd=0.5, cumulative_usd=2.0, allowed=True)
    over_preflight = BudgetPreflight(estimated_cost_usd=2.0, cumulative_usd=8.5, allowed=False)

    assert opus_escalation_allowed(failing_arms, ok_preflight) is True
    # A cheaper arm passed → no Opus spend, even within budget.
    assert opus_escalation_allowed(passing_arms, ok_preflight) is False
    # Both failed but the budget would be exceeded → NO top-up, refuse.
    assert opus_escalation_allowed(failing_arms, over_preflight) is False


# ---------------------------------------------------------------------------
# Budget pre-flight + ledger discipline
# ---------------------------------------------------------------------------


def _seed_ledger(tmp_path: Path, cumulative: float) -> Path:
    path = tmp_path / "spend-ledger.csv"
    ledger.append_row(
        path,
        {
            "date": "2026-08-23",
            "session_id": "syn-session",
            "activity": "seed",
            "issue": "21",
            "model": "claude-haiku-4-5",
            "mode": "live",
            "calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": cumulative,
        },
    )
    return path


def test_preflight_prices_through_pricing_module_and_enforces_the_cap(tmp_path: Path):
    """Hand-computed: 100K Haiku input tokens live = $0.10; from $8.90
    cumulative that reaches the $9.00 cap exactly → refused (fail
    closed). A $0.05 batched run (100K input at 50% off) stays under →
    allowed."""
    ledger_path = _seed_ledger(tmp_path, 8.90)

    over = preflight_budget(
        [
            {
                "model": "claude-haiku-4-5",
                "input_tokens": 100_000,
                "output_tokens": 0,
                "mode": "live",
            }
        ],
        ledger_path=ledger_path,
    )
    assert over.estimated_cost_usd == pytest.approx(0.10)
    assert over.cumulative_usd == pytest.approx(8.90)
    assert over.threshold_usd == pytest.approx(9.00)
    assert over.allowed is False

    under = preflight_budget(
        [
            {
                "model": "claude-haiku-4-5",
                "input_tokens": 100_000,
                "output_tokens": 0,
                "mode": "batch",
            }
        ],
        ledger_path=ledger_path,
    )
    assert under.estimated_cost_usd == pytest.approx(0.05)
    assert under.allowed is True


def test_preflight_rejects_unknown_models_loudly(tmp_path: Path):
    """evals/pricing.py is the single pricing table: an unknown model
    must refuse, never estimate $0."""
    ledger_path = _seed_ledger(tmp_path, 0.10)
    with pytest.raises(ValueError):
        preflight_budget(
            [
                {
                    "model": "claude-mystery-9",
                    "input_tokens": 1_000,
                    "output_tokens": 0,
                    "mode": "live",
                }
            ],
            ledger_path=ledger_path,
        )


def test_live_runs_append_to_the_ledger_with_priced_rows(tmp_path: Path):
    """A batch-mode run appends one M8 row priced through
    evals/pricing.py: 1M in + 100K out on Haiku batched =
    (1.0 + 0.5) * 0.5 = $0.75; cumulative advances."""
    ledger_path = _seed_ledger(tmp_path, 0.10)
    row = record_run_spend(
        ledger_path,
        mode="batch",
        model="claude-haiku-4-5",
        activity="release-eval-arm",
        usage={"input_tokens": 1_000_000, "output_tokens": 100_000},
        calls=94,
        session_id="syn-run",
    )
    assert row is not None
    assert float(row["cost_usd"]) == pytest.approx(0.75)
    assert ledger.cumulative_usd(ledger_path) == pytest.approx(0.85)


def test_replay_runs_cost_zero_and_skip_the_ledger(tmp_path: Path):
    """fake/replay runs never touch the ledger — no row, no file."""
    ledger_path = tmp_path / "spend-ledger.csv"
    for mode in ("replay", "fake"):
        result = record_run_spend(
            ledger_path,
            mode=mode,
            model="claude-haiku-4-5",
            activity="release-eval-arm",
            usage={"input_tokens": 1_000_000, "output_tokens": 100_000},
            calls=94,
            session_id="syn-run",
        )
        assert result is None
    assert not ledger_path.exists()
