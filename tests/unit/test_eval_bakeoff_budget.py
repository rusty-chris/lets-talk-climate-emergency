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

from evals import harness, ledger
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
from evals.harness import (
    AnswerPathDeps,
    BudgetExceededError,
    BudgetPreflight,
    load_and_validate_gold,
    preflight_budget,
    record_run_spend,
)
from evals.judges import SONNET_ARM_MODEL
from rag.provider import FakeAdapter
from rag.retrieval import HonestRefusal
from tests._eval_harness_fixtures import FakeBatchClient, write_synthetic_gold


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


# ---------------------------------------------------------------------------
# review-21 #238 (major): every spending adapter mode is ledgered; unknown
# modes refuse loudly — a typo can never silently under-count the cap.
# ---------------------------------------------------------------------------


def test_recording_mode_spend_is_ledgered(tmp_path: Path):
    """A recording session burns real tokens (replay-fixture capture is
    an explicit spend activity): record_run_spend(mode='recording')
    appends a row priced at LIVE (non-batch) rates — 1M in + 100K out on
    Haiku = $1.00 + $0.50 = $1.50 (#238)."""
    ledger_path = tmp_path / "spend-ledger.csv"
    row = record_run_spend(
        ledger_path,
        mode="recording",
        model="claude-haiku-4-5",
        activity="replay-fixture-capture",
        usage={"input_tokens": 1_000_000, "output_tokens": 100_000},
        calls=10,
        session_id="syn-recording",
    )
    assert row is not None, "recording spends real money: the ledger row is mandatory"
    assert float(row["cost_usd"]) == pytest.approx(1.50)
    assert ledger.cumulative_usd(ledger_path) == pytest.approx(1.50)


def test_unknown_spend_mode_refused_loudly(tmp_path: Path):
    """A mode outside the closed vocabulary ('Live', 'batched', any
    typo) raises ValueError naming the offender — never a silent
    unledgered return (#238). The ledger file stays untouched."""
    ledger_path = tmp_path / "spend-ledger.csv"
    for bad_mode in ("Live", "batched"):
        with pytest.raises(ValueError) as excinfo:
            record_run_spend(
                ledger_path,
                mode=bad_mode,
                model="claude-haiku-4-5",
                activity="release-eval-arm",
                usage={"input_tokens": 1_000, "output_tokens": 100},
                calls=1,
                session_id="syn-run",
            )
        assert bad_mode in str(excinfo.value)
    assert not ledger_path.exists()


# ---------------------------------------------------------------------------
# review-21 #236 (blocker): a planned-calls estimator is the single honest
# estimate source, and the release orchestrator re-reads the ledger at
# every spend segment — a pre-flight is never a stale bearer token.
# ---------------------------------------------------------------------------


def test_planned_calls_estimator_covers_generation_and_judges(tmp_path: Path):
    """estimate_planned_calls derives the pre-flight's planned_calls
    from the gold sets themselves: one batched generation entry per
    answerable item at GENERATION_MAX_TOKENS_DEFAULT output, plus one
    batched judge entry per gate-driven judge request (see #241) whose
    input estimate covers the actual prompt (the severity judge embeds
    the whole rubric) — never caller-invented numbers (#236)."""
    from evals.judges import SEVERITY_RUBRIC_PATH
    from rag.generation import GENERATION_MAX_TOKENS_DEFAULT

    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    gold = load_and_validate_gold(qa_path, charts_path)

    estimator = getattr(harness, "estimate_planned_calls", None)
    assert estimator is not None, (
        "evals.harness.estimate_planned_calls(gold, arm_model=...) is the single "
        "honest source preflight_budget's planned_calls come from (#236)"
    )
    plan = list(estimator(gold, arm_model="claude-haiku-4-5"))

    generation_calls = [call for call in plan if call.get("purpose") == "generation"]
    judge_calls = [call for call in plan if call.get("purpose") == "judge"]

    # Synthetic gold: 3 answerable items (single_passage, multi_passage,
    # severity) -> 3 generation calls, Batches mandatory.
    assert len(generation_calls) == 3
    for call in generation_calls:
        assert call["model"] == "claude-haiku-4-5"
        assert call["output_tokens"] == GENERATION_MAX_TOKENS_DEFAULT
        assert call["mode"] == "batch"

    # Gate-driven judge fan-out (#241): faithfulness + confidence on the 3
    # answered items, severity_fidelity on the 1 severity-annotated item.
    assert len(judge_calls) == 7
    rubric_floor_tokens = len(SEVERITY_RUBRIC_PATH.read_text(encoding="utf-8")) // 4
    severity_calls = [call for call in judge_calls if call.get("kind") == "severity_fidelity"]
    assert len(severity_calls) == 1
    assert severity_calls[0]["input_tokens"] >= rubric_floor_tokens, (
        "the severity judge prompt embeds the rubric verbatim; an estimate below "
        "the rubric's own token floor is dishonest"
    )
    for call in judge_calls:
        assert call["model"] == SONNET_ARM_MODEL  # cross-judge for the Haiku arm
        assert call["mode"] == "batch"

    # The estimate is consumable by (and priced through) the pre-flight.
    ledger_path = _seed_ledger(tmp_path, 0.10)
    preflight = preflight_budget(plan, ledger_path=ledger_path)
    # Hand-computed output-token floor: 3 x 1024 gen tokens at Haiku batch
    # ($5/MTok out * 0.5) + 7 x 512 judge tokens at Sonnet batch
    # ($15/MTok out * 0.5) = $0.00768 + $0.02688.
    assert preflight.estimated_cost_usd >= 0.00768 + 0.02688
    assert preflight.allowed is True


def test_preflight_rechecked_between_arms(tmp_path: Path):
    """The release orchestrator re-reads the ledger before EVERY spend
    segment: when arm 1's run pushes cumulative past the $9.00 cap, the
    judge batch is never submitted and arm 2 refuses with
    BudgetExceededError — a pre-run pre-flight object is never trusted
    as a bearer token (#236)."""
    orchestrator = getattr(harness, "run_release_eval", None)
    assert orchestrator is not None, (
        "evals.harness.run_release_eval is the live orchestration seam "
        "(arms -> pre-flight -> batch submit -> poll -> collect -> gates -> "
        "ledger); without it every live run is hand-written ungated glue (#236)"
    )

    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    gold = load_and_validate_gold(qa_path, charts_path)
    ledger_path = _seed_ledger(tmp_path, 0.10)

    refusal = HonestRefusal(
        refusal_text="The synthetic corpus does not cover this.",
        covered_topics=("synthetic warming",),
        top_score=0.05,
        threshold=0.4,
    )
    crossed: list[bool] = []
    adapters: dict[str, FakeAdapter] = {}

    def deps_factory(arm_model: str) -> AnswerPathDeps:
        adapter = FakeAdapter(
            structured_results=[
                {"scope": "in_scope", "rewritten_query": "synthetic query"} for _ in gold.qa_items
            ]
        )
        adapters[arm_model] = adapter

        def retrieve(_decision):
            if not crossed:
                # Arm 1's first spend lands mid-run: the ledger crosses the cap.
                crossed.append(True)
                ledger.append_row(
                    ledger_path,
                    {
                        "date": "2026-08-23",
                        "session_id": "syn-arm-1",
                        "activity": "release-eval-arm",
                        "issue": "21",
                        "model": "claude-haiku-4-5",
                        "mode": "live",
                        "calls": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 9.50,
                    },
                )
            return refusal

        return AnswerPathDeps(adapter=adapter, retrieve=retrieve)

    batch_client = FakeBatchClient()
    with pytest.raises(BudgetExceededError):
        orchestrator(
            gold,
            arm_models=("claude-haiku-4-5", "claude-sonnet-5"),
            deps_factory=deps_factory,
            plan_chart=lambda request: {"kind": "spec", "spec": {"chart_id": "syn-chart"}},
            batch_client=batch_client,
            mode="live",
            ledger_path=ledger_path,
            journal_dir=tmp_path / "journals",
            session_id="syn-release",
        )

    haiku = adapters.get("claude-haiku-4-5")
    assert haiku is not None and haiku.calls, "arm 1 ran before the cap was crossed"
    sonnet = adapters.get("claude-sonnet-5")
    assert sonnet is None or sonnet.calls == [], "arm 2 must refuse before any adapter call"
    assert batch_client.create_calls == [], (
        "no judge batch may be submitted once the re-read ledger is past the cap"
    )
