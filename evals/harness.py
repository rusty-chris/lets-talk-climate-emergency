"""Release-eval harness core (issue #21): gold loading, the answer-path
runner, the run journal, and the budget pre-flight.

IMPLEMENTATION.md §4.4 is the constitution here: **evals are not tests**.
The LLM-judge metrics are release gates run against the live model per
release/corpus version; they never run under pytest. What *is* unit-tested
(tests/unit/test_eval_harness_*.py) is this module's deterministic
machinery: gold-set loading refuses malformed inputs, the runner drives
the real pipeline through injectable adapters (Fake/Replay in tests;
Recording/live only via explicit opt-in + a passing budget pre-flight),
runs are resumable via the journal, and every live/recording run is
priced through evals/pricing.py against the $9.00 cap
(evals.ledger.BUDGET_REFUSAL_THRESHOLD_USD) BEFORE it starts.

Red phase: contracts pinned, behaviour raises NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals import gold_selection
from evals.ledger import BUDGET_REFUSAL_THRESHOLD_USD

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIMATE_QA_PATH = gold_selection.CLIMATE_QA_PATH
CHART_REQUESTS_PATH = REPO_ROOT / "evals" / "gold" / "chart_requests.yaml"
SPEND_LEDGER_PATH = REPO_ROOT / "evals" / "spend-ledger.csv"

#: The refusal release gate's minimum n (DESIGN §6.1 as amended, findings
#: #192/#193): the `gate` ∩ `retrieval_refusal` subset must hold at least
#: 20 items so a single flake (19/20 = 95%) survives the strict >90% gate.
MIN_REFUSAL_GATE_ITEMS = 20

#: Adapter modes the runner accepts. `fake`/`replay` are the test-tier
#: modes (deterministic, $0, no ledger row). `recording`/`live` touch the
#: network and are refused without explicit opt-in AND a passing
#: BudgetPreflight (DESIGN §9 fail-closed philosophy).
ADAPTER_MODES = ("fake", "replay", "recording", "live")
OFFLINE_MODES = frozenset({"fake", "replay"})
LIVE_MODES = frozenset({"recording", "live"})


class HarnessError(RuntimeError):
    """Base class for harness refusals — loud, never silent."""


class GoldValidationError(HarnessError):
    """A gold set is missing, malformed, or fails the gate arithmetic —
    the harness refuses to run rather than evaluate against bad gold."""


class LiveRunRefusedError(HarnessError):
    """A live/recording run was requested without the explicit opt-in
    contract (mode + passing budget pre-flight)."""


class BudgetExceededError(LiveRunRefusedError):
    """The pre-flight estimate would take cumulative spend past the
    $9.00 cap — the run refuses to start (cost-plan M8; NO top-up)."""


@dataclass(frozen=True)
class GoldSets:
    """The validated gold sets, plus the route-aware id selections the
    gates consume (selection seam: evals/gold_selection.py)."""

    qa_items: tuple[Mapping[str, Any], ...]
    chart_items: tuple[Mapping[str, Any], ...]
    refusal_gate_ids: tuple[str, ...]
    refusal_calibration_ids: tuple[str, ...]
    canned_out_of_scope_ids: tuple[str, ...]


def load_and_validate_gold(
    qa_path: Path = CLIMATE_QA_PATH,
    charts_path: Path = CHART_REQUESTS_PATH,
) -> GoldSets:
    """Load both gold files and refuse to run on anything malformed.

    Contract (pinned by tests/unit/test_eval_harness_gold.py):

    - a missing gold file raises GoldValidationError naming the path;
    - every no_answer item must carry a known ``expected_route``
      (gold_selection.EXPECTED_ROUTES) and ``subset`` — the #192/#193
      route vocabulary; violations raise GoldValidationError;
    - gate arithmetic: the ``gate`` ∩ ``retrieval_refusal`` subset must
      contain at least MIN_REFUSAL_GATE_ITEMS items, and be disjoint
      from the calibration subset;
    - multi_passage items must declare ``recall_semantics`` (all_gold |
      any_gold) — the harness has NO default (finding #196);
    - chart items must declare exactly one of ``spec`` / ``refusal``
      per their ``expected`` field.
    """
    raise NotImplementedError("issue #21 green phase")


@dataclass(frozen=True)
class ItemResult:
    """One gold item driven through the answer path: the evidence unit
    the judges and gates consume, and the journal's resume record."""

    item_id: str
    arm_model: str
    route: str
    refused: bool = False
    answer_text: str | None = None
    citations: tuple[Mapping[str, Any], ...] = ()
    transcript: tuple[Mapping[str, Any], ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    validation: Mapping[str, Any] | None = None
    usage: Mapping[str, int] | None = None


class RunJournal:
    """Append-only JSONL journal keyed by item id — the resumability
    seam. A crashed or budget-refused run resumes by skipping every
    item already journalled (zero adapter calls for skipped items)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def completed_item_ids(self) -> frozenset[str]:
        raise NotImplementedError("issue #21 green phase")

    def record(self, result: ItemResult) -> None:
        raise NotImplementedError("issue #21 green phase")

    def load_results(self) -> tuple[ItemResult, ...]:
        raise NotImplementedError("issue #21 green phase")


@dataclass(frozen=True)
class AnswerPathDeps:
    """The injectable seams the runner drives the REAL pipeline through
    (mirrors service.app.ServiceDeps shape-for-shape; IMPLEMENTATION §1).

    ``adapter`` is the ProviderAdapter (Fake/Replay in tests); classify
    and generation go through it. ``retrieve`` is
    ``QueryDecision -> RetrievedPassages | HonestRefusal``.
    ``validate_exchange`` is the #13 citation-support seam producing a
    ValidationOutcome-shaped record per exchange.
    """

    adapter: Any
    retrieve: Callable[[Any], Any]
    validate_exchange: Callable[[Any, Sequence[Mapping[str, Any]]], Any] | None = None


@dataclass(frozen=True)
class BudgetPreflight:
    """The M8 pre-flight verdict: computed BEFORE any live/recording
    run, from evals/pricing.py estimates + the committed ledger."""

    estimated_cost_usd: float
    cumulative_usd: float
    threshold_usd: float = BUDGET_REFUSAL_THRESHOLD_USD
    allowed: bool = False
    planned_calls: tuple[Mapping[str, Any], ...] = field(default=())


def preflight_budget(
    planned_calls: Sequence[Mapping[str, Any]],
    *,
    ledger_path: Path = SPEND_LEDGER_PATH,
) -> BudgetPreflight:
    """Price a planned run against the $9.00 cap before it starts.

    ``planned_calls``: mappings with ``model``, ``input_tokens``,
    ``output_tokens`` and ``mode`` (live|batch), priced through
    evals.pricing.estimate_cost_usd — never any other table. The verdict
    is ``allowed=False`` whenever ledger cumulative + estimate exceeds
    ``threshold_usd`` (estimate-inclusive: a run that WOULD cross the
    cap refuses to start, not just one starting past it).
    """
    raise NotImplementedError("issue #21 green phase")


def record_run_spend(
    ledger_path: Path,
    *,
    mode: str,
    model: str,
    activity: str,
    usage: Mapping[str, int],
    calls: int,
    session_id: str,
    notes: str = "",
) -> Mapping[str, Any] | None:
    """Ledger discipline for one completed run segment.

    live/recording (mode live|batch) → appends an M8 row via
    evals.ledger.append_row, priced through evals/pricing.py, and
    returns the row. fake/replay runs cost $0 and MUST NOT touch the
    ledger — returns None with the ledger file unchanged.
    """
    raise NotImplementedError("issue #21 green phase")


def run_answer_path(
    qa_items: Iterable[Mapping[str, Any]],
    deps: AnswerPathDeps,
    *,
    arm_model: str,
    mode: str = "replay",
    journal: RunJournal | None = None,
    preflight: BudgetPreflight | None = None,
) -> tuple[ItemResult, ...]:
    """Drive gold items through the real answer pipeline, one item →
    one ItemResult (transcript + citations + route classification).

    Contract (pinned by tests/unit/test_eval_harness_runner.py):

    - classification goes through ``deps.adapter.structured`` and cited
      generation through ``deps.adapter.generate`` — the runner drives
      the pipeline's provider seam, never a parallel reimplementation;
    - deterministic: identical inputs (adapter programme, gold items)
      produce identical results run-to-run;
    - resumable: items already in ``journal`` are skipped with ZERO
      adapter calls; fresh results are journalled as they complete;
    - modes ``recording``/``live`` raise LiveRunRefusedError without a
      pre-flight, and BudgetExceededError on a failing one — BEFORE any
      adapter call. ``fake``/``replay`` need neither and never touch
      the ledger. Unknown modes refuse loudly.
    """
    raise NotImplementedError("issue #21 green phase")


@dataclass(frozen=True)
class ChartItemResult:
    """One chart gold item through the planner path: the planned spec
    (or refusal payload), plus the skipped-visibly record for items
    blocked on #23/#117 (never silently absent)."""

    item_id: str
    outcome: str  # "spec" | "refusal" | "skipped_blocked"
    planned: Mapping[str, Any] | None = None
    blocked_reason: str | None = None


def run_chart_path(
    chart_items: Iterable[Mapping[str, Any]],
    plan_chart: Callable[[str], Any],
    *,
    journal: RunJournal | None = None,
) -> tuple[ChartItemResult, ...]:
    """Drive the chart gold set through the injected planner seam.

    Items carrying ``blocked_on`` (the flagship, blocked on #23/#117)
    are returned as ``skipped_blocked`` with their reason — visible in
    every report, never dropped. Deterministic and resumable like
    run_answer_path.
    """
    raise NotImplementedError("issue #21 green phase")
