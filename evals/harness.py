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

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from evals import gold_selection, ledger, pricing
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
    qa_path = Path(qa_path)
    charts_path = Path(charts_path)
    if not qa_path.is_file():
        raise GoldValidationError(
            f"climate-QA gold file not found at {qa_path}: the harness refuses to "
            "evaluate against absent gold"
        )
    if not charts_path.is_file():
        raise GoldValidationError(
            f"chart-requests gold file not found at {charts_path}: the harness refuses "
            "to evaluate against absent gold"
        )

    try:
        qa_items = list(gold_selection.load_climate_qa_items(qa_path))
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise GoldValidationError(f"climate-QA gold {qa_path} is unreadable: {error}") from error

    # The route vocabulary + subset validation lives in gold_selection (the
    # single selection seam, finding #192) — surface its refusals as gold
    # validation errors.
    try:
        refusal_gate_ids = gold_selection.gate_item_ids(qa_items)
        refusal_calibration_ids = gold_selection.calibration_item_ids(qa_items)
    except gold_selection.GoldSelectionError as error:
        raise GoldValidationError(str(error)) from error

    if len(refusal_gate_ids) < MIN_REFUSAL_GATE_ITEMS:
        raise GoldValidationError(
            f"the gate ∩ retrieval_refusal subset holds {len(refusal_gate_ids)} items; "
            f"the >90% refusal gate needs at least {MIN_REFUSAL_GATE_ITEMS} so a single "
            "flake (19/20) still clears the strict gate (findings #192/#193)"
        )
    overlap = set(refusal_gate_ids) & set(refusal_calibration_ids)
    if overlap:
        raise GoldValidationError(
            f"the gate and calibration refusal subsets overlap on {sorted(overlap)}; "
            "calibrating on a gated item certifies a path production never takes"
        )

    canned_out_of_scope_ids = tuple(
        item["id"]
        for item in qa_items
        if item.get("category") == "no_answer"
        and item.get("expected_route") == gold_selection.CANNED_OUT_OF_SCOPE
    )

    for item in qa_items:
        if item.get("category") == "multi_passage" and "recall_semantics" not in item:
            raise GoldValidationError(
                f"multi_passage gold item {item.get('id')!r} does not declare "
                "recall_semantics (all_gold | any_gold): the harness has no default "
                "(finding #196)"
            )

    chart_items = _load_and_validate_chart_items(charts_path)

    return GoldSets(
        qa_items=tuple(qa_items),
        chart_items=chart_items,
        refusal_gate_ids=refusal_gate_ids,
        refusal_calibration_ids=refusal_calibration_ids,
        canned_out_of_scope_ids=canned_out_of_scope_ids,
    )


def _load_and_validate_chart_items(charts_path: Path) -> tuple[Mapping[str, Any], ...]:
    """Chart gold items, each declaring the behaviour payload its
    ``expected`` field promises (spec item -> ``spec``; refusal item ->
    ``refusal``)."""
    try:
        chart_items = yaml.safe_load(charts_path.read_text(encoding="utf-8"))["items"]
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise GoldValidationError(
            f"chart-requests gold {charts_path} is unreadable: {error}"
        ) from error
    for item in chart_items:
        item_id = item.get("id")
        expected = item.get("expected")
        if expected == "spec" and "spec" not in item:
            raise GoldValidationError(
                f"chart gold item {item_id!r} is expected 'spec' but declares no 'spec' payload"
            )
        if expected == "refusal" and "refusal" not in item:
            raise GoldValidationError(
                f"chart gold item {item_id!r} is expected 'refusal' but declares no "
                "'refusal' payload"
            )
        if expected not in ("spec", "refusal"):
            raise GoldValidationError(
                f"chart gold item {item_id!r} carries expected {expected!r}; "
                "expected one of ('spec', 'refusal')"
            )
    return tuple(chart_items)


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

    #: The ItemResult fields carried as tuples that JSON round-trips as
    #: lists — restored to tuples on load so a resumed result equals a
    #: freshly-computed one.
    _TUPLE_FIELDS = ("citations", "transcript", "retrieved_chunk_ids")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def completed_item_ids(self) -> frozenset[str]:
        return frozenset(result.item_id for result in self.load_results())

    def record(self, result: ItemResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def load_results(self) -> tuple[ItemResult, ...]:
        if not self.path.is_file():
            return ()
        results: list[ItemResult] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            for field_name in self._TUPLE_FIELDS:
                value = data.get(field_name)
                if isinstance(value, list):
                    data[field_name] = tuple(
                        tuple(item) if isinstance(item, list) else item for item in value
                    )
            results.append(ItemResult(**data))
        return tuple(results)


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
    estimated = sum(
        pricing.estimate_cost_usd(
            call["model"],
            input_tokens=int(call.get("input_tokens", 0)),
            output_tokens=int(call.get("output_tokens", 0)),
            mode=call.get("mode", "live"),
            cache_read_tokens=int(call.get("cache_read_tokens", 0)),
            cache_creation_tokens=int(call.get("cache_creation_tokens", 0)),
        )
        for call in planned_calls
    )
    cumulative = ledger.cumulative_usd(Path(ledger_path))
    # Estimate-inclusive and STRICTLY under the cap (RATIFIED): a run that
    # WOULD reach the cap refuses to start — the cap protects real money.
    allowed = (cumulative + estimated) < BUDGET_REFUSAL_THRESHOLD_USD
    return BudgetPreflight(
        estimated_cost_usd=estimated,
        cumulative_usd=cumulative,
        threshold_usd=BUDGET_REFUSAL_THRESHOLD_USD,
        allowed=allowed,
        planned_calls=tuple(dict(call) for call in planned_calls),
    )


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
    if mode not in pricing._MODES:
        # Only live|batch segments spend money; fake/replay cost $0 and the
        # ledger must stay untouched (no row, no file).
        return None
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    cache_read_tokens = int(usage.get("cache_read_tokens", 0))
    cache_creation_tokens = int(usage.get("cache_creation_tokens", 0))
    cost = pricing.estimate_cost_usd(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        mode=mode,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )
    return ledger.append_row(
        Path(ledger_path),
        {
            "date": date.today().isoformat(),
            "session_id": session_id,
            "activity": activity,
            "issue": "21",
            "model": model,
            "mode": mode,
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cost_usd": cost,
            "notes": notes,
        },
    )


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
    if mode not in ADAPTER_MODES:
        raise HarnessError(f"unknown adapter mode {mode!r}; expected one of {ADAPTER_MODES}")
    if mode in LIVE_MODES:
        # Fail closed BEFORE any adapter call (DESIGN §9): live/recording
        # needs the explicit opt-in AND a passing budget pre-flight.
        if preflight is None:
            raise LiveRunRefusedError(
                f"mode {mode!r} touches the network: it requires an explicit "
                "budget pre-flight (evals.harness.preflight_budget) before any "
                "adapter call — refusing"
            )
        if not preflight.allowed:
            raise BudgetExceededError(
                f"mode {mode!r} refused: the pre-flight estimate "
                f"(${preflight.estimated_cost_usd:.4f} on top of "
                f"${preflight.cumulative_usd:.4f}) would cross the "
                f"${preflight.threshold_usd:.2f} cap — no top-up (cost-plan M8)"
            )

    completed_by_id: dict[str, ItemResult] = {}
    if journal is not None:
        completed_by_id = {result.item_id: result for result in journal.load_results()}

    results: list[ItemResult] = []
    for item in qa_items:
        item_id = item["id"]
        if item_id in completed_by_id:
            # Resume: journalled items are returned as-is, ZERO adapter calls.
            results.append(completed_by_id[item_id])
            continue
        result = _drive_answer_item(item, deps, arm_model)
        if journal is not None:
            journal.record(result)
        results.append(result)
    return tuple(results)


def _eval_document_block(passage: Any) -> dict[str, Any]:
    """A minimal citations-enabled generation document block for the eval
    runner: the reranked passage body under an all-or-none citations flag
    (the seam contract), without the live builder's richer provenance the
    synthetic gold does not carry."""
    payload = passage.payload
    text = payload.get("text") or payload.get("body") or ""
    return {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": text}]},
        "citations": {"enabled": True},
    }


def _drive_answer_item(item: Mapping[str, Any], deps: AnswerPathDeps, arm_model: str) -> ItemResult:
    """One gold item through the real classify -> route -> retrieve ->
    cited-generation pipeline, folded into an ItemResult."""
    from rag.generation import GENERATION_MAX_TOKENS_DEFAULT, resolve_citations
    from rag.query import Route, process_query
    from rag.retrieval import HonestRefusal

    item_id = item["id"]
    question = item["question"]
    decision = process_query(deps.adapter, question)
    route = decision.route.value

    if decision.route is Route.CANNED:
        return ItemResult(
            item_id=item_id,
            arm_model=arm_model,
            route=route,
            refused=True,
            answer_text=decision.canned_response,
            transcript=(
                {"role": "user", "content": question},
                {"role": "assistant", "content": decision.canned_response},
            ),
        )

    if decision.route is not Route.RETRIEVAL:
        # CHART requests are the chart path's job, not the answer path's —
        # record the routing visibly without a generation call.
        return ItemResult(
            item_id=item_id,
            arm_model=arm_model,
            route=route,
            transcript=({"role": "user", "content": question},),
        )

    retrieval_result = deps.retrieve(decision)
    if isinstance(retrieval_result, HonestRefusal):
        # §3.5: the refusal path never spends a generation call.
        return ItemResult(
            item_id=item_id,
            arm_model=arm_model,
            route=route,
            refused=True,
            answer_text=retrieval_result.refusal_text,
            transcript=(
                {"role": "user", "content": question},
                {"role": "assistant", "content": retrieval_result.refusal_text},
            ),
        )

    passages = retrieval_result.passages
    answer = deps.adapter.generate(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": decision.retrieval_query or question}],
            }
        ],
        documents=[_eval_document_block(passage) for passage in passages],
        config={"model": arm_model, "max_tokens": GENERATION_MAX_TOKENS_DEFAULT},
    )
    cited = resolve_citations(answer, retrieval_result)
    citations = tuple(
        {
            "cited_text": passage.cited_text,
            "chunk_id": passage.chunk_id,
            "document_index": passage.document_index,
        }
        for passage in cited
    )
    return ItemResult(
        item_id=item_id,
        arm_model=arm_model,
        route=route,
        refused=False,
        answer_text=answer.text,
        citations=citations,
        retrieved_chunk_ids=tuple(passage.chunk_id for passage in passages),
        transcript=(
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.text},
        ),
        usage=dict(answer.usage) if answer.usage else None,
    )


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
    completed_ids: frozenset[str] = journal.completed_item_ids() if journal else frozenset()
    results: list[ChartItemResult] = []
    for item in chart_items:
        item_id = item["id"]
        if item_id in completed_ids:
            continue
        if item.get("blocked_on"):
            # The flagship stays visible as skipped, never dropped, never a pass.
            results.append(
                ChartItemResult(
                    item_id=item_id,
                    outcome="skipped_blocked",
                    blocked_reason=item.get("blocked_reason"),
                )
            )
            continue
        planned = plan_chart(item["request"])
        outcome = planned.get("kind") if isinstance(planned, Mapping) else None
        results.append(
            ChartItemResult(
                item_id=item_id,
                outcome=outcome or "spec",
                planned=dict(planned) if isinstance(planned, Mapping) else None,
            )
        )
    return tuple(results)
