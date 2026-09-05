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
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from evals import gold_selection, ledger, pricing
from evals.ledger import BUDGET_REFUSAL_THRESHOLD_USD

#: Gold categories whose route legitimately includes the voices layer —
#: the voices-separation gate exempts these (finding #243).
VOICES_CATEGORIES = frozenset({"voices_action"})

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

#: The modes record_run_spend will price + append a ledger row for
#: (finding #238): the live/recording adapter modes plus the ``batch``
#: pricing mode a Batches segment records under. Everything outside this
#: set and OFFLINE_MODES is a typo and refuses loudly.
LEDGERED_SPEND_MODES = frozenset({"batch", "live", "recording"})


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
    #: The generation call's document set as {chunk_id, source_type}
    #: mappings (finding #243): the evidence the DESIGN §6.2 voices-
    #: separation gate is computed from — captured from the passages
    #: actually sent to generate, never reconstructed.
    documents: tuple[Mapping[str, Any], ...] = ()
    validation: Mapping[str, Any] | None = None
    usage: Mapping[str, int] | None = None
    #: Review #316: the delivered #12 SSE transcript (text/citation/usage/
    #: footer events) — journalled so a citation failure is attributable
    #: from artifacts offline, and the batched-verdict collector fix (#309)
    #: has a transcript to re-score against.
    sse_transcript: tuple[Mapping[str, Any], ...] = ()
    #: Review #317: a max_tokens truncation is a deterministic, scoreable
    #: FAILURE (refused=False, truncated=True) — journalled with its real
    #: usage so it never re-runs into the identical wall on resume.
    truncated: bool = False


class RunJournal:
    """Append-only JSONL journal keyed by item id — the resumability
    seam. A crashed or budget-refused run resumes by skipping every
    item already journalled (zero adapter calls for skipped items).

    Records both answer-path (:class:`ItemResult`) and chart-path
    (:class:`ChartItemResult`) results, discriminated by a ``_record_type``
    tag so a single journal round-trips either kind (findings #237/#243).
    """

    #: The tuple-carried fields that JSON round-trips as lists — restored
    #: to tuples on load so a resumed result equals a freshly-computed one.
    _TUPLE_FIELDS = (
        "citations",
        "transcript",
        "retrieved_chunk_ids",
        "documents",
        "sse_transcript",
    )

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def completed_item_ids(self) -> frozenset[str]:
        return frozenset(result.item_id for result in self.load_results())

    def record(self, result: ItemResult | ChartItemResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(result)
        data["_record_type"] = type(result).__name__
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_results(self) -> tuple[ItemResult | ChartItemResult, ...]:
        if not self.path.is_file():
            return ()
        lines = self.path.read_text(encoding="utf-8").split("\n")
        entries = [(number, line) for number, line in enumerate(lines, start=1) if line.strip()]
        results: list[ItemResult | ChartItemResult] = []
        for position, (line_number, line) in enumerate(entries):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                if position == len(entries) - 1:
                    # A run killed mid-record leaves a truncated FINAL line:
                    # drop it (the item safely re-runs) but surface it loudly
                    # (finding #246 — tolerate the tail, never a raw crash).
                    warnings.warn(
                        f"journal {self.path} has a truncated final line "
                        f"(line {line_number}); dropping it — the item re-runs",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                raise HarnessError(
                    f"journal {self.path} is corrupt at line {line_number}: {error} — "
                    "refusing to silently drop an interior (paid) record (finding #246)"
                ) from error
            results.append(self._record_from_data(data, line_number))
        return tuple(results)

    @staticmethod
    def _record_from_data(data: dict[str, Any], line_number: int) -> ItemResult | ChartItemResult:
        record_type = data.pop("_record_type", ItemResult.__name__)
        cls = ChartItemResult if record_type == ChartItemResult.__name__ else ItemResult
        for field_name in RunJournal._TUPLE_FIELDS:
            value = data.get(field_name)
            if isinstance(value, list):
                data[field_name] = tuple(
                    tuple(item) if isinstance(item, list) else item for item in value
                )
        known = {field_info.name for field_info in fields(cls)}
        unknown = set(data) - known
        if unknown:
            # Schema drift (finding #246): a field this dataclass does not
            # know is a loud HarnessError naming the field, never a bare
            # TypeError from the constructor.
            raise HarnessError(
                f"journal line {line_number} carries unknown {cls.__name__} field(s) "
                f"{sorted(unknown)}: schema drift — refusing (finding #246)"
            )
        return cls(**data)


def _judge_verdict_to_dict(verdict: Any) -> dict[str, Any]:
    """A JudgeVerdict as a JSON-serialisable mapping (review #316)."""
    return {
        "custom_id": verdict.custom_id,
        "kind": verdict.kind,
        "item_id": verdict.item_id,
        "scored": bool(verdict.scored),
        "verdict": dict(verdict.verdict) if verdict.verdict is not None else None,
        "failure_reason": verdict.failure_reason,
        "usage": dict(verdict.usage) if verdict.usage is not None else None,
    }


def _judge_verdict_from_dict(data: Mapping[str, Any]) -> Any:
    """Reconstruct a JudgeVerdict from a journalled mapping (review #316)."""
    from evals.judges import JudgeVerdict

    return JudgeVerdict(
        custom_id=data["custom_id"],
        kind=data["kind"],
        item_id=data["item_id"],
        scored=bool(data["scored"]),
        verdict=data.get("verdict"),
        failure_reason=data.get("failure_reason"),
        usage=data.get("usage"),
    )


class JudgesJournal:
    """Append-only JSONL journal for the release run's judge batches
    (review #316) — the seam that makes a resumed run REUSE an
    already-paid batch instead of re-submitting it (the live run lost
    $0.34, ~20% of its whole $1.71 spend, to exactly this gap).

    Two record kinds per arm, both keyed by ``arm_model``:

    - a ``submission`` record (batch id + the submitted custom_id set),
      written BEFORE the first poll so a crash while polling never orphans
      the paid batch — a resume collects it by id via ``results()`` (free
      for 29 days);
    - a ``collection`` record (the folded per-request verdicts), written
      after collection so a fully-resumed run reconstructs the verdicts
      with ZERO batch calls, leaving the ledger unchanged.

    Corruption follows the #246 conventions RunJournal pins: a truncated
    final line warns and the run completes; interior corruption raises
    HarnessError naming the line (never a silent drop of paid verdicts).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_submission(self, arm_model: str, batch_id: str, custom_ids: Sequence[str]) -> None:
        self._append(
            {
                "_kind": "submission",
                "arm_model": arm_model,
                "batch_id": batch_id,
                "custom_ids": list(custom_ids),
            }
        )

    def record_collection(self, arm_model: str, batch_id: str, verdicts: Mapping[str, Any]) -> None:
        self._append(
            {
                "_kind": "collection",
                "arm_model": arm_model,
                "batch_id": batch_id,
                "verdicts": [_judge_verdict_to_dict(verdict) for verdict in verdicts.values()],
            }
        )

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").split("\n")
        entries = [(number, line) for number, line in enumerate(lines, start=1) if line.strip()]
        records: list[dict[str, Any]] = []
        for position, (line_number, line) in enumerate(entries):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                if position == len(entries) - 1:
                    # A run killed mid-write leaves a truncated FINAL line:
                    # drop it (the batch is re-collected by id) but surface it
                    # loudly (finding #246 — tolerate the tail, never a raw crash).
                    warnings.warn(
                        f"judges journal {self.path} has a truncated final line "
                        f"(line {line_number}); dropping it",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                raise HarnessError(
                    f"judges journal {self.path} is corrupt at line {line_number}: {error} — "
                    "refusing to silently drop an interior (paid) record (finding #246)"
                ) from error
        return records

    def state_for_arm(self, arm_model: str) -> tuple[str | None, dict[str, Any] | None]:
        """The (batch_id, collected verdicts) known for an arm — later
        records win. ``verdicts`` is None when only a submission was
        journalled (kill-after-submit); the batch is then re-collected by
        id, never re-created."""
        batch_id: str | None = None
        verdicts: dict[str, Any] | None = None
        for record in self._load():
            if record.get("arm_model") != arm_model:
                continue
            if record.get("batch_id"):
                batch_id = record["batch_id"]
            if record.get("_kind") == "collection":
                verdicts = {
                    entry["custom_id"]: _judge_verdict_from_dict(entry)
                    for entry in record.get("verdicts", [])
                }
        return batch_id, verdicts


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

    live/recording runs spend real money and append an M8 row via
    evals.ledger.append_row, priced through evals/pricing.py, and
    return the row. fake/replay runs cost $0 and MUST NOT touch the
    ledger — return None with the ledger file unchanged. Any mode outside
    the closed adapter vocabulary raises loudly (finding #238): a typo can
    never silently under-count the cap.
    """
    if mode in OFFLINE_MODES:
        # fake/replay cost $0 and the ledger must stay untouched (no row,
        # no file) — the explicit skip-list, not an accident of membership.
        return None
    if mode not in LEDGERED_SPEND_MODES:
        raise ValueError(
            f"record_run_spend: unknown mode {mode!r}; expected one of "
            f"{sorted(LEDGERED_SPEND_MODES)} (spend) or {sorted(OFFLINE_MODES)} (skip) — "
            "a silently unledgered spend erodes the $9.00 cap (finding #238)"
        )
    # `recording` spends real tokens at live (non-batch) rates; map it onto
    # the pricing vocabulary (evals.pricing.MODES = live|batch).
    pricing_mode = "live" if mode == "recording" else mode
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    cache_read_tokens = int(usage.get("cache_read_tokens", 0))
    cache_creation_tokens = int(usage.get("cache_creation_tokens", 0))
    cost = pricing.estimate_cost_usd(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        mode=pricing_mode,
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
            "mode": pricing_mode,
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
      generation through ``deps.adapter.generate_stream`` on the
      production-built request — the runner drives the pipeline's
      provider seam (the STREAMED production path, so validation is fed
      the true SSE transcript per the #303 ratification note), never a
      parallel reimplementation;
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
        for result in journal.load_results():
            # #235: a journal recorded under a different arm must never be
            # returned as this arm's answers (judge == generator, corrupted
            # bake-off). Fail closed, naming both ids, BEFORE any adapter
            # call — the cheapest possible mistake, caught loudly.
            if isinstance(result, ItemResult) and result.arm_model != arm_model:
                raise HarnessError(
                    f"journal {journal.path} was recorded under arm_model "
                    f"{result.arm_model!r} but this run is arm_model {arm_model!r}: a shared "
                    "journal must never return one arm's answers as another's (finding #235)"
                )
            completed_by_id[result.item_id] = result

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


def eval_no_budget_guard(model: str) -> None:
    """The eval's explicit, NAMED no-op budget guard for non-default
    bake-off arms (ADR-015 / finding #234): the release-eval tier
    genuinely wants no budget behaviour, and passes a guard whose name
    says so rather than bypassing the production builder's best-mode
    policy. Never refuses; the $9.00 cap is enforced by the pre-flight."""
    return None


#: The corpus vintage stamped into the eval GroundedAnswer's §3.5 footer.
#: The eval harness runs against synthetic/replayed corpora, so the footer
#: vintage is not a measured field here (segmentation never reads the
#: footer event); a live release run stamps the real manifest vintage.
_EVAL_CORPUS_VINTAGE = "the evaluated corpus snapshot"


def _assert_transcript_complete(item_id: str, sse_transcript: Sequence[Mapping[str, Any]]) -> None:
    """Refuse an error-terminated delivery loudly (#303 ratification note
    6): production never validates a transcript carrying an ``error``
    event (truncated/failed answers are not delivered answers), so the
    eval must never score one either — the item is NOT journalled and
    safely re-runs on resume."""
    for event in sse_transcript:
        if event.get("event") == "error":
            error_type = (event.get("data") or {}).get("type")
            raise HarnessError(
                f"item {item_id!r}: the generation stream terminated with error "
                f"{error_type!r} — an incomplete delivery is not scoreable evidence "
                "(#303 transcript-fidelity note); the item re-runs on resume"
            )


def _validation_record(outcome: Any) -> dict[str, Any]:
    """Derive the ItemResult.validation record the citation_support gate
    consumes from a #13 ValidationOutcome: the segmented factual-sentence
    count (the pooled denominator — preserved even when degraded, finding
    #239) and the count of those sentences an entailment verdict supported
    (zero on a degraded/unvalidated outcome — fail-closed).

    Review #316: the validator's per-pair verdicts ride the record too
    ({pair_index, sentence_index, document_index, supported}) — enough,
    with the journalled SSE transcript, to recompute {supported, factual}
    offline and to attribute a citation failure from artifacts alone."""
    factual_sentences = [sentence for sentence in outcome.sentences if sentence.factual]
    factual = len(factual_sentences)
    if not outcome.validated:
        return {
            "validated": False,
            "supported": 0,
            "factual": factual,
            "degraded_reason": outcome.degraded_reason,
        }
    supported_sentences = {
        verdict.sentence_index for verdict in outcome.verdicts if verdict.supported
    }
    supported = sum(1 for sentence in factual_sentences if sentence.index in supported_sentences)
    return {
        "validated": True,
        "supported": supported,
        "factual": factual,
        "verdicts": [
            {
                "pair_index": verdict.pair_index,
                "sentence_index": verdict.sentence_index,
                "document_index": verdict.document_index,
                "supported": verdict.supported,
            }
            for verdict in outcome.verdicts
        ],
    }


#: Review #317: the SSE ``error`` type a max_tokens truncation surfaces
#: (rag.generation.answer_stream_to_sse). It is transport-complete and
#: deterministic — distinct from a genuine transport error, which stays a
#: non-scoreable HarnessError (#303).
_TRUNCATED_ERROR_TYPE = "truncated"


def _truncation_error(sse_transcript: Sequence[Mapping[str, Any]]) -> bool:
    """True when the delivered transcript ends in a max_tokens truncation."""
    return any(
        event.get("event") == "error"
        and (event.get("data") or {}).get("type") == _TRUNCATED_ERROR_TYPE
        for event in sse_transcript
    )


def _degraded_truncation_validation(sse_transcript: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The fail-closed validation record for a truncated delivery (#317):
    the delivered factual sentences pool with ZERO supported (#239's
    degraded arithmetic), never validated, reason naming the truncation."""
    from rag.citation_validator import segment_answer_sentences

    sentences = segment_answer_sentences(sse_transcript)
    factual = sum(1 for sentence in sentences if sentence.factual)
    return {
        "validated": False,
        "supported": 0,
        "factual": factual,
        "degraded_reason": "generation truncated at max_tokens (output budget) — fail-closed",
    }


def _drive_answer_item(item: Mapping[str, Any], deps: AnswerPathDeps, arm_model: str) -> ItemResult:
    """One gold item through the real classify -> route -> retrieve ->
    cited-generation pipeline, folded into an ItemResult."""
    from rag.generation import (
        GENERATION_MAX_TOKENS_DEFAULT,
        GENERATION_MODEL_DEFAULT,
        GenerationConfig,
        answer_stream_to_sse,
        build_generation_request,
        resolve_citations,
    )
    from rag.provider import accumulate_answer_from_stream_events
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
    # #234: the eval generation request IS the production builder's output —
    # build_generation_request field-for-field (committed system prompt on
    # the system channel, the ORIGINAL question, titled document blocks
    # with source_type/consensus_position context, the §3.4 ≤8-doc bound,
    # citations enabled). Non-default arms go THROUGH the best-mode policy
    # with the named no-op guard, never around the builder.
    is_non_default = not (
        arm_model == GENERATION_MODEL_DEFAULT
        or arm_model.startswith(GENERATION_MODEL_DEFAULT + "-")
    )
    config = GenerationConfig(
        model=arm_model,
        max_tokens=GENERATION_MAX_TOKENS_DEFAULT,
        best_mode_enabled=is_non_default,
        budget_guard=eval_no_budget_guard if is_non_default else None,
    )
    request = build_generation_request(retrieval_result, question, config=config)
    # #303 ratification note 6: generation drives the STREAMED production
    # seam so validation is fed the TRUE SSE transcript — the same
    # answer_stream_to_sse translation production delivers (text/citation
    # events in transport arrival order, usage, footer) — never a flat
    # reconstruction that hangs every citation on the final sentence.
    transport_events = list(deps.adapter.generate_stream(**request))
    sse_transcript = tuple(
        answer_stream_to_sse(
            iter(transport_events),
            retrieved=retrieval_result,
            corpus_vintage=_EVAL_CORPUS_VINTAGE,
        )
    )
    # The folded answer/usage view is the transport-side production twin
    # (one event vocabulary, no drift): message_start input usage merged
    # with message_delta output usage for the ledger row. It is computed
    # before the transcript-fidelity gate so a truncation's REAL usage (in
    # message_delta, which arrives before the terminal error event) is
    # captured for the ledger, never estimated (#317).
    answer = accumulate_answer_from_stream_events(transport_events)
    documents = tuple(
        {"chunk_id": passage.chunk_id, "source_type": passage.payload.get("source_type")}
        for passage in passages
    )
    retrieved_chunk_ids = tuple(passage.chunk_id for passage in passages)

    if _truncation_error(sse_transcript):
        # #317: a max_tokens truncation is DETERMINISTIC — re-running it is
        # pure double spend. Journal it as a scoreable FAILED item
        # (refused=False, truncated=True) with its real usage, instead of
        # the unjournalled HarnessError that re-ran qa-sp-06 into the
        # identical wall on every live resume. Validation degrades
        # fail-closed WITHOUT an entailment call (a truncated delivery is
        # never validated).
        truncated_validation = (
            _degraded_truncation_validation(sse_transcript)
            if deps.validate_exchange is not None
            else None
        )
        return ItemResult(
            item_id=item_id,
            arm_model=arm_model,
            route=route,
            refused=False,
            truncated=True,
            answer_text=answer.text,
            validation=truncated_validation,
            retrieved_chunk_ids=retrieved_chunk_ids,
            documents=documents,
            transcript=(
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer.text},
            ),
            sse_transcript=sse_transcript,
            usage=dict(answer.usage) if answer.usage else None,
        )
    # Any OTHER error-terminated delivery is a genuine, non-deterministic
    # failure: it stays a non-scoreable HarnessError, NOT journalled (#303).
    _assert_transcript_complete(item_id, sse_transcript)
    cited = resolve_citations(answer, retrieval_result)
    citations = tuple(
        {
            "cited_text": passage.cited_text,
            "chunk_id": passage.chunk_id,
            "document_index": passage.document_index,
        }
        for passage in cited
    )

    # #303: drive the citation-support validation seam for the answered
    # exchange exactly like production (service.app), feeding the #13
    # validator production's GroundedAnswer reassembly of the delivered
    # transcript plus that SAME transcript, and record the derived
    # {validated, supported, factual} on the ItemResult — the ONLY honest
    # feed for the citation_support release gate. Refusals are never
    # validated (they return early above).
    validation: dict[str, Any] | None = None
    if deps.validate_exchange is not None:
        # Production's own reassembly (service.app._grounded_answer_from_sse)
        # — imported, never re-implemented, so the eval validates exactly
        # the artefact production validates.
        from service.app import _grounded_answer_from_sse

        grounded = _grounded_answer_from_sse(sse_transcript, retrieval_result)
        outcome = deps.validate_exchange(grounded, sse_transcript)
        validation = _validation_record(outcome)
        # Review #312: an answered (non-refused), ZERO-citation exchange on
        # a no_answer gold item is a generation-level honest decline — its
        # passage-meta/referral sentences can never be entailed by a corpus
        # chunk, and the refusal gate already fails the item, so the gate
        # excludes it from the citation pool (never double-counted). An
        # uncited answer on an ANSWERABLE item is NOT a decline: it stays
        # pooled fail-closed. Structured decline signals may STRENGTHEN this
        # later, never replace it (ratified minimum contract).
        if item.get("category") == "no_answer" and not citations:
            validation["generation_decline"] = True

    return ItemResult(
        item_id=item_id,
        arm_model=arm_model,
        route=route,
        refused=False,
        answer_text=answer.text,
        citations=citations,
        validation=validation,
        retrieved_chunk_ids=retrieved_chunk_ids,
        # #243: the generation document set's per-document source_type —
        # the voices-separation gate's evidence, captured from the passages
        # actually sent to generate.
        documents=documents,
        transcript=(
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.text},
        ),
        sse_transcript=sse_transcript,
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
    run_answer_path: journalled items are returned as-is with ZERO
    planner calls, and fresh results are journalled as they complete
    (finding #237 — resume must not re-pay every planner call, and must
    not starve the gate denominator by dropping completed items).
    """
    completed_by_id: dict[str, ChartItemResult] = {}
    if journal is not None:
        completed_by_id = {
            result.item_id: result
            for result in journal.load_results()
            if isinstance(result, ChartItemResult)
        }
    results: list[ChartItemResult] = []
    for item in chart_items:
        item_id = item["id"]
        if item_id in completed_by_id:
            # Resume: journalled charts are returned as-is, ZERO planner calls.
            results.append(completed_by_id[item_id])
            continue
        if item.get("blocked_on"):
            # The flagship stays visible as skipped, never dropped, never a pass.
            result = ChartItemResult(
                item_id=item_id,
                outcome="skipped_blocked",
                blocked_reason=item.get("blocked_reason"),
            )
        else:
            planned = plan_chart(item["request"])
            outcome = planned.get("kind") if isinstance(planned, Mapping) else None
            result = ChartItemResult(
                item_id=item_id,
                outcome=outcome or "spec",
                planned=dict(planned) if isinstance(planned, Mapping) else None,
            )
        if journal is not None:
            journal.record(result)
        results.append(result)
    return tuple(results)


# ---------------------------------------------------------------------------
# Gate-input mapping helpers (finding #243) + the deterministic metrics the
# release orchestrator wires from the run records.
# ---------------------------------------------------------------------------


def voices_gate_input(
    item_results: Sequence[ItemResult],
    gold_items: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map ItemResults onto the shape ``voices_separation_violations``
    consumes (finding #243): each run carries its generation document set
    and a ``route_is_voices`` flag derived from the gold item's category
    (voices_action). A voices chunk in a non-voices item's document set is
    a violation; the same documents on a voices item are exempt."""
    category_by_id = {item["id"]: item.get("category") for item in gold_items}
    return [
        {
            "item_id": result.item_id,
            "route_is_voices": category_by_id.get(result.item_id) in VOICES_CATEGORIES,
            "documents": [dict(document) for document in result.documents],
        }
        for result in item_results
    ]


def chart_gate_records(
    gold_chart_items: Sequence[Mapping[str, Any]],
    chart_results: Sequence[ChartItemResult],
) -> dict[str, list[dict[str, Any]]]:
    """Derive the chart gate inputs by COMPARING each planner outcome to
    its gold item (finding #242 — never a hardwired 'match'):

    - spec-expected items: ``match`` iff the planned spec equals the gold
      spec, else ``mismatch``; blocked flagships are ``skipped_blocked``;
    - refusal-expected items: ``refused_with_nearest`` iff the planner
      refused, else ``mismatch``.
    """
    results_by_id = {result.item_id: result for result in chart_results}
    spec_records: list[dict[str, Any]] = []
    refusal_records: list[dict[str, Any]] = []
    for gold_item in gold_chart_items:
        item_id = gold_item["id"]
        result = results_by_id.get(item_id)
        if result is None:
            continue
        if result.outcome == "skipped_blocked":
            spec_records.append(
                {
                    "item_id": item_id,
                    "status": "skipped_blocked",
                    "blocked_reason": result.blocked_reason,
                }
            )
            continue
        planned = result.planned or {}
        if gold_item.get("expected") == "refusal":
            status = "refused_with_nearest" if planned.get("kind") == "refusal" else "mismatch"
            refusal_records.append({"item_id": item_id, "status": status})
        else:
            status = "match" if planned.get("spec") == gold_item.get("spec") else "mismatch"
            spec_records.append({"item_id": item_id, "status": status})
    return {"spec": spec_records, "refusal": refusal_records}


#: The calibrated IPCC confidence vocabulary the proxy metric looks for
#: (DESIGN §6.1/§6.2). Multi-word terms are matched as adjacent runs.
_CALIBRATED_VOCABULARY = (
    "very likely",
    "very unlikely",
    "likely",
    "unlikely",
    "high confidence",
    "medium confidence",
    "low confidence",
    "virtually certain",
)


def compute_retrieval_metrics(
    answer_results: Sequence[ItemResult],
    gold_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Arm-level recall@8 / MRR / nDCG@8 over the run's retrieved chunk
    ids vs each item's gold_chunk_ids (finding #242: these implemented
    metrics must actually be invoked by a runner). Items without gold
    chunk ids (refusals, charts) are excluded from the average."""
    from evals.metrics import mrr, ndcg_at_k, recall_at_k

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for result in answer_results:
        gold_item = gold_by_id.get(result.item_id, {})
        gold_ids = gold_item.get("gold_chunk_ids")
        if not gold_ids:
            continue
        semantics = gold_item.get("recall_semantics") or (
            "all_gold" if gold_item.get("category") == "multi_passage" else "any_gold"
        )
        recalls.append(
            1.0
            if recall_at_k(result.retrieved_chunk_ids, gold_ids, k=8, semantics=semantics)
            else 0.0
        )
        reciprocal_ranks.append(mrr(result.retrieved_chunk_ids, gold_ids))
        ndcgs.append(ndcg_at_k(result.retrieved_chunk_ids, gold_ids, k=8))

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "recall_at_8": _mean(recalls),
        "mrr": _mean(reciprocal_ranks),
        "ndcg_at_8": _mean(ndcgs),
    }


def compute_calibrated_term_rate(answer_results: Sequence[ItemResult]) -> float:
    """The fraction of answered items whose answer carries at least one
    calibrated-confidence term (finding #242/#244: the fixed proxy metric
    is invoked and reported)."""
    from evals.metrics import calibrated_term_preserved

    answered = [
        result
        for result in answer_results
        if not result.refused and (result.answer_text or "").strip()
    ]
    if not answered:
        return 0.0
    carrying = sum(
        1
        for result in answered
        if any(
            calibrated_term_preserved(result.answer_text, result.answer_text, term)
            for term in _CALIBRATED_VOCABULARY
        )
    )
    return carrying / len(answered)


# ---------------------------------------------------------------------------
# The planned-calls estimator + the live release orchestrator (finding #236).
# ---------------------------------------------------------------------------


def _predicted_item_result(item: Mapping[str, Any]) -> Any:
    """A predicted ItemResult stand-in for the estimator: no_answer items
    refuse, everything else answers — enough for the gate-driven judge
    fan-out to size the batch honestly before a single token is spent."""
    from types import SimpleNamespace

    is_refusal = item.get("category") == "no_answer"
    return SimpleNamespace(
        item_id=item["id"],
        refused=is_refusal,
        answer_text="" if is_refusal else "predicted answer under evaluation",
    )


def estimate_planned_calls(gold: GoldSets, *, arm_model: str) -> list[dict[str, Any]]:
    """The single honest source of ``preflight_budget``'s planned_calls
    (finding #236): one batched generation entry per answerable gold item
    (output GENERATION_MAX_TOKENS_DEFAULT), plus one batched judge entry
    per gate-driven judge request (finding #241), each input estimate
    derived from the ACTUAL built prompt (so the severity judge's estimate
    covers the rubric it embeds). Never caller-invented numbers."""
    from evals.judges import _JUDGE_MAX_TOKENS, build_judge_requests
    from rag.generation import (
        GENERATION_MAX_TOKENS_DEFAULT,
        estimate_tokens_lower_bound,
        load_system_prompt,
    )

    plan: list[dict[str, Any]] = []
    system_tokens = estimate_tokens_lower_bound(load_system_prompt())
    answerable = [item for item in gold.qa_items if item.get("category") != "no_answer"]
    for item in answerable:
        plan.append(
            {
                "purpose": "generation",
                "model": arm_model,
                "input_tokens": system_tokens + len(str(item.get("question", ""))) // 4,
                "output_tokens": GENERATION_MAX_TOKENS_DEFAULT,
                "mode": "batch",
            }
        )
    gold_by_id = {item["id"]: item for item in gold.qa_items}
    predicted = [_predicted_item_result(item) for item in gold.qa_items]
    for request in build_judge_requests(predicted, gold_by_id, arm_model=arm_model):
        plan.append(
            {
                "purpose": "judge",
                "kind": request.kind,
                "model": request.judge_model,
                "input_tokens": len(request.prompt) // 4,
                "output_tokens": _JUDGE_MAX_TOKENS,
                "mode": "batch",
            }
        )
    return plan


def _aggregate_usage(results: Sequence[ItemResult]) -> dict[str, int]:
    """Sum the token usage across an arm's ItemResults for the ledger row."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    for result in results:
        for key in totals:
            totals[key] += int((result.usage or {}).get(key, 0))
    return totals


def _route_accuracy_gate(classifier_summary: Mapping[str, Any] | None) -> Any:
    """route_accuracy fed from the classifier-accuracy summary
    (evals/scripts/classifier_accuracy.py's ``release_gate_passes``);
    an ABSENT summary is BLOCKED — never silently dropped from the
    battery (issue #303, fail-closed like the pending owner audit): the
    classifier gate cannot vanish from the verdict just because nobody
    ran the accuracy eval."""
    from evals import gates

    if classifier_summary is None:
        return gates.GateResult(
            name="route_accuracy",
            status=gates.GATE_BLOCKED,
            reason=(
                "classifier-accuracy summary absent for this run: the route-accuracy "
                "release gate cannot be measured (#303)"
            ),
        )
    return gates.route_accuracy_gate(classifier_summary)


def _citation_support_gate(answer_results: Sequence[ItemResult]) -> Any:
    """citation_support fed from the #13 validator outcomes carried on
    ItemResult.validation. A run where validation NEVER executed (no
    answered exchange carries a validation record) is BLOCKED — never
    passed, never silently absent: the product's core guarantee must
    block release exactly like the pending owner audit when unmeasured
    (issue #303)."""
    from evals import gates

    records = [
        {"item_id": result.item_id, **dict(result.validation)}
        for result in answer_results
        if result.validation is not None
    ]
    if not records:
        return gates.GateResult(
            name="citation_support",
            status=gates.GATE_BLOCKED,
            reason="citation-support validation never executed for this run (#303)",
        )
    return gates.citation_support_gate(records)


def _severity_gate(severity_records: Sequence[Mapping[str, Any]] | None) -> Any:
    """severity fed from judged-severity records ({item_id, expected,
    judged, scored}). An ABSENT feed is BLOCKED — never scored as an
    empty 0/0 failure and never silently dropped (fail-closed, parallel
    to route_accuracy and citation_support): with the owner audit
    complete (2026-09-04) an unmeasured severity gate must block release,
    not fail it on vacuous arithmetic. While the audit packet says
    pending, the gate's own owner-guard (finding #197) takes precedence
    and its pending reason is reported. The release orchestrator feeds
    ``severity_records_from_verdicts`` over the collected judge batch;
    the offline suite feeds its simulated exact-match records."""
    from evals import gates, severity_audit

    if severity_records is None:
        try:
            severity_audit.assert_owner_severity_audit_complete()
        except severity_audit.SeverityAuditPendingError:
            # BLOCKED with the finding-#197 pending reason.
            return gates.severity_gate([])
        return gates.GateResult(
            name="severity",
            status=gates.GATE_BLOCKED,
            reason=(
                "no judged severity records supplied for this run: the severity "
                "release gate cannot be measured — blocked, never an empty "
                "pass/fail (fail-closed like route_accuracy/citation_support)"
            ),
        )
    return gates.severity_gate(list(severity_records))


def build_gate_battery(
    gold: GoldSets,
    answer_results: Sequence[ItemResult],
    chart_records: Mapping[str, list[dict[str, Any]]],
    *,
    chart_faithfulness_records: Sequence[Mapping[str, Any]] = (),
    classifier_summary: Mapping[str, Any] | None = None,
    severity_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[Any]:
    """The ONE release gate battery, single-sourced into BOTH orchestration
    paths (``run_release_eval`` AND scripts/run_evals.py's
    ``run_offline_suite``) so membership can never drift (issue #303 — two
    independently-spelled batteries had already lost chart_faithfulness
    from one path).

    Every gate is DERIVED from the run records (finding #242 — never
    fabricated): the refusal pair, the canned decline, route_accuracy
    (from the classifier summary), citation_support (from the validation
    records on ItemResult.validation), severity (from ``severity_records``
    — the judged {item_id, expected, judged, scored} feed), the chart trio
    and voices separation. route_accuracy, citation_support AND severity
    are BLOCKED-when-absent rather than dropped — an unmeasured release
    gate blocks, it never vanishes (and the owner-audit guard, finding
    #197, still precedes any severity scoring)."""
    from evals import gates
    from evals.metrics import voices_separation_violations

    answerable_ids = {item["id"] for item in gold.qa_items if item.get("category") != "no_answer"}
    gate_ids = set(gold.refusal_gate_ids)
    canned_ids = set(gold.canned_out_of_scope_ids)

    battery: list[Any] = [
        gates.refusal_gate(
            {
                result.item_id: result.refused
                for result in answer_results
                if result.item_id in gate_ids
            },
            gate_item_ids=gold.refusal_gate_ids,
        ),
        gates.false_refusal_gate(
            {
                result.item_id: result.refused
                for result in answer_results
                if result.item_id in answerable_ids
            }
        ),
    ]
    if canned_ids:
        battery.append(
            gates.canned_out_of_scope_check(
                {
                    result.item_id: (result.route == "canned" or result.refused)
                    for result in answer_results
                    if result.item_id in canned_ids
                }
            )
        )
    battery.append(_route_accuracy_gate(classifier_summary))
    battery.append(_citation_support_gate(answer_results))
    battery.append(_severity_gate(severity_records))
    if chart_records["spec"]:
        battery.append(gates.chart_spec_gate(chart_records["spec"]))
    if chart_faithfulness_records:
        battery.append(gates.chart_faithfulness_gate(list(chart_faithfulness_records)))
    if chart_records["refusal"]:
        battery.append(gates.chart_refusal_gate(chart_records["refusal"]))
    battery.append(
        gates.voices_separation_gate(
            voices_separation_violations(voices_gate_input(answer_results, gold.qa_items))
        )
    )
    return battery


#: Fixture-id fragments naming a rendered-value transform — those series
#: carry the looser post-transform faithfulness tolerance, the rest are
#: pass-through (mirrors the #242 offline classification).
_TRANSFORM_FIXTURE_MARKERS = (
    "splice",
    "rolling",
    "convert",
    "rebaselin",
    "resample",
    "anomaly",
    "degf",
    "_gt_",
)

CHART_FIXTURES_PATH = REPO_ROOT / "evals" / "gold" / "chart_fixtures.json"


def compute_chart_faithfulness_records() -> list[dict[str, Any]]:
    """Rendered-value faithfulness records vs the committed
    ``evals/gold/chart_fixtures.json`` (finding #242): the independent
    fixture generator is re-run over the committed synthetic CSVs and its
    values compared to the committed fixtures, so a drift in either is
    caught. Each series point becomes one record with its tolerance kind.
    Shared by BOTH orchestration paths so chart_faithfulness is one gate
    on one battery (issue #303)."""
    from evals.scripts import compute_chart_fixtures

    committed = json.loads(CHART_FIXTURES_PATH.read_text(encoding="utf-8")).get("fixtures", {})
    fresh = compute_chart_fixtures.compute_fixtures().get("fixtures", {})
    records: list[dict[str, Any]] = []
    for fixture_id, body in committed.items():
        kind = (
            "post_transform"
            if any(marker in fixture_id for marker in _TRANSFORM_FIXTURE_MARKERS)
            else "pass_through"
        )
        fresh_series = fresh.get(fixture_id, {}).get("series", {})
        for series_name, series in body.get("series", {}).items():
            fresh_points = fresh_series.get(series_name, {}).get("points", [])
            for index, point in enumerate(series.get("points", [])):
                actual = fresh_points[index][1] if index < len(fresh_points) else None
                records.append(
                    {
                        "item_id": f"{fixture_id}:{series_name}:{index}",
                        "kind": kind,
                        "expected": float(point[1]),
                        "actual": float(actual) if actual is not None else float("inf"),
                    }
                )
    return records


def affordable_arm_projection(
    reference_usage: Mapping[str, int],
    target_arm: str,
    gold: GoldSets,
    *,
    ledger_path: Path = SPEND_LEDGER_PATH,
) -> tuple[bool, float]:
    """Project a non-first arm's FULL cost from a reference arm's MEASURED
    geometry, priced at the target arm's OWN rates (review #317).

    The Sonnet arm DNF'd at 5/94 items after ~$0.83 of foreseeable spend
    that the static planned-calls estimator waved through — the Haiku arm's
    measured token geometry, repriced at Sonnet rates, already showed the
    arm + its judge batch could not fit the remaining $9.00 cap. This
    projects the target arm's generation from the reference arm's actual
    usage (batch pricing) plus the target arm's judge-batch estimate, and
    reports whether ``cumulative + projection`` still clears the cap.
    Returns ``(affordable, projected_usd)``."""
    generation = pricing.estimate_cost_usd(
        target_arm,
        input_tokens=int(reference_usage.get("input_tokens", 0)),
        output_tokens=int(reference_usage.get("output_tokens", 0)),
        cache_read_tokens=int(reference_usage.get("cache_read_tokens", 0)),
        cache_creation_tokens=int(reference_usage.get("cache_creation_tokens", 0)),
        mode="batch",
    )
    judge = sum(
        pricing.estimate_cost_usd(
            call["model"],
            input_tokens=int(call.get("input_tokens", 0)),
            output_tokens=int(call.get("output_tokens", 0)),
            mode=call.get("mode", "batch"),
        )
        for call in estimate_planned_calls(gold, arm_model=target_arm)
        if call.get("purpose") == "judge"
    )
    projected = generation + judge
    cumulative = ledger.cumulative_usd(Path(ledger_path))
    affordable = (cumulative + projected) < BUDGET_REFUSAL_THRESHOLD_USD
    return affordable, projected


def _resolve_judge_verdicts(
    arm_model: str,
    judge_requests: Sequence[Any],
    batch_client: Any,
    planned_calls: Sequence[Mapping[str, Any]],
    *,
    judges_journal: JudgesJournal | None,
    mode: str,
    ledger_path: Path,
) -> dict[str, Any]:
    """Collect the arm's judge verdicts, reusing a journalled batch when
    one exists (review #316) so a resumed run never re-pays a batch whose
    results are already retrievable by id.

    - fully-collected in the journal → reconstruct, ZERO batch calls;
    - submitted-but-not-collected (kill-after-submit) → collect by id via
      ``results()`` (free for 29 days), NEVER ``create`` again;
    - otherwise → fresh pre-flight + submit, journal the submission BEFORE
      the first poll, collect, journal the collected verdicts."""
    from evals.judges import collect_judge_verdicts, submit_judge_batch

    # Live/recording batches pace polling with the collector's real waiter (a
    # tight no-sleep loop would hammer the Batches API); fake/replay batch
    # doubles end immediately, so tests never sleep.
    collect_kwargs: dict[str, Any] = {} if mode in LIVE_MODES else {"waiter": lambda: None}
    submitted_custom_ids = {request.custom_id for request in judge_requests}

    if judges_journal is not None:
        batch_id, journalled_verdicts = judges_journal.state_for_arm(arm_model)
        if journalled_verdicts is not None and submitted_custom_ids <= set(journalled_verdicts):
            # The $0.34 duplicate batch becomes impossible: the paid verdicts
            # are already on disk, so a fully-resumed run submits nothing.
            return {
                request.custom_id: journalled_verdicts[request.custom_id]
                for request in judge_requests
            }
        if batch_id is not None:
            verdicts = collect_judge_verdicts(
                batch_id, judge_requests, batch_client, **collect_kwargs
            )
            judges_journal.record_collection(arm_model, batch_id, verdicts)
            return verdicts

    judge_preflight = preflight_budget(planned_calls, ledger_path=ledger_path)
    if mode in LIVE_MODES and not judge_preflight.allowed:
        raise BudgetExceededError(
            f"release eval refused the judge batch for arm {arm_model!r}: the "
            "re-read ledger would cross the $9.00 cap (finding #236)"
        )
    batch_id = submit_judge_batch(judge_requests, batch_client, preflight=judge_preflight)
    if judges_journal is not None:
        # BEFORE the first poll (#316): a crash while polling must never
        # orphan the paid batch — resume collects it by id.
        judges_journal.record_submission(arm_model, batch_id, sorted(submitted_custom_ids))
    verdicts = collect_judge_verdicts(batch_id, judge_requests, batch_client, **collect_kwargs)
    if judges_journal is not None:
        judges_journal.record_collection(arm_model, batch_id, verdicts)
    return verdicts


def run_release_eval(
    gold: GoldSets,
    *,
    arm_models: Sequence[str],
    deps_factory: Callable[[str], AnswerPathDeps],
    plan_chart: Callable[[str], Any],
    batch_client: Any,
    mode: str = "live",
    ledger_path: Path = SPEND_LEDGER_PATH,
    journal_dir: Path | None = None,
    session_id: str = "release-eval",
    results_mode: str | None = None,
    classifier_summary: Mapping[str, Any] | None = None,
    escalation_arm_model: str | None = None,
) -> dict[str, Any]:
    """The ONE release-eval orchestration path (finding #236): for each
    arm it re-reads the ledger and re-computes a fresh budget pre-flight
    BEFORE every spend segment (the answer path AND the judge batch), so a
    pre-flight is never a stale bearer token — arm 1 crossing the cap
    mid-run means arm 2 refuses with BudgetExceededError before any adapter
    call, and no judge batch is created past the cap. Fake/replay modes
    spend $0 and never touch the ledger; the tested path IS the shipped
    --live path with only the adapter/batch-client seams swapped.

    Each arm's verdict comes from the ONE shared ``build_gate_battery``
    (issue #303): its citation_support gate is fed from the run's own
    validation outcomes (``deps.validate_exchange`` drives the #13
    validator per answered exchange; absent validation is BLOCKED) and its
    route_accuracy gate from ``classifier_summary`` (absent is BLOCKED),
    and its severity gate from the collected judge batch's severity-kind
    verdicts joined to the audited gold labels
    (``evals.judges.severity_records_from_verdicts``; a run with no
    judge batch stays BLOCKED-unmeasured — the #308 fail-closed pin).

    Opus escalation is opt-in (``escalation_arm_model``): after the cheaper
    arms run, the decision goes through ``gates.opus_escalation_allowed``
    exactly once, with the cheaper arms' ArmResults and a freshly-computed
    BudgetPreflight for the escalation arm; only its True verdict drives
    the escalation arm (ratified escalation-only, NO top-up)."""
    from evals import gates, report
    from evals.judges import build_judge_requests, severity_records_from_verdicts

    ledger_path = Path(ledger_path)

    # The chart path is model-independent: plan once and reuse for every arm.
    chart_journal = RunJournal(journal_dir / "charts.jsonl") if journal_dir is not None else None
    chart_results = run_chart_path(gold.chart_items, plan_chart, journal=chart_journal)
    chart_records = chart_gate_records(gold.chart_items, chart_results)
    # chart_faithfulness is model-independent too (committed fixtures vs a
    # fresh recompute) — one gate on the one battery (issue #303).
    faithfulness_records = compute_chart_faithfulness_records()

    # Review #316: the judge-batch journal — a resumed run reuses paid-for
    # verdicts (or collects a submitted batch by id) instead of re-creating it.
    judges_journal = (
        JudgesJournal(journal_dir / "judges.jsonl") if journal_dir is not None else None
    )

    gold_by_id = {item["id"]: item for item in gold.qa_items}
    arms: list[Any] = []
    arm_extras: list[dict[str, Any]] = []
    # Review #317: the first arm to run becomes the affordability reference —
    # its MEASURED geometry projects every later arm before that arm spends.
    reference_usage: dict[str, int] | None = None

    def drive_arm(arm_model: str, *, check_affordability: bool = False) -> None:
        nonlocal reference_usage
        # The planned-calls estimate is a pure function of (gold, arm_model)
        # and nothing mutates gold across this arm, so build it ONCE and reuse
        # it for both segment pre-flights (finding #297) — this rebuilds every
        # judge prompt (per-severity rubric-file reads included) only once per
        # arm. The #236 pin is the fresh LEDGER read per segment, which each
        # preflight_budget call below still does.
        planned_calls = estimate_planned_calls(gold, arm_model=arm_model)
        # Segment 1 (answer path): re-read the ledger, fresh pre-flight. The
        # #236 hard stop comes FIRST — a ledger that has already crossed the
        # cap refuses loudly (BudgetExceededError), before the softer #317
        # affordability projection is even consulted.
        preflight = preflight_budget(planned_calls, ledger_path=ledger_path)
        if mode in LIVE_MODES and not preflight.allowed:
            raise BudgetExceededError(
                f"release eval refused before arm {arm_model!r}: the re-read ledger "
                f"(${preflight.cumulative_usd:.4f}) + estimate "
                f"(${preflight.estimated_cost_usd:.4f}) would cross the "
                f"${preflight.threshold_usd:.2f} cap — no top-up (finding #236)"
            )
        # Review #317: the #236 static estimator uses the production 1024-token
        # output budget, which sails through for a big-output arm; project THIS
        # non-first arm's full cost from the reference arm's MEASURED geometry
        # priced at this arm's rates instead. If it cannot fit the remaining
        # budget, refuse the WHOLE arm here — no deps_factory, no adapter call,
        # no judge batch, no ledger row — recorded as a DNF-unaffordable arm
        # verdict (never silently dropped, never run item-by-item into the wall).
        if check_affordability and reference_usage is not None:
            affordable, projected = affordable_arm_projection(
                reference_usage, arm_model, gold, ledger_path=ledger_path
            )
            if not affordable:
                arms.append(
                    gates.ArmResult(
                        model=arm_model,
                        gates=(),
                        cost_usd=float(projected),
                        arm_verdict="dnf-unaffordable",
                        reason=(
                            f"dnf-unaffordable: the projection from the reference arm's "
                            f"measured geometry (${projected:.2f} at {arm_model}'s rates + "
                            f"judge batch) cannot fit the remaining budget under the "
                            f"${BUDGET_REFUSAL_THRESHOLD_USD:.2f} cap"
                        ),
                    )
                )
                arm_extras.append({"retrieval_metrics": {}, "calibrated_term_preserved_rate": 0.0})
                return
        deps = deps_factory(arm_model)
        answer_journal = (
            RunJournal(journal_dir / f"{arm_model}-answers.jsonl")
            if journal_dir is not None
            else None
        )
        # Journal-resumed items make ZERO adapter calls, so their usage must
        # never be re-ledgered on resume (the ledger twin of finding #237's
        # "resume must not re-pay"): only the FRESH results of this run
        # contribute to the spend row.
        resumed_ids = (
            answer_journal.completed_item_ids() if answer_journal is not None else frozenset()
        )
        answer_results = run_answer_path(
            gold.qa_items,
            deps,
            arm_model=arm_model,
            mode=mode,
            journal=answer_journal,
            preflight=preflight,
        )
        fresh_results = [result for result in answer_results if result.item_id not in resumed_ids]
        if mode in LIVE_MODES and fresh_results:
            record_run_spend(
                ledger_path,
                mode="batch",
                model=arm_model,
                activity="release-eval-generation",
                usage=_aggregate_usage(fresh_results),
                calls=len(fresh_results),
                session_id=session_id,
            )
        # Review #317: the first arm to complete its answer path becomes the
        # affordability reference — its FULL measured geometry (resumed items
        # included, so the projection is stable across resumes) prices every
        # later arm before that arm is allowed to spend.
        if reference_usage is None:
            reference_usage = _aggregate_usage(answer_results)

        # Segment 2 (judge batch): collect the arm's verdicts, reusing a
        # journalled batch when one exists (#316) — a resumed run never
        # re-creates a batch whose results are retrievable by id. The fresh
        # path re-reads the ledger for its own pre-flight inside the helper.
        judge_requests = build_judge_requests(answer_results, gold_by_id, arm_model=arm_model)
        severity_records: list[dict[str, Any]] | None = None
        if judge_requests:
            verdicts = _resolve_judge_verdicts(
                arm_model,
                judge_requests,
                batch_client,
                planned_calls,
                judges_journal=judges_journal,
                mode=mode,
                ledger_path=ledger_path,
            )
            # The live severity feed: severity-kind verdicts joined to the
            # audited gold labels. An empty mapping (no severity judge
            # request was ever built) stays None — the shared battery then
            # reports severity BLOCKED-unmeasured (the #308 fail-closed pin),
            # never a vacuous 0/0.
            severity_records = severity_records_from_verdicts(verdicts, gold_by_id) or None

        battery = build_gate_battery(
            gold,
            answer_results,
            chart_records,
            chart_faithfulness_records=faithfulness_records,
            classifier_summary=classifier_summary,
            severity_records=severity_records,
        )
        arms.append(
            gates.ArmResult(
                model=arm_model,
                gates=tuple(battery),
                cost_usd=float(preflight.estimated_cost_usd),
            )
        )
        arm_extras.append(
            {
                "retrieval_metrics": compute_retrieval_metrics(answer_results, gold_by_id),
                "calibrated_term_preserved_rate": compute_calibrated_term_rate(answer_results),
            }
        )

    for index, arm_model in enumerate(arm_models):
        # #317: every arm after the first is affordability-projected from the
        # reference arm's measured geometry before it spends.
        drive_arm(arm_model, check_affordability=index > 0)

    # Opus escalation (opt-in): consult the ratified policy ONCE with the
    # cheaper arms and a fresh pre-flight for the escalation arm; obey it.
    if escalation_arm_model is not None:
        escalation_preflight = preflight_budget(
            estimate_planned_calls(gold, arm_model=escalation_arm_model), ledger_path=ledger_path
        )
        if gates.opus_escalation_allowed(tuple(arms), escalation_preflight):
            drive_arm(escalation_arm_model)

    selected = gates.select_production_model(arms)
    verdict_gates = next(
        (list(arm.gates) for arm in arms if arm.model == selected),
        list(arms[0].gates),
    )
    verdict = gates.release_verdict(verdict_gates)
    payload = report.build_results_payload(
        arms, verdict=verdict, selected_model=selected, mode=results_mode
    )
    for arm_payload, extras in zip(payload["arms"], arm_extras, strict=True):
        arm_payload.update(extras)
    return payload
