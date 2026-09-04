"""Issue #303 red phase (Fable): wiring the release-gate gap.

The 2026-09 phase-gate audit (issue #303) found the release contract
spelled out twice with drifted membership — evals/harness.py's
``_arm_gate_battery`` (used by ``run_release_eval``) vs
scripts/run_evals.py's inline offline battery (which alone carries
``chart_faithfulness_gate``) — while three built, unit-tested release
gates are wired into NEITHER path: ``citation_support_gate`` (the
product's core guarantee, RATIFIED at 0.95 pooled-sentence as a RELEASE
GATE — issue #21 orchestrator ratification item 4), ``route_accuracy_gate``
and ``opus_escalation_allowed``. The harness seam built to feed the
citation gate (``AnswerPathDeps.validate_exchange`` /
``ItemResult.validation``) is dead: never invoked, never populated.

**Orchestrator decision (binding): WIRE, don't delete.**

Contracts pinned here:

1. ONE shared battery builder — ``evals.harness.build_gate_battery`` —
   single-sourced into BOTH orchestration paths (``run_release_eval``
   and ``scripts/run_evals.py``'s ``run_offline_suite``), membership
   pinned by gate name so the chart_faithfulness-style silent drift
   becomes impossible.
2. ``citation_support`` fed from the #13 validator outcomes carried on
   ``ItemResult.validation``: ``run_answer_path`` drives the
   ``validate_exchange`` seam for every answered exchange exactly like
   production (a real GroundedAnswer + the #12-shaped delivered
   transcript; fake/replay-driven under pytest), and a release run
   where validation never executed reports the gate BLOCKED — never
   passed, never silently absent.
3. ``route_accuracy`` fed from the classifier-accuracy JSON summary
   (evals/scripts/classifier_accuracy.py's ``release_gate_passes``);
   an absent summary is BLOCKED, never absent.
4. Opus escalation decided by ``gates.opus_escalation_allowed`` INSIDE
   the release orchestrator: the escalation arm is driven only when
   the cheaper arms failed AND the freshly-computed pre-flight fits.

No test here touches the network or mutates a committed gold set
(IMPLEMENTATION.md §4.4; synthetic gold only, ratification item 8).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from evals import harness
from evals.gates import (
    GATE_BLOCKED,
    GATE_FAILED,
    GATE_PASSED,
    GATE_STATUSES,
    ArmResult,
)
from evals.harness import (
    AnswerPathDeps,
    GoldSets,
    ItemResult,
    RunJournal,
    load_and_validate_gold,
    run_answer_path,
)
from evals.judges import OPUS_ARM_MODEL
from rag.citation_validator import AnswerSentence, PairVerdict, ValidationOutcome
from rag.generation import GroundedAnswer
from rag.provider import AnswerWithCitations, Citation, FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    production_passage_payload,
    transport_stream_for_answer,
    write_synthetic_gold,
)

ARM_MODEL = "claude-haiku-4-5"

#: The pinned release battery membership — identical on BOTH paths
#: (issue #303: "one gate battery, single-sourced"). The chart trio is
#: chart_spec + chart_faithfulness + chart_refusal.
EXPECTED_BATTERY_NAMES = frozenset(
    {
        "route_accuracy",
        "citation_support",
        "refusal",
        "false_refusal",
        "canned_out_of_scope",
        "severity",
        "chart_spec",
        "chart_faithfulness",
        "chart_refusal",
        "voices_separation",
    }
)

CLASSIFIER_SUMMARY_PASSING = {
    "release_gate_passes": True,
    "per_class": {"in_scope": 1.0, "unsafe": 1.0},
}

CLASSIFIER_SUMMARY_FAILING = {
    "release_gate_passes": False,
    "per_class": {"in_scope": 1.0, "unsafe": 0.9},
}

CHART_RECORDS = {
    "spec": [
        {"item_id": "syn-chart-01", "status": "match"},
        {
            "item_id": "syn-chart-flagship",
            "status": "skipped_blocked",
            "blocked_reason": "synthetic stand-in for the #117 fixture embargo",
        },
    ],
    "refusal": [{"item_id": "syn-chart-02", "status": "refused_with_nearest"}],
}

CHART_FAITHFULNESS_RECORDS = [
    {"item_id": "syn-fixture:series:0", "kind": "pass_through", "expected": 1.0, "actual": 1.0},
]

ANSWER = AnswerWithCitations(
    text="The synthetic planet warmed 1.1 degrees. It is very likely to continue.",
    citations=(
        Citation(cited_text="synthetic passage", document_index=0, document_title="Syn doc"),
    ),
)

#: The streamed delivery of ANSWER — the runner drives the production
#: streamed seam (#303 ratification note 6).
ANSWER_STREAM = transport_stream_for_answer(ANSWER)

PASSAGES = RetrievedPassages(
    passages=(
        RerankedPassage(
            chunk_id="syn_doc:0001",
            rerank_score=0.9,
            clears_threshold=True,
            payload=production_passage_payload("syn_doc:0001"),
        ),
    )
)

REFUSAL = HonestRefusal(
    refusal_text="The synthetic corpus does not cover this.",
    covered_topics=("synthetic warming",),
    top_score=0.05,
    threshold=0.4,
)

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

REFUSAL_ITEM = {
    "id": "syn-na-g-01",
    "category": "no_answer",
    "question": "Synthetic unanswerable question?",
    "expected_behaviour": "refusal",
    "subset": "gate",
    "expected_route": "retrieval_refusal",
}

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic planet warming",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validated_outcome(*, supported: int, factual: int) -> ValidationOutcome:
    """A genuine #13 ValidationOutcome (the shape production's
    validate_exchange returns): ``factual`` cited factual sentences of
    which the first ``supported`` carry an entailment-supported verdict."""
    sentences = tuple(
        AnswerSentence(index=i, text=f"Synthetic factual sentence {i}.", document_indices=(0,))
        for i in range(factual)
    )
    verdicts = tuple(
        PairVerdict(pair_index=i, sentence_index=i, document_index=0, supported=(i < supported))
        for i in range(factual)
    )
    return ValidationOutcome(
        validated=True,
        sentences=sentences,
        verdicts=verdicts,
        support_rate=(supported / factual) if factual else None,
        model=ARM_MODEL,
    )


def _degraded_outcome(*, factual: int, reason: str) -> ValidationOutcome:
    sentences = tuple(
        AnswerSentence(index=i, text=f"Synthetic factual sentence {i}.", document_indices=(0,))
        for i in range(factual)
    )
    return ValidationOutcome(
        validated=False,
        sentences=sentences,
        degraded_reason=reason,
        model=ARM_MODEL,
    )


def _synthetic_gold(tmp_path: Path) -> GoldSets:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _fabricated_answer_results(
    gold: GoldSets,
    *,
    validation_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[ItemResult, ...]:
    """ItemResults shaped like a clean run over the synthetic gold:
    no_answer items refused, everything else answered with a report-type
    document set; answered items optionally carry validation records."""
    validation_by_id = validation_by_id or {}
    results: list[ItemResult] = []
    for item in gold.qa_items:
        if item.get("category") == "no_answer":
            results.append(
                ItemResult(
                    item_id=item["id"],
                    arm_model=ARM_MODEL,
                    route="retrieval",
                    refused=True,
                    answer_text="The synthetic corpus does not cover this.",
                )
            )
        else:
            results.append(
                ItemResult(
                    item_id=item["id"],
                    arm_model=ARM_MODEL,
                    route="retrieval",
                    refused=False,
                    answer_text=ANSWER.text,
                    documents=({"chunk_id": "syn_doc:0001", "source_type": "report"},),
                    validation=validation_by_id.get(item["id"]),
                )
            )
    return tuple(results)


def _battery_builder():
    builder = getattr(harness, "build_gate_battery", None)
    assert builder is not None, (
        "evals.harness.build_gate_battery must exist: the ONE shared battery "
        "builder both run_release_eval and scripts/run_evals.py assemble their "
        "gates from (issue #303 — two independently-spelled batteries have "
        "already drifted; wiring decision: WIRE, don't delete)"
    )
    return builder


def _build_battery(
    gold: GoldSets,
    answer_results,
    *,
    classifier_summary=CLASSIFIER_SUMMARY_PASSING,
    severity_records=None,
):
    return _battery_builder()(
        gold,
        answer_results,
        CHART_RECORDS,
        chart_faithfulness_records=CHART_FAITHFULNESS_RECORDS,
        classifier_summary=classifier_summary,
        severity_records=severity_records,
    )


def _gate_by_name(battery, name: str):
    matches = [gate for gate in battery if gate.name == name]
    assert matches, (
        f"the release battery must carry a {name!r} gate; got only "
        f"{sorted(gate.name for gate in battery)} — an absent gate is exactly "
        "the silent gap issue #303 closes"
    )
    assert len(matches) == 1, f"duplicate {name!r} gates in one battery"
    return matches[0]


def _fully_validated(gold: GoldSets) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: {"validated": True, "supported": 2, "factual": 2}
        for item in gold.qa_items
        if item.get("category") != "no_answer"
    }


# ---------------------------------------------------------------------------
# 1. One gate battery, single-sourced
# ---------------------------------------------------------------------------


def test_shared_battery_builder_membership_is_the_full_release_contract(tmp_path: Path):
    """build_gate_battery assembles EVERY ratified release gate — the
    refusal pair, the canned-decline check, severity, the chart trio,
    voices separation, route_accuracy AND citation_support — with every
    status inside the closed vocabulary."""
    gold = _synthetic_gold(tmp_path)
    battery = _build_battery(
        gold, _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold))
    )
    names = [gate.name for gate in battery]
    assert EXPECTED_BATTERY_NAMES <= set(names), (
        f"battery membership {sorted(names)} is missing "
        f"{sorted(EXPECTED_BATTERY_NAMES - set(names))}"
    )
    for gate in battery:
        assert gate.status in GATE_STATUSES


def test_offline_suite_single_sources_the_battery_builder():
    """scripts/run_evals.py must assemble its offline battery from THE
    harness builder — the same function object, not a second spelled-out
    list (the drift that already lost chart_faithfulness from one path)."""
    from tests.unit.test_eval_report import _load_run_evals_module

    run_evals = _load_run_evals_module()
    builder = _battery_builder()
    offline_builder = getattr(run_evals, "build_gate_battery", None)
    assert offline_builder is not None, (
        "scripts/run_evals.py must import build_gate_battery from evals.harness "
        "(issue #303: one battery definition instead of two)"
    )
    assert offline_builder is builder, (
        "scripts/run_evals.py must use evals.harness.build_gate_battery ITSELF, "
        "not a copy — single-sourcing is the point"
    )


def test_offline_and_release_paths_assemble_identical_battery_membership(tmp_path: Path):
    """The offline suite and the release orchestrator, run over the SAME
    synthetic gold, produce byte-identical gate-name membership — and
    both carry the full pinned set. Divergence (a gate present on one
    path only) must be impossible."""
    import json

    from tests.unit.test_eval_report import _load_run_evals_module

    run_evals = _load_run_evals_module()
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    out_dir = tmp_path / "offline-out"
    run_evals.run_offline_suite(qa_path, charts_path, out_dir)
    offline_payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    (offline_arm,) = offline_payload["arms"]
    offline_names = {gate["name"] for gate in offline_arm["gates"]}

    gold = load_and_validate_gold(qa_path, charts_path)
    release_payload = _run_release_eval_fake(
        gold,
        tmp_path,
        validate=lambda result, sse_events: _validated_outcome(supported=2, factual=2),
    )
    release_names = {gate["name"] for gate in release_payload["arms"][0]["gates"]}

    assert offline_names == release_names, (
        f"gate batteries diverged: offline-only {sorted(offline_names - release_names)}, "
        f"release-only {sorted(release_names - offline_names)} — issue #303's exact trap"
    )
    assert EXPECTED_BATTERY_NAMES <= offline_names


# ---------------------------------------------------------------------------
# 2. citation_support wired: the ItemResult.validation feed
# ---------------------------------------------------------------------------


def test_run_answer_path_validates_answered_exchanges_like_production():
    """The dead seam comes alive: for an answered exchange the runner
    invokes deps.validate_exchange exactly once with (a real
    GroundedAnswer, the #12-shaped delivered transcript) — the SAME
    contract service.app drives — and records the derived outcome on
    ItemResult.validation ({validated, supported, factual})."""
    calls: list[tuple[Any, tuple[Any, ...]]] = []

    def fake_validate(result, sse_events):
        calls.append((result, tuple(sse_events)))
        return _validated_outcome(supported=2, factual=2)

    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: PASSAGES, validate_exchange=fake_validate
    )
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL)

    assert len(calls) == 1, (
        "run_answer_path must drive the validate_exchange seam exactly once per "
        "answered exchange (issue #303: the seam is currently dead — eval answers "
        "are never validated, so the ratified citation-support release gate has "
        "no honest feed)"
    )
    validated_result, sse_events = calls[0]
    assert isinstance(validated_result, GroundedAnswer), (
        "the exchange must be validated LIKE PRODUCTION: the #13 validator "
        "short-circuits anything that is not a GroundedAnswer to a skipped "
        "outcome with zero calls, so feeding it any other shape silently "
        "validates nothing"
    )
    assert validated_result.text == ANSWER.text
    assert validated_result.cited_passages, "the resolved citations must ride the GroundedAnswer"
    assert validated_result.cited_passages[0].payload["body"] == "synthetic passage"

    delivered_text = "".join(
        str(event.get("data", {}).get("text", ""))
        for event in sse_events
        if event.get("event") == "text"
    )
    assert delivered_text == ANSWER.text, (
        "the transcript handed to the validator must be the #12-shaped delivery "
        "of the answer text — segmentation runs over what was delivered"
    )
    citation_indices = [
        event["data"]["document_index"] for event in sse_events if event.get("event") == "citation"
    ]
    assert citation_indices == [0], (
        "every resolved citation must appear as a #12 citation event carrying its document_index"
    )

    assert result.validation is not None, (
        "ItemResult.validation must carry the validation outcome — the "
        "citation_support gate's only honest feed"
    )
    assert result.validation.get("validated") is True
    assert result.validation.get("supported") == 2
    assert result.validation.get("factual") == 2


def test_run_answer_path_records_degraded_validation_fail_closed():
    """A degraded validator outcome is recorded fail-closed: zero
    supported, the factual sentence count preserved (the #239
    denominator contract), and the degraded_reason carried — the gate
    then counts these sentences against release."""

    def degraded_validate(result, sse_events):
        return _degraded_outcome(factual=2, reason="ProviderError: synthetic transport failure")

    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: PASSAGES, validate_exchange=degraded_validate
    )
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL)

    assert result.validation is not None
    assert result.validation.get("validated") is False
    assert result.validation.get("supported") == 0
    assert result.validation.get("factual") == 2, (
        "a degraded exchange must still carry its segmented factual-sentence "
        "count: citation_support_gate refuses countless degraded records "
        "(finding #239) — dropping the count would crash every degraded run"
    )
    assert "ProviderError" in str(result.validation.get("degraded_reason"))


def test_run_answer_path_never_validates_refusals():
    """The §3.5 refusal path spends no generation call and no validation
    call: validate_exchange is never invoked and validation stays None."""

    def poisoned_validate(result, sse_events):
        raise AssertionError("validate_exchange must never run for a refusal")

    adapter = FakeAdapter(structured_results=[CLASSIFICATION_IN_SCOPE])
    deps = AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: REFUSAL, validate_exchange=poisoned_validate
    )
    (result,) = run_answer_path([REFUSAL_ITEM], deps, arm_model=ARM_MODEL)
    assert result.refused is True
    assert result.validation is None


def test_validation_outcome_survives_journal_resume(tmp_path: Path):
    """The journal round-trips the validation record: a resumed item
    returns its recorded validation with ZERO validate calls — resume
    must never silently drop the citation gate's paid-for evidence."""
    journal = RunJournal(tmp_path / "journal.jsonl")

    adapter = FakeAdapter(
        generate_stream_results=[ANSWER_STREAM], structured_results=[CLASSIFICATION_IN_SCOPE]
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=lambda result, sse_events: _validated_outcome(supported=2, factual=2),
    )
    (first,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, journal=journal)
    assert first.validation is not None, "red pre-condition: the seam records validation"

    def poisoned_validate(result, sse_events):
        raise AssertionError("resume must make zero validate calls")

    resume_adapter = FakeAdapter()
    resume_deps = AnswerPathDeps(
        adapter=resume_adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=poisoned_validate,
    )
    (resumed,) = run_answer_path(
        [ANSWERABLE_ITEM], resume_deps, arm_model=ARM_MODEL, journal=journal
    )
    assert resumed.validation == first.validation


def test_citation_support_gate_fed_from_validation_records(tmp_path: Path):
    """The battery builder feeds citation_support_gate the pooled-sentence
    arithmetic RATIFIED for release (issue #21 ratification item 4):
    validated exchanges contribute supported/factual; a degraded exchange
    contributes its factual sentences with ZERO supported. 5 supported of
    7 pooled factual sentences is below the 0.95 threshold -> FAILED."""
    from evals.gates import CITATION_SUPPORT_THRESHOLD

    gold = _synthetic_gold(tmp_path)
    validation_by_id = {
        "syn-sp-01": {"validated": True, "supported": 2, "factual": 2},
        "syn-mp-01": {"validated": True, "supported": 3, "factual": 3},
        "syn-sev-01": {
            "validated": False,
            "supported": 0,
            "factual": 2,
            "degraded_reason": "ProviderError: synthetic transport failure",
        },
    }
    battery = _build_battery(
        gold, _fabricated_answer_results(gold, validation_by_id=validation_by_id)
    )
    gate = _gate_by_name(battery, "citation_support")
    assert gate.status == GATE_FAILED
    assert gate.numerator == 5
    assert gate.denominator == 7
    assert gate.threshold == CITATION_SUPPORT_THRESHOLD
    evidence_ids = {entry.get("item_id") for entry in gate.evidence}
    assert {"syn-sp-01", "syn-mp-01", "syn-sev-01"} <= evidence_ids, (
        "the gate's evidence must link every answered item back to its record"
    )

    fully_supported = _build_battery(
        gold, _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold))
    )
    assert _gate_by_name(fully_supported, "citation_support").status == GATE_PASSED


def test_citation_support_blocked_when_validation_never_ran(tmp_path: Path):
    """A release run where validation was never executed (no answered
    item carries a validation record) reports citation_support BLOCKED —
    never passed, never absent: an unmeasured core guarantee must block
    release exactly like the pending owner severity audit does."""
    from evals.gates import release_verdict

    gold = _synthetic_gold(tmp_path)
    battery = _build_battery(gold, _fabricated_answer_results(gold, validation_by_id=None))
    gate = _gate_by_name(battery, "citation_support")
    assert gate.status == GATE_BLOCKED
    assert gate.reason, "the BLOCKED gate must say why"
    assert "validat" in gate.reason.lower()
    assert release_verdict(list(battery)) != "passed"


# ---------------------------------------------------------------------------
# 3. route_accuracy wired: the classifier-summary feed
# ---------------------------------------------------------------------------


def test_route_accuracy_gate_fed_from_classifier_summary(tmp_path: Path):
    """The battery's route_accuracy gate consumes the classifier-accuracy
    JSON summary (evals/scripts/classifier_accuracy.py): its
    release_gate_passes flips the gate."""
    gold = _synthetic_gold(tmp_path)
    results = _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold))

    passing = _build_battery(gold, results, classifier_summary=CLASSIFIER_SUMMARY_PASSING)
    assert _gate_by_name(passing, "route_accuracy").status == GATE_PASSED

    failing = _build_battery(gold, results, classifier_summary=CLASSIFIER_SUMMARY_FAILING)
    assert _gate_by_name(failing, "route_accuracy").status == GATE_FAILED


def test_route_accuracy_blocked_when_no_classifier_summary(tmp_path: Path):
    """No classifier summary -> route_accuracy BLOCKED, present in the
    battery — the classifier gate must never vanish from the release
    verdict just because nobody ran the accuracy eval."""
    gold = _synthetic_gold(tmp_path)
    battery = _build_battery(
        gold,
        _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold)),
        classifier_summary=None,
    )
    gate = _gate_by_name(battery, "route_accuracy")
    assert gate.status == GATE_BLOCKED
    assert gate.reason, "the BLOCKED gate must say why"


# ---------------------------------------------------------------------------
# severity wired: the judged-severity feed (owner audit complete 2026-09-04)
# ---------------------------------------------------------------------------


def test_severity_gate_scores_supplied_judged_records(tmp_path: Path):
    """With the owner audit complete, a supplied judged-severity feed is
    SCORED by the battery's severity gate — the offline suite's simulated
    exact-match feed passes; a two-level error in the feed fails."""
    gold = _synthetic_gold(tmp_path)
    results = _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold))

    exact = [
        {"item_id": "syn-sev-01", "expected": "serious", "judged": "serious", "scored": True},
    ]
    passing = _build_battery(gold, results, severity_records=exact)
    gate = _gate_by_name(passing, "severity")
    assert gate.status == GATE_PASSED
    assert (gate.numerator, gate.denominator) == (1, 1)

    two_level = [
        {
            "item_id": "syn-sev-01",
            "expected": "emergency-level",
            "judged": "reassuring",
            "scored": True,
        },
    ]
    failing = _build_battery(gold, results, severity_records=two_level)
    assert _gate_by_name(failing, "severity").status == GATE_FAILED


def test_severity_blocked_when_no_judged_records(tmp_path: Path):
    """No judged-severity feed -> severity BLOCKED, present in the
    battery — with the owner audit complete an unmeasured severity gate
    must block release exactly like an absent classifier summary or
    never-run validation, never fail on a vacuous empty 0/0 and never
    vanish (fail-closed, orchestrator adjudication on PR #308)."""
    from evals.gates import release_verdict

    gold = _synthetic_gold(tmp_path)
    battery = _build_battery(
        gold,
        _fabricated_answer_results(gold, validation_by_id=_fully_validated(gold)),
        severity_records=None,
    )
    gate = _gate_by_name(battery, "severity")
    assert gate.status == GATE_BLOCKED
    assert gate.reason, "the BLOCKED gate must say why"
    assert "severity" in gate.reason
    assert release_verdict(list(battery)) != "passed"


# ---------------------------------------------------------------------------
# Release-orchestrator wiring (items 1, 2 and 4 end-to-end, fake mode, $0)
# ---------------------------------------------------------------------------


def _release_deps_factory(gold: GoldSets, *, validate=None, refuse_everything=False, driven=None):
    def deps_factory(arm_model: str) -> AnswerPathDeps:
        if driven is not None:
            driven.append(arm_model)
        adapter = FakeAdapter(
            generate_stream_results=[ANSWER_STREAM] * len(gold.qa_items),
            structured_results=[
                {"scope": "in_scope", "rewritten_query": item["question"]} for item in gold.qa_items
            ],
        )

        def retrieve(decision):
            query = decision.retrieval_query or ""
            if refuse_everything or "unanswerable" in query or "out-of-scope" in query:
                return REFUSAL
            return PASSAGES

        return AnswerPathDeps(adapter=adapter, retrieve=retrieve, validate_exchange=validate)

    return deps_factory


def _run_release_eval_fake(
    gold: GoldSets,
    tmp_path: Path,
    *,
    validate=None,
    refuse_everything=False,
    driven=None,
    **kwargs,
):
    return harness.run_release_eval(
        gold,
        arm_models=(ARM_MODEL,),
        deps_factory=_release_deps_factory(
            gold, validate=validate, refuse_everything=refuse_everything, driven=driven
        ),
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=FakeBatchClient(),
        mode="fake",
        ledger_path=tmp_path / "spend-ledger.csv",
        journal_dir=None,
        session_id="review-303-fake",
        **kwargs,
    )


def _assert_orchestrator_accepts(parameter: str) -> None:
    signature = inspect.signature(harness.run_release_eval)
    assert parameter in signature.parameters, (
        f"run_release_eval must accept {parameter!r} (issue #303 wiring); "
        f"got parameters {sorted(signature.parameters)}"
    )


def test_release_eval_battery_carries_full_membership_and_live_citation_feed(tmp_path: Path):
    """run_release_eval's per-arm battery is the FULL pinned set, with
    citation_support fed from the run's own validation outcomes (all
    supported here -> passed) and route_accuracy fed from the supplied
    classifier summary."""
    _assert_orchestrator_accepts("classifier_summary")
    gold = _synthetic_gold(tmp_path)
    payload = _run_release_eval_fake(
        gold,
        tmp_path,
        validate=lambda result, sse_events: _validated_outcome(supported=2, factual=2),
        classifier_summary=CLASSIFIER_SUMMARY_PASSING,
    )
    (arm,) = payload["arms"]
    names = {gate["name"] for gate in arm["gates"]}
    assert EXPECTED_BATTERY_NAMES <= names, (
        f"release battery {sorted(names)} is missing {sorted(EXPECTED_BATTERY_NAMES - names)}"
    )
    citation = next(gate for gate in arm["gates"] if gate["name"] == "citation_support")
    assert citation["status"] == "passed"
    assert citation["denominator"] and citation["denominator"] > 0
    route = next(gate for gate in arm["gates"] if gate["name"] == "route_accuracy")
    assert route["status"] == "passed"


def test_release_eval_without_validation_reports_citation_support_blocked(tmp_path: Path):
    """A release run whose deps carry no validate_exchange (validation
    never executed) reports citation_support BLOCKED in the payload —
    the verdict cannot be 'passed' while the core guarantee is
    unmeasured."""
    gold = _synthetic_gold(tmp_path)
    payload = _run_release_eval_fake(gold, tmp_path, validate=None)
    (arm,) = payload["arms"]
    citation = next((gate for gate in arm["gates"] if gate["name"] == "citation_support"), None)
    assert citation is not None, (
        "citation_support must appear in the battery even when validation never "
        "ran — absent is exactly the silent gap issue #303 closes"
    )
    assert citation["status"] == "blocked"
    assert payload["release_verdict"] != "passed"


# ---------------------------------------------------------------------------
# 4. opus_escalation_allowed wired into the bake-off selection path
# ---------------------------------------------------------------------------


def test_release_eval_drives_escalation_arm_when_cheaper_arms_fail(tmp_path: Path):
    """With every cheaper arm failing its gates (wrong refusals on the
    answerable items) and the fresh pre-flight fitting the remaining
    budget, the orchestrator drives the escalation arm and reports it
    after the cheaper arms."""
    _assert_orchestrator_accepts("escalation_arm_model")
    gold = _synthetic_gold(tmp_path)
    driven: list[str] = []
    payload = harness.run_release_eval(
        gold,
        arm_models=("claude-haiku-4-5", "claude-sonnet-5"),
        deps_factory=_release_deps_factory(gold, refuse_everything=True, driven=driven),
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=FakeBatchClient(),
        mode="fake",
        ledger_path=tmp_path / "spend-ledger.csv",
        journal_dir=None,
        session_id="review-303-escalation",
        escalation_arm_model=OPUS_ARM_MODEL,
    )
    assert driven[:2] == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert OPUS_ARM_MODEL in driven, (
        "both cheaper arms failed and the budget fits: the ratified "
        "escalation-only Opus arm must actually run"
    )
    models = [arm["model"] for arm in payload["arms"]]
    assert models == ["claude-haiku-4-5", "claude-sonnet-5", OPUS_ARM_MODEL]


def test_release_eval_escalation_decision_goes_through_opus_escalation_allowed(
    tmp_path: Path, monkeypatch
):
    """The orchestrator's escalation decision IS gates.opus_escalation_allowed:
    it is called exactly once with the cheaper arms' ArmResults and a
    freshly-computed BudgetPreflight for the escalation arm, and its
    verdict is obeyed — False means the escalation arm is never driven
    (that pure function already pins 'no run when a cheaper arm passed'
    and 'no top-up past the cap')."""
    _assert_orchestrator_accepts("escalation_arm_model")
    gold = _synthetic_gold(tmp_path)
    recorded: list[tuple[Any, Any]] = []

    def refusing_escalation_policy(cheaper_arms, preflight):
        recorded.append((tuple(cheaper_arms), preflight))
        return False

    monkeypatch.setattr("evals.gates.opus_escalation_allowed", refusing_escalation_policy)

    driven: list[str] = []
    payload = harness.run_release_eval(
        gold,
        arm_models=("claude-haiku-4-5", "claude-sonnet-5"),
        deps_factory=_release_deps_factory(gold, refuse_everything=True, driven=driven),
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=FakeBatchClient(),
        mode="fake",
        ledger_path=tmp_path / "spend-ledger.csv",
        journal_dir=None,
        session_id="review-303-escalation-refused",
        escalation_arm_model=OPUS_ARM_MODEL,
    )

    assert len(recorded) == 1, (
        "the escalation decision must go through gates.opus_escalation_allowed "
        "exactly once — never a parallel reimplementation"
    )
    cheaper_arms, preflight = recorded[0]
    assert [arm.model for arm in cheaper_arms] == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert all(isinstance(arm, ArmResult) for arm in cheaper_arms)
    assert hasattr(preflight, "allowed"), (
        "the policy must receive a real BudgetPreflight computed for the "
        "escalation arm from the re-read ledger (finding #236: never a stale "
        "bearer token)"
    )
    assert OPUS_ARM_MODEL not in driven, "a False verdict means the Opus arm never runs"
    assert [arm["model"] for arm in payload["arms"]] == ["claude-haiku-4-5", "claude-sonnet-5"]


def test_release_eval_without_escalation_arm_never_consults_the_policy(tmp_path: Path, monkeypatch):
    """No escalation arm configured -> the policy is never consulted and
    the payload carries exactly the requested arms (the pre-#303 calls
    stay valid: escalation is opt-in)."""

    def poisoned_policy(cheaper_arms, preflight):
        raise AssertionError("no escalation arm was configured")

    monkeypatch.setattr("evals.gates.opus_escalation_allowed", poisoned_policy)
    gold = _synthetic_gold(tmp_path)
    payload = _run_release_eval_fake(gold, tmp_path, refuse_everything=True)
    assert [arm["model"] for arm in payload["arms"]] == [ARM_MODEL]
