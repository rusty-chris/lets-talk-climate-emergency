"""Release-gate arithmetic and the bake-off selection (issue #21).

Everything here is PURE (IMPLEMENTATION.md §4.4): gates apply thresholds
to synthetic-testable run/verdict records; the release verdict is the
conjunction of gate results; the bake-off selection is a pure function
over per-arm gate results + ledger costs. No gate touches the network.

Gate statuses: ``passed`` / ``failed`` / ``blocked`` / ``skipped``.
``blocked`` is a first-class outcome — the severity gate REFUSES to
evaluate (status blocked, never passed) while
evals/gold/severity-audit-packet.md still carries
``owner_severity_audit: pending`` (finding #197, via
evals.severity_audit.assert_owner_severity_audit_complete). The release
verdict reports pending-owner-audit as BLOCKED, never as a pass.

Thresholds (DESIGN §6.2 as amended):
- refusal gate: STRICTLY >90% on the 20-item ``gate`` ∩
  ``retrieval_refusal`` subset (#192/#193) — canned out-of-scope
  declines are the classifier gate's, measured separately, never here;
- false-refusal: STRICTLY <5% on answerable items;
- severity: ≥90% exact-or-adjacent AND zero two-level errors — one
  reassuring↔emergency-level disagreement blocks release regardless of
  the aggregate;
- chart data faithfulness: 100% vs committed fixtures, tolerance 1e-9
  relative pass-through / 1e-6 post-transform; flagship items blocked
  on #23/#117 are skipped-visibly, never counted as passes.

Red phase: contracts pinned, behaviour raises NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import severity_audit

GATE_PASSED = "passed"
GATE_FAILED = "failed"
GATE_BLOCKED = "blocked"
GATE_SKIPPED = "skipped"
GATE_STATUSES = (GATE_PASSED, GATE_FAILED, GATE_BLOCKED, GATE_SKIPPED)

#: The 3-point ordinal lead-severity scale, ordered (evals/gold/severity-rubric.md).
#: Adjacent levels are one step apart; reassuring<->emergency-level is the
#: two-level error that blocks release regardless of the aggregate.
SEVERITY_LEVELS = ("reassuring", "serious", "emergency-level")

REFUSAL_GATE_THRESHOLD = 0.90  # strictly greater-than
FALSE_REFUSAL_THRESHOLD = 0.05  # strictly less-than
SEVERITY_AGREEMENT_THRESHOLD = 0.90  # greater-or-equal, exact-or-adjacent
#: The owner-ratified four-part citation gate (DESIGN §6.2 as amended,
#: issue #325 decision 2026-09-07) — SUPERSEDES the flat 0.95
#: citation_support target. See ``citation_entailment_precision_gate`` etc.
CITATION_ENTAILMENT_PRECISION_THRESHOLD = 0.95  # >= HARD, over ATTACHED factual sentences
UNCITED_FACTUAL_RATE_CEILING = 0.35  # <= HARD ceiling, ratcheted down over time
VERIFIED_CLAIM_GROUP_COVERAGE_THRESHOLD = 0.75  # >= HARD
CHART_TOLERANCE_PASS_THROUGH = 1e-9  # relative
CHART_TOLERANCE_POST_TRANSFORM = 1e-6  # relative


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome with its per-item evidence.

    ``evidence`` carries one record per counted (or skipped/unscored)
    item — the report links every gate number back to items.
    """

    name: str
    status: str
    numerator: int | None = None
    denominator: int | None = None
    threshold: float | None = None
    reason: str | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()


def refusal_gate(
    gate_item_outcomes: Mapping[str, bool],
    *,
    gate_item_ids: Sequence[str],
) -> GateResult:
    """>90% (strict) refusal on the retrieval_refusal gate subset.

    ``gate_item_ids`` comes from gold_selection.gate_item_ids — ONLY
    retrieval_refusal items; passing an outcome for an id outside the
    subset (e.g. a canned_out_of_scope item) is a contract violation
    and raises ValueError. 19/20 (95%) passes; 18/20 (90%) fails the
    strict gate.
    """
    allowed = set(gate_item_ids)
    leaked = set(gate_item_outcomes) - allowed
    if leaked:
        raise ValueError(
            f"refusal_gate received outcomes for ids outside the retrieval_refusal "
            f"gate subset: {sorted(leaked)} — canned_out_of_scope declines are the "
            "classifier gate's, measured separately (finding #192)"
        )
    denominator = len(gate_item_ids)
    numerator = sum(1 for item_id in gate_item_ids if gate_item_outcomes.get(item_id, False))
    rate = numerator / denominator if denominator else 0.0
    status = GATE_PASSED if rate > REFUSAL_GATE_THRESHOLD else GATE_FAILED
    evidence = tuple(
        {"item_id": item_id, "refused": bool(gate_item_outcomes.get(item_id, False))}
        for item_id in gate_item_ids
    )
    return GateResult(
        name="refusal",
        status=status,
        numerator=numerator,
        denominator=denominator,
        threshold=REFUSAL_GATE_THRESHOLD,
        evidence=evidence,
    )


def false_refusal_gate(answerable_outcomes: Mapping[str, bool]) -> GateResult:
    """<5% (strict) false refusals over answerable gold items
    (``refused`` True means the pipeline wrongly refused)."""
    denominator = len(answerable_outcomes)
    numerator = sum(1 for wrongly_refused in answerable_outcomes.values() if wrongly_refused)
    rate = numerator / denominator if denominator else 0.0
    status = GATE_PASSED if rate < FALSE_REFUSAL_THRESHOLD else GATE_FAILED
    evidence = tuple(
        {"item_id": item_id, "false_refusal": bool(wrongly_refused)}
        for item_id, wrongly_refused in answerable_outcomes.items()
    )
    return GateResult(
        name="false_refusal",
        status=status,
        numerator=numerator,
        denominator=denominator,
        threshold=FALSE_REFUSAL_THRESHOLD,
        evidence=evidence,
    )


def canned_out_of_scope_check(canned_outcomes: Mapping[str, bool]) -> GateResult:
    """The classifier-route decline check, measured SEPARATELY from the
    refusal gate (finding #192): every canned_out_of_scope gold item
    must produce the canned decline end-to-end."""
    denominator = len(canned_outcomes)
    numerator = sum(1 for declined in canned_outcomes.values() if declined)
    status = GATE_PASSED if denominator and numerator == denominator else GATE_FAILED
    evidence = tuple(
        {"item_id": item_id, "declined": bool(declined)}
        for item_id, declined in canned_outcomes.items()
    )
    return GateResult(
        name="canned_out_of_scope",
        status=status,
        numerator=numerator,
        denominator=denominator,
        evidence=evidence,
    )


def route_accuracy_gate(classifier_summary: Mapping[str, Any]) -> GateResult:
    """The classifier release gate, consuming the JSON summary of
    evals/scripts/classifier_accuracy.py (issue #10: 100% unsafe
    recall, scope AND subtype): passed iff the summary's
    ``release_gate_passes`` is true; evidence carries the per-class
    accuracy."""
    passes = bool(classifier_summary.get("release_gate_passes"))
    per_class = classifier_summary.get("per_class", {})
    evidence = tuple(
        {"class": class_name, "accuracy": accuracy} for class_name, accuracy in per_class.items()
    )
    return GateResult(
        name="route_accuracy",
        status=GATE_PASSED if passes else GATE_FAILED,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# The owner-ratified four-part citation gate (issue #325, DESIGN §6.2 as
# amended 2026-09-07). All FOUR are recomputed at the GATE LAYER from the
# run's per-sentence validation records ({index, paragraph, factual,
# attached} plus per-pair verdicts) — the #13 validator's segmentation,
# pairing and entailment (and the per-sentence UI chips/badges) are
# UNCHANGED. The flat citation_support gate is SUPERSEDED.
#
# Carried-over semantics: #312 declines are excluded from every part's
# arithmetic while staying visible in the evidence; #239 degraded
# exchanges pool fail-closed (attached with zero supported, groups never
# verified, no entailed citation) and a record that cannot supply
# per-sentence data raises rather than shrinking a pool.
# ---------------------------------------------------------------------------


def _require_sentence_data(
    validation_outcomes: Sequence[Mapping[str, Any]], gate_name: str
) -> None:
    """#239 fail-closed, carried to the #325 schema: a DEGRADED
    (unvalidated) record that carries no per-sentence ``sentences`` data
    cannot feed the gate-layer recompute — it raises (naming the items)
    rather than vanishing from a denominator, forcing a re-run. Declines
    are excluded entirely (exempt); a VALIDATED record without per-sentence
    data simply pools nothing (segmentation runs before the entailment
    call, so a real degraded exchange always carries the data — the guard
    catches the stale-journal / crashed-validator shape)."""
    missing = [
        record.get("item_id")
        for record in validation_outcomes
        if not record.get("generation_decline")
        and not record.get("validated")
        and "sentences" not in record
    ]
    if missing:
        raise ValueError(
            f"{gate_name}: records {missing} carry no per-sentence data; the gate "
            "refuses to let them vanish from the recompute (fail-closed, finding "
            "#239) — supply the segmented sentence records"
        )


def _entailed_sentence_indices(record: Mapping[str, Any]) -> set[Any]:
    """The sentence indices carrying at least one entailment-supported
    verdict (a degraded record has no verdicts, so the set is empty —
    fail-closed)."""
    return {
        verdict.get("sentence_index")
        for verdict in record.get("verdicts", ())
        if verdict.get("supported")
    }


def _decline_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """#312: an excluded decline, kept VISIBLE in every part's evidence."""
    return {"item_id": record.get("item_id"), "generation_decline": True}


def claim_groups(sentences: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, ...], ...]:
    """The ratified claim-group fold (issue #325), PURE over per-sentence
    records ({index, paragraph, factual}): a claim group is a MAXIMAL RUN
    OF CONTIGUOUS FACTUAL SENTENCES IN ONE PARAGRAPH. A non-factual
    sentence breaks the run and joins no group; a paragraph boundary
    splits an otherwise-contiguous run."""
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_paragraph: Any = None
    for sentence in sentences:
        if not sentence.get("factual"):
            if current:
                groups.append(tuple(current))
                current = []
            current_paragraph = None
            continue
        paragraph = sentence.get("paragraph", 0)
        if current and paragraph == current_paragraph:
            current.append(sentence.get("index"))
        else:
            if current:
                groups.append(tuple(current))
            current = [sentence.get("index")]
            current_paragraph = paragraph
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def citation_entailment_precision_gate(
    validation_outcomes: Sequence[Mapping[str, Any]],
) -> GateResult:
    """Part 1a (>= 0.95 HARD): of the ATTACHED factual sentences (a
    citation attached), the fraction with an entailment-supported verdict.
    Degraded attachments pool with ZERO supported (#239); an empty
    denominator (no attached factual sentences) FAILS closed — a run that
    never attaches a citation cannot demonstrate precision."""
    _require_sentence_data(validation_outcomes, "citation_entailment_precision")
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in validation_outcomes:
        if record.get("generation_decline"):
            evidence.append(_decline_evidence(record))
            continue
        entailed = _entailed_sentence_indices(record)
        attached_factual = [
            sentence
            for sentence in (record.get("sentences") or ())
            if sentence.get("factual") and sentence.get("attached")
        ]
        supported = sum(1 for sentence in attached_factual if sentence.get("index") in entailed)
        denominator += len(attached_factual)
        numerator += supported
        entry: dict[str, Any] = {
            "item_id": record.get("item_id"),
            "attached_factual": len(attached_factual),
            "entailed": supported,
        }
        if record.get("degraded_reason"):
            entry["degraded_reason"] = record["degraded_reason"]
        evidence.append(entry)
    rate = numerator / denominator if denominator else 0.0
    status = (
        GATE_PASSED
        if denominator and rate >= CITATION_ENTAILMENT_PRECISION_THRESHOLD
        else GATE_FAILED
    )
    return GateResult(
        name="citation_entailment_precision",
        status=status,
        numerator=numerator,
        denominator=denominator,
        threshold=CITATION_ENTAILMENT_PRECISION_THRESHOLD,
        evidence=tuple(evidence),
    )


def uncited_factual_rate_gate(
    validation_outcomes: Sequence[Mapping[str, Any]],
) -> GateResult:
    """Part 1b (<= 0.35 HARD CEILING): of the pooled factual sentences
    (post-#312/#328 cleaning), the fraction with NO citation attached.
    A CEILING gate passes AT its threshold and fails ABOVE it. Degraded
    exchanges pool their factual sentences fail-closed."""
    _require_sentence_data(validation_outcomes, "uncited_factual_rate")
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in validation_outcomes:
        if record.get("generation_decline"):
            evidence.append(_decline_evidence(record))
            continue
        factual = [
            sentence for sentence in (record.get("sentences") or ()) if sentence.get("factual")
        ]
        uncited = [sentence for sentence in factual if not sentence.get("attached")]
        denominator += len(factual)
        numerator += len(uncited)
        entry: dict[str, Any] = {
            "item_id": record.get("item_id"),
            "factual": len(factual),
            "uncited": len(uncited),
        }
        if record.get("degraded_reason"):
            entry["degraded_reason"] = record["degraded_reason"]
        evidence.append(entry)
    rate = numerator / denominator if denominator else 0.0
    status = GATE_PASSED if rate <= UNCITED_FACTUAL_RATE_CEILING else GATE_FAILED
    return GateResult(
        name="uncited_factual_rate",
        status=status,
        numerator=numerator,
        denominator=denominator,
        threshold=UNCITED_FACTUAL_RATE_CEILING,
        evidence=tuple(evidence),
    )


def verified_claim_group_coverage_gate(
    validation_outcomes: Sequence[Mapping[str, Any]],
) -> GateResult:
    """Part 2 (>= 0.75 HARD): of the run's claim groups (``claim_groups``
    over each record's sentences), the fraction VERIFIED — a group is
    verified iff at least one member sentence carries an ENTAILED citation
    (attachment alone never verifies). Degraded groups are never verified;
    an empty denominator FAILS closed."""
    _require_sentence_data(validation_outcomes, "verified_claim_group_coverage")
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in validation_outcomes:
        if record.get("generation_decline"):
            evidence.append(_decline_evidence(record))
            continue
        entailed = _entailed_sentence_indices(record)
        groups = claim_groups(record.get("sentences") or ())
        verified = sum(1 for group in groups if any(index in entailed for index in group))
        denominator += len(groups)
        numerator += verified
        entry: dict[str, Any] = {
            "item_id": record.get("item_id"),
            "groups": len(groups),
            "verified": verified,
        }
        if record.get("degraded_reason"):
            entry["degraded_reason"] = record["degraded_reason"]
        evidence.append(entry)
    rate = numerator / denominator if denominator else 0.0
    status = (
        GATE_PASSED
        if denominator and rate >= VERIFIED_CLAIM_GROUP_COVERAGE_THRESHOLD
        else GATE_FAILED
    )
    return GateResult(
        name="verified_claim_group_coverage",
        status=status,
        numerator=numerator,
        denominator=denominator,
        threshold=VERIFIED_CLAIM_GROUP_COVERAGE_THRESHOLD,
        evidence=tuple(evidence),
    )


def citation_invariants_gate(
    validation_outcomes: Sequence[Mapping[str, Any]],
    citation_events: Sequence[Mapping[str, Any]],
) -> GateResult:
    """Part 3 (invariants): ZERO zero-width citation spans (a span-carrying
    event whose ``answer_block_start == answer_block_end`` — the #322/#326
    regression class; legacy spanless events carry no extent and are
    tolerated) AND every answered exchange carries >= 1 entailed citation.
    Generation declines are exempt and visible; degraded exchanges fail
    closed (no verdicts => no entailed citation)."""
    evidence: list[Mapping[str, Any]] = []
    zero_width = False
    for event in citation_events:
        start = event.get("answer_block_start")
        end = event.get("answer_block_end")
        if start is not None and end is not None and int(start) == int(end):
            zero_width = True
            evidence.append(
                {
                    "item_id": event.get("item_id"),
                    "zero_width": True,
                    "answer_block_start": start,
                    "answer_block_end": end,
                }
            )
    missing_entailed = False
    for record in validation_outcomes:
        if record.get("generation_decline"):
            evidence.append(_decline_evidence(record))
            continue
        if not _entailed_sentence_indices(record):
            missing_entailed = True
            entry: dict[str, Any] = {"item_id": record.get("item_id"), "entailed_citation": False}
            if record.get("degraded_reason"):
                entry["degraded_reason"] = record["degraded_reason"]
            evidence.append(entry)
    failed = zero_width or missing_entailed
    reasons = []
    if zero_width:
        reasons.append("a zero-width citation span is unattributable to any sentence")
    if missing_entailed:
        reasons.append("an answered exchange carries no entailed citation")
    return GateResult(
        name="citation_invariants",
        status=GATE_FAILED if failed else GATE_PASSED,
        reason="; ".join(reasons) or None,
        evidence=tuple(evidence),
    )


def severity_gate(
    judged_items: Sequence[Mapping[str, Any]],
    *,
    packet_path: Path = severity_audit.PACKET_PATH,
) -> GateResult:
    """≥90% exact-or-adjacent ordinal agreement, zero two-level errors.

    THE OWNER GATE comes first: while the audit packet still says
    ``owner_severity_audit: pending``, this gate returns status
    ``blocked`` (reason naming finding #197) WITHOUT scoring anything —
    unaudited labels never pass or fail a release. Once complete:
    ``judged_items`` records carry ``expected`` and ``judged`` levels
    (reassuring | serious | emergency-level) plus ``scored``; a single
    reassuring↔emergency-level error fails the gate regardless of the
    aggregate; unscored (judge-degraded) items are never agreement.
    """
    try:
        severity_audit.assert_owner_severity_audit_complete(packet_path)
    except severity_audit.SeverityAuditPendingError:
        return GateResult(
            name="severity",
            status=GATE_BLOCKED,
            reason=(
                "owner severity audit pending (finding #197): the release severity "
                "gate refuses to score unaudited labels"
            ),
        )

    level_index = {level: index for index, level in enumerate(SEVERITY_LEVELS)}
    denominator = len(judged_items)
    numerator = 0
    two_level_error = False
    evidence: list[Mapping[str, Any]] = []
    for record in judged_items:
        expected = record.get("expected")
        judged = record.get("judged")
        scored = bool(record.get("scored")) and judged is not None
        agreement = False
        is_two_level = False
        if scored and expected in level_index and judged in level_index:
            distance = abs(level_index[expected] - level_index[judged])
            agreement = distance <= 1
            is_two_level = distance == 2
        if agreement:
            numerator += 1
        if is_two_level:
            two_level_error = True
        evidence.append(
            {
                "item_id": record.get("item_id"),
                "expected": expected,
                "judged": judged,
                "scored": scored,
                "agreement": agreement,
                "two_level_error": is_two_level,
            }
        )
    rate = numerator / denominator if denominator else 0.0
    passes = denominator > 0 and rate >= SEVERITY_AGREEMENT_THRESHOLD and not two_level_error
    reason = (
        "a reassuring<->emergency-level two-level error blocks release" if two_level_error else None
    )
    return GateResult(
        name="severity",
        status=GATE_PASSED if passes else GATE_FAILED,
        numerator=numerator,
        denominator=denominator,
        threshold=SEVERITY_AGREEMENT_THRESHOLD,
        reason=reason,
        evidence=tuple(evidence),
    )


def chart_spec_gate(chart_outcomes: Sequence[Mapping[str, Any]]) -> GateResult:
    """Planned-spec accuracy vs gold specs (deterministic compare);
    skipped_blocked items (flagship, #23/#117) appear in the evidence
    as skipped with their reason and are excluded from the
    denominator — visible, never passes."""
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in chart_outcomes:
        status = record.get("status")
        if status == "skipped_blocked":
            evidence.append(
                {
                    "item_id": record.get("item_id"),
                    "status": "skipped_blocked",
                    "blocked_reason": record.get("blocked_reason"),
                }
            )
            continue
        denominator += 1
        matched = status == "match"
        if matched:
            numerator += 1
        evidence.append({"item_id": record.get("item_id"), "status": status})
    status = GATE_PASSED if denominator and numerator == denominator else GATE_FAILED
    return GateResult(
        name="chart_spec",
        status=status,
        numerator=numerator,
        denominator=denominator,
        evidence=tuple(evidence),
    )


def _within_relative_tolerance(expected: float, actual: float, tolerance: float) -> bool:
    """Relative-error tolerance check, falling back to absolute error when
    the expected value is zero."""
    if expected == 0:
        return abs(actual) <= tolerance
    return abs(actual - expected) / abs(expected) <= tolerance


def chart_faithfulness_gate(chart_value_records: Sequence[Mapping[str, Any]]) -> GateResult:
    """100% rendered-value match vs committed gold fixtures within the
    stated relative tolerances: 1e-9 pass-through, 1e-6 post-transform.
    A 2e-6 post-transform deviation fails (issue #21 TDD plan item 4).
    Records declare ``kind``: pass_through | post_transform."""
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in chart_value_records:
        kind = record.get("kind")
        # #245: a CLOSED vocabulary. A misspelled/absent kind must never
        # silently take the 1000x-looser post-transform tolerance.
        if kind not in ("pass_through", "post_transform"):
            raise ValueError(
                f"chart_faithfulness_gate: item {record.get('item_id')!r} carries kind "
                f"{kind!r}; expected one of ('pass_through', 'post_transform') — an "
                "unknown kind must not fall into the looser tolerance (finding #245)"
            )
        tolerance = (
            CHART_TOLERANCE_PASS_THROUGH
            if kind == "pass_through"
            else CHART_TOLERANCE_POST_TRANSFORM
        )
        expected = float(record.get("expected", 0.0))
        actual = float(record.get("actual", 0.0))
        matched = _within_relative_tolerance(expected, actual, tolerance)
        denominator += 1
        if matched:
            numerator += 1
        evidence.append(
            {
                "item_id": record.get("item_id"),
                "kind": kind,
                "expected": expected,
                "actual": actual,
                "matched": matched,
            }
        )
    status = GATE_PASSED if denominator and numerator == denominator else GATE_FAILED
    return GateResult(
        name="chart_faithfulness",
        status=status,
        numerator=numerator,
        denominator=denominator,
        evidence=tuple(evidence),
    )


def chart_refusal_gate(chart_outcomes: Sequence[Mapping[str, Any]]) -> GateResult:
    """Correct refusal + nearest-dataset suggestion on the
    unavailable-data chart golds (deterministic; finding #194
    sub-schema)."""
    numerator = 0
    denominator = 0
    evidence: list[Mapping[str, Any]] = []
    for record in chart_outcomes:
        status = record.get("status")
        denominator += 1
        correct = status == "refused_with_nearest"
        if correct:
            numerator += 1
        evidence.append({"item_id": record.get("item_id"), "status": status})
    status = GATE_PASSED if denominator and numerator == denominator else GATE_FAILED
    return GateResult(
        name="chart_refusal",
        status=status,
        numerator=numerator,
        denominator=denominator,
        evidence=tuple(evidence),
    )


def voices_separation_gate(violations: Sequence[Mapping[str, Any]]) -> GateResult:
    """100% separation: any violation record fails the gate."""
    violations = tuple(violations)
    return GateResult(
        name="voices_separation",
        status=GATE_PASSED if not violations else GATE_FAILED,
        numerator=len(violations),
        denominator=len(violations),
        evidence=violations,
    )


def release_verdict(gates: Sequence[GateResult]) -> str:
    """The conjunction: ``failed`` if any gate failed; else ``blocked``
    if any gate is blocked (pending owner audit is BLOCKED, not
    passed); else ``passed``. An empty gate list is a contract
    violation (ValueError) — a release cannot pass on zero gates."""
    if not gates:
        raise ValueError(
            "release_verdict requires at least one gate; a release cannot pass on zero"
        )
    # #240: every status is a member of the closed vocabulary — a typo
    # ('pased') must raise, never silently convert failures into passes.
    for gate in gates:
        if gate.status not in GATE_STATUSES:
            raise ValueError(
                f"release_verdict: gate {gate.name!r} has unknown status {gate.status!r}; "
                f"expected one of {GATE_STATUSES} — a misspelled status must never pass"
            )
    statuses = {gate.status for gate in gates}
    if GATE_FAILED in statuses:
        return "failed"
    # #240: a SKIPPED gate means work remains before release — it maps to
    # blocked (never passed), so arm_passes and release_verdict agree.
    if GATE_BLOCKED in statuses or GATE_SKIPPED in statuses:
        return "blocked"
    return "passed"


# ---------------------------------------------------------------------------
# Model bake-off (issue #21 orchestrator comments)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    """One bake-off arm: its model, its gate results, and its measured
    cost (from the ledger rows the run appended).

    Review #317: an arm the affordability projection refused before it
    spent anything carries ``arm_verdict='dnf-unaffordable'`` (with a
    ``reason``) and NO gates — a reported outcome, never a silent
    disappearance, and never a pass (``arm_passes`` needs gates)."""

    model: str
    gates: tuple[GateResult, ...]
    cost_usd: float
    arm_verdict: str | None = None
    reason: str | None = None


def arm_passes(arm: ArmResult) -> bool:
    """True only when EVERY gate is ``passed`` — blocked or skipped
    gates never count as passing an arm."""
    return bool(arm.gates) and all(gate.status == GATE_PASSED for gate in arm.gates)


def select_production_model(arms: Sequence[ArmResult]) -> str | None:
    """Cheapest-passing-wins: among arms where arm_passes, the one with
    the lowest cost_usd (per-query cost proxy); None when no arm passes
    — the harness then escalates to the client with the numbers rather
    than picking one."""
    passing = [arm for arm in arms if arm_passes(arm)]
    if not passing:
        return None
    return min(passing, key=lambda arm: arm.cost_usd).model


def opus_escalation_allowed(
    cheaper_arms: Sequence[ArmResult],
    preflight: Any,
) -> bool:
    """The Opus arm is escalation-only (ratified: NO top-up): allowed
    only when (a) no cheaper arm passed all gates, (b) some cheaper arm
    actually FAILED a gate, AND (c) the Opus run's budget pre-flight is
    allowed within the REMAINING budget under the $9.00 cap. Never
    allowed merely to compare.

    Escalation triggers on a model-capability FAILURE only (orchestrator
    ratification of #303, decision 4): a gate that is merely BLOCKED —
    the owner-pending severity audit — is not something a stronger model
    can fix, so a battery whose only non-pass is a BLOCKED gate never
    escalates. A FAILED gate is the honest capability signal."""
    if any(arm_passes(arm) for arm in cheaper_arms):
        return False
    failed = any(any(gate.status == GATE_FAILED for gate in arm.gates) for arm in cheaper_arms)
    if not failed:
        return False
    return bool(getattr(preflight, "allowed", False))
