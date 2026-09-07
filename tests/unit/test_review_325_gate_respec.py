"""Issue #325 red phase (Fable): the OWNER-RATIFIED four-part citation
gate replaces the flat 0.95 citation_support gate.

OWNER DECISION (issue #325, comment recorded 2026-09-07, option (b)):
the release citation gate becomes FOUR distinct, separately-reported
parts, all recomputed at the GATE layer from the run's per-sentence
validation records — validator segmentation / pairing / entailment and
the per-sentence UI chips/badges are UNCHANGED:

1. ``citation_entailment_precision`` >= 0.95 HARD — denominator: the
   ATTACHED factual sentences (sentences a citation attached to);
   numerator: those with at least one entailment-supported verdict.
   (Resmoke measured 169/175 = 96.6% on Haiku.)
2. ``uncited_factual_rate`` <= 0.35 HARD ceiling — denominator: the
   pooled factual sentences (post-#312 cleaning); numerator: factual
   sentences NO citation attached to. (Resmoke measured 87/262 = 33.2%.)
3. ``verified_claim_group_coverage`` >= 0.75 HARD — a claim group is a
   MAXIMAL RUN OF CONTIGUOUS FACTUAL SENTENCES IN ONE PARAGRAPH (the
   ratified definition); a group is verified iff >=1 member sentence has
   an entailed citation. (Resmoke measured 80.2%.)
4. ``citation_invariants`` — zero zero-width spans across the run's
   citation events AND every answered exchange carries >=1 entailed
   citation.

Carried-over semantics, pinned here so the re-spec cannot shed them:
- #312: ``generation_decline`` records are EXCLUDED from every part's
  arithmetic while staying visible in the evidence (the refusal gate is
  that item's scorer). The honest-decline shape (zero citations) must
  therefore never fail the >=1-entailed-citation invariant.
- #239 fail-closed: degraded/unvalidated exchanges pool their sentences
  against release (attached sentences count in the precision denominator
  with ZERO supported; their groups are never verified; the exchange
  fails the entailed-citation invariant); a record that cannot supply
  per-sentence data raises, it never vanishes from a denominator.
- #303 BLOCKED-when-unmeasured: a run with no validation records reports
  ALL FOUR parts blocked — never passed, never absent.

RECORD-SCHEMA ADDITION (flagged, pinned here): today's journalled
validation record carries only aggregate counts plus per-pair verdicts —
per-sentence factual/attached flags and PARAGRAPH boundaries are not
recoverable from it, so the claim-group fold cannot run on the journal.
The record gains a ``sentences`` field — one entry per segmented
sentence: ``{index, paragraph, factual, attached}`` — derived by the
harness from the #13 outcome, with ``paragraph`` stamped on
``AnswerSentence`` by ``segment_answer_sentences`` from the delivered
text's blank-line paragraph boundaries (metadata enrichment only:
sentence texts, factual classification, attachment and pairing are
byte-identical to today, pinned below).

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.harness import AnswerPathDeps, ItemResult, run_answer_path
from rag.citation_validator import ValidationOutcome, segment_answer_sentences
from rag.generation import answer_stream_to_sse
from rag.provider import FakeAdapter
from rag.retrieval import RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    production_passage_payload,
    write_synthetic_gold,
)

ARM_MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Record builders (the gate-layer feed: validation records with the new
# per-sentence ``sentences`` field)
# ---------------------------------------------------------------------------


def _sentence(index: int, *, paragraph: int = 0, factual: bool = True, attached: bool = False):
    return {"index": index, "paragraph": paragraph, "factual": factual, "attached": attached}


def _record(
    item_id: str,
    sentences: list[dict[str, Any]],
    *,
    entailed: tuple[int, ...] = (),
    unentailed: tuple[int, ...] = (),
    validated: bool = True,
    degraded_reason: str | None = None,
    generation_decline: bool = False,
) -> dict[str, Any]:
    """One gate-layer validation record: per-sentence data plus per-pair
    verdicts (supported for ``entailed`` sentence indices, unsupported
    for ``unentailed`` ones — both must be attached sentences)."""
    verdicts = []
    for sentence_index in (*entailed, *unentailed):
        verdicts.append(
            {
                "pair_index": len(verdicts),
                "sentence_index": sentence_index,
                "document_index": 0,
                "supported": sentence_index in entailed,
            }
        )
    factual = sum(1 for sentence in sentences if sentence["factual"])
    record: dict[str, Any] = {
        "item_id": item_id,
        "validated": validated,
        "supported": len(entailed),
        "factual": factual,
        "sentences": sentences,
        "verdicts": verdicts,
    }
    if degraded_reason is not None:
        record["degraded_reason"] = degraded_reason
        record["validated"] = False
        record["supported"] = 0
        record["verdicts"] = []
    if generation_decline:
        record["generation_decline"] = True
    return record


def _uniform_record(
    item_id: str,
    *,
    factual: int,
    attached: int,
    entailed: int,
    paragraph: int = 0,
) -> dict[str, Any]:
    """``factual`` contiguous one-paragraph factual sentences of which the
    first ``attached`` are attached and the first ``entailed`` of those
    are entailment-supported (the rest of the attached get unsupported
    verdicts)."""
    sentences = [
        _sentence(index, paragraph=paragraph, attached=index < attached) for index in range(factual)
    ]
    return _record(
        item_id,
        sentences,
        entailed=tuple(range(entailed)),
        unentailed=tuple(range(entailed, attached)),
    )


def _gate_functions():
    """The four ratified gate functions (red: they do not exist yet)."""
    from evals import gates

    return (
        gates.citation_entailment_precision_gate,
        gates.uncited_factual_rate_gate,
        gates.verified_claim_group_coverage_gate,
        gates.citation_invariants_gate,
    )


# ---------------------------------------------------------------------------
# Part 1a — citation_entailment_precision: >=0.95 over ATTACHED factual
# sentences, boundary arithmetic pinned in both directions.
# ---------------------------------------------------------------------------


class TestCitationEntailmentPrecisionGate:
    def test_threshold_constant_is_ratified(self):
        from evals.gates import CITATION_ENTAILMENT_PRECISION_THRESHOLD

        assert CITATION_ENTAILMENT_PRECISION_THRESHOLD == 0.95

    def test_passes_at_exactly_the_threshold(self):
        """19 entailed of 20 attached factual sentences = 0.95 exactly —
        a greater-or-equal gate PASSES at its threshold."""
        from evals.gates import GATE_PASSED, citation_entailment_precision_gate

        result = citation_entailment_precision_gate(
            [_uniform_record("syn-a", factual=20, attached=20, entailed=19)]
        )
        assert result.name == "citation_entailment_precision"
        assert (result.numerator, result.denominator) == (19, 20)
        assert result.threshold == 0.95
        assert result.status == GATE_PASSED

    def test_fails_below_the_threshold(self):
        from evals.gates import GATE_FAILED, citation_entailment_precision_gate

        result = citation_entailment_precision_gate(
            [_uniform_record("syn-a", factual=20, attached=20, entailed=18)]
        )
        assert (result.numerator, result.denominator) == (18, 20)
        assert result.status == GATE_FAILED

    def test_denominator_is_attached_factual_sentences_only(self):
        """Uncited factual sentences belong to part 2's pool, NOT this
        denominator — that split is the whole point of the re-spec (the
        old gate mixed 'bad citation' and 'no citation' in one number)."""
        from evals.gates import GATE_PASSED, citation_entailment_precision_gate

        # 10 factual sentences, only 4 attached, all 4 entailed: 4/4.
        result = citation_entailment_precision_gate(
            [_uniform_record("syn-a", factual=10, attached=4, entailed=4)]
        )
        assert (result.numerator, result.denominator) == (4, 4)
        assert result.status == GATE_PASSED

    def test_degraded_attached_sentences_pool_with_zero_supported(self):
        """#239 carried over: a degraded exchange's attached factual
        sentences count in the denominator with ZERO in the numerator —
        19 entailed of 19+2 attached = 0.9048 < 0.95 fails."""
        from evals.gates import GATE_FAILED, citation_entailment_precision_gate

        degraded = _record(
            "syn-deg",
            [_sentence(0, attached=True), _sentence(1, attached=True)],
            degraded_reason="validator call failed",
        )
        result = citation_entailment_precision_gate(
            [_uniform_record("syn-a", factual=19, attached=19, entailed=19), degraded]
        )
        assert (result.numerator, result.denominator) == (19, 21)
        assert result.status == GATE_FAILED

    def test_empty_denominator_fails_closed(self):
        """A run with NO attached factual sentences cannot demonstrate
        precision: 0/0 is FAILED, never a vacuous pass."""
        from evals.gates import GATE_FAILED, citation_entailment_precision_gate

        result = citation_entailment_precision_gate(
            [_uniform_record("syn-a", factual=3, attached=0, entailed=0)]
        )
        assert result.status == GATE_FAILED


# ---------------------------------------------------------------------------
# Part 1b — uncited_factual_rate: <=0.35 ceiling over the pooled factual
# sentences (post-#312 cleaning), boundary pinned in both directions.
# ---------------------------------------------------------------------------


class TestUncitedFactualRateGate:
    def test_ceiling_constant_is_ratified(self):
        from evals.gates import UNCITED_FACTUAL_RATE_CEILING

        assert UNCITED_FACTUAL_RATE_CEILING == 0.35

    def test_passes_at_exactly_the_ceiling(self):
        """7 uncited of 20 pooled factual sentences = 0.35 exactly — a
        CEILING gate passes AT its threshold and fails ABOVE it (the
        opposite direction from the two >= gates; the report must not
        render this one as if bigger were better)."""
        from evals.gates import GATE_PASSED, uncited_factual_rate_gate

        result = uncited_factual_rate_gate(
            [_uniform_record("syn-a", factual=20, attached=13, entailed=13)]
        )
        assert result.name == "uncited_factual_rate"
        assert (result.numerator, result.denominator) == (7, 20)
        assert result.threshold == 0.35
        assert result.status == GATE_PASSED

    def test_fails_above_the_ceiling(self):
        from evals.gates import GATE_FAILED, uncited_factual_rate_gate

        result = uncited_factual_rate_gate(
            [_uniform_record("syn-a", factual=20, attached=12, entailed=12)]
        )
        assert (result.numerator, result.denominator) == (8, 20)
        assert result.status == GATE_FAILED

    def test_non_factual_sentences_never_pool(self):
        """The post-#312/#328 cleaning IS the denominator definition:
        non-factual sentences (furniture, passage-meta, referral, stance)
        contribute to neither side."""
        from evals.gates import uncited_factual_rate_gate

        sentences = [
            _sentence(0, attached=True),
            _sentence(1, factual=False),
            _sentence(2),
            _sentence(3, factual=False),
        ]
        result = uncited_factual_rate_gate([_record("syn-a", sentences, entailed=(0,))])
        assert (result.numerator, result.denominator) == (1, 2)

    def test_degraded_record_without_sentence_data_raises(self):
        """#239's fail-closed guard survives the re-spec: a record that
        carries no per-sentence data cannot feed the recompute — it
        raises naming the item (a stale-journal or crashed-validator
        record must force a re-run, never silently shrink a pool)."""
        from evals.gates import uncited_factual_rate_gate

        records = [
            _uniform_record("syn-ok", factual=20, attached=20, entailed=20),
            {
                "item_id": "syn-deg-01",
                "validated": False,
                "degraded_reason": "validator crashed before sentence segmentation",
            },
        ]
        with pytest.raises(ValueError) as excinfo:
            uncited_factual_rate_gate(records)
        assert "syn-deg-01" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Part 2 — the claim-group fold (pure) and its coverage gate.
# ---------------------------------------------------------------------------


class TestClaimGroupFold:
    """``claim_groups`` is a PURE function over per-sentence records
    ({index, paragraph, factual}): the ratified definition — a claim
    group is a maximal run of contiguous factual sentences in one
    paragraph."""

    def test_non_factual_sentence_splits_a_run_and_joins_no_group(self):
        from evals.gates import claim_groups

        sentences = [
            _sentence(0),
            _sentence(1),
            _sentence(2, factual=False),
            _sentence(3),
        ]
        assert claim_groups(sentences) == ((0, 1), (3,))

    def test_paragraph_boundary_splits_contiguous_indices(self):
        """Sentence 3 follows sentence 2 immediately, but in a new
        paragraph — the ratified unit is per-paragraph, so the run
        splits (the resmoke fold measured 80.2% on exactly this rule)."""
        from evals.gates import claim_groups

        sentences = [
            _sentence(0, paragraph=0),
            _sentence(1, paragraph=0),
            _sentence(2, paragraph=0),
            _sentence(3, paragraph=1),
            _sentence(4, paragraph=1),
        ]
        assert claim_groups(sentences) == ((0, 1, 2), (3, 4))

    def test_single_sentence_groups_and_empty_input(self):
        from evals.gates import claim_groups

        assert claim_groups([_sentence(0)]) == ((0,),)
        assert claim_groups([]) == ()
        # A furniture-only answer folds to zero groups.
        assert claim_groups([_sentence(0, factual=False)]) == ()


class TestVerifiedClaimGroupCoverageGate:
    def test_threshold_constant_is_ratified(self):
        from evals.gates import VERIFIED_CLAIM_GROUP_COVERAGE_THRESHOLD

        assert VERIFIED_CLAIM_GROUP_COVERAGE_THRESHOLD == 0.75

    @staticmethod
    def _four_group_record(verified_groups: int) -> dict[str, Any]:
        """Four two-sentence groups (one per paragraph); the first
        ``verified_groups`` have their FIRST member attached+entailed —
        the second member stays uncited, pinning 'a group is verified if
        >=1 member sentence has an entailed citation'."""
        sentences = []
        entailed = []
        for group in range(4):
            first = 2 * group
            sentences.append(_sentence(first, paragraph=group, attached=group < verified_groups))
            sentences.append(_sentence(first + 1, paragraph=group))
            if group < verified_groups:
                entailed.append(first)
        return _record("syn-a", sentences, entailed=tuple(entailed))

    def test_passes_at_exactly_the_threshold(self):
        from evals.gates import GATE_PASSED, verified_claim_group_coverage_gate

        result = verified_claim_group_coverage_gate([self._four_group_record(3)])
        assert result.name == "verified_claim_group_coverage"
        assert (result.numerator, result.denominator) == (3, 4)
        assert result.threshold == 0.75
        assert result.status == GATE_PASSED

    def test_fails_below_the_threshold(self):
        from evals.gates import GATE_FAILED, verified_claim_group_coverage_gate

        result = verified_claim_group_coverage_gate([self._four_group_record(2)])
        assert (result.numerator, result.denominator) == (2, 4)
        assert result.status == GATE_FAILED

    def test_attachment_without_entailment_never_verifies_a_group(self):
        """The ratified rule says ENTAILED citation — an attached-but-
        unsupported verdict is precision's failure, not this gate's
        rescue."""
        from evals.gates import verified_claim_group_coverage_gate

        record = _record(
            "syn-a",
            [
                _sentence(0, paragraph=0, attached=True),
                _sentence(1, paragraph=1, attached=True),
            ],
            entailed=(1,),
            unentailed=(0,),
        )
        result = verified_claim_group_coverage_gate([record])
        assert (result.numerator, result.denominator) == (1, 2)

    def test_degraded_record_groups_are_never_verified(self):
        """Fail-closed: without verdicts nothing is entailed, so a
        degraded exchange's groups all count unverified."""
        from evals.gates import GATE_FAILED, verified_claim_group_coverage_gate

        degraded = _record(
            "syn-deg",
            [_sentence(0, attached=True), _sentence(1)],
            degraded_reason="validator call failed",
        )
        result = verified_claim_group_coverage_gate([degraded])
        assert (result.numerator, result.denominator) == (0, 1)
        assert result.status == GATE_FAILED


# ---------------------------------------------------------------------------
# Part 3 — citation_invariants: zero zero-width spans; every answered
# exchange >=1 entailed citation.
# ---------------------------------------------------------------------------


class TestCitationInvariantsGate:
    @staticmethod
    def _healthy_records():
        return [_uniform_record("syn-a", factual=3, attached=2, entailed=2)]

    def test_passes_on_span_carrying_events_and_entailed_exchanges(self):
        from evals.gates import GATE_PASSED, citation_invariants_gate

        events = [
            {"item_id": "syn-a", "answer_block_start": 0, "answer_block_end": 62},
            {"item_id": "syn-a", "answer_block_start": 63, "answer_block_end": 120},
        ]
        result = citation_invariants_gate(self._healthy_records(), events)
        assert result.name == "citation_invariants"
        assert result.status == GATE_PASSED

    def test_a_single_zero_width_span_fails_and_names_the_item(self):
        """The #322/#326 regression class: a citation event whose block
        extent is empty (start == end) is unattributable to any sentence.
        The resmoke MUST-be-0 condition (0/162, 0/75 measured) is now a
        release invariant."""
        from evals.gates import GATE_FAILED, citation_invariants_gate

        events = [
            {"item_id": "syn-a", "answer_block_start": 0, "answer_block_end": 62},
            {"item_id": "syn-a", "answer_block_start": 62, "answer_block_end": 62},
        ]
        result = citation_invariants_gate(self._healthy_records(), events)
        assert result.status == GATE_FAILED
        assert any(
            entry.get("item_id") == "syn-a" and entry.get("zero_width") for entry in result.evidence
        ), "the zero-width event must be named in the evidence"

    def test_legacy_spanless_events_are_not_zero_width(self):
        """A citation journalled before #310/#326 carries no block extent
        at all — that is the tolerated legacy shape (it falls back to
        last-text-char attachment), NOT an empty extent; only start==end
        trips the invariant. (Documented test-author reading of the
        ratified 'zero zero-width spans': the resmoke counted spans.)"""
        from evals.gates import GATE_PASSED, citation_invariants_gate

        events = [{"item_id": "syn-a", "document_index": 0}]
        result = citation_invariants_gate(self._healthy_records(), events)
        assert result.status == GATE_PASSED

    def test_answered_exchange_without_an_entailed_citation_fails(self):
        from evals.gates import GATE_FAILED, citation_invariants_gate

        no_entailment = _record(
            "syn-bare",
            [_sentence(0, attached=True)],
            unentailed=(0,),
        )
        result = citation_invariants_gate([*self._healthy_records(), no_entailment], [])
        assert result.status == GATE_FAILED
        assert any(entry.get("item_id") == "syn-bare" for entry in result.evidence), (
            "the exchange with no entailed citation must be named"
        )

    def test_degraded_exchange_fails_the_entailed_citation_invariant(self):
        """Fail-closed: an unvalidated exchange cannot demonstrate an
        entailed citation."""
        from evals.gates import GATE_FAILED, citation_invariants_gate

        degraded = _record(
            "syn-deg",
            [_sentence(0, attached=True)],
            degraded_reason="validator call failed",
        )
        result = citation_invariants_gate([degraded], [])
        assert result.status == GATE_FAILED

    def test_generation_declines_are_exempt_and_visible(self):
        """#312/#313: an honest decline is an AUTHORITATIVE refusal with
        zero citations BY CONSTRUCTION — the >=1-entailed-citation
        invariant must not fail every honest decline (that would make
        honesty a release blocker). Excluded from the arithmetic, visible
        in the evidence."""
        from evals.gates import GATE_PASSED, citation_invariants_gate

        decline = _record(
            "qa-na-g-05",
            [_sentence(0, factual=False), _sentence(1, factual=False)],
            generation_decline=True,
        )
        result = citation_invariants_gate([*self._healthy_records(), decline], [])
        assert result.status == GATE_PASSED
        assert any(
            entry.get("item_id") == "qa-na-g-05" and entry.get("generation_decline")
            for entry in result.evidence
        )


# ---------------------------------------------------------------------------
# #312 decline exclusion carried over to ALL parts' arithmetic.
# ---------------------------------------------------------------------------


def test_generation_decline_records_are_excluded_from_every_part():
    """A decline record must not perturb any of the four numbers, and
    stays visible in each part's evidence (never double-counted with the
    refusal gate; never silently dropped)."""
    precision_gate, uncited_gate, coverage_gate, invariants_gate = _gate_functions()

    healthy = _uniform_record("syn-ok", factual=4, attached=3, entailed=3)
    decline = _record(
        "qa-na-g-05",
        [_sentence(0, factual=False), _sentence(1, factual=False), _sentence(2)],
        generation_decline=True,
    )
    records = [healthy, decline]

    precision = precision_gate(records)
    assert (precision.numerator, precision.denominator) == (3, 3)
    uncited = uncited_gate(records)
    assert (uncited.numerator, uncited.denominator) == (1, 4), (
        "the decline's sentences (even a factual-looking one) must not pool"
    )
    coverage = coverage_gate(records)
    assert (coverage.numerator, coverage.denominator) == (1, 1)

    for result in (precision, uncited, coverage, invariants_gate(records, [])):
        assert any(
            entry.get("item_id") == "qa-na-g-05" and entry.get("generation_decline")
            for entry in result.evidence
        ), f"{result.name}: the excluded decline must stay visible in the evidence"


# ---------------------------------------------------------------------------
# The resmoke-shaped fixture: the measured arithmetic shape (uncited
# ~1/3, cited clusters carrying uncited members, coverage ~0.8) passes
# all four parts — the spec was ratified BECAUSE Haiku measures this way.
# ---------------------------------------------------------------------------


def _resmoke_shaped_records() -> list[dict[str, Any]]:
    """Two answered items shaped like the resmoke journals: pooled
    factual 12, attached 8 (all entailed), uncited 4 (rate 0.333),
    5 claim groups of which 4 verified (0.8)."""
    item_a = _record(
        "qa-mp-01",
        [
            # p0: cited cluster containing uncited members -> verified.
            _sentence(0, paragraph=0, attached=True),
            _sentence(1, paragraph=0, attached=True),
            _sentence(2, paragraph=0),
            # p0: a stance sentence splits the run (#328's class)...
            _sentence(3, paragraph=0, factual=False),
            # ...leaving a verified single-sentence group.
            _sentence(4, paragraph=0, attached=True),
            # p1: a wholly-uncited group -> unverified.
            _sentence(5, paragraph=1),
            _sentence(6, paragraph=1),
            # p2: a verified two-sentence group.
            _sentence(7, paragraph=2, attached=True),
            _sentence(8, paragraph=2, attached=True),
        ],
        entailed=(0, 1, 4, 7, 8),
    )
    item_b = _record(
        "qa-sp-01",
        [
            _sentence(0, attached=True),
            _sentence(1, attached=True),
            _sentence(2, attached=True),
            _sentence(3),
        ],
        entailed=(0, 1, 2),
    )
    return [item_a, item_b]


def _span_events(item_id: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item_id,
            "answer_block_start": 100 * index,
            "answer_block_end": 100 * index + 80,
        }
        for index in range(count)
    ]


def test_resmoke_shaped_run_passes_all_four_parts():
    from evals.gates import GATE_PASSED

    precision_gate, uncited_gate, coverage_gate, invariants_gate = _gate_functions()
    records = _resmoke_shaped_records()
    events = [*_span_events("qa-mp-01", 5), *_span_events("qa-sp-01", 3)]

    precision = precision_gate(records)
    assert (precision.numerator, precision.denominator) == (8, 8)
    uncited = uncited_gate(records)
    assert (uncited.numerator, uncited.denominator) == (4, 12)  # 0.333 <= 0.35
    coverage = coverage_gate(records)
    assert (coverage.numerator, coverage.denominator) == (4, 5)  # 0.8 >= 0.75
    invariants = invariants_gate(records, events)
    for result in (precision, uncited, coverage, invariants):
        assert result.status == GATE_PASSED, f"{result.name}: {result.status}"


def test_release_verdict_conjunction_includes_all_four():
    """The four parts are FOUR gates in the verdict conjunction: a feed
    failing ONLY claim-group coverage (precision 6/6, uncited 2/8 = 0.25,
    but one of two groups wholly uncited) must fail the release."""
    from evals.gates import GATE_FAILED, GATE_PASSED, release_verdict

    precision_gate, uncited_gate, coverage_gate, invariants_gate = _gate_functions()
    record = _record(
        "syn-a",
        [
            *[_sentence(index, paragraph=0, attached=True) for index in range(6)],
            _sentence(6, paragraph=1),
            _sentence(7, paragraph=1),
        ],
        entailed=(0, 1, 2, 3, 4, 5),
    )
    battery = [
        precision_gate([record]),
        uncited_gate([record]),
        coverage_gate([record]),
        invariants_gate([record], _span_events("syn-a", 6)),
    ]
    statuses = {gate.name: gate.status for gate in battery}
    assert statuses == {
        "citation_entailment_precision": GATE_PASSED,
        "uncited_factual_rate": GATE_PASSED,
        "verified_claim_group_coverage": GATE_FAILED,
        "citation_invariants": GATE_PASSED,
    }
    assert release_verdict(battery) == "failed"


def test_all_four_parts_report_distinctly_in_payload_and_results_md():
    """Four rows, four names, four scores — the report must never fold
    the parts back into one opaque number (measurement honesty is the
    point of the re-spec)."""
    from evals.gates import ArmResult
    from evals.report import build_results_payload, render_results_md

    precision_gate, uncited_gate, coverage_gate, invariants_gate = _gate_functions()
    records = _resmoke_shaped_records()
    events = [*_span_events("qa-mp-01", 5), *_span_events("qa-sp-01", 3)]
    arm = ArmResult(
        model=ARM_MODEL,
        gates=(
            precision_gate(records),
            uncited_gate(records),
            coverage_gate(records),
            invariants_gate(records, events),
        ),
        cost_usd=0.0,
    )
    payload = build_results_payload([arm], verdict="passed", selected_model=ARM_MODEL)
    names = [gate["name"] for gate in payload["arms"][0]["gates"]]
    assert names == [
        "citation_entailment_precision",
        "uncited_factual_rate",
        "verified_claim_group_coverage",
        "citation_invariants",
    ]
    rendered = render_results_md(payload)
    assert "| citation_entailment_precision | PASSED | 8/8 |" in rendered
    assert "| uncited_factual_rate | PASSED | 4/12 |" in rendered
    assert "| verified_claim_group_coverage | PASSED | 4/5 |" in rendered
    assert "| citation_invariants | PASSED |" in rendered


# ---------------------------------------------------------------------------
# Segmentation metadata: AnswerSentence.paragraph (the ONLY validator-side
# edit — enrichment; segmentation behaviour byte-identical).
# ---------------------------------------------------------------------------


TWO_PARAGRAPH_TEXT = (
    "Warming reached 1.2C by 2024. Sea level rose 20cm over the same period.\n\n"
    "Heatwaves have become more frequent since the 1980s."
)


class TestParagraphStamping:
    def test_sentences_carry_their_paragraph_index(self):
        sentences = segment_answer_sentences(
            [{"event": "text", "data": {"text": TWO_PARAGRAPH_TEXT}}]
        )
        assert [sentence.paragraph for sentence in sentences] == [0, 0, 1], (
            "blank-line-delimited paragraphs of the DELIVERED text are the "
            "ratified claim-group boundary; the segmenter must stamp them"
        )

    def test_segmentation_behaviour_is_otherwise_unchanged(self):
        """The owner decision holds segmentation/pairing/entailment fixed:
        texts, order, factual flags and attachment must be exactly what
        today's rule produces for the same delivery."""
        events = [
            {"event": "text", "data": {"text": TWO_PARAGRAPH_TEXT}},
            {
                "event": "citation",
                "data": {"document_index": 0, "answer_block_start": 0, "answer_block_end": 28},
            },
        ]
        sentences = segment_answer_sentences(events)
        assert [sentence.text for sentence in sentences] == [
            "Warming reached 1.2C by 2024.",
            "Sea level rose 20cm over the same period.",
            "Heatwaves have become more frequent since the 1980s.",
        ]
        assert [sentence.factual for sentence in sentences] == [True, True, True]
        assert [sentence.document_indices for sentence in sentences] == [(0,), (), ()]

    def test_single_paragraph_answers_are_paragraph_zero(self):
        sentences = segment_answer_sentences(
            [{"event": "text", "data": {"text": "One fact. Another fact."}}]
        )
        assert [sentence.paragraph for sentence in sentences] == [0, 0]


# ---------------------------------------------------------------------------
# The harness record schema: run_answer_path journals per-sentence data
# (the claim-group fold's feed) for validated AND degraded outcomes.
# ---------------------------------------------------------------------------


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

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic planet warming",
}

#: A block-structured transport delivery: block 0 is one cited sentence,
#: block 1 delivers the rest of paragraph 0 plus paragraph 1 uncited.
BLOCK_STRUCTURED_STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
    {"type": "content_block_start", "index": 0},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Fact one is stated here."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "citations_delta",
            "citation": {"cited_text": "synthetic passage", "document_index": 0},
        },
    },
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1},
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {
            "type": "text_delta",
            "text": " Fact two is uncited.\n\nFact three carries no citation.",
        },
    },
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 24}},
    {"type": "message_stop"},
]


def _run_with_validator(validator):
    adapter = FakeAdapter(
        generate_stream_results=[list(BLOCK_STRUCTURED_STREAM)],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: PASSAGES, validate_exchange=validator
    )
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, mode="fake")
    return result


EXPECTED_SENTENCE_RECORDS = [
    {"index": 0, "paragraph": 0, "factual": True, "attached": True},
    {"index": 1, "paragraph": 0, "factual": True, "attached": False},
    {"index": 2, "paragraph": 1, "factual": True, "attached": False},
]


def test_validation_record_carries_per_sentence_data():
    """The journalled record gains ``sentences`` — {index, paragraph,
    factual, attached} per segmented sentence — so the claim-group fold
    (and a future offline re-audit) runs from artifacts alone."""

    def validator(grounded, sse_events):
        return ValidationOutcome(
            validated=True,
            sentences=segment_answer_sentences(sse_events),
            verdicts=(),
        )

    result = _run_with_validator(validator)
    assert result.validation is not None
    assert result.validation.get("sentences") == EXPECTED_SENTENCE_RECORDS


def test_degraded_validation_record_still_carries_per_sentence_data():
    """#239's rationale extends to the new field: segmentation happens
    before the entailment call, so even a degraded outcome journals its
    per-sentence data — the fold and pools never lose a degraded item."""

    def degraded_validator(grounded, sse_events):
        return ValidationOutcome(
            validated=False,
            sentences=segment_answer_sentences(sse_events),
            degraded_reason="ProviderError: synthetic transport failure",
        )

    result = _run_with_validator(degraded_validator)
    assert result.validation is not None
    assert result.validation.get("validated") is False
    assert result.validation.get("sentences") == EXPECTED_SENTENCE_RECORDS


# ---------------------------------------------------------------------------
# Battery wiring: the four parts replace citation_support; the invariants
# gate is fed the run's OWN citation events (from the journalled SSE
# transcripts); unmeasured -> all four BLOCKED (#303).
# ---------------------------------------------------------------------------

FOUR_PART_NAMES = frozenset(
    {
        "citation_entailment_precision",
        "uncited_factual_rate",
        "verified_claim_group_coverage",
        "citation_invariants",
    }
)

CHART_RECORDS = {
    "spec": [{"item_id": "syn-chart-01", "status": "match"}],
    "refusal": [{"item_id": "syn-chart-02", "status": "refused_with_nearest"}],
}


def _gold(tmp_path):
    from evals.harness import load_and_validate_gold

    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _battery(gold, answer_results):
    from evals.harness import build_gate_battery

    return build_gate_battery(
        gold,
        answer_results,
        CHART_RECORDS,
        classifier_summary={"release_gate_passes": True, "per_class": {}},
        severity_records=[
            {"item_id": "syn-sev-01", "expected": "serious", "judged": "serious", "scored": True}
        ],
    )


def _answered(item_id: str, validation, sse_transcript=()):
    return ItemResult(
        item_id=item_id,
        arm_model=ARM_MODEL,
        route="retrieval",
        refused=False,
        answer_text="synthetic",
        documents=({"chunk_id": "syn_doc:0001", "source_type": "report"},),
        validation=validation,
        sse_transcript=tuple(sse_transcript),
    )


def _refused(item_id: str):
    return ItemResult(
        item_id=item_id,
        arm_model=ARM_MODEL,
        route="retrieval",
        refused=True,
        answer_text="The synthetic corpus does not cover this.",
    )


def _battery_results(gold, *, transcripts_by_id=None):
    transcripts_by_id = transcripts_by_id or {}
    results = []
    for item in gold.qa_items:
        if item.get("category") == "no_answer":
            results.append(_refused(item["id"]))
        else:
            record = _uniform_record(item["id"], factual=2, attached=2, entailed=2)
            record.pop("item_id")
            results.append(_answered(item["id"], record, transcripts_by_id.get(item["id"], ())))
    return results


def test_battery_replaces_citation_support_with_the_four_parts(tmp_path):
    gold = _gold(tmp_path)
    battery = _battery(gold, _battery_results(gold))
    names = [gate.name for gate in battery]
    assert FOUR_PART_NAMES <= set(names), (
        f"battery {sorted(names)} is missing {sorted(FOUR_PART_NAMES - set(names))}"
    )
    assert "citation_support" not in names, (
        "the flat citation_support gate is SUPERSEDED by the ratified four-part "
        "spec (issue #325) — keeping both would double-count one signal and the "
        "old one fails a passing arm"
    )


def test_battery_feeds_invariants_from_the_runs_own_citation_events(tmp_path):
    """A zero-width span in an answered item's journalled SSE transcript
    must fail citation_invariants — the gate is DERIVED from run records
    (finding #242), never fabricated."""
    from evals.gates import GATE_FAILED, GATE_PASSED

    gold = _gold(tmp_path)
    zero_width_transcript = (
        {"event": "text", "data": {"text": "Fact one is stated here."}},
        {
            "event": "citation",
            "data": {"document_index": 0, "answer_block_start": 10, "answer_block_end": 10},
        },
    )
    battery = _battery(
        gold,
        _battery_results(gold, transcripts_by_id={"syn-sp-01": zero_width_transcript}),
    )
    (invariants,) = [gate for gate in battery if gate.name == "citation_invariants"]
    assert invariants.status == GATE_FAILED
    assert any(
        entry.get("item_id") == "syn-sp-01" and entry.get("zero_width")
        for entry in invariants.evidence
    )

    clean = _battery(gold, _battery_results(gold))
    (invariants,) = [gate for gate in clean if gate.name == "citation_invariants"]
    assert invariants.status == GATE_PASSED


def test_all_four_parts_blocked_when_validation_never_ran(tmp_path):
    """#303 carried over: an unmeasured citation guarantee blocks release
    — each of the four parts is PRESENT and BLOCKED with a reason, and
    the verdict cannot be 'passed'."""
    from evals.gates import GATE_BLOCKED, release_verdict

    gold = _gold(tmp_path)
    results = []
    for item in gold.qa_items:
        if item.get("category") == "no_answer":
            results.append(_refused(item["id"]))
        else:
            results.append(_answered(item["id"], None))
    battery = _battery(gold, results)
    for name in sorted(FOUR_PART_NAMES):
        (gate,) = [candidate for candidate in battery if candidate.name == name]
        assert gate.status == GATE_BLOCKED, f"{name} must block when unmeasured"
        assert gate.reason and "validat" in gate.reason.lower()
    assert release_verdict(list(battery)) != "passed"


# ---------------------------------------------------------------------------
# The offline suite's simulated feed, extended coherently: its citation
# events carry REAL block extents, so the invariants gate is exercised on
# span data — never vacuously green. (The all-four-pass end-to-end pin
# lives in tests/integration/test_review_303_offline_battery.py.)
# ---------------------------------------------------------------------------


def test_offline_adapter_stream_delivers_span_carrying_citations():
    from tests.unit.test_eval_report import _load_run_evals_module

    run_evals = _load_run_evals_module()
    adapter = run_evals._OfflineAdapter()
    events = list(
        answer_stream_to_sse(
            adapter.generate_stream(messages=[], documents=[], config=None),
            retrieved=run_evals._offline_retrieve(None),
            corpus_vintage="0000-00-00",
        )
    )
    citations = [event for event in events if event.get("event") == "citation"]
    assert citations, "the offline delivery must cite (the simulated-honest feed)"
    for event in citations:
        data = event["data"]
        start = data.get("answer_block_start")
        end = data.get("answer_block_end")
        assert start is not None and end is not None and end > start, (
            "the offline transport must open/close its answer block so its "
            "citation carries a real non-zero-width extent — otherwise the "
            "citation_invariants gate would pass on legacy-spanless data it "
            "never actually examined"
        )
