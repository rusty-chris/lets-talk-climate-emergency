"""Issue #325: docs-as-code pins for the DESIGN §6.2 citation-gate
amendment.

DESIGN.md is the contract (ORCHESTRATION step 5); the owner's ratified
four-part re-spec (decision recorded 2026-09-07 on issue #325) is only
real once §6.2 records it. Characterisation anchors, not phrasing: each
test asserts a load-bearing statement — the four parts with their exact
thresholds, the ratified claim-group definition, the decision date, the
survey calibration line that justified abandoning the 0.95 per-sentence
target, and the supersession itself — so later doc edits cannot silently
roll the gate back to the unreachable v2 target.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _section_6_2() -> str:
    text = (REPO_ROOT / "DESIGN.md").read_text(encoding="utf-8")
    start = text.index("### 6.2")
    end = text.index("### 6.3", start)
    return text[start:end]


class TestDesignCitationGateAmendment:
    def test_amendment_records_the_owner_decision_and_date(self) -> None:
        section = _section_6_2()
        assert "#325" in section, "§6.2 must cite the deciding issue"
        assert "2026-09-07" in section, "§6.2 must record the owner decision date"
        assert re.search(r"amend|ratif", section, re.IGNORECASE)

    def test_part_1_entailment_precision_with_threshold_and_denominator(self) -> None:
        section = _section_6_2()
        assert re.search(r"citation_entailment_precision.{0,80}0\.95", section, re.DOTALL), (
            "§6.2 must state citation_entailment_precision >= 0.95"
        )
        assert re.search(r"attached.{0,80}factual sentences", section, re.IGNORECASE | re.DOTALL), (
            "§6.2 must state the precision denominator: attached factual sentences"
        )

    def test_part_2_uncited_rate_ceiling_with_ratchet(self) -> None:
        section = _section_6_2()
        assert re.search(r"uncited_factual_rate.{0,80}0\.35", section, re.DOTALL), (
            "§6.2 must state the uncited_factual_rate <= 0.35 ceiling"
        )
        assert re.search(r"ratchet", section, re.IGNORECASE), (
            "§6.2 must record the owner's ratchet-down-over-time intent for the "
            "ceiling — the 0.35 is a starting point, not a resting place"
        )

    def test_part_3_claim_group_coverage_with_the_ratified_definition(self) -> None:
        section = _section_6_2()
        assert re.search(r"verified_claim_group_coverage.{0,80}0\.75", section, re.DOTALL), (
            "§6.2 must state verified_claim_group_coverage >= 0.75"
        )
        assert re.search(
            r"maximal run of contiguous factual sentences.{0,60}paragraph",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§6.2 must record the ratified claim-group definition verbatim-in-substance"
        assert re.search(
            r"(>=|≥|at least) ?1?.{0,40}entailed citation.{0,120}(verified|group)|"
            r"(verified|group).{0,120}(>=|≥|at least) ?1?.{0,40}entailed citation",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§6.2 must state when a group counts as verified (>=1 entailed member)"

    def test_part_4_invariants(self) -> None:
        section = _section_6_2()
        assert re.search(r"zero.{0,20}zero-width", section, re.IGNORECASE | re.DOTALL), (
            "§6.2 must record the zero zero-width-spans invariant"
        )
        assert re.search(
            r"(every|each) answered exchange.{0,80}entailed citation",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§6.2 must record the >=1-entailed-citation-per-answered-exchange invariant"

    def test_survey_calibration_line_recorded(self) -> None:
        """The one-liner that justified the re-spec: no measured system
        reaches 95% per-sentence citation recall (academic best 69.3%
        long-form; best deployed 68.7%) — the numbers anchor the decision
        to its evidence so a future 'just raise it back to 0.95' has to
        argue with the survey."""
        section = _section_6_2()
        assert "69.3" in section and "68.7" in section
        assert re.search(
            r"no measured system|no (published|deployed) system", section, re.IGNORECASE
        )

    def test_old_per_sentence_target_recorded_as_superseded(self) -> None:
        """§6.2 must say the flat 0.95-of-all-factual-sentences
        citation-support gate is superseded/replaced — not leave two
        contradictory release gates in the contract."""
        section = _section_6_2()
        assert re.search(r"supersed|replac", section, re.IGNORECASE)

    def test_gate_layer_recompute_scope_recorded(self) -> None:
        """The decision's scope guard: validator segmentation, pairing and
        entailment (and the per-sentence UI chips/badges) are unchanged —
        the re-spec is a gate-layer recompute."""
        section = _section_6_2()
        assert re.search(
            r"gate-?layer.{0,40}recompute|recompute.{0,40}gate-?layer",
            section,
            re.IGNORECASE | re.DOTALL,
        )
        assert re.search(
            r"(segmentation|validator).{0,80}unchanged|UI unchanged",
            section,
            re.IGNORECASE | re.DOTALL,
        )
