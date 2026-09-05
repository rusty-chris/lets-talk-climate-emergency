"""Issue #313: docs-as-code pins for the refusal-redesign amendment.

DESIGN.md §3.5 and DECISIONS.md ADR-010 are the contract (ORCHESTRATION
step 5); the redesign is only real once they record it. Characterisation
anchors, not phrasing: each test asserts a load-bearing statement of the
amended contract and the live-evidence date, so later doc edits cannot
silently roll the design back to the falsified v2 wording.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _design() -> str:
    return (REPO_ROOT / "DESIGN.md").read_text(encoding="utf-8")


def _decisions() -> str:
    return (REPO_ROOT / "DECISIONS.md").read_text(encoding="utf-8")


def _section_3_5(text: str) -> str:
    start = text.index("### 3.5")
    end = text.index("### 3.6", start)
    return text[start:end]


class TestDesignRefusalSection:
    def test_section_3_5_is_amended_not_the_v2_wording(self) -> None:
        section = _section_3_5(_design())
        assert "unchanged from v2" not in section, (
            "the falsified v2 refusal wording must not survive the amendment"
        )
        assert re.search(r"amended.{0,40}2026-09", section, re.IGNORECASE), (
            "§3.5 must record when (2026-09) and that it was amended"
        )
        assert "#313" in section

    def test_authoritative_signal_is_the_structured_decline(self) -> None:
        section = _section_3_5(_design())
        assert re.search(r"authoritative", section, re.IGNORECASE)
        assert re.search(
            r"structured.{0,60}decline|decline.{0,60}structured", section, re.IGNORECASE | re.DOTALL
        )
        assert "GENERATION_DECLINE_MARKER" in section, (
            "§3.5 must name the marker constant the contract hangs on"
        )

    def test_threshold_recorded_as_prefilter_with_degradation(self) -> None:
        section = _section_3_5(_design())
        assert re.search(r"pre-?filter", section, re.IGNORECASE)
        assert re.search(
            r"(inseparable|separab\w+).{0,300}(no longer|never|not).{0,80}"
            r"(brick|block)",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§3.5 must record that an inseparable corpus no longer bricks the release"
        assert re.search(
            r"(missing|malformed|failed).{0,120}(degrade|disabled|warning)",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§3.5 must record the degrade-not-block artifact behaviour"

    def test_gates_measure_the_authoritative_signal(self) -> None:
        section = _section_3_5(_design())
        assert re.search(
            r"refusal gate.{0,400}(pre-?filter.{0,80}OR.{0,80}decline|"
            r"decline.{0,80}OR.{0,80}pre-?filter)",
            section,
            re.IGNORECASE | re.DOTALL,
        ), "§3.5 must state the gate counts pre-filter refusal OR structured decline"

    def test_honesty_invariants_recorded(self) -> None:
        section = _section_3_5(_design())
        assert re.search(
            r"decline never carries citations|never carries citations, badges",
            section,
            re.IGNORECASE,
        )

    def test_live_evidence_date_recorded(self) -> None:
        assert "2026-09-04/05" in _section_3_5(_design()), (
            "the live-run evidence date anchors the amendment to its proof"
        )


class TestAdr010Amendment:
    def _adr_010(self) -> str:
        text = _decisions()
        start = text.index("## ADR-010")
        end = text.index("## ADR-011", start)
        return text[start:end]

    def test_amendment_block_exists_with_date_and_issue(self) -> None:
        adr = self._adr_010()
        assert re.search(r"amendment.{0,60}2026-09", adr, re.IGNORECASE)
        assert "#313" in adr

    def test_amendment_records_the_live_evidence(self) -> None:
        adr = self._adr_010()
        assert "10/10" in adr, "the 10/10 honest live declines are the amendment's evidence"
        assert "0.3885" in adr and "0.00142" in adr, (
            "the overlapping live geometry is the falsifying measurement"
        )
        assert re.search(r"0\.0195.{0,80}diagnostic", adr, re.IGNORECASE | re.DOTALL), (
            "the recorded 0.0195 stays diagnostic-only"
        )

    def test_amendment_states_the_new_signal_and_demotion(self) -> None:
        adr = self._adr_010()
        assert re.search(r"authoritative refusal signal", adr, re.IGNORECASE)
        assert re.search(
            r"demoted?.{0,80}pre-?filter|pre-?filter.{0,120}demot", adr, re.IGNORECASE | re.DOTALL
        )
