"""Footprint feature RED — the /footprint page rendering.

``service.transparency.render_footprint_page`` renders the §5/§9 page
from the owner-approved methodology (docs/FOOTPRINT-METHODOLOGY.md,
BINDING) over the live ``FootprintTotals``: headline totals (ranges,
labelled estimated; the ONE measured line beside them), the
measured/estimated/unknown honesty table, the formula + the §3
constants interpolated from ``service.footprint`` (never hand-copied
figures), the verbatim Anthropic-uncertainty paragraph, the two grids
with the verbatim market-vs-location treatment, the three sourced
anchors, the §7 exclusions, and the revision note. Every-page
invariants (ADR-018 pair adjacency, the §4.11 disclaimer verbatim,
links to the other transparency routes) apply exactly as on the other
four surfaces. ``totals=None`` (unreadable ledger journal) renders the
honest unavailable notice — never silent zeros.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

import service.footprint as footprint
from service.footprint import (
    ANTHROPIC_GRID_ASSUMPTION_SENTENCE,
    ANTHROPIC_UNCERTAINTY_PARAGRAPH,
    FOOTPRINT_FACTORS_VERSION,
    FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE,
    MARKET_VS_LOCATION_SENTENCE,
    TRAINING_EXCLUSION_PHRASE,
    FootprintTotals,
)
from service.transparency import (
    CREDIT_PAIR_MAX_SEPARATION,
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    STEWARD_CREDIT_TEXT,
    TRANSPARENCY_ROUTES,
    render_footprint_page,
)
from tests._transparency_fixtures import chars_between, contains_verbatim, page_text

#: A realistic lifetime aggregate: ~10M input-class + 1M output tokens
#: over 12,345 answers, with 3.6 measured CPU-hours of retrieval compute.
TOTALS = FootprintTotals(
    since="2026-09-20",
    exchanges=12_345,
    input_tokens=6_000_000,
    output_tokens=1_000_000,
    cache_read_input_tokens=4_000_000,
    cache_creation_input_tokens=500_000,
    cpu_seconds=12_960.0,
)


def rendered() -> str:
    return render_footprint_page(totals=TOTALS)


#: A genuinely fresh ledger: the service booted, nobody has asked yet.
#: This is the page's launch-day state — NOT the unavailable state.
FRESH_TOTALS = FootprintTotals(
    since=None,
    exchanges=0,
    input_tokens=0,
    output_tokens=0,
    cache_read_input_tokens=0,
    cache_creation_input_tokens=0,
    cpu_seconds=0.0,
)

#: First-day totals: one answered question (500 in / 40 out).
TINY_TOTALS = FootprintTotals(
    since="2026-09-20",
    exchanges=1,
    input_tokens=500,
    output_tokens=40,
    cache_read_input_tokens=0,
    cache_creation_input_tokens=0,
    cpu_seconds=0.0,
)

#: A scientific-notation number as rendered text ("6.5e-06", "1.9E-05").
SCIENTIFIC_NOTATION = re.compile(r"\d(?:\.\d+)?[eE][-+]\d")


class TestHeadlineTotals:
    def test_totals_render_since_date_and_answer_count(self) -> None:
        text = page_text(rendered())
        assert "2026-09-20" in text
        # The N answers line (rendered with or without thousands separator).
        assert "12,345" in text or "12345" in text
        assert "answers" in text

    def test_headline_energy_is_an_estimated_range(self) -> None:
        # §9.1: application lifetime energy as an est. range, labelled.
        text = page_text(rendered()).lower()
        assert "estimated" in text or "est." in text
        assert "kwh" in text
        # A range needs an en-dash between two figures somewhere in the
        # headline; a single point total would violate always-a-range.
        assert "–" in page_text(rendered())

    def test_carbon_total_is_present_with_its_unit(self) -> None:
        # gCO2e belongs ON THE PAGE (beside the disclosed grid
        # assumption) — it is only the FOOTER that must not carry it.
        text = page_text(rendered())
        assert "CO2e" in text

    def test_the_one_measured_line(self) -> None:
        # §9.1: "of which our own server's retrieval compute:
        # {measured CPU-hours} (measured) ≈ est. C–D kWh".
        text = page_text(rendered()).lower()
        assert "(measured)" in text or "measured" in text
        # 12,960 CPU-seconds = 3.6 CPU-hours.
        assert "3.6" in text

    def test_unavailable_totals_render_the_honest_notice(self) -> None:
        html_out = render_footprint_page(totals=None)
        assert contains_verbatim(html_out, FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE)
        text = page_text(html_out)
        # Never silent zeros dressed as totals.
        assert "over 0 answers" not in text
        assert "since None" not in text

    def test_unavailable_state_keeps_the_methodology_sections(self) -> None:
        html_out = render_footprint_page(totals=None)
        assert contains_verbatim(html_out, ANTHROPIC_UNCERTAINTY_PARAGRAPH)
        assert contains_verbatim(html_out, NON_AFFILIATION_DISCLAIMER)


class TestFreshAndSmallLedgerRegister:
    """Review finding #361 — register violations on fresh/small ledgers.

    The launch-window page (the one that gets screenshotted) must obey
    the same register rules as everything else: no literal "None" on the
    public page, no scientific notation (§9: figures land in a human
    range "without scientific notation"), and English that survives a
    count of one. The honest fresh state is a dedicated branch — e.g.
    "no answers counted yet" — never ``est. 0–0 kWh … since None, over 0
    answers``. WORDING DECISION (the fresh-state phrase) flagged in the
    red-phase report.
    """

    def test_fresh_ledger_never_renders_the_word_none(self) -> None:
        text = page_text(render_footprint_page(totals=FRESH_TOTALS))
        assert "since None" not in text, "the literal string 'None' is on the public page"
        assert "over 0 answers" not in text

    def test_fresh_ledger_degrades_to_an_honest_empty_state(self) -> None:
        html_out = render_footprint_page(totals=FRESH_TOTALS)
        text = page_text(html_out)
        # Not a fabricated zero range dressed as a lifetime total…
        assert "0–0" not in text, "a fresh ledger must not render 'est. 0–0' totals"
        # …and not the unavailable notice either: a fresh ledger is a
        # healthy, honest zero state, not a broken counter.
        assert not contains_verbatim(html_out, FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE)
        assert "no answers counted yet" in text.lower()

    def test_fresh_state_keeps_the_methodology_sections(self) -> None:
        html_out = render_footprint_page(totals=FRESH_TOTALS)
        assert contains_verbatim(html_out, ANTHROPIC_UNCERTAINTY_PARAGRAPH)
        assert contains_verbatim(html_out, NON_AFFILIATION_DISCLAIMER)

    def test_small_totals_never_render_scientific_notation(self) -> None:
        # 500 in / 40 out: the kWh/kg figures sit far below 1e-4, where
        # "%g" flips to scientific — the §9 register rule says never.
        text = page_text(render_footprint_page(totals=TINY_TOTALS))
        match = SCIENTIFIC_NOTATION.search(text)
        assert match is None, f"scientific notation on the public page: {match.group(0)!r}"

    def test_one_answer_is_singular(self) -> None:
        text = page_text(render_footprint_page(totals=TINY_TOTALS))
        assert "over 1 answers" not in text, "'over 1 answers' — singular/plural violation"
        assert "over 1 answer" in text

    def test_many_answers_stay_plural(self) -> None:
        # The existing headline shape survives the fix.
        text = page_text(rendered())
        assert "answers" in text


class TestLocalSliceInstrumentationHonesty:
    """Review finding #362 — an absence dressed as a measurement.

    Ratified decision 7 on #358 defers the ``getrusage`` wiring to
    deploy, so ``cpu_seconds`` is a constant 0.0 that nothing measures —
    yet the page renders "0.0 CPU-hours (measured)" and the honesty
    table asserts the CPU time "is measured on the box". Until
    ``cpu_seconds > 0``, the page must say the truth instead: the
    counter is not yet instrumented ("not yet instrumented" pinned;
    exact surrounding wording is the implementer's — flagged). With a
    real nonzero measurement, the current measured line renders.
    """

    def test_zero_cpu_seconds_is_not_presented_as_a_measured_zero(self) -> None:
        text = page_text(render_footprint_page(totals=replace(TOTALS, cpu_seconds=0.0)))
        assert "0.0 CPU-hours (measured)" not in text, (
            "nothing measures cpu_seconds yet — a constant zero must never "
            "be presented as a measurement"
        )
        assert "not yet instrumented" in text.lower()

    def test_honesty_table_matches_the_uninstrumented_state(self) -> None:
        # The Measured column's "is measured on the box" claim is false
        # until the counter is wired — the table row must move with the
        # same branch as the headline line.
        text = page_text(render_footprint_page(totals=replace(TOTALS, cpu_seconds=0.0)))
        assert "is measured on the box" not in text

    def test_a_real_measurement_renders_the_measured_line(self) -> None:
        # TOTALS carries 12,960 measured CPU-seconds (3.6 CPU-hours):
        # the §9.1 measured line renders, and the placeholder does not.
        text = page_text(rendered())
        assert "(measured)" in text
        assert "not yet instrumented" not in text.lower()

    def test_fresh_ledger_is_also_uninstrumented(self) -> None:
        # The launch-day page (fresh ledger, cpu_seconds 0.0) carries the
        # honest placeholder, not a measured zero.
        text = page_text(render_footprint_page(totals=FRESH_TOTALS))
        assert "0.0 CPU-hours (measured)" not in text


class TestFooterScopeDisclosure:
    """Review finding #360 (disclosure half) — the footer/totals scope
    split, stated ON THE PAGE beside the honesty table.

    Ratified decision 4 on #358 accepts that the chat FOOTER shows the
    wire-visible (answer-generation) usage only, with "the full picture
    on the page". That is only honest if the page SAYS so: the reader
    must learn (a) the footer's per-answer figure covers the
    answer-generation call only, (b) the classifier and validation calls
    are counted in the totals on this page, and (c) best-mode answers'
    footer figures use the default-model factors (the wire usage event
    carries no model, so an Opus answer's footer is computed with
    Haiku-class factors — its true central sits ABOVE the displayed high
    end). Exact prose fragments are pinned; WORDING DECISION flagged in
    the red-phase report.
    """

    def test_footer_scope_is_disclosed(self) -> None:
        text = page_text(rendered())
        assert "the answer-generation call only" in text, (
            "/footprint must disclose that the chat footer's per-answer figure "
            "covers the answer-generation call only"
        )

    def test_classifier_and_validation_scope_is_disclosed(self) -> None:
        text = page_text(rendered())
        assert "classifier and validation calls are counted in the totals on this page" in text, (
            "/footprint must state where the classifier/validation tokens are "
            "counted — the reader is otherwise led to believe the footer "
            "covers everything"
        )

    def test_best_mode_footer_factor_gap_is_disclosed(self) -> None:
        text = page_text(rendered())
        assert "default-model factors" in text, (
            "/footprint must disclose that best-mode answers' footer figures "
            "are computed with the default-model factors"
        )

    def test_the_disclosure_survives_the_unavailable_state(self) -> None:
        # The scope split is methodology, not a live total — it must be
        # disclosed even when the counter file cannot be read.
        text = page_text(render_footprint_page(totals=None))
        assert "the answer-generation call only" in text


class TestHonestyTable:
    """§9.2 — the measured / estimated / unknown centrepiece."""

    def test_the_three_columns_are_present(self) -> None:
        text = page_text(rendered())
        for label in ("Measured", "Estimated", "Unknown"):
            assert label in text, f"honesty table lacks the {label} column"

    def test_token_counts_are_the_measured_row(self) -> None:
        text = page_text(rendered()).lower()
        assert "token" in text
        assert "provider-reported" in text or "measured" in text

    def test_anthropic_energy_per_token_is_the_unknown(self) -> None:
        text = page_text(rendered()).lower()
        assert "anthropic" in text
        assert "unknown" in text

    def test_local_slice_labelled_measured_times_estimated(self) -> None:
        # §5's exact honest label for the only measured-energy slice.
        assert contains_verbatim(rendered(), "measured CPU time × estimated per-vCPU wattage")


class TestMethodSection:
    def test_constants_are_interpolated_not_hand_copied(self, monkeypatch) -> None:
        """The retention-constants pattern: the page reads the factor
        module attributes AT CALL TIME, so a factor change (or this
        monkeypatch) re-renders — published figures can never silently
        diverge from the code that computes them."""
        sentinel = footprint.EnergyFactor(low=0.111, central=0.555, high=1.555)
        monkeypatch.setattr(footprint, "E_OUT_WH_PER_1K", sentinel)
        text = page_text(render_footprint_page(totals=TOTALS))
        assert "0.555" in text, "the page did not re-render the patched factor"

    def test_every_factor_range_is_published(self) -> None:
        text = page_text(rendered())
        # §3 table: central values and both range ends, verbatim figures.
        for figure in (
            "0.5",
            "0.1",
            "1.5",
            "0.02",
            "0.005",
            "0.07",
            "0.08",
            "287",
            "384",
            "450",
            "2.2",
            "1.13",
        ):
            assert figure in text, f"§3 figure {figure} missing from the method section"

    def test_provenance_is_named(self) -> None:
        text = page_text(rendered())
        for source in ("Jegham", "Google", "Epoch", "ML.ENERGY", "Ember", "Hetzner", "EPA", "IEA"):
            assert source in text, f"the method section does not name {source}"

    def test_the_uncertainty_paragraph_is_verbatim(self) -> None:
        assert contains_verbatim(rendered(), ANTHROPIC_UNCERTAINTY_PARAGRAPH)

    def test_the_two_grids_are_verbatim(self) -> None:
        html_out = rendered()
        assert contains_verbatim(html_out, ANTHROPIC_GRID_ASSUMPTION_SENTENCE)
        assert contains_verbatim(html_out, MARKET_VS_LOCATION_SENTENCE)

    def test_opus_extrapolation_is_flagged(self) -> None:
        # §3.4: the non-default-model multipliers are the weakest numbers
        # in the table and must be labelled as extrapolated.
        assert "extrapolat" in page_text(rendered()).lower()


class TestAnchors:
    def test_all_three_anchors_with_sources_inline(self) -> None:
        text = page_text(rendered()).lower()
        assert "streaming" in text
        assert "iea" in text
        assert "metre" in text
        assert "epa" in text
        assert "tea" in text or "kettle" in text

    def test_no_rejected_anchor_sneaks_in(self) -> None:
        # §8 explicitly rejected trees / smartphone charges / water drops.
        text = page_text(rendered()).lower()
        assert "trees absorb" not in text
        assert "smartphone" not in text


class TestExclusions:
    def test_the_exclusion_list_is_disclosed(self) -> None:
        text = page_text(rendered()).lower()
        assert TRAINING_EXCLUSION_PHRASE in text  # training: not included, not zero
        assert "training" in text
        assert "device" in text  # the visitor's own device
        assert "water" in text  # unknown, listed as unknown
        assert "embodied" in text  # serving-hardware embodied carbon

    def test_water_stays_in_the_unknown_column_not_an_anchor(self) -> None:
        text = page_text(rendered()).lower()
        assert "water" in text and "unknown" in text


class TestRevisionNote:
    def test_factors_version_is_rendered(self) -> None:
        assert FOOTPRINT_FACTORS_VERSION in page_text(rendered())


class TestEveryPageInvariants:
    def test_steward_credit_paired_with_noncommercial_note(self) -> None:
        text = page_text(rendered())
        assert STEWARD_CREDIT_TEXT in text
        assert NONCOMMERCIAL_NOTE in text
        assert (
            chars_between(text, STEWARD_CREDIT_TEXT, NONCOMMERCIAL_NOTE)
            <= CREDIT_PAIR_MAX_SEPARATION
        ), "the ADR-018 credit and non-commercial note are separated on /footprint"

    def test_nonaffiliation_disclaimer_verbatim(self) -> None:
        assert contains_verbatim(rendered(), NON_AFFILIATION_DISCLAIMER)

    def test_links_every_other_transparency_route(self) -> None:
        html_out = rendered()
        for route in TRANSPARENCY_ROUTES:
            if route == "/footprint":
                continue
            assert route in html_out, f"/footprint does not link {route}"

    def test_no_secrets_or_key_shaped_strings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC-NOT-REAL-000")
        assert "sk-ant-" not in rendered()

    def test_totals_are_the_only_dynamic_content(self) -> None:
        """Privacy: the page renders counts and dates only — no
        identifiers, no question/answer content can ever reach it
        (structurally: FootprintTotals has no content-bearing field)."""
        field_names = set(FootprintTotals.__dataclass_fields__)
        assert field_names == {
            "since",
            "exchanges",
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cpu_seconds",
        }
