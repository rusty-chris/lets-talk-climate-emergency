"""Issue #18 RED — the ADR-018 footer invariant and framing furniture.

Pins ``ui.footer``: the "Built by Rusty Data" credit and the
free/open-source/non-commercial note are an INSEPARABLE pair (ADR-018,
owner-confirmed 2026-08-16) — unrepresentable apart at the type level
and always rendered together; the §4.11 non-affiliation disclaimer is
verbatim; the transparency links are routes into #22's static surfaces
(never duplicated content).
"""

from __future__ import annotations

import pytest

import ui.footer
from ui.footer import (
    DONATIONS_NOTE_SUFFIX,
    DONATIONS_URL,
    NONCOMMERCIAL_NOTE,
    RUSTY_DATA_URL,
    STEWARD_CREDIT_TEXT,
    STEWARD_MARK_PATH,
    TRANSPARENCY_ROUTES,
    FooterInvariantError,
    StewardCredit,
    build_page_footer,
    footer_link_line,
    render_footer_lines,
)


class TestStewardCreditPair:
    def test_credit_without_the_noncommercial_note_is_unrepresentable(self) -> None:
        with pytest.raises(FooterInvariantError):
            StewardCredit(credit_text="Built by Rusty Data", noncommercial_note="")

    def test_note_without_the_credit_is_unrepresentable(self) -> None:
        with pytest.raises(FooterInvariantError):
            StewardCredit(
                credit_text="", noncommercial_note="A free, open-source, non-commercial project."
            )

    def test_whitespace_only_halves_are_rejected_too(self) -> None:
        with pytest.raises(FooterInvariantError):
            StewardCredit(credit_text="Built by Rusty Data", noncommercial_note="   ")


class TestBuildPageFooter:
    def test_footer_carries_the_credit_pair(self) -> None:
        footer = build_page_footer()
        assert footer.credit.credit_text == "Built by Rusty Data"
        note = footer.credit.noncommercial_note.lower()
        assert "free" in note
        assert "open-source" in note or "open source" in note
        assert "non-commercial" in note

    def test_footer_carries_the_design_4_11_non_affiliation_disclaimer_verbatim(self) -> None:
        footer = build_page_footer()
        assert footer.non_affiliation == (
            "Not affiliated with or endorsed by the National Emergency Briefing "
            "campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or "
            "the IPCC. All sources cited and linked."
        )

    def test_footer_links_the_transparency_routes_not_content(self) -> None:
        footer = build_page_footer()
        assert footer.transparency_routes == ("/about", "/privacy", "/sources", "/voices")


class TestRenderFooterLines:
    def test_rendered_footer_always_pairs_credit_with_the_note(self) -> None:
        rendered = "\n".join(render_footer_lines(build_page_footer()))
        assert "Built by Rusty Data" in rendered
        lowered = rendered.lower()
        assert "non-commercial" in lowered
        assert "free" in lowered
        assert "open-source" in lowered or "open source" in lowered

    def test_rendered_footer_includes_disclaimer_and_transparency_routes(self) -> None:
        rendered = "\n".join(render_footer_lines(build_page_footer()))
        assert "Not affiliated with or endorsed by" in rendered
        for route in ("/about", "/privacy", "/sources", "/voices"):
            assert route in rendered


class TestFooterLinks:
    """Review finding #228 RED — the transparency routes must be reachable.

    The footer currently renders '/about · /privacy · /sources · /voices'
    as dead caption text on the Streamlit origin; the pages are served by
    the api at the SITE_URL origin. /privacy is a legal surface (DESIGN
    §9 UK-GDPR): it must be a real link from the page that logs queries.
    """

    def test_transparency_links_render_as_absolute_markdown_links(self) -> None:
        line = footer_link_line(build_page_footer(), "https://site.example")

        assert "[About](https://site.example/about)" in line
        assert "[Privacy](https://site.example/privacy)" in line
        assert "[Sources](https://site.example/sources)" in line
        assert "[Voices](https://site.example/voices)" in line
        # Every TRANSPARENCY_ROUTES member exactly once — none dropped,
        # none duplicated.
        for route in TRANSPARENCY_ROUTES:
            assert line.count(f"https://site.example{route})") == 1

    def test_trailing_slash_base_url_joins_cleanly(self) -> None:
        line = footer_link_line(build_page_footer(), "https://site.example/")
        assert "https://site.example/about" in line
        assert "//about" not in line

    def test_link_line_carries_no_credit_pair(self) -> None:
        """The ADR-018 credit/non-commercial pair stays on its OWN line
        (render_footer_lines); the link line is links only — the pair's
        inseparability invariant is not diluted by mixing surfaces."""
        line = footer_link_line(build_page_footer(), "https://site.example")
        assert "Built by Rusty Data" not in line


class TestRustyDataMarkAndLink:
    """Rusty Data branding RED — the credit gains the owner's mark + a
    live link to rustydata.ai, WITHOUT weakening any ADR-018 pin: the
    credit text is unchanged, the non-commercial note stays on the same
    rendered line (inseparable), and the link is enrichment only.
    """

    def test_rusty_data_url_is_pinned(self) -> None:
        assert RUSTY_DATA_URL == "https://rustydata.ai"

    def test_credit_text_is_unchanged_by_the_enrichment(self) -> None:
        """The link wraps the EXISTING credit; the wording never moves."""
        assert STEWARD_CREDIT_TEXT == "Built by Rusty Data"

    def test_credit_line_links_the_credit_text_to_rustydata_ai(self) -> None:
        """The first rendered line carries the credit as a real markdown
        link on the pinned URL — label exactly the credit text."""
        credit_line = render_footer_lines(build_page_footer())[0]
        assert f"[{STEWARD_CREDIT_TEXT}]({RUSTY_DATA_URL})" in credit_line

    def test_link_never_splits_the_adr018_pair(self) -> None:
        """The enriched credit line still carries the non-commercial note
        beside the (now linked) credit — ONE line, never split."""
        credit_line = render_footer_lines(build_page_footer())[0]
        assert STEWARD_CREDIT_TEXT in credit_line
        lowered = credit_line.lower()
        assert "non-commercial" in lowered
        assert "free" in lowered

    def test_mark_asset_is_checked_in_and_is_an_svg(self) -> None:
        """The footer-scale mark ships in the repo (ui/static), SVG —
        the owner's own hex-nut mark from the rusty_data_website repo."""
        assert STEWARD_MARK_PATH.suffix == ".svg"
        assert STEWARD_MARK_PATH.is_file(), (
            f"the Rusty Data mark asset is not checked in at {STEWARD_MARK_PATH}"
        )
        content = STEWARD_MARK_PATH.read_text(encoding="utf-8")
        assert "<svg" in content

    def test_mark_asset_records_its_provenance(self) -> None:
        """Licence hygiene: the asset names its source (the owner's own
        rusty_data_website repo), so reuse is traceably licence-clean."""
        content = STEWARD_MARK_PATH.read_text(encoding="utf-8")
        assert "rusty_data_website" in content

    def test_mark_asset_is_self_contained(self) -> None:
        """The SVG fetches nothing and scripts nothing — it inlines
        cleanly anywhere (the transparency pages' no-external-requests
        convention)."""
        content = STEWARD_MARK_PATH.read_text(encoding="utf-8")
        assert "<script" not in content
        assert "href" not in content  # no xlink/external references
        # The only URL an inline mark may carry is the SVG namespace.
        for token in content.split():
            if "http" in token:
                assert "www.w3.org" in token, f"external reference in the mark: {token}"


class TestDonationsGate:
    """The donations wording ships config-gated and OFF: the owner has no
    donation account yet, so NOTHING user-visible changes until he
    supplies a URL. When it flips on, the suffix joins the SAME
    inseparable note — the ADR-018 pair is never diluted."""

    def test_gate_defaults_off(self) -> None:
        assert DONATIONS_URL is None

    def test_suffix_wording_is_pinned_for_later(self) -> None:
        assert DONATIONS_NOTE_SUFFIX == " — donations cover running costs"

    def test_nothing_user_visible_while_the_gate_is_off(self) -> None:
        footer = build_page_footer()
        assert footer.credit.noncommercial_note == NONCOMMERCIAL_NOTE
        rendered = "\n".join(render_footer_lines(footer)).lower()
        assert "donation" not in rendered

    def test_enabled_gate_appends_the_suffix_inside_the_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the owner fills DONATIONS_URL, the suffix rides INSIDE the
        non-commercial note (same StewardCredit half, same rendered line)
        — enabling donations can never separate the ADR-018 pair.
        NOTE for the enable day: the note's verbatim-on-page pins
        (test_transparency_pages.py) key on NONCOMMERCIAL_NOTE and will
        prompt a conscious wording review then — by design."""
        monkeypatch.setattr(ui.footer, "DONATIONS_URL", "https://example.org/donate")
        footer = build_page_footer()
        note = footer.credit.noncommercial_note
        assert "non-commercial" in note
        assert note.endswith("donations cover running costs.")
        credit_line = render_footer_lines(footer)[0]
        assert STEWARD_CREDIT_TEXT in credit_line
        assert "donations cover running costs" in credit_line
