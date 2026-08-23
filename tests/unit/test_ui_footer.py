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

from ui.footer import (
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
