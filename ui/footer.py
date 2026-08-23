"""The site footer & framing furniture (issue #18, DESIGN §7.3, ADR-018).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_ui_footer.py`` pins the contract.

The ADR-018 invariant this module makes UNREPRESENTABLE to violate: the
"Built by Rusty Data" steward credit and the free/open-source/
non-commercial note are an inseparable pair. :class:`StewardCredit`
refuses construction (raises :class:`FooterInvariantError`) when either
half is blank, and :func:`render_footer_lines` emits them together —
there is no code path that renders the credit without the note.

:func:`build_page_footer` assembles the one footer EVERY page view model
carries (landing, chat, chart — pinned per page in the unit suites):

- the steward credit pair (ADR-018, owner-confirmed 2026-08-16);
- the DESIGN §4.11 non-affiliation disclaimer, verbatim;
- the transparency links — routes only, ``/about`` ``/privacy``
  ``/sources`` ``/voices`` (#22 serves the pages; #19 owns their
  content — the UI links routes, never duplicates content).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "STEWARD_CREDIT_TEXT",
    "NONCOMMERCIAL_NOTE",
    "NON_AFFILIATION_DISCLAIMER",
    "TRANSPARENCY_ROUTES",
    "FooterInvariantError",
    "StewardCredit",
    "PageFooter",
    "build_page_footer",
    "render_footer_lines",
]

#: ADR-018 (owner-confirmed 2026-08-16): the credit and the note are a pair.
STEWARD_CREDIT_TEXT = "Built by Rusty Data"
NONCOMMERCIAL_NOTE = "A free, open-source, non-commercial project."

#: DESIGN §4.11, verbatim — everywhere sources or the campaign appear.
NON_AFFILIATION_DISCLAIMER = (
    "Not affiliated with or endorsed by the National Emergency Briefing "
    "campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or "
    "the IPCC. All sources cited and linked."
)

#: The #22 static surfaces the footer links to (routes, not content).
TRANSPARENCY_ROUTES: tuple[str, ...] = ("/about", "/privacy", "/sources", "/voices")


class FooterInvariantError(Exception):
    """The ADR-018 credit/non-commercial pair was about to be split."""


@dataclass(frozen=True)
class StewardCredit:
    """The inseparable ADR-018 pair; construction validates both halves."""

    credit_text: str
    noncommercial_note: str

    def __post_init__(self) -> None:
        # RED phase: validation is part of the pinned behaviour and is not
        # implemented yet — the unit suite drives it in (a blank half must
        # raise FooterInvariantError).
        pass


@dataclass(frozen=True)
class PageFooter:
    """The one footer every page view model carries."""

    credit: StewardCredit
    non_affiliation: str
    transparency_routes: tuple[str, ...]


def build_page_footer() -> PageFooter:
    """The assembled ADR-018/§7.3 footer (see module docs for the contract)."""
    raise NotImplementedError("issue #18 red phase: build_page_footer is not implemented yet")


def render_footer_lines(footer: PageFooter) -> tuple[str, ...]:
    """Pure render: the footer's display lines; credit and note always together."""
    raise NotImplementedError("issue #18 red phase: render_footer_lines is not implemented yet")
