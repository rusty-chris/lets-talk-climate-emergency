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

- the steward credit pair (ADR-018, owner-confirmed 2026-08-16) —
  ENRICHED with the Rusty Data branding: the credit renders as a live
  markdown link to :data:`RUSTY_DATA_URL` and the shell draws the
  footer-scale mark at :data:`STEWARD_MARK_PATH` beside it (the pair's
  inseparability is untouched: the link wraps the unchanged credit text
  on the same line as the note);
- the DESIGN §4.11 non-affiliation disclaimer, verbatim;
- the transparency links — routes only, ``/about`` ``/privacy``
  ``/sources`` ``/voices`` (#22 serves the pages; #19 owns their
  content — the UI links routes, never duplicates content).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "STEWARD_CREDIT_TEXT",
    "NONCOMMERCIAL_NOTE",
    "NON_AFFILIATION_DISCLAIMER",
    "RUSTY_DATA_URL",
    "STEWARD_MARK_PATH",
    "DONATIONS_URL",
    "DONATIONS_NOTE_SUFFIX",
    "TRANSPARENCY_ROUTES",
    "FooterInvariantError",
    "StewardCredit",
    "PageFooter",
    "build_page_footer",
    "render_footer_lines",
    "footer_link_line",
]

#: ADR-018 (owner-confirmed 2026-08-16): the credit and the note are a pair.
STEWARD_CREDIT_TEXT = "Built by Rusty Data"
NONCOMMERCIAL_NOTE = "A free, open-source, non-commercial project."

#: Rusty Data branding: the credit is a live link to the steward's site.
#: The link ENRICHES the ADR-018 credit — the credit text is unchanged and
#: the non-commercial note stays on the same rendered line.
RUSTY_DATA_URL = "https://rustydata.ai"

#: The footer-scale Rusty Data mark (hex nut + rust data core), the
#: owner's own mark copied from his rusty_data_website repo (provenance
#: recorded inside the SVG) — licence-clean reuse; the shell draws it
#: with st.image at footer scale, never as a banner.
STEWARD_MARK_PATH = Path(__file__).resolve().parent / "static" / "rusty_data_mark.svg"

#: Donations gate — OFF until the owner supplies a real donation URL (no
#: donation account exists yet, so nothing user-visible changes while
#: this is None). When the owner fills it, the suffix joins the
#: non-commercial note INSIDE the ADR-018 pair (same StewardCredit half,
#: same rendered line) — enabling donations can never split the pair.
#: NOTE for the enable day: the transparency pages' verbatim
#: NONCOMMERCIAL_NOTE pins will prompt a conscious wording review then.
DONATIONS_URL: str | None = None
DONATIONS_NOTE_SUFFIX = " — donations cover running costs"


def _noncommercial_note_text() -> str:
    """The note with the config-gated donations suffix (gate defaults OFF).

    Reads :data:`DONATIONS_URL` at call time (the retention-constants
    pattern) so flipping the gate — or a test's monkeypatch — takes
    effect without touching any render path.
    """
    if DONATIONS_URL:
        return NONCOMMERCIAL_NOTE.rstrip(".") + DONATIONS_NOTE_SUFFIX + "."
    return NONCOMMERCIAL_NOTE


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
        # ADR-018: the credit and the non-commercial note are one indivisible
        # pair. A blank (or whitespace-only) half would render the credit
        # without its non-commercial framing — unrepresentable by contract.
        if not self.credit_text.strip() or not self.noncommercial_note.strip():
            raise FooterInvariantError(
                "ADR-018: the steward credit and the non-commercial note are "
                "inseparable — neither half may be blank"
            )


@dataclass(frozen=True)
class PageFooter:
    """The one footer every page view model carries."""

    credit: StewardCredit
    non_affiliation: str
    transparency_routes: tuple[str, ...]


def build_page_footer() -> PageFooter:
    """The assembled ADR-018/§7.3 footer (see module docs for the contract)."""
    return PageFooter(
        credit=StewardCredit(
            credit_text=STEWARD_CREDIT_TEXT,
            noncommercial_note=_noncommercial_note_text(),
        ),
        non_affiliation=NON_AFFILIATION_DISCLAIMER,
        transparency_routes=TRANSPARENCY_ROUTES,
    )


def footer_link_line(footer: PageFooter, base_url: str) -> str:
    """The transparency routes as REAL markdown links (review finding #228).

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_footer.py::TestFooterLinks`` pin the contract:
    every route in :data:`TRANSPARENCY_ROUTES` renders once as a labelled
    absolute markdown link resolving against ``base_url`` (the api/site
    origin — the pages are served by the api service, NOT the Streamlit
    origin, so relative routes 404). A trailing slash on ``base_url``
    joins cleanly (never ``//about``). ``/privacy`` is a legal surface
    (DESIGN §9 UK-GDPR): a deployment where the privacy notice is not
    reachable from the page that logs the queries undermines the
    legitimate-interests position.
    """
    base = base_url.rstrip("/")
    links = [
        # Label = the capitalised route name (ratified decision 6); the
        # target is an absolute URL on the api/site origin. Trailing-slash
        # bases join cleanly (``base`` is stripped above), so never //about.
        f"[{route.lstrip('/').capitalize()}]({base}{route})"
        for route in footer.transparency_routes
    ]
    return " · ".join(links)


def render_footer_lines(footer: PageFooter) -> tuple[str, ...]:
    """Pure render: the footer's display lines; credit and note always together."""
    return (
        # The ADR-018 pair on ONE line — the credit is never emitted without
        # its non-commercial note beside it. The credit is a live markdown
        # link to the steward's site (Rusty Data branding): the link wraps
        # the unchanged credit text and never separates it from the note.
        f"[{footer.credit.credit_text}]({RUSTY_DATA_URL}) — {footer.credit.noncommercial_note}",
        footer.non_affiliation,
        " · ".join(footer.transparency_routes),
    )
