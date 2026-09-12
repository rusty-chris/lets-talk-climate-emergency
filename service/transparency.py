"""The four transparency pages — /about /privacy /sources /voices (issue #19).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_transparency_pages.py``,
``tests/unit/test_transparency_sources.py`` and
``tests/unit/test_service_transparency_routes.py`` pin the contract.

DESIGN §7.3's "credibility furniture", rendered ONCE at build/startup
from the project's single sources of truth — never hand-copied prose
that can drift from the code and manifests it describes:

- **Rendering format (decision, flagged in the #19 red-phase report):**
  server-rendered static HTML strings, assembled in this module and
  served by the FastAPI static routes with ``text/html``. No template
  engine, no markdown pipeline, no client JS — the pages are read-only
  furniture that must serve for $0 in the paused state, exactly like
  ``/health`` (DESIGN §9's read-only-not-dark rule).
- **/about** — what the product is (ADR-022 name + tagline), how it
  works honestly (retrieval → citations → runtime citation-support
  validation with "unverified" badges — worded to the validator's REAL
  semantics per review finding #251: factual sentences are checked when
  the pass runs, and when it cannot run the answer is delivered with no
  badges and flagged as unvalidated, never presented as checked), the
  Appendix-B guaranteed-vs-measured text VERBATIM plus the latest
  published eval numbers (``evals/RESULTS.md`` — a missing results file
  fails the build via :class:`TransparencyBuildError`, never renders
  blanks), the corpus-tier story with exclusions and reasons (the
  Ripple et al. line keyed to the RECORDED letters-sent state, review
  finding #254: "permission to be requested" until
  :func:`read_permission_letters_record` reads the checked-in
  ``letters/SENDING-RECORD.md`` as sent — the page never claims a
  request that has not been made), the NEB-campaign alignment framing, the
  ADR-018 non-commercial statement + Rusty Data credit (inseparable
  pair), and the §4.11 disclaimer verbatim. Issue #56: /about also
  mentions the thumbs feedback and what it is used for (improving the
  service through evaluation) — the red suite pins the mention, not
  exact prose.
- **/privacy** — DESIGN §9 made visible: the
  :data:`service.exchange_log.LOGGING_DISCLOSURE` line verbatim (the
  one-line chat disclosure is UNCHANGED by issue #56 — it is pinned
  verbatim to the DESIGN §9 sentence; the feedback disclosure is a
  SEPARATE added sentence, :data:`FEEDBACK_LOGGING_DISCLOSURE`,
  rendered verbatim in the "What we log" section — issue #56, edit
  documented there); the
  retention figures INTERPOLATED AT CALL TIME from
  ``service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS`` and
  ``service.rate_limit.IP_HASH_RETENTION_DAYS`` (module-attribute
  reads, so a retention-constant change re-renders here and can never
  silently diverge — pinned by monkeypatching the source modules);
  the hashed-IP explanation (rotating salt, never joined to query
  logs); the lawful basis ("legitimate interests"); the UK-GDPR contact
  point (:data:`PRIVACY_CONTACT_EMAIL` — owner ruling, 2026-09-12: filled
  to ``privacy@rustydata.ai``, an alias the owner creates on his domain
  before go-public; tracked outside this repo, not a boot precondition);
  and the data controller identity (:data:`DATA_CONTROLLER_NAME` "Rusty
  Data Ltd" with :data:`DATA_CONTROLLER_ICO_REGISTRATION` "ZB268445",
  the owner's ICO-registered limited company — same owner ruling —
  rendered beside the contact point).
- **/sources** — GENERATED from ``corpus/manifest.yaml`` +
  ``datasets/manifest.yaml`` (the single sources of truth): every
  active document with title / manifest-verbatim ``attribution_text`` /
  licence / ``canonical_url``, grouped by tier;
  ``permitted_context: non-commercial-educational`` entries carry their
  non-commercial note; the commented pending-source skeleton is NEVER
  listed (only real, signed-off entries); every dataset with its
  attribution, licence and fetch provenance (origin URL + access date —
  ADR-023: fetched at build, verified against pinned hashes, never
  committed). A manifest addition appears with ZERO code change.
- **/voices** — the DESIGN §2.5 Voices & action surface. The voices
  content (``voices/voices.yaml`` + first-party prose) MERGED with the
  owner's editorial sign-off (PR #198), so the published page renders
  it through the ``voices/render.py`` seam: every entity's name,
  one-liner and prose; links and person links as real ``href`` anchors;
  every snapshot fact WITH its ``as_of`` date (the deferred §2.5
  contract from the #19 red phase — a figure never renders undated);
  the "About the movement" attribution framing; and the issue #260
  prototype note (:data:`VOICES_PROTOTYPE_NOTE`, linking
  :data:`VOICES_PROTOTYPE_NOTE_ISSUE_URL`) — on /voices ONLY.
  ``build_transparency_pages`` reads the checked-in
  :data:`VOICES_CONTENT_PATH` at startup (``voices_path=None``); a
  missing or unparseable voices.yaml fails the build loudly
  (:class:`TransparencyBuildError` naming the path, the #249 pattern) —
  the "awaiting editorial sign-off" placeholder
  (:data:`VOICES_PLACEHOLDER_NOTICE`) is retired from the build and
  survives only as the pure-function ``voices_content=None`` state.
  The #255 invariant is PRESERVED across this supersession: provided
  content is never silently swallowed — it renders, or the call fails
  loudly; the placeholder never serves over content.

**Every page** carries: the ADR-018 steward-credit pair (credit text
never further than :data:`CREDIT_PAIR_MAX_SEPARATION` characters from
the non-commercial note — the pair is inseparable on the rendered
artefact, not just in the dataclasses; the credit is a live
:data:`RUSTY_DATA_URL` anchor beside the inline :data:`STEWARD_MARK_SVG`
mark — enrichment only, the pair pins are unchanged), the §4.11
non-affiliation disclaimer verbatim, and links to the other
transparency routes. The
constants here are deliberately DUPLICATED from ``ui.footer`` (the
service image must not import the UI package); the parity test pins
them equal, mirroring the wire-vocabulary parity pattern of
``tests/unit/test_ui_shell_hygiene.py`` — ``ui.footer`` is pure
(streamlit-free, enforced there), so the parity test imports it
without dragging streamlit into the service suite.

**No secrets, no identifiers:** the rendered pages are static per
build; they must never embed environment values (API keys), raw IPs, or
anything from :data:`service.exchange_log.FORBIDDEN_IDENTIFIER_FIELDS`.

Serving contract (``service.app``): ``ServiceDeps.transparency`` is an
optional :class:`TransparencyPages`; when provided, the four static
routes serve ITS html (both live and paused modes, ``text/html``, never
rate-limited, zero adapter calls, nothing written to the exchange log);
``None`` keeps the interim pre-#19 placeholders so the composed stack
stays serving until the implementation wires ``service.main`` to build
the real pages at startup.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PRODUCT_NAME",
    "PRODUCT_TAGLINE",
    "FEEDBACK_LOGGING_DISCLOSURE",
    "STEWARD_CREDIT_TEXT",
    "NONCOMMERCIAL_NOTE",
    "RUSTY_DATA_URL",
    "STEWARD_MARK_SVG",
    "DONATIONS_URL",
    "DONATIONS_NOTE_SUFFIX",
    "NON_AFFILIATION_DISCLAIMER",
    "TRANSPARENCY_ROUTES",
    "GUARANTEED_VS_MEASURED_TEXT",
    "PRIVACY_CONTACT_EMAIL",
    "DATA_CONTROLLER_NAME",
    "DATA_CONTROLLER_ICO_REGISTRATION",
    "VOICES_PLACEHOLDER_NOTICE",
    "VOICES_PROTOTYPE_NOTE",
    "VOICES_PROTOTYPE_NOTE_ISSUE_URL",
    "VOICES_CONTENT_PATH",
    "CREDIT_PAIR_MAX_SEPARATION",
    "PERMISSION_LETTERS_RECORD_PATH",
    "TransparencyBuildError",
    "TransparencyPages",
    "read_permission_letters_record",
    "render_about_page",
    "render_privacy_page",
    "render_sources_page",
    "render_voices_page",
    "render_footprint_page",
    "build_transparency_pages",
]

#: ADR-022: the product name, exact string (short form is UI chrome only).
PRODUCT_NAME = "Let's Talk About the Climate Emergency"

#: ADR-022 tagline, verbatim.
PRODUCT_TAGLINE = (
    "The emergency briefing you haven't had — answers from the science, with receipts."
)

#: ADR-018 pair — duplicated from ``ui.footer`` (see module doc); the
#: parity test pins these equal so the two images cannot drift.
STEWARD_CREDIT_TEXT = "Built by Rusty Data"
NONCOMMERCIAL_NOTE = "A free, open-source, non-commercial project."

#: Rusty Data branding — duplicated from ``ui.footer`` (parity-pinned):
#: every page's footer wraps the credit text in a live anchor on this URL.
#: The link ENRICHES the ADR-018 credit; the credit text is unchanged and
#: the non-commercial note stays inside the same footer paragraph.
RUSTY_DATA_URL = "https://rustydata.ai"

#: Donations gate — duplicated from ``ui.footer`` (parity-pinned) and OFF
#: until the owner supplies a real donation URL (no donation account
#: exists yet: nothing user-visible changes while this is None). Read at
#: call time by :func:`_page_footer` (the retention-constants pattern).
#: When enabled, the suffix joins the non-commercial note INSIDE the
#: ADR-018 pair — enabling donations can never split credit from note.
#: NOTE for the enable day: the verbatim NONCOMMERCIAL_NOTE page pins
#: will prompt a conscious wording review then.
DONATIONS_URL: str | None = None
DONATIONS_NOTE_SUFFIX = " — donations cover running costs"

#: The Rusty Data mark, inlined on every page so the pages stay
#: self-contained (no external requests — the pages' standing
#: convention). DUPLICATED byte-for-byte from the canonical asset at
#: ``ui/static/rusty_data_mark.svg`` (the service image never reads the
#: UI package's files at runtime); the parity test in
#: tests/unit/test_service_transparency_routes.py pins the two equal so
#: the marks cannot drift. Provenance is recorded inside the SVG: the
#: owner's own mark from his rusty_data_website repo — licence-clean.
STEWARD_MARK_SVG = """<!--
  Rusty Data brand mark: a hex nut (rusty hardware) holding a rust
  "data core" seated in its bore, on the dark botanical-ink tile.
  Provenance: copied from the owner's own website repo
  (rusty_data_website: src/app/icon.svg, the site favicon; cf.
  src/components/HexNutMark.tsx), with only footer-scale sizing
  (width/height), accessibility attributes (role/aria-label) and a
  namespaced gradient id added. Chris McWilliams owns both projects,
  so reuse of the mark here is licence-clean.
-->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"
     width="20" height="20" role="img" aria-label="Rusty Data">
  <defs>
    <radialGradient id="rusty-data-core" cx="38%" cy="34%" r="75%">
      <stop offset="0%" stop-color="#E0854F"/>
      <stop offset="100%" stop-color="#B85E31"/>
    </radialGradient>
  </defs>
  <rect width="200" height="200" rx="46" fill="#0C110E"/>
  <path d="M150,100 L125,56.7 L75,56.7 L50,100 L75,143.3 L125,143.3 Z
           M128,100 a28,28 0 1 0 -56,0 a28,28 0 1 0 56,0 Z"
        fill="#8FBFA4" fill-rule="evenodd"
        stroke="#8FBFA4" stroke-width="6" stroke-linejoin="round"/>
  <circle cx="100" cy="100" r="16" fill="url(#rusty-data-core)"/>
</svg>
"""

#: DESIGN §4.11, verbatim — on every transparency page.
NON_AFFILIATION_DISCLAIMER = (
    "Not affiliated with or endorsed by the National Emergency Briefing "
    "campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or "
    "the IPCC. All sources cited and linked."
)

#: The transparency routes; parity-pinned against
#: ``ui.footer.TRANSPARENCY_ROUTES`` (the #18 footer links these).
#: ``/footprint`` (the owner-approved footprint indicator,
#: docs/FOOTPRINT-METHODOLOGY.md) is the one route NOT rendered into the
#: startup-built :class:`TransparencyPages`: its headline totals move
#: with every answer, so the service renders it PER REQUEST from the
#: live ``service.footprint`` ledger (still static-honest: pure local
#: interpolation, zero adapter calls, serves in both modes).
TRANSPARENCY_ROUTES: tuple[str, ...] = (
    "/about",
    "/privacy",
    "/sources",
    "/voices",
    "/footprint",
)

#: DESIGN Appendix B, verbatim (markdown emphasis stripped) — the /about
#: guaranteed-vs-measured one-liner.
GUARANTEED_VS_MEASURED_TEXT = (
    "Guaranteed: every citation points to real text we retrieved from a "
    "named, clearly-licensed source, and every chart is rendered by our own "
    "code from named public datasets — the model writes neither the numbers "
    "nor the pixels. Measured (and published): how often cited text actually "
    "supports each sentence, how faithfully the science's calibrated "
    "uncertainty is preserved, and whether answers convey the severity the "
    "sources state — no more, no less. We show these numbers rather than "
    "claiming perfection."
)

#: Issue #56 — the /privacy feedback-collection sentence, VERBATIM on the
#: page (the "What we log" section). Every clause is enforced elsewhere in
#: code, so the sentence stays honest by construction: stored on the
#: record (``ExchangeLog.record_feedback``), no identifier (the record
#: shape pin), deleted with the record (the 90-day purge carries it),
#: stripped before eval promotion (``detach_for_harvest`` drops
#: ``feedback``). The chat surface's one-line
#: ``service.exchange_log.LOGGING_DISCLOSURE`` is deliberately UNCHANGED —
#: it is pinned verbatim to DESIGN §9's sentence; feedback disclosure is
#: this separate added sentence. Wording DECISION flagged in the #56
#: red-phase notes.
FEEDBACK_LOGGING_DISCLOSURE = (
    "If you rate an answer with the thumbs up/down buttons, the rating is "
    "stored on that conversation's anonymous log record — it adds no "
    "identifier, is deleted with the record, and is stripped out before "
    "any exchange is promoted into our published evaluation sets."
)

#: The UK-GDPR contact point on /privacy. FILLED (owner ruling,
#: 2026-09-12): ``privacy@rustydata.ai``, an alias on the controller's
#: domain — the owner creates the mailbox itself before the repo/site
#: goes public (tracked outside this repo; not a boot precondition). This
#: retires the NAMED placeholder the #19 red-phase report and the
#: deployment runbook flagged as an owner action. The page renders THIS
#: constant, so any future change is a one-line edit here and nowhere
#: else.
PRIVACY_CONTACT_EMAIL = "privacy@rustydata.ai"

#: The data controller named on /privacy (owner ruling, 2026-09-12):
#: Rusty Data Ltd, the owner's ICO-registered limited company. Rendered
#: beside :data:`PRIVACY_CONTACT_EMAIL` and :data:`DATA_CONTROLLER_ICO_REGISTRATION`.
DATA_CONTROLLER_NAME = "Rusty Data Ltd"

#: Rusty Data Ltd's ICO registration reference (owner ruling,
#: 2026-09-12), rendered beside the controller name on /privacy.
DATA_CONTROLLER_ICO_REGISTRATION = "ZB268445"

#: The honest /voices state while the voices content (PR #198) awaited
#: the owner's editorial sign-off. #198 has MERGED signed-off, so this
#: placeholder is RETIRED from the build (build_transparency_pages reads
#: :data:`VOICES_CONTENT_PATH` and fails loudly if it cannot); it
#: survives only as the pure-function ``voices_content=None`` state.
VOICES_PLACEHOLDER_NOTICE = (
    "The Voices & action content — the people and campaigns publicly "
    "communicating the climate emergency — is awaiting editorial sign-off "
    "and will appear here once approved."
)

#: Issue #260 — the prototype note the PUBLISHED /voices page carries
#: (and no other page): the movement descriptions are first-party prose
#: still under editorial review, with a link to the tracking issue.
#: WORDING DECISION flagged in the voices-route-wiring red-phase report
#: (the red suite pins the substance — "first-party" + "editorial
#: review" — plus this exact string verbatim on the page).
VOICES_PROTOTYPE_NOTE = (
    "Prototype note: the movement descriptions on this page are first-party "
    "prose written by this project and are still under editorial review — "
    "individual details may be corrected as they are verified."
)

#: Where the #260 prototype note links to (the review checklist issue).
VOICES_PROTOTYPE_NOTE_ISSUE_URL = (
    "https://github.com/rusty-chris/lets-talk-climate-emergency/issues/260"
)

#: The checked-in voices content (PR #198, owner-signed-off) that
#: ``build_transparency_pages`` reads at startup when ``voices_path`` is
#: None — mirroring :data:`PERMISSION_LETTERS_RECORD_PATH`. A missing or
#: unparseable file fails the build loudly (the #249 pattern), never a
#: silent fall-back to the placeholder.
VOICES_CONTENT_PATH = Path(__file__).resolve().parents[1] / "voices" / "voices.yaml"

#: ADR-018 on the rendered artefact: the credit text and the
#: non-commercial note must sit within this many characters of each
#: other in every page's HTML (the pair reads as one statement).
CREDIT_PAIR_MAX_SEPARATION = 200

#: Review finding #254 — the checked-in record of whether the
#: permission letters (``letters/*.md``, an ORCHESTRATION.md
#: stop-and-ask owner action) have actually been SENT. Mirrors the
#: severity-audit-packet owner gate (finding #197): a header line
#: ``permission_letters_sent: pending`` until the owner sends the
#: letters and flips it to ``permission_letters_sent: sent <YYYY-MM-DD>``.
#: The /about Ripple exclusion wording is keyed to this recorded state,
#: so the public page can never claim "permission requested" before the
#: request exists.
PERMISSION_LETTERS_RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "letters" / "SENDING-RECORD.md"
)


class TransparencyBuildError(Exception):
    """Building the transparency pages failed loudly (missing eval
    results file, unreadable manifest) — never a page with blanks."""


def read_permission_letters_record(record_path: Path | None = None) -> bool:
    """The recorded letters-sent state (review finding #254).

    RED-phase contract stub: raises ``NotImplementedError``; the failing
    suite in ``tests/unit/test_transparency_pages.py``
    (``TestRippleLettersGate``) pins the contract:

    - ``record_path`` ``None`` reads the checked-in
      :data:`PERMISSION_LETTERS_RECORD_PATH`.
    - A ``permission_letters_sent: pending`` header line returns
      ``False``; ``permission_letters_sent: sent <YYYY-MM-DD>`` returns
      ``True``.
    - A missing record, a record without the header line, or an unknown
      status value raises :class:`TransparencyBuildError` NAMING THE
      PATH — an absent or malformed record is never treated as sent
      (the #197 severity-audit discipline).
    """
    path = record_path if record_path is not None else PERMISSION_LETTERS_RECORD_PATH
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TransparencyBuildError(
            f"permission-letters sending record not found or unreadable at {path} "
            "— an absent record is NEVER treated as sent (review finding #254)"
        ) from exc
    header = next(
        (
            line.strip()
            for line in content.splitlines()
            if line.strip().startswith("permission_letters_sent:")
        ),
        None,
    )
    if header is None:
        raise TransparencyBuildError(
            f"permission-letters sending record at {path} carries no "
            "'permission_letters_sent:' header line — refusing to treat as sent"
        )
    status = header.split(":", 1)[1].strip()
    if status == "pending":
        return False
    if status.startswith("sent"):
        return True
    raise TransparencyBuildError(
        f"permission-letters sending record at {path} has an unrecognised status "
        f"{status!r} — expected 'pending' or 'sent <YYYY-MM-DD>', never treated as sent"
    )


@dataclass(frozen=True)
class TransparencyPages:
    """The four rendered pages, built once at startup and served as-is."""

    about_html: str
    privacy_html: str
    sources_html: str
    voices_html: str

    def as_route_map(self) -> dict[str, str]:
        """``{route: html}`` for the four STATIC startup-built pages.

        ``/footprint`` is deliberately absent: its totals are live, so
        the service renders it per request (see
        :data:`TRANSPARENCY_ROUTES`'s note and
        :func:`render_footprint_page`).
        """
        return {
            "/about": self.about_html,
            "/privacy": self.privacy_html,
            "/sources": self.sources_html,
            "/voices": self.voices_html,
        }


#: The tier order the /sources page renders documents in; only tiers
#: that actually carry documents get a heading.
_SOURCE_TIER_ORDER: tuple[str, ...] = ("A", "B", "C")

#: permitted_context values that carry the ADR-018 non-commercial note
#: beside the entry on /sources.
_NONCOMMERCIAL_CONTEXTS = frozenset({"non-commercial-educational"})


def _page_head(page_title: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(page_title)} — {html.escape(PRODUCT_NAME)}</title>\n"
        "</head>\n<body>\n"
    )


def _transparency_nav() -> str:
    """Links to every transparency route (a page linking itself is harmless
    and keeps the nav identical across pages)."""
    links = " · ".join(
        f'<a href="{route}">{html.escape(route.lstrip("/").capitalize())}</a>'
        for route in TRANSPARENCY_ROUTES
    )
    return f'<nav class="transparency-links">{links}</nav>\n'


def _page_footer() -> str:
    """The every-page furniture: the ADR-018 pair (adjacent, inseparable —
    now with the inline Rusty Data mark and a live rustydata.ai anchor
    wrapping the unchanged credit text), the §4.11 disclaimer verbatim,
    and the transparency route links."""
    # Donations gate, read at call time (the retention-constants pattern):
    # OFF by default — the suffix appears only once the owner fills
    # DONATIONS_URL, and it joins the note INSIDE the pair paragraph.
    if DONATIONS_URL:
        note = NONCOMMERCIAL_NOTE.rstrip(".") + DONATIONS_NOTE_SUFFIX + "."
    else:
        note = NONCOMMERCIAL_NOTE
    return (
        "<footer>\n"
        # The ADR-018 credit and non-commercial note render as ONE
        # statement — the pair is never split on the artefact. The mark is
        # inline SVG (self-contained, zero external requests) and the
        # credit is a real anchor on the steward's site.
        f'<p class="steward-credit">{STEWARD_MARK_SVG}'
        f'<a href="{RUSTY_DATA_URL}">{html.escape(STEWARD_CREDIT_TEXT)}</a> — '
        f"{html.escape(note)}</p>\n"
        f'<p class="non-affiliation">{html.escape(NON_AFFILIATION_DISCLAIMER)}</p>\n'
        f"{_transparency_nav()}"
        "</footer>\n</body>\n</html>\n"
    )


def render_about_page(
    *,
    eval_results_text: str,
    corpus_vintage: str,
    permission_letters_sent: bool = False,
) -> str:
    """Pure: the /about HTML (see module doc for the pinned content).

    ``eval_results_text`` is the published ``evals/RESULTS.md`` content
    (verdict + gate numbers surface on the page); ``corpus_vintage``
    renders in the "answers reflect sources as of" line.

    ``permission_letters_sent`` (review finding #254) is the recorded
    letters-sent state (:func:`read_permission_letters_record`): while
    ``False`` — the default, and the state until the owner sends the
    letters — the Ripple et al. exclusion reads "permission to be
    requested — kept link-only until written permission is granted";
    only ``True`` renders the DESIGN §7.3 "permission requested"
    wording. The page must never claim a request that has not been
    made.
    """
    # #254: the Ripple exclusion reason follows the RECORDED letters-sent
    # state — the page never asserts a request that has not been made.
    if permission_letters_sent:
        ripple_reason = "permission requested — kept link-only until written permission is granted"
    else:
        ripple_reason = (
            "permission to be requested — kept link-only until written permission is granted"
        )
    return (
        _page_head("About")
        + "<main>\n"
        + "<h1>About this briefing</h1>\n"
        + f"<p>{html.escape(PRODUCT_NAME)} — {html.escape(PRODUCT_TAGLINE)}</p>\n"
        + "<h2>How it works</h2>\n"
        # #251: worded to the #13 validator's REAL semantics — checked WHEN
        # the pass runs; when it cannot run, delivered with no badges and
        # flagged as unvalidated, never presented as checked.
        + "<p>We retrieve passages from a named, clearly-licensed corpus, "
        "then generate an answer that cites the source text it draws on. "
        "Factual sentences are checked by a runtime citation-support "
        "validation pass when it runs: any sentence the cited passages do "
        "not actually support is badged &quot;unverified&quot; rather than "
        "presented as settled. When that validation pass cannot run, the "
        "answer is delivered with no badges and is flagged as unvalidated — "
        "never presented as checked. Charts are rendered by our own code "
        "from named public datasets — the model writes neither the numbers "
        "nor the pixels.</p>\n"
        + "<h2>Your feedback</h2>\n"
        # Issue #56: /about mentions the thumbs feedback and what it is for.
        + "<p>You can rate any answer with the thumbs up and thumbs down "
        "buttons. We use this feedback to improve the service — a rating "
        "helps us find the answers that fell short and evaluate them. Your "
        "rating is stored anonymously and adds no identifier; see the "
        '<a href="/privacy">privacy notice</a> for the detail.</p>\n'
        + "<h2>Guaranteed vs measured</h2>\n"
        + f"<p>{html.escape(GUARANTEED_VS_MEASURED_TEXT)}</p>\n"
        + "<h2>Latest published evaluation results</h2>\n"
        + "<p>We publish these numbers rather than claiming perfection; they "
        "come straight from our evaluation run:</p>\n"
        + f"<pre>{html.escape(eval_results_text)}</pre>\n"
        + "<h2>What is not in the corpus, and why</h2>\n"
        + "<ul>\n"
        + f"<li>Ripple et al.: {html.escape(ripple_reason)}.</li>\n"
        + "<li>IPCC AR6 full text: link-only pending a licensing check.</li>\n"
        + "</ul>\n"
        + "<h2>Our mission</h2>\n"
        + "<p>This project aligns with the National Emergency Briefing "
        "campaign (nebriefing.org). It is a free, open-source, "
        "non-commercial public-education project — see the "
        "non-affiliation note below.</p>\n"
        + f"<p>Answers reflect the sources as of {html.escape(corpus_vintage)}.</p>\n"
        + "</main>\n"
        + _page_footer()
    )


def _render_document(document: Mapping[str, Any]) -> str:
    """One corpus document as a /sources list item, all fields manifest-verbatim."""
    title = html.escape(str(document.get("title", "")))
    attribution = html.escape(str(document.get("attribution_text", "")))
    licence = html.escape(str(document.get("licence", "")))
    canonical_url = str(document.get("canonical_url", ""))
    url_escaped = html.escape(canonical_url)
    parts = [
        "<li>\n",
        f"<strong>{title}</strong><br>\n",
        f'<span class="attribution">{attribution}</span><br>\n',
        f"Licence: {licence}<br>\n",
        f'<a href="{url_escaped}">{url_escaped}</a>\n',
    ]
    if document.get("permitted_context") in _NONCOMMERCIAL_CONTEXTS:
        # ADR-018: the non-commercial nature is load-bearing — say so beside
        # the entry, using the literal "non-commercial" the pin looks for.
        parts.append(
            '<br><span class="nc-note">Licensed for non-commercial, educational use only.</span>\n'
        )
    parts.append("</li>\n")
    return "".join(parts)


#: #250 — the marker every provisional (``permitted_context != "open"``)
#: dataset carries beside its attribution; the pack invariant excludes it
#: from charts until written licence confirmation lands (issue #23).
DATASET_PENDING_MARKER = "licence confirmation pending — not used in charts"
#: #250 — the two /sources dataset section headings.
DATASET_PACK_HEADING = "Chart datasets"
DATASET_PROVISIONAL_HEADING = "Datasets under licence confirmation"


def _render_provenance(segments: list[Mapping[str, Any]]) -> str:
    """The manifest's per-segment ``provenance`` block, rendered so a
    licence text's "see provenance below" reference never dangles (#250):
    each segment names its origin, period and credit."""
    items = []
    for segment in segments:
        origin = html.escape(str(segment.get("origin", "")))
        period = html.escape(str(segment.get("period", "")))
        credit = html.escape(str(segment.get("credit", "")))
        items.append(f"<li>{origin} ({period}) — credit: {credit}</li>\n")
    return '<div class="provenance">Provenance:<ul>\n' + "".join(items) + "</ul></div>\n"


def _render_dataset(entry: Mapping[str, Any]) -> str:
    """One chart dataset as a /sources list item, with fetch provenance.

    ``permitted_context != "open"`` entries carry the #250 pending marker
    beside their attribution; entries with a manifest ``provenance`` block
    render it so no "see provenance" reference dangles.
    """
    attribution = html.escape(str(entry.get("attribution_text", "")))
    licence = html.escape(str(entry.get("licence", "")))
    url = str(entry.get("url", ""))
    url_escaped = html.escape(url)
    parts = [
        "<li>\n",
        f'<span class="attribution">{attribution}</span><br>\n',
    ]
    if entry.get("permitted_context") != "open":
        parts.append(
            f'<span class="pending-marker">{html.escape(DATASET_PENDING_MARKER)}</span><br>\n'
        )
    # The provenance block renders BESIDE the attribution (before the — often
    # long — licence text), so a "see provenance below" reference can never
    # dangle far from the entry it belongs to (#250).
    segments = list(entry.get("provenance") or [])
    if segments:
        parts.append(_render_provenance(segments))
    parts.extend(
        [
            f"Licence: {licence}<br>\n",
            f'Fetched from <a href="{url_escaped}">{url_escaped}</a> '
            "and verified against a pinned sha256 hash.\n",
        ]
    )
    parts.append("</li>\n")
    return "".join(parts)


def render_privacy_page(
    *,
    contact_email: str = PRIVACY_CONTACT_EMAIL,
    controller_name: str = DATA_CONTROLLER_NAME,
    controller_ico_registration: str = DATA_CONTROLLER_ICO_REGISTRATION,
) -> str:
    """Pure: the /privacy HTML.

    Retention figures are read from
    ``service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS`` and
    ``service.rate_limit.IP_HASH_RETENTION_DAYS`` as module attributes
    AT CALL TIME (the no-silent-divergence pin), never hand-copied.

    ``controller_name`` / ``controller_ico_registration`` (owner ruling,
    2026-09-12) name the data controller — Rusty Data Ltd, ICO
    registration reference ZB268445 by default — in the same paragraph
    as ``contact_email``, mirroring the contact-email override pattern.
    """
    # Read the retention figures from the source modules AT CALL TIME, via
    # the module objects, so a constant change (or a test's monkeypatch)
    # re-renders here — the numbers can never silently diverge from the
    # code that enforces them.
    import service.exchange_log as exchange_log
    import service.rate_limit as rate_limit

    exchange_days = exchange_log.EXCHANGE_LOG_RETENTION_DAYS
    ip_hash_days = rate_limit.IP_HASH_RETENTION_DAYS
    contact = html.escape(contact_email)
    controller = html.escape(controller_name)
    controller_registration = html.escape(controller_ico_registration)
    return (
        _page_head("Privacy")
        + "<main>\n"
        + "<h1>Privacy notice</h1>\n"
        + f"<p>{html.escape(exchange_log.LOGGING_DISCLOSURE)}</p>\n"
        + "<h2>What we log, and why</h2>\n"
        + "<p>We log your question and our answer, the retrieved source "
        "chunks and the citations, plus usage and cost — anonymously, to "
        "operate the service and improve it through evaluation. We store no "
        "IP address, cookie, account or other identifier alongside a "
        "conversation.</p>\n"
        # Issue #56: the feedback-collection sentence, verbatim.
        + f"<p>{html.escape(FEEDBACK_LOGGING_DISCLOSURE)}</p>\n"
        + "<h2>Retention</h2>\n"
        + f"<p>Exchange logs are kept for {exchange_days} days, then "
        "permanently deleted. Rate-limiting keeps only hashed request "
        f"counts, held for at most {ip_hash_days} days. One exception "
        "applies to the deletion bound, disclosed next: a few exchanges "
        "may be promoted into our published evaluation sets. Before "
        "promotion each is hand-reviewed; any exchange containing personal "
        "details is excluded entirely, and promoted content is irreversibly "
        "detached from its timestamps and identifiers first. Because the "
        "detached, anonymised excerpts no longer reference any conversation, "
        "they may then be retained beyond the deletion bound above.</p>\n"
        + "<h2>How rate-limiting handles IP addresses</h2>\n"
        + "<p>To enforce a per-visitor request limit we compute a hash of "
        "the request IP using a rotating salt. The hash is never joined to "
        "the conversation logs, and no raw IP address is ever stored.</p>\n"
        + "<h2>Lawful basis</h2>\n"
        + "<p>We process this data under our legitimate interests (UK GDPR "
        "Article 6(1)(f)) in running and improving an anonymous "
        "public-education service.</p>\n"
        + "<h2>Your rights and contact</h2>\n"
        + f"<p>The data controller for this service is {controller} "
        f"(ICO registration reference {controller_registration}). Under the "
        f"UK GDPR you can contact us at {contact}. You may "
        "also complain to the Information Commissioner's Office (the ICO), "
        "the UK's data-protection regulator.</p>\n" + "</main>\n" + _page_footer()
    )


def render_sources_page(
    *,
    corpus_manifest: Mapping[str, Any],
    datasets_manifest: Mapping[str, Any],
    corpus_vintage: str,
) -> str:
    """Pure: the /sources HTML, generated from the parsed manifests.

    ``corpus_manifest`` / ``datasets_manifest`` are the
    ``yaml.safe_load`` results of the two manifest files (the raw
    mappings — titles included; the commented pending-source skeleton is
    invisible to YAML and therefore can never leak onto the page).
    """
    documents = list(corpus_manifest.get("documents") or [])
    datasets = dict(datasets_manifest.get("datasets") or {})
    access_date = str(datasets_manifest.get("access_date", ""))

    body = [
        _page_head("Sources"),
        "<main>\n",
        "<h1>Source library</h1>\n",
        "<p>Every answer is grounded in this named, clearly-licensed corpus. "
        "This page is generated directly from our source manifests, so a new "
        "source appears here the moment its entry lands; entries awaiting "
        "written licence confirmation are marked as such.</p>\n",
        f"<p>Answers reflect the sources as of {html.escape(corpus_vintage)}.</p>\n",
        "<h2>Corpus documents</h2>\n",
    ]
    # Group by manifest tier, in a fixed order; only tiers that carry
    # documents get a heading.
    by_tier: dict[str, list[Mapping[str, Any]]] = {}
    for document in documents:
        by_tier.setdefault(str(document.get("source_tier", "")), []).append(document)
    # #253: a document whose tier is outside the rendered order would be
    # grouped under an unknown key and silently never rendered — an active,
    # cited document vanishing from the public attribution surface. Fail
    # LOUDLY instead, naming every offending document and its tier.
    unknown_tiers = [
        (str(document.get("id", "")), tier)
        for tier, tier_documents in by_tier.items()
        if tier not in _SOURCE_TIER_ORDER
        for document in tier_documents
    ]
    if unknown_tiers:
        offenders = ", ".join(f"{doc_id} (source_tier {tier!r})" for doc_id, tier in unknown_tiers)
        raise TransparencyBuildError(
            "transparency build failed: corpus document(s) carry a source_tier "
            f"outside the rendered order {_SOURCE_TIER_ORDER}: {offenders} — a tier "
            "typo must never silently drop an attribution from /sources"
        )
    for tier in _SOURCE_TIER_ORDER:
        tier_documents = by_tier.get(tier)
        if not tier_documents:
            continue
        body.append(f"<h3>Tier {html.escape(tier)}</h3>\n<ul>\n")
        body.extend(_render_document(document) for document in tier_documents)
        body.append("</ul>\n")

    # #250: the chart-pack section names ONLY datasets the charts are
    # actually built from (``in_chart_pack: true``); everything else — the
    # open-provisional entries awaiting written licence confirmation (#23) —
    # renders under its own honest heading below, never masquerading as a
    # charted source.
    pack_entries = [entry for entry in datasets.values() if entry.get("in_chart_pack")]
    provisional_entries = [entry for entry in datasets.values() if not entry.get("in_chart_pack")]
    body.append(f"<h2>{html.escape(DATASET_PACK_HEADING)}</h2>\n")
    body.append(
        "<p>Chart datasets are fetched at build time from their origin URLs "
        "and verified against pinned sha256 hashes (ADR-023); the raw data "
        f"files are never committed. Access date: {html.escape(access_date)}.</p>\n"
    )
    body.append("<ul>\n")
    body.extend(_render_dataset(entry) for entry in pack_entries)
    body.append("</ul>\n")
    if provisional_entries:
        body.append(f"<h2>{html.escape(DATASET_PROVISIONAL_HEADING)}</h2>\n")
        body.append(
            "<p>These datasets are not used in any chart. Their licences are "
            "not yet confirmed in writing — the pack invariant excludes them "
            "until the written confirmation lands (issue #23).</p>\n"
        )
        body.append("<ul>\n")
        body.extend(_render_dataset(entry) for entry in provisional_entries)
        body.append("</ul>\n")
    body.append("</main>\n")
    body.append(_page_footer())
    return "".join(body)


def _render_voices_entity(entity: Any) -> str:
    """One entity as a /voices section: name, one-liner, prose paragraphs,
    every snapshot fact WITH its ``as_of`` date, named people and entity
    links as real ``href`` anchors. Every YAML-sourced string is escaped."""
    parts = [
        f'<section id="{html.escape(entity.id)}">\n',
        f"<h2>{html.escape(entity.name)}</h2>\n",
        f"<p>{html.escape(entity.one_liner)}</p>\n",
    ]
    for paragraph in entity.prose.strip().split("\n\n"):
        text = " ".join(paragraph.split())
        if text:
            parts.append(f"<p>{html.escape(text)}</p>\n")
    for fact in entity.snapshot_facts:
        # §2.5: rendered_sentence() carries the figure AND its as_of date
        # in one sentence, so a snapshot fact never renders undated.
        parts.append(f"<p>{html.escape(fact.rendered_sentence())}</p>\n")
    for person in entity.people:
        parts.append(
            f"<p>{html.escape(person.name)} — {html.escape(person.one_liner)} "
            f'(<a href="{html.escape(person.link)}">{html.escape(person.link)}</a>)</p>\n'
        )
    if entity.links:
        items = "".join(
            f'<li><a href="{html.escape(link["url"])}">{html.escape(link["label"])}</a></li>\n'
            for link in entity.links
        )
        parts.append(f"<ul>\n{items}</ul>\n")
    parts.append("</section>\n")
    return "".join(parts)


def render_voices_page(*, voices_content: Any | None = None) -> str:
    """Pure: the /voices HTML.

    CONTRACT (voices-route wiring; PR #198 merged with the owner's
    editorial sign-off — pinned by the red suite in
    ``tests/unit/test_transparency_voices_route.py``):

    - ``voices_content`` a ``voices.render.VoicesLibrary`` (the
      ``load_voices`` result) renders the PUBLISHED page: every entity's
      name, one-liner and prose paragraphs; entity links and person
      links as real ``href`` anchors; every snapshot fact rendered WITH
      its ``as_of`` date (the deferred §2.5 contract — a figure never
      renders undated); the "About the movement" attribution framing;
      the issue #260 :data:`VOICES_PROTOTYPE_NOTE` verbatim with a link
      to :data:`VOICES_PROTOTYPE_NOTE_ISSUE_URL`; and NO placeholder
      notice. Every YAML-sourced string is HTML-escaped.
    - ``None`` still renders the flagged placeholder around
      :data:`VOICES_PLACEHOLDER_NOTICE` (a pure-function state, retired
      from the build — ``build_transparency_pages`` never reaches it).
    - Any other non-None value raises ``TypeError`` naming
      ``VoicesLibrary``.

    The review-finding #255 invariant is PRESERVED across this
    supersession (its raises-``NotImplementedError``-naming-#198 pin is
    superseded now the seam exists): provided content is never silently
    swallowed — it renders, or the call fails loudly; the placeholder
    never serves over content.
    """
    if voices_content is None:
        return (
            _page_head("Voices")
            + "<main>\n"
            + "<h1>Voices &amp; action</h1>\n"
            + f"<p>{html.escape(VOICES_PLACEHOLDER_NOTICE)}</p>\n"
            + "<p>Voices are about the movement — the people and campaigns "
            "publicly communicating the climate emergency — and are kept "
            "structurally separate from the assessed scientific evidence: they "
            "are never treated as scientific support for a factual claim.</p>\n"
            + "</main>\n"
            + _page_footer()
        )

    # Local import: the service image must not carry a hard top-level
    # dependency on the voices package for its unrelated routes (mirrors
    # the lazy ``yaml``/``service.exchange_log`` imports elsewhere here).
    from voices.render import VoicesLibrary

    if not isinstance(voices_content, VoicesLibrary):
        raise TypeError(
            "render_voices_page: voices_content must be a voices.render.VoicesLibrary "
            f"(or None for the placeholder) — got {type(voices_content).__name__}; "
            "refusing to silently serve the placeholder over unrecognised content (#255)"
        )

    library = voices_content
    body = [
        _page_head("Voices"),
        "<main>\n",
        "<h1>Voices &amp; action</h1>\n",
        f'<p class="voices-prototype-note">{html.escape(VOICES_PROTOTYPE_NOTE)} '
        f'See <a href="{VOICES_PROTOTYPE_NOTE_ISSUE_URL}">issue #260</a>.</p>\n',
        f"<p>{html.escape(library.attribution_text)}</p>\n",
    ]
    body.extend(_render_voices_entity(entity) for entity in library.entities)
    body.append("</main>\n")
    body.append(_page_footer())
    return "".join(body)


def build_transparency_pages(
    *,
    corpus_manifest_path: Path,
    datasets_manifest_path: Path,
    eval_results_path: Path,
    corpus_vintage: str,
    contact_email: str = PRIVACY_CONTACT_EMAIL,
    controller_name: str = DATA_CONTROLLER_NAME,
    controller_ico_registration: str = DATA_CONTROLLER_ICO_REGISTRATION,
    voices_path: Path | None = None,
    letters_record_path: Path | None = None,
) -> TransparencyPages:
    """Load the sources of truth and render all four pages (startup step).

    Raises :class:`TransparencyBuildError` NAMING THE PATH when
    ``eval_results_path`` does not exist (a release without published
    eval results must fail the build, not render blanks — issue #19
    acceptance criterion) or when a manifest cannot be loaded.

    ``controller_name`` / ``controller_ico_registration`` (owner ruling,
    2026-09-12) thread straight through to :func:`render_privacy_page`;
    the defaults are Rusty Data Ltd / ZB268445.

    ``letters_record_path`` (review finding #254): the permission-letters
    sending record, read via :func:`read_permission_letters_record`
    (``None`` reads the checked-in
    :data:`PERMISSION_LETTERS_RECORD_PATH`); its recorded state is
    threaded into :func:`render_about_page` as
    ``permission_letters_sent``, so the /about Ripple wording follows
    the record, never an assumption.

    ``voices_path`` (voices-route wiring — REPLACES the pre-#198
    ``voices_content`` parameter; the #255 supersession is documented on
    :func:`render_voices_page`): the voices.yaml to load through
    ``voices.render.load_voices`` and render as the published /voices
    page. ``None`` — the startup default ``service.main`` uses — reads
    the checked-in :data:`VOICES_CONTENT_PATH`. A MISSING or unparseable
    voices.yaml raises :class:`TransparencyBuildError` NAMING THE PATH
    (the #249 pattern): now the signed-off content exists, silently
    serving the "awaiting editorial sign-off" placeholder would be a
    false statement on the public artefact.
    """
    import yaml

    if not eval_results_path.is_file():
        raise TransparencyBuildError(
            "transparency build failed: published eval results file not found "
            f"at {eval_results_path} — a release without eval results must fail "
            "the build, not render blank numbers"
        )
    try:
        eval_results_text = eval_results_path.read_text(encoding="utf-8")
        corpus_manifest = yaml.safe_load(corpus_manifest_path.read_text(encoding="utf-8"))
        datasets_manifest = yaml.safe_load(datasets_manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TransparencyBuildError(
            f"transparency build failed loading a source of truth: {exc}"
        ) from exc

    # #254: the /about Ripple wording follows the RECORDED letters-sent
    # state, read from the checked-in sending record — never an assumption.
    permission_letters_sent = read_permission_letters_record(letters_record_path)

    # Voices-route wiring: the checked-in voices.yaml is now the build's
    # source of truth (voices_path=None mirrors letters_record_path). A
    # missing or unparseable file fails the build LOUDLY, naming the path
    # (the #249 pattern) — the "awaiting editorial sign-off" placeholder
    # must never serve over content that in fact exists but failed to load.
    from voices.render import VoicesError, load_voices

    resolved_voices_path = voices_path if voices_path is not None else VOICES_CONTENT_PATH
    try:
        voices_library = load_voices(resolved_voices_path)
    except (OSError, yaml.YAMLError, VoicesError) as exc:
        raise TransparencyBuildError(
            f"transparency build failed loading the voices content at {resolved_voices_path}: {exc}"
        ) from exc

    return TransparencyPages(
        about_html=render_about_page(
            eval_results_text=eval_results_text,
            corpus_vintage=corpus_vintage,
            permission_letters_sent=permission_letters_sent,
        ),
        privacy_html=render_privacy_page(
            contact_email=contact_email,
            controller_name=controller_name,
            controller_ico_registration=controller_ico_registration,
        ),
        sources_html=render_sources_page(
            corpus_manifest=corpus_manifest,
            datasets_manifest=datasets_manifest,
            corpus_vintage=corpus_vintage,
        ),
        voices_html=render_voices_page(voices_content=voices_library),
    )


def render_footprint_page(*, totals: Any | None) -> str:
    """Pure: the /footprint page HTML (owner-approved methodology,
    docs/FOOTPRINT-METHODOLOGY.md §5/§9 — BINDING).

    RED-phase contract stub: raises ``NotImplementedError``; the failing
    suite in ``tests/unit/test_footprint_page.py`` pins the contract.

    ``totals`` is a ``service.footprint.FootprintTotals`` (the live
    aggregate the service reads per request — this page is the one
    transparency surface rendered per GET, not at startup) or ``None``
    when the ledger journal is unreadable — the honest unavailable
    state.

    Pinned structure (§9's eight sections, in order):

    1. **Headline totals** — application lifetime energy and carbon as
       est. ranges (kWh / kg CO2e) "since {totals.since}, over N
       answers", every figure labelled *estimated*; beside them the ONE
       measured line: the accumulated retrieval CPU-hours (measured) ≈
       its est. Wh range. ``totals=None`` renders
       ``service.footprint.FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE`` instead
       — NEVER silent zeros presented as totals.
    2. **Measured / estimated / unknown** — the §1 three-column honesty
       table (token counts measured; local CPU-seconds measured with an
       estimated wattage conversion; Anthropic energy-per-token unknown
       → estimated range; serving region unknown → disclosed
       assumption; Hetzner electricity supplier-claimed).
    3. **How the estimate is built** — the §2 formula in prose plus the
       §3 constants INTERPOLATED from ``service.footprint`` (module
       attributes at call time, the retention-constants pattern — the
       published figures can never drift from the code that computes
       them), each with central value, range, and provenance.
    4. The §9.4 uncertainty statement,
       ``ANTHROPIC_UNCERTAINTY_PARAGRAPH`` VERBATIM.
    5. **The two grids** — ``ANTHROPIC_GRID_ASSUMPTION_SENTENCE`` and
       ``MARKET_VS_LOCATION_SENTENCE`` VERBATIM.
    6. **Equivalents** — the three §8 anchors (video streaming / metres
       driven / mugs of tea), each with its source named inline
       (IEA/Kamiya; US EPA; first-principles kettle physics).
    7. **What is not counted** — the §7 exclusions: training
       (``TRAINING_EXCLUSION_PHRASE``), index build-time compute,
       network transfer and the visitor's device, Anthropic water use
       (unknown), serving-hardware embodied carbon.
    8. **Revision note** — ``FOOTPRINT_FACTORS_VERSION`` rendered, so a
       factor update is a visible, dated event.

    Every-page invariants apply exactly as on the other four surfaces
    (the ADR-018 credit/non-commercial pair adjacent within
    :data:`CREDIT_PAIR_MAX_SEPARATION`, the §4.11 disclaimer verbatim,
    the transparency nav, the Rusty Data anchor + inline mark — i.e.
    rendered through the same ``_page_footer`` furniture); no secrets,
    no identifiers, nothing user-derived — ``totals`` is counts only.
    """
    # Import the estimation model at CALL TIME (the retention-constants
    # pattern): the §3 factors and verbatim sentences are read as module
    # attributes on every render, so the published figures can never drift
    # from the code that computes them (a factor change re-renders here).
    import service.footprint as fp

    body: list[str] = [_page_head("Footprint"), "<main>\n"]
    body.append("<h1>Energy &amp; carbon footprint</h1>\n")
    body.append(
        "<p>Every answer costs energy. Here is our honest estimate — every "
        "figure labelled measured, estimated, or unknown, and estimates "
        "always shown as ranges.</p>\n"
    )

    # 1. Headline totals (live per request) — application lifetime.
    body.append("<h2>Since we launched</h2>\n")
    if totals is None:
        notice = html.escape(fp.FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE)
        body.append(f'<p class="totals-unavailable">{notice}</p>\n')
    else:
        lifetime_api = fp.api_energy_wh(
            [
                {
                    "input_tokens": totals.input_tokens,
                    "output_tokens": totals.output_tokens,
                    "cache_read_input_tokens": totals.cache_read_input_tokens,
                    "cache_creation_input_tokens": totals.cache_creation_input_tokens,
                }
            ]
        )
        lifetime_local = fp.local_energy_wh(totals.cpu_seconds)
        lifetime_co2e = fp.co2e_grams(lifetime_api, lifetime_local)
        total_energy = fp.sum_wh_ranges([lifetime_api, lifetime_local])
        cpu_hours = totals.cpu_seconds / 3600
        body.append(
            "<p>Estimated energy: "
            f"<strong>est. {_kwh(total_energy.low)}–{_kwh(total_energy.high)} kWh</strong> "
            f"(central est. {_kwh(total_energy.central)} kWh); estimated carbon: "
            f"<strong>est. {_kg(lifetime_co2e.low)}–{_kg(lifetime_co2e.high)} kg CO2e</strong> "
            f"(central est. {_kg(lifetime_co2e.central)} kg CO2e) — since "
            f"{html.escape(str(totals.since))}, over {totals.exchanges:,} answers. "
            "Every figure here is <em>estimated</em>.</p>\n"
        )
        body.append(
            "<p>Of which our own server's retrieval compute: "
            f"<strong>{cpu_hours:.1f} CPU-hours (measured)</strong> "
            f"≈ est. {_kwh(lifetime_local.low)}–{_kwh(lifetime_local.high)} kWh — "
            "measured CPU time, estimated wattage.</p>\n"
        )

    # 2. The measured / estimated / unknown honesty table (the centrepiece).
    body.append("<h2>What we measure, estimate, and cannot know</h2>\n")
    body.append(
        "<table><thead><tr>"
        "<th>Measured</th><th>Estimated</th><th>Unknown</th>"
        "</tr></thead><tbody>\n"
    )
    body.append(
        "<tr>"
        "<td>Token counts per answer — input, output and cache tokens for "
        "every model call — are provider-reported facts (measured). Our "
        "server's retrieval CPU time is measured on the box.</td>"
        "<td>Everything that converts tokens and CPU-seconds into watt-hours "
        "and grams of CO2e: the per-token energy factors, the per-vCPU "
        "wattage, the grid carbon intensity — each a sourced, ranged "
        "estimate. Our local slice is "
        "<strong>measured CPU time × estimated per-vCPU wattage</strong>.</td>"
        "<td>The energy Anthropic's servers actually spend per token is "
        "<strong>unknown</strong> — Anthropic publishes none. The serving "
        "region (and so the grid) is unknown; we disclose our assumption. "
        "Anthropic's water use is unknown, and stays in this column rather "
        "than an anchor.</td>"
        "</tr>\n"
    )
    body.append("</tbody></table>\n")

    # 3. How the estimate is built — the §2 formula + §3 constants,
    # interpolated from the factor module (never hand-copied figures).
    body.append("<h2>How the estimate is built</h2>\n")
    body.append(
        "<p>Per answer we sum, over every metered model call: input tokens "
        "(and cache-write tokens) × an input-energy factor, cache-read "
        "tokens × that factor × a cache-read fraction, and output tokens × "
        "an output-energy factor. Local energy is measured CPU-seconds × a "
        "per-vCPU wattage × the data-centre PUE. Carbon is energy × grid "
        "carbon intensity, applied on both supply chains (location-based). "
        "Low and high bounds propagate the low/high end of every factor "
        "together, so the range is an honest propagation, not a point "
        "estimate with decoration.</p>\n"
    )
    body.append(
        "<table><thead><tr><th>Factor</th><th>Central</th><th>Range</th><th>Provenance</th></tr></thead><tbody>\n"
    )
    body.append(
        _factor_row(
            "Output energy (Wh / 1k output tokens, Haiku-class)",
            fp.E_OUT_WH_PER_1K,
            "ML.ENERGY measurements, Google's production median, Epoch AI's "
            "GPT-4o estimate, and Jegham et al. (Claude 3.7 Sonnet), scaled "
            "for a Haiku-class model.",
        )
    )
    body.append(
        _factor_row(
            "Input energy (Wh / 1k input tokens, incl. cache writes)",
            fp.E_IN_WH_PER_1K,
            "Jegham et al.-derived output:input energy ratio (~21:1), "
            "cross-checked against EcoLogits.",
        )
    )
    body.append(
        _factor_row(
            "Cache-read fraction (of an uncached input token)",
            fp.CACHE_READ_FACTOR,
            "Prefill-residual engineering estimate — explicitly NOT "
            "Anthropic's billing ratio (a price, not an energy measurement).",
        )
    )
    body.append(
        _factor_row(
            "Anthropic grid carbon intensity (gCO2e/kWh)",
            fp.CIF_API_G_PER_KWH,
            "US grid average (Ember 2024); low end AWS-region-weighted "
            "(Jegham et al.), high end a fossil-heavier-region margin.",
        )
    )
    body.append(
        _factor_row(
            "Per-vCPU wattage (W, at typical utilisation)",
            fp.W_PER_VCPU,
            "Cloud Carbon Footprint per-vCPU coefficients across AWS/GCP/Azure.",
        )
    )
    body.append(
        _factor_row(
            "Hetzner data-centre PUE",
            fp.PUE_HETZNER,
            "Hetzner's supplier-published average PUE (1.13).",
        )
    )
    body.append("</tbody></table>\n")
    body.append(
        "<p>Non-default models scale these factors and are flagged as the "
        "weakest numbers in the table: a Sonnet best-mode answer ×2, an "
        f"Opus best-mode answer ×{_num(fp.OPUS_ENERGY_MULTIPLIER_LOW)}–"
        f"{_num(fp.OPUS_ENERGY_MULTIPLIER_HIGH)} — an "
        "<em>extrapolated</em> widening (Jegham et al. never measured "
        "Opus), never a measured figure.</p>\n"
    )

    # 4. The biggest uncertainty, stated plainly (verbatim).
    body.append("<h2>The biggest uncertainty</h2>\n")
    body.append(f"<p>{html.escape(fp.ANTHROPIC_UNCERTAINTY_PARAGRAPH)}</p>\n")

    # 5. The two grids (verbatim).
    body.append("<h2>The two grids</h2>\n")
    body.append(f"<p>{html.escape(fp.ANTHROPIC_GRID_ASSUMPTION_SENTENCE)}</p>\n")
    body.append(f"<p>{html.escape(fp.MARKET_VS_LOCATION_SENTENCE)}</p>\n")

    # 6. Everyday-equivalent anchors, each with its source inline.
    body.append("<h2>What that is like</h2>\n")
    body.append(
        "<ul>\n"
        "<li>Seconds of video streaming — about 20–30 seconds per answer "
        "(IEA / Kamiya 2020 fact-check, ~36 gCO2e per streaming hour, "
        "viewing device included).</li>\n"
        "<li>Metres driven by a typical passenger car — roughly one metre "
        "per answer (US EPA Greenhouse Gas Equivalencies, ~0.25 g/metre).</li>\n"
        "<li>Mugs of tea — about 60 answers to one mug (first-principles "
        "kettle physics: heating 250 ml, ~0.031 kWh at a realistic kettle "
        "efficiency).</li>\n"
        "</ul>\n"
    )

    # 7. What is not counted (the §7 exclusions).
    body.append("<h2>What is not counted</h2>\n")
    body.append(
        "<ul>\n"
        f"<li>Model <strong>training</strong>: {html.escape(fp.TRAINING_EXCLUSION_PHRASE)} "
        "— amortised training energy is unknowable for a closed model.</li>\n"
        "<li>Our one-off embedding/index build-time compute (small; could be "
        "added as a constant later).</li>\n"
        "<li>Network transfer and the visitor's own device — tiny for a text "
        "chat, but listed here as excluded, not silently dropped.</li>\n"
        "<li>Anthropic's water use — <strong>unknown</strong>, listed as "
        "unknown rather than guessed.</li>\n"
        "<li>Serving-hardware embodied carbon.</li>\n"
        "</ul>\n"
    )

    # 8. Revision note — the dated factors version.
    body.append(
        "<h2>Revision note</h2>\n"
        f"<p>These factors are versioned in the repository: "
        f"<strong>{html.escape(fp.FOOTPRINT_FACTORS_VERSION)}</strong>. A "
        "factor change is a visible, dated commit — this page moves with "
        "the code.</p>\n"
    )

    body.append("</main>\n")
    body.append(_page_footer())
    return "".join(body)


def _num(value: float) -> str:
    """A factor value with trailing ``.0`` stripped (3.0 → "3")."""
    text = f"{value:g}"
    return text


def _factor_row(label: str, factor: Any, provenance: str) -> str:
    """One §3-table row on the /footprint method section, figures verbatim
    from the factor module (interpolated at call time)."""
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{_num(factor.central)}</td>"
        f"<td>{_num(factor.low)} – {_num(factor.high)}</td>"
        f"<td>{html.escape(provenance)}</td>"
        "</tr>\n"
    )


def _kwh(wh: float) -> str:
    """Watt-hours → a kWh figure for the headline totals (never scientific)."""
    kwh = wh / 1000
    if kwh == 0:
        return "0"
    return f"{kwh:.3g}"


def _kg(grams: float) -> str:
    """Grams CO2e → a kg figure for the headline totals."""
    kg = grams / 1000
    if kg == 0:
        return "0"
    return f"{kg:.3g}"
