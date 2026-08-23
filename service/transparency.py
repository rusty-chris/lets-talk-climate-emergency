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
  validation with "unverified" badges), the Appendix-B
  guaranteed-vs-measured text VERBATIM plus the latest published eval
  numbers (``evals/RESULTS.md`` — a missing results file fails the
  build via :class:`TransparencyBuildError`, never renders blanks), the
  corpus-tier story with exclusions and reasons ("Ripple et al.:
  permission requested"), the NEB-campaign alignment framing, the
  ADR-018 non-commercial statement + Rusty Data credit (inseparable
  pair), and the §4.11 disclaimer verbatim.
- **/privacy** — DESIGN §9 made visible: the
  :data:`service.exchange_log.LOGGING_DISCLOSURE` line verbatim; the
  retention figures INTERPOLATED AT CALL TIME from
  ``service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS`` and
  ``service.rate_limit.IP_HASH_RETENTION_DAYS`` (module-attribute
  reads, so a retention-constant change re-renders here and can never
  silently diverge — pinned by monkeypatching the source modules);
  the hashed-IP explanation (rotating salt, never joined to query
  logs); the lawful basis ("legitimate interests"); and the UK-GDPR
  contact point (:data:`PRIVACY_CONTACT_EMAIL` — a NAMED placeholder
  constant the owner fills before launch, flagged as an owner action).
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
  content (``voices/voices.yaml`` + first-party prose) is HELD for
  owner editorial sign-off (PR #198, an ORCHESTRATION.md stop-and-ask
  point) and is NOT on main; until it merges,
  ``render_voices_page(voices_content=None)`` serves an honestly
  flagged placeholder (:data:`VOICES_PLACEHOLDER_NOTICE`, "awaiting
  editorial sign-off") that invents NO campaign facts, expert names or
  ``as_of`` figures. Once #198 merges, non-None ``voices_content``
  consumes its render seam and every snapshot fact renders with its
  ``as_of`` date (that behaviour is pinned when the seam lands — a
  contract note here, deliberately not a test over unmerged content).

**Every page** carries: the ADR-018 steward-credit pair (credit text
never further than :data:`CREDIT_PAIR_MAX_SEPARATION` characters from
the non-commercial note — the pair is inseparable on the rendered
artefact, not just in the dataclasses), the §4.11 non-affiliation
disclaimer verbatim, and links to the other transparency routes. The
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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PRODUCT_NAME",
    "PRODUCT_TAGLINE",
    "STEWARD_CREDIT_TEXT",
    "NONCOMMERCIAL_NOTE",
    "NON_AFFILIATION_DISCLAIMER",
    "TRANSPARENCY_ROUTES",
    "GUARANTEED_VS_MEASURED_TEXT",
    "PRIVACY_CONTACT_EMAIL",
    "VOICES_PLACEHOLDER_NOTICE",
    "CREDIT_PAIR_MAX_SEPARATION",
    "TransparencyBuildError",
    "TransparencyPages",
    "render_about_page",
    "render_privacy_page",
    "render_sources_page",
    "render_voices_page",
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

#: DESIGN §4.11, verbatim — on every transparency page.
NON_AFFILIATION_DISCLAIMER = (
    "Not affiliated with or endorsed by the National Emergency Briefing "
    "campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or "
    "the IPCC. All sources cited and linked."
)

#: The four routes this module renders; parity-pinned against
#: ``ui.footer.TRANSPARENCY_ROUTES`` (the #18 footer links these).
TRANSPARENCY_ROUTES: tuple[str, ...] = ("/about", "/privacy", "/sources", "/voices")

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

#: The UK-GDPR contact point on /privacy. A NAMED placeholder: the owner
#: substitutes the real published contact address before launch (an
#: ORCHESTRATION.md owner-action item — flagged in the #19 red-phase
#: report and the deployment runbook). The page renders THIS constant,
#: so the fill is a one-line change here and nowhere else.
PRIVACY_CONTACT_EMAIL = "privacy-contact-PENDING-owner-decision@example.invalid"

#: The honest /voices state while the voices content (PR #198) awaits
#: the owner's editorial sign-off. No invented campaign facts.
VOICES_PLACEHOLDER_NOTICE = (
    "The Voices & action content — the people and campaigns publicly "
    "communicating the climate emergency — is awaiting editorial sign-off "
    "and will appear here once approved."
)

#: ADR-018 on the rendered artefact: the credit text and the
#: non-commercial note must sit within this many characters of each
#: other in every page's HTML (the pair reads as one statement).
CREDIT_PAIR_MAX_SEPARATION = 200


class TransparencyBuildError(Exception):
    """Building the transparency pages failed loudly (missing eval
    results file, unreadable manifest) — never a page with blanks."""


@dataclass(frozen=True)
class TransparencyPages:
    """The four rendered pages, built once at startup and served as-is."""

    about_html: str
    privacy_html: str
    sources_html: str
    voices_html: str

    def as_route_map(self) -> dict[str, str]:
        """``{route: html}`` for the four :data:`TRANSPARENCY_ROUTES`."""
        return {
            "/about": self.about_html,
            "/privacy": self.privacy_html,
            "/sources": self.sources_html,
            "/voices": self.voices_html,
        }


def render_about_page(*, eval_results_text: str, corpus_vintage: str) -> str:
    """Pure: the /about HTML (see module doc for the pinned content).

    ``eval_results_text`` is the published ``evals/RESULTS.md`` content
    (verdict + gate numbers surface on the page); ``corpus_vintage``
    renders in the "answers reflect sources as of" line.
    """
    raise NotImplementedError("issue #19: implement the /about renderer")


def render_privacy_page(*, contact_email: str = PRIVACY_CONTACT_EMAIL) -> str:
    """Pure: the /privacy HTML.

    Retention figures are read from
    ``service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS`` and
    ``service.rate_limit.IP_HASH_RETENTION_DAYS`` as module attributes
    AT CALL TIME (the no-silent-divergence pin), never hand-copied.
    """
    raise NotImplementedError("issue #19: implement the /privacy renderer")


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
    raise NotImplementedError("issue #19: implement the /sources renderer")


def render_voices_page(*, voices_content: Any | None = None) -> str:
    """Pure: the /voices HTML.

    ``None`` (the state until PR #198's editorial sign-off) renders the
    flagged placeholder around :data:`VOICES_PLACEHOLDER_NOTICE`; a
    non-None value consumes the #198 voices render seam once merged.
    """
    raise NotImplementedError("issue #19: implement the /voices renderer")


def build_transparency_pages(
    *,
    corpus_manifest_path: Path,
    datasets_manifest_path: Path,
    eval_results_path: Path,
    corpus_vintage: str,
    contact_email: str = PRIVACY_CONTACT_EMAIL,
    voices_content: Any | None = None,
) -> TransparencyPages:
    """Load the sources of truth and render all four pages (startup step).

    Raises :class:`TransparencyBuildError` NAMING THE PATH when
    ``eval_results_path`` does not exist (a release without published
    eval results must fail the build, not render blanks — issue #19
    acceptance criterion) or when a manifest cannot be loaded.
    """
    raise NotImplementedError("issue #19: implement the transparency page build")
