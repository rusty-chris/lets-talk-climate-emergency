"""Issue #19 RED — /about, /privacy and /voices content, and the
invariants EVERY transparency page carries.

Pins ``service.transparency`` (contract stubs; behaviour raises
``NotImplementedError`` until the implementer lands):

- /about: ADR-022 name + tagline, the Appendix-B guaranteed-vs-measured
  text verbatim, the honest how-it-works framing (retrieval, citations,
  "unverified" badges), the published eval numbers from RESULTS.md (a
  missing results file fails the BUILD, never renders blanks), the
  exclusions-with-reasons line ("Ripple et al.: permission requested"),
  the NEB-campaign framing, and the corpus vintage.
- /privacy: LOGGING_DISCLOSURE verbatim; retention figures interpolated
  AT CALL TIME from the retention constants (monkeypatching the source
  module changes the page — hand-copied numbers cannot pass); hashed-IP
  explanation; lawful basis; the NAMED owner-contact constant.
- /voices: the honestly-flagged placeholder (PR #198's content awaits
  editorial sign-off and is NOT on main) — no invented campaign facts.
- every page: the ADR-018 credit/non-commercial pair ADJACENT on the
  rendered artefact, the §4.11 disclaimer verbatim, links to the other
  transparency routes, and no secrets.
"""

from __future__ import annotations

import pytest

import service.exchange_log
import service.rate_limit
from service.exchange_log import LOGGING_DISCLOSURE
from service.transparency import (
    CREDIT_PAIR_MAX_SEPARATION,
    GUARANTEED_VS_MEASURED_TEXT,
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    PRIVACY_CONTACT_EMAIL,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    STEWARD_CREDIT_TEXT,
    TRANSPARENCY_ROUTES,
    VOICES_PLACEHOLDER_NOTICE,
    TransparencyBuildError,
    build_transparency_pages,
    render_about_page,
    render_privacy_page,
    render_sources_page,
    render_voices_page,
)
from tests._transparency_fixtures import (
    FIXTURE_RESULTS_MD,
    FIXTURE_RESULTS_SCORES,
    FIXTURE_RESULTS_VERDICT,
    chars_between,
    contains_verbatim,
    load_real_corpus_manifest,
    load_real_datasets_manifest,
    page_text,
    write_fixture_manifests,
    write_fixture_results,
)

CORPUS_VINTAGE = "2026-08-01"


def about_page() -> str:
    return render_about_page(eval_results_text=FIXTURE_RESULTS_MD, corpus_vintage=CORPUS_VINTAGE)


def sources_page() -> str:
    return render_sources_page(
        corpus_manifest=load_real_corpus_manifest(),
        datasets_manifest=load_real_datasets_manifest(),
        corpus_vintage=CORPUS_VINTAGE,
    )


#: name -> zero-arg renderer, for the every-page invariants.
ALL_PAGES = {
    "about": about_page,
    "privacy": render_privacy_page,
    "sources": sources_page,
    "voices": render_voices_page,
}


class TestAboutPage:
    def test_about_names_the_product_and_tagline(self) -> None:
        text = page_text(about_page())
        assert PRODUCT_NAME in text
        assert contains_verbatim(about_page(), PRODUCT_TAGLINE)

    def test_about_includes_guaranteed_vs_measured_text(self) -> None:
        """DESIGN Appendix B, verbatim — the honest framing is the point."""
        assert contains_verbatim(about_page(), GUARANTEED_VS_MEASURED_TEXT)

    def test_about_explains_the_mechanics_honestly(self) -> None:
        """RAG + citations + validation badges: the page says what the
        system does (retrieves licensed text, cites it, badges sentences
        that fail citation-support validation as "unverified")."""
        text = page_text(about_page()).lower()
        assert "retriev" in text
        assert "cite" in text
        assert "unverified" in text

    def test_about_renders_latest_eval_results(self) -> None:
        """The published numbers surface — guaranteed vs MEASURED needs
        the measurements on the page."""
        text = page_text(about_page())
        assert FIXTURE_RESULTS_VERDICT in text
        for score in FIXTURE_RESULTS_SCORES:
            assert score in text, f"eval score {score} missing from /about"

    def test_about_lists_exclusions_with_reasons(self) -> None:
        """Credibility furniture: what is NOT in the corpus, and why."""
        text = page_text(about_page())
        assert "Ripple et al." in text
        assert "permission requested" in text
        assert chars_between(text, "Ripple et al.", "permission requested") < 200, (
            "the Ripple exclusion must carry its reason beside it"
        )

    def test_about_carries_the_neb_alignment_framing(self) -> None:
        """The mission framing names the campaign the project aligns with
        (DESIGN §1) — beside the §4.11 non-affiliation disclaimer, which
        the every-page invariant pins separately."""
        text = page_text(about_page())
        assert "National Emergency Briefing" in text
        assert "nebriefing.org" in text

    def test_about_shows_the_corpus_vintage(self) -> None:
        text = page_text(about_page())
        assert CORPUS_VINTAGE in text
        assert "sources as of" in text.lower()


class TestPrivacyPage:
    def test_privacy_page_states_required_elements(self) -> None:
        """§9 in one place: disclosure verbatim, both retention figures,
        lawful basis, what-is-logged."""
        text = page_text(render_privacy_page())
        assert LOGGING_DISCLOSURE in text
        assert f"{service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS} days" in text
        assert f"{service.rate_limit.IP_HASH_RETENTION_DAYS} days" in text
        lowered = text.lower()
        assert "legitimate interests" in lowered
        assert "question" in lowered and "answer" in lowered  # what we log

    def test_privacy_retention_figures_come_from_the_constants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The no-silent-divergence pin: change the retention constants
        and the page changes — the figures are interpolated at call time
        from service.exchange_log / service.rate_limit, never
        hand-copied prose."""
        monkeypatch.setattr(service.exchange_log, "EXCHANGE_LOG_RETENTION_DAYS", 123)
        monkeypatch.setattr(service.rate_limit, "IP_HASH_RETENTION_DAYS", 5)
        text = page_text(render_privacy_page())
        assert "123 days" in text
        assert "5 days" in text
        assert "90 days" not in text, "a hand-copied 90-day figure survived a constant change"
        assert "7 days" not in text, "a hand-copied 7-day figure survived a constant change"

    def test_privacy_explains_hashed_ips(self) -> None:
        text = page_text(render_privacy_page()).lower()
        assert "hash" in text
        assert "rotating salt" in text
        assert "never joined" in text

    def test_privacy_renders_the_named_contact_constant(self) -> None:
        """The UK-GDPR contact point renders from PRIVACY_CONTACT_EMAIL
        (the owner fills the constant; the page follows)."""
        assert PRIVACY_CONTACT_EMAIL in page_text(render_privacy_page())

    def test_privacy_contact_follows_the_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = page_text(render_privacy_page(contact_email="owner-filled@example.test"))
        assert "owner-filled@example.test" in text
        assert PRIVACY_CONTACT_EMAIL not in text

    def test_privacy_names_uk_gdpr_and_the_ico(self) -> None:
        text = page_text(render_privacy_page())
        assert "UK GDPR" in text
        assert "Information Commissioner" in text


class TestVoicesPlaceholder:
    """PR #198's voices content awaits the owner's editorial sign-off and
    is NOT on main (voices/ is scaffolding only) — the route serves an
    honestly-flagged placeholder, never invented content."""

    def test_voices_placeholder_is_clearly_flagged(self) -> None:
        rendered = render_voices_page(voices_content=None)
        assert contains_verbatim(rendered, VOICES_PLACEHOLDER_NOTICE)
        assert "awaiting editorial sign-off" in page_text(rendered)

    def test_voices_placeholder_explains_the_separation(self) -> None:
        """The §2.5 promise stands even before content lands: voices are
        about the movement, never evidence for scientific claims."""
        text = page_text(render_voices_page(voices_content=None)).lower()
        assert "about the movement" in text
        assert "scientific" in text

    def test_voices_placeholder_invents_no_campaign_facts(self) -> None:
        """No expert names, no signature counts, no snapshot facts — the
        first-party prose is the owner's to approve (ORCHESTRATION.md
        stop-and-ask), not ours to improvise."""
        text = page_text(render_voices_page(voices_content=None))
        for invented in ("Packham", "petition", "signatures", "as_of"):
            assert invented not in text, (
                f"placeholder /voices page invents campaign content: {invented!r}"
            )


class TestEveryPageInvariants:
    @pytest.mark.parametrize("name", sorted(ALL_PAGES))
    def test_steward_credit_paired_with_noncommercial_note(self, name: str) -> None:
        """ADR-018 on the artefact: the pair renders adjacently, on every
        page — never the credit alone."""
        text = page_text(ALL_PAGES[name]())
        assert STEWARD_CREDIT_TEXT in text
        assert NONCOMMERCIAL_NOTE in text
        assert (
            chars_between(text, STEWARD_CREDIT_TEXT, NONCOMMERCIAL_NOTE)
            <= CREDIT_PAIR_MAX_SEPARATION
        ), f"/{name}: the ADR-018 credit and non-commercial note are separated"

    @pytest.mark.parametrize("name", sorted(ALL_PAGES))
    def test_nonaffiliation_disclaimer_on_every_source_surface(self, name: str) -> None:
        assert contains_verbatim(ALL_PAGES[name](), NON_AFFILIATION_DISCLAIMER), (
            f"/{name} is missing the §4.11 disclaimer verbatim"
        )

    @pytest.mark.parametrize("name", sorted(ALL_PAGES))
    def test_every_page_links_the_transparency_routes(self, name: str) -> None:
        rendered = ALL_PAGES[name]()
        for route in TRANSPARENCY_ROUTES:
            if route == f"/{name}":
                continue
            assert route in rendered, f"/{name} does not link {route}"

    def test_noncommercial_statement_present(self) -> None:
        """The non-commercial commitment is stated in prose on /about,
        beyond the footer pair (ADR-018: the framing is load-bearing for
        the Tier-B licensing calculus)."""
        text = page_text(about_page()).lower()
        assert "non-commercial" in text
        assert "free" in text
        assert "open-source" in text or "open source" in text


class TestBuildSeam:
    def test_build_produces_all_four_pages(self, tmp_path) -> None:
        corpus_path, datasets_path = write_fixture_manifests(tmp_path)
        pages = build_transparency_pages(
            corpus_manifest_path=corpus_path,
            datasets_manifest_path=datasets_path,
            eval_results_path=write_fixture_results(tmp_path),
            corpus_vintage=CORPUS_VINTAGE,
        )
        route_map = pages.as_route_map()
        assert set(route_map) == set(TRANSPARENCY_ROUTES)
        for route, rendered in route_map.items():
            assert rendered.strip(), f"{route} built empty"

    def test_missing_results_file_fails_the_build(self, tmp_path) -> None:
        """The acceptance criterion: no RESULTS.md → the build FAILS,
        naming the path — never a page with blank numbers."""
        corpus_path, datasets_path = write_fixture_manifests(tmp_path)
        missing = tmp_path / "no-such-RESULTS.md"
        with pytest.raises(TransparencyBuildError) as excinfo:
            build_transparency_pages(
                corpus_manifest_path=corpus_path,
                datasets_manifest_path=datasets_path,
                eval_results_path=missing,
                corpus_vintage=CORPUS_VINTAGE,
            )
        assert str(missing) in str(excinfo.value)

    def test_built_pages_carry_no_secrets(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Static pages travel everywhere: no env secrets, no key-shaped
        strings, ever."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC-NOT-REAL-000")
        corpus_path, datasets_path = write_fixture_manifests(tmp_path)
        pages = build_transparency_pages(
            corpus_manifest_path=corpus_path,
            datasets_manifest_path=datasets_path,
            eval_results_path=write_fixture_results(tmp_path),
            corpus_vintage=CORPUS_VINTAGE,
        )
        for route, rendered in pages.as_route_map().items():
            assert "sk-ant-" not in rendered, f"{route} embeds a key-shaped string"
