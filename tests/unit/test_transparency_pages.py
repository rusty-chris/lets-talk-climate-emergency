"""Issue #19 RED — /about, /privacy and /voices content, and the
invariants EVERY transparency page carries.

Pins ``service.transparency`` (contract stubs; behaviour raises
``NotImplementedError`` until the implementer lands):

- /about: ADR-022 name + tagline, the Appendix-B guaranteed-vs-measured
  text verbatim, the honest how-it-works framing (retrieval, citations,
  "unverified" badges — with NO overclaim of the validator's degraded
  path, review finding #251), the published eval numbers from RESULTS.md
  (a missing results file fails the BUILD, never renders blanks), the
  exclusions-with-reasons line (Ripple et al. wording keyed to the
  RECORDED letters-sent state, review finding #254), the NEB-campaign
  framing, and the corpus vintage.
- /privacy: LOGGING_DISCLOSURE verbatim; retention figures interpolated
  AT CALL TIME from the retention constants (monkeypatching the source
  module changes the page — hand-copied numbers cannot pass); the DESIGN
  §9 eval-harvest disclosure (review finding #252); hashed-IP
  explanation; lawful basis; the NAMED owner-contact constant.
- /voices: the honestly-flagged placeholder (PR #198's content awaits
  editorial sign-off and is NOT on main) — no invented campaign facts,
  and a non-None voices_content is REFUSED loudly, never silently
  swallowed (review finding #255).
- every page: the ADR-018 credit/non-commercial pair ADJACENT on the
  rendered artefact, the §4.11 disclaimer verbatim, links to the other
  transparency routes, and no secrets.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import service.exchange_log
import service.rate_limit
from service.exchange_log import (
    LOGGING_DISCLOSURE,
    build_exchange_record,
    detach_for_harvest,
)
from service.transparency import (
    CREDIT_PAIR_MAX_SEPARATION,
    GUARANTEED_VS_MEASURED_TEXT,
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    PERMISSION_LETTERS_RECORD_PATH,
    PRIVACY_CONTACT_EMAIL,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    STEWARD_CREDIT_TEXT,
    TRANSPARENCY_ROUTES,
    VOICES_PLACEHOLDER_NOTICE,
    TransparencyBuildError,
    build_transparency_pages,
    read_permission_letters_record,
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
        """Credibility furniture: what is NOT in the corpus, and why.

        UPDATED for review finding #254 (the invariant — Ripple line +
        reason beside it — is preserved; only the reason wording moves):
        the letters are unsent drafts, so the DEFAULT rendering must
        read "permission to be requested", not the past-tense claim.
        ``TestRippleLettersGate`` pins the recorded-state gate."""
        text = page_text(about_page())
        assert "Ripple et al." in text
        assert "permission to be requested" in text
        assert chars_between(text, "Ripple et al.", "permission to be requested") < 200, (
            "the Ripple exclusion must carry its reason beside it"
        )

    def test_about_makes_no_unsent_permission_claim_by_default(self) -> None:
        """#254: while the recorded letters-sent state is False (the
        default, and the state on main today — letters/02-oup-bioscience.md
        is an unsent draft), the page must never assert "permission
        requested": the claim would be false to the exact counterparties
        the project is about to write to."""
        assert "permission requested" not in page_text(about_page())

    def test_about_does_not_overclaim_validation_coverage(self) -> None:
        """#251: "Every answer sentence is checked … is badged
        'unverified'" is a GUARANTEE the merged #13 validator does not
        make — on its designed degraded path (validate_exchange returns
        validated=False with degraded_reason) zero badge events are
        emitted. The universal claim must be retired."""
        text = page_text(about_page()).lower()
        assert "every answer sentence is checked" not in text
        assert "every answer sentence" not in text, (
            "/about still carries a universal every-answer-sentence claim the "
            "validator's degraded path makes false"
        )

    def test_about_discloses_the_degraded_validation_path(self) -> None:
        """#251, the honest replacement contract: factual sentences are
        checked WHEN the validation pass runs; when it cannot run the
        answer is delivered with no badges and flagged as unvalidated —
        never presented as checked. (Appendix B's guaranteed-vs-measured
        text is untouched; a separate test pins it verbatim.)"""
        text = page_text(about_page()).lower()
        assert "factual sentences are checked" in text
        assert "cannot run" in text
        assert "no badges" in text
        assert "flagged as unvalidated" in text

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
        lawful basis, what-is-logged.

        UPDATED for review finding #252 (all prior pins preserved): the
        retention paragraph may no longer present the deletion bound as
        unconditional — the eval-harvest qualifier must sit in the same
        paragraph as the interpolated retention figure."""
        text = page_text(render_privacy_page())
        assert LOGGING_DISCLOSURE in text
        assert f"{service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS} days" in text
        assert f"{service.rate_limit.IP_HASH_RETENTION_DAYS} days" in text
        lowered = text.lower()
        assert "legitimate interests" in lowered
        assert "question" in lowered and "answer" in lowered  # what we log
        # #252: the harvest qualifier is beside the figure it qualifies,
        # not buried elsewhere — "kept N days then deleted" must not read
        # as unconditional on the page.
        assert (
            chars_between(
                lowered,
                f"{service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS} days",
                "hand-reviewed",
            )
            <= 600
        ), "the eval-harvest qualifier is not in the retention paragraph"

    def test_privacy_discloses_the_eval_harvest_flow(self) -> None:
        """#252: the DESIGN §9 harvest flow is privacy-notice material —
        exchanges promoted into the published eval sets are hand-reviewed,
        personal-detail exchanges are excluded entirely, and promoted
        content is irreversibly detached from timestamps and identifiers,
        after which the anonymised excerpts may be retained beyond the
        log-retention bound. The disclosure must match what
        ``exchange_log.detach_for_harvest`` ACTUALLY does."""
        # What the code actually strips: the detached record keeps only
        # content fields — no timestamp, no exchange_id (the join key).
        detached = detach_for_harvest(
            build_exchange_record(
                question="synthetic question",
                route="retrieval",
                answer_text="synthetic answer",
                retrieved_chunk_ids=[],
                citations=[],
                validation={},
                usage_records=[],
                exclude_from_harvest=False,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        assert "timestamp" not in detached
        assert "exchange_id" not in detached

        text = page_text(render_privacy_page())
        lowered = text.lower()
        assert "hand-reviewed" in lowered
        assert "irreversibly detached" in lowered
        # …detached from exactly what the code detaches from:
        assert "timestamps" in lowered
        assert "identifiers" in lowered
        # …and honest about the consequence: retention past the bound.
        assert "retained beyond" in lowered
        # Personal-detail exclusion, stated within the harvest paragraph.
        assert chars_between(lowered, "hand-reviewed", "personal") <= 400, (
            "the personal-details exclusion is not part of the harvest disclosure"
        )

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

    def test_voices_page_refuses_content_until_the_seam_lands(self) -> None:
        """#255: the signature advertises a #198 render seam that does
        not exist — a caller passing real approved content today gets a
        green build and the placeholder served OVER it, silently. Until
        the seam merges, a non-None voices_content must raise
        NotImplementedError naming PR #198; None still renders the
        placeholder (the ratified state — unchanged)."""
        with pytest.raises(NotImplementedError) as excinfo:
            render_voices_page(voices_content={"campaigns": []})
        assert "198" in str(excinfo.value)
        rendered = render_voices_page(voices_content=None)
        assert contains_verbatim(rendered, VOICES_PLACEHOLDER_NOTICE)

    def test_build_seam_propagates_the_voices_refusal(self, tmp_path) -> None:
        """#255, the trap scenario: the #198 wiring goes through
        build_transparency_pages — the refusal must surface there too,
        never a successful build serving the placeholder over content."""
        corpus_path, datasets_path = write_fixture_manifests(tmp_path)
        with pytest.raises(NotImplementedError):
            build_transparency_pages(
                corpus_manifest_path=corpus_path,
                datasets_manifest_path=datasets_path,
                eval_results_path=write_fixture_results(tmp_path),
                corpus_vintage=CORPUS_VINTAGE,
                voices_content={"campaigns": []},
            )


class TestRippleLettersGate:
    """Review finding #254: /about said "Ripple et al.: permission
    requested" while letters/02-oup-bioscience.md sat as an unsent
    draft — nothing tied the public claim to the letters going out.

    FLAGGED DECISION (state-recording mechanism, for orchestrator
    ratification): a checked-in sending record,
    ``letters/SENDING-RECORD.md``, carrying a header line
    ``permission_letters_sent: pending`` that the OWNER flips to
    ``permission_letters_sent: sent <YYYY-MM-DD>`` after performing the
    ORCHESTRATION.md stop-and-ask act of sending — mirroring the
    severity-audit-packet owner gate (finding #197). The /about wording
    is keyed to this recorded state via
    ``read_permission_letters_record`` and the ``letters_record_path``
    build seam; a missing or malformed record is refused loudly, never
    treated as sent."""

    def _record(self, tmp_path, header_line: str, name: str = "SENDING-RECORD.md") -> Path:
        path = tmp_path / name
        path.write_text(
            f"{header_line}\n\n# Permission-letters sending record (synthetic fixture)\n",
            encoding="utf-8",
        )
        return path

    def test_pending_record_reads_false(self, tmp_path) -> None:
        record = self._record(tmp_path, "permission_letters_sent: pending")
        assert read_permission_letters_record(record) is False

    def test_sent_record_reads_true(self, tmp_path) -> None:
        record = self._record(tmp_path, "permission_letters_sent: sent 2026-09-01")
        assert read_permission_letters_record(record) is True

    def test_missing_record_is_never_treated_as_sent(self, tmp_path) -> None:
        """The #197 discipline: an absent record refuses loudly, naming
        the path — it never quietly reads as either state."""
        missing = tmp_path / "no-such-SENDING-RECORD.md"
        with pytest.raises(TransparencyBuildError) as excinfo:
            read_permission_letters_record(missing)
        assert str(missing) in str(excinfo.value)

    def test_malformed_record_refuses_loudly(self, tmp_path) -> None:
        record = self._record(tmp_path, "permission_letters_sent: banana")
        with pytest.raises(TransparencyBuildError):
            read_permission_letters_record(record)
        headerless = self._record(tmp_path, "# no header line here", name="headerless.md")
        with pytest.raises(TransparencyBuildError):
            read_permission_letters_record(headerless)

    def test_checked_in_sending_record_exists_and_parses(self) -> None:
        """The repo carries the record (the implementer creates it,
        recording today's true state: pending). This pin is
        state-agnostic so the owner's later flip to sent keeps it
        green; malformed or missing would raise."""
        assert PERMISSION_LETTERS_RECORD_PATH.is_file(), (
            "letters/SENDING-RECORD.md is not checked in — the /about wording "
            "has no recorded state to key on"
        )
        assert isinstance(read_permission_letters_record(), bool)

    def test_about_wording_follows_the_recorded_sent_state(self) -> None:
        """Once the record says sent, the DESIGN §7.3 wording is honest
        and renders; the unsent wording is retired."""
        text = page_text(
            render_about_page(
                eval_results_text=FIXTURE_RESULTS_MD,
                corpus_vintage=CORPUS_VINTAGE,
                permission_letters_sent=True,
            )
        )
        assert "Ripple et al." in text
        assert "permission requested" in text
        assert "permission to be requested" not in text

    def test_build_threads_the_letters_record_into_about(self, tmp_path) -> None:
        """build_transparency_pages reads the record (letters_record_path
        seam) and threads the state into render_about_page — the page
        follows the record, never an assumption."""
        corpus_path, datasets_path = write_fixture_manifests(tmp_path)

        def build(record: Path):
            return build_transparency_pages(
                corpus_manifest_path=corpus_path,
                datasets_manifest_path=datasets_path,
                eval_results_path=write_fixture_results(tmp_path),
                corpus_vintage=CORPUS_VINTAGE,
                letters_record_path=record,
            )

        pending = build(self._record(tmp_path, "permission_letters_sent: pending"))
        assert "permission to be requested" in page_text(pending.about_html)
        sent = build(
            self._record(tmp_path, "permission_letters_sent: sent 2026-09-01", name="sent.md")
        )
        assert "permission requested" in page_text(sent.about_html)


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
