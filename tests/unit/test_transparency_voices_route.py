"""Voices-route wiring RED — the published /voices page (unit tier).

PR #198 (the voices layer: ``voices/voices.yaml`` + ``voices/render.py``)
has MERGED with the owner's editorial sign-off, so the /voices
transparency page stops serving the "awaiting editorial sign-off"
placeholder and RENDERS the merged content through the voices seam.
This suite pins that wiring, failing until the implementer lands it:

1. ``render_voices_page(voices_content=<VoicesLibrary>)`` renders the
   PUBLISHED page: every entity name, one-liner and prose paragraph;
   entity links and person links as real ``href`` anchors; every
   snapshot fact WITH its ``as_of`` date — the §2.5 as_of-rendering
   contract deliberately DEFERRED in the #19 red phase until the #198
   seam merged (a figure never renders undated); and the "About the
   movement" attribution framing.
2. The issue #260 prototype note (``VOICES_PROTOTYPE_NOTE``) renders
   verbatim with a link to the tracking issue — on /voices ONLY, never
   on /about, /privacy or /sources.
3. The placeholder path retires from the build:
   ``build_transparency_pages`` reads the checked-in
   ``VOICES_CONTENT_PATH`` by default (the ``service.main`` startup
   wiring passes nothing); a missing or unparseable voices.yaml fails
   the build loudly with ``TransparencyBuildError`` naming the path
   (the #249 pattern), never a green build silently serving the
   placeholder.
4. The every-page invariants hold on the published page (ADR-018 pair,
   §4.11 verbatim, transparency route links, no secrets), and every
   YAML-sourced string is HTML-escaped (pinned with a synthetic
   injection entity).
5. The Alliance of World Scientists / Ripple entry's link-only
   discipline survives rendering: its section links out but never
   quotes a scientific figure.

The review-finding #255 invariant is PRESERVED across the supersession
of its raises-NotImplementedError pin (updated, documented, in
tests/unit/test_transparency_pages.py): provided content is never
silently swallowed — it renders, or the call fails loudly.

FLAGGED DECISIONS (voices-route-wiring red phase, for orchestrator
ratification):

- ``VOICES_PROTOTYPE_NOTE`` wording (this suite pins the exact string
  verbatim on the page, plus its substance — "first-party" and
  "editorial review" — at the constant): "Prototype note: the movement
  descriptions on this page are first-party prose written by this
  project and are still under editorial review — individual details may
  be corrected as they are verified." It links
  ``VOICES_PROTOTYPE_NOTE_ISSUE_URL`` (issue #260).
- ``build_transparency_pages`` takes ``voices_path`` (a Path, REPLACING
  the pre-#198 ``voices_content`` parameter); ``None`` reads the
  checked-in ``VOICES_CONTENT_PATH`` — mirroring the
  ``letters_record_path``/``PERMISSION_LETTERS_RECORD_PATH`` seam — so
  the existing ``service.main`` startup call needs no new argument.
- ``render_voices_page`` accepts a ``voices.render.VoicesLibrary``; any
  other non-None value raises ``TypeError`` (loud, per #255 — never the
  placeholder over it).
"""

from __future__ import annotations

import datetime
import html
import re

import pytest
import yaml

from service.transparency import (
    CREDIT_PAIR_MAX_SEPARATION,
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    STEWARD_CREDIT_TEXT,
    VOICES_CONTENT_PATH,
    VOICES_PLACEHOLDER_NOTICE,
    VOICES_PROTOTYPE_NOTE,
    VOICES_PROTOTYPE_NOTE_ISSUE_URL,
    TransparencyBuildError,
    build_transparency_pages,
    render_about_page,
    render_privacy_page,
    render_sources_page,
    render_voices_page,
)
from tests._transparency_fixtures import (
    FIXTURE_RESULTS_MD,
    chars_between,
    contains_verbatim,
    load_real_corpus_manifest,
    load_real_datasets_manifest,
    page_text,
    write_fixture_manifests,
    write_fixture_results,
)
from voices.render import SnapshotFact, VoicesEntity, VoicesLibrary, load_voices

CORPUS_VINTAGE = "2026-08-01"


# --------------------------------------------------------------------------- #
# Helpers + fixtures
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    """Comparable prose: emphasis asterisks dropped, whitespace collapsed.

    The YAML prose carries ``*emphasis*`` asterisks; a renderer may keep
    them literally (escaped) or map the pair to a tag — both normalise
    identically here, so the pin is on the WORDS, not the styling."""
    return " ".join(text.replace("*", " ").split())


def visible_text(rendered_html: str) -> str:
    """Rendered HTML -> comparable visible text: tags stripped, entities
    unescaped, then :func:`_norm`."""
    return _norm(html.unescape(re.sub(r"<[^>]+>", " ", rendered_html)))


def min_gap(text: str, first: str, second: str) -> int:
    """Smallest character gap between occurrences of the two needles."""
    starts_first = [m.start() for m in re.finditer(re.escape(first), text)]
    starts_second = [m.start() for m in re.finditer(re.escape(second), text)]
    assert starts_first, f"missing {first!r}"
    assert starts_second, f"missing {second!r}"
    return min(abs(a - b) for a in starts_first for b in starts_second)


def fact_value_variants(fact: SnapshotFact) -> tuple[str, ...]:
    """Honest renderings of the figure: comma-grouped or plain."""
    return (f"{fact.value:,.0f}", f"{fact.value:.0f}")


def fact_date_variants(fact: SnapshotFact) -> tuple[str, ...]:
    """Honest renderings of the as_of date: human-readable or ISO."""
    return (fact.as_of.strftime("%-d %B %Y"), fact.as_of.isoformat())


def assert_fact_renders_dated(text: str, fact: SnapshotFact, where: str) -> None:
    """The deferred §2.5 contract: the figure is on the page AND its
    as_of date sits beside it — a fact never renders without its date."""
    values = [v for v in fact_value_variants(fact) if v in text]
    assert values, f"{where}: snapshot fact {fact.key!r} value not rendered"
    dates = [d for d in fact_date_variants(fact) if d in text]
    assert dates, f"{where}: snapshot fact {fact.key!r} rendered without its as_of date"
    gap = min(min_gap(text, value, date) for value in values for date in dates)
    assert gap <= 400, (
        f"{where}: snapshot fact {fact.key!r} renders its as_of date {gap} chars "
        "from the figure — the date must sit beside the number it dates (§2.5)"
    )


@pytest.fixture(scope="module")
def library() -> VoicesLibrary:
    """The REAL merged voices content, loaded through the #198 seam."""
    return load_voices(VOICES_CONTENT_PATH)


@pytest.fixture(scope="module")
def published_page(library: VoicesLibrary) -> str:
    """The published /voices HTML rendered from the real library."""
    return render_voices_page(voices_content=library)


def synthetic_entity(**overrides) -> VoicesEntity:
    """A schema-shaped invented entity (Aurelian Basin universe)."""
    base = dict(
        id="syn-aurelian-basin-collective",
        name="Aurelian Basin Voices Collective (synthetic)",
        category="campaign",
        canonical_url="https://example.invalid/aurelian-voices",
        one_liner="An invented movement entity for the voices-route red suite.",
        prose=(
            "An invented first paragraph describing an invented movement in "
            "the Aurelian Basin universe.\n\n"
            "An invented second paragraph, present to pin that every prose "
            "paragraph reaches the page."
        ),
        links=({"label": "Aurelian Basin home", "url": "https://example.invalid/aurelian-voices"},),
    )
    base.update(overrides)
    return VoicesEntity(**base)


def synthetic_library(*entities: VoicesEntity) -> VoicesLibrary:
    return VoicesLibrary(
        version=1,
        source_type="voices",
        attribution_text=(
            "About the movement — synthetic fixture attribution (Aurelian "
            "Basin universe). Not scientific evidence."
        ),
        entities=entities or (synthetic_entity(),),
    )


def build_pages(tmp_path, **overrides):
    """build_transparency_pages over the fixture manifests + results."""
    corpus_path, datasets_path = write_fixture_manifests(tmp_path)
    kwargs = dict(
        corpus_manifest_path=corpus_path,
        datasets_manifest_path=datasets_path,
        eval_results_path=write_fixture_results(tmp_path),
        corpus_vintage=CORPUS_VINTAGE,
    )
    kwargs.update(overrides)
    return build_transparency_pages(**kwargs)


# --------------------------------------------------------------------------- #
# 1. The published page renders the merged content through the seam
# --------------------------------------------------------------------------- #


class TestPublishedVoicesRendersTheMergedContent:
    def test_every_entity_name_renders(self, library, published_page) -> None:
        text = visible_text(published_page)
        for entity in library.entities:
            assert _norm(entity.name) in text, f"entity {entity.id!r} missing from /voices"

    def test_every_one_liner_renders(self, library, published_page) -> None:
        text = visible_text(published_page)
        for entity in library.entities:
            assert _norm(entity.one_liner) in text, f"{entity.id}: one-liner not rendered"

    def test_every_prose_paragraph_renders(self, library, published_page) -> None:
        """The owner signed off this exact prose; the page carries it —
        every paragraph, no silent truncation."""
        text = visible_text(published_page)
        for entity in library.entities:
            for paragraph in entity.prose.strip().split("\n\n"):
                needle = _norm(paragraph)
                if not needle:
                    continue
                assert needle in text, (
                    f"{entity.id}: prose paragraph not rendered: {needle[:60]!r}…"
                )

    def test_entity_links_render_as_real_hrefs(self, library, published_page) -> None:
        """/voices is the action surface — 'what can I watch or join?'
        must be clickable, not bare URL prose."""
        text = visible_text(published_page)
        for entity in library.entities:
            for link in entity.links:
                assert f'href="{html.escape(link["url"])}"' in published_page, (
                    f"{entity.id}: link {link['url']!r} is not a real href"
                )
                assert _norm(link["label"]) in text, (
                    f"{entity.id}: link label {link['label']!r} not rendered"
                )

    def test_people_render_with_real_hrefs(self, library, published_page) -> None:
        """Every named person appears with their one-liner, and their
        verification link is a real anchor."""
        text = visible_text(published_page)
        for entity in library.entities:
            for person in entity.people:
                assert _norm(person.name) in text, f"{entity.id}: {person.name!r} missing"
                assert _norm(person.one_liner) in text, (
                    f"{entity.id}: {person.name!r} one-liner not rendered"
                )
                assert f'href="{html.escape(person.link)}"' in published_page, (
                    f"{entity.id}: {person.name!r} link is not a real href"
                )

    def test_every_snapshot_fact_renders_with_its_as_of_date(self, library, published_page) -> None:
        """The §2.5 as_of-rendering contract, DEFERRED in the #19 red
        phase until the #198 seam merged — now due: every snapshot fact
        on the page states how current its figure is."""
        text = visible_text(published_page)
        facts = [(e.id, f) for e in library.entities for f in e.snapshot_facts]
        assert facts, "the merged voices content must carry snapshot facts"
        for entity_id, fact in facts:
            assert_fact_renders_dated(text, fact, where=f"/voices {entity_id}")

    def test_synthetic_fact_never_renders_without_its_date(self) -> None:
        """Same contract over a synthetic library with a distinctive
        figure — the pin cannot be satisfied by coincidence in the real
        prose."""
        fact = SnapshotFact(
            key="syn_basin_pledges",
            label="pledges to the Aurelian Basin charter (synthetic)",
            value=87654321,
            as_of=datetime.date(2099, 5, 4),
            source_url="https://example.invalid/basin-pledges",
        )
        entity = synthetic_entity(snapshot_facts=(fact,))
        text = visible_text(render_voices_page(voices_content=synthetic_library(entity)))
        assert_fact_renders_dated(text, fact, where="synthetic /voices")

    def test_about_the_movement_attribution_framing_present(self, published_page) -> None:
        """The §2.5 framing survives the placeholder's retirement: the
        page still says what this content IS (about the movement) and is
        NOT (scientific evidence)."""
        lowered = visible_text(published_page).lower()
        assert "about the movement" in lowered
        assert "scientific" in lowered


# --------------------------------------------------------------------------- #
# 2. The issue #260 prototype note — on /voices, and only /voices
# --------------------------------------------------------------------------- #


class TestPrototypeNote:
    def test_note_constant_carries_the_260_substance(self) -> None:
        """Guards the FLAGGED wording's substance through any rewording:
        #260 requires the note to say the descriptions are first-party
        prose under editorial review."""
        assert "first-party" in VOICES_PROTOTYPE_NOTE
        assert "editorial review" in VOICES_PROTOTYPE_NOTE
        assert VOICES_PROTOTYPE_NOTE_ISSUE_URL.endswith("/issues/260")

    def test_note_renders_verbatim_and_links_issue_260(self, published_page) -> None:
        assert contains_verbatim(published_page, VOICES_PROTOTYPE_NOTE), (
            "/voices does not carry the #260 prototype note verbatim"
        )
        assert f'href="{VOICES_PROTOTYPE_NOTE_ISSUE_URL}"' in published_page, (
            "/voices prototype note does not link issue #260"
        )

    def test_note_absent_from_the_other_three_pages(self) -> None:
        """The note qualifies the voices prose specifically; on /about,
        /privacy or /sources it would falsely qualify content that is
        NOT under editorial review."""
        other_pages = {
            "about": render_about_page(
                eval_results_text=FIXTURE_RESULTS_MD, corpus_vintage=CORPUS_VINTAGE
            ),
            "privacy": render_privacy_page(),
            "sources": render_sources_page(
                corpus_manifest=load_real_corpus_manifest(),
                datasets_manifest=load_real_datasets_manifest(),
                corpus_vintage=CORPUS_VINTAGE,
            ),
        }
        for name, rendered in other_pages.items():
            assert not contains_verbatim(rendered, VOICES_PROTOTYPE_NOTE), (
                f"/{name} carries the /voices-only prototype note"
            )
            assert VOICES_PROTOTYPE_NOTE_ISSUE_URL not in rendered, (
                f"/{name} links the /voices-only prototype-note issue"
            )

    def test_placeholder_retired_from_the_published_page(self, published_page) -> None:
        """The signed-off content and the 'awaiting sign-off' notice are
        mutually exclusive statements — the published page must not say
        both."""
        assert not contains_verbatim(published_page, VOICES_PLACEHOLDER_NOTICE)
        assert "awaiting editorial sign-off" not in visible_text(published_page)


# --------------------------------------------------------------------------- #
# 3. The build feeds the checked-in voices.yaml; absence fails loudly
# --------------------------------------------------------------------------- #


class TestBuildFeedsTheCheckedInVoicesContent:
    def test_voices_content_path_is_the_checked_in_yaml(self) -> None:
        """The build's default source of truth is the repo's signed-off
        file, and it loads through the #198 seam."""
        assert VOICES_CONTENT_PATH.name == "voices.yaml"
        assert VOICES_CONTENT_PATH.is_file(), (
            "voices/voices.yaml is not checked in — the build has no content to feed"
        )
        assert load_voices(VOICES_CONTENT_PATH).entities

    def test_build_default_serves_the_published_voices_page(self, tmp_path) -> None:
        """The startup wiring pin: ``service.main`` passes no voices
        argument, so the DEFAULT build must already feed the checked-in
        content — real entities and the #260 note on the page, the
        placeholder gone."""
        voices_html = build_pages(tmp_path).voices_html
        text = visible_text(voices_html)
        assert "The National Emergency Briefing" in text
        assert "Chris Packham" in text
        assert contains_verbatim(voices_html, VOICES_PROTOTYPE_NOTE)
        assert not contains_verbatim(voices_html, VOICES_PLACEHOLDER_NOTICE), (
            "the default build still serves the placeholder over the signed-off content"
        )

    def test_build_renders_the_content_at_an_explicit_voices_path(self, tmp_path) -> None:
        """``voices_path`` is honoured, not decorative: a synthetic
        voices.yaml renders ITS entities, not the checked-in ones."""
        synthetic = {
            "version": 1,
            "source_type": "voices",
            "attribution_text": (
                "About the movement — synthetic fixture attribution (Aurelian "
                "Basin universe). Not scientific evidence."
            ),
            "entities": [
                {
                    "id": "syn-aurelian-basin-collective",
                    "name": "Aurelian Basin Voices Collective (synthetic)",
                    "category": "campaign",
                    "canonical_url": "https://example.invalid/aurelian-voices",
                    "one_liner": "An invented movement entity for the voices-route red suite.",
                    "prose": "An invented paragraph describing an invented movement.",
                }
            ],
        }
        path = tmp_path / "synthetic-voices.yaml"
        path.write_text(yaml.safe_dump(synthetic, allow_unicode=True), encoding="utf-8")
        voices_html = build_pages(tmp_path, voices_path=path).voices_html
        text = visible_text(voices_html)
        assert "Aurelian Basin Voices Collective (synthetic)" in text
        assert "Chris Packham" not in text, "explicit voices_path was ignored"
        assert not contains_verbatim(voices_html, VOICES_PLACEHOLDER_NOTICE)

    def test_missing_voices_yaml_fails_the_build_loudly(self, tmp_path) -> None:
        """The #249 pattern: with the signed-off content merged, a build
        that cannot find it must FAIL naming the path — silently serving
        'awaiting editorial sign-off' would be a false public statement."""
        missing = tmp_path / "no-such-voices.yaml"
        with pytest.raises(TransparencyBuildError) as excinfo:
            build_pages(tmp_path, voices_path=missing)
        assert str(missing) in str(excinfo.value)

    @pytest.mark.parametrize(
        ("filename", "content"),
        [
            ("not-yaml.yaml", "entities: [unclosed\n  - {{{"),
            ("schema-invalid.yaml", "version: 1\nsource_type: voices\n"),
        ],
        ids=["yaml-syntax-error", "schema-violation"],
    )
    def test_unparseable_voices_yaml_fails_the_build_loudly(
        self, tmp_path, filename: str, content: str
    ) -> None:
        """Same discipline for a file that exists but cannot serve:
        TransparencyBuildError naming the path, never the placeholder."""
        broken = tmp_path / filename
        broken.write_text(content, encoding="utf-8")
        with pytest.raises(TransparencyBuildError) as excinfo:
            build_pages(tmp_path, voices_path=broken)
        assert str(broken) in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 4. Every-page invariants hold on the published page; YAML is escaped
# --------------------------------------------------------------------------- #


class TestPublishedPageInvariants:
    def test_adr018_pair_renders_adjacently(self, published_page) -> None:
        text = page_text(published_page)
        assert STEWARD_CREDIT_TEXT in text
        assert NONCOMMERCIAL_NOTE in text
        assert (
            chars_between(text, STEWARD_CREDIT_TEXT, NONCOMMERCIAL_NOTE)
            <= CREDIT_PAIR_MAX_SEPARATION
        ), "published /voices: the ADR-018 credit and non-commercial note are separated"

    def test_nonaffiliation_disclaimer_verbatim(self, published_page) -> None:
        assert contains_verbatim(published_page, NON_AFFILIATION_DISCLAIMER), (
            "published /voices is missing the §4.11 disclaimer verbatim"
        )

    def test_links_the_other_transparency_routes(self, published_page) -> None:
        for route in ("/about", "/privacy", "/sources"):
            assert route in published_page, f"published /voices does not link {route}"

    def test_built_voices_page_embeds_no_secrets(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The published page is static and travels everywhere — the
        no-secrets invariant holds on the CONTENT-bearing build too."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYNTHETIC-NOT-REAL-000")
        voices_html = build_pages(tmp_path).voices_html
        assert contains_verbatim(voices_html, VOICES_PROTOTYPE_NOTE)  # the published page
        assert "sk-ant-" not in voices_html, "published /voices embeds a key-shaped string"

    def test_yaml_strings_are_html_escaped(self) -> None:
        """A hostile-looking string anywhere in the YAML must reach the
        page inert (synthetic injection entity — the voices.yaml is
        first-party, but the renderer must not RELY on that)."""
        entity = synthetic_entity(
            name='Basin <script>alert("voices")</script> Collective (synthetic)',
            one_liner='An invented entity with <img src=x onerror="alert(1)"> markup.',
            links=(
                {
                    "label": '"><script>evil()</script> Basin link',
                    "url": "https://example.invalid/aurelian-voices",
                },
            ),
        )
        rendered = render_voices_page(voices_content=synthetic_library(entity))
        assert "<script" not in rendered, "a YAML string reached the page as live markup"
        assert 'onerror="alert(1)"' not in rendered
        assert "&lt;script&gt;" in rendered, "the hostile string was dropped, not escaped"
        # The inert text still reads back out as the authored string.
        assert 'Basin <script>alert("voices")</script> Collective' in visible_text(rendered)


# --------------------------------------------------------------------------- #
# 5. The AWS/Ripple link-only discipline survives rendering
# --------------------------------------------------------------------------- #

#: Markers of a scientific figure — the thing the link-only Ripple/AWS
#: entry must never quote (its prose promises "no scientific figure from
#: them is quoted here"; DESIGN §2.5 keeps voices structurally apart
#: from the assessed evidence).
FORBIDDEN_SCIENTIFIC_FIGURE_MARKERS = (
    "°C",
    "°F",
    " ppm",
    "GtCO2",
    "GtCO₂",
    "gigaton",
    "gigatonne",
    "W/m2",
    "W/m²",
)

#: A number wearing a unit of scientific measurement (percentages are
#: the Ripple warnings' signature "vital signs" framing).
PERCENT_FIGURE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|per\s?cent)", re.IGNORECASE)


class TestLinkOnlyDisciplineSurvivesRendering:
    def test_aws_entity_is_still_link_only_with_no_snapshot_facts(self, library) -> None:
        """The content-boundary half of the discipline, re-pinned at the
        transparency seam: the AWS/Ripple entry stays link_only and
        carries no snapshot figures at all."""
        aws = next(e for e in library.entities if e.id == "alliance-of-world-scientists")
        assert aws.link_only, "the Ripple/AWS entry must stay link_only (DESIGN §2.5)"
        assert aws.snapshot_facts == (), (
            "the link-only Ripple/AWS entry must carry no snapshot figures"
        )

    def test_aws_section_links_out_but_quotes_no_scientific_figure(self, library) -> None:
        """The RENDERED half: an AWS-only page (its whole visible body is
        the entry's section plus furniture) carries the link as a real
        href and not one scientific-figure pattern."""
        aws = next(e for e in library.entities if e.id == "alliance-of-world-scientists")
        aws_only = VoicesLibrary(
            version=library.version,
            source_type=library.source_type,
            attribution_text=library.attribution_text,
            entities=(aws,),
        )
        rendered = render_voices_page(voices_content=aws_only)
        assert f'href="{html.escape(aws.links[0]["url"])}"' in rendered, (
            "the AWS/Ripple section lost its read-at-the-source link"
        )
        text = visible_text(rendered)
        for marker in FORBIDDEN_SCIENTIFIC_FIGURE_MARKERS:
            assert marker not in text, (
                f"the link-only AWS/Ripple section quotes a scientific figure ({marker!r})"
            )
        assert not PERCENT_FIGURE.search(text), (
            "the link-only AWS/Ripple section quotes a percentage figure"
        )

    def test_full_published_page_quotes_no_scientific_figure_units(
        self, library, published_page
    ) -> None:
        """Voices-wide: movement counts (signatures, MPs, screenings) are
        fine; numbers wearing scientific units are not — on the whole
        rendered page, for the real yaml."""
        text = visible_text(published_page)
        for marker in FORBIDDEN_SCIENTIFIC_FIGURE_MARKERS:
            assert marker not in text, f"published /voices quotes a scientific figure ({marker!r})"
        assert not PERCENT_FIGURE.search(text), "published /voices quotes a percentage figure"


# --------------------------------------------------------------------------- #
# The composed service serves the published page (still unit tier: the
# app harness fakes every adapter; no network, no Docker)
# --------------------------------------------------------------------------- #


class TestComposedServiceServesThePublishedPage:
    def test_voices_route_serves_the_published_content(self, tmp_path) -> None:
        """End of the wire: a service booted with the built pages serves
        the signed-off content and the #260 note on GET /voices — not
        the placeholder."""
        from fastapi.testclient import TestClient

        from tests._service_fixtures import make_harness

        pages = build_pages(tmp_path)
        harness = make_harness(tmp_path, transparency=pages)
        response = TestClient(harness.app).get("/voices")
        assert response.status_code == 200
        assert contains_verbatim(response.text, VOICES_PROTOTYPE_NOTE)
        assert "The National Emergency Briefing" in visible_text(response.text)
        assert not contains_verbatim(response.text, VOICES_PLACEHOLDER_NOTICE)
