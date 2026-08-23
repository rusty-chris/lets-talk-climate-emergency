"""The voices layer content + ingestion (issue #8, DESIGN §2.5) — unit.

These tests pin the data contract of the voices layer and its ingestion
labelling, over BOTH the real ``voices/voices.yaml`` and synthetic
fixtures built in-test for the refusal paths:

1. ``test_voices_yaml_schema_valid`` — every entity has a name, canonical
   link and one-line description (and every named person a link).
2. ``test_snapshot_facts_carry_as_of_dates`` — a numeric snapshot fact
   without ``as_of`` refuses; the real file's facts all carry one and
   render WITH it.
3. ``test_voices_chunks_labelled_source_type_voices`` — ingesting the
   voices content through the #7 chunker marks every chunk
   ``source_type: voices``.
4. ``test_voices_attribution_is_about_the_movement`` — voices chunks
   carry the "About the movement" attribution, never a scientific-source
   attribution.

The prose itself is editorially reviewed (voices/EDITORIAL_CHECKLIST.md),
not asserted here — but the checklist's coverage of every entity IS
pinned, so it cannot drift out of step with the content.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import yaml

from ingestion.pipeline import ChunkRecord
from tests._ingestion_fixtures import config, manifest_entry
from voices.render import (
    ABOUT_THE_MOVEMENT,
    VoicesError,
    ingest_voices_entities,
    load_voices,
    render_entity_html,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_VOICES = REPO_ROOT / "voices" / "voices.yaml"


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def library():
    return load_voices(REAL_VOICES)


def _write_voices(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "voices.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_voices(**entity_overrides) -> dict:
    """A schema-valid one-entity voices doc; overrides mutate the entity."""
    entity = {
        "id": "syn-entity",
        "name": "Synthetic Campaign (invented)",
        "category": "campaign",
        "canonical_url": "https://example.invalid/campaign",
        "one_liner": "An invented campaign used only to exercise the schema.",
        "prose": (
            "This invented paragraph is long enough to survive the chunker's "
            "tiny-chunk floor so the ingestion tests have real body text to "
            "assert on, and it mentions nothing real at all."
        ),
    }
    entity.update(entity_overrides)
    return {
        "version": 1,
        "source_type": "voices",
        "attribution_text": (
            "About the movement — invented first-party attribution for the tests."
        ),
        "entities": [entity],
    }


# --------------------------------------------------------------------------- #
# 1. Schema validity
# --------------------------------------------------------------------------- #


def test_voices_yaml_schema_valid(library) -> None:
    """Every entity carries a name, a canonical link and a one-line
    description; every named person carries a name, one-liner and link."""
    assert library.source_type == "voices"
    assert ABOUT_THE_MOVEMENT.lower() in library.attribution_text.lower()
    assert library.entities, "the real voices layer must define entities"

    for entity in library.entities:
        assert entity.name.strip(), f"{entity.id}: empty name"
        assert entity.canonical_url.startswith("http"), f"{entity.id}: bad canonical_url"
        assert entity.one_liner.strip(), f"{entity.id}: empty one_liner"
        assert entity.prose.strip(), f"{entity.id}: empty prose"
        for person in entity.people:
            assert person.name.strip(), f"{entity.id}: person with no name"
            assert person.one_liner.strip(), f"{entity.id}: {person.name} has no one-liner"
            assert person.link.startswith("http"), f"{entity.id}: {person.name} has no link"


def test_named_experts_and_link_only_entities_present(library) -> None:
    """DESIGN §2.5 name-checks: the NEB campaign, the named experts, the
    link-only Ripple/AWS entry, warming stripes, Climate Majority Project,
    Covering Climate Now and David King/CCAG are all represented."""
    ids = {e.id for e in library.entities}
    for required in (
        "neb-campaign",
        "neb-experts",
        "chris-packham",
        "alliance-of-world-scientists",
        "warming-stripes",
        "climate-majority-project",
        "covering-climate-now",
        "ccag-david-king",
    ):
        assert required in ids, f"voices layer is missing the {required!r} entity (DESIGN §2.5)"

    experts = next(e for e in library.entities if e.id == "neb-experts")
    expert_names = " ".join(p.name for p in experts.people)
    for surname in ("Anderson", "Lenton", "Fowler", "Seddon", "Behrens", "Khan", "Mann"):
        assert surname in expert_names, f"named NEB expert {surname} missing"

    aws = next(e for e in library.entities if e.id == "alliance-of-world-scientists")
    assert aws.link_only, "the Ripple/AWS entry must be marked link_only (DESIGN §2.5)"


def test_missing_required_entity_field_refuses(tmp_path) -> None:
    for field in ("name", "canonical_url", "one_liner", "prose"):
        data = _minimal_voices()
        del data["entities"][0][field]
        with pytest.raises(VoicesError, match=field):
            load_voices(_write_voices(tmp_path, data))


def test_person_without_link_refuses(tmp_path) -> None:
    data = _minimal_voices(
        people=[{"name": "Dr Invented", "one_liner": "invented role"}]  # no link
    )
    with pytest.raises(VoicesError, match="link"):
        load_voices(_write_voices(tmp_path, data))


def test_attribution_without_about_the_movement_refuses(tmp_path) -> None:
    data = _minimal_voices()
    data["attribution_text"] = "Meridian Climate Assessment, Cycle 3"  # scientific-looking
    with pytest.raises(VoicesError, match="About the movement"):
        load_voices(_write_voices(tmp_path, data))


def test_editorial_checklist_covers_every_entity(library) -> None:
    """Process guard: the editorial sign-off checklist names every entity,
    so a new entity cannot be added without a review line for it."""
    checklist = (REPO_ROOT / "voices" / "EDITORIAL_CHECKLIST.md").read_text(encoding="utf-8")
    for entity in library.entities:
        assert entity.id in checklist, f"editorial checklist omits entity {entity.id!r}"


# --------------------------------------------------------------------------- #
# 2. Snapshot facts carry (and render with) as_of dates
# --------------------------------------------------------------------------- #


def test_snapshot_facts_carry_as_of_dates(library) -> None:
    """Every numeric snapshot fact in the real file carries a valid
    ``as_of`` date and a source url."""
    facts = [f for e in library.entities for f in e.snapshot_facts]
    assert facts, "the voices layer must carry snapshot facts (petition/MP/screening counts)"
    for fact in facts:
        assert isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool)
        assert isinstance(fact.as_of, datetime.date), f"{fact.key}: as_of not a date"
        assert fact.source_url.startswith("http"), f"{fact.key}: no source url"


def test_numeric_snapshot_fact_without_as_of_refuses(tmp_path) -> None:
    data = _minimal_voices(
        snapshot_facts=[
            {
                "key": "petition_signatures",
                "label": "signatures",
                "value": 100000,
                "source_url": "https://example.invalid/petition",
                # as_of deliberately omitted
            }
        ]
    )
    with pytest.raises(VoicesError, match="as_of"):
        load_voices(_write_voices(tmp_path, data))


def test_snapshot_facts_render_with_as_of(library) -> None:
    """A rendered entity states each snapshot fact WITH its as_of date and
    its value (DESIGN §2.5 cadence): the number never appears undated."""
    for entity in library.entities:
        if not entity.snapshot_facts:
            continue
        rendered = render_entity_html(entity)
        for fact in entity.snapshot_facts:
            as_of = fact.as_of.strftime("%-d %B %Y")
            assert as_of in rendered, f"{entity.id}/{fact.key}: as_of not rendered"
            assert f"{fact.value:,.0f}" in rendered, f"{entity.id}/{fact.key}: value not rendered"


# --------------------------------------------------------------------------- #
# 3 + 4. Ingestion labelling
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def voices_chunks(library) -> list[ChunkRecord]:
    return list(ingest_voices_entities(library, config=config()).chunks)


def test_voices_chunks_labelled_source_type_voices(voices_chunks) -> None:
    """Every chunk ingested from the voices layer is source_type: voices —
    the structural label the retrieval include-list keys on (#11)."""
    assert voices_chunks, "voices ingestion produced no chunks"
    assert all(c.source_type == "voices" for c in voices_chunks)
    # And it is a scalar string, never an array (finding #174 class).
    assert all(isinstance(c.source_type, str) for c in voices_chunks)


def test_voices_attribution_is_about_the_movement(voices_chunks) -> None:
    """Voices chunks carry the "About the movement" attribution and never
    a scientific-source attribution string."""
    science_attribution = manifest_entry("syn-doc")["attribution_text"]
    for chunk in voices_chunks:
        attribution = chunk.citation_metadata["attribution_text"]
        assert ABOUT_THE_MOVEMENT in attribution, f"{chunk.doc_id}: attribution not labelled"
        assert attribution != science_attribution
        # A voices chunk must never carry the evidence source type either.
        assert chunk.source_type != "evidence"


def test_voices_chunk_ids_are_deterministic(library) -> None:
    """Two ingests over the same content produce identical chunk ids —
    the idempotent-reindex property carries to voices too."""
    first = [c.chunk_id for c in ingest_voices_entities(library, config=config()).chunks]
    second = [c.chunk_id for c in ingest_voices_entities(library, config=config()).chunks]
    assert first == second and len(set(first)) == len(first)


# --------------------------------------------------------------------------- #
# 5. The real voices.yaml ingests and is retrievable (end-to-end, in-memory)
# --------------------------------------------------------------------------- #
#
# Runs in the unit tier because it needs neither Docker nor model weights:
# an in-memory Qdrant index + the deterministic HashEmbeddingModel, exactly
# like tests/unit/test_retrieval_voices_filter.py. It ingests the ACTUAL
# voices/voices.yaml (not a fixture) and exercises the #9 index + #11
# retrieval so "who is calling for a briefing?" returns voices chunks — and
# so those same chunks never leak onto a science route.


def _index_voices_with_evidence(library):
    """Build an in-memory index over the real voices chunks plus the
    synthetic evidence fixture corpus. Returns (client, model)."""
    from rag.query import ScopeClass  # noqa: F401  (import proves availability)
    from tests._indexing_fixtures import HashEmbeddingModel, build, fixture_corpus, fresh_client

    voices = ingest_voices_entities(library, config=config())
    ev_chunks, ev_records = fixture_corpus()

    chunks = [*ev_chunks, *voices.chunks]
    records = {**ev_records, **dict(voices.documents)}

    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)
    return client, model


def test_real_voices_yaml_ingests_and_is_retrievable(library) -> None:
    """The 'who is calling for a briefing?' query on the voices route
    retrieves voices chunks from the real content; the same query on a
    science route returns none (the §2.5/#11 separation)."""
    from rag.query import ScopeClass
    from rag.retrieval import RetrievedPassages
    from tests._retrieval_fixtures import TableReranker, decision, run_retrieve
    from tests._retrieval_fixtures import config as rconfig

    client, model = _index_voices_with_evidence(library)
    # Score the campaign's own vocabulary top so a leak would be loud.
    reranker = TableReranker(
        [("televised national emergency briefing", 0.99), ("National Emergency Briefing", 0.98)],
        default=0.05,
    )
    query = "who is calling for an emergency briefing and what can I do"

    voices_result = run_retrieve(
        client,
        decision(query, scope=ScopeClass.VOICES),
        model=model,
        reranker=reranker,
        cfg=rconfig(),
    )
    assert isinstance(voices_result, RetrievedPassages)
    voices_hits = [p for p in voices_result.passages if p.payload["source_type"] == "voices"]
    assert voices_hits, "the voices route must retrieve voices chunks for the briefing query"
    assert voices_result.passages[0].payload["source_type"] == "voices", (
        "the reranker scored voices text top, so a voices chunk must lead the voices-route result"
    )
    assert ABOUT_THE_MOVEMENT in voices_hits[0].payload["citation_metadata"]["attribution_text"]

    # Separation: the same pull query on a science route yields zero voices.
    science_result = run_retrieve(
        client,
        decision(query, scope=ScopeClass.IN_SCOPE),
        model=model,
        reranker=reranker,
        cfg=rconfig(),
    )
    assert isinstance(science_result, RetrievedPassages)
    assert all(p.payload["source_type"] != "voices" for p in science_result.passages), (
        "voices chunks must never be served on a science route (DESIGN §2.5 / #11)"
    )
