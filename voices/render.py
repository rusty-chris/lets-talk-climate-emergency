"""Load, validate, render and ingest the voices layer (DESIGN.md §2.5).

The voices layer is first-party descriptive text about the people and
campaigns publicly communicating the climate emergency. It is authored by
this project (``voices/voices.yaml``), so it is freely licensable and
freely ingestable — but it is a DISTINCT source from the scientific
corpus:

* every chunk is labelled ``source_type: voices`` and carries the "About
  the movement" attribution, never a scientific-source attribution; the
  retrieval layer (#11) structurally refuses to serve voices chunks on a
  science route;
* snapshot facts (petition counts, MP counts, screening counts) carry
  ``as_of`` dates and are RENDERED with them, so an answer built from a
  voices chunk states how current the number is (DESIGN §2.5 cadence).

Ingestion reuses the #7 production pipeline exactly — ``parse_html`` ->
``chunk_document`` -> ``build_citation_blocks`` — so voices chunks are the
same ``ChunkRecord`` shape the indexer (#9) and retrieval (#11) already
consume. It does NOT go through the sha256-pinned fetch path: that path
exists to protect against drift of EXTERNAL sources, and voices text is
first-party and in-repo. The manifest entry each entity carries is
synthesised here (``permitted_context: open``, ``source_type: voices``).
"""

from __future__ import annotations

import datetime
import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ingestion.blocks import build_citation_blocks
from ingestion.pipeline import (
    ChunkRecord,
    DocumentIngestRecord,
    IngestConfig,
    IngestResult,
    chunk_document,
    parse_html,
)

__all__ = [
    "Person",
    "SnapshotFact",
    "VoicesEntity",
    "VoicesError",
    "VoicesLibrary",
    "ingest_voices",
    "load_voices",
    "render_entity_html",
]

#: The label every voices chunk is attributed under (DESIGN §2.5 "About
#: the movement"). The loader requires the file's ``attribution_text`` to
#: contain this phrase, so the attribution can never silently drift into
#: something that reads like a scientific citation.
ABOUT_THE_MOVEMENT = "About the movement"

#: The one source_type value voices content may ever carry. Kept in step
#: with ``rag.retrieval.KNOWN_SOURCE_TYPES`` — voices is a first-class,
#: closed-vocabulary source type, never a free string.
VOICES_SOURCE_TYPE = "voices"

#: First-party licence string recorded on every voices document. Voices
#: text is authored in-repo, so it ships as ``permitted_context: open``.
VOICES_LICENCE = (
    "First-party text authored by Let's Talk About the Climate Emergency "
    "(freely licensable; not a third-party work)"
)


class VoicesError(ValueError):
    """A voices.yaml schema violation. The message names the offending
    entity id (or the file) and the missing/invalid field."""


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotFact:
    """A mutable count that carries its own currency date (DESIGN §2.5).

    ``value`` is numeric; ``as_of`` is the date the number was verified.
    ``qualifier`` (e.g. "more than") lets an approximate count render
    honestly. Rendered text always includes ``as_of`` so the answer states
    how current the figure is.
    """

    key: str
    label: str
    value: float
    as_of: datetime.date
    source_url: str
    qualifier: str | None = None
    note: str | None = None

    def rendered_sentence(self) -> str:
        amount = f"{self.value:,.0f}" if float(self.value).is_integer() else f"{self.value:,}"
        qualifier = f"{self.qualifier} " if self.qualifier else ""
        as_of = self.as_of.strftime("%-d %B %Y")
        return (
            f"As of {as_of}, there were {qualifier}{amount} {self.label} "
            f"(source: {self.source_url})."
        )


@dataclass(frozen=True)
class Person:
    """One named individual with a one-line description and a link."""

    name: str
    one_liner: str
    link: str
    verify_at_signoff: bool = False


@dataclass(frozen=True)
class VoicesEntity:
    """One entity cluster in the voices layer."""

    id: str
    name: str
    category: str
    canonical_url: str
    one_liner: str
    prose: str
    links: tuple[Mapping[str, str], ...] = ()
    snapshot_facts: tuple[SnapshotFact, ...] = ()
    people: tuple[Person, ...] = ()
    link_only: bool = False

    @property
    def doc_id(self) -> str:
        return f"voices-{self.id}"


@dataclass(frozen=True)
class VoicesLibrary:
    """The whole validated voices layer."""

    version: int
    source_type: str
    attribution_text: str
    entities: tuple[VoicesEntity, ...] = field(default=())


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def _require_str(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VoicesError(f"{where}: required field {key!r} is missing or empty")
    return value.strip()


def _parse_snapshot_fact(raw: Mapping[str, Any], where: str) -> SnapshotFact:
    key = _require_str(raw, "key", where)
    fact_where = f"{where} snapshot fact {key!r}"
    label = _require_str(raw, "label", fact_where)
    source_url = _require_str(raw, "source_url", fact_where)

    if (
        "value" not in raw
        or isinstance(raw["value"], bool)
        or not isinstance(raw["value"], (int, float))
    ):
        raise VoicesError(f"{fact_where}: 'value' must be a number")

    # The invariant this whole layer's cadence rests on (DESIGN §2.5): a
    # numeric snapshot fact WITHOUT an as_of date is refused — a number
    # that cannot state how current it is must never reach an answer.
    as_of_raw = raw.get("as_of")
    if not as_of_raw:
        raise VoicesError(
            f"{fact_where}: numeric snapshot facts must carry an 'as_of' date "
            "(DESIGN §2.5) — refusing an undated count"
        )
    try:
        as_of = datetime.date.fromisoformat(str(as_of_raw))
    except ValueError as exc:
        raise VoicesError(f"{fact_where}: 'as_of' {as_of_raw!r} is not an ISO date") from exc

    qualifier = raw.get("qualifier")
    note = raw.get("note")
    return SnapshotFact(
        key=key,
        label=label,
        value=raw["value"],
        as_of=as_of,
        source_url=source_url,
        qualifier=qualifier.strip() if isinstance(qualifier, str) and qualifier.strip() else None,
        note=note.strip() if isinstance(note, str) and note.strip() else None,
    )


def _parse_person(raw: Mapping[str, Any], where: str) -> Person:
    name = _require_str(raw, "name", where)
    person_where = f"{where} person {name!r}"
    return Person(
        name=name,
        one_liner=_require_str(raw, "one_liner", person_where),
        link=_require_str(raw, "link", person_where),
        verify_at_signoff=bool(raw.get("verify_at_signoff", False)),
    )


def _parse_links(raw: Any, where: str) -> tuple[Mapping[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise VoicesError(f"{where}: 'links' must be a list")
    links: list[Mapping[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise VoicesError(f"{where}: each link must be a mapping with 'label' and 'url'")
        links.append(
            {"label": _require_str(item, "label", where), "url": _require_str(item, "url", where)}
        )
    return tuple(links)


def _parse_entity(raw: Mapping[str, Any]) -> VoicesEntity:
    if not isinstance(raw, Mapping):
        raise VoicesError("each entity must be a mapping")
    entity_id = _require_str(raw, "id", "entity")
    where = f"entity {entity_id!r}"
    snapshot_raw = raw.get("snapshot_facts") or []
    if not isinstance(snapshot_raw, list):
        raise VoicesError(f"{where}: 'snapshot_facts' must be a list")
    people_raw = raw.get("people") or []
    if not isinstance(people_raw, list):
        raise VoicesError(f"{where}: 'people' must be a list")
    return VoicesEntity(
        id=entity_id,
        name=_require_str(raw, "name", where),
        category=_require_str(raw, "category", where),
        canonical_url=_require_str(raw, "canonical_url", where),
        one_liner=_require_str(raw, "one_liner", where),
        prose=_require_str(raw, "prose", where),
        links=_parse_links(raw.get("links"), where),
        snapshot_facts=tuple(_parse_snapshot_fact(item, where) for item in snapshot_raw),
        people=tuple(_parse_person(item, where) for item in people_raw),
        link_only=bool(raw.get("link_only", False)),
    )


def load_voices(path: str | Path) -> VoicesLibrary:
    """Load and validate ``voices/voices.yaml`` into a typed library.

    Raises :class:`VoicesError` naming the offending entity/field on any
    schema violation: a missing name/canonical_url/one_liner/prose, a
    person without a link, a numeric snapshot fact without an ``as_of``
    date, or an ``attribution_text`` that does not carry the "About the
    movement" label.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise VoicesError(f"{path}: top level must be a mapping")

    source_type = raw.get("source_type")
    if source_type != VOICES_SOURCE_TYPE:
        raise VoicesError(
            f"{path}: 'source_type' must be {VOICES_SOURCE_TYPE!r}, got {source_type!r}"
        )

    attribution_text = _require_str(raw, "attribution_text", str(path))
    if ABOUT_THE_MOVEMENT.lower() not in attribution_text.lower():
        raise VoicesError(
            f"{path}: 'attribution_text' must carry the {ABOUT_THE_MOVEMENT!r} label "
            "(DESIGN §2.5) so voices answers are never rendered as a scientific citation"
        )

    entities_raw = raw.get("entities") or []
    if not isinstance(entities_raw, list) or not entities_raw:
        raise VoicesError(f"{path}: 'entities' must be a non-empty list")

    entities = tuple(_parse_entity(item) for item in entities_raw)
    ids = [e.id for e in entities]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise VoicesError(f"{path}: duplicate entity ids {sorted(duplicates)}")

    return VoicesLibrary(
        version=int(raw.get("version", 1)),
        source_type=source_type,
        attribution_text=attribution_text,
        entities=entities,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_entity_html(entity: VoicesEntity) -> str:
    """Render one entity to a single-section HTML document for ingestion.

    Everything lives under ONE numbered ``<h2>`` heading so the
    structure-aware chunker packs it into ``source_type: voices`` chunks
    as a single section. The heading is numbered deliberately: an
    UNNUMBERED opening heading (or a bare ``<h1>`` title) arms the #7
    front-matter "affiliation wall" stripper, which would drop the
    people/links paragraphs as non-prose boilerplate — a numbered opening
    heading is treated as real structure and keeps every block. Snapshot
    facts are rendered as sentences that INCLUDE their ``as_of`` date
    (DESIGN §2.5), so a retrieved voices chunk states how current each
    figure is.
    """
    parts: list[str] = [f"<h2>1 {html.escape(entity.name)}</h2>"]
    parts.append(f"<p>{html.escape(entity.one_liner)}</p>")

    for paragraph in entity.prose.strip().split("\n\n"):
        text = " ".join(paragraph.split())
        if text:
            parts.append(f"<p>{html.escape(text)}</p>")

    for fact in entity.snapshot_facts:
        parts.append(f"<p>{html.escape(fact.rendered_sentence())}</p>")

    for person in entity.people:
        parts.append(
            f"<p>{html.escape(person.name)} — {html.escape(person.one_liner)} "
            f"({html.escape(person.link)})</p>"
        )

    if entity.links:
        rendered = "; ".join(
            f"{html.escape(link['label'])}: {html.escape(link['url'])}" for link in entity.links
        )
        parts.append(f"<p>Find out more — {rendered}.</p>")

    body = "\n".join(parts)
    return f"<html><body>{body}</body></html>"


def entity_manifest_entry(library: VoicesLibrary, entity: VoicesEntity) -> dict[str, Any]:
    """The synthetic manifest entry an entity is ingested under.

    Carries exactly the keys the #7 chunker/citation-metadata path reads,
    with the voices labelling (``source_type: voices``, the "About the
    movement" attribution) baked in. ``consensus_position: assessed`` is
    inert here — voices chunks never contribute scientific claims — but the
    field is present because the chunker propagates it.
    """
    return {
        "id": entity.doc_id,
        "title": entity.name,
        "licence": VOICES_LICENCE,
        "attribution_text": library.attribution_text,
        "canonical_url": entity.canonical_url,
        "permitted_context": "open",
        "consensus_position": "assessed",
        "source_type": library.source_type,
    }


# ---------------------------------------------------------------------------
# Ingestion (reuses the #7 pipeline)
# ---------------------------------------------------------------------------


def ingest_voices_entities(
    library: VoicesLibrary,
    config: IngestConfig | None = None,
) -> IngestResult:
    """Chunk every entity through the #7 pipeline into a voices IngestResult.

    Each entity is rendered to HTML, parsed via :func:`parse_html`, and
    chunked via the production :func:`chunk_document` under a synthesised
    ``source_type: voices`` manifest entry. Every resulting chunk therefore
    carries ``source_type == "voices"`` and the "About the movement"
    attribution, and one citation block is emitted per chunk. No network,
    no sha256 fetch, no Docling — voices text is first-party and in-repo.
    """
    config = config or IngestConfig()
    chunks: list[ChunkRecord] = []
    documents: dict[str, DocumentIngestRecord] = {}
    for entity in library.entities:
        entry = entity_manifest_entry(library, entity)
        sdoc = parse_html(render_entity_html(entity), entity.doc_id, title=entity.name)
        doc_chunks = chunk_document(sdoc, entry, config)
        documents[entity.doc_id] = DocumentIngestRecord(
            doc_id=entity.doc_id,
            parse_backend="html",
            chunk_count=len(doc_chunks),
        )
        chunks.extend(doc_chunks)
    blocks = build_citation_blocks(chunks)
    return IngestResult(chunks=tuple(chunks), blocks=tuple(blocks), documents=documents)


def ingest_voices(
    path: str | Path,
    config: IngestConfig | None = None,
) -> IngestResult:
    """Load ``voices/voices.yaml`` at ``path`` and ingest it (convenience)."""
    return ingest_voices_entities(load_voices(path), config=config)


def snapshot_facts(library: VoicesLibrary) -> Iterable[tuple[VoicesEntity, SnapshotFact]]:
    """Every (entity, snapshot fact) pair — used by tests and refresh tools."""
    for entity in library.entities:
        for fact in entity.snapshot_facts:
            yield entity, fact
