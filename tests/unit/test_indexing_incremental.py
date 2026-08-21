"""Incremental re-embedding by content hash (issue #9 TDD plan 1–2) — RED.

The incremental mechanism is asserted as OBSERVED embedder behaviour (a
RecordingEmbeddingModel counts every text encoded), never by trusting
the build report alone. Unit tier: deterministic hash-fake embedder, an
embedded in-memory Qdrant client (the same interface as the Docker
server), no Docker, no network, no model weights.
"""

from __future__ import annotations

import dataclasses

import pytest

from rag.indexing import DuplicateChunkIdError, indexed_chunk_ids
from tests._indexing_fixtures import (
    COLLECTION,
    RecordingEmbeddingModel,
    build,
    fixture_corpus,
    fresh_client,
    titled_doc_chunks,
)


def test_unchanged_content_hash_skips_reembedding() -> None:
    """TDD plan 1: a second build over identical chunks embeds ZERO texts.

    The acceptance criterion behind `test_full_reindex_idempotent`, at
    unit scale: the no-change re-run must not re-encode anything — the
    recording embedder sees no texts at all on the second run, and the
    report says so.
    """
    chunks, records = fixture_corpus()
    client = fresh_client()

    first_model = RecordingEmbeddingModel()
    first = build(client, chunks, records, model=first_model)
    assert first.embedded_count == len(chunks)
    assert sorted(first_model.encoded_texts) == sorted(c.embedding_text for c in chunks)

    second_model = RecordingEmbeddingModel()
    second = build(client, chunks, records, model=second_model)
    assert second_model.encoded_texts == [], (
        "a no-change rebuild re-encoded chunks — the content-hash skip is not working"
    )
    assert second.embedded_count == 0
    assert second.skipped_unchanged_count == len(chunks)
    assert second.indexed_chunk_count == len(chunks)


def test_changed_chunk_reembeds_only_that_chunk() -> None:
    """TDD plan 2: change ONE section's text — exactly one chunk is
    re-encoded; the stale chunk id (content-addressed, so the change
    moves it) is deleted; every unchanged chunk is skipped untouched."""
    chunks_v1, records = fixture_corpus()
    chunks_v2, _ = fixture_corpus(
        closer=(
            "A revised invented closing sentence stating that attribution confidence "
            "for the Aurelian drying signal is now rated higher than before."
        )
    )
    ids_v1 = {c.chunk_id for c in chunks_v1}
    changed = [c for c in chunks_v2 if c.chunk_id not in ids_v1]
    assert len(changed) == 1, "fixture self-check: exactly one chunk differs between versions"

    client = fresh_client()
    build(client, chunks_v1, records, model=RecordingEmbeddingModel())

    model = RecordingEmbeddingModel()
    report = build(client, chunks_v2, records, model=model)

    assert model.encoded_texts == [changed[0].embedding_text], (
        "the incremental build must re-encode exactly the changed chunk and nothing else"
    )
    assert report.embedded_count == 1
    assert report.skipped_unchanged_count == len(chunks_v2) - 1
    assert report.deleted_count == 1, "the superseded content-addressed id must be deleted"
    assert indexed_chunk_ids(client, COLLECTION) == {c.chunk_id for c in chunks_v2}


def test_removed_chunk_is_deleted_without_reembedding_the_rest() -> None:
    """A chunk id that leaves the corpus leaves the index: its point is
    deleted, and the removal costs zero re-encoding of survivors."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    removed_doc = "syn-idx-filler-c"
    kept_chunks = [c for c in chunks if c.doc_id != removed_doc]
    kept_records = {doc_id: r for doc_id, r in records.items() if doc_id != removed_doc}
    assert len(kept_chunks) < len(chunks), "fixture self-check: something was removed"

    model = RecordingEmbeddingModel()
    report = build(client, kept_chunks, kept_records, model=model)

    assert model.encoded_texts == []
    assert report.deleted_count == len(chunks) - len(kept_chunks)
    assert report.indexed_chunk_count == len(kept_chunks)
    assert indexed_chunk_ids(client, COLLECTION) == {c.chunk_id for c in kept_chunks}


def test_context_header_change_reembeds_despite_unchanged_chunk_id() -> None:
    """The content hash covers the FULL embedded text, not just the body.

    `chunk_id` hashes doc id + section path + body — a document retitle
    changes every context header (hence `embedding_text`, the text the
    embedder actually sees) while leaving every chunk id unchanged. An
    implementation that keys the skip on chunk-id presence alone would
    silently serve vectors of text that no longer exists. The retitled
    build must re-encode.
    """
    chunks_v1, records = titled_doc_chunks("Original invented title")
    chunks_v2, _ = titled_doc_chunks("Renamed invented title")
    assert [c.chunk_id for c in chunks_v1] == [c.chunk_id for c in chunks_v2]
    assert [c.embedding_text for c in chunks_v1] != [c.embedding_text for c in chunks_v2]

    client = fresh_client()
    build(client, chunks_v1, records, model=RecordingEmbeddingModel())

    model = RecordingEmbeddingModel()
    report = build(client, chunks_v2, records, model=model)

    assert sorted(model.encoded_texts) == sorted(c.embedding_text for c in chunks_v2), (
        "header-only changes were skipped — the content hash must cover embedding_text, "
        "not merely prove the chunk_id already exists"
    )
    assert report.embedded_count == len(chunks_v2)


def test_indexer_embeds_embedding_text_never_the_bare_body() -> None:
    """What gets encoded is `ChunkRecord.embedding_text` (context header +
    body — the #7 contract for the embedded representation), never the
    header-less body."""
    chunks, records = fixture_corpus()
    model = RecordingEmbeddingModel()
    build(fresh_client(), chunks, records, model=model)

    expected = {c.embedding_text for c in chunks}
    encoded = set(model.encoded_texts)
    assert encoded == expected
    stray_bodies = {c.body for c in chunks} & encoded
    assert stray_bodies == set(), "bare chunk bodies were embedded instead of embedding_text"


def test_duplicate_incoming_chunk_ids_refuse_loudly() -> None:
    """Id uniqueness is a precondition the indexer ENFORCES, never assumes.

    Review #7 found real chunk-id collisions in the chunker (fix landing
    in parallel). Because the chunk id is the upsert key, a silent upsert
    of colliding ids drops one chunk's evidence from the index with no
    trace. The build must refuse the whole run — naming the colliding
    id — and write nothing.
    """
    chunks, records = fixture_corpus()
    victim, imposter = chunks[0], chunks[1]
    collided = dataclasses.replace(imposter, chunk_id=victim.chunk_id)
    poisoned = [victim, collided, *chunks[2:]]

    client = fresh_client()
    model = RecordingEmbeddingModel()
    with pytest.raises(DuplicateChunkIdError, match=victim.chunk_id):
        build(client, poisoned, records, model=model)

    assert model.encoded_texts == [], "a refused build must not spend embedding work"
    assert client.collection_exists(COLLECTION) is False, (
        "a refused build must leave the store exactly as it found it — refusal comes "
        "before any write, including collection creation"
    )
