"""Collection schema, payload metadata carriage, and the #143 hand-review
enforcement point (issue #9) — RED.

Unit tier: embedded in-memory Qdrant client (identical interface to the
Docker server), deterministic hash-fake embedder, no weights.
"""

from __future__ import annotations

import pytest

from rag.indexing import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    IndexingError,
    UnreviewedDegradedChunksError,
    get_chunk_point,
    indexed_chunk_ids,
)
from tests._indexing_fixtures import (
    COLLECTION,
    HashEmbeddingModel,
    build,
    degraded_records,
    fixture_corpus,
    fresh_client,
)


def test_collection_uses_named_dense_and_sparse_vectors() -> None:
    """ADR-007 schema: ONE collection, TWO named vectors per point — the
    bge-m3 dense embedding and the bge-m3 LEARNED SPARSE embedding (not
    BM25). The dense size is the embedding model's dense_dim; every
    stored point carries both named vectors."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    info = client.get_collection(COLLECTION)
    dense_config = info.config.params.vectors
    assert DENSE_VECTOR_NAME in dense_config, "the dense vector must be NAMED, not default"
    assert dense_config[DENSE_VECTOR_NAME].size == model.dense_dim
    sparse_config = info.config.params.sparse_vectors
    assert sparse_config is not None and SPARSE_VECTOR_NAME in sparse_config

    point = get_chunk_point(client, COLLECTION, chunks[0].chunk_id)
    assert len(point.dense) == model.dense_dim
    assert point.sparse, "the learned-sparse vector must be stored per point"
    assert all(isinstance(k, int) and v > 0 for k, v in point.sparse.items())


def test_payload_carries_full_chunk_metadata() -> None:
    """DESIGN §2.4 metadata round-trips into the payload, field for field.

    The payload is the ONLY carrier of citation metadata through
    retrieval (issue #9 acceptance: 'metadata intact'), so every §2.4
    field must be present and equal to the ChunkRecord's value —
    including consensus_position and source_type, which downstream
    guardrails (#11's voices filter, §3.3 rule 8) filter on.
    """
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=HashEmbeddingModel())

    beyond = next(c for c in chunks if c.consensus_position == "beyond-assessed-range")
    voices = next(c for c in chunks if c.source_type == "voices")

    for chunk in (beyond, voices, chunks[0]):
        payload = get_chunk_point(client, COLLECTION, chunk.chunk_id).payload
        assert payload["chunk_id"] == chunk.chunk_id
        assert payload["doc_id"] == chunk.doc_id
        assert list(payload["section_path"]) == list(chunk.section_path)
        assert payload["context_header"] == chunk.context_header
        assert payload["body"] == chunk.body
        assert payload["token_count"] == chunk.token_count
        assert list(payload["confidence_markers"]) == list(chunk.confidence_markers)
        assert list(payload["block_types"]) == list(chunk.block_types)
        assert payload["consensus_position"] == chunk.consensus_position
        assert payload["source_type"] == chunk.source_type
        citation = payload["citation_metadata"]
        for key in ("title", "licence", "attribution_text", "canonical_url", "permitted_context"):
            assert citation[key] == chunk.citation_metadata[key], key

    assert get_chunk_point(client, COLLECTION, voices.chunk_id).payload["source_type"] == "voices"


def test_payload_carries_parse_provenance_flags() -> None:
    """The #143 flags are on EVERY payload, queryable after the fact:
    parse_backend, degraded_fallback and needs_hand_review come from the
    document's ingest record — False/clean here, so their presence (not
    their truth) is what this pins."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=HashEmbeddingModel())

    payload = get_chunk_point(client, COLLECTION, chunks[0].chunk_id).payload
    assert payload["parse_backend"] == "html"
    assert payload["degraded_fallback"] is False
    assert payload["needs_hand_review"] is False


def test_build_refuses_unreviewed_degraded_chunks() -> None:
    """Issue #143's enforcement point: a document flagged
    needs_hand_review (the loud PyMuPDF fallback) must NOT reach the
    index — amended DESIGN §2.4 requires hand review BEFORE indexing.
    The build refuses, naming the document, and writes nothing."""
    chunks, records = fixture_corpus()
    flagged = degraded_records(records, "syn-idx-basin")

    client = fresh_client()
    with pytest.raises(UnreviewedDegradedChunksError, match="syn-idx-basin"):
        build(client, chunks, flagged, model=HashEmbeddingModel())

    assert client.collection_exists(COLLECTION) is False, (
        "a refused build must write nothing — not even the collection schema"
    )


def test_build_refuses_chunk_without_document_record() -> None:
    """Provenance is never optional: a chunk whose doc_id has no ingest
    record cannot be given the #143 flags, so the build refuses loudly
    instead of guessing them."""
    chunks, records = fixture_corpus()
    missing = dict(records)
    del missing["syn-idx-voices"]

    client = fresh_client()
    with pytest.raises(IndexingError, match="syn-idx-voices"):
        build(client, chunks, missing, model=HashEmbeddingModel())
    assert client.collection_exists(COLLECTION) is False


def test_same_corpus_builds_identical_collection_state_twice() -> None:
    """Determinism: the same fixture corpus built into two fresh stores
    yields identical collection state — same chunk ids, payloads and
    both vectors, point for point. (The reproducibility property ADR-005
    chose local embedding for, at the index level.)"""
    chunks, records = fixture_corpus()

    def state(client):
        ids = indexed_chunk_ids(client, COLLECTION)
        return {
            chunk_id: (
                dict(get_chunk_point(client, COLLECTION, chunk_id).payload),
                get_chunk_point(client, COLLECTION, chunk_id).dense,
                dict(get_chunk_point(client, COLLECTION, chunk_id).sparse),
            )
            for chunk_id in ids
        }

    client_a = fresh_client()
    client_b = fresh_client()
    build(client_a, chunks, records, model=HashEmbeddingModel())
    build(client_b, chunks, records, model=HashEmbeddingModel())

    state_a = state(client_a)
    state_b = state(client_b)
    assert set(state_a) == {c.chunk_id for c in chunks}
    assert state_a == state_b
