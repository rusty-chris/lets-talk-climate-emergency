"""Hybrid indexing against the real composed Qdrant server (issue #9 TDD
plan 4–7) — RED. Integration tier (auto-marked by this directory's
conftest).

The compose file's pinned qdrant/qdrant:v1.19.0 serves on
127.0.0.1:6333; the embedder stays the DETERMINISTIC hash fake — the
review verdict scopes #9 to the indexing MECHANISM over the synthetic
fixture corpus, and the single real-model check lives in
test_bge_m3_smoke.py. Every test uses its own uniquely named collection
and deletes it afterwards, so runs never see each other's state.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from rag.indexing import (
    DEFAULT_TOP_K,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_chunk_point,
    hybrid_query,
    indexed_chunk_ids,
)
from tests._docker import require_docker
from tests._indexing_fixtures import (
    FIXTURE_CORPUS_VERSION,
    HashEmbeddingModel,
    RecordingEmbeddingModel,
    build,
    fixture_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QDRANT_URL = "http://127.0.0.1:6333"


@pytest.fixture(scope="module")
def qdrant_server():
    """The composed Qdrant service, healthy; stopped again afterwards
    unless it was already running before the tests."""
    require_docker()
    was_running = bool(
        subprocess.run(
            ["docker", "compose", "ps", "-q", "--status", "running", "qdrant"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    up = subprocess.run(
        ["docker", "compose", "up", "-d", "--wait", "qdrant"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert up.returncode == 0, up.stdout + up.stderr
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        yield client
        client.close()
    finally:
        if not was_running:
            subprocess.run(
                ["docker", "compose", "stop", "qdrant"], cwd=REPO_ROOT, capture_output=True
            )


@pytest.fixture
def collection(qdrant_server):
    """A unique collection name per test, deleted on the way out."""
    name = f"test-idx-{uuid.uuid4().hex[:12]}"
    yield name
    if qdrant_server.collection_exists(name):
        qdrant_server.delete_collection(name)


def test_dense_and_sparse_vectors_stored_per_chunk(qdrant_server, collection) -> None:
    """TDD plan 4 (scoped per the review verdict: mechanism over the
    synthetic corpus, deterministic embedder): the real server stores
    BOTH named vectors for every fixture chunk, and the §2.4 payload
    round-trips the wire intact."""
    chunks, records = fixture_corpus()
    model = HashEmbeddingModel()
    build(qdrant_server, chunks, records, model=model, collection=collection)

    info = qdrant_server.get_collection(collection)
    assert DENSE_VECTOR_NAME in info.config.params.vectors
    assert info.config.params.vectors[DENSE_VECTOR_NAME].size == model.dense_dim
    assert SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})

    assert indexed_chunk_ids(qdrant_server, collection) == {c.chunk_id for c in chunks}
    for chunk in (chunks[0], chunks[-1]):
        point = get_chunk_point(qdrant_server, collection, chunk.chunk_id)
        assert len(point.dense) == model.dense_dim
        assert point.sparse
        assert point.payload["body"] == chunk.body
        assert point.payload["source_type"] == chunk.source_type
        assert list(point.payload["section_path"]) == list(chunk.section_path)


def test_hybrid_query_returns_rrf_fused_top40_with_metadata(qdrant_server, collection) -> None:
    """TDD plan 5 / acceptance criterion: the server-side RRF-fused
    hybrid query returns the §3.2 top-40 with chunk metadata intact
    (section path, attribution, source_type)."""
    chunks, records = fixture_corpus()
    model = HashEmbeddingModel()
    build(qdrant_server, chunks, records, model=model, collection=collection)

    results = hybrid_query(
        qdrant_server,
        collection,
        "invented aurelian basin rainfall and warming query",
        embedding_model=model,
        expected_corpus_version=FIXTURE_CORPUS_VERSION,
    )

    assert len(results) == DEFAULT_TOP_K
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
    by_id = {c.chunk_id: c for c in chunks}
    for hit in results:
        chunk = by_id[hit.chunk_id]
        assert list(hit.payload["section_path"]) == list(chunk.section_path)
        assert (
            hit.payload["citation_metadata"]["attribution_text"]
            == (chunk.citation_metadata["attribution_text"])
        )
        assert hit.payload["source_type"] == chunk.source_type


def test_known_query_hits_expected_fixture_document_family(qdrant_server, collection) -> None:
    """TDD plan 6, the named acceptance test: a query in the synthetic
    attribution review's distinctive vocabulary retrieves that document
    family first — deterministic because both the corpus and the
    embedder are."""
    chunks, records = fixture_corpus()
    model = HashEmbeddingModel()
    build(qdrant_server, chunks, records, model=model, collection=collection)

    results = hybrid_query(
        qdrant_server,
        collection,
        "what explains the attribution of aurelian basin drying",
        embedding_model=model,
        expected_corpus_version=FIXTURE_CORPUS_VERSION,
    )

    assert results[0].payload["doc_id"] == "syn-idx-attribution"
    top_five_docs = [r.payload["doc_id"] for r in results[:5]]
    assert top_five_docs.count("syn-idx-attribution") >= 2, (
        f"the attribution family should dominate the ranking head, got {top_five_docs}"
    )


def test_full_reindex_idempotent(qdrant_server, collection) -> None:
    """TDD plan 7 / acceptance criterion `test_full_reindex_idempotent`:
    a no-change rebuild against the real server embeds NOTHING (observed
    at the embedder seam) and leaves the collection identical, and both
    runs record the wall-clock the build log requires."""
    chunks, records = fixture_corpus()
    first = build(
        qdrant_server, chunks, records, model=RecordingEmbeddingModel(), collection=collection
    )

    def state():
        return {
            chunk_id: (
                dict(get_chunk_point(qdrant_server, collection, chunk_id).payload),
                get_chunk_point(qdrant_server, collection, chunk_id).dense,
                dict(get_chunk_point(qdrant_server, collection, chunk_id).sparse),
            )
            for chunk_id in indexed_chunk_ids(qdrant_server, collection)
        }

    before = state()
    model = RecordingEmbeddingModel()
    second = build(qdrant_server, chunks, records, model=model, collection=collection)

    assert model.encoded_texts == [], "the no-change rebuild re-encoded chunks"
    assert second.embedded_count == 0
    assert second.skipped_unchanged_count == len(chunks)
    assert state() == before, "the rebuild must leave the collection byte-identical"
    assert first.wall_clock_seconds >= 0.0
    assert second.wall_clock_seconds >= 0.0
