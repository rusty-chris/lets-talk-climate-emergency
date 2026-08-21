"""Index versioning tied to corpus version (issue #9 TDD plan 3) — RED.

An index answers for exactly one corpus vintage. The build records the
corpus version (and the embedding model id); retrieval states what it
expects and refuses on mismatch — stale-index answers are a silent
integrity failure, so the mismatch is an exception, never a warning.
"""

from __future__ import annotations

import pytest

from rag.indexing import (
    IndexingError,
    IndexVersionMismatchError,
    get_index_corpus_version,
    hybrid_query,
)
from tests._indexing_fixtures import (
    COLLECTION,
    FIXTURE_CORPUS_VERSION,
    HashEmbeddingModel,
    build,
    fixture_corpus,
    fresh_client,
)


def test_index_version_records_corpus_version() -> None:
    """TDD plan 3: the collection carries the corpus version it was built
    from, readable back verbatim."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    report = build(client, chunks, records, model=HashEmbeddingModel())

    assert get_index_corpus_version(client, COLLECTION) == FIXTURE_CORPUS_VERSION
    assert report.corpus_version == FIXTURE_CORPUS_VERSION


def test_rebuild_under_new_corpus_version_updates_recorded_version() -> None:
    """A re-build under a bumped corpus version moves the recorded
    version with it — the version describes the CURRENT contents."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=HashEmbeddingModel())
    build(client, chunks, records, model=HashEmbeddingModel(), corpus_version="fixture-corpus-v2")

    assert get_index_corpus_version(client, COLLECTION) == "fixture-corpus-v2"


def test_query_against_mismatched_corpus_version_refuses() -> None:
    """TDD plan 3, the detectable-mismatch half: querying an index built
    from corpus v1 while expecting v2 raises, naming BOTH versions so the
    operator can see which side is stale. No results are returned."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    with pytest.raises(IndexVersionMismatchError) as excinfo:
        hybrid_query(
            client,
            COLLECTION,
            "invented aurelian probe query",
            embedding_model=model,
            expected_corpus_version="fixture-corpus-v2",
        )
    message = str(excinfo.value)
    assert FIXTURE_CORPUS_VERSION in message
    assert "fixture-corpus-v2" in message


def test_query_under_different_embedding_model_id_refuses() -> None:
    """Vectors from different models share no space: a query embedded by
    a model other than the one that built the index would search garbage
    convincingly. The recorded model id makes that refuse loudly."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=HashEmbeddingModel())

    class OtherModel(HashEmbeddingModel):
        model_id = "some-other-embedder-v9"

    with pytest.raises(IndexingError, match="some-other-embedder-v9"):
        hybrid_query(
            client,
            COLLECTION,
            "invented aurelian probe query",
            embedding_model=OtherModel(),
            expected_corpus_version=FIXTURE_CORPUS_VERSION,
        )


def test_unversioned_collection_is_never_treated_as_current() -> None:
    """A collection this module didn't build (no recorded version) has
    unknown vintage; asking its version raises rather than defaulting."""
    client = fresh_client()
    from qdrant_client import models

    client.create_collection(
        collection_name="alien-collection",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )
    with pytest.raises(IndexingError):
        get_index_corpus_version(client, "alien-collection")
