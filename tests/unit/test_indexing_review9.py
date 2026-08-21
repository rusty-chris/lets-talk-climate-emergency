"""Adversarial-review findings on the #9 embedding + indexing feature — RED.

One test (family) per review finding on PR #154, red before each fix:

- #159: an iterator passed as ``chunks`` must not bypass the pre-write
  refusals (the multi-pass checks silently see an exhausted stream).
- #167: the ``__meta`` companion-collection namespace is reserved; a
  chunk build into it must refuse before any write, and a foreign point
  without a ``chunk_id`` payload must fail with a named error.
- #158: ``source_type`` is a closed vocabulary end-to-end — unknown,
  miscased or null values refuse the build, and the query filter helper
  accepts only known values (it never normalises).
- #155: a metadata-only change (unchanged embedding text) must refresh
  the stored payload without re-encoding anything.
- #156: rebuilding under a different embedding model id (or dense_dim)
  must refuse unless the caller explicitly asks for a full reindex —
  never re-stamp the meta over stale vectors.
- #157: a crash between the first store mutation and the final meta
  write must leave the index loudly unqueryable, never confidently
  mislabeled with the previous corpus version.
- #163: the bge-m3 revision is pinned (full commit hash) and forwarded
  to the loader as the resolved snapshot path.

Unit tier: deterministic fakes, in-memory Qdrant, no weights, no network.
"""

from __future__ import annotations

import dataclasses

import pytest

from ingestion.pipeline import DocumentIngestRecord
from rag.indexing import (
    IndexingError,
    UnreviewedDegradedChunksError,
    get_chunk_point,
    hybrid_query,
    indexed_chunk_ids,
)
from tests._indexing_fixtures import (
    COLLECTION,
    HashEmbeddingModel,
    RecordingEmbeddingModel,
    build,
    degraded_records,
    fixture_corpus,
    fresh_client,
)


def test_build_materialises_input_before_refusal_checks() -> None:
    """Finding #159: `build_index` iterates its input several times; a
    generator/iterator argument exhausts on the first pass, so the
    needs-hand-review refusal sees an empty stream and unreviewed
    degraded chunks get INDEXED before the run dies late on an
    unrelated TypeError. The iterator case must behave identically to
    the list case: refuse loudly, write nothing."""
    chunks, records = fixture_corpus()
    flagged = degraded_records(records, "syn-idx-basin")
    client = fresh_client()
    model = RecordingEmbeddingModel()

    with pytest.raises(UnreviewedDegradedChunksError, match="syn-idx-basin"):
        build(client, iter(chunks), flagged, model=model)

    assert model.encoded_texts == [], "a refused build must not spend embedding work"
    assert client.collection_exists(COLLECTION) is False, (
        "a refused build must leave the store exactly as it found it — the iterator "
        "input bypassed the pre-write refusals"
    )


def test_build_refuses_reserved_meta_collection_name() -> None:
    """Finding #167: index metadata lives in a companion collection named
    `<collection>__meta`, so a chunk build INTO a name ending in the
    reserved suffix collides with another collection's metadata store
    (foreign meta points in a chunk collection, KeyError crashes on the
    next rebuild). The API boundary must refuse the name up front,
    naming the reserved suffix, before any write."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    model = RecordingEmbeddingModel()

    with pytest.raises(IndexingError, match="__meta"):
        build(client, chunks, records, model=model, collection="anything__meta")

    assert model.encoded_texts == [], "a refused build must not spend embedding work"
    assert client.collection_exists("anything__meta") is False, (
        "the reserved-name refusal must come before any write"
    )


def test_foreign_point_without_chunk_id_fails_with_named_error() -> None:
    """Finding #167 (second half): a point without a `chunk_id` payload
    (a foreign or corrupted collection) must surface as a named
    IndexingError diagnosing the collection, never a bare KeyError far
    from the cause."""
    from qdrant_client import models as qmodels

    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    # Pollute the chunk collection with a foreign point (the shape the
    # meta-namespace collision produced before the guard existed).
    client.upsert(
        collection_name=COLLECTION,
        points=[
            qmodels.PointStruct(
                id=str(__import__("uuid").uuid4()),
                vector={"dense": [0.0] * RecordingEmbeddingModel().dense_dim},
                payload={"corpus_version": "foreign-v1"},
            )
        ],
    )

    with pytest.raises(IndexingError, match=COLLECTION):
        build(client, chunks, records, model=RecordingEmbeddingModel())
    with pytest.raises(IndexingError, match=COLLECTION):
        indexed_chunk_ids(client, COLLECTION)


@pytest.mark.parametrize("bad_value", ["Voices", "propaganda", None, ""])
def test_build_refuses_unknown_source_type(bad_value) -> None:
    """Finding #158: the #11 voices exclusion is a case-sensitive exact
    blocklist over the source_type payload field, so any value outside
    the closed vocabulary (miscased, misspelt, null) silently escapes
    it and gets served as evidence. The indexer must enforce the
    vocabulary the way it enforces needs_hand_review: refuse the build
    loudly, naming the value and the document, before any write."""
    chunks, records = fixture_corpus()
    poisoned = [
        dataclasses.replace(c, source_type=bad_value) if c.doc_id == "syn-idx-basin" else c
        for c in chunks
    ]
    client = fresh_client()
    model = RecordingEmbeddingModel()

    with pytest.raises(IndexingError, match="syn-idx-basin"):
        build(client, poisoned, records, model=model)

    assert model.encoded_texts == []
    assert client.collection_exists(COLLECTION) is False, (
        "the source_type vocabulary refusal must come before any write"
    )


def test_source_types_vocabulary_is_closed_and_exact() -> None:
    """The module declares the closed vocabulary once; DESIGN §2.5/§3.2
    know exactly two source layers today."""
    from rag.indexing import SOURCE_TYPES

    assert SOURCE_TYPES == frozenset({"evidence", "voices"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exclude_source_types": ("Voices",)},
        {"include_source_types": ("evidence", "campaign")},
    ],
)
def test_query_filter_accepts_only_known_source_types(kwargs) -> None:
    """Finding #158 (query side): the filter helper never normalises —
    an unknown or miscased source_type in include/exclude lists is a
    caller bug that would silently filter nothing, so it refuses."""
    from rag.indexing import hybrid_query

    chunks, records = fixture_corpus()
    client = fresh_client()
    model = RecordingEmbeddingModel()
    build(client, chunks, records, model=model)

    with pytest.raises(IndexingError, match="source_type"):
        hybrid_query(
            client,
            COLLECTION,
            "invented aurelian probe query",
            embedding_model=model,
            expected_corpus_version="fixture-corpus-v1",
            **kwargs,
        )


def test_metadata_only_change_updates_stored_payload_without_reembedding() -> None:
    """Finding #155: the incremental skip keys on the embedding-text hash
    and, on a match, skips the chunk ENTIRELY — so a metadata-only
    change (a doc reclassified to voices, text unchanged) never reaches
    the index, and the voices exclusion keeps serving it as evidence.
    The skip may only ever skip the EMBEDDING work: the stored payload
    must still be refreshed, and filters must see the new source_type."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    # Reclassify one document to the voices layer; every byte of every
    # embedding_text is unchanged.
    reclassified = [
        dataclasses.replace(c, source_type="voices") if c.doc_id == "syn-idx-basin" else c
        for c in chunks
    ]
    assert [c.embedding_text for c in reclassified] == [c.embedding_text for c in chunks]

    model = RecordingEmbeddingModel()
    build(client, reclassified, records, model=model, corpus_version="fixture-corpus-v2")

    assert model.encoded_texts == [], (
        "unchanged text must not be re-encoded — the vectors are legitimately reused"
    )
    basin_chunk = next(c for c in reclassified if c.doc_id == "syn-idx-basin")
    stored = get_chunk_point(client, COLLECTION, basin_chunk.chunk_id)
    assert stored.payload["source_type"] == "voices", (
        "the metadata-only change never reached the stored payload — the "
        "content-hash skip must skip embedding work only, never the metadata write"
    )

    results = hybrid_query(
        client,
        COLLECTION,
        "aurelian basin surface temperatures reservoir inflows",
        embedding_model=model,
        expected_corpus_version="fixture-corpus-v2",
        exclude_source_types=("voices",),
    )
    assert all(r.payload["doc_id"] != "syn-idx-basin" for r in results), (
        "the voices exclusion still served the reclassified document as evidence"
    )


def test_licence_and_parse_flag_corrections_reach_the_stored_payload() -> None:
    """Finding #155, the licensing/#143 halves: the stored payload is the
    ONLY carrier of citation metadata and parse-provenance flags through
    retrieval. A licence correction or a reviewed re-parse with
    byte-identical text must land in the payload on rebuild."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    target = next(c for c in chunks if c.doc_id == "syn-idx-attribution")
    corrected_citation = {
        **dict(target.citation_metadata),
        "licence": "CORRECTED-LICENCE-1.0",
        "attribution_text": "Corrected invented attribution line.",
    }
    corrected = [
        dataclasses.replace(c, citation_metadata=corrected_citation)
        if c.chunk_id == target.chunk_id
        else c
        for c in chunks
    ]
    reparsed_records = {
        **records,
        "syn-idx-basin": DocumentIngestRecord(
            doc_id="syn-idx-basin",
            parse_backend="pymupdf",
            degraded_fallback=True,
            needs_hand_review=False,
        ),
    }

    model = RecordingEmbeddingModel()
    build(client, corrected, reparsed_records, model=model)

    assert model.encoded_texts == []
    stored_citation = get_chunk_point(client, COLLECTION, target.chunk_id).payload[
        "citation_metadata"
    ]
    assert stored_citation["licence"] == "CORRECTED-LICENCE-1.0"
    assert stored_citation["attribution_text"] == "Corrected invented attribution line."

    basin_chunk = next(c for c in chunks if c.doc_id == "syn-idx-basin")
    basin_payload = get_chunk_point(client, COLLECTION, basin_chunk.chunk_id).payload
    assert basin_payload["parse_backend"] == "pymupdf"
    assert basin_payload["degraded_fallback"] is True
    assert basin_payload["needs_hand_review"] is False


class _OtherHashModel(HashEmbeddingModel):
    """A second, incompatible embedding model: different id, different
    vector space (different dense_dim)."""

    model_id = "totally-different-embedder-v2"
    dense_dim = 32


def test_rebuild_under_new_model_id_reembeds_everything_or_refuses() -> None:
    """Finding #156: the content hash ignores the embedding model, so a
    rebuild under a DIFFERENT model re-embeds nothing, reuses every
    old-model vector, and re-stamps the meta with the new model id —
    subverting the very model-mismatch refusal the meta exists for.
    Without an explicit reindex the build must refuse loudly, leaving
    the recorded model id unchanged; embedded_count == 0 plus a
    re-stamped id must be impossible."""
    from rag.indexing import get_index_corpus_version

    chunks, records = fixture_corpus()
    client = fresh_client()
    model_a = RecordingEmbeddingModel()
    build(client, chunks, records, model=model_a)

    model_b = RecordingEmbeddingModel(_OtherHashModel())
    with pytest.raises(IndexingError, match="totally-different-embedder-v2"):
        build(client, chunks, records, model=model_b)

    assert model_b.encoded_texts == [], "a refused build must not spend embedding work"
    # The old index is untouched and still answers under the OLD model.
    assert get_index_corpus_version(client, COLLECTION) == "fixture-corpus-v1"
    hits = hybrid_query(
        client,
        COLLECTION,
        "invented aurelian probe query",
        embedding_model=model_a,
        expected_corpus_version="fixture-corpus-v1",
    )
    assert hits, "the refused rebuild must leave the old-model index intact"
    # The new model must NOT pass the query-side mismatch check.
    with pytest.raises(IndexingError, match="totally-different-embedder-v2"):
        hybrid_query(
            client,
            COLLECTION,
            "invented aurelian probe query",
            embedding_model=model_b,
            expected_corpus_version="fixture-corpus-v1",
        )


def test_rebuild_with_explicit_reindex_flag_reembeds_everything() -> None:
    """Finding #156, the sanctioned path: an operator changing models
    passes reindex=True and gets a FULL re-embed under the new model —
    every chunk re-encoded (observed at the seam), the collection
    recreated at the new dense_dim, the meta recording the new model.
    Old-model vectors never survive into the new model's index."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    model_b = RecordingEmbeddingModel(_OtherHashModel())
    report = build(client, chunks, records, model=model_b, reindex=True)

    assert sorted(model_b.encoded_texts) == sorted(c.embedding_text for c in chunks), (
        "reindex=True must re-encode every chunk under the new model"
    )
    assert report.embedded_count == len(chunks)
    assert report.embedding_model_id == "totally-different-embedder-v2"
    stored = get_chunk_point(client, COLLECTION, chunks[0].chunk_id)
    assert len(stored.dense) == _OtherHashModel.dense_dim, (
        "the collection must be recreated at the new model's dense_dim"
    )
    hits = hybrid_query(
        client,
        COLLECTION,
        "invented aurelian probe query",
        embedding_model=model_b,
        expected_corpus_version="fixture-corpus-v1",
    )
    assert hits


def test_rebuild_under_same_id_but_different_dense_dim_refuses() -> None:
    """Finding #156, the dimension half: the incremental skip is only
    valid when model id AND dense_dim both match the recorded build;
    a same-id model with a different dimensionality must refuse too
    (nothing gets re-embedded that could trip the server's dim check)."""

    class SameIdDifferentDim(HashEmbeddingModel):
        dense_dim = 32

    chunks, records = fixture_corpus()
    client = fresh_client()
    build(client, chunks, records, model=RecordingEmbeddingModel())

    model = RecordingEmbeddingModel(SameIdDifferentDim())
    with pytest.raises(IndexingError, match="dense_dim|dimension"):
        build(client, chunks, records, model=model)
    assert model.encoded_texts == []


class _CrashingClient:
    """Delegates every call to a real in-memory client, but raises an
    injected ConnectionError the first time any of the named methods is
    called — the seam for simulating a mid-build crash."""

    def __init__(self, inner, fail_on: set[str]) -> None:
        self._inner = inner
        self._fail_on = set(fail_on)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._fail_on and callable(attr):

            def _boom(*args, **kwargs):
                raise ConnectionError(f"injected mid-build failure in {name}()")

            return _boom
        return attr


def _v2_corpus():
    return fixture_corpus(
        closer=(
            "A revised invented closing sentence stating that attribution confidence "
            "for the Aurelian drying signal is now rated higher than before."
        )
    )


def test_interrupted_build_never_leaves_old_version_stamp_over_new_contents() -> None:
    """Finding #157: build_index mutates upsert -> delete -> meta write.
    A crash after the first write (here: injected in the stale-delete
    step of a v1->v2 rebuild, i.e. between chunk upsert and meta write)
    used to leave v2/mixed contents served under the intact v1 stamp —
    the version check then actively lies. An interrupted build must
    leave the index loudly UNQUERYABLE: a query under the old version
    raises, and so does one under the new."""
    chunks_v1, records = fixture_corpus()
    chunks_v2, _ = _v2_corpus()
    client = fresh_client()
    model = RecordingEmbeddingModel()
    build(client, chunks_v1, records, model=model)

    crashing = _CrashingClient(client, fail_on={"delete"})
    with pytest.raises(ConnectionError):
        build(crashing, chunks_v2, records, model=model, corpus_version="fixture-corpus-v2")

    for expected in ("fixture-corpus-v1", "fixture-corpus-v2"):
        with pytest.raises(IndexingError):
            hybrid_query(
                client,
                COLLECTION,
                "invented aurelian probe query",
                embedding_model=model,
                expected_corpus_version=expected,
            )


def test_crash_during_chunk_upsert_leaves_index_unqueryable() -> None:
    """Finding #157, the earlier window: a crash during the chunk upsert
    itself (marker written, contents part-mutated) must equally refuse
    queries under the previous stamp."""
    chunks_v1, records = fixture_corpus()
    chunks_v2, _ = _v2_corpus()
    client = fresh_client()
    model = RecordingEmbeddingModel()
    build(client, chunks_v1, records, model=model)

    class _CrashOnChunkUpsert(_CrashingClient):
        def __getattr__(self, name):
            if name == "upsert":
                inner_upsert = self._inner.upsert

                def _upsert(*args, **kwargs):
                    collection = kwargs.get("collection_name", args[0] if args else None)
                    if collection == COLLECTION:
                        raise ConnectionError("injected crash in the chunk upsert")
                    return inner_upsert(*args, **kwargs)

                return _upsert
            return super().__getattr__(name)

    crashing = _CrashOnChunkUpsert(client, fail_on=set())
    with pytest.raises(ConnectionError):
        build(crashing, chunks_v2, records, model=model, corpus_version="fixture-corpus-v2")

    with pytest.raises(IndexingError):
        hybrid_query(
            client,
            COLLECTION,
            "invented aurelian probe query",
            embedding_model=model,
            expected_corpus_version="fixture-corpus-v1",
        )


def test_interrupted_build_is_detectable_and_rebuildable() -> None:
    """Finding #157, recovery: the interrupted state is detectable (the
    version read refuses too) and a plain re-run of build_index over the
    same input completes the rebuild — after which queries under the new
    version work and the contents are exactly the v2 corpus."""
    from rag.indexing import get_index_corpus_version

    chunks_v1, records = fixture_corpus()
    chunks_v2, _ = _v2_corpus()
    client = fresh_client()
    model = RecordingEmbeddingModel()
    build(client, chunks_v1, records, model=model)

    crashing = _CrashingClient(client, fail_on={"delete"})
    with pytest.raises(ConnectionError):
        build(crashing, chunks_v2, records, model=model, corpus_version="fixture-corpus-v2")

    with pytest.raises(IndexingError):
        get_index_corpus_version(client, COLLECTION)

    build(client, chunks_v2, records, model=model, corpus_version="fixture-corpus-v2")
    assert get_index_corpus_version(client, COLLECTION) == "fixture-corpus-v2"
    assert indexed_chunk_ids(client, COLLECTION) == {c.chunk_id for c in chunks_v2}
    hits = hybrid_query(
        client,
        COLLECTION,
        "invented aurelian probe query",
        embedding_model=model,
        expected_corpus_version="fixture-corpus-v2",
    )
    assert hits
