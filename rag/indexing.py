"""Embedding + hybrid indexing (issue #9) — contract stubs, RED phase.

DESIGN §3.2 / ADR-005 / ADR-007: chunks are embedded locally with bge-m3,
which natively emits BOTH a dense vector and a LEARNED SPARSE (lexical
term-weight) vector per text — the sparse channel is bge-m3's learned
representation, **never classical BM25**. Both vectors are stored in one
Qdrant collection under named vectors, and hybrid queries are fused
**server-side** with reciprocal-rank fusion (RRF) into a top-40 that
feeds the #11 reranker.

Contract points the issue-9 red suite pins (each stub's docstring
carries its own detail):

- **EmbeddingModel seam.** All embedding flows through the
  :class:`EmbeddingModel` protocol. The real model
  (:class:`Bgem3EmbeddingModel`) is ONE implementation; every unit-tier
  test injects a deterministic fake instead. Importing this module must
  never import torch/transformers/FlagEmbedding — the heavy stack loads
  lazily inside ``Bgem3EmbeddingModel`` only (IMPLEMENTATION.md §1/§3).
- **Store seam.** Functions take a ``qdrant_client.QdrantClient``; the
  unit tier passes an embedded/in-memory client
  (``QdrantClient(":memory:")`` — the same interface as the Docker
  server, verified against qdrant-client 1.19 local mode), the
  integration tier the composed server (v1.19.0; pin the client and
  server together, compose finding #34).
- **Incremental by content hash** over the *embedded text*
  (``ChunkRecord.embedding_text`` — header INCLUDED), not merely by
  chunk id: a context-header change re-embeds even though the
  content+provenance ``chunk_id`` is unchanged.
- **Id uniqueness is a hard precondition**: duplicate incoming chunk
  ids refuse the whole build loudly (:class:`DuplicateChunkIdError`)
  before any write — never a silent last-writer-wins upsert. (Review
  #7 found real collisions in the chunker; the indexer must not paper
  over them while that fix lands in parallel.)
- **Hand-review enforcement (#143)**: chunks from a document whose
  ingest record carries ``needs_hand_review=True`` refuse the build
  (:class:`UnreviewedDegradedChunksError`); the
  ``parse_backend``/``degraded_fallback``/``needs_hand_review`` flags
  are carried on every payload so they stay queryable after the fact.
- **Index versioning**: the build records the corpus version; queries
  state the corpus version they expect and refuse on mismatch
  (:class:`IndexVersionMismatchError`) instead of silently answering
  from a stale index.
- **Voices hook point**: hybrid queries accept payload filtering by
  ``source_type`` (include/exclude). This module pins the *capability*;
  the voices-filter *policy* (which queries exclude what) is #11's,
  in ``rag/filters.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ingestion.pipeline import ChunkRecord, DocumentIngestRecord

__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "DEFAULT_TOP_K",
    "BGE_M3_MODEL_ID",
    "BGE_M3_DENSE_DIM",
    "IndexingError",
    "DuplicateChunkIdError",
    "UnreviewedDegradedChunksError",
    "IndexVersionMismatchError",
    "Embedding",
    "EmbeddingModel",
    "Bgem3EmbeddingModel",
    "IndexBuildReport",
    "IndexedPoint",
    "RetrievedChunk",
    "build_index",
    "hybrid_query",
    "indexed_chunk_ids",
    "get_chunk_point",
    "get_index_corpus_version",
]

#: Named-vector keys of the collection schema (ADR-007). One collection,
#: two named vectors per point: the bge-m3 dense embedding and the bge-m3
#: learned sparse (lexical weight) embedding. Not BM25.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

#: DESIGN §3.2: the RRF-fused hybrid result set is a top-40; the #11
#: cross-encoder reranker cuts it to the top-8 fed to generation.
DEFAULT_TOP_K = 40

#: ADR-005: the pinned local embedding model and its dense dimensionality.
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_DENSE_DIM = 1024


class IndexingError(RuntimeError):
    """Base class for loud indexing refusals (never warnings, never silent)."""


class DuplicateChunkIdError(IndexingError):
    """Two incoming chunks share a ``chunk_id``.

    Chunk-id uniqueness is the precondition the whole incremental
    mechanism rests on (the id is the upsert key); a collision would
    silently drop one chunk's text from the index. The build refuses
    before writing anything, and the message names at least one
    colliding id so the chunker defect is diagnosable.
    """


class UnreviewedDegradedChunksError(IndexingError):
    """Chunks from a degraded-parse document reached the indexer unreviewed.

    Amended DESIGN §2.4 (issue #143): a PyMuPDF-parsed document must be
    hand-reviewed *before indexing*. A build whose input contains chunks
    of a document flagged ``needs_hand_review=True`` refuses, naming the
    document id(s) — this module is the enforcement point #143 requires.
    """


class IndexVersionMismatchError(IndexingError):
    """The collection's recorded corpus version differs from the caller's.

    Issue #9 scope: index versioning tied to corpus version. Retrieval
    against an index built from a different corpus version refuses
    loudly (naming both versions) instead of silently serving stale or
    mixed-vintage evidence.
    """


@dataclass(frozen=True)
class Embedding:
    """One text's paired bge-m3-shaped representation.

    ``dense``: the dense vector, length == the producing model's
    ``dense_dim``. ``sparse``: the LEARNED SPARSE representation — a
    mapping of vocabulary token index -> positive weight (bge-m3's
    lexical weights; never a BM25 score).
    """

    dense: tuple[float, ...]
    sparse: Mapping[int, float]


class EmbeddingModel(Protocol):
    """The embedding seam (IMPLEMENTATION.md §1: ``Embedder`` protocol).

    Implementations: :class:`Bgem3EmbeddingModel` (real weights,
    integration/production only) and the unit tier's deterministic
    hash-based fake (``tests/_indexing_fixtures.HashEmbeddingModel`` —
    never downloads weights). ``encode`` returns exactly one
    :class:`Embedding` per input text, in input order; every dense
    vector has length ``dense_dim``.

    ``model_id`` identifies the producing model; an index records it so
    a query embedded under a different model is detectable (embeddings
    from different models share no space).
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dense_dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[Embedding]: ...


class Bgem3EmbeddingModel:
    """The real local bge-m3 embedding model (ADR-005). CPU, pinned revision.

    Contract (pinned by the single real-model integration smoke,
    ``test_bge_m3_smoke_dense_dim_and_sparse_format``):

    - ``model_id`` == :data:`BGE_M3_MODEL_ID`; ``dense_dim`` ==
      :data:`BGE_M3_DENSE_DIM` (1024);
    - ``encode`` returns, per text, a dense vector of exactly 1024
      floats and a NON-EMPTY sparse mapping of int token index ->
      strictly positive float weight (the learned lexical weights);
    - the heavy stack (torch / FlagEmbedding) is imported lazily inside
      this class only — importing :mod:`rag.indexing` stays cheap and
      weight-free (unit-tier rule, IMPLEMENTATION.md §3);
    - construction fails with a clear error when the model weights are
      not available locally; it never silently downloads multi-GB
      weights inside a test run.
    """

    def __init__(self, model_id: str = BGE_M3_MODEL_ID) -> None:
        raise NotImplementedError("issue #9: implemented in the green phase")

    @property
    def model_id(self) -> str:
        raise NotImplementedError("issue #9: implemented in the green phase")

    @property
    def dense_dim(self) -> int:
        raise NotImplementedError("issue #9: implemented in the green phase")

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        raise NotImplementedError("issue #9: implemented in the green phase")


@dataclass(frozen=True)
class IndexBuildReport:
    """What one :func:`build_index` run did — the idempotency evidence.

    ``embedded_count``: texts actually encoded this run (0 on a
    no-change re-run — the ``test_full_reindex_idempotent`` acceptance
    criterion). ``skipped_unchanged_count``: incoming chunks skipped
    because their stored content hash matched. ``deleted_count``:
    stale points removed because their chunk id left the corpus.
    ``indexed_chunk_count``: chunks in the collection after the run.
    ``wall_clock_seconds``: the build duration the acceptance criteria
    require recorded in the build log.
    """

    corpus_version: str
    embedding_model_id: str
    embedded_count: int
    skipped_unchanged_count: int
    deleted_count: int
    indexed_chunk_count: int
    wall_clock_seconds: float


@dataclass(frozen=True)
class IndexedPoint:
    """One chunk's stored state, read back by :func:`get_chunk_point`."""

    chunk_id: str
    payload: Mapping[str, Any]
    dense: tuple[float, ...]
    sparse: Mapping[int, float]


@dataclass(frozen=True)
class RetrievedChunk:
    """One fused hybrid hit: the chunk id, the RRF fused score, and the
    chunk's full stored payload (metadata intact, DESIGN §3.2)."""

    chunk_id: str
    score: float
    payload: Mapping[str, Any]


def build_index(
    client: Any,
    collection_name: str,
    chunks: Sequence[ChunkRecord],
    document_records: Mapping[str, DocumentIngestRecord],
    *,
    embedding_model: EmbeddingModel,
    corpus_version: str,
) -> IndexBuildReport:
    """Build or incrementally update the hybrid index over ``chunks``.

    ``client`` is a ``qdrant_client.QdrantClient`` — embedded
    (``":memory:"``/path, the unit tier and the compose-free mode) or
    the Docker server (integration/production); behaviour is identical
    across both (ADR-007's one-interface property).
    ``document_records`` is the per-document provenance from the ingest
    run (``IngestResult.documents``): every chunk's ``doc_id`` must have
    a record, and the record's ``parse_backend`` /
    ``degraded_fallback`` / ``needs_hand_review`` land on the chunk's
    payload.

    Contract (the red suite pins each point):

    - **Refusals, before any write:**
      duplicate incoming chunk ids -> :class:`DuplicateChunkIdError`;
      any chunk whose document record has ``needs_hand_review=True`` ->
      :class:`UnreviewedDegradedChunksError` naming the document (#143);
      a chunk whose ``doc_id`` has no record -> :class:`IndexingError`
      (provenance is never optional). A refused build leaves the
      collection exactly as it found it.
    - **Collection schema:** named dense vector :data:`DENSE_VECTOR_NAME`
      (size == ``embedding_model.dense_dim``, cosine distance) plus
      named sparse vector :data:`SPARSE_VECTOR_NAME` — the learned
      sparse channel, never BM25.
    - **Payload = the full §2.4 chunk metadata**, per point: ``chunk_id``,
      ``doc_id``, ``section_path``, ``context_header``, ``body``,
      ``token_count``, ``confidence_markers``, ``block_types``,
      ``consensus_position``, ``source_type``, ``citation_metadata``
      (licence, attribution_text, canonical_url, permitted_context,
      title), plus ``parse_backend``, ``degraded_fallback`` and
      ``needs_hand_review`` from the document record — all queryable
      via payload filters.
    - **What gets embedded** is ``ChunkRecord.embedding_text`` (context
      header + body — the text the #7 cap covers), never the bare body.
    - **Incremental by content hash:** a chunk whose embedded text is
      unchanged since the stored point is NOT re-encoded (the embedding
      model sees zero texts for it); a changed chunk is re-encoded and
      upserted; a chunk id no longer in the corpus is deleted. The hash
      covers the full embedding text, so a header-only change (e.g. a
      retitled document) re-embeds even though ``chunk_id`` — a
      content+provenance hash of the *body* — is unchanged.
    - **Versioning:** the collection records ``corpus_version`` and
      ``embedding_model.model_id``; re-building the same corpus under a
      new version updates the recorded version. (Qdrant point ids are
      UUIDs/ints, not arbitrary strings — the point id is derived
      deterministically from ``chunk_id``, and ``chunk_id`` lives in
      the payload; all read paths in this module speak chunk ids.)
    - **Deterministic:** building the same chunks twice into two fresh
      collections yields identical stored state (ids, payloads, both
      vectors).
    """
    raise NotImplementedError("issue #9: implemented in the green phase")


def hybrid_query(
    client: Any,
    collection_name: str,
    query_text: str,
    *,
    embedding_model: EmbeddingModel,
    expected_corpus_version: str,
    top_k: int = DEFAULT_TOP_K,
    include_source_types: Iterable[str] | None = None,
    exclude_source_types: Iterable[str] = (),
) -> list[RetrievedChunk]:
    """Hybrid dense + learned-sparse retrieval, fused with RRF (DESIGN §3.2).

    Embeds ``query_text`` once through ``embedding_model``, runs BOTH
    channels against the collection, and returns the reciprocal-rank
    -fused top ``top_k`` (default: the §3.2 top-40) as
    :class:`RetrievedChunk` records, best-fused first. Fusion is
    performed by Qdrant's Query API (``FusionQuery``/RRF over dense and
    sparse prefetches) — server-side, never re-implemented in
    application code (ADR-007's deciding feature).

    - Every hit carries its full stored payload — the §2.4 metadata
      round-trips retrieval intact (section path, attribution,
      ``source_type``, ``consensus_position``, degraded flags, …).
    - **Version refusal:** if the collection's recorded corpus version
      differs from ``expected_corpus_version``, raises
      :class:`IndexVersionMismatchError` naming both versions — before
      any search work.
    - **Source-type hook (#11):** ``include_source_types`` /
      ``exclude_source_types`` filter hits by the ``source_type``
      payload field as part of the query (applied to both channel
      prefetches, so filtered-out chunks never occupy fused ranks).
      This is the structural hook the voices filter builds on; policy
      stays out of this module.
    - A query against a collection built under a *different embedding
      model id* refuses (:class:`IndexingError`): vectors from
      different models share no space, so the search would be garbage.
    """
    raise NotImplementedError("issue #9: implemented in the green phase")


def indexed_chunk_ids(client: Any, collection_name: str) -> frozenset[str]:
    """All chunk ids currently stored in the collection.

    The read-side contract the incremental tests assert deletion and
    upsert behaviour through (chunk ids, not Qdrant point ids).
    """
    raise NotImplementedError("issue #9: implemented in the green phase")


def get_chunk_point(client: Any, collection_name: str, chunk_id: str) -> IndexedPoint:
    """Read one chunk's stored payload and both named vectors back.

    Raises :class:`IndexingError` when ``chunk_id`` is not indexed.
    """
    raise NotImplementedError("issue #9: implemented in the green phase")


def get_index_corpus_version(client: Any, collection_name: str) -> str:
    """The corpus version recorded when the collection was last built.

    Raises :class:`IndexingError` for a collection this module never
    built (no recorded version — an index of unknown vintage is never
    silently treated as current).
    """
    raise NotImplementedError("issue #9: implemented in the green phase")
