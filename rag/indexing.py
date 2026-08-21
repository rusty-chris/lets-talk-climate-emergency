"""Embedding + hybrid indexing (issue #9): local bge-m3 dense + learned
sparse embedding, incremental-by-content-hash Qdrant indexing, and
server-side RRF hybrid retrieval.

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

import hashlib
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import models

from ingestion.pipeline import ChunkRecord, DocumentIngestRecord

__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "DEFAULT_TOP_K",
    "BGE_M3_MODEL_ID",
    "BGE_M3_DENSE_DIM",
    "BGE_M3_REVISION",
    "SOURCE_TYPES",
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

#: ADR-005's rationale is "open weights (reproducible forever — pin the
#: revision)", so the revision IS pinned (finding #163): the full commit
#: hash of the Hugging Face hub revision the corpus is embedded under,
#: verified against both the hub and the CI-cached snapshot on
#: 2026-08-21. Under an unpinned load, a fresh machine could fetch
#: different weights under the same recorded model id — undetectable by
#: the model-mismatch refusal, silently invalidating published eval
#: numbers. Bump deliberately, together with a full reindex.
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

#: The CLOSED ``source_type`` vocabulary (review finding #158). DESIGN
#: §2.5/§3.2 define exactly two source layers: the RAG evidence corpus
#: (``evidence``) and the first-party voices layer (``voices``). The #11
#: voices exclusion is an exact, case-sensitive blocklist over this
#: payload field, so any value outside this set — miscased, misspelt,
#: null — would silently escape it; the indexer refuses such chunks at
#: build time and the query filter helper refuses such filter values.
#: Extend this set deliberately (with the matching filter policy), never
#: by letting arbitrary manifest strings through.
SOURCE_TYPES = frozenset({"evidence", "voices"})

# ---------------------------------------------------------------------------
# Internal storage conventions (not part of the public contract, but kept
# together here since every function below relies on them).
# ---------------------------------------------------------------------------

#: Fixed namespace for deriving Qdrant point ids from chunk ids (UUID5) so
#: point ids are stable across runs/processes without ever storing an
#: arbitrary string as a point id — Qdrant point ids are UUIDs/ints only.
#: The chunk id itself always lives in the payload (``chunk_id``), so every
#: read path in this module speaks chunk ids, never the derived point id.
_POINT_ID_NAMESPACE = uuid.UUID("2f6a2f3e-9c2f-4a1a-9b7a-2b6a2b7a2b7a")

#: The single sentinel point id inside a collection's companion metadata
#: collection (see ``_meta_collection_name``) that carries the recorded
#: corpus version + embedding model id (index versioning, TDD plan 3).
_META_POINT_ID = str(uuid.uuid5(_POINT_ID_NAMESPACE, "__rag_indexing_meta__"))

#: The metadata collection's own (unused for search) vector name/size —
#: Qdrant collections require at least one vector config; this one never
#: participates in retrieval.
_META_VECTOR_NAME = "meta"


def _point_id(chunk_id: str) -> str:
    """Derive a Qdrant point id deterministically from a chunk id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _content_hash(embedding_text: str) -> str:
    """The incremental-rebuild content hash, over the FULL embedded text
    (context header included) — see :func:`build_index`."""
    return hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()


#: Reserved suffix of the companion metadata collections (finding #167):
#: a CHUNK collection must never be named `<x>__meta`, or it collides
#: with x's metadata store — build_index refuses such names up front.
_RESERVED_META_SUFFIX = "__meta"


def _meta_collection_name(collection_name: str) -> str:
    """The companion collection carrying a single metadata point.

    Qdrant has no generic collection-level key/value metadata API, so the
    recorded corpus version + embedding model id (index versioning) live
    as the payload of one point in a small dedicated collection instead
    of a sentinel point mixed into the chunk collection — this keeps
    every chunk-collection read path (scroll/search/count) chunk-only,
    with nothing to filter out.
    """
    return f"{collection_name}__meta"


def _read_meta_payload(client: Any, collection_name: str) -> Mapping[str, Any] | None:
    """The raw recorded metadata payload, or ``None`` when none exists.

    The build path reads through this (it must see whatever is recorded,
    including a mismatched model id it is about to refuse on); the query
    path layers its refusals on top in :func:`_read_index_meta`.
    """
    meta_name = _meta_collection_name(collection_name)
    if not client.collection_exists(meta_name):
        return None
    points = client.retrieve(meta_name, ids=[_META_POINT_ID], with_payload=True)
    if not points:
        return None
    return points[0].payload


def _read_index_meta(client: Any, collection_name: str) -> Mapping[str, Any]:
    """Read back the recorded metadata payload (``corpus_version``,
    ``embedding_model_id``, ``dense_dim``, ``embedding_model_revision``).

    Raises :class:`IndexingError` when the collection carries no
    recorded metadata — either it was never built by this module, or it
    is a stale/foreign collection of unknown vintage — or when a build
    is in progress / was interrupted (finding #157).
    """
    payload = _read_meta_payload(client, collection_name)
    if payload is None:
        raise IndexingError(
            f"collection {collection_name!r} has no recorded index metadata "
            "(no corpus_version/embedding_model_id) — it was never built by "
            "rag.indexing.build_index, so its vintage is unknown and it is "
            "never treated as current"
        )
    if payload.get("building"):
        raise IndexingError(
            f"collection {collection_name!r} has a build in progress or was "
            "left by an INTERRUPTED build (the build-in-progress marker was "
            "written but never finalized, finding #157) — its contents are "
            "of mixed vintage and must not be served; re-run build_index "
            "over the intended corpus to finish the rebuild"
        )
    return payload


def _write_index_meta(
    client: Any,
    collection_name: str,
    *,
    corpus_version: str,
    embedding_model_id: str,
    dense_dim: int,
    embedding_model_revision: str | None,
    building: bool = False,
) -> None:
    """Record/replace the collection's corpus version + embedding model
    identity (id, dense_dim and weights revision — the triple the
    incremental skip is only valid under, findings #156/#163).

    ``building=True`` writes the build-in-progress marker (finding
    #157): the point upsert replaces the whole payload, so the final
    ``building=False`` write at the end of a successful build clears
    the marker atomically with recording the real vintage.
    """
    meta_name = _meta_collection_name(collection_name)
    if not client.collection_exists(meta_name):
        client.create_collection(
            collection_name=meta_name,
            vectors_config={
                _META_VECTOR_NAME: models.VectorParams(size=1, distance=models.Distance.COSINE)
            },
        )
    payload: dict[str, Any] = {
        "corpus_version": corpus_version,
        "embedding_model_id": embedding_model_id,
        "dense_dim": dense_dim,
    }
    if embedding_model_revision is not None:
        payload["embedding_model_revision"] = embedding_model_revision
    if building:
        payload["building"] = True
    client.upsert(
        collection_name=meta_name,
        points=[
            models.PointStruct(
                id=_META_POINT_ID,
                vector={_META_VECTOR_NAME: [0.0]},
                payload=payload,
            )
        ],
    )


def _require_known_source_types(values: Sequence[Any], *, argument: str) -> None:
    """Refuse filter values outside the closed vocabulary (finding #158).

    The filter is exact and case-sensitive on the server, so an unknown
    or miscased value here would silently match nothing — a caller bug
    that must never pass as a working filter. No normalisation is ever
    applied: the vocabulary is closed, not fuzzy.
    """
    unknown = [v for v in values if v not in SOURCE_TYPES]
    if unknown:
        raise IndexingError(
            f"{argument} contains unknown source_type value(s) "
            f"{unknown!r} — the closed vocabulary is "
            f"{sorted(SOURCE_TYPES)} (exact, case-sensitive; finding #158), "
            "and an unknown value would silently filter nothing"
        )


def _source_type_filter(
    include_source_types: Iterable[str] | None, exclude_source_types: Iterable[str]
) -> models.Filter | None:
    """Build the payload filter applied to EACH channel's Prefetch.

    Empirically verified against qdrant-client 1.19 local mode: a
    top-level ``query_filter`` passed to ``query_points`` is ignored
    when the top-level ``query`` is a ``FusionQuery`` — the filter must
    live on every ``Prefetch`` instead, so excluded chunks never occupy
    a fused rank in the first place.
    """
    must = []
    must_not = []
    if include_source_types is not None:
        include_list = list(include_source_types)
        _require_known_source_types(include_list, argument="include_source_types")
        must.append(
            models.FieldCondition(key="source_type", match=models.MatchAny(any=include_list))
        )
    exclude_list = list(exclude_source_types)
    _require_known_source_types(exclude_list, argument="exclude_source_types")
    if exclude_list:
        must_not.append(
            models.FieldCondition(key="source_type", match=models.MatchAny(any=exclude_list))
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _payload_chunk_id(record: Any, collection_name: str) -> str:
    """A scrolled point's ``chunk_id``, or a named refusal (finding #167).

    Every point this module writes carries ``chunk_id`` in its payload;
    a point without one means the collection holds foreign or corrupted
    data (e.g. another index's metadata sentinel) — diagnose it loudly
    instead of dying on a bare ``KeyError`` far from the cause.
    """
    chunk_id = (record.payload or {}).get("chunk_id")
    if chunk_id is None:
        raise IndexingError(
            f"collection {collection_name!r} contains a point (id "
            f"{record.id!r}) with no chunk_id payload — it holds foreign or "
            "corrupted data this module never wrote, so the build/read "
            "refuses rather than guess"
        )
    return chunk_id


def _scroll_all(client: Any, collection_name: str, *, with_payload: Any = True) -> list[Any]:
    """Scroll an entire collection's points (paginated), returning all of them."""
    records: list[Any] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name,
            limit=256,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return records


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

    def __init__(self, model_id: str = BGE_M3_MODEL_ID, revision: str = BGE_M3_REVISION) -> None:
        snapshot = _bge_m3_snapshot_dir(model_id, revision)
        if snapshot is None:
            raise IndexingError(
                f"{model_id} weights at the PINNED revision {revision} are "
                "not cached locally under the Hugging Face hub cache "
                "(HF_HUB_CACHE / HF_HOME) — fetch them first (e.g. "
                f"`huggingface-cli download {model_id} --revision {revision}`); "
                "Bgem3EmbeddingModel never triggers an implicit multi-GB "
                "download on construction, and never loads an unpinned "
                "snapshot (finding #163: different weights under the same "
                "model id are undetectable downstream)."
            )

        from FlagEmbedding import BGEM3FlagModel

        # Belt-and-suspenders: force offline mode for the load itself too, so
        # a partially-cached snapshot fails loudly instead of silently
        # fetching the missing pieces over the network. The model is loaded
        # from the pinned revision's snapshot DIRECTORY (FlagEmbedding does
        # not forward a revision kwarg to from_pretrained, so the path is
        # how the pin is enforced — finding #163).
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = BGEM3FlagModel(
                str(snapshot),
                use_fp16=False,  # ADR-005: CPU-only.
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        self._model_id = model_id
        self._revision = revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        """The pinned hub revision (full commit hash) the weights were
        loaded from; recorded in the index meta so cross-revision
        query-vs-index mismatches refuse (finding #163)."""
        return self._revision

    @property
    def dense_dim(self) -> int:
        return BGE_M3_DENSE_DIM

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        texts = list(texts)
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        embeddings = []
        for i in range(len(texts)):
            dense = tuple(float(v) for v in dense_vecs[i])
            sparse = {
                int(token_id): float(weight) for token_id, weight in lexical_weights[i].items()
            }
            embeddings.append(Embedding(dense=dense, sparse=sparse))
        return embeddings


def _hf_hub_cache_dir() -> Path:
    """The Hugging Face hub cache directory, honouring HF_HUB_CACHE/HF_HOME
    (same convention as ``tests/_weights.py``, duplicated here because
    production code never imports from ``tests/``)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _bge_m3_snapshot_dir(model_id: str, revision: str) -> Path | None:
    """The cached snapshot directory of ``model_id`` at exactly the
    pinned ``revision``, or ``None`` when it is not cached. Any OTHER
    cached revision does not count (finding #163)."""
    snapshot = (
        _hf_hub_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    )
    if snapshot.is_dir() and any(snapshot.iterdir()):
        return snapshot
    return None


def _chunk_payload(chunk: ChunkRecord, record: DocumentIngestRecord) -> dict[str, Any]:
    """The full stored payload for one chunk — the §2.4 metadata plus the
    #143 document-record flags plus the incremental content hash. Derived
    in exactly one place so the build can diff stored payloads against
    freshly derived ones (finding #155)."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_path": list(chunk.section_path),
        "context_header": chunk.context_header,
        "body": chunk.body,
        "token_count": chunk.token_count,
        "confidence_markers": list(chunk.confidence_markers),
        "block_types": list(chunk.block_types),
        "consensus_position": chunk.consensus_position,
        "source_type": chunk.source_type,
        "citation_metadata": dict(chunk.citation_metadata),
        "parse_backend": record.parse_backend,
        "degraded_fallback": record.degraded_fallback,
        "needs_hand_review": record.needs_hand_review,
        "content_hash": _content_hash(chunk.embedding_text),
    }


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
    ``payload_refreshed_count``: chunks whose embedding text was
    unchanged (no re-encoding) but whose stored payload was stale and
    was rewritten — metadata-only corrections (finding #155).
    """

    corpus_version: str
    embedding_model_id: str
    embedded_count: int
    skipped_unchanged_count: int
    deleted_count: int
    indexed_chunk_count: int
    wall_clock_seconds: float
    payload_refreshed_count: int = 0


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
    chunks: Iterable[ChunkRecord],
    document_records: Mapping[str, DocumentIngestRecord],
    *,
    embedding_model: EmbeddingModel,
    corpus_version: str,
    reindex: bool = False,
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
      content+provenance hash of the *body* — is unchanged. A hash
      match reuses the stored VECTORS only (finding #155): the freshly
      derived payload is still diffed against the stored one and
      rewritten when it differs, so metadata-only corrections
      (``source_type`` reclassification, licence fixes, #143 flags)
      always reach the index without re-encoding anything.
    - **Versioning:** the collection records ``corpus_version`` and
      ``embedding_model.model_id``; re-building the same corpus under a
      new version updates the recorded version. (Qdrant point ids are
      UUIDs/ints, not arbitrary strings — the point id is derived
      deterministically from ``chunk_id``, and ``chunk_id`` lives in
      the payload; all read paths in this module speak chunk ids.)
    - **Deterministic:** building the same chunks twice into two fresh
      collections yields identical stored state (ids, payloads, both
      vectors).
    - **Model identity (finding #156):** the incremental skip is only
      valid while ``embedding_model.model_id`` and ``dense_dim`` match
      the recorded build — old vectors share no space with a new
      model's. A rebuild under a different model REFUSES loudly unless
      ``reindex=True``, which drops the collection and re-embeds
      everything from scratch under the new model. Refusal (not silent
      full re-embed) is deliberate: an accidental wrong-model run would
      otherwise silently spend a full corpus re-embed and swap the
      index's vector space with no operator decision anywhere, and a
      changed ``dense_dim`` needs a collection re-create the
      incremental path cannot do. ``reindex=True`` always rebuilds from
      scratch, model change or not.
    """
    start = time.perf_counter()

    # Materialise FIRST (review finding #159): the refusal checks below make
    # several passes over `chunks`, and Python does not enforce the type
    # hint — a generator argument would exhaust on the first pass, so the
    # needs-hand-review / missing-record checks would see an empty stream
    # and silently pass, indexing gated content before dying late on an
    # unrelated TypeError. One defensive list() closes the hole.
    chunks = list(chunks)

    # --- Refusals, before any write (read-only over the arguments) --------
    if collection_name.endswith(_RESERVED_META_SUFFIX):
        raise IndexingError(
            f"collection name {collection_name!r} ends with the reserved "
            f"suffix {_RESERVED_META_SUFFIX!r} — names ending in it are "
            "companion metadata collections (finding #167), so building a "
            "chunk index there would corrupt another index's metadata"
        )

    by_chunk_id: dict[str, ChunkRecord] = {}
    for chunk in chunks:
        collision = by_chunk_id.get(chunk.chunk_id)
        if collision is not None:
            raise DuplicateChunkIdError(
                f"duplicate incoming chunk id {chunk.chunk_id!r} (documents "
                f"{collision.doc_id!r} and {chunk.doc_id!r}) — the indexer "
                "refuses the whole build rather than risk a silent "
                "last-writer-wins upsert dropping one chunk's evidence"
            )
        by_chunk_id[chunk.chunk_id] = chunk

    missing_doc_ids = sorted({c.doc_id for c in chunks if c.doc_id not in document_records})
    if missing_doc_ids:
        raise IndexingError(
            "chunk(s) reference document id(s) with no ingest record — "
            "provenance is never optional, so the #143 flags cannot be "
            f"attached and the build refuses: {', '.join(missing_doc_ids)}"
        )

    unreviewed_doc_ids = sorted(
        {c.doc_id for c in chunks if document_records[c.doc_id].needs_hand_review}
    )
    if unreviewed_doc_ids:
        raise UnreviewedDegradedChunksError(
            "document(s) flagged needs_hand_review (a loud PyMuPDF-degraded "
            "parse, issue #143) must be hand-reviewed before indexing: "
            f"{', '.join(unreviewed_doc_ids)}"
        )

    # Finding #174: refuse NON-STRING source_type values by name FIRST — a
    # list value (e.g. a YAML manifest's `source_type: [voices, evidence]`)
    # is unhashable, so the vocabulary set-comprehension below would die
    # with a bare TypeError instead of this named refusal. Arrays are also
    # the dangerous case at query time: Qdrant match conditions accept a
    # payload array when ANY element matches, so a stored array containing
    # "evidence" would bypass the #11 include-list on science routes.
    non_string_source_types = sorted(
        {(c.doc_id, repr(c.source_type)) for c in chunks if not isinstance(c.source_type, str)}
    )
    if non_string_source_types:
        described = ", ".join(
            f"{doc_id!r} carries source_type {value}" for doc_id, value in non_string_source_types
        )
        raise IndexingError(
            "chunk(s) carry a non-string source_type — the field is a single "
            "scalar string from the closed vocabulary, never a list or other "
            "type (Qdrant matches an array payload when ANY element matches, "
            "so an array would bypass the #11 include-list filter — review "
            f"finding #174): {described}. Fix the manifest entry rather than "
            "indexing the chunk"
        )

    unknown_source_types = sorted(
        {(c.doc_id, c.source_type) for c in chunks if c.source_type not in SOURCE_TYPES},
        key=repr,
    )
    if unknown_source_types:
        described = ", ".join(
            f"{doc_id!r} carries source_type {value!r}" for doc_id, value in unknown_source_types
        )
        raise IndexingError(
            "chunk(s) carry a source_type outside the closed vocabulary "
            f"{sorted(SOURCE_TYPES)} (exact, case-sensitive — finding #158): "
            f"{described}. The #11 voices exclusion filters on this field "
            "exactly, so an unknown value would silently escape it; fix the "
            "manifest entry rather than indexing the chunk"
        )

    # --- Model-identity gate over the incremental path (finding #156) -----
    model_revision = getattr(embedding_model, "revision", None)
    stored_meta = _read_meta_payload(client, collection_name)
    if stored_meta is not None and not reindex:
        stored_model_id = stored_meta.get("embedding_model_id")
        stored_dense_dim = stored_meta.get("dense_dim")
        stored_revision = stored_meta.get("embedding_model_revision")
        if stored_model_id != embedding_model.model_id:
            raise IndexingError(
                f"collection {collection_name!r} was built with embedding "
                f"model {stored_model_id!r}; this build's model is "
                f"{embedding_model.model_id!r}. The incremental skip would "
                "reuse every stored vector and re-stamp the meta — vectors "
                "from different models share no space, so that index would "
                "be convincing garbage. Pass reindex=True to drop the "
                "collection and re-embed everything under the new model, or "
                "build into a fresh collection"
            )
        if stored_dense_dim is not None and stored_dense_dim != embedding_model.dense_dim:
            raise IndexingError(
                f"collection {collection_name!r} was built at dense_dim "
                f"{stored_dense_dim}; this build's model {embedding_model.model_id!r} "
                f"emits dense_dim {embedding_model.dense_dim}. A dimension "
                "change needs a collection re-create the incremental path "
                "cannot do — pass reindex=True to rebuild from scratch"
            )
        if stored_revision != model_revision:
            raise IndexingError(
                f"collection {collection_name!r} was built under weights "
                f"revision {stored_revision!r}; this build's model "
                f"{embedding_model.model_id!r} carries revision "
                f"{model_revision!r} (finding #163) — two revisions of the "
                "same model id still share no vector space, so pass "
                "reindex=True to re-embed everything under the new revision"
            )
    if reindex:
        # The sanctioned destructive path: drop chunks AND meta, rebuild
        # from scratch (used for model changes; valid for any full rebuild).
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        meta_name = _meta_collection_name(collection_name)
        if client.collection_exists(meta_name):
            client.delete_collection(meta_name)

    # --- Invalidate before mutating (finding #157) -------------------------
    # The FIRST write of every run is the build-in-progress marker: any
    # crash between here and the final meta write leaves the index loudly
    # unqueryable (_read_index_meta refuses on the marker) instead of
    # confidently mislabeled with the previous corpus version over new or
    # mixed contents. The final meta write below replaces the marker
    # payload wholesale, finalizing the build atomically.
    _write_index_meta(
        client,
        collection_name,
        corpus_version=corpus_version,
        embedding_model_id=embedding_model.model_id,
        dense_dim=embedding_model.dense_dim,
        embedding_model_revision=model_revision,
        building=True,
    )

    # --- Ensure the collection schema exists -------------------------------
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedding_model.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )

    # --- Diff against stored state (content hash over embedding_text) -----
    existing_records = _scroll_all(client, collection_name, with_payload=True)
    existing_payloads = {
        _payload_chunk_id(record, collection_name): record.payload for record in existing_records
    }
    incoming_ids = set(by_chunk_id)
    stale_ids = [chunk_id for chunk_id in existing_payloads if chunk_id not in incoming_ids]
    new_payloads = {
        chunk_id: _chunk_payload(chunk, document_records[chunk.doc_id])
        for chunk_id, chunk in by_chunk_id.items()
    }
    to_embed = [
        chunk
        for chunk_id, chunk in by_chunk_id.items()
        if existing_payloads.get(chunk_id, {}).get("content_hash")
        != new_payloads[chunk_id]["content_hash"]
    ]
    # Finding #155: a content-hash match licenses reusing the stored
    # VECTORS only, never a stale payload — the payload carries fields
    # (source_type, citation_metadata, the #143 flags) that are not part
    # of the embedded text, and metadata-only corrections must still
    # reach the index or the voices/licensing invariants silently break.
    embedded_ids = {chunk.chunk_id for chunk in to_embed}
    to_refresh_payload = [
        chunk_id
        for chunk_id, chunk in by_chunk_id.items()
        if chunk_id not in embedded_ids
        and chunk_id in existing_payloads
        and existing_payloads[chunk_id] != new_payloads[chunk_id]
    ]

    # --- Embed only what changed --------------------------------------------
    embeddings = (
        embedding_model.encode([chunk.embedding_text for chunk in to_embed]) if to_embed else []
    )

    # --- Upsert changed/new points -----------------------------------------
    if to_embed:
        points = []
        for chunk, embedding in zip(to_embed, embeddings, strict=True):
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: list(embedding.dense),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=list(embedding.sparse.keys()),
                            values=list(embedding.sparse.values()),
                        ),
                    },
                    payload=new_payloads[chunk.chunk_id],
                )
            )
        client.upsert(collection_name=collection_name, points=points)

    # --- Refresh stale payloads of unchanged-text chunks (finding #155) ----
    for chunk_id in to_refresh_payload:
        client.overwrite_payload(
            collection_name=collection_name,
            payload=new_payloads[chunk_id],
            points=models.PointIdsList(points=[_point_id(chunk_id)]),
        )

    # --- Delete points whose chunk id left the corpus -----------------------
    if stale_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[_point_id(chunk_id) for chunk_id in stale_ids]
            ),
        )

    # --- Record versioning ---------------------------------------------------
    _write_index_meta(
        client,
        collection_name,
        corpus_version=corpus_version,
        embedding_model_id=embedding_model.model_id,
        dense_dim=embedding_model.dense_dim,
        embedding_model_revision=model_revision,
    )

    return IndexBuildReport(
        corpus_version=corpus_version,
        embedding_model_id=embedding_model.model_id,
        embedded_count=len(to_embed),
        skipped_unchanged_count=len(chunks) - len(to_embed),
        deleted_count=len(stale_ids),
        indexed_chunk_count=len(chunks),
        wall_clock_seconds=time.perf_counter() - start,
        payload_refreshed_count=len(to_refresh_payload),
    )


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
    meta = _read_index_meta(client, collection_name)
    stored_corpus_version = meta["corpus_version"]
    stored_model_id = meta["embedding_model_id"]
    if stored_corpus_version != expected_corpus_version:
        raise IndexVersionMismatchError(
            f"index {collection_name!r} was built from corpus_version "
            f"{stored_corpus_version!r}; this query expected "
            f"{expected_corpus_version!r} — refusing to serve a "
            "stale/mismatched-vintage index"
        )
    if stored_model_id != embedding_model.model_id:
        raise IndexingError(
            f"index {collection_name!r} was built with embedding model "
            f"{stored_model_id!r}; the query embedding model is "
            f"{embedding_model.model_id!r} — vectors from different models "
            "share no space, so the search would be garbage"
        )
    stored_revision = meta.get("embedding_model_revision")
    query_revision = getattr(embedding_model, "revision", None)
    if stored_revision != query_revision:
        raise IndexingError(
            f"index {collection_name!r} was built under weights revision "
            f"{stored_revision!r}; the query model carries revision "
            f"{query_revision!r} (finding #163) — two revisions of the same "
            "model id still share no vector space, so the search would be "
            "garbage; rebuild the index under the current pinned revision"
        )

    [query_embedding] = embedding_model.encode([query_text])
    channel_filter = _source_type_filter(include_source_types, exclude_source_types)

    prefetch = [
        models.Prefetch(
            query=list(query_embedding.dense),
            using=DENSE_VECTOR_NAME,
            filter=channel_filter,
            limit=top_k,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(query_embedding.sparse.keys()),
                values=list(query_embedding.sparse.values()),
            ),
            using=SPARSE_VECTOR_NAME,
            filter=channel_filter,
            limit=top_k,
        ),
    ]

    response = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(chunk_id=point.payload["chunk_id"], score=point.score, payload=point.payload)
        for point in response.points
    ]


def indexed_chunk_ids(client: Any, collection_name: str) -> frozenset[str]:
    """All chunk ids currently stored in the collection.

    The read-side contract the incremental tests assert deletion and
    upsert behaviour through (chunk ids, not Qdrant point ids).
    """
    records = _scroll_all(client, collection_name, with_payload=["chunk_id"])
    return frozenset(_payload_chunk_id(record, collection_name) for record in records)


def get_chunk_point(client: Any, collection_name: str, chunk_id: str) -> IndexedPoint:
    """Read one chunk's stored payload and both named vectors back.

    Raises :class:`IndexingError` when ``chunk_id`` is not indexed.
    """
    points = client.retrieve(
        collection_name,
        ids=[_point_id(chunk_id)],
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        raise IndexingError(f"chunk {chunk_id!r} is not indexed in collection {collection_name!r}")
    point = points[0]
    vector = point.vector
    dense = tuple(float(v) for v in vector[DENSE_VECTOR_NAME])
    sparse_vector = vector[SPARSE_VECTOR_NAME]
    sparse = {
        int(index): float(value)
        for index, value in zip(sparse_vector.indices, sparse_vector.values, strict=True)
    }
    return IndexedPoint(chunk_id=chunk_id, payload=point.payload, dense=dense, sparse=sparse)


def get_index_corpus_version(client: Any, collection_name: str) -> str:
    """The corpus version recorded when the collection was last built.

    Raises :class:`IndexingError` for a collection this module never
    built (no recorded version — an index of unknown vintage is never
    silently treated as current).
    """
    return _read_index_meta(client, collection_name)["corpus_version"]
