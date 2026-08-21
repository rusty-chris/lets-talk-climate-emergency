"""Shared doubles and builders for the issue-11 retrieval red suite.

Everything here is synthetic (the fictional Aurelian-Basin universe of
the #24/#9 fixtures). The reranker doubles implement the
`rag.retrieval.Reranker` seam:

- `HashReranker` — a deterministic pseudo-score per passage text (no
  weights, no network, no randomness). Its ordering is unrelated to the
  RRF fused order, which is exactly what the "result order follows the
  reranker, not RRF" test needs.
- `TableReranker` — programmable: passages matching a marker substring
  get that marker's score, everything else the default. The instrument
  for threshold-boundary and relevance-ordering tests.

Both record every `score` call, so tests assert the reranker was called
exactly once per query and saw exactly the (already voices-filtered)
candidate set — observed behaviour, never a trusted report field.

`QueryRecordingClient` wraps a Qdrant client and records every
`query_points` call so the structural voices filter can be asserted IN
THE QUERY (an include-list `must` condition on `source_type` on every
channel Prefetch — finding #158: include-list-first, so unknown or
malformed source_type values fail closed), not inferred from the shape
of the results.

`implant_rogue_chunk` upserts a point directly into the collection,
bypassing `build_index` — simulating index data whose `source_type` is
mis-cased, unknown, null or missing. The retrieval contract must fail
closed on such data regardless of whether the parallel data-side
hardening (#158's indexer vocabulary check) has landed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from qdrant_client import models

from rag.query import Classification, QueryDecision, ScopeClass, route_classification
from rag.retrieval import RetrievalConfig, retrieve
from tests._indexing_fixtures import (
    COLLECTION,
    FIXTURE_CORPUS_VERSION,
    HashEmbeddingModel,
    build,
    fixture_corpus,
    fresh_client,
)

#: A query in the voices doc's own distinctive invented vocabulary — the
#: strongest possible pull toward the chunks the structural filter must
#: exclude on non-voices routes.
VOICES_PULL_QUERY = "the lanternwood briefing campaign organisers reading aloud"

#: Marker substrings unique to one fixture chunk body each (see
#: tests/_indexing_fixtures.py's document builders).
BASIN_WARMING_MARKER = "eastern escarpment"
ATTRIBUTION_MARKER = "counterfactual simulations"
VOICES_MARKER = "market square"

#: Coverage lines for the honest-refusal template tests (synthetic).
COVERAGE_TOPICS = (
    "observed warming across the Aurelian Basin",
    "attribution of the basin drying trend",
    "assessed ranges for committed basin warming",
)


class HashReranker:
    """Deterministic fake reranker: score = a stable hash of the passage
    text mapped into (0, 1). Ordering is deterministic yet unrelated to
    the hybrid RRF order."""

    model_id = "hash-fake-reranker-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(passages)))
        return [self._score_one(p) for p in passages]

    @staticmethod
    def _score_one(passage: str) -> float:
        digest = hashlib.sha256(passage.encode("utf-8")).hexdigest()
        # Strictly inside (0, 1), like the real sigmoid scores.
        return (int(digest[:12], 16) + 1) / (16**12 + 2)


class TableReranker:
    """Programmable fake reranker: first matching marker substring wins.

    `table` is a sequence of (marker, score) pairs checked in order
    against each passage text; unmatched passages get `default`.
    """

    model_id = "table-fake-reranker-v1"

    def __init__(self, table: Sequence[tuple[str, float]] = (), default: float = 0.05) -> None:
        self.table = tuple(table)
        self.default = default
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(passages)))
        scores = []
        for passage in passages:
            for marker, marked_score in self.table:
                if marker in passage:
                    scores.append(marked_score)
                    break
            else:
                scores.append(self.default)
        return scores

    @property
    def scored_texts(self) -> list[str]:
        return [text for _query, passages in self.calls for text in passages]


class QueryRecordingClient:
    """Proxy around a Qdrant client recording every query_points call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.query_calls: list[dict[str, Any]] = []

    def query_points(self, *args: Any, **kwargs: Any) -> Any:
        self.query_calls.append({"args": args, "kwargs": kwargs})
        return self._inner.query_points(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def call_prefetches(call: dict[str, Any]) -> list[models.Prefetch]:
    """The Prefetch list of one recorded query_points call, however passed."""
    prefetch = call["kwargs"].get("prefetch")
    if prefetch is None:
        for arg in call["args"]:
            if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], models.Prefetch):
                prefetch = arg
                break
    assert prefetch, f"query_points call carries no Prefetch list: {call}"
    return list(prefetch)


def prefetch_include_values(prefetch: models.Prefetch) -> frozenset[str] | None:
    """The include-list this channel Prefetch restricts `source_type` to.

    Returns the set of values of a `must` FieldCondition on the
    `source_type` payload key (MatchAny or MatchValue), or None when the
    prefetch carries no such include restriction — i.e. it would serve
    chunks of ANY source_type, which is the fail-open shape the #158
    contract forbids.
    """
    if prefetch.filter is None or not prefetch.filter.must:
        return None
    conditions = prefetch.filter.must
    if not isinstance(conditions, list):
        conditions = [conditions]
    for condition in conditions:
        if not isinstance(condition, models.FieldCondition) or condition.key != "source_type":
            continue
        match = condition.match
        values = getattr(match, "any", None)
        if values is not None:
            return frozenset(values)
        value = getattr(match, "value", None)
        if value is not None:
            return frozenset({value})
    return None


def recorded_include_lists(client: QueryRecordingClient) -> list[frozenset[str] | None]:
    """One include-list (or None) per channel Prefetch of every recorded
    store query, in call order."""
    assert client.query_calls, "no query_points call was recorded"
    return [
        prefetch_include_values(prefetch)
        for call in client.query_calls
        for prefetch in call_prefetches(call)
    ]


def implant_rogue_chunk(
    client: Any,
    model: Any,
    *,
    chunk_id: str,
    body: str,
    source_type_value: Any = "voices",
    include_source_type_key: bool = True,
    collection_name: str = COLLECTION,
) -> str:
    """Upsert one point straight into the collection, bypassing build_index.

    Simulates index data with a malformed/unknown/mis-cased/null/missing
    — or array-valued (finding #174) — `source_type` (finding #158) that
    an exclusion blocklist would fail open on. Returns the implanted
    chunk_id.
    """
    import uuid

    from rag.indexing import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    [embedding] = model.encode([body])
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": f"rogue-{chunk_id}",
        "body": body,
        "section_path": ["1 Rogue"],
        "context_header": "Rogue synthetic document",
    }
    if include_source_type_key:
        payload["source_type"] = source_type_value
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"rogue::{chunk_id}")),
                vector={
                    DENSE_VECTOR_NAME: list(embedding.dense),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ),
                },
                payload=payload,
            )
        ],
    )
    return chunk_id


def decision(
    query: str,
    scope: ScopeClass = ScopeClass.IN_SCOPE,
) -> QueryDecision:
    """A QueryDecision built by the REAL #10 router over a synthetic
    classification — never a hand-rolled imitation of the routing
    contract."""
    return route_classification(Classification(scope=scope, rewritten_query=query))


def config(
    refusal_threshold: float = 0.0,
    corpus_coverage: tuple[str, ...] = COVERAGE_TOPICS,
    **overrides: Any,
) -> RetrievalConfig:
    """A RetrievalConfig with an explicit threshold.

    The 0.0 default here is a TEST convenience meaning "never refuse"
    (every real score is strictly positive); production thresholds come
    only from the eval-derived artifact (#20/#21).
    """
    return RetrievalConfig(
        refusal_threshold=refusal_threshold,
        corpus_coverage=corpus_coverage,
        **overrides,
    )


def indexed_corpus(client: Any | None = None):
    """The >40-chunk fixture corpus built into an in-memory index.

    Returns (client, embedding_model, chunks). `client` may be a
    QueryRecordingClient-wrapped instance; the index is built through it.
    """
    client = client if client is not None else fresh_client()
    model = HashEmbeddingModel()
    chunks, records = fixture_corpus()
    build(client, chunks, records, model=model)
    return client, model, chunks


def reviewed_degraded_corpus(flagged_doc_id: str = "syn-idx-basin"):
    """The fixture corpus with one document parsed by the LOUD degraded
    fallback and then hand-reviewed (#143: degraded_fallback=True,
    needs_hand_review=False — the only degraded state the indexer
    admits). Returns (client, embedding_model, chunks, flagged_doc_id).
    """
    from ingestion.pipeline import DocumentIngestRecord

    client = fresh_client()
    model = HashEmbeddingModel()
    chunks, records = fixture_corpus()
    records = dict(records)
    records[flagged_doc_id] = DocumentIngestRecord(
        doc_id=flagged_doc_id,
        parse_backend="pymupdf",
        degraded_fallback=True,
        needs_hand_review=False,
        warnings=(f"{flagged_doc_id}: PyMuPDF fallback parse — hand-reviewed and approved",),
    )
    build(client, chunks, records, model=model)
    return client, model, chunks, flagged_doc_id


def run_retrieve(client, decision_value, *, model, reranker, cfg, collection_name=COLLECTION):
    """rag.retrieval.retrieve with the fixture-index defaults filled in."""
    return retrieve(
        client,
        collection_name,
        decision_value,
        embedding_model=model,
        reranker=reranker,
        config=cfg,
        expected_corpus_version=FIXTURE_CORPUS_VERSION,
    )
