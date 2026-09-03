"""Shared builders for the issue-57 semantic-cache red suites.

Everything is synthetic and LOCAL (zero network, zero provider calls,
never a model weight): the embedder seam is faked two ways —

- :class:`VectorEmbedder` programs exact vectors per question so the
  threshold geometry (0.96 hits, 0.94 misses, orthogonal entries) is
  pinned precisely;
- ``tests._indexing_fixtures.HashEmbeddingModel`` (word-overlap cosine)
  drives the realistic adversarial near-miss pairs, where similarity
  emerges from the text itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from rag.indexing import Embedding
from service.semantic_cache import SemanticCache

#: A fixed aware-UTC "now" (matches tests/_service_fixtures.T0).
CACHE_T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)

CACHE_CORPUS_VERSION = "corpus-2026-08-01"


def normalise(question: str) -> str:
    """The cache's pinned key normalisation: collapsed whitespace."""
    return " ".join(question.split())


class VectorEmbedder:
    """Programmable fake embedder: normalised text -> exact dense vector.

    An unprogrammed text raises ``KeyError`` loudly — a lookup for a
    question the test did not anticipate is a test bug, never a silent
    zero-vector.
    """

    model_id = "vector-fake-embedder-v1"
    dense_dim = 4

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = {normalise(text): tuple(vector) for text, vector in vectors.items()}
        self.encoded: list[str] = []

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        results = []
        for text in texts:
            self.encoded.append(text)
            results.append(Embedding(dense=self.vectors[normalise(text)], sparse={}))
        return results


def store_kwargs(
    question: str = "Why are scientists calling this an emergency?",
    **overrides: Any,
) -> dict[str, Any]:
    """A complete, valid ``SemanticCache.store`` argument set."""
    values: dict[str, Any] = dict(
        question=question,
        answer_text="The basin has very likely warmed by one point nine degrees.",
        footer="Verify against the sources below. Answers reflect sources as of 2026-08-01.",
        citations=(
            {
                "type": "char_location",
                "cited_text": "very likely warmed",
                "document_index": 0,
                "document_title": "Synthetic Basin Assessment",
                "chunk_id": "syn-gen-doc::c0000",
                "clears_threshold": True,
                "degraded_fallback": False,
                "needs_hand_review": False,
            },
        ),
        badges=(),
        sources=(
            {
                "doc_id": "syn-gen-doc",
                "chunk_id": "syn-gen-doc::c0000",
                "title": "Synthetic Basin Assessment",
                "attribution_text": "Synthetic Basin Assessment (2026)",
                "canonical_url": "https://example.test/basin",
                "source_type": "evidence",
                "source_tier": "A",
                "permitted_context": "open",
                "excerpt": "The basin has very likely warmed.",
                "excerpt_truncated": False,
            },
        ),
        generated_on="2026-08-20",
        source_exchange_id="src-exchange-0001",
    )
    values.update(overrides)
    return values


def make_cache(
    vectors: Mapping[str, Sequence[float]] | None = None,
    *,
    embedder: Any | None = None,
    corpus_version: str = CACHE_CORPUS_VERSION,
    threshold: float | None = None,
    max_entries: int | None = None,
    clock: Any | None = None,
    entries: Sequence[Any] = (),
) -> SemanticCache:
    """A SemanticCache over a fake LOCAL embedder and a frozen clock."""
    from tests._service_fixtures import FrozenClock

    kwargs: dict[str, Any] = dict(
        embedding_model=embedder if embedder is not None else VectorEmbedder(vectors or {}),
        corpus_version=corpus_version,
        clock=clock or FrozenClock(CACHE_T0),
        entries=entries,
    )
    if threshold is not None:
        kwargs["threshold"] = threshold
    if max_entries is not None:
        kwargs["max_entries"] = max_entries
    return SemanticCache(**kwargs)
