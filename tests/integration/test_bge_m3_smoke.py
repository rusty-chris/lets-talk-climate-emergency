"""The ONE real-model check for issue #9 — RED. Integration tier.

The review verdict scopes everything else in the #9 suite to the
deterministic fake behind the EmbeddingModel seam; this single smoke
pins what only the real weights can prove — that bge-m3's output has
the shape the whole index schema is built around (ADR-005: 1024-dim
dense + non-empty learned sparse lexical weights).

Skips on a dev machine without the cached weights; FAILS in CI when the
weights are absent (tests/_weights.py — the #32 convention: a skip must
never green a CI tier that was meant to test something).
"""

from __future__ import annotations

from rag.indexing import BGE_M3_DENSE_DIM, BGE_M3_MODEL_ID, Bgem3EmbeddingModel
from tests._weights import require_bge_m3_weights


def test_bge_m3_smoke_dense_dim_and_sparse_format() -> None:
    """Real bge-m3 over two short synthetic sentences: per text, exactly
    one embedding whose dense vector has exactly 1024 float components
    and whose sparse mapping is a non-empty {int token index: strictly
    positive float weight} — the learned lexical weights, not BM25.
    Everything downstream (collection schema, hybrid fusion) assumes
    exactly this shape."""
    require_bge_m3_weights()

    model = Bgem3EmbeddingModel()
    assert model.model_id == BGE_M3_MODEL_ID
    assert model.dense_dim == BGE_M3_DENSE_DIM

    texts = [
        "Invented probe sentence about Aurelian Basin warming trends.",
        "A second synthetic sentence with distinctive vocabulary: lanternwood.",
    ]
    embeddings = model.encode(texts)

    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert len(embedding.dense) == BGE_M3_DENSE_DIM
        assert all(isinstance(v, float) for v in embedding.dense)
        assert any(v != 0.0 for v in embedding.dense)
        assert embedding.sparse, "bge-m3 must emit non-empty learned sparse weights"
        assert all(
            isinstance(index, int) and index >= 0 and isinstance(weight, float) and weight > 0.0
            for index, weight in embedding.sparse.items()
        )
    # Different texts embed differently — a constant output would satisfy
    # every shape check above while being useless for retrieval.
    assert embeddings[0].dense != embeddings[1].dense
    assert dict(embeddings[0].sparse) != dict(embeddings[1].sparse)
