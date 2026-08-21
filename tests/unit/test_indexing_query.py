"""Hybrid RRF query mechanics + the voices-filter hook point (issue #9) — RED.

Unit tier: the embedded in-memory Qdrant client honours the exact
interface of the Docker server (fusion included), so the fused-query
mechanism is pinned here cheaply; the integration twin re-runs the
contract against the real composed server.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.indexing import DEFAULT_TOP_K, Embedding, hybrid_query
from tests._indexing_fixtures import (
    COLLECTION,
    FIXTURE_CORPUS_VERSION,
    HashEmbeddingModel,
    build,
    fixture_corpus,
    fresh_client,
)


def _query(client, model, text, **kwargs):
    return hybrid_query(
        client,
        COLLECTION,
        text,
        embedding_model=model,
        expected_corpus_version=FIXTURE_CORPUS_VERSION,
        **kwargs,
    )


def test_hybrid_query_returns_fused_top40_with_metadata_intact() -> None:
    """DESIGN §3.2: the fused result set is a top-40 (the corpus holds
    more than 40 chunks, so exactly 40 come back), best-fused first,
    every hit carrying its full stored §2.4 payload."""
    chunks, records = fixture_corpus()
    assert len(chunks) > DEFAULT_TOP_K, "fixture self-check: corpus must exceed the top-40"
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    results = _query(client, model, "invented aurelian basin warming query")

    assert len(results) == DEFAULT_TOP_K
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
    assert len({r.chunk_id for r in results}) == DEFAULT_TOP_K, "fused results must be distinct"

    by_id = {c.chunk_id: c for c in chunks}
    for hit in results:
        chunk = by_id[hit.chunk_id]
        assert hit.payload["doc_id"] == chunk.doc_id
        assert list(hit.payload["section_path"]) == list(chunk.section_path)
        assert hit.payload["body"] == chunk.body
        assert hit.payload["source_type"] == chunk.source_type
        assert hit.payload["consensus_position"] == chunk.consensus_position
        assert (
            hit.payload["citation_metadata"]["attribution_text"]
            == (chunk.citation_metadata["attribution_text"])
        )


class _ChannelModel:
    """Programmable seam double whose dense and sparse channels DISAGREE.

    Each known text gets a one-hot dense vector and a single-index
    sparse vector, chosen so one document matches the query on the dense
    channel only and another on the sparse channel only — the input that
    catches a 'hybrid' implementation quietly searching one channel.
    """

    model_id = "channel-probe-embedder"
    dense_dim = 4

    def __init__(self, table: dict[str, tuple[int, int]]) -> None:
        # text -> (dense axis, sparse index)
        self._table = table

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        out = []
        for text_value in texts:
            axis, sparse_index = self._table[text_value]
            dense = tuple(1.0 if i == axis else 0.0 for i in range(self.dense_dim))
            out.append(Embedding(dense=dense, sparse={sparse_index: 1.0}))
        return out


def test_hybrid_query_fuses_both_channels() -> None:
    """RRF fusion genuinely consults BOTH channels: a dense-only match
    and a sparse-only match must both out-rank a chunk matching neither.
    A dense-only (or sparse-only) implementation fails this — the other
    channel's winner sinks to noise rank."""
    chunks, records = fixture_corpus()
    dense_only = chunks[0]  # shares the query's dense axis, disjoint sparse index
    sparse_only = chunks[1]  # shares the query's sparse index, orthogonal dense axis
    neither = chunks[2]
    corpus = [dense_only, sparse_only, neither]
    query_text = "channel probe query"

    model = _ChannelModel(
        {
            dense_only.embedding_text: (0, 101),
            sparse_only.embedding_text: (1, 202),
            neither.embedding_text: (2, 303),
            query_text: (0, 202),
        }
    )
    client = fresh_client()
    build(client, corpus, records, model=model)

    results = _query(client, model, query_text, top_k=3)

    top_two = {r.chunk_id for r in results[:2]}
    assert top_two == {dense_only.chunk_id, sparse_only.chunk_id}, (
        f"fusion must surface the dense-channel and sparse-channel winners together; "
        f"got top two {top_two}"
    )


def test_known_query_hits_attribution_family_in_memory() -> None:
    """The retrieval smoke check at unit scale (the named acceptance test
    runs at integration tier over the same corpus): a query in the
    attribution family's distinctive invented vocabulary retrieves that
    document's chunks first."""
    chunks, records = fixture_corpus()
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    results = _query(client, model, "what explains the attribution of aurelian basin drying")

    assert results[0].payload["doc_id"] == "syn-idx-attribution"
    top_five_docs = [r.payload["doc_id"] for r in results[:5]]
    assert top_five_docs.count("syn-idx-attribution") >= 2, (
        f"the attribution family should dominate the head of the ranking, got {top_five_docs}"
    )


def test_source_type_exclude_filter_hook() -> None:
    """The voices hook point (#11 consumes this; the CAPABILITY is pinned
    here, the policy there): excluding source_type 'voices' removes every
    voices chunk from the fused results — structurally, in the store
    query, not by post-hoc trimming (excluded chunks must not eat fused
    ranks, so the full top-40 is still delivered)."""
    chunks, records = fixture_corpus()
    voices_ids = {c.chunk_id for c in chunks if c.source_type == "voices"}
    assert voices_ids, "fixture self-check: the corpus carries voices chunks"
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    # A query in the voices doc's own vocabulary — the strongest pull
    # towards the chunks the filter must exclude.
    results = _query(
        client,
        model,
        "the lanternwood briefing campaign organisers reading aloud",
        exclude_source_types=("voices",),
    )

    assert {r.chunk_id for r in results} & voices_ids == set()
    assert all(r.payload["source_type"] != "voices" for r in results)
    assert len(results) == DEFAULT_TOP_K, (
        "excluded chunks must not occupy fused ranks — the filter belongs inside the "
        "channel prefetches, not after fusion"
    )


def test_source_type_include_filter_hook() -> None:
    """The inverse hook (a voices-classified query biases TO voices,
    DESIGN §3.1): restricting to source_type 'voices' returns exactly
    the voices chunks."""
    chunks, records = fixture_corpus()
    voices_ids = {c.chunk_id for c in chunks if c.source_type == "voices"}
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, chunks, records, model=model)

    results = _query(
        client,
        model,
        "the lanternwood briefing campaign organisers reading aloud",
        include_source_types=("voices",),
    )

    assert {r.chunk_id for r in results} == voices_ids
    assert all(r.payload["source_type"] == "voices" for r in results)
