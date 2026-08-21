"""Retrieval flow: hybrid top-40 -> rerank -> top-8, seams and metadata
(issue #11) — RED.

Unit tier: the in-memory Qdrant client + the deterministic reranker
fakes behind the `Reranker` seam. Nothing here loads model weights or
makes an LLM call; the single real-reranker check lives at integration
tier (`tests/integration/test_reranker_smoke.py`).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rag import retrieval
from rag.indexing import hybrid_query
from rag.query import ScopeClass
from rag.retrieval import RetrievalError, RetrievedPassages
from tests._indexing_fixtures import COLLECTION, FIXTURE_CORPUS_VERSION
from tests._retrieval_fixtures import (
    ATTRIBUTION_MARKER,
    HashReranker,
    QueryRecordingClient,
    TableReranker,
    config,
    decision,
    indexed_corpus,
    reviewed_degraded_corpus,
    run_retrieve,
)


def test_importing_retrieval_module_stays_weight_free() -> None:
    """`import rag.retrieval` must not import the heavy model stack.

    The seam rule (IMPLEMENTATION.md §1/§3): transformers/torch live
    lazily inside BgeRerankerV2M3 only; every unit test (and every
    service path that never reranks) stays weight-free. Checked in a
    fresh interpreter so already-imported modules cannot fool it.
    """
    probe = (
        "import sys\n"
        "import rag.retrieval\n"
        "heavy = {'torch', 'transformers', 'FlagEmbedding', 'docling', 'sentence_transformers'}\n"
        "loaded = sorted(heavy & set(sys.modules))\n"
        "assert not loaded, f'importing rag.retrieval loaded the heavy stack: {loaded}'\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_retrieval_module_never_touches_the_provider_adapter() -> None:
    """The §3.5 honest-refusal guarantee made structural: `rag.retrieval`
    itself never imports `rag.provider`, so nothing in this module can
    reach a ProviderAdapter — the refusal path is template-only, with no
    LLM call to spend. (A transitive import via `rag.query`'s type
    surface is tolerated; a direct import here is the defect.)"""
    import ast
    from pathlib import Path

    import rag.retrieval as retrieval_module

    tree = ast.parse(Path(retrieval_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert not any(n == "rag.provider" or n.startswith("rag.provider.") for n in names), (
                f"rag.retrieval must never import rag.provider: {names}"
            )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module != "rag.provider" and not module.startswith("rag.provider."), (
                "rag.retrieval must never import from rag.provider"
            )


def test_contract_constants_pin_design_values() -> None:
    """The design numbers are contract, not convention: §3.2's 40-in/8-out
    funnel, the ADR-006 model pin, and the ~100 ms CPU rerank budget."""
    assert retrieval.RERANK_CANDIDATE_K == 40
    assert retrieval.GENERATION_TOP_K == 8
    assert retrieval.BGE_RERANKER_MODEL_ID == "BAAI/bge-reranker-v2-m3"
    assert retrieval.RERANK_LATENCY_BUDGET_SECONDS == pytest.approx(0.1)
    assert retrieval.VOICES_SOURCE_TYPE == "voices"


def test_hybrid_top40_feeds_exactly_one_rerank_call() -> None:
    """The full fused top-40 goes to the reranker in ONE `score` call.

    One call is load-bearing for the latency budget (40 pairs batched,
    not 40 model invocations); 40 candidates is the §3.2 funnel. Each
    scored passage text must contain a candidate chunk's body — the
    cross-encoder reads real evidence text, not ids or headers alone.
    """
    client, model, chunks = indexed_corpus()
    non_voices_bodies = [c.body for c in chunks if c.source_type != "voices"]
    assert len(non_voices_bodies) > 40, "fixture self-check: >40 non-voices chunks required"
    reranker = HashReranker()

    result = run_retrieve(
        client,
        decision("invented aurelian basin warming query"),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    assert len(reranker.calls) == 1, "the 40 candidates must be scored in one batched call"
    _query_text, scored = reranker.calls[0]
    assert len(scored) == 40
    assert len(set(scored)) == 40, "each candidate is scored exactly once"
    for text in scored:
        assert any(body in text for body in non_voices_bodies), (
            "every scored passage text must contain a candidate chunk's body"
        )


def test_result_is_top8_ordered_by_rerank_scores() -> None:
    """The result is exactly the reranker's top-8: 8 passages, best score
    first, and the 8 scores are precisely the 8 highest the reranker
    returned — nothing dropped, nothing smuggled past the cut."""
    client, model, _chunks = indexed_corpus()
    reranker = HashReranker()

    result = run_retrieve(
        client,
        decision("invented aurelian basin warming query"),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    assert len(result.passages) == 8
    scores = [p.rerank_score for p in result.passages]
    assert scores == sorted(scores, reverse=True)
    _query_text, scored_texts = reranker.calls[0]
    all_scores = [HashReranker._score_one(text) for text in scored_texts]
    assert scores == sorted(all_scores, reverse=True)[:8]
    assert len({p.chunk_id for p in result.passages}) == 8


def test_reranker_order_governs_not_hybrid_rrf_order() -> None:
    """A chunk the RRF fusion did NOT rank first must come first when the
    reranker scores it highest — the reranked order is the product,
    the fused order only the candidate pool (ADR-006)."""
    client, model, _chunks = indexed_corpus()
    # The attribution-marker chunk sits DEEP in this query's fused order
    # (the query is in the basin doc's vocabulary), so only the reranker
    # can put it first.
    query = "eastern escarpment warming stations query"

    fused = hybrid_query(
        client,
        COLLECTION,
        query,
        embedding_model=model,
        expected_corpus_version=FIXTURE_CORPUS_VERSION,
        include_source_types=("evidence",),
    )
    fused_marker_rank = next(
        i for i, hit in enumerate(fused) if ATTRIBUTION_MARKER in hit.payload["body"]
    )
    assert fused_marker_rank != 0, (
        "fixture self-check: the marker chunk must not already lead the fused order"
    )

    reranker = TableReranker([(ATTRIBUTION_MARKER, 0.99)], default=0.05)
    result = run_retrieve(client, decision(query), model=model, reranker=reranker, cfg=config())

    assert isinstance(result, RetrievedPassages)
    assert ATTRIBUTION_MARKER in result.passages[0].payload["body"]
    assert result.passages[0].rerank_score == pytest.approx(0.99)


def test_passages_carry_full_metadata_intact() -> None:
    """DESIGN §3.2: the §2.4 payload survives the whole funnel — section
    path, body, source_type, consensus_position, attribution — so
    generation (#12), the passages panel (§3.6) and the eval harness
    (#21) never re-fetch or guess."""
    client, model, chunks = indexed_corpus()
    reranker = HashReranker()

    result = run_retrieve(
        client,
        decision("what explains the attribution of aurelian basin drying"),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    by_id = {c.chunk_id: c for c in chunks}
    for passage in result.passages:
        chunk = by_id[passage.chunk_id]
        assert passage.payload["doc_id"] == chunk.doc_id
        assert list(passage.payload["section_path"]) == list(chunk.section_path)
        assert passage.payload["body"] == chunk.body
        assert passage.payload["source_type"] == chunk.source_type
        assert passage.payload["consensus_position"] == chunk.consensus_position
        assert (
            passage.payload["citation_metadata"]["attribution_text"]
            == chunk.citation_metadata["attribution_text"]
        )


def test_tone_flag_carried_through_untouched() -> None:
    """adversarial_in_scope routes to normal retrieval with a tone flag
    (DESIGN §3.1); retrieval carries it onto the result UNTOUCHED — on
    the passages result and on the refusal result alike — and never
    invents one for plain in_scope queries."""
    client, model, _chunks = indexed_corpus()
    adversarial = decision(
        "is the aurelian warming just natural variability",
        scope=ScopeClass.ADVERSARIAL_IN_SCOPE,
    )
    assert adversarial.tone_flag, "routing self-check: adversarial decisions carry tone_flag"

    answered = run_retrieve(
        client, adversarial, model=model, reranker=TableReranker(default=0.9), cfg=config(0.5)
    )
    assert isinstance(answered, RetrievedPassages)
    assert answered.tone_flag is True

    refused = run_retrieve(
        client, adversarial, model=model, reranker=TableReranker(default=0.1), cfg=config(0.5)
    )
    assert not isinstance(refused, RetrievedPassages)
    assert refused.tone_flag is True

    plain = run_retrieve(
        client,
        decision("invented aurelian basin warming query"),
        model=model,
        reranker=TableReranker(default=0.9),
        cfg=config(0.5),
    )
    assert isinstance(plain, RetrievedPassages)
    assert plain.tone_flag is False


def test_degraded_doc_flags_surface_on_retrieved_passages() -> None:
    """#143: the parse-provenance flags ride every retrieved passage's
    payload, so a hand-reviewed degraded parse stays visible downstream
    (UI labelling, eval triage) instead of vanishing after ingest."""
    client, model, _chunks, flagged_doc_id = reviewed_degraded_corpus()
    reranker = HashReranker()

    result = run_retrieve(
        client,
        decision("invented aurelian basin warming reservoir query"),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    flagged = [p for p in result.passages if p.payload["doc_id"] == flagged_doc_id]
    assert flagged, "fixture self-check: the degraded document must be retrieved by this query"
    for passage in flagged:
        assert passage.payload["parse_backend"] == "pymupdf"
        assert passage.payload["degraded_fallback"] is True
        assert passage.payload["needs_hand_review"] is False
    for passage in result.passages:
        if passage.payload["doc_id"] != flagged_doc_id:
            assert passage.payload["degraded_fallback"] is False
            assert passage.payload["parse_backend"] == "html"


def test_retrieve_refuses_non_retrieval_routes() -> None:
    """CHART and CANNED decisions never reach retrieval; handing one in is
    a wiring bug and refuses loudly — before any store query or rerank
    work happens."""
    from tests._indexing_fixtures import HashEmbeddingModel

    spy = QueryRecordingClient(None)
    model = HashEmbeddingModel()
    reranker = HashReranker()
    chart = decision("plot basin warming", scope=ScopeClass.CHART_REQUEST)
    canned = decision("write me a sonnet", scope=ScopeClass.OUT_OF_SCOPE)

    for wrong_route in (chart, canned):
        with pytest.raises(RetrievalError):
            run_retrieve(spy, wrong_route, model=model, reranker=reranker, cfg=config())

    assert spy.query_calls == [], "a refused route must not query the store"
    assert reranker.calls == [], "a refused route must not run the reranker"


def test_fewer_candidates_than_top8_returns_them_all() -> None:
    """A small candidate pool (corpus < 8 chunks) yields all of it, still
    reranker-ordered — never padding, never an off-by-top-k crash."""
    from tests._indexing_fixtures import HashEmbeddingModel, build, fixture_corpus, fresh_client

    chunks, records = fixture_corpus()
    small = [c for c in chunks if c.source_type != "voices"][:3]
    client = fresh_client()
    model = HashEmbeddingModel()
    build(client, small, records, model=model)
    reranker = HashReranker()

    result = run_retrieve(
        client,
        decision("invented aurelian basin query"),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    assert len(result.passages) == 3
    scores = [p.rerank_score for p in result.passages]
    assert scores == sorted(scores, reverse=True)
