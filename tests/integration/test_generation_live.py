"""Live generation checks (issue #12) — `live` tier, per release / on
demand only, never per PR (IMPLEMENTATION.md §3; the CI integration job
selects `-m "integration and not live"`).

Two checks that only a real key can prove:

1. **Prompt-cache smoke** (the orchestrator note on #12): Haiku 4.5's
   minimum cacheable prefix is 4096 tokens and caching fails SILENTLY
   below it — the builder-level arithmetic in the unit tier is a
   guard, but only `usage.cache_read_input_tokens > 0` on the second of
   two identical-prefix calls proves the §9 cost assumption fires.
2. **`test_live_cited_answer_end_to_end`** (issue #12 acceptance):
   index -> retrieve -> generate against the live API streams a cited
   answer whose citations resolve to supplied blocks.

Skip-marked awaiting a key (the #162 visible-gap convention): without
ANTHROPIC_API_KEY these SKIP loudly rather than vanish, so the gap stays
visible in every live run's output.
"""

from __future__ import annotations

import os

import pytest

from rag.generation import (
    GenerationConfig,
    GroundedAnswer,
    generate_grounded_answer,
)
from rag.provider import AnthropicAdapter
from rag.retrieval import RetrievedPassages
from tests._generation_fixtures import CORPUS_VINTAGE, make_retrieved
from tests._retrieval_fixtures import HashReranker, config, decision, indexed_corpus, run_retrieve

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="live generation checks need ANTHROPIC_API_KEY — "
        "skip-marked awaiting a key (the #162 visible-gap convention)",
    ),
]


def test_prompt_cache_smoke_second_identical_prefix_call_reads_cache():
    """The 4096-floor smoke contract: two generate calls sharing the
    static prefix but asking DIFFERENT questions; the second must report
    `cache_read_input_tokens > 0`. Zero here means the prefix is under
    the floor or a silent invalidator leaked volatile bytes into it —
    either way the §9 'less with prompt caching' assumption is dead and
    this test is the alarm."""
    adapter = AnthropicAdapter()
    cfg = GenerationConfig()
    retrieved = make_retrieved(3)

    first = generate_grounded_answer(
        adapter,
        retrieved,
        "How much has the invented basin warmed?",
        config=cfg,
        corpus_vintage=CORPUS_VINTAGE,
    )
    assert isinstance(first, GroundedAnswer)
    assert first.usage is not None

    second = generate_grounded_answer(
        adapter,
        retrieved,
        "What is the assessed range for committed basin warming?",
        config=cfg,
        corpus_vintage=CORPUS_VINTAGE,
    )
    assert isinstance(second, GroundedAnswer)
    assert second.usage is not None
    assert second.usage.get("cache_read_input_tokens", 0) > 0, (
        "second identical-prefix call read nothing from cache: the static prefix "
        "is under Haiku 4.5's 4096-token floor or a volatile byte leaked into it"
    )


def test_live_cited_answer_end_to_end():
    """Issue #12 acceptance: one real query through the indexed (fixture)
    corpus — in-memory Qdrant index, hybrid retrieval, rerank — streams a
    cited answer from the live model whose citations resolve to the
    supplied blocks. Runs per release and on demand, never per PR."""
    client, model, _chunks = indexed_corpus()
    retrieved = run_retrieve(
        client,
        decision("How much have surface temperatures risen across the Aurelian Basin?"),
        model=model,
        reranker=HashReranker(),
        cfg=config(),
    )
    assert isinstance(retrieved, RetrievedPassages)

    answer = generate_grounded_answer(
        AnthropicAdapter(),
        retrieved,
        "How much have surface temperatures risen across the Aurelian Basin?",
        config=GenerationConfig(),
        corpus_vintage=CORPUS_VINTAGE,
    )
    assert isinstance(answer, GroundedAnswer)
    assert answer.text.strip()
    assert answer.cited_passages, "live answer carried no resolvable citations"
    supplied_ids = {p.chunk_id for p in retrieved.passages}
    for cited in answer.cited_passages:
        assert cited.chunk_id in supplied_ids
    assert CORPUS_VINTAGE in answer.footer
