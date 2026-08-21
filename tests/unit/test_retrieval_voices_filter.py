"""The structural voices filter (issue #11) — RED.

DESIGN §3.2 / §2.5: voices/evidence separation is a STRUCTURAL
invariant, enforced in code inside the store query — never a prompt
behaviour, never post-hoc trimming. Spike-03 proved the §3.4 corollary
live: citations are all-or-none across document blocks, so a voices
chunk cannot even ride along with citations disabled — it must never
reach the generation document set at all.

Include-list-first (review finding #158): `source_type` is an
unvalidated free string upstream, and Qdrant's match conditions are
exact and case-sensitive — a `must_not "voices"` blocklist fails OPEN on
`"Voices"`, null, or a missing key. The contract pinned here therefore
puts an INCLUDE list of known-good source types on every store query:
science routes serve only `EVIDENCE_SOURCE_TYPES`; voices routes serve
evidence AND voices (bias, never exclusion of evidence); anything
outside the closed vocabulary fails CLOSED on every route — independent
of the parallel data-side hardening.

The eval-facing guarantee (§6.2 voices separation): on every non-voices
route the returned top-8 provably contains zero `source_type: voices`
chunks.

Unit tier: in-memory Qdrant + deterministic fakes. The fake reranker is
always programmed to score voices/rogue text HIGHEST, so any leak past
the structural filter wins the ranking and fails the test loudly — the
filter is never rescued by bad chunks merely ranking low.
"""

from __future__ import annotations

from qdrant_client import models

from rag.query import ScopeClass
from rag.retrieval import (
    EVIDENCE_SOURCE_TYPES,
    KNOWN_SOURCE_TYPES,
    RetrievedPassages,
    permitted_source_types,
)
from tests._indexing_fixtures import fresh_client
from tests._retrieval_fixtures import (
    BASIN_WARMING_MARKER,
    VOICES_MARKER,
    VOICES_PULL_QUERY,
    QueryRecordingClient,
    TableReranker,
    call_prefetches,
    config,
    decision,
    implant_rogue_chunk,
    indexed_corpus,
    recorded_include_lists,
    run_retrieve,
)

#: The two non-voices scopes that reach retrieval (DESIGN §3.1 routing);
#: the structural filter must hold on EVERY one of them.
NON_VOICES_RETRIEVAL_SCOPES = (ScopeClass.IN_SCOPE, ScopeClass.ADVERSARIAL_IN_SCOPE)

#: A marker for implanted rogue chunks, in the pull query's vocabulary.
ROGUE_MARKER = "lanternwood briefing organisers"


def _voices_boosting_reranker() -> TableReranker:
    """Scores voices/rogue text top — a leaked chunk would rank #1."""
    return TableReranker([(VOICES_MARKER, 0.99), ("lanternwood", 0.98)], default=0.1)


def test_source_type_policy_is_include_list_first() -> None:
    """The pure policy core (IMPLEMENTATION.md §1), include-list-first
    (finding #158): science routes may serve ONLY the known-good
    evidence source types; voices routes add voices but never drop
    evidence; the vocabulary is closed on both."""
    assert EVIDENCE_SOURCE_TYPES == ("evidence",)
    assert "voices" not in EVIDENCE_SOURCE_TYPES
    assert set(KNOWN_SOURCE_TYPES) == {"evidence", "voices"}

    for scope in NON_VOICES_RETRIEVAL_SCOPES:
        assert permitted_source_types(decision("q", scope=scope)) == EVIDENCE_SOURCE_TYPES
    voices_decision = decision("who is calling for the briefing", scope=ScopeClass.VOICES)
    assert voices_decision.voices_bias, "routing self-check: voices scope sets voices_bias"
    permitted = permitted_source_types(voices_decision)
    assert set(permitted) == set(KNOWN_SOURCE_TYPES)
    assert set(EVIDENCE_SOURCE_TYPES) <= set(permitted), "voices routes never drop evidence"


def test_voices_chunks_removed_for_non_voices_query() -> None:
    """Acceptance criterion (issue #11): a non-voices query in the voices
    document's own vocabulary — with the reranker rigged to score voices
    text top — still yields zero voices chunks in the result."""
    client, model, chunks = indexed_corpus()
    voices_ids = {c.chunk_id for c in chunks if c.source_type == "voices"}
    assert voices_ids, "fixture self-check: the corpus carries voices chunks"

    for scope in NON_VOICES_RETRIEVAL_SCOPES:
        result = run_retrieve(
            client,
            decision(VOICES_PULL_QUERY, scope=scope),
            model=model,
            reranker=_voices_boosting_reranker(),
            cfg=config(),
        )
        assert isinstance(result, RetrievedPassages)
        assert {p.chunk_id for p in result.passages} & voices_ids == set()
        assert all(p.payload["source_type"] != "voices" for p in result.passages)


def test_voices_chunks_retained_for_voices_query() -> None:
    """The control: the same pull query on the voices route retrieves the
    voices chunks — the filter is routing policy, not a blanket ban."""
    client, model, chunks = indexed_corpus()
    voices_ids = {c.chunk_id for c in chunks if c.source_type == "voices"}

    result = run_retrieve(
        client,
        decision(VOICES_PULL_QUERY, scope=ScopeClass.VOICES),
        model=model,
        reranker=_voices_boosting_reranker(),
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    retained = {p.chunk_id for p in result.passages} & voices_ids
    assert retained, "a voices-route query must be able to retrieve voices chunks"
    assert result.passages[0].chunk_id in voices_ids, (
        "the reranker scored voices text top, so a voices chunk must lead the result"
    )


def test_voices_route_biases_but_does_not_exclude_evidence() -> None:
    """§2.5/#10 review: voices_bias means bias TOWARD the voices source —
    evidence chunks stay retrievable on a voices route. A voices query
    that also matches evidence vocabulary returns both kinds."""
    client, model, _chunks = indexed_corpus()
    reranker = TableReranker([(VOICES_MARKER, 0.99), (BASIN_WARMING_MARKER, 0.98)], default=0.1)

    result = run_retrieve(
        client,
        decision(
            "lanternwood briefing organisers and eastern escarpment warming stations",
            scope=ScopeClass.VOICES,
        ),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    assert isinstance(result, RetrievedPassages)
    source_types = {p.payload["source_type"] for p in result.passages}
    assert "voices" in source_types
    assert source_types & set(EVIDENCE_SOURCE_TYPES), (
        "voices routes must not exclude evidence chunks — bias, never exclusion"
    )


def test_voices_filtering_is_in_the_store_query_not_post_hoc() -> None:
    """The filter lives IN THE QUERY (the #9 Prefetch-level capability):

    - every channel Prefetch of a science-route store query carries an
      include restriction on `source_type` of exactly the evidence
      include list — never no restriction, never a blocklist standing
      alone;
    - the reranker never sees a voices body (filtered before scoring,
      not trimmed after);
    - filtered-out chunks never occupy fused ranks, so the reranker
      still receives the full top-40 of evidence candidates.
    """
    spy = QueryRecordingClient(fresh_client())
    _client, model, chunks = indexed_corpus(client=spy)
    voices_bodies = [c.body for c in chunks if c.source_type == "voices"]
    reranker = _voices_boosting_reranker()

    run_retrieve(
        spy,
        decision(VOICES_PULL_QUERY),
        model=model,
        reranker=reranker,
        cfg=config(),
    )

    include_lists = recorded_include_lists(spy)
    assert include_lists, "the science route must query the store"
    for include_list in include_lists:
        assert include_list == frozenset(EVIDENCE_SOURCE_TYPES), (
            f"every channel Prefetch must include-restrict source_type to exactly "
            f"{EVIDENCE_SOURCE_TYPES}; got {include_list} — None means fail-open (#158)"
        )
    assert len(reranker.calls) == 1
    _query_text, scored = reranker.calls[0]
    for text in scored:
        assert all(body not in text for body in voices_bodies), (
            "the reranker must never be handed voices text on a non-voices route"
        )
    assert len(scored) == 40, (
        "filtered-out chunks must not eat fused ranks — the include list belongs in "
        "the channel prefetches, so the evidence top-40 arrives intact"
    )


def test_voices_route_include_list_covers_evidence_and_voices() -> None:
    """Voices routes stay fail-closed too: every channel Prefetch carries
    an include list drawn from the closed vocabulary (no unrestricted
    prefetch anywhere), and across the route's store queries both
    evidence and voices are served — bias by inclusion, never an
    evidence exclusion and never a fail-open query."""
    spy = QueryRecordingClient(fresh_client())
    _client, model, _chunks = indexed_corpus(client=spy)

    run_retrieve(
        spy,
        decision(VOICES_PULL_QUERY, scope=ScopeClass.VOICES),
        model=model,
        reranker=_voices_boosting_reranker(),
        cfg=config(),
    )

    include_lists = recorded_include_lists(spy)
    assert include_lists, "the voices route must still query the store"
    for include_list in include_lists:
        assert include_list is not None, "no prefetch may serve unrestricted source types (#158)"
        assert include_list <= frozenset(KNOWN_SOURCE_TYPES)
    union: set[str] = set().union(*include_lists)
    assert union == set(KNOWN_SOURCE_TYPES), (
        "across the voices route's store queries both evidence and voices must be servable"
    )


def test_unknown_source_type_fails_closed_on_every_route() -> None:
    """Finding #158 made contract: chunks whose `source_type` is
    mis-cased, unknown, null or missing — implanted directly into the
    index, bypassing build_index, exactly as a blocklist bypass would —
    are served on NO route, even with the reranker rigged to score them
    top. The include list is what makes this fail CLOSED; no data-side
    fix is assumed."""
    client, model, _chunks = indexed_corpus()
    rogue_ids = {
        implant_rogue_chunk(
            client,
            model,
            chunk_id="rogue-cased",
            body=f"Mis-cased {ROGUE_MARKER} chunk reading aloud in the market square.",
            source_type_value="Voices",
        ),
        implant_rogue_chunk(
            client,
            model,
            chunk_id="rogue-unknown",
            body=f"Unknown-type {ROGUE_MARKER} chunk reading aloud in the market square.",
            source_type_value="campaign-material",
        ),
        implant_rogue_chunk(
            client,
            model,
            chunk_id="rogue-null",
            body=f"Null-type {ROGUE_MARKER} chunk reading aloud in the market square.",
            source_type_value=None,
        ),
        implant_rogue_chunk(
            client,
            model,
            chunk_id="rogue-missing",
            body=f"Missing-key {ROGUE_MARKER} chunk reading aloud in the market square.",
            include_source_type_key=False,
        ),
    }
    reranker = TableReranker([(ROGUE_MARKER, 0.999), (VOICES_MARKER, 0.99)], default=0.1)

    for scope in (*NON_VOICES_RETRIEVAL_SCOPES, ScopeClass.VOICES):
        result = run_retrieve(
            client,
            decision(VOICES_PULL_QUERY, scope=scope),
            model=model,
            reranker=reranker,
            cfg=config(),
        )
        assert isinstance(result, RetrievedPassages)
        leaked = {p.chunk_id for p in result.passages} & rogue_ids
        assert leaked == set(), (
            f"chunks outside the closed source_type vocabulary must fail closed on "
            f"every route; leaked on {scope}: {leaked}"
        )


def test_filter_applies_before_generation_document_set() -> None:
    """Acceptance criterion (issue #11): the document set handed to the
    generation call is post-filter, asserted AT THE SEAM — the
    RetrievedPassages tuple IS what #12 turns into document blocks
    (spike-03: citations are all-or-none, so there is no
    citations-disabled side door for a voices block).

    Even with the reranker rigged so any leaked voices chunk would top
    the ranking, the generation document set on every non-voices route
    is exactly `GENERATION_TOP_K` passages, every one of a known-good
    evidence source type.
    """
    client, model, chunks = indexed_corpus()
    voices_bodies = {c.body for c in chunks if c.source_type == "voices"}

    for scope in NON_VOICES_RETRIEVAL_SCOPES:
        result = run_retrieve(
            client,
            decision(VOICES_PULL_QUERY, scope=scope),
            model=model,
            reranker=_voices_boosting_reranker(),
            cfg=config(),
        )
        assert isinstance(result, RetrievedPassages)
        generation_document_set = result.passages
        assert len(generation_document_set) == 8
        for passage in generation_document_set:
            assert passage.payload["source_type"] in EVIDENCE_SOURCE_TYPES
            assert passage.payload["body"] not in voices_bodies


def test_include_list_uses_the_prefetch_capability_verbatim() -> None:
    """Guard against a bespoke re-implementation: the recorded prefetch
    filters must be the #9 `models.Filter` shape (a `must` FieldCondition
    on the `source_type` payload key), i.e. the include list rides
    `hybrid_query`'s `include_source_types` hook rather than any
    client-side filtering layer."""
    spy = QueryRecordingClient(fresh_client())
    _client, model, _chunks = indexed_corpus(client=spy)

    run_retrieve(
        spy,
        decision(VOICES_PULL_QUERY),
        model=model,
        reranker=_voices_boosting_reranker(),
        cfg=config(),
    )

    for call in spy.query_calls:
        for prefetch in call_prefetches(call):
            assert isinstance(prefetch.filter, models.Filter)
            assert prefetch.filter.must, "each channel prefetch carries the include condition"
