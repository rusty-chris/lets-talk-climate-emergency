"""The semantic response cache — pure module contract (issue #57, SotA rec 5).

RED suite over ``service.semantic_cache``: the LOCAL similarity lookup
(injectable embedder, cosine over dense vectors, conservative
threshold), the adversarial near-miss guarantee, corpus-version binding
(wholesale, fail-closed), the clean-exchanges-only admission gate, the
LRU bound, the #56 thumbs-down eviction seam, and the §9 90-day purge.
Zero network anywhere: both embedders are deterministic in-process
fakes.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from service.app import ANSWER_KIND_CACHED
from service.exchange_log import EXCHANGE_LOG_RETENTION_DAYS
from service.semantic_cache import (
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_ROUTE,
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SemanticCache,
    SemanticCacheEntry,
    SemanticCacheHit,
    cacheable_exchange,
)
from tests._indexing_fixtures import HashEmbeddingModel
from tests._semantic_cache_fixtures import (
    CACHE_CORPUS_VERSION,
    CACHE_T0,
    VectorEmbedder,
    make_cache,
    store_kwargs,
)
from tests._service_fixtures import FrozenClock

QUESTION = "Why are scientists calling this an emergency?"

#: Unit vectors at exact cosines to the stored entry's (1, 0, 0, 0).
ENTRY_VECTOR = (1.0, 0.0, 0.0, 0.0)


def _at_cosine(cosine: float) -> tuple[float, float, float, float]:
    """A unit vector whose cosine with ENTRY_VECTOR is exactly ``cosine``."""
    return (cosine, math.sqrt(1.0 - cosine * cosine), 0.0, 0.0)


class TestPinnedConstants:
    def test_threshold_is_the_flagged_conservative_value(self) -> None:
        # DECISION FLAGGED (#57 red notes): 0.95 cosine, the SotA
        # review's "conservative ~0.95+". Changing it is a ratified
        # decision, never a drive-by.
        assert SEMANTIC_CACHE_SIMILARITY_THRESHOLD == 0.95

    def test_entry_bound_is_the_flagged_value(self) -> None:
        # DECISION FLAGGED (#57 red notes): 500 entries, LRU.
        assert SEMANTIC_CACHE_MAX_ENTRIES == 500

    def test_route_kind_parity_with_the_service_wire(self) -> None:
        # One vocabulary: the log route, the wire answer kind and the
        # cache module's constant can never drift apart.
        assert SEMANTIC_CACHE_ROUTE == ANSWER_KIND_CACHED == "cached"


class TestCacheHitRequiresSimilarityAboveThreshold:
    """Issue #57 TDD plan item 1 (fixture embeddings, exact geometry)."""

    def _warm_cache(self, extra_vectors: dict | None = None) -> SemanticCache:
        vectors = {QUESTION: ENTRY_VECTOR}
        vectors.update(extra_vectors or {})
        cache = make_cache(vectors)
        cache.store(**store_kwargs(QUESTION))
        return cache

    def test_cache_hit_requires_similarity_above_threshold(self) -> None:
        cache = self._warm_cache(
            {
                "a question at cosine ninety six": _at_cosine(0.96),
                "a question at cosine ninety four": _at_cosine(0.94),
            }
        )
        hit = cache.lookup("a question at cosine ninety six")
        assert isinstance(hit, SemanticCacheHit)
        assert hit.similarity == pytest.approx(0.96)
        assert cache.lookup("a question at cosine ninety four") is None, (
            "0.94 < the 0.95 threshold: a near question is NOT the same "
            "question — serving it anyway is the integrity failure the "
            "conservative threshold exists to prevent"
        )

    def test_exact_repeat_hits_at_full_similarity(self) -> None:
        cache = self._warm_cache()
        hit = cache.lookup(QUESTION)
        assert isinstance(hit, SemanticCacheHit)
        assert hit.similarity == pytest.approx(1.0)

    def test_hit_replays_the_stored_content_verbatim(self) -> None:
        kwargs = store_kwargs(QUESTION)
        cache = self._warm_cache()
        hit = cache.lookup(QUESTION)
        assert hit is not None
        entry = hit.entry
        assert entry.question == QUESTION
        assert entry.answer_text == kwargs["answer_text"]
        assert entry.footer == kwargs["footer"]
        assert entry.citations == tuple(kwargs["citations"])
        assert entry.badges == tuple(kwargs["badges"])
        assert entry.sources == tuple(kwargs["sources"])
        assert entry.generated_on == kwargs["generated_on"]
        assert entry.source_exchange_id == kwargs["source_exchange_id"]

    def test_whitespace_variants_share_one_normalised_key(self) -> None:
        cache = self._warm_cache()
        hit = cache.lookup("  Why are   scientists calling this an emergency?  ")
        assert isinstance(hit, SemanticCacheHit)
        assert hit.similarity == pytest.approx(1.0)

    def test_blank_question_never_embeds_and_never_hits(self) -> None:
        embedder = VectorEmbedder({QUESTION: ENTRY_VECTOR})
        cache = make_cache(embedder=embedder)
        cache.store(**store_kwargs(QUESTION))
        encoded_after_store = list(embedder.encoded)
        assert cache.lookup("   ") is None
        assert embedder.encoded == encoded_after_store, (
            "a blank question is a guaranteed miss; embedding it is wasted "
            "work and (with a strict fake) a crash"
        )

    def test_best_entry_wins_when_several_clear_the_threshold(self) -> None:
        vectors = {
            QUESTION: ENTRY_VECTOR,
            "an orthogonal cached question": (0.0, 1.0, 0.0, 0.0),
            "a probe near the first entry": _at_cosine(0.98),
        }
        cache = make_cache(vectors)
        cache.store(**store_kwargs(QUESTION, source_exchange_id="src-first"))
        cache.store(
            **store_kwargs("an orthogonal cached question", source_exchange_id="src-second")
        )
        hit = cache.lookup("a probe near the first entry")
        assert hit is not None
        assert hit.entry.source_exchange_id == "src-first"
        assert hit.similarity == pytest.approx(0.98)

    def test_lookup_uses_only_the_injected_local_embedder(self) -> None:
        # The $0 guarantee is structural: the ONLY embedding path is the
        # injected LOCAL seam — the fake records exactly one encode of
        # the normalised question per lookup, and no other collaborator
        # exists to call.
        embedder = VectorEmbedder({QUESTION: ENTRY_VECTOR})
        cache = make_cache(embedder=embedder)
        cache.store(**store_kwargs(QUESTION))
        embedder.encoded.clear()
        cache.lookup("  Why are scientists   calling this an emergency?")
        assert embedder.encoded == [QUESTION]


class TestNearMissQuestionsMustNotHit:
    """Issue #57 TDD plan item 2: the adversarial pairs, over the
    word-overlap hash embedder (similarity emerges from the text)."""

    ADVERSARIAL_PAIRS = [
        # The issue's canonical pair: presence vs timing.
        ("Is it too late?", "When is it too late?"),
        # Negation-flavoured myth vs its plain-question sibling.
        ("Didn't warming pause?", "Did warming pause?"),
        # One swapped content word changes the answer entirely.
        ("How much has the planet warmed?", "How much has the ocean warmed?"),
    ]

    def _hash_cache(self, question: str) -> SemanticCache:
        cache = make_cache(embedder=HashEmbeddingModel())
        cache.store(**store_kwargs(question))
        return cache

    def test_near_miss_questions_must_not_hit(self) -> None:
        for cached_question, near_miss in self.ADVERSARIAL_PAIRS:
            cache = self._hash_cache(cached_question)
            assert cache.lookup(near_miss) is None, (
                f"{near_miss!r} must NOT be served {cached_question!r}'s cached "
                "answer — a similar question is not the same question"
            )

    def test_the_controls_still_hit(self) -> None:
        for cached_question, _ in self.ADVERSARIAL_PAIRS:
            cache = self._hash_cache(cached_question)
            hit = cache.lookup(f"  {cached_question}  ")
            assert isinstance(hit, SemanticCacheHit), (
                f"an exact restatement of {cached_question!r} must hit — a cache "
                "that never hits is dead code, not a safe cache"
            )


def _seed_entry(corpus_version: str) -> SemanticCacheEntry:
    kwargs = store_kwargs(QUESTION)
    return SemanticCacheEntry(
        question=kwargs["question"],
        embedding=ENTRY_VECTOR,
        answer_text=kwargs["answer_text"],
        footer=kwargs["footer"],
        citations=tuple(kwargs["citations"]),
        badges=tuple(kwargs["badges"]),
        sources=tuple(kwargs["sources"]),
        generated_on=kwargs["generated_on"],
        corpus_version=corpus_version,
        source_exchange_id=kwargs["source_exchange_id"],
        stored_at=CACHE_T0,
    )


class TestCacheInvalidatedOnCorpusVersionChange:
    """Issue #57 TDD plan item 3: stale citations are a licensing and
    honesty hazard — version binding fails CLOSED, wholesale."""

    def test_store_stamps_the_current_corpus_version(self) -> None:
        cache = make_cache({QUESTION: ENTRY_VECTOR})
        cache.store(**store_kwargs(QUESTION))
        hit = cache.lookup(QUESTION)
        assert hit is not None
        assert hit.entry.corpus_version == CACHE_CORPUS_VERSION

    def test_cache_invalidated_on_corpus_version_change(self) -> None:
        # A cache constructed for the NEW corpus release discards seeded
        # entries from the old release WHOLESALE at construction — not
        # lazily, not per-lookup, never served.
        stale = _seed_entry("corpus-2026-05-01")
        cache = make_cache(
            {QUESTION: ENTRY_VECTOR},
            corpus_version=CACHE_CORPUS_VERSION,
            entries=[stale],
        )
        assert len(cache) == 0
        assert cache.lookup(QUESTION) is None

    def test_matching_version_seed_entries_do_serve(self) -> None:
        # The seed seam is real (a future persistence layer restores
        # through it): same-release entries load and serve.
        current = _seed_entry(CACHE_CORPUS_VERSION)
        cache = make_cache(
            {QUESTION: ENTRY_VECTOR},
            corpus_version=CACHE_CORPUS_VERSION,
            entries=[current],
        )
        assert len(cache) == 1
        hit = cache.lookup(QUESTION)
        assert hit is not None
        assert hit.entry.answer_text == current.answer_text


CLEAN_VALIDATION = {
    "citation_support_rate": 1.0,
    "factual_sentence_count": 2,
    "unverified_sentence_indices": [],
    "validated": True,
    "skipped_reason": None,
    "degraded_reason": None,
    "model": "claude-haiku-4-5",
    "usage": {"input_tokens": 200, "output_tokens": 20},
    "cost_usd": 0.0003,
}


class TestUnvalidatedOrRefusalExchangesNeverCached:
    """Issue #57 TDD plan item 4: the admission gate is fail-closed —
    only provably clean grounded exchanges may be replayed."""

    def _gate(self, **overrides) -> bool:
        values = dict(
            route="retrieval",
            history=[],
            validation=dict(CLEAN_VALIDATION),
            error_terminated=False,
        )
        values.update(overrides)
        return cacheable_exchange(**values)

    def test_a_clean_first_turn_grounded_exchange_is_cacheable(self) -> None:
        assert self._gate() is True

    def test_non_retrieval_routes_are_never_cacheable(self) -> None:
        for route in ("canned", "chart", "cached_starter", "cached"):
            assert self._gate(route=route) is False, (
                f"route {route!r} must never enter the semantic cache — canned "
                "has its own $0 path, chart specs have permalinks, and a "
                "cached serving must never re-cache itself"
            )

    def test_refusals_are_never_cacheable(self) -> None:
        # A retrieval-route refusal produces NO validation record (the
        # validator never ran; nothing was generated): the empty mapping
        # fails the gate — an honest refusal is not an answer to replay.
        assert self._gate(validation={}) is False

    def test_follow_up_turns_are_never_cacheable(self) -> None:
        # FLAGGED decision 2: first-turn questions only — with history in
        # play the raw question is not the whole context, and the cache
        # keys on the raw question.
        assert self._gate(history=[{"role": "user", "content": "earlier turn"}]) is False

    def test_error_terminated_streams_are_never_cacheable(self) -> None:
        assert self._gate(error_terminated=True) is False

    def test_unvalidated_or_degraded_exchanges_are_never_cacheable(self) -> None:
        assert self._gate(validation={**CLEAN_VALIDATION, "validated": False}) is False
        assert (
            self._gate(
                validation={
                    **CLEAN_VALIDATION,
                    "validated": False,
                    "degraded_reason": "validator transport error",
                }
            )
            is False
        ), "a degraded validation is an UNVERIFIED answer — never replayable"
        assert (
            self._gate(
                validation={
                    **CLEAN_VALIDATION,
                    "skipped_reason": "no factual sentences",
                }
            )
            is False
        ), "a skipped validation never proved the answer clean — fail closed"


class TestLruBound:
    ORTHOGONAL = {
        "first cached question": (1.0, 0.0, 0.0, 0.0),
        "second cached question": (0.0, 1.0, 0.0, 0.0),
        "third cached question": (0.0, 0.0, 1.0, 0.0),
    }

    def test_bound_evicts_the_least_recently_used_entry(self) -> None:
        cache = make_cache(self.ORTHOGONAL, max_entries=2)
        cache.store(**store_kwargs("first cached question", source_exchange_id="src-1"))
        cache.store(**store_kwargs("second cached question", source_exchange_id="src-2"))
        assert len(cache) == 2
        # A hit bumps recency: "first" becomes most recently used.
        assert cache.lookup("first cached question") is not None
        cache.store(**store_kwargs("third cached question", source_exchange_id="src-3"))
        assert len(cache) == 2, "the bound is a hard invariant, never exceeded"
        assert cache.lookup("second cached question") is None, (
            "the least-recently-USED entry (second: stored second, never hit) is the one evicted"
        )
        assert cache.lookup("first cached question") is not None
        assert cache.lookup("third cached question") is not None

    def test_restoring_the_same_question_replaces_its_entry(self) -> None:
        cache = make_cache({QUESTION: ENTRY_VECTOR}, max_entries=2)
        cache.store(**store_kwargs(QUESTION, answer_text="the first stored answer"))
        cache.store(**store_kwargs(QUESTION, answer_text="the refreshed stored answer"))
        assert len(cache) == 1, "one normalised question holds ONE entry"
        hit = cache.lookup(QUESTION)
        assert hit is not None
        assert hit.entry.answer_text == "the refreshed stored answer"


class TestThumbsDownEviction:
    """The #56 interplay: a "down" verdict poisons the cached answer."""

    def _warm(self) -> SemanticCache:
        cache = make_cache({QUESTION: ENTRY_VECTOR})
        cache.store(**store_kwargs(QUESTION, source_exchange_id="src-under-test"))
        return cache

    def test_thumbs_down_on_the_source_exchange_evicts(self) -> None:
        cache = self._warm()
        assert cache.handle_thumbs_down("src-under-test") is True
        assert cache.lookup(QUESTION) is None
        assert cache.handle_thumbs_down("src-under-test") is False, (
            "a second down on the same id has nothing left to evict"
        )

    def test_thumbs_down_on_a_recorded_serving_evicts_the_source(self) -> None:
        cache = self._warm()
        cache.record_serving("serving-exchange-9", "src-under-test")
        assert cache.handle_thumbs_down("serving-exchange-9") is True
        assert cache.lookup(QUESTION) is None

    def test_unknown_exchange_ids_evict_nothing(self) -> None:
        cache = self._warm()
        assert cache.handle_thumbs_down("never-issued-id") is False
        assert cache.lookup(QUESTION) is not None

    def test_direct_evict_by_source_exchange_id(self) -> None:
        cache = self._warm()
        assert cache.evict("src-under-test") is True
        assert cache.evict("src-under-test") is False
        assert cache.lookup(QUESTION) is None


class TestRetentionPurge:
    """Cached content follows the exchange log's §9 90-day bound."""

    def test_purge_expired_drops_only_over_age_entries(self) -> None:
        clock = FrozenClock(CACHE_T0)
        vectors = {
            "an early cached question": (1.0, 0.0, 0.0, 0.0),
            "a later cached question": (0.0, 1.0, 0.0, 0.0),
        }
        cache = make_cache(vectors, clock=clock)
        cache.store(**store_kwargs("an early cached question"))
        clock.advance(timedelta(days=50))
        cache.store(**store_kwargs("a later cached question"))
        clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS - 50 + 1))

        assert cache.purge_expired() == 1
        assert cache.lookup("an early cached question") is None
        assert cache.lookup("a later cached question") is not None
        assert cache.purge_expired() == 0
