"""Issue #291 resolution red phase (Fable, live-release-run session).

The #291 live eval (2026-09-04, real bge-m3 pinned weights, the cache's
own normalise->embed->cosine path) measured the 0.95 threshold against
real geometry and it did NOT hold. Recorded margins
(data/release-run/threshold_gate_291.json of that run):

- adversarial near-misses AT OR ABOVE 0.95 (would have served a cached
  answer across a different question — the misinformation failure #57's
  docstring names):
    0.9838  "Will the AMOC collapse this century?"
            vs "Could the AMOC collapse this century?"      (modality)
    0.9760  "Is it too late to act on climate change?"
            vs "Is it NOT too late to act on climate change?" (negation)
    0.9758  "Didn't global warming stop after 1998?"
            vs "Global warming didn't stop after 1998, right?" (negation)
    0.9699  "Is it too late to act on climate change?"
            vs "When is it too late to act on climate change?" (modality)
- true-paraphrase controls all scored >= 0.9926 (whitespace 1.0,
  punctuation-only 0.9928/0.9926, benign rephrase 0.9976/0.9970); the
  one exception is a lowercased-acronym casing variant at 0.8861, which
  stays a miss (safe direction, accepted).

Orchestrator adjudication (2026-09-04, recorded in the run session):
1. SEMANTIC_CACHE_SIMILARITY_THRESHOLD 0.95 -> 0.99 — above the worst
   recorded near-miss (0.9838), below the true-paraphrase floor
   (0.9926).
2. A LEXICAL VETO as belt-and-braces (both margins are thin): before a
   hit serves, the pair's negation/modality token profiles must MATCH;
   a differing profile is a MISS regardless of cosine. The veto token
   list is a pinned constant; contractions normalise (a token ending
   n't contributes its base word AND "not", so isn't/didn't/won't all
   carry negation).
3. The recorded near-miss pairs above are regression-pinned as misses;
   the paraphrase controls stay hits (except the casing variant, and
   except could/might — a modality flip the ratified veto catches by
   design: a conservative miss, never a cross-question serve).
"""

from __future__ import annotations

import math

import pytest

from service.semantic_cache import (
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SemanticCache,
    SemanticCacheHit,
)
from tests._semantic_cache_fixtures import make_cache, store_kwargs

#: The four live-recorded >=0.95 adversarial pairs (entry, query, live cosine).
LIVE_NEAR_MISSES = [
    ("Will the AMOC collapse this century?", "Could the AMOC collapse this century?", 0.9838),
    (
        "Is it too late to act on climate change?",
        "Is it NOT too late to act on climate change?",
        0.9760,
    ),
    (
        "Didn't global warming stop after 1998?",
        "Global warming didn't stop after 1998, right?",
        0.9758,
    ),
    (
        "Is it too late to act on climate change?",
        "When is it too late to act on climate change?",
        0.9699,
    ),
]


def _at_cosine(cosine: float) -> tuple[float, float, float, float]:
    return (cosine, math.sqrt(1.0 - cosine * cosine), 0.0, 0.0)


def _pair_cache(entry: str, query: str, cosine: float) -> SemanticCache:
    """A cache holding ``entry``, with ``query`` programmed at exactly the
    live-recorded cosine to it."""
    cache = make_cache({entry: (1.0, 0.0, 0.0, 0.0), query: _at_cosine(cosine)})
    cache.store(**store_kwargs(entry))
    return cache


class TestRatifiedThreshold:
    def test_threshold_is_the_291_ratified_value(self) -> None:
        # RATIFIED 2026-09-04 (orchestrator adjudication on the #291 live
        # margins): 0.99, above the worst recorded adversarial near-miss
        # (0.9838) and below the true-paraphrase floor (0.9926).
        assert SEMANTIC_CACHE_SIMILARITY_THRESHOLD == 0.99


class TestLiveRecordedNearMissesStayMisses:
    @pytest.mark.parametrize(("entry", "query", "cosine"), LIVE_NEAR_MISSES)
    def test_recorded_near_miss_pair_is_a_miss(self, entry, query, cosine) -> None:
        """Each live-recorded >=0.95 pair misses — via the raised
        threshold, the lexical veto, or both. Serving any of them would
        be exactly the cross-question misinformation failure the live
        eval demonstrated 0.95 permits."""
        cache = _pair_cache(entry, query, cosine)
        assert cache.lookup(query) is None, (
            f"{query!r} scored {cosine} against {entry!r} on real bge-m3 geometry — "
            "it MUST miss under the ratified 0.99 + lexical-veto gate"
        )

    @pytest.mark.parametrize(
        ("entry", "query"),
        [
            # Modality flip: will vs could.
            ("Will the AMOC collapse this century?", "Could the AMOC collapse this century?"),
            # Negation insertion: NOT appears on one side only.
            (
                "Is it too late to act on climate change?",
                "Is it NOT too late to act on climate change?",
            ),
        ],
    )
    def test_veto_pairs_miss_even_at_perfect_cosine(self, entry, query) -> None:
        """The veto is belt-and-braces UNDER the threshold: a pair whose
        negation/modality profile differs is a miss REGARDLESS of
        cosine — pinned at cosine 1.0, where only the veto can act."""
        shared = (1.0, 0.0, 0.0, 0.0)
        cache = make_cache({entry: shared, query: shared})
        cache.store(**store_kwargs(entry))
        assert cache.lookup(query) is None, (
            "a negation/modality-profile mismatch must veto the serve even at "
            "cosine 1.0 (ratified belt-and-braces; both live margins are thin)"
        )

    def test_contraction_negation_is_normalised(self) -> None:
        """isn't/didn't-style contractions carry negation: an entry with
        "didn't" and a query without any negation token differ in
        profile and must miss at any cosine."""
        entry = "Didn't global warming stop after 1998?"
        query = "Global warming stopped after 1998, right?"
        shared = (1.0, 0.0, 0.0, 0.0)
        cache = make_cache({entry: shared, query: shared})
        cache.store(**store_kwargs(entry))
        assert cache.lookup(query) is None


class TestParaphraseControlsStillHit:
    def test_punctuation_variant_hits_at_the_recorded_margin(self) -> None:
        """The live punctuation-only control scored 0.9928 — above the
        ratified 0.99, same veto profile: it must still hit."""
        entry = "How much has the planet actually warmed so far?"
        query = "How much has the planet actually warmed so far"
        cache = _pair_cache(entry, query, 0.9928)
        hit = cache.lookup(query)
        assert isinstance(hit, SemanticCacheHit)
        assert hit.similarity == pytest.approx(0.9928)

    def test_benign_rephrase_hits_at_the_recorded_margin(self) -> None:
        """The live benign-rephrase control ("what exactly is" vs "what
        is ..., exactly") scored 0.9976 with an identical veto profile
        ("is" on both sides): still a hit."""
        entry = "What exactly is a climate tipping point?"
        query = "What is a climate tipping point, exactly?"
        cache = _pair_cache(entry, query, 0.9976)
        hit = cache.lookup(query)
        assert isinstance(hit, SemanticCacheHit)

    def test_exact_repeat_still_hits_at_full_similarity(self) -> None:
        entry = "Why are scientists calling this an emergency?"
        cache = make_cache({entry: (1.0, 0.0, 0.0, 0.0)})
        cache.store(**store_kwargs(entry))
        hit = cache.lookup(entry)
        assert isinstance(hit, SemanticCacheHit)
        assert hit.similarity == pytest.approx(1.0)

    def test_could_might_rephrase_is_a_conservative_veto_miss(self) -> None:
        """could->might scored 0.9970 live (a genuine paraphrase) but IS
        a modality-token change, so the ratified veto misses it — the
        safe direction (one extra live answer, never a wrong serve).
        Pinned so the trade-off is a decision, not an accident."""
        entry = "How bad could sea level rise get by the end of the century?"
        query = "How bad might sea level rise get by the end of the century?"
        cache = _pair_cache(entry, query, 0.9970)
        assert cache.lookup(query) is None


class TestVetoSelectsAmongEntries:
    def test_a_vetoed_closer_entry_never_shadows_a_clean_hit(self) -> None:
        """The veto excludes an entry from contention; it does not kill
        the lookup: a non-vetoed entry that clears the threshold still
        serves even when a vetoed entry sits at higher cosine."""
        vetoed_entry = "Could the AMOC collapse this century?"
        clean_entry = "Will the AMOC collapse this century?"
        query = "Will the AMOC collapse this century soon?"
        cache = make_cache(
            {
                vetoed_entry: (1.0, 0.0, 0.0, 0.0),
                clean_entry: _at_cosine(0.995),
                query: (1.0, 0.0, 0.0, 0.0),  # cosine 1.0 to the vetoed entry
            }
        )
        cache.store(**store_kwargs(vetoed_entry, source_exchange_id="src-vetoed"))
        cache.store(**store_kwargs(clean_entry, source_exchange_id="src-clean"))
        hit = cache.lookup(query)
        assert isinstance(hit, SemanticCacheHit)
        assert hit.entry.source_exchange_id == "src-clean", (
            "the will/could-profile mismatch vetoes the cosine-1.0 entry; the "
            "matching-profile entry at 0.995 is the honest serve"
        )


class TestVetoTokenConstant:
    def test_veto_token_list_is_pinned(self) -> None:
        from service.semantic_cache import SEMANTIC_CACHE_VETO_TOKENS

        # The ratified list: negation (not/never/no; n't via contraction
        # normalisation) + modality (will/would/could/might/may/is; isn't
        # via contraction normalisation). A frozenset constant — changing
        # membership is a ratified decision, never a drive-by.
        assert isinstance(SEMANTIC_CACHE_VETO_TOKENS, frozenset)
        assert {
            "not",
            "never",
            "no",
            "will",
            "would",
            "could",
            "might",
            "may",
            "is",
        } <= SEMANTIC_CACHE_VETO_TOKENS
