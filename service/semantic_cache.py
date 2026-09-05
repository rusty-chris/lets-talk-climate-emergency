"""The semantic response cache over answered grounded exchanges (issue #57).

The behaviour is pinned by ``tests/unit/test_semantic_cache.py``,
``tests/unit/test_service_semantic_cache.py``,
``tests/unit/test_ui_semantic_cache.py`` and
``tests/integration/test_semantic_cache_pause.py``.

SotA adoption rec 5 (reviews/sota-portfolio-review-2026-08.md B4,
client-approved as issue #57): before any LLM spend, embed the incoming
question with the LOCAL bge-m3 embedder (DESIGN §3.2 — embeddings are
local, a lookup costs $0) and serve a previously answered, validated
grounded exchange VERBATIM when cosine similarity clears a conservative
threshold. The classic failure — serving a cached answer to a *similar
but different* question — is an integrity risk for an anti-misinformation
product, so every guard here is conservative and fail-closed.

## The honest-cache invariants (each pinned by the red suites)

- **Only clean exchanges enter.** :func:`cacheable_exchange` admits only
  first-turn (empty-history) ``retrieval``-route grounded exchanges whose
  runtime citation-support validation COMPLETED un-degraded
  (``validated`` true, no ``degraded_reason``, no ``skipped_reason``)
  and whose stream terminated cleanly (no ``error`` event). Refusals,
  canned responses, chart responses, paused furniture and cached
  servings themselves are NEVER cached (canned has its own $0 path;
  chart specs have permalinks; a refusal is not an answer).
- **A hit replays VERBATIM.** The stored answer text, citation event
  data, badge event data (an "unverified" badge earned by the original
  answer rides every replay — honesty is not laundered by caching), the
  #220 sources-panel entries (already licence-bounded at storage time)
  and the footer are served byte-identical, marked as a cached answer
  and dated with the ORIGINAL answer's date — never presented as fresh.
- **Corpus-version bound, fail closed.** Every entry is stamped with the
  corpus version it was answered under; entries seeded from any OTHER
  version are discarded wholesale at construction, and ``lookup``
  double-checks the stamp before serving (a stale citation set is a
  licensing and honesty hazard — DESIGN §2.1/§3.6).
- **Privacy adds nothing.** An entry stores exactly the content the
  exchange log already stores (question text, answer, citations) plus a
  locally computed embedding — no identifiers, no new join surface.
  Entries follow the DESIGN §9 90-day bound via :meth:`purge_expired`
  (wired into ``service.retention.run_retention_pass``).
- **$0 and metered.** A cache hit makes ZERO provider-adapter calls of
  any kind and records ZERO spend; it still counts against the per-IP
  rate limit like any chat request. Because a hit is free it also
  serves while the daily budget is PAUSED (issue #57 TDD plan item 5).
- **Thumbs-down evicts (#56 interplay).** Every serving logs its own
  exchange record (fresh ``exchange_id``, ``cached_from`` naming the
  source exchange); a "down" verdict landing on the SOURCE exchange or
  on ANY serving of it evicts the entry via
  :meth:`handle_thumbs_down`. "Up" never evicts.
- **Bounded.** At most :data:`SEMANTIC_CACHE_MAX_ENTRIES` entries; the
  least-recently-used entry (a hit or a store counts as use) is evicted
  first.

## Flagged decisions (orchestrator ratification, #57 red-phase report)

1. **Threshold** :data:`SEMANTIC_CACHE_SIMILARITY_THRESHOLD` = 0.99
   cosine over the DENSE bge-m3 vectors, plus the
   :data:`SEMANTIC_CACHE_VETO_TOKENS` lexical veto (RATIFIED 2026-09-04,
   the issue #291 resolution: the original 0.95 was live-evaluated
   against the real embedder and did not hold — adversarial near-misses
   reached 0.9838 while true paraphrases stayed >= 0.9926). A hit
   requires ``similarity >= threshold`` AND an equal veto profile.
2. **Lookup keys on the RAW question, first turn only, BEFORE the
   classifier.** Issue #57's text says "rewritten" queries, but the
   rewrite is itself a paid Haiku call: embedding the rewritten query
   would make every "$0 cache hit" cost ~$0.001 and make paused-mode
   serving impossible (paused mode makes zero adapter calls). So the
   cache keys on the whitespace-normalised raw question and is consulted
   ONLY when the conversation history is empty — exactly the
   landing-page starter funnel the SotA review targets, where the raw
   question IS the whole context and reference resolution has nothing
   to resolve. Follow-up turns never consult and never populate the
   cache.
3. **Eviction bound** :data:`SEMANTIC_CACHE_MAX_ENTRIES` = 500 (LRU).
4. **Feedback linkage**: a serving's exchange record carries a new
   top-level ``cached_from`` key (the source ``exchange_id``; ``None``
   on every other route) and logs the SOURCE'S canonical question text,
   never the visitor's raw variant. Eviction resolves serving ids to
   their source via the in-cache servings map (:meth:`record_serving`).
5. **In-memory only (MVP)**: no persistence file; the ``entries``
   constructor parameter is the seed/restore seam. The compose smoke
   stack runs with the cache OFF (``CLIMATE_CHAT_SEMANTIC_CACHE=0``) so
   the seeded replay smoke stays deterministic.
6. **Paused-mode order**: the semantic cache is consulted before the
   starter cache; a miss falls through to the unchanged §7.1
   starter/furniture behaviour.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rag.indexing import EmbeddingModel
from service.exchange_log import EXCHANGE_LOG_RETENTION_DAYS

__all__ = [
    "SEMANTIC_CACHE_SIMILARITY_THRESHOLD",
    "SEMANTIC_CACHE_VETO_TOKENS",
    "SEMANTIC_CACHE_MAX_ENTRIES",
    "SEMANTIC_CACHE_ROUTE",
    "SemanticCacheEntry",
    "SemanticCacheHit",
    "SemanticCache",
    "cacheable_exchange",
]

#: RATIFIED 2026-09-04 (issue #291 resolution — orchestrator adjudication
#: on the live-release-run margins): the cosine threshold over DENSE
#: bge-m3 vectors is **0.99**. The #291 live eval measured the original
#: 0.95 against the REAL pinned embedder and it did not hold: adversarial
#: near-misses scored up to 0.9838 (will/could modality flip; negation
#: insertions ~0.976) while every true-paraphrase control scored >=
#: 0.9926 — 0.99 sits above the worst near-miss and below the paraphrase
#: floor. Both margins are thin, hence the lexical veto below as
#: belt-and-braces. A hit requires ``similarity >= threshold`` AND a
#: matching veto-token profile.
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.99

#: RATIFIED 2026-09-04 (issue #291 resolution, item 2): the lexical-veto
#: token list — negation (not/never/no) + modality (will/would/could/
#: might/may/is). Before a hit serves, both questions' veto-token
#: profiles (the subset of these tokens each contains, contractions
#: normalised: a token ending ``n't`` contributes its base word AND
#: ``not``, so isn't/didn't/won't all carry negation) must be EQUAL; a
#: differing profile is a MISS regardless of cosine. Changing membership
#: is a ratified decision, never a drive-by.
SEMANTIC_CACHE_VETO_TOKENS = frozenset(
    {"not", "never", "no", "will", "would", "could", "might", "may", "is"}
)

#: DECISION (flagged for ratification, #57 red notes): the entry-count
#: bound. Least-recently-used (hit or store = use) evicted first.
SEMANTIC_CACHE_MAX_ENTRIES = 500

#: The exchange-log ``route`` value AND the wire ``answer`` event
#: ``kind`` for a semantic-cache serving (pinned equal to
#: ``service.app.ANSWER_KIND_CACHED`` by the red suite).
SEMANTIC_CACHE_ROUTE = "cached"


@dataclass(frozen=True)
class SemanticCacheEntry:
    """One cached answered exchange — the same content the exchange log
    stores, plus the locally computed dense query embedding.

    ``citations`` / ``badges`` / ``sources`` hold the ORIGINAL wire
    event data mappings verbatim (the #12 citation events' data, the
    #13 badge events' data, and the #220 sources event's entry list) so
    a hit can replay them byte-identical. ``generated_on`` is the ISO
    date the source answer was generated (the honesty marker's date);
    ``corpus_version`` stamps the corpus the answer cited;
    ``source_exchange_id`` joins the entry to the source exchange's log
    record; ``stored_at`` drives the 90-day purge.
    """

    question: str
    embedding: tuple[float, ...]
    answer_text: str
    footer: str
    citations: tuple[Mapping[str, Any], ...]
    badges: tuple[Mapping[str, Any], ...]
    sources: tuple[Mapping[str, Any], ...]
    generated_on: str
    corpus_version: str
    source_exchange_id: str
    stored_at: datetime
    #: Precomputed L2 norm of ``embedding`` (finding #290): the cosine scan
    #: reduces to a dot product divided by the two cached norms, so a lookup
    #: never recomputes an entry's norm (identical for the entry's lifetime).
    embedding_norm: float = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "embedding_norm",
            math.sqrt(sum(component * component for component in self.embedding)),
        )


@dataclass(frozen=True)
class SemanticCacheHit:
    """One lookup hit: the entry to replay plus the similarity that won it."""

    entry: SemanticCacheEntry
    similarity: float


def cacheable_exchange(
    *,
    route: str,
    history: Sequence[Any],
    validation: Mapping[str, Any],
    error_terminated: bool,
) -> bool:
    """Pure gate: may this completed exchange enter the semantic cache?

    ``tests/unit/test_semantic_cache.py`` pins:

    - ``route != "retrieval"`` -> False (canned, chart, cached_starter,
      cached servings and paused furniture are NEVER cached; a
      retrieval-route REFUSAL never reaches this gate with a validation
      record and fails the validated check below).
    - non-empty ``history`` -> False (first-turn questions only —
      flagged decision 2).
    - ``error_terminated`` -> False (a partial answer is not a cacheable
      answer).
    - ``validation`` must show a COMPLETED, un-degraded run: a truthy
      ``validated``, ``degraded_reason`` None/absent AND
      ``skipped_reason`` None/absent; an empty mapping (no validation
      ran) -> False. Fail closed: only provably clean exchanges cache.
    """
    if route != "retrieval":
        return False
    if history:
        return False
    if error_terminated:
        return False
    # Fail closed: a completed, un-degraded, un-skipped validation only.
    if not validation or not validation.get("validated"):
        return False
    if validation.get("degraded_reason") or validation.get("skipped_reason"):
        return False
    return True


def _normalise(question: str) -> str:
    """The cache's pinned key normalisation: collapsed whitespace."""
    return " ".join(question.split())


_VETO_WORD_RE = re.compile(r"[a-z']+")


def _veto_profile(question: str) -> frozenset[str]:
    """The question's negation/modality token profile (#291 resolution).

    Lowercased word tokens; a token ending ``n't`` contributes its base
    word AND ``not`` (isn't -> is + not; didn't -> did + not), then the
    profile is the intersection with :data:`SEMANTIC_CACHE_VETO_TOKENS`.
    Two questions may only serve one another when their profiles are
    EQUAL — a differing profile vetoes the hit regardless of cosine.
    """
    tokens: set[str] = set()
    for token in _VETO_WORD_RE.findall(question.lower()):
        if token.endswith("n't"):
            tokens.add(token[:-3])
            tokens.add("not")
        else:
            tokens.add(token)
    return frozenset(tokens & SEMANTIC_CACHE_VETO_TOKENS)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product over two dense vectors of EQUAL dimension.

    ``strict=True`` makes a dimension mismatch raise ``ValueError`` (finding
    #290): a shorter or longer vector — a corrupt seed, or an embedder swap
    without cache invalidation — must fail loudly, never score a plausible
    similarity over the truncated shared prefix.
    """
    return sum(a * b for a, b in zip(left, right, strict=True))


class SemanticCache:
    """Bounded in-memory similarity cache over answered grounded exchanges.

    ``embedding_model`` is the injectable LOCAL embedder seam
    (``rag.indexing.EmbeddingModel`` — the real ``Bgem3EmbeddingModel``
    in production, a deterministic fake in every unit test; lookups make
    zero network calls and cost $0). ``clock`` is the injected aware-UTC
    time source (stamps ``stored_at``; drives :meth:`purge_expired`).

    ``entries`` is the seed/restore seam: seeded entries whose
    ``corpus_version`` differs from ``corpus_version`` are discarded
    WHOLESALE at construction (issue #57: cache invalidated on corpus
    release — fail closed, never a stale citation).
    """

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        corpus_version: str,
        clock: Callable[[], datetime],
        threshold: float = SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
        max_entries: int = SEMANTIC_CACHE_MAX_ENTRIES,
        entries: Sequence[SemanticCacheEntry] = (),
    ) -> None:
        self.embedding_model = embedding_model
        self.corpus_version = corpus_version
        self.threshold = threshold
        self.max_entries = max_entries
        self._clock = clock
        # One lock guards every access to _entries/_servings — the cache is
        # called from FastAPI threadpool workers (the sync /chat and /feedback
        # endpoints, the SSE generator's lookup/store, the lifespan retention
        # task), where a store landing mid-scan would otherwise raise
        # "OrderedDict mutated during iteration" and kill a stream. The embed
        # call stays OUTSIDE it (the expensive part; it touches no shared
        # state). Mirrors ExchangeLog._lock (finding #286).
        self._lock = threading.Lock()
        # LRU by normalised question: least-recently-used at the front.
        self._entries: OrderedDict[str, SemanticCacheEntry] = OrderedDict()
        # serving exchange_id -> source exchange_id (flagged decision 4).
        self._servings: dict[str, str] = {}
        # source exchange_id -> its serving ids, so a departing entry's joins
        # are pruned in O(joins) rather than leaking forever (finding #288).
        self._servings_by_source: dict[str, set[str]] = {}
        # Seed/restore seam: entries answered under ANY OTHER corpus version
        # are discarded WHOLESALE here (fail closed — never a stale citation).
        for entry in entries:
            if entry.corpus_version == corpus_version:
                self._entries[_normalise(entry.question)] = entry

    def __len__(self) -> int:
        """The live entry count (never exceeds ``max_entries``)."""
        with self._lock:
            return len(self._entries)

    def _embed(self, normalised_question: str) -> tuple[float, ...]:
        """The LOCAL, $0 dense embedding of one normalised question."""
        return tuple(self.embedding_model.encode([normalised_question])[0].dense)

    def lookup(self, question: str) -> SemanticCacheHit | None:
        """The $0 similarity lookup for one incoming question.

        Pinned contract: whitespace-normalise ``question`` (blank ->
        None, no embed call), embed it with the LOCAL injected embedder,
        cosine over the DENSE vectors against every entry; return the
        best entry as a :class:`SemanticCacheHit` iff its similarity
        ``>= threshold`` AND its negation/modality veto profile equals
        the query's (:func:`_veto_profile` — the #291 lexical veto: a
        differing profile is a miss regardless of cosine, the vetoed
        entry simply leaves contention) AND its ``corpus_version`` stamp
        matches the cache's (fail closed), bumping its LRU recency;
        otherwise None. Never a provider-adapter call, never a network
        call.
        """
        key = _normalise(question)
        if not key:
            # A blank question is a guaranteed miss: never embed it.
            return None
        with self._lock:
            if not self._entries:
                # An empty cache is a guaranteed miss: never pay the (multi-GB,
                # lazily-loaded) embedder to embed a question with nothing to
                # score it against — the first /chat of every process (finding
                # #287).
                return None
        # The embed is the expensive part and touches no shared state, so it
        # runs OUTSIDE the lock (finding #286).
        query = self._embed(key)
        query_norm = math.sqrt(sum(component * component for component in query))
        if query_norm == 0.0:
            # A null query vector is a guaranteed miss (fail closed).
            return None
        # The #291 lexical veto profile, computed once per lookup.
        query_profile = _veto_profile(key)
        with self._lock:
            best_key: str | None = None
            best: SemanticCacheEntry | None = None
            best_similarity = -1.0
            for entry_key, entry in self._entries.items():
                # Fail closed: an entry from any other corpus version never
                # serves (belt-and-braces over the construction-time discard).
                if entry.corpus_version != self.corpus_version:
                    continue
                if entry.embedding_norm == 0.0:
                    continue
                if _veto_profile(_normalise(entry.question)) != query_profile:
                    # #291 lexical veto (RATIFIED): a negation/modality
                    # profile mismatch is a MISS for this entry regardless
                    # of cosine — near-identical embeddings across a
                    # negation or will/could flip measured >= 0.95 on real
                    # bge-m3 geometry, and serving across one is the
                    # misinformation failure the cache must never commit.
                    continue
                # Cosine reduces to a dot product over the two precomputed
                # norms (finding #290); a dimension mismatch raises loudly.
                similarity = _dot(query, entry.embedding) / (query_norm * entry.embedding_norm)
                # A hit requires similarity >= threshold; the best clearing
                # entry wins (first-max on a tie).
                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_key = entry_key
                    best = entry
            if best is None or best_key is None:
                return None
            # A hit counts as use: bump the entry's LRU recency.
            self._entries.move_to_end(best_key)
            return SemanticCacheHit(entry=best, similarity=best_similarity)

    def store(
        self,
        *,
        question: str,
        answer_text: str,
        footer: str,
        citations: Sequence[Mapping[str, Any]],
        badges: Sequence[Mapping[str, Any]],
        sources: Sequence[Mapping[str, Any]],
        generated_on: str,
        source_exchange_id: str,
    ) -> None:
        """Admit one CLEAN exchange (caller enforces :func:`cacheable_exchange`).

        Pinned contract: embeds the whitespace-normalised question
        locally, stamps the cache's ``corpus_version`` and the clock's
        ``stored_at``, stores the wire payloads verbatim, and evicts the
        least-recently-used entry when the bound is exceeded (the store
        itself counts as the new entry's use).
        """
        key = _normalise(question)
        # The embed is the expensive part and touches no shared state (finding
        # #286): compute it BEFORE taking the lock.
        embedding = self._embed(key)
        entry = SemanticCacheEntry(
            question=question,
            embedding=embedding,
            answer_text=answer_text,
            footer=footer,
            citations=tuple(dict(citation) for citation in citations),
            badges=tuple(dict(badge) for badge in badges),
            sources=tuple(dict(source) for source in sources),
            generated_on=generated_on,
            corpus_version=self.corpus_version,
            source_exchange_id=source_exchange_id,
            stored_at=self._clock(),
        )
        with self._lock:
            # One normalised question holds ONE entry: a re-store replaces it
            # and counts as the freshest use (moved to the most-recent end). A
            # replacement under a NEW source id retires the superseded entry's
            # joins (finding #288 — they would otherwise resolve to a source no
            # entry carries).
            superseded = self._entries.get(key)
            if superseded is not None and superseded.source_exchange_id != source_exchange_id:
                self._drop_source_joins(superseded.source_exchange_id)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                # Evict the least-recently-used entry (front of the ordering).
                _key, displaced = self._entries.popitem(last=False)
                self._drop_source_joins(displaced.source_exchange_id)

    def record_serving(self, serving_exchange_id: str, source_exchange_id: str) -> None:
        """Join one serving's fresh exchange_id to its source entry.

        Pinned contract: after a hit is served (and logged under its own
        new ``exchange_id``), the app records the join here so a later
        thumbs-down on the SERVING evicts the SOURCE entry (flagged
        decision 4 — the linkage lives in-cache, no log lookup needed).
        """
        with self._lock:
            self._servings[serving_exchange_id] = source_exchange_id
            self._servings_by_source.setdefault(source_exchange_id, set()).add(serving_exchange_id)

    def _drop_source_joins(self, source_exchange_id: str) -> None:
        """Prune every serving join of a source entry that just left the cache.

        Called (with the lock held) whenever an entry leaves ``_entries`` for
        ANY reason (evict, purge, LRU displacement, re-store replacement) so
        the joins never outlive their entry (finding #288).
        """
        for serving_id in self._servings_by_source.pop(source_exchange_id, ()):
            self._servings.pop(serving_id, None)

    def handle_thumbs_down(self, exchange_id: str) -> bool:
        """The #56 interplay: a "down" verdict evicts the entry it hits.

        Pinned contract: ``exchange_id`` naming a SOURCE entry evicts
        that entry; an id recorded via :meth:`record_serving` evicts the
        serving's source entry; any other id is a no-op. Returns True
        iff an entry was evicted. The app calls this ONLY for "down"
        verdicts — "up" never evicts (pinned at the /feedback route).
        """
        with self._lock:
            # A serving id resolves to its source entry; a source id evicts
            # itself. Resolution and eviction share the one critical section
            # (a non-reentrant lock, so the resolve never re-enters evict).
            source_exchange_id = self._servings.get(exchange_id, exchange_id)
            return self._evict_locked(source_exchange_id)

    def evict(self, source_exchange_id: str) -> bool:
        """Drop the entry stored under ``source_exchange_id`` (True if found)."""
        with self._lock:
            return self._evict_locked(source_exchange_id)

    def _evict_locked(self, source_exchange_id: str) -> bool:
        """Drop the entry for ``source_exchange_id`` (caller holds the lock)."""
        for key, entry in self._entries.items():
            if entry.source_exchange_id == source_exchange_id:
                del self._entries[key]
                self._drop_source_joins(source_exchange_id)
                return True
        return False

    def purge_expired(self) -> int:
        """The §9 retention bound: drop entries older than the exchange
        log's ``EXCHANGE_LOG_RETENTION_DAYS`` at the injected clock's
        now; return the count dropped. Wired into
        ``service.retention.run_retention_pass`` (pinned there)."""
        cutoff = self._clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS)
        with self._lock:
            expired = [
                (key, entry.source_exchange_id)
                for key, entry in self._entries.items()
                if entry.stored_at <= cutoff
            ]
            for key, source_exchange_id in expired:
                del self._entries[key]
                self._drop_source_joins(source_exchange_id)
            return len(expired)
