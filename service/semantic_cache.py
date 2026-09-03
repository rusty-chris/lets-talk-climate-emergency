"""The semantic response cache over answered grounded exchanges (issue #57).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_semantic_cache.py``,
``tests/unit/test_service_semantic_cache.py``,
``tests/unit/test_ui_semantic_cache.py`` and
``tests/integration/test_semantic_cache_pause.py`` pin the contract.

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

1. **Threshold** :data:`SEMANTIC_CACHE_SIMILARITY_THRESHOLD` = 0.95
   cosine over the DENSE bge-m3 vectors (the SotA review's
   "conservative ~0.95+"); a hit requires ``similarity >= threshold``.
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rag.indexing import EmbeddingModel

__all__ = [
    "SEMANTIC_CACHE_SIMILARITY_THRESHOLD",
    "SEMANTIC_CACHE_MAX_ENTRIES",
    "SEMANTIC_CACHE_ROUTE",
    "SemanticCacheEntry",
    "SemanticCacheHit",
    "SemanticCache",
    "cacheable_exchange",
]

#: DECISION (flagged for ratification, #57 red notes): the conservative
#: cosine threshold over DENSE bge-m3 vectors. A hit requires
#: ``similarity >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD``; the SotA
#: review's near-miss adversarial pairs (e.g. "is it too late?" vs
#: "when is it too late?") must land BELOW it.
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.95

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

    RED-phase contract stub; ``tests/unit/test_semantic_cache.py`` pins:

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
    raise NotImplementedError("issue #57 red phase: cacheable_exchange is not implemented yet")


class SemanticCache:
    """Bounded in-memory similarity cache over answered grounded exchanges.

    RED-phase contract stub; behaviour raises ``NotImplementedError``.
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
        # RED note for the implementer: seeded entries from any OTHER
        # corpus version must be dropped here, wholesale.
        self._seed: tuple[SemanticCacheEntry, ...] = tuple(entries)
        # serving exchange_id -> source exchange_id (flagged decision 4).
        self._servings: dict[str, str] = {}

    def __len__(self) -> int:
        """The live entry count (never exceeds ``max_entries``)."""
        raise NotImplementedError("issue #57 red phase: SemanticCache.__len__ is not implemented")

    def lookup(self, question: str) -> SemanticCacheHit | None:
        """The $0 similarity lookup for one incoming question.

        Pinned contract: whitespace-normalise ``question`` (blank ->
        None, no embed call), embed it with the LOCAL injected embedder,
        cosine over the DENSE vectors against every entry; return the
        best entry as a :class:`SemanticCacheHit` iff its similarity
        ``>= threshold`` AND its ``corpus_version`` stamp matches the
        cache's (fail closed), bumping its LRU recency; otherwise None.
        Never a provider-adapter call, never a network call.
        """
        raise NotImplementedError("issue #57 red phase: SemanticCache.lookup is not implemented")

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
        raise NotImplementedError("issue #57 red phase: SemanticCache.store is not implemented")

    def record_serving(self, serving_exchange_id: str, source_exchange_id: str) -> None:
        """Join one serving's fresh exchange_id to its source entry.

        Pinned contract: after a hit is served (and logged under its own
        new ``exchange_id``), the app records the join here so a later
        thumbs-down on the SERVING evicts the SOURCE entry (flagged
        decision 4 — the linkage lives in-cache, no log lookup needed).
        """
        raise NotImplementedError(
            "issue #57 red phase: SemanticCache.record_serving is not implemented"
        )

    def handle_thumbs_down(self, exchange_id: str) -> bool:
        """The #56 interplay: a "down" verdict evicts the entry it hits.

        Pinned contract: ``exchange_id`` naming a SOURCE entry evicts
        that entry; an id recorded via :meth:`record_serving` evicts the
        serving's source entry; any other id is a no-op. Returns True
        iff an entry was evicted. The app calls this ONLY for "down"
        verdicts — "up" never evicts (pinned at the /feedback route).
        """
        raise NotImplementedError(
            "issue #57 red phase: SemanticCache.handle_thumbs_down is not implemented"
        )

    def evict(self, source_exchange_id: str) -> bool:
        """Drop the entry stored under ``source_exchange_id`` (True if found)."""
        raise NotImplementedError("issue #57 red phase: SemanticCache.evict is not implemented")

    def purge_expired(self) -> int:
        """The §9 retention bound: drop entries older than the exchange
        log's ``EXCHANGE_LOG_RETENTION_DAYS`` at the injected clock's
        now; return the count dropped. Wired into
        ``service.retention.run_retention_pass`` (pinned there)."""
        raise NotImplementedError(
            "issue #57 red phase: SemanticCache.purge_expired is not implemented"
        )
