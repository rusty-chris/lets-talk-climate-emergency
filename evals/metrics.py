"""Deterministic eval metrics (issue #21): pure arithmetic, fully
unit-tested against synthetic run records with hand-computed values
(IMPLEMENTATION.md §4.4 — the harness's arithmetic is pure).

Retrieval metrics honour the per-item ``recall_semantics`` declaration
(review finding #196): ``all_gold`` counts an item recalled only when
EVERY gold chunk surfaces in the top-k; ``any_gold`` when any one does.
There is NO default — multi-passage items must declare it in the gold
file, and callers must pass it explicitly.

Red phase: contracts pinned, behaviour raises NotImplementedError.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

RECALL_ALL_GOLD = "all_gold"
RECALL_ANY_GOLD = "any_gold"
RECALL_SEMANTICS = (RECALL_ALL_GOLD, RECALL_ANY_GOLD)

VOICES_SOURCE_TYPE = "voices"

#: Negation tokens that flip a calibrated term's polarity when they sit
#: immediately before it (issue #21 TDD plan item 7). Kept small and
#: explicit — a proxy, not a parser.
_NEGATION_TOKENS = frozenset(
    {"not", "no", "never", "without", "nor", "cannot", "n't", "dont", "doesnt", "isnt"}
)
_WORD_PATTERN = re.compile(r"[a-z']+")


class MetricInputError(ValueError):
    """A metric was fed inputs it must refuse (unknown semantics, empty
    gold) — loud refusal, never a silent 0.0."""


def recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    k: int = 8,
    semantics: str,
) -> bool:
    """Whether one item counts as recalled at cutoff k.

    ``semantics`` is required (finding #196: no default): all_gold —
    every gold id within the top-k; any_gold — at least one. Unknown
    semantics or empty gold ids raise MetricInputError.
    """
    if semantics not in RECALL_SEMANTICS:
        raise MetricInputError(
            f"unknown recall semantics {semantics!r}; expected one of {RECALL_SEMANTICS} "
            "(finding #196: multi-passage gold declares it, the harness has no default)"
        )
    if not gold_ids:
        raise MetricInputError("recall_at_k requires at least one gold id, got none")
    top_k = set(retrieved_ids[:k])
    if semantics == RECALL_ALL_GOLD:
        return all(gold_id in top_k for gold_id in gold_ids)
    return any(gold_id in top_k for gold_id in gold_ids)


def mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first gold id in the retrieved list
    (1-indexed); 0.0 when no gold id was retrieved."""
    gold = set(gold_ids)
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], *, k: int = 8) -> float:
    """Binary-relevance nDCG@k: DCG over gold hits at their retrieved
    ranks / ideal DCG for min(len(gold), k) hits."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1)
        if retrieved_id in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def voices_separation_violations(
    runs: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """The DESIGN §6.2 structural check: for non-voices queries, zero
    ``source_type: voices`` chunks in the generation call's document
    set. Returns one violation record per offending (item, chunk) —
    empty means the check passes. Voices-directed queries are exempt.
    """
    violations: list[Mapping[str, Any]] = []
    for run in runs:
        if run.get("route_is_voices"):
            continue
        for document in run.get("documents", ()):
            if document.get("source_type") == VOICES_SOURCE_TYPE:
                violations.append(
                    {"item_id": run.get("item_id"), "chunk_id": document.get("chunk_id")}
                )
    return tuple(violations)


def _term_is_negated(tokens: Sequence[str], term: str) -> bool:
    """Whether the first occurrence of ``term`` is directly preceded by a
    negation token (the polarity proxy)."""
    for index, token in enumerate(tokens):
        if token == term:
            return index > 0 and tokens[index - 1] in _NEGATION_TOKENS
    return False


def calibrated_term_preserved(source_text: str, answer_text: str, term: str) -> bool:
    """Negation-aware calibrated-language proxy (issue #21 TDD plan
    item 7): the term counts as preserved only when the answer carries
    it in the same polarity as the source — "not likely" in an answer
    against "likely" in the source is NOT preserved (and vice versa).
    """
    term = term.lower()
    answer_tokens = _WORD_PATTERN.findall(answer_text.lower())
    if term not in answer_tokens:
        return False
    source_tokens = _WORD_PATTERN.findall(source_text.lower())
    return _term_is_negated(source_tokens, term) == _term_is_negated(answer_tokens, term)
