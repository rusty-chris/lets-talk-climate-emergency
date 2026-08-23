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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

RECALL_ALL_GOLD = "all_gold"
RECALL_ANY_GOLD = "any_gold"
RECALL_SEMANTICS = (RECALL_ALL_GOLD, RECALL_ANY_GOLD)

VOICES_SOURCE_TYPE = "voices"


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
    raise NotImplementedError("issue #21 green phase")


def mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first gold id in the retrieved list
    (1-indexed); 0.0 when no gold id was retrieved."""
    raise NotImplementedError("issue #21 green phase")


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], *, k: int = 8) -> float:
    """Binary-relevance nDCG@k: DCG over gold hits at their retrieved
    ranks / ideal DCG for min(len(gold), k) hits."""
    raise NotImplementedError("issue #21 green phase")


def voices_separation_violations(
    runs: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """The DESIGN §6.2 structural check: for non-voices queries, zero
    ``source_type: voices`` chunks in the generation call's document
    set. Returns one violation record per offending (item, chunk) —
    empty means the check passes. Voices-directed queries are exempt.
    """
    raise NotImplementedError("issue #21 green phase")


def calibrated_term_preserved(source_text: str, answer_text: str, term: str) -> bool:
    """Negation-aware calibrated-language proxy (issue #21 TDD plan
    item 7): the term counts as preserved only when the answer carries
    it in the same polarity as the source — "not likely" in an answer
    against "likely" in the source is NOT preserved (and vice versa).
    """
    raise NotImplementedError("issue #21 green phase")
