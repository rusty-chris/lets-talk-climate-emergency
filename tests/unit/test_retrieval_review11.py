"""Adversarial-review fixes for the #11 retrieval service — RED.

Review findings #172–#178 (review of PR #169). Each test block names the
finding it pins:

- **#172** — the refusal gate must fail CLOSED on a garbage signal:
  NaN/inf reranker scores, a wrong-length score list, or a non-finite
  configured threshold are a typed refusal-of-the-run
  (:class:`RetrievalError`), never a served answer. "An answered result
  implies the top passage cleared the threshold" is an enforced
  invariant, not an accident of finite arithmetic.

Unit tier: in-memory Qdrant + deterministic fakes behind the `Reranker`
seam (IMPLEMENTATION.md §1) — no weights, no network.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from rag.retrieval import (
    HonestRefusal,
    RetrievalError,
    RetrievedPassages,
)
from tests._retrieval_fixtures import (
    BASIN_WARMING_MARKER,
    TableReranker,
    config,
    decision,
    indexed_corpus,
    run_retrieve,
)

QUERY = "invented aurelian basin warming attribution query"

NAN = float("nan")
INF = float("inf")


class ScriptedReranker:
    """A fake reranker whose scores come from an arbitrary script — the
    instrument for adversarial score vectors (wrong lengths, NaN, inf)
    that the well-behaved fakes can never emit."""

    model_id = "scripted-fake-reranker-v1"

    def __init__(self, script: Callable[[Sequence[str]], list[float]]) -> None:
        self._script = script

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return self._script(list(passages))


def _repeating(pattern: Sequence[float]) -> ScriptedReranker:
    """Scores cycling through ``pattern``, one per passage."""
    return ScriptedReranker(
        lambda passages: [pattern[i % len(pattern)] for i in range(len(passages))]
    )


# ---------------------------------------------------------------------------
# Finding #172 — the gate fails CLOSED on non-finite scores.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reranker",
    [
        pytest.param(TableReranker(default=NAN), id="all-scores-nan"),
        pytest.param(
            TableReranker([(BASIN_WARMING_MARKER, NAN)], default=0.1),
            id="one-nan-among-low",
        ),
        pytest.param(TableReranker(default=INF), id="all-scores-inf"),
        pytest.param(
            TableReranker([(BASIN_WARMING_MARKER, -INF)], default=0.1),
            id="one-negative-inf",
        ),
    ],
)
def test_gate_refuses_or_raises_on_nan_scores(reranker) -> None:
    """Finding #172: every NaN comparison is False, so a NaN top score
    passes `top_score < threshold` and the gate answers — the honesty
    gate inverts exactly when the signal is garbage. Non-finite scores
    must be a typed refusal-of-the-run naming the reranker, never an
    answer (and never an answer led by a NaN passage)."""
    client, model, _chunks = indexed_corpus()

    with pytest.raises(RetrievalError) as excinfo:
        run_retrieve(
            client,
            decision(QUERY),
            model=model,
            reranker=reranker,
            cfg=config(refusal_threshold=0.9),
        )
    assert reranker.model_id in str(excinfo.value), (
        "the loud refusal must name the reranker that produced the "
        "non-finite score, so the defect is attributable"
    )


def test_answered_result_top_passage_always_clears_threshold() -> None:
    """Finding #172, the invariant stated by the docstrings and violated
    under NaN: `isinstance(result, RetrievedPassages)` implies
    `result.passages[0].clears_threshold is True`. Property-style over
    adversarial score vectors — every run either answers with a cleared
    top passage, refuses honestly, or raises the typed error; no fourth
    outcome exists."""
    client, model, _chunks = indexed_corpus()

    adversarial_patterns = [
        [NAN],
        [NAN, 0.1, 0.1, 0.1],
        [INF, 0.5],
        [-INF, 0.95],
        [0.95, NAN, 0.2],
        [0.05, 0.04, 0.03],
        [0.95, 0.91, 0.05],
        [0.9, 0.9, 0.9],
        [0.9001, 0.0001],
    ]
    for pattern in adversarial_patterns:
        reranker = _repeating(pattern)
        try:
            result = run_retrieve(
                client,
                decision(QUERY),
                model=model,
                reranker=reranker,
                cfg=config(refusal_threshold=0.9),
            )
        except RetrievalError:
            continue  # loud typed refusal-of-the-run: acceptable outcome
        assert isinstance(result, (RetrievedPassages, HonestRefusal))
        if isinstance(result, RetrievedPassages):
            assert result.passages, "an answered result carries passages"
            assert result.passages[0].clears_threshold is True, (
                f"answered implies the top passage cleared the threshold; "
                f"violated for score pattern {pattern}"
            )
            assert math.isfinite(result.passages[0].rerank_score), (
                "an answered result must never be led by a non-finite score"
            )


@pytest.mark.parametrize("bad_threshold", [NAN, INF, -INF])
def test_non_finite_threshold_refused_loudly(bad_threshold) -> None:
    """Finding #172: a NaN threshold means the gate never fires again,
    silently (every comparison False); -inf answers everything, +inf
    refuses everything. A non-finite threshold is a typed configuration
    error at construction — never a silent behaviour change."""
    with pytest.raises(RetrievalError):
        config(refusal_threshold=bad_threshold)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(lambda n: n - 1, id="one-short"),
        pytest.param(lambda n: n + 1, id="one-long"),
        pytest.param(lambda n: 0, id="empty"),
    ],
)
def test_wrong_length_score_list_raises_retrieval_error(shape) -> None:
    """Finding #172 (loudness): a Reranker returning a wrong-length score
    list is a broken seam contract — it must die as a named
    RetrievalError identifying the reranker, not a bare ValueError from
    zip(strict=True)."""
    client, model, _chunks = indexed_corpus()
    reranker = ScriptedReranker(lambda passages: [0.5] * shape(len(passages)))

    with pytest.raises(RetrievalError) as excinfo:
        run_retrieve(
            client,
            decision(QUERY),
            model=model,
            reranker=reranker,
            cfg=config(refusal_threshold=0.5),
        )
    assert reranker.model_id in str(excinfo.value)
