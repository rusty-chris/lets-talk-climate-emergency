"""The real bge-reranker-v2-m3 checks for issue #11 — RED. Integration tier.

Everything else in the #11 suite runs against deterministic fakes behind
the `Reranker` seam; these two tests pin what only the real weights can
prove:

1. the cross-encoder loads via `transformers` directly
   (`AutoModelForSequenceClassification` + `AutoTokenizer` with
   sigmoid(logit) scores — FlagEmbedding's `FlagReranker` wrapper calls
   the removed `tokenizer.prepare_for_model` and is broken under
   transformers 5.x; spike-03 findings, deviation 4), emits
   query-comparable scores in (0, 1) (ADR-006's wording: NOT calibrated
   probabilities — query-comparable relevance scores, the property
   thresholding needs and RRF lacks), and ranks the relevant synthetic
   passage first;
2. the ~100 ms / 40-pair CPU budget (ADR-006) is MEASURED and RECORDED
   into a perf log on every run, and ASSERTED only on the demo hardware
   profile (issue #11 acceptance criteria) — CI hardware records
   evidence, it never gates on its own speed.

Skips on a dev machine without the cached weights; FAILS in CI when the
weights are absent (tests/_weights.py — the #32 convention: a skip must
never green a CI tier that was meant to test something).
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pytest

from rag.retrieval import (
    BGE_RERANKER_MODEL_ID,
    RERANK_CANDIDATE_K,
    RERANK_LATENCY_BUDGET_SECONDS,
    BgeRerankerV2M3,
    record_rerank_latency,
)
from tests._weights import require_bge_reranker_weights

#: The env var the release runbook sets on the demo hardware profile;
#: only then is the ~100 ms budget asserted (never on CI hardware).
PERF_PROFILE_ENV = "CLIMATE_CHAT_PERF_PROFILE"
DEMO_PROFILE = "demo"

QUERY = "How much have surface temperatures risen across the Aurelian Basin?"

#: Synthetic passages (invented Aurelian-Basin universe): exactly one is
#: relevant to QUERY; the distractors include near-topic evidence, a
#: voices-style passage and off-topic filler.
RELEVANT = (
    "Surface temperatures across the Aurelian Basin have very likely risen by "
    "one point nine degrees since the fictional baseline period, with the "
    "strongest warming recorded over the eastern escarpment stations."
)
DISTRACTORS = [
    "Reservoir inflows in the basin declined in twenty one of the last twenty "
    "five invented water years, straining the lowland irrigation districts.",
    "The Lanternwood Briefing campaign gathered invented residents each month "
    "to read the assessment aloud in the market square.",
    "Attribution studies compare observed rainfall decline against invented "
    "counterfactual simulations of the drying trend.",
    "The synthetic filler committee publishes an annual lanternwood almanac of "
    "entirely unrelated fictional gardening advice.",
]


def test_reranker_orders_relevant_fixture_chunk_first() -> None:
    """Real bge-reranker-v2-m3 over synthetic passages: one float per
    passage, every score strictly inside (0, 1) — the sigmoid scale the
    refusal threshold is calibrated in — and the relevant passage
    out-scores every distractor, ranking first."""
    require_bge_reranker_weights()

    reranker = BgeRerankerV2M3()
    assert reranker.model_id == BGE_RERANKER_MODEL_ID

    passages = [RELEVANT, *DISTRACTORS]
    scores = reranker.score(QUERY, passages)

    assert len(scores) == len(passages)
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 < s < 1.0 for s in scores), (
        "scores are sigmoid(logit) — strictly inside (0, 1), the scale the "
        "threshold artifact lives in"
    )
    relevant_score, *distractor_scores = scores
    assert all(relevant_score > s for s in distractor_scores), (
        f"the relevant passage must out-score every distractor: {scores}"
    )
    assert max(scores) == relevant_score


@pytest.mark.perf
def test_rerank_latency_recorded(tmp_path: Path) -> None:
    """Acceptance criterion (issue #11): rerank latency over a full
    40-passage candidate set is measured and recorded in the perf log.
    The ~100 ms budget is documented in every record and asserted ONLY
    when the demo hardware profile is declared — a slow CI box records
    honest evidence instead of flaking the tier."""
    require_bge_reranker_weights()

    reranker = BgeRerankerV2M3()
    passages = [RELEVANT, *DISTRACTORS] * 8  # 40 pairs — the §3.2 candidate set
    assert len(passages) == RERANK_CANDIDATE_K

    reranker.score(QUERY, passages[:2])  # warm-up: exclude one-off model init cost

    started = time.perf_counter()
    scores = reranker.score(QUERY, passages)
    elapsed = time.perf_counter() - started
    assert len(scores) == RERANK_CANDIDATE_K

    profile = os.environ.get(PERF_PROFILE_ENV, "unasserted-ci-or-dev")
    log_path = tmp_path / "perf-log.csv"
    record = record_rerank_latency(
        log_path,
        passage_count=RERANK_CANDIDATE_K,
        wall_clock_seconds=elapsed,
        hardware_profile=profile,
    )

    assert record["passage_count"] == RERANK_CANDIDATE_K
    assert record["wall_clock_seconds"] == pytest.approx(elapsed)
    assert record["budget_seconds"] == pytest.approx(RERANK_LATENCY_BUDGET_SECONDS)
    assert record["within_budget"] == (elapsed <= RERANK_LATENCY_BUDGET_SECONDS)
    assert record["hardware_profile"] == profile

    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, "one measurement, one perf-log row"
    row = rows[0]
    assert int(row["passage_count"]) == RERANK_CANDIDATE_K
    assert float(row["wall_clock_seconds"]) == pytest.approx(elapsed, rel=1e-6)
    assert float(row["budget_seconds"]) == pytest.approx(RERANK_LATENCY_BUDGET_SECONDS)
    assert row["hardware_profile"] == profile

    if profile == DEMO_PROFILE:
        assert elapsed <= RERANK_LATENCY_BUDGET_SECONDS, (
            f"demo hardware profile blew the rerank budget: {elapsed:.3f}s for "
            f"{RERANK_CANDIDATE_K} pairs (budget {RERANK_LATENCY_BUDGET_SECONDS}s)"
        )
