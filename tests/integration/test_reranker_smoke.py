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

#: Off-topic filler sentences (invented gardening-almanac universe) used to
#: build chunks at the chunker's real max size (~500 whitespace words —
#: `ingestion.chunk.ChunkConfig.max_tokens` counts words, finding #175).
FILLER_SENTENCES = [
    "The lanternwood almanac committee recommends pruning ornamental hedge "
    "rows before the first invented frost settles over the allotments.",
    "Volunteer marrow growers exchanged entirely fictional compost recipes "
    "at the annual synthetic gardening fair in the village hall.",
    "A well balanced potting mixture, the almanac insists, contains loam "
    "sand and imaginary leaf mould in equal generous measures.",
    "Seasonal advice for invented greenhouse keepers covers ventilation "
    "shading watering cans and the polishing of decorative gnomes.",
    "The fictional rose society awards a silver trowel each year for the "
    "most extravagantly scented imaginary climbing variety.",
]


def _filler_words(word_count: int) -> list[str]:
    """Deterministic off-topic filler of exactly ``word_count`` words."""
    words: list[str] = []
    sentence_index = 0
    while len(words) < word_count:
        words.extend(FILLER_SENTENCES[sentence_index % len(FILLER_SENTENCES)].split())
        sentence_index += 1
    return words[:word_count]


def _max_size_chunk_with_tail(relevant_sentence: str) -> str:
    """A chunk at the chunker's max word budget whose ONLY relevant
    content is its final sentence — the §2.4 shape where a section's
    calibrated headline finding lands in the last paragraph."""
    from ingestion.chunk import ChunkConfig

    budget = ChunkConfig().max_tokens  # whitespace WORDS (finding #175)
    relevant_words = relevant_sentence.split()
    return " ".join(_filler_words(budget - len(relevant_words)) + relevant_words)


def _max_size_filler_chunk() -> str:
    from ingestion.chunk import ChunkConfig

    return " ".join(_filler_words(ChunkConfig().max_tokens))


def test_reranker_orders_relevant_fixture_chunk_first() -> None:
    """Real bge-reranker-v2-m3 over synthetic passages: one float per
    passage, every score strictly inside (0, 1) — the sigmoid scale the
    refusal threshold is calibrated in — and the relevant passage
    out-scores every distractor, ranking first."""
    require_bge_reranker_weights()

    reranker = BgeRerankerV2M3()
    assert reranker.model_id == BGE_RERANKER_MODEL_ID
    # Finding #178 (the #163 pattern): the loaded identity records the
    # pinned hub revision the snapshot was loaded from.
    from rag.retrieval import BGE_RERANKER_REVISION

    assert reranker.revision == BGE_RERANKER_REVISION

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


def test_reranker_sees_full_max_size_chunk() -> None:
    """Review finding #175: a ~500-WORD chunk (the chunker's real budget,
    `ingestion.chunk.estimate_tokens` counts whitespace words) tokenises
    to ~960 XLM-R subword tokens, but the joint pair was truncated at
    512 — the reranker read roughly the first half of every real-size
    chunk. Measured consequence: the SAME relevant sentence scored
    0.992 at the START of a long chunk and 2.1e-05 (bit-identical to
    pure filler) at its END. Silent ranking inversion and false
    refusals on exactly the chunks the corpus will serve.

    Contract: a max-size chunk whose ONLY relevant content is its final
    sentence must out-score a pure-filler chunk of the same size by a
    wide margin, and must rank first."""
    require_bge_reranker_weights()

    reranker = BgeRerankerV2M3()
    tail_chunk = _max_size_chunk_with_tail(RELEVANT)
    filler_chunk = _max_size_filler_chunk()

    tail_score, filler_score = reranker.score(QUERY, [tail_chunk, filler_chunk])

    assert all(0.0 < s < 1.0 for s in (tail_score, filler_score))
    assert tail_score > filler_score + 0.3, (
        f"the relevant sentence in the chunk's TAIL must out-score pure "
        f"filler by a wide margin — a truncated head-only read makes them "
        f"indistinguishable (finding #175): tail {tail_score!r} vs filler "
        f"{filler_score!r}"
    )
    assert tail_score > 0.5, (
        f"a chunk containing the directly-relevant sentence must score as "
        f"relevant, wherever the sentence sits: {tail_score!r}"
    )


def _realistic_candidate_set() -> list[str]:
    """40 DISTINCT chunks at the chunker's max word budget (finding #176):
    the real §3.2 candidate set is 40 distinct ~500-word chunks, not 5
    short sentences repeated 8x — an order of magnitude more tokens per
    batch, and transformer cost grows superlinearly with sequence
    length. One chunk carries the relevant sentence in its tail (the
    finding-#175 shape), the rest are distinct rotations of filler."""
    from ingestion.chunk import ChunkConfig

    budget = ChunkConfig().max_tokens
    passages: list[str] = []
    for index in range(RERANK_CANDIDATE_K):
        words = _filler_words(budget + index)
        words = words[index:] + words[:index]  # distinct rotation per chunk
        passages.append(" ".join(words[:budget]))
    passages[7] = _max_size_chunk_with_tail(RELEVANT)
    return passages


@pytest.mark.perf
def test_rerank_latency_recorded() -> None:
    """Acceptance criterion (issue #11) as finding #176 re-pins it: rerank
    latency over a REPRESENTATIVE candidate set — 40 distinct chunks at
    the chunker's max size, each tokenising past the pair cap so the
    windowed scoring path is actually exercised — is measured and
    recorded in a perf log that OUTLIVES the test (the repo-level
    perf-evidence home, env-var overridable; previously the log went to
    pytest tmp_path and was destroyed at teardown). The ~100 ms budget
    is documented in every record and asserted ONLY when the demo
    hardware profile is declared — a slow CI box records honest
    evidence instead of flaking the tier."""
    require_bge_reranker_weights()

    from rag.retrieval import default_perf_log_path

    reranker = BgeRerankerV2M3()
    passages = _realistic_candidate_set()
    assert len(passages) == RERANK_CANDIDATE_K
    assert len(set(passages)) == RERANK_CANDIDATE_K, (
        "the measured batch must be 40 DISTINCT passages, not repeats (finding #176)"
    )
    pair_cap = BgeRerankerV2M3._MAX_PAIR_TOKENS
    for passage in passages:
        subwords = len(reranker._tokenizer(passage, add_special_tokens=False)["input_ids"])
        assert subwords > pair_cap, (
            f"every measured passage must tokenise past the {pair_cap}-token "
            f"pair cap so the measurement covers the real windowed workload; "
            f"got {subwords} subword tokens"
        )

    reranker.score(QUERY, passages[:2])  # warm-up: exclude one-off model init cost

    started = time.perf_counter()
    scores = reranker.score(QUERY, passages)
    elapsed = time.perf_counter() - started
    assert len(scores) == RERANK_CANDIDATE_K

    profile = os.environ.get(PERF_PROFILE_ENV, "unasserted-ci-or-dev")
    log_path = default_perf_log_path()
    rows_before = 0
    if log_path.exists():
        with log_path.open() as handle:
            rows_before = len(list(csv.DictReader(handle)))
    record = record_rerank_latency(
        passage_count=RERANK_CANDIDATE_K,
        wall_clock_seconds=elapsed,
        hardware_profile=profile,
    )

    assert record["passage_count"] == RERANK_CANDIDATE_K
    assert record["wall_clock_seconds"] == pytest.approx(elapsed)
    assert record["budget_seconds"] == pytest.approx(RERANK_LATENCY_BUDGET_SECONDS)
    assert record["within_budget"] == (elapsed <= RERANK_LATENCY_BUDGET_SECONDS)
    assert record["hardware_profile"] == profile

    # The evidence persists at the resolved repo-level (or env-var) home,
    # appended — the row must still be there after the test ends.
    assert log_path.exists(), f"the perf log must persist at {log_path}"
    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == rows_before + 1, "one measurement appends exactly one perf-log row"
    row = rows[-1]
    assert int(row["passage_count"]) == RERANK_CANDIDATE_K
    assert float(row["wall_clock_seconds"]) == pytest.approx(elapsed, rel=1e-6)
    assert float(row["budget_seconds"]) == pytest.approx(RERANK_LATENCY_BUDGET_SECONDS)
    assert row["hardware_profile"] == profile

    if profile == DEMO_PROFILE:
        assert elapsed <= RERANK_LATENCY_BUDGET_SECONDS, (
            f"demo hardware profile blew the rerank budget: {elapsed:.3f}s for "
            f"{RERANK_CANDIDATE_K} pairs (budget {RERANK_LATENCY_BUDGET_SECONDS}s)"
        )
