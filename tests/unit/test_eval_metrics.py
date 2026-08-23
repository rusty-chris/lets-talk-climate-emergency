"""Issue #21 red phase: deterministic metric arithmetic (TDD plan items
1, 5, 7) — pure functions against synthetic records with hand-computed
expected values (IMPLEMENTATION.md §4.4)."""

from __future__ import annotations

import math

import pytest

from evals.metrics import (
    MetricInputError,
    calibrated_term_preserved,
    mrr,
    ndcg_at_k,
    recall_at_k,
    voices_separation_violations,
)


def test_recall_mrr_ndcg_hand_computed_values():
    """Issue #21 TDD plan item 1."""
    retrieved = ["a", "b", "c", "d"]

    # recall honours the per-item semantics (finding #196; no default).
    assert recall_at_k(retrieved, ["a", "c"], semantics="all_gold") is True
    assert recall_at_k(retrieved, ["a", "z"], semantics="all_gold") is False
    assert recall_at_k(retrieved, ["z", "c"], semantics="any_gold") is True
    assert recall_at_k(retrieved, ["z"], semantics="any_gold") is False
    # k cuts the retrieved list: gold at rank 3 is missed at k=2.
    assert recall_at_k(retrieved, ["c"], k=2, semantics="any_gold") is False
    with pytest.raises(TypeError):
        recall_at_k(retrieved, ["a"])  # semantics is required, keyword-only
    with pytest.raises(MetricInputError):
        recall_at_k(retrieved, ["a"], semantics="most_gold")
    with pytest.raises(MetricInputError):
        recall_at_k(retrieved, [], semantics="any_gold")

    # MRR: first gold at rank 3 → 1/3; absent → 0.0.
    assert mrr(["x", "y", "g"], ["g"]) == pytest.approx(1 / 3)
    assert mrr(["x", "y"], ["g"]) == 0.0

    # nDCG@8, binary relevance, hand-computed:
    # hits at ranks 2 and 4 → DCG = 1/log2(3) + 1/log2(5);
    # ideal (2 gold) = 1/log2(2) + 1/log2(3).
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "g1", "y", "g2"], ["g1", "g2"]) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["g1", "g2"], ["g1", "g2"]) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ["g1"]) == 0.0


def test_voices_separation_check_flags_violation():
    """Issue #21 TDD plan item 5: one voices chunk in a science query's
    document set is a violation; voices-directed queries are exempt."""
    science_run = {
        "item_id": "syn-sp-01",
        "route_is_voices": False,
        "documents": [
            {"chunk_id": "syn_doc:0001", "source_type": "report"},
            {"chunk_id": "voices_doc:0001", "source_type": "voices"},
        ],
    }
    violations = voices_separation_violations([science_run])
    assert len(violations) == 1
    assert violations[0]["item_id"] == "syn-sp-01"
    assert violations[0]["chunk_id"] == "voices_doc:0001"

    voices_run = dict(science_run, item_id="syn-va-01", route_is_voices=True)
    assert voices_separation_violations([voices_run]) == ()

    clean_run = {
        "item_id": "syn-sp-02",
        "route_is_voices": False,
        "documents": [{"chunk_id": "syn_doc:0002", "source_type": "report"}],
    }
    assert voices_separation_violations([clean_run]) == ()


def test_calibrated_term_proxy_negation():
    """Issue #21 TDD plan item 7: 'not likely' against 'likely' is NOT
    preserved — polarity must match."""
    source = "It is likely that synthetic warming continues."
    assert calibrated_term_preserved(source, "It is likely warming continues.", "likely") is True
    assert calibrated_term_preserved(source, "It is not likely to continue.", "likely") is False
    assert calibrated_term_preserved(source, "No calibrated language here.", "likely") is False

    negated_source = "It is not likely that the synthetic feedback dominates."
    assert calibrated_term_preserved(negated_source, "It is not likely.", "likely") is True
    assert calibrated_term_preserved(negated_source, "It is likely.", "likely") is False


# ---------------------------------------------------------------------------
# review-21 #244 (minor): the calibrated-term proxy handles its PRIMARY
# vocabulary — multi-word IPCC terms — and apostrophe negation forms.
# ---------------------------------------------------------------------------


def test_calibrated_term_proxy_handles_multi_word_terms():
    """DESIGN §6.1/§6.2 name 'very likely' and 'high confidence' as the
    calibrated vocabulary: a multi-word term preserved verbatim must
    count as preserved, and a negated echo must not (#244)."""
    source = "It is very likely that synthetic warming continues."
    assert (
        calibrated_term_preserved(source, "It is very likely warming continues.", "very likely")
        is True
    )
    assert (
        calibrated_term_preserved(source, "It is not very likely warming continues.", "very likely")
        is False
    )

    confidence_source = "Scientists have high confidence in the synthetic record."
    assert (
        calibrated_term_preserved(
            confidence_source,
            "There is high confidence in the synthetic record.",
            "high confidence",
        )
        is True
    )


def test_calibrated_term_proxy_detects_apostrophe_negations():
    """The tokenizer keeps apostrophes, so the negation vocabulary must
    match "isn't"/"doesn't" as they actually tokenize — "It isn't
    likely" against a positive source is a polarity flip, never
    preserved (#244: fail-open in the exact direction the negation trap
    exists to catch)."""
    source = "It is likely warming continues."
    assert (
        calibrated_term_preserved(source, "It isn't likely warming continues.", "likely") is False
    )
    assert (
        calibrated_term_preserved(source, "It doesn't likely continue warming.", "likely") is False
    )


# ---------------------------------------------------------------------------
# review-21 #243 (major): the voices-separation gate is computable from the
# run records the harness keeps — ItemResult.documents through the
# runner→gate mapping helper.
# ---------------------------------------------------------------------------


def test_voices_gate_computed_from_item_results_flags_violation():
    """ItemResults carrying per-document source_type feed
    voices_separation_violations via the harness's gate-input mapping:
    a voices chunk in a science item's generation document set is a
    violation; the same documents on a voices-category gold item are
    exempt (route_is_voices derived from the gold category) (#243)."""
    from evals.harness import ItemResult, voices_gate_input

    documents = (
        {"chunk_id": "syn_doc:0001", "source_type": "report"},
        {"chunk_id": "voices_doc:0001", "source_type": "voices"},
    )
    science = ItemResult(
        item_id="syn-sp-01",
        arm_model="claude-haiku-4-5",
        route="retrieval",
        documents=documents,
    )
    voices = ItemResult(
        item_id="syn-va-01",
        arm_model="claude-haiku-4-5",
        route="retrieval",
        documents=documents,
    )
    gold_items = [
        {"id": "syn-sp-01", "category": "single_passage"},
        {"id": "syn-va-01", "category": "voices_action"},
    ]

    runs = voices_gate_input([science, voices], gold_items)
    violations = voices_separation_violations(runs)
    assert [(entry["item_id"], entry["chunk_id"]) for entry in violations] == [
        ("syn-sp-01", "voices_doc:0001")
    ]
