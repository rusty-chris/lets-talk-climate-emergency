"""Refusal gate + threshold calibration mechanism (issue #11) — RED.

ADR-006/ADR-010: the gate thresholds on the reranker's QUERY-COMPARABLE
relevance scores (not calibrated probabilities, and never RRF rank
artefacts). The threshold VALUE is #20/#21's to set — calibrated on the
gold set's no-answer calibration items and delivered through a committed
eval-derived artifact. This suite pins the MECHANISM: gate arithmetic,
partial-support flagging, the template-only refusal payload, calibration
reproducibility, the artifact round-trip, and the §6.1
calibration/gate-item disjointness guard.

Unit tier: injected scores and synthetic chunk lists throughout
(IMPLEMENTATION.md §1 — filter and gate logic are pure).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from rag.retrieval import (
    CalibrationGateOverlapError,
    HonestRefusal,
    RetrievalConfig,
    RetrievalError,
    RetrievedPassages,
    ThresholdCalibration,
    build_refusal_text,
    calibrate_refusal_threshold,
    check_calibration_gate_split,
    load_threshold_artifact,
    save_threshold_artifact,
)
from tests._retrieval_fixtures import (
    ATTRIBUTION_MARKER,
    BASIN_WARMING_MARKER,
    COVERAGE_TOPICS,
    TableReranker,
    config,
    decision,
    indexed_corpus,
    run_retrieve,
)

QUERY = "invented aurelian basin warming attribution query"


def test_refusal_below_threshold() -> None:
    """Every reranked score below the threshold -> a typed HonestRefusal
    recording the top score and the threshold that fired — never a
    passages result, never an exception."""
    client, model, _chunks = indexed_corpus()

    result = run_retrieve(
        client,
        decision(QUERY),
        model=model,
        reranker=TableReranker(default=0.2),
        cfg=config(refusal_threshold=0.6),
    )

    assert isinstance(result, HonestRefusal)
    assert result.top_score == pytest.approx(0.2)
    assert result.threshold == pytest.approx(0.6)


def test_no_refusal_at_threshold() -> None:
    """The boundary answers: a top score exactly AT the threshold clears
    the gate (refusal is strictly `top < threshold`), so the calibrated
    threshold value itself is answerable territory."""
    client, model, _chunks = indexed_corpus()

    result = run_retrieve(
        client,
        decision(QUERY),
        model=model,
        reranker=TableReranker([(BASIN_WARMING_MARKER, 0.6)], default=0.2),
        cfg=config(refusal_threshold=0.6),
    )

    assert isinstance(result, RetrievedPassages)
    assert result.passages[0].rerank_score == pytest.approx(0.6)
    assert result.passages[0].clears_threshold is True


def test_refusal_response_names_covered_topics() -> None:
    """§3.5: the honest refusal says what the corpus DOES cover. The
    payload carries the configured coverage topics, every topic appears
    in the user-facing text, and the text is exactly the pure template's
    output — a template, not a model behaviour."""
    client, model, _chunks = indexed_corpus()

    result = run_retrieve(
        client,
        decision(QUERY),
        model=model,
        reranker=TableReranker(default=0.1),
        cfg=config(refusal_threshold=0.9, corpus_coverage=COVERAGE_TOPICS),
    )

    assert isinstance(result, HonestRefusal)
    assert result.covered_topics == COVERAGE_TOPICS
    for topic in COVERAGE_TOPICS:
        assert topic in result.refusal_text
    assert result.refusal_text == build_refusal_text(COVERAGE_TOPICS), (
        "the refusal text is the pure template's output — no LLM call, no drift"
    )
    assert "cover" in result.refusal_text.lower()


def test_refusal_template_is_pure_and_deterministic() -> None:
    """The template function is pure: identical topics -> identical text;
    every topic named exactly."""
    first = build_refusal_text(COVERAGE_TOPICS)
    second = build_refusal_text(COVERAGE_TOPICS)
    assert first == second
    for topic in COVERAGE_TOPICS:
        assert topic in first


def test_partial_support_flags_which_passages_cleared() -> None:
    """§3.5 partial support: when the top-8 straddles the threshold the
    result still answers (top cleared), but each passage carries
    `clears_threshold` so generation can name what is and is not
    supported, and `partial_support` is True."""
    client, model, _chunks = indexed_corpus()

    result = run_retrieve(
        client,
        decision(QUERY),
        model=model,
        reranker=TableReranker(
            [(BASIN_WARMING_MARKER, 0.9), (ATTRIBUTION_MARKER, 0.7)], default=0.1
        ),
        cfg=config(refusal_threshold=0.5),
    )

    assert isinstance(result, RetrievedPassages)
    cleared = [p for p in result.passages if p.clears_threshold]
    uncleared = [p for p in result.passages if not p.clears_threshold]
    assert cleared and uncleared, "fixture self-check: the top-8 must straddle the threshold"
    for passage in result.passages:
        assert passage.clears_threshold is (passage.rerank_score >= 0.5)
    assert result.partial_support is True


def test_full_support_is_not_flagged_partial() -> None:
    """All passages at-or-above threshold -> partial_support False and
    every clears_threshold True — the flag means straddling, not merely
    'a threshold exists'."""
    client, model, _chunks = indexed_corpus()

    result = run_retrieve(
        client,
        decision(QUERY),
        model=model,
        reranker=TableReranker(default=0.8),
        cfg=config(refusal_threshold=0.5),
    )

    assert isinstance(result, RetrievedPassages)
    assert all(p.clears_threshold for p in result.passages)
    assert result.partial_support is False


def test_threshold_calibration_reproducible() -> None:
    """ADR-010: calibration is scripted and reproducible, never hand-tuned
    magic. Identical score inputs -> an identical artifact; for
    separable inputs the threshold refuses every no-answer item
    (score < threshold) and passes every answerable one
    (score >= threshold); every consumed item id is recorded."""
    no_answer = {"gold-na-01": 0.12, "gold-na-02": 0.18, "gold-na-03": 0.09}
    answerable = {"gold-a-01": 0.55, "gold-a-02": 0.61}

    first = calibrate_refusal_threshold(no_answer, answerable)
    second = calibrate_refusal_threshold(dict(no_answer), dict(answerable))

    assert first == second, "identical inputs must yield an identical threshold artifact"
    assert isinstance(first, ThresholdCalibration)
    assert max(no_answer.values()) < first.threshold <= min(answerable.values())
    assert set(first.calibration_item_ids) == set(no_answer) | set(answerable)


def test_threshold_artifact_round_trips_as_json(tmp_path: Path) -> None:
    """The eval-derived artifact is the config source: save/load
    round-trips exactly, and the on-disk form is plain JSON a human (and
    the #21 harness) can audit."""
    calibration = calibrate_refusal_threshold(
        {"gold-na-01": 0.12, "gold-na-02": 0.18}, {"gold-a-01": 0.55}
    )
    path = tmp_path / "refusal-threshold.json"

    save_threshold_artifact(calibration, path)
    loaded = load_threshold_artifact(path)

    assert loaded == calibration
    on_disk = json.loads(path.read_text())
    assert isinstance(on_disk, dict), "the artifact must be an auditable JSON object"


def test_threshold_artifact_missing_or_malformed_refuses(tmp_path: Path) -> None:
    """No artifact, no threshold, no service: a missing or malformed
    artifact raises RetrievalError — the loader never invents a
    fallback value."""
    with pytest.raises(RetrievalError):
        load_threshold_artifact(tmp_path / "never-written.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json at all")
    with pytest.raises(RetrievalError):
        load_threshold_artifact(malformed)


def test_calibration_items_disjoint_from_gate_items() -> None:
    """DESIGN §6.1: threshold-calibration items and refusal-gate eval
    items share no ids — a threshold tuned on the gate's own items would
    make the >90%/<5% release gates circular. Overlap refuses loudly,
    naming the shared id; disjoint sets pass."""
    calibration_ids = ("gold-na-01", "gold-na-02", "gold-a-01")
    disjoint_gate_ids = ("gold-na-10", "gold-na-11", "gold-a-07")

    assert check_calibration_gate_split(calibration_ids, disjoint_gate_ids) is None

    overlapping_gate_ids = ("gold-na-10", "gold-na-02")
    with pytest.raises(CalibrationGateOverlapError) as excinfo:
        check_calibration_gate_split(calibration_ids, overlapping_gate_ids)
    assert "gold-na-02" in str(excinfo.value)


def test_refusal_threshold_is_injected_never_defaulted() -> None:
    """The threshold is eval-derived config (#20/#21), not a constant:
    RetrievalConfig requires it (and the coverage list) explicitly —
    there is no default to fall back on anywhere in the module."""
    fields = {f.name: f for f in dataclasses.fields(RetrievalConfig)}
    assert fields["refusal_threshold"].default is dataclasses.MISSING
    assert fields["refusal_threshold"].default_factory is dataclasses.MISSING
    assert fields["corpus_coverage"].default is dataclasses.MISSING

    with pytest.raises(TypeError):
        RetrievalConfig()  # type: ignore[call-arg]
