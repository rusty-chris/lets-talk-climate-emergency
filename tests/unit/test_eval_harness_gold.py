"""Issue #21 red phase: gold-set loading + validation contract.

The harness must refuse to run against malformed or missing gold — a
release eval scored against bad gold is worse than no eval. The
committed gold sets must load and validate as-is (the fixed #192/#193
route semantics, the n≥20 gate arithmetic, chart spec/refusal
exclusivity); every refusal case is exercised on SYNTHETIC gold files
(tests/_eval_harness_fixtures.py), never by corrupting the real ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import gold_selection
from evals.harness import (
    CHART_REQUESTS_PATH,
    CLIMATE_QA_PATH,
    MIN_REFUSAL_GATE_ITEMS,
    GoldValidationError,
    load_and_validate_gold,
)
from tests._eval_harness_fixtures import (
    synthetic_chart_items,
    synthetic_qa_items,
    write_synthetic_gold,
)


def test_real_gold_sets_load_and_validate():
    """The committed evals/gold files ARE the harness's input: they must
    load, carry the fixed route vocabulary, and satisfy the gate
    arithmetic (gate ∩ retrieval_refusal ≥ 20, disjoint from
    calibration; 15 chart items)."""
    gold = load_and_validate_gold()
    assert len(gold.qa_items) >= 75
    assert len(gold.chart_items) == 15
    assert len(gold.refusal_gate_ids) >= MIN_REFUSAL_GATE_ITEMS
    assert not set(gold.refusal_gate_ids) & set(gold.refusal_calibration_ids)
    # The selection seam is gold_selection — the harness must not
    # reimplement the filter (finding #192: one seam, no divergence).
    items = gold_selection.load_climate_qa_items(CLIMATE_QA_PATH)
    assert gold.refusal_gate_ids == gold_selection.gate_item_ids(items)
    assert gold.refusal_calibration_ids == gold_selection.calibration_item_ids(items)
    # canned_out_of_scope ids are selected too — measured separately,
    # never inside the refusal gate subset.
    assert gold.canned_out_of_scope_ids
    assert not set(gold.canned_out_of_scope_ids) & set(gold.refusal_gate_ids)
    assert CHART_REQUESTS_PATH.name == "chart_requests.yaml"


def test_missing_gold_file_refused(tmp_path: Path):
    qa_path, charts_path = write_synthetic_gold(tmp_path)
    missing = tmp_path / "nowhere" / "climate_qa.yaml"
    with pytest.raises(GoldValidationError) as excinfo:
        load_and_validate_gold(missing, charts_path)
    assert "climate_qa.yaml" in str(excinfo.value)
    with pytest.raises(GoldValidationError):
        load_and_validate_gold(qa_path, tmp_path / "nowhere" / "chart_requests.yaml")


def test_malformed_route_vocabulary_refused(tmp_path: Path):
    """A no_answer item with an unknown expected_route is a hard refusal
    (the #192 route vocabulary is closed)."""
    items = synthetic_qa_items(bad_route_on="syn-na-g-01")
    qa_path, charts_path = write_synthetic_gold(tmp_path, qa_items=items)
    with pytest.raises(GoldValidationError) as excinfo:
        load_and_validate_gold(qa_path, charts_path)
    assert "syn-na-g-01" in str(excinfo.value)


def test_gate_subset_arithmetic_requires_twenty_items(tmp_path: Path):
    """19 gate ∩ retrieval_refusal items fail the n≥20 arithmetic — a
    shrunken gate subset cannot survive a single flake under the strict
    >90% gate (#193), so loading refuses."""
    items = synthetic_qa_items(gate_refusal_items=19)
    qa_path, charts_path = write_synthetic_gold(tmp_path, qa_items=items)
    with pytest.raises(GoldValidationError) as excinfo:
        load_and_validate_gold(qa_path, charts_path)
    assert "20" in str(excinfo.value)

    # Exactly 20 satisfies the arithmetic.
    qa_path, charts_path = write_synthetic_gold(
        tmp_path / "ok", qa_items=synthetic_qa_items(gate_refusal_items=20)
    )
    gold = load_and_validate_gold(qa_path, charts_path)
    assert len(gold.refusal_gate_ids) == 20
    # canned_out_of_scope items are never in the gate subset (fixed
    # #192/#193 semantics), but are selected for their separate check.
    assert all(item_id.startswith("syn-na-g-") for item_id in gold.refusal_gate_ids)
    assert set(gold.canned_out_of_scope_ids) == {"syn-na-oos-01", "syn-na-oos-02"}


def test_multi_passage_requires_recall_semantics(tmp_path: Path):
    """Finding #196: recall semantics are declared per item; the harness
    has NO default — a multi_passage item without the declaration is
    malformed gold."""
    items = synthetic_qa_items(drop_recall_semantics=True)
    qa_path, charts_path = write_synthetic_gold(tmp_path, qa_items=items)
    with pytest.raises(GoldValidationError) as excinfo:
        load_and_validate_gold(qa_path, charts_path)
    assert "recall_semantics" in str(excinfo.value)


def test_chart_item_requires_declared_behaviour(tmp_path: Path):
    """A chart item whose expected behaviour payload is missing (spec
    item without a spec / refusal item without a refusal) is malformed."""
    charts = synthetic_chart_items(strip_behaviour_on="syn-chart-01")
    qa_path, charts_path = write_synthetic_gold(tmp_path, chart_items=charts)
    with pytest.raises(GoldValidationError) as excinfo:
        load_and_validate_gold(qa_path, charts_path)
    assert "syn-chart-01" in str(excinfo.value)
