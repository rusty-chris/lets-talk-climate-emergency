"""Route-aware selection of no-answer gold items (review finding #192).

The reranker-calibration and refusal-gate machinery must consume ONLY
items annotated ``expected_route: retrieval_refusal`` — an item the
classifier diverts to a canned out-of-scope decline never produces a
reranker score, so feeding it to `calibrate_refusal_threshold` (or
counting it in the >90% refusal gate) certifies a path production never
takes. `evals.gold_selection` is the single selection seam the #21
harness consumes; these tests pin its contract against both synthetic
items and the committed gold set.
"""

from __future__ import annotations

import pytest

from evals import gold_selection


def _item(item_id: str, subset: str, route: str | None) -> dict:
    body = {
        "id": item_id,
        "category": "no_answer",
        "question": f"synthetic question {item_id}",
        "expected_behaviour": "refusal",
        "subset": subset,
    }
    if route is not None:
        body["expected_route"] = route
    return body


SYNTHETIC_ITEMS = [
    _item("na-c-1", "calibration", "retrieval_refusal"),
    _item("na-c-2", "calibration", "canned_out_of_scope"),
    _item("na-c-3", "calibration", "retrieval_refusal"),
    _item("na-g-1", "gate", "canned_out_of_scope"),
    _item("na-g-2", "gate", "retrieval_refusal"),
    {
        "id": "sp-1",
        "category": "single_passage",
        "question": "an answerable question",
        "expected_behaviour": "answer",
        "gold_chunk_ids": ["doc:abc"],
    },
]


def test_calibration_selection_is_route_filtered_and_order_preserving():
    assert gold_selection.calibration_item_ids(SYNTHETIC_ITEMS) == ("na-c-1", "na-c-3")


def test_gate_selection_is_route_filtered():
    assert gold_selection.gate_item_ids(SYNTHETIC_ITEMS) == ("na-g-2",)


def test_selection_refuses_missing_route():
    items = [*SYNTHETIC_ITEMS, _item("na-c-4", "calibration", None)]
    with pytest.raises(gold_selection.GoldSelectionError) as excinfo:
        gold_selection.calibration_item_ids(items)
    assert "na-c-4" in str(excinfo.value)
    assert "expected_route" in str(excinfo.value)


def test_selection_refuses_unknown_route():
    items = [*SYNTHETIC_ITEMS, _item("na-g-3", "gate", "reranker_refusal")]
    with pytest.raises(gold_selection.GoldSelectionError) as excinfo:
        gold_selection.gate_item_ids(items)
    assert "na-g-3" in str(excinfo.value)


def test_selection_refuses_missing_subset():
    broken = _item("na-x-1", "calibration", "retrieval_refusal")
    del broken["subset"]
    with pytest.raises(gold_selection.GoldSelectionError) as excinfo:
        gold_selection.calibration_item_ids([*SYNTHETIC_ITEMS, broken])
    assert "na-x-1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Against the committed gold set: calibration consumes only items that
# reach retrieval (finding #192's test_calibration_items_reach_retrieval).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold_items() -> list[dict]:
    return gold_selection.load_climate_qa_items()


def test_calibration_items_reach_retrieval(gold_items):
    by_id = {item["id"]: item for item in gold_items}
    selected = gold_selection.calibration_item_ids(gold_items)
    assert selected, "the calibration selection over the gold set must be non-empty"
    for item_id in selected:
        item = by_id[item_id]
        assert item["expected_route"] == "retrieval_refusal", item_id
        assert item["subset"] == "calibration", item_id


def test_gate_items_reach_retrieval(gold_items):
    by_id = {item["id"]: item for item in gold_items}
    selected = gold_selection.gate_item_ids(gold_items)
    assert selected, "the gate selection over the gold set must be non-empty"
    for item_id in selected:
        item = by_id[item_id]
        assert item["expected_route"] == "retrieval_refusal", item_id
        assert item["subset"] == "gate", item_id


def test_canned_items_never_selected(gold_items):
    canned = {
        item["id"]
        for item in gold_items
        if item["category"] == "no_answer" and item.get("expected_route") == "canned_out_of_scope"
    }
    selected = set(gold_selection.calibration_item_ids(gold_items)) | set(
        gold_selection.gate_item_ids(gold_items)
    )
    assert canned, "the gold set records its canned out-of-scope no-answer items"
    assert selected.isdisjoint(canned)
