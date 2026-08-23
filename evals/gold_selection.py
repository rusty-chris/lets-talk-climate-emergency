"""Route-aware selection of no-answer gold items (review finding #192).

The gold set's no-answer items exercise two different refusal
mechanisms, recorded per item as ``expected_route``:

- ``canned_out_of_scope`` — the classifier labels the query
  ``out_of_scope`` and ``rag/query.py::route_classification`` returns
  the canned decline. Retrieval never runs, so no reranker score
  exists. These items are covered by the classifier's own labelled-set
  gate (``tests/fixtures/classifier/labelled_queries.yaml``); in the
  end-to-end no-answer run they are checked to produce the canned
  decline, reported separately from the reranker gate.
- ``retrieval_refusal`` — in-scope climate material today's corpus
  cannot answer. Retrieval runs and the reranker refusal gate
  (``rag/retrieval.py::calibrate_refusal_threshold`` + the DESIGN §6.2
  >90% release gate) must fire.

The threshold calibration and the release refusal gate consume ONLY
``retrieval_refusal`` items — calibrating on (or gating with) queries
the classifier diverts would certify a path production never takes.
This module is the single selection seam: the #21 harness and the
gold-set meta-tests both draw calibration/gate item ids from here, so
the filter cannot silently diverge between them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIMATE_QA_PATH = REPO_ROOT / "evals" / "gold" / "climate_qa.yaml"

CANNED_OUT_OF_SCOPE = "canned_out_of_scope"
RETRIEVAL_REFUSAL = "retrieval_refusal"
EXPECTED_ROUTES = (CANNED_OUT_OF_SCOPE, RETRIEVAL_REFUSAL)
NO_ANSWER_SUBSETS = ("calibration", "gate")


class GoldSelectionError(ValueError):
    """A no-answer gold item is missing or mis-declares its routing
    metadata — refused loudly rather than silently mis-selected."""


def load_climate_qa_items(path: Path = CLIMATE_QA_PATH) -> list[dict[str, Any]]:
    """The committed climate-QA gold items, as plain dicts."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))["items"]


def _no_answer_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in items:
        if item.get("category") != "no_answer":
            continue
        item_id = item.get("id", "<missing id>")
        subset = item.get("subset")
        if subset not in NO_ANSWER_SUBSETS:
            raise GoldSelectionError(
                f"no_answer gold item {item_id!r} carries subset {subset!r}; "
                f"expected one of {NO_ANSWER_SUBSETS} (DESIGN §6.1)"
            )
        route = item.get("expected_route")
        if route not in EXPECTED_ROUTES:
            raise GoldSelectionError(
                f"no_answer gold item {item_id!r} carries expected_route "
                f"{route!r}; every no-answer item must declare one of "
                f"{EXPECTED_ROUTES} (review finding #192)"
            )
        selected.append(item)
    return selected


def _retrieval_refusal_ids(items: Iterable[dict[str, Any]], subset: str) -> tuple[str, ...]:
    return tuple(
        item["id"]
        for item in _no_answer_items(items)
        if item["subset"] == subset and item["expected_route"] == RETRIEVAL_REFUSAL
    )


def calibration_item_ids(items: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Ids of the items the reranker threshold calibration consumes:
    ``subset: calibration`` AND ``expected_route: retrieval_refusal``,
    in gold-file order. Canned out-of-scope items never appear — they
    produce no reranker score to calibrate on (finding #192)."""
    return _retrieval_refusal_ids(items, "calibration")


def gate_item_ids(items: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Ids of the items the DESIGN §6.2 >90% refusal release gate
    counts: ``subset: gate`` AND ``expected_route: retrieval_refusal``,
    in gold-file order. Canned out-of-scope items never appear — their
    decline is the classifier's, gated by the labelled query set, not
    by the reranker threshold this gate certifies (finding #192)."""
    return _retrieval_refusal_ids(items, "gate")
