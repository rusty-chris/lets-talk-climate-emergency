"""Shared synthetic fixtures for the issue #21 eval-harness tests.

SYNTHETIC FIXTURE — authored for this project's tests. Everything here
is invented: the gold items are tiny stand-ins with the same schema
vocabulary as the real evals/gold files (routes, subsets, categories),
never copies of them; the batch client is a Message-Batches-shaped
double so no judge test can reach the network (IMPLEMENTATION.md §4.4).

The helpers write small synthetic gold YAML files into a test-provided
directory — the harness's gold-loading tests exercise malformed and
undersized variants without ever touching the committed gold sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

SYNTHETIC_MARKER = "SYNTHETIC GOLD SET — authored for this project's tests"


def production_passage_payload(
    chunk_id: str,
    *,
    source_type: str = "report",
    consensus_position: str = "assessed",
    body: str = "synthetic passage",
) -> dict[str, Any]:
    """One stored chunk payload in the PRODUCTION shape `build_index`
    writes (§2.4 metadata + the #143 parse-provenance flags) — the shape
    `rag.generation.build_generation_request` requires (review #21,
    issue #234: the eval fixtures must carry production-shape payloads;
    the builder is never weakened to fit thinner fixtures)."""
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split(":", 1)[0],
        "section_path": ["1 Synthetic section"],
        "context_header": "Synthetic Assessment (invented) > 1 Synthetic section",
        "body": body,
        "token_count": 12,
        "confidence_markers": ["very likely"],
        "block_types": ["text"],
        "consensus_position": consensus_position,
        "source_type": source_type,
        "citation_metadata": {
            "licence": "CC-BY-4.0",
            "attribution_text": "Synthetic Assessment Cycle 1 (invented)",
            "canonical_url": "https://example.invalid/synthetic-assessment",
            "permitted_context": "public-noncommercial",
        },
        "parse_backend": "docling",
        "degraded_fallback": False,
        "needs_hand_review": False,
        "content_hash": f"synthetic-hash-{chunk_id}",
    }


def synthetic_qa_items(
    *,
    gate_refusal_items: int = 20,
    calibration_refusal_items: int = 2,
    canned_items: int = 2,
    bad_route_on: str | None = None,
    drop_recall_semantics: bool = False,
) -> list[dict[str, Any]]:
    """A minimal, schema-shaped climate-QA item list.

    ``gate_refusal_items`` controls the ``gate`` ∩ ``retrieval_refusal``
    count (the n≥20 arithmetic under test). ``bad_route_on`` swaps one
    named item's expected_route for an unknown token; ``drop_recall_semantics``
    removes the multi-passage declaration (finding #196).
    """
    items: list[dict[str, Any]] = [
        {
            "id": "syn-sp-01",
            "category": "single_passage",
            "question": "How warm is the synthetic planet?",
            "expected_behaviour": "answer",
            "gold_chunk_ids": ["syn_doc:0001"],
        },
        {
            "id": "syn-mp-01",
            "category": "multi_passage",
            "question": "Synthesise two synthetic passages.",
            "expected_behaviour": "answer",
            "gold_chunk_ids": ["syn_doc:0002", "syn_doc:0003"],
            "recall_semantics": "all_gold",
        },
        {
            "id": "syn-sev-01",
            "category": "severity",
            "question": "Is the synthetic situation basically fine?",
            "expected_behaviour": "answer",
            "gold_chunk_ids": ["syn_doc:0004"],
            "severity": {
                "expected_lead": "serious",
                "bait": "soft-pedal",
                "source": {"chunk_id": "syn_doc:0004", "quote": "synthetic passage"},
            },
        },
    ]
    if drop_recall_semantics:
        del items[1]["recall_semantics"]
    for index in range(gate_refusal_items):
        items.append(
            {
                "id": f"syn-na-g-{index + 1:02d}",
                "category": "no_answer",
                "question": f"Synthetic unanswerable gate question {index + 1}?",
                "expected_behaviour": "refusal",
                "subset": "gate",
                "expected_route": "retrieval_refusal",
            }
        )
    for index in range(calibration_refusal_items):
        items.append(
            {
                "id": f"syn-na-c-{index + 1:02d}",
                "category": "no_answer",
                "question": f"Synthetic unanswerable calibration question {index + 1}?",
                "expected_behaviour": "refusal",
                "subset": "calibration",
                "expected_route": "retrieval_refusal",
            }
        )
    for index in range(canned_items):
        items.append(
            {
                "id": f"syn-na-oos-{index + 1:02d}",
                "category": "no_answer",
                "question": f"Synthetic out-of-scope question {index + 1}?",
                "expected_behaviour": "refusal",
                "subset": "gate" if index % 2 == 0 else "calibration",
                "expected_route": "canned_out_of_scope",
            }
        )
    if bad_route_on is not None:
        for item in items:
            if item["id"] == bad_route_on:
                item["expected_route"] = "definitely_not_a_route"
                break
        else:
            raise AssertionError(f"no synthetic item {bad_route_on!r} to corrupt")
    return items


def synthetic_chart_items(
    *,
    include_blocked_flagship: bool = True,
    strip_behaviour_on: str | None = None,
) -> list[dict[str, Any]]:
    """A minimal chart gold list: one spec item, one refusal item and
    (by default) one blocked flagship-shaped item (#23/#117)."""
    items: list[dict[str, Any]] = [
        {
            "id": "syn-chart-01",
            "request": "Plot the synthetic series",
            "expected": "spec",
            "manifest": "synthetic",
            "fixture": "syn_fixture_01",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        {
            "id": "syn-chart-02",
            "request": "Plot data we do not have",
            "expected": "refusal",
            "manifest": "synthetic",
            "refusal": {
                "requested_data": "synthetic missing series",
                "nearest_dataset_first": "syn_temp_modern",
                "curation_gap_logged": True,
            },
        },
    ]
    if include_blocked_flagship:
        items.append(
            {
                "id": "syn-chart-flagship",
                "request": "The synthetic flagship request",
                "expected": "refusal",
                "manifest": "real",
                "refusal": {
                    "reason_names_issue": "issue #23",
                    "blocked_pairs": ["syn_pair"],
                    "curation_gap_logged": True,
                },
                "blocked_on": "issue-23-licence-confirmations",
                "blocked_reason": "synthetic stand-in for the #117 fixture embargo",
            }
        )
    if strip_behaviour_on is not None:
        for item in items:
            if item["id"] == strip_behaviour_on:
                item.pop("spec", None)
                item.pop("refusal", None)
                break
        else:
            raise AssertionError(f"no synthetic chart item {strip_behaviour_on!r}")
    return items


def write_synthetic_gold(
    directory: Path,
    *,
    qa_items: Sequence[Mapping[str, Any]] | None = None,
    chart_items: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write small synthetic gold YAML files; returns (qa_path, charts_path)."""
    directory.mkdir(parents=True, exist_ok=True)
    qa_path = directory / "climate_qa.yaml"
    charts_path = directory / "chart_requests.yaml"
    qa_payload = {
        "marker": SYNTHETIC_MARKER,
        "corpus_documents": ["syn_doc"],
        "items": [
            dict(item) for item in (qa_items if qa_items is not None else synthetic_qa_items())
        ],
    }
    chart_payload = {
        "marker": SYNTHETIC_MARKER,
        "items": [
            dict(item)
            for item in (chart_items if chart_items is not None else synthetic_chart_items())
        ],
    }
    qa_path.write_text(yaml.safe_dump(qa_payload, sort_keys=False), encoding="utf-8")
    charts_path.write_text(yaml.safe_dump(chart_payload, sort_keys=False), encoding="utf-8")
    return qa_path, charts_path


# ---------------------------------------------------------------------------
# Message-Batches-shaped double (judges tests)
# ---------------------------------------------------------------------------


def succeeded_batch_result(custom_id: str, verdict_json_text: str) -> SimpleNamespace:
    """A `succeeded` batch result whose message carries one text block."""
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=[SimpleNamespace(type="text", text=verdict_json_text)]),
        ),
    )


def failed_batch_result(custom_id: str, result_type: str = "errored") -> SimpleNamespace:
    """An errored/canceled/expired batch result."""
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type))


class FakeBatchClient:
    """A double mirroring ``client.messages.batches``: create/retrieve/results.

    Records every ``create`` call (the batched-not-per-item invariant);
    serves the programmed results in whatever order they were given —
    collectors must key on custom_id, never on position.

    ``ops`` is the ordered method log (issue #236: collection must poll
    ``retrieve`` until the batch is terminal BEFORE calling ``results``).
    ``program_processing_statuses`` queues the status each successive
    ``retrieve`` reports; once exhausted (or when nothing is programmed,
    the pre-#236 default) every ``retrieve`` reports ``ended``.
    """

    def __init__(self, results: Iterable[Any] = ()) -> None:
        self.create_calls: list[list[Any]] = []
        self.retrieve_calls: list[str] = []
        self.ops: list[str] = []
        self._results = list(results)
        self._batch_id = "synbatch_001"
        self._processing_statuses: list[str] = []

    def program_results(self, results: Iterable[Any]) -> None:
        self._results = list(results)

    def program_processing_statuses(self, statuses: Iterable[str]) -> None:
        self._processing_statuses = list(statuses)

    def create(self, *, requests: Sequence[Any]) -> SimpleNamespace:
        self.ops.append("create")
        self.create_calls.append(list(requests))
        return SimpleNamespace(id=self._batch_id, processing_status="in_progress")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        assert batch_id == self._batch_id, f"unknown batch {batch_id!r}"
        self.ops.append("retrieve")
        self.retrieve_calls.append(batch_id)
        if self._processing_statuses:
            status = self._processing_statuses.pop(0)
        else:
            status = "ended"
        return SimpleNamespace(id=batch_id, processing_status=status)

    def results(self, batch_id: str) -> Iterable[Any]:
        assert batch_id == self._batch_id, f"unknown batch {batch_id!r}"
        self.ops.append("results")
        return iter(self._results)
