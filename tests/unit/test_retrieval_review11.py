"""Adversarial-review fixes for the #11 retrieval service — RED.

Review findings #172–#178 (review of PR #169). Each test block names the
finding it pins:

- **#172** — the refusal gate must fail CLOSED on a garbage signal:
  NaN/inf reranker scores, a wrong-length score list, or a non-finite
  configured threshold are a typed refusal-of-the-run
  (:class:`RetrievalError`), never a served answer. "An answered result
  implies the top passage cleared the threshold" is an enforced
  invariant, not an accident of finite arithmetic.
- **#173** — `load_threshold_artifact` validates hard: the threshold
  must be a finite non-bool number strictly inside (0, 1), the item ids
  a JSON array of strings, and the artifact must carry the expected
  schema-version field. Tampered or malformed artifacts refuse loudly;
  nothing is coerced (no bool->1.0, no string->float, no string
  shredded into characters).

Unit tier: in-memory Qdrant + deterministic fakes behind the `Reranker`
seam (IMPLEMENTATION.md §1) — no weights, no network.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag.retrieval import (
    HonestRefusal,
    RetrievalError,
    RetrievedPassages,
    calibrate_refusal_threshold,
    load_threshold_artifact,
    save_threshold_artifact,
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


# ---------------------------------------------------------------------------
# Finding #173 — the threshold-artifact loader refuses tampered content.
# ---------------------------------------------------------------------------


def _write_artifact(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "refusal-threshold.json"
    path.write_text(content)
    return path


def _artifact_json(
    threshold: object = 0.42,
    item_ids: object = None,
    schema_version: object = "KEEP",
) -> str:
    """A hand-built artifact document (raw JSON text, so non-standard
    literals like NaN/Infinity can be expressed)."""
    ids = item_ids if item_ids is not None else ["gold-na-01", "gold-a-01"]
    document: dict[str, object] = {
        "threshold": threshold,
        "calibration_item_ids": ids,
    }
    if schema_version == "KEEP":
        # Whatever schema marker save_threshold_artifact writes today.
        reference = json.loads(_saved_reference_artifact())
        for key, value in reference.items():
            if key not in ("threshold", "calibration_item_ids"):
                document[key] = value
    elif schema_version is not None:
        document["schema_version"] = schema_version
    return json.dumps(document)


def _saved_reference_artifact() -> str:
    """The exact on-disk form save_threshold_artifact writes for a real
    calibration — the shape every tampering below deviates from."""
    import tempfile

    calibration = calibrate_refusal_threshold({"gold-na-01": 0.12}, {"gold-a-01": 0.55})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reference.json"
        save_threshold_artifact(calibration, path)
        return path.read_text()


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_threshold_artifact_rejects_non_finite_threshold(tmp_path: Path, literal: str) -> None:
    """Finding #173: json.loads accepts the non-standard NaN/Infinity
    literals by default and float() passes them through — a NaN
    threshold silently kills the refusal gate forever. Non-finite
    thresholds must refuse loudly at load."""
    raw = _artifact_json().replace("0.42", literal)
    path = _write_artifact(tmp_path, raw)
    with pytest.raises(RetrievalError):
        load_threshold_artifact(path)


@pytest.mark.parametrize("value", [-5.0, 0.0, 1.0, 100, -0.001, 1.0001])
def test_threshold_artifact_rejects_out_of_scale_threshold(tmp_path: Path, value) -> None:
    """Finding #173: reranker scores are sigmoid outputs strictly inside
    (0, 1) — a threshold at or outside those bounds can never have come
    from calibrate_refusal_threshold over real scores, and any value
    <= 0 disables refusal outright. Out-of-scale thresholds refuse."""
    path = _write_artifact(tmp_path, _artifact_json(threshold=value))
    with pytest.raises(RetrievalError):
        load_threshold_artifact(path)


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param({"threshold": "0.99"}, id="string-threshold"),
        pytest.param({"threshold": True}, id="bool-threshold"),
        pytest.param({"threshold": None}, id="null-threshold"),
        pytest.param({"item_ids": "gold-na-01"}, id="string-item-ids-shredded"),
        pytest.param({"item_ids": 7}, id="numeric-item-ids"),
        pytest.param({"item_ids": ["gold-na-01", 2]}, id="non-string-id-in-list"),
        pytest.param({"item_ids": [["gold-na-01"]]}, id="nested-list-id"),
    ],
)
def test_threshold_artifact_rejects_wrong_types(tmp_path: Path, tamper: dict) -> None:
    """Finding #173: no silent coercion. A string threshold, a bool
    threshold (float(True) == 1.0), or calibration_item_ids that are
    not a JSON array of strings — tuple('gold-na-01') shreds the id
    into single characters, silently defeating the §6.1 disjointness
    guard — all refuse with the typed RetrievalError."""
    kwargs: dict[str, object] = {}
    if "threshold" in tamper:
        kwargs["threshold"] = tamper["threshold"]
    if "item_ids" in tamper:
        kwargs["item_ids"] = tamper["item_ids"]
    path = _write_artifact(tmp_path, _artifact_json(**kwargs))
    with pytest.raises(RetrievalError):
        load_threshold_artifact(path)


def test_threshold_artifact_rejects_non_object_document(tmp_path: Path) -> None:
    """Finding #173: the artifact must be a JSON object — a bare list or
    scalar refuses instead of dying on an untyped lookup."""
    for content in ("[0.42]", '"0.42"', "0.42"):
        path = _write_artifact(tmp_path, content)
        with pytest.raises(RetrievalError):
            load_threshold_artifact(path)


def test_threshold_artifact_requires_schema_version(tmp_path: Path) -> None:
    """Finding #173: the artifact carries an explicit schema-version
    field, written by save_threshold_artifact and required by the
    loader — a document missing it (hand-built, or from a different
    tool) refuses rather than being guessed at; so does an unknown
    version."""
    saved = json.loads(_saved_reference_artifact())
    version_keys = [k for k in saved if k not in ("threshold", "calibration_item_ids")]
    assert version_keys, (
        "save_threshold_artifact must write a schema/version field "
        "alongside threshold and calibration_item_ids"
    )

    path = _write_artifact(tmp_path, _artifact_json(schema_version=None))
    with pytest.raises(RetrievalError):
        load_threshold_artifact(path)

    path = _write_artifact(tmp_path, _artifact_json(schema_version=999))
    with pytest.raises(RetrievalError):
        load_threshold_artifact(path)


def test_threshold_artifact_still_round_trips_real_calibrations(tmp_path: Path) -> None:
    """Finding #173 (no behaviour change for honest artifacts): what
    save_threshold_artifact writes from a real calibration still loads
    back exactly."""
    calibration = calibrate_refusal_threshold(
        {"gold-na-01": 0.12, "gold-na-02": 0.18}, {"gold-a-01": 0.55}
    )
    path = tmp_path / "refusal-threshold.json"
    save_threshold_artifact(calibration, path)
    assert load_threshold_artifact(path) == calibration
