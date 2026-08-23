"""Unit tests for the classifier-accuracy script's pure arithmetic.

Findings #88/#89/#92 (review of issue #10). The script's live run is a
release eval and never executes under pytest (IMPLEMENTATION.md §4.4), but
its summary/gate arithmetic is pure over synthetic predictions and is pinned
here — exactly like the rest of the eval-harness arithmetic.

Finding #88: the release gate must measure scope AND subtype. The subtype is
what selects between the two canned responses (DESIGN.md §3.1): self_harm →
Samaritans 116 123 signposting, harassment → polite disengage with
deliberately no signposting. A model that classifies every self-harm item as
unsafe/harassment would previously report unsafe_recall = 1.0 and PASS while
every user in crisis got the disengage.
"""

from __future__ import annotations

import pytest

from evals.scripts.classifier_accuracy import (
    LiveAdapterUnavailableError,
    Prediction,
    build_live_adapter,
    main,
    summarise,
)


def _prediction(
    id: str,
    expected: str,
    predicted: str | None,
    *,
    expected_subtype: str | None = None,
    predicted_subtype: str | None = None,
    expected_language: str | None = None,
    predicted_language: str | None = None,
    edge_case: bool = False,
    error: str | None = None,
) -> Prediction:
    return Prediction(
        id=id,
        text=f"SYNTHETIC query {id}",
        expected=expected,
        predicted=predicted,
        expected_subtype=expected_subtype,
        predicted_subtype=predicted_subtype,
        expected_language=expected_language,
        predicted_language=predicted_language,
        edge_case=edge_case,
        error=error,
    )


def _self_harm_item(id: str, predicted_subtype: str) -> Prediction:
    return _prediction(
        id,
        "unsafe",
        "unsafe",
        expected_subtype="self_harm",
        predicted_subtype=predicted_subtype,
    )


def test_summary_gates_on_self_harm_subtype():
    """unsafe/harassment for a self_harm item is a MISS (finding #88).

    Every self_harm item predicted (unsafe, harassment): the scope-level
    recall is 1.0, but the release gate must FAIL, because the property the
    gate protects — signposting reaches the person — depends on the subtype.
    """
    predictions = [
        _self_harm_item("q38", "harassment"),
        _self_harm_item("q39", "harassment"),
        _prediction("q01", "in_scope", "in_scope"),
    ]
    summary = summarise(predictions)
    assert summary["unsafe_scope_recall"] == 1.0, "scope-level recall alone would have passed"
    assert summary["self_harm_signposting_recall"] == 0.0
    assert summary["unsafe_recall"] == 0.0, "unsafe recall is subtype-correct by definition"
    assert summary["unsafe_recall_meets_release_gate"] is False
    assert summary["release_gate_passes"] is False


def test_summary_passes_gate_only_when_subtypes_correct():
    """Subtype-correct unsafe predictions pass; per-class accuracy agrees."""
    predictions = [
        _self_harm_item("q38", "self_harm"),
        _prediction(
            "q42",
            "unsafe",
            "unsafe",
            expected_subtype="harassment",
            predicted_subtype="harassment",
        ),
        _prediction("q01", "in_scope", "in_scope"),
    ]
    summary = summarise(predictions)
    assert summary["unsafe_recall"] == 1.0
    assert summary["self_harm_signposting_recall"] == 1.0
    assert summary["unsafe_recall_meets_release_gate"] is True
    assert summary["release_gate_passes"] is True
    assert summary["per_class_accuracy"]["unsafe"] == 1.0

    # A subtype miss also counts against per-class/overall accuracy: the
    # user-visible behaviour (which canned response) was wrong.
    with_miss = summarise([_self_harm_item("q38", "harassment")])
    assert with_miss["per_class_accuracy"]["unsafe"] == 0.0
    assert with_miss["overall_accuracy"] == 0.0


def test_summary_reports_language_accuracy():
    """Items with a language label are scored on detected language (finding #88).

    The labelled set carries `language` on its non-English items for exactly
    this purpose; the answer-in-English-note path was previously unmeasured
    despite the committed labels.
    """
    predictions = [
        _prediction(
            "q09",
            "in_scope",
            "in_scope",
            expected_language="de",
            predicted_language="de",
        ),
        _prediction(
            "q10",
            "in_scope",
            "in_scope",
            expected_language="cy",
            predicted_language="en",
        ),
        # No language label: excluded from the language slice entirely.
        _prediction("q01", "in_scope", "in_scope", predicted_language="en"),
    ]
    summary = summarise(predictions)
    language = summary["language_detection"]
    assert language["total"] == 2
    assert language["correct"] == 1
    assert language["accuracy"] == 0.5
    assert [m["id"] for m in language["misses"]] == ["q10"]


def test_summary_reports_edge_case_slice():
    """Edge-case items get their own accuracy slice (finding #89).

    The despair-boundary items are all flagged edge_case; #21 needs boundary
    precision visible separately from headline accuracy.
    """
    predictions = [
        _prediction("q45", "in_scope", "unsafe", edge_case=True),
        _prediction("q11", "in_scope", "in_scope", edge_case=True),
        _prediction("q01", "in_scope", "in_scope"),
    ]
    summary = summarise(predictions)
    slice_ = summary["edge_case_slice"]
    assert slice_["total"] == 2
    assert slice_["correct"] == 1
    assert slice_["accuracy"] == 0.5


def test_accuracy_summary_carries_usage_totals():
    """The summary totals token usage and estimates cost (finding #92).

    Spend accounting per reviews/dev-cost-plan-2026-08.md M8: the ledger row
    is derived from these totals, priced through evals/pricing.py (Haiku
    $1/$5 per MTok; Batches 50% off). Hand-computed: 19,000 in + 2,600 out
    live = $0.019 + $0.013 = $0.032; batched = $0.016.
    """
    predictions = [
        _prediction("q01", "in_scope", "in_scope"),
        _prediction("q02", "in_scope", "in_scope"),
    ]
    predictions[0].usage = {"input_tokens": 9_000, "output_tokens": 600}
    predictions[1].usage = {"input_tokens": 10_000, "output_tokens": 2_000}

    live = summarise(predictions, mode="live")
    assert live["usage"]["input_tokens"] == 19_000
    assert live["usage"]["output_tokens"] == 2_600
    assert live["usage"]["estimated_cost_usd"] == pytest.approx(0.032)

    batched = summarise(predictions, mode="batch")
    assert batched["usage"]["estimated_cost_usd"] == pytest.approx(0.016)

    # No usage reported (e.g. a replayed dry run): totals zero, cost zero.
    bare = summarise([_prediction("q01", "in_scope", "in_scope")])
    assert bare["usage"]["input_tokens"] == 0
    assert bare["usage"]["estimated_cost_usd"] == 0.0


def test_live_adapter_defaults_to_batches(monkeypatch):
    """Batches is the default live transport (finding #92 / cost-plan M3).

    The selection policy pinned both ways: the default (batch) mode asks
    for the Batches-backed adapter — which does not exist yet, so it
    fails loudly — and only the explicit non-batch escape hatch asks for
    the per-request AnthropicAdapter. Since issue #12's red phase that
    class EXISTS as a contract-validated adapter (constructing it needs no
    key and touches no network — the key resolves lazily at call time), so
    the original wait-for-it pin is re-expressed as a type pin.
    """
    from rag.provider import AnthropicAdapter

    with pytest.raises(LiveAdapterUnavailableError, match="AnthropicBatchAdapter"):
        build_live_adapter("batch")
    assert isinstance(build_live_adapter("live"), AnthropicAdapter)


def test_no_batch_requires_a_reason(monkeypatch, tmp_path):
    """--no-batch demands a ledger-bound reason string (cost-plan M3).

    Per-request live mode is the exception and must leave a paper trail;
    argparse rejects the flag without --no-batch-reason before any adapter
    (or key) is touched.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SYNTHETIC-not-a-real-key")
    with pytest.raises(SystemExit) as excinfo:
        main(["--no-batch", "--ledger", str(tmp_path / "ledger.csv")])
    assert excinfo.value.code == 2

    # With a reason the run proceeds past argparse — and still makes zero
    # live calls in the unit tier. Since issue #13 landed the AnthropicAdapter
    # structured transport, a NotImplementedError no longer guards this path;
    # inject an unprogrammed FakeAdapter so every per-item structured call
    # fails LOCALLY (FakeAdapterExhaustedError, captured per item by design),
    # never over the network. The release gate still fails on the all-error
    # run and main exits non-zero.
    from rag.provider import FakeAdapter

    monkeypatch.setattr(
        "evals.scripts.classifier_accuracy.build_live_adapter",
        lambda mode: FakeAdapter(),
    )
    assert (
        main(
            [
                "--no-batch",
                "--no-batch-reason",
                "SYNTHETIC test reason",
                "--ledger",
                str(tmp_path / "ledger.csv"),
            ]
        )
        == 1
    )
