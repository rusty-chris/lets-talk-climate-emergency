"""Issue #351 red phase (Fable): the judge collector folds unscored items
silently and returns no usage (recurring — runs 3 AND 4).

Run 4: the batch collector folded 2/160 verdicts unscored — qa-sev-14
severity_fidelity ("succeeded result carried no text block", now
gate-adjacent: an unscored severity verdict counts AGAINST the >=90%
severity gate) and qa-va-01 faithfulness ("malformed verdict: not valid
JSON") — and returned usage: null on EVERY verdict, scored ones included,
so the run again needed a manual true-usage correction ledger row
(judge_usage_correction.json: 161,468 in / 11,476 out, $0.328272). Root
cause of the usage loss: the collector reads ``result.usage``, but the
live Batches API carries usage on ``result.message.usage`` — so every
per-verdict usage silently reads None. Three red prongs per the issue:

1. the collector surfaces per-item unscored REASONS in the results
   payload (parse failure vs missing result vs errored request), never a
   silent fold;
2. gate-adjacent unscored severity items trigger ONE targeted re-judge
   (a single live call, pre-flighted, ledgered) before the gate computes;
3. the collector extracts and returns the batch's REAL usage for the
   ledger — no manual correction rows.

FLAGGED CONTRACT (test-author decisions):

- New seams (stubs in evals/judges.py, repo red-phase convention):
  ``JudgeCollection`` + ``collect_judge_batch`` (the verdict fold of
  collect_judge_verdicts plus ``unscored`` reason entries and the batch
  ``usage`` totals) and ``rejudge_gate_adjacent_severity``.
  ``collect_judge_verdicts``'s dict return keeps its shape for existing
  callers; the batch collection is the new ledger-ready wrapper.
- Per-verdict usage is read from ``entry.result.message.usage`` (the live
  shape) for every succeeded result INCLUDING succeeded-but-malformed
  ones — those were billed; errored/missing results carry none. The
  batch total sums exactly those.
- "Gate-adjacent" = any unscored ``severity_fidelity`` verdict: severity
  verdicts are gate evidence and unscored always counts against the
  gate, so every severity fold-out is worth exactly ONE targeted live
  call (single attempt, no retry loop; fail-to-unscored preserved).
  Non-severity unscored verdicts are never re-judged.
- The re-judge is a spend seam: pre-flight discipline identical to
  submit_judge_batch (LiveRunRefusedError without one,
  BudgetExceededError on a failing one, zero adapter calls either way),
  and the replacement verdict carries the live call's usage so the run
  ledgers it.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evals.harness import BudgetExceededError, BudgetPreflight, LiveRunRefusedError
from evals.judges import (
    _JUDGE_MAX_TOKENS,
    JudgeRequest,
    JudgeVerdict,
    collect_judge_batch,
    collect_judge_verdicts,
    rejudge_gate_adjacent_severity,
    severity_records_from_verdicts,
    submit_judge_batch,
)
from rag.provider import AnswerWithCitations, FakeAdapter
from tests._eval_harness_fixtures import FakeBatchClient

ALLOWED_PREFLIGHT = BudgetPreflight(estimated_cost_usd=0.01, cumulative_usd=1.0, allowed=True)
REFUSED_PREFLIGHT = BudgetPreflight(estimated_cost_usd=5.0, cumulative_usd=8.9, allowed=False)

ARM = "claude-haiku-4-5"
JUDGE_MODEL = "claude-sonnet-5"

#: The run-4 fold-out custom ids, verbatim shape ({arm}__{kind}__{item}).
SEV_14_ID = f"{ARM}__severity_fidelity__qa-sev-14"
VA_01_ID = f"{ARM}__faithfulness__qa-va-01"
SEV_OK_ID = f"{ARM}__severity_fidelity__qa-sev-01"


def _request(custom_id: str, kind: str, item_id: str) -> JudgeRequest:
    return JudgeRequest(
        custom_id=custom_id,
        kind=kind,
        item_id=item_id,
        judge_model=JUDGE_MODEL,
        prompt=f"SYNTHETIC judge prompt for {item_id}",
        schema={"type": "object"},
    )


#: Per-result usages, distinct so mis-attribution shows in the totals.
USAGE_OK = {
    "input_tokens": 900,
    "output_tokens": 60,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}
USAGE_NO_TEXT = {
    "input_tokens": 1100,
    "output_tokens": 4,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}
USAGE_BAD_JSON = {
    "input_tokens": 1300,
    "output_tokens": 90,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


def live_shaped_result(
    custom_id: str,
    *,
    text: str | None,
    usage: dict[str, int],
) -> SimpleNamespace:
    """A ``succeeded`` batch result in the LIVE API shape: the message
    carries the content blocks AND the usage (``result.message.usage``);
    there is NO ``result.usage`` — the attribute the current collector
    reads. ``text=None`` programs the run-4 qa-sev-14 shape (a succeeded
    result whose content has no text block)."""
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=content, usage=dict(usage)),
        ),
    )


def errored_result(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


# ---------------------------------------------------------------------------
# Prong 3 — real usage, per verdict and per batch.
# ---------------------------------------------------------------------------


class TestCollectorUsageExtraction:
    def test_scored_verdict_usage_is_read_from_the_result_message(self) -> None:
        """Run 4 journalled usage: null on all 158 SCORED verdicts. The
        live Batches API puts usage on result.message.usage; the
        collector must read it there."""
        requests = [_request(SEV_OK_ID, "severity_fidelity", "qa-sev-01")]
        client = FakeBatchClient(
            results=[
                live_shaped_result(
                    SEV_OK_ID, text=json.dumps({"judged_lead": "serious"}), usage=USAGE_OK
                )
            ]
        )
        batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
        verdicts = collect_judge_verdicts(batch_id, requests, client)
        assert verdicts[SEV_OK_ID].scored is True
        assert verdicts[SEV_OK_ID].usage == USAGE_OK, (
            "per-verdict usage must come from result.message.usage (the live "
            "shape) — run 4 read result.usage and journalled None for every "
            "paid verdict"
        )

    def test_succeeded_but_unscored_verdicts_still_carry_their_paid_usage(self) -> None:
        """The two run-4 fold-outs were SUCCEEDED results — billed — and
        their usage vanished with the fold. Unscored never means unpaid."""
        requests = [
            _request(SEV_14_ID, "severity_fidelity", "qa-sev-14"),
            _request(VA_01_ID, "faithfulness", "qa-va-01"),
        ]
        client = FakeBatchClient(
            results=[
                live_shaped_result(SEV_14_ID, text=None, usage=USAGE_NO_TEXT),
                live_shaped_result(VA_01_ID, text="this is not json {", usage=USAGE_BAD_JSON),
            ]
        )
        batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
        verdicts = collect_judge_verdicts(batch_id, requests, client)
        assert verdicts[SEV_14_ID].scored is False
        assert verdicts[SEV_14_ID].usage == USAGE_NO_TEXT
        assert verdicts[VA_01_ID].scored is False
        assert verdicts[VA_01_ID].usage == USAGE_BAD_JSON


def _mixed_batch() -> tuple[list[JudgeRequest], FakeBatchClient]:
    """The run-4 shapes side by side: one scored, one succeeded-no-text,
    one succeeded-bad-JSON, one errored, one missing entirely."""
    requests = [
        _request(SEV_OK_ID, "severity_fidelity", "qa-sev-01"),
        _request(SEV_14_ID, "severity_fidelity", "qa-sev-14"),
        _request(VA_01_ID, "faithfulness", "qa-va-01"),
        _request(f"{ARM}__faithfulness__qa-sp-01", "faithfulness", "qa-sp-01"),
        _request(f"{ARM}__faithfulness__qa-sp-02", "faithfulness", "qa-sp-02"),
    ]
    client = FakeBatchClient(
        results=[
            live_shaped_result(
                SEV_OK_ID, text=json.dumps({"judged_lead": "serious"}), usage=USAGE_OK
            ),
            live_shaped_result(SEV_14_ID, text=None, usage=USAGE_NO_TEXT),
            live_shaped_result(VA_01_ID, text="this is not json {", usage=USAGE_BAD_JSON),
            errored_result(f"{ARM}__faithfulness__qa-sp-01"),
            # qa-sp-02 has no result at all.
        ]
    )
    return requests, client


class TestCollectJudgeBatch:
    def test_batch_usage_totals_are_the_ledger_numbers(self) -> None:
        """The batch total sums every returned result's real usage —
        scored AND succeeded-but-unscored (billed); errored/missing
        contribute nothing. This is the ledger row: no manual
        judge_usage_correction.json rows (runs 3 and 4 both needed one)."""
        requests, client = _mixed_batch()
        batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
        collection = collect_judge_batch(batch_id, requests, client)
        expected_totals = {
            key: USAGE_OK[key] + USAGE_NO_TEXT[key] + USAGE_BAD_JSON[key] for key in USAGE_OK
        }
        assert dict(collection.usage) == expected_totals

    def test_unscored_reasons_are_surfaced_never_silently_folded(self) -> None:
        """2/160 folded silently in run 4: the collection must name every
        fold-out with its distinct reason class — parse failure vs
        missing text block vs errored request vs missing result."""
        requests, client = _mixed_batch()
        batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
        collection = collect_judge_batch(batch_id, requests, client)

        unscored = {entry["custom_id"]: entry for entry in collection.unscored}
        assert set(unscored) == {
            SEV_14_ID,
            VA_01_ID,
            f"{ARM}__faithfulness__qa-sp-01",
            f"{ARM}__faithfulness__qa-sp-02",
        }, "every unscored verdict appears; no scored verdict does"
        for entry in unscored.values():
            assert entry["reason"], "every fold-out names its reason"
            assert entry["kind"] and entry["item_id"], "entries are self-describing"
        reasons = {custom_id: entry["reason"] for custom_id, entry in unscored.items()}
        assert reasons[SEV_14_ID] != reasons[VA_01_ID], (
            "a missing text block and malformed JSON are DIFFERENT failure "
            "classes; the payload must distinguish them"
        )
        assert (
            len(
                {
                    reasons[VA_01_ID],
                    reasons[f"{ARM}__faithfulness__qa-sp-01"],
                    reasons[f"{ARM}__faithfulness__qa-sp-02"],
                }
            )
            == 3
        ), "parse failure, errored request and missing result read differently"

    def test_verdict_fold_matches_collect_judge_verdicts(self) -> None:
        """The collection is the SAME fold plus ledger metadata — one
        fold, no second spelling (the #303 single-predicate lesson)."""
        requests, client = _mixed_batch()
        batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
        collection = collect_judge_batch(batch_id, requests, client)
        reference = collect_judge_verdicts(batch_id, requests, client)
        assert set(collection.verdicts) == set(reference)
        for custom_id, verdict in collection.verdicts.items():
            assert verdict.scored == reference[custom_id].scored, custom_id
            assert verdict.failure_reason == reference[custom_id].failure_reason, custom_id


# ---------------------------------------------------------------------------
# Prong 2 — the gate-adjacent severity re-judge: one pre-flighted live
# call, fake-pinned; fail-to-unscored preserved.
# ---------------------------------------------------------------------------


def _unscored(custom_id: str, kind: str, item_id: str, reason: str) -> JudgeVerdict:
    return JudgeVerdict(
        custom_id=custom_id,
        kind=kind,
        item_id=item_id,
        scored=False,
        verdict=None,
        failure_reason=reason,
    )


def _scored(custom_id: str, kind: str, item_id: str) -> JudgeVerdict:
    return JudgeVerdict(
        custom_id=custom_id,
        kind=kind,
        item_id=item_id,
        scored=True,
        verdict={"judged_lead": "serious"},
        usage=USAGE_OK,
    )


def _run4_verdicts() -> dict[str, JudgeVerdict]:
    return {
        SEV_OK_ID: _scored(SEV_OK_ID, "severity_fidelity", "qa-sev-01"),
        SEV_14_ID: _unscored(
            SEV_14_ID,
            "severity_fidelity",
            "qa-sev-14",
            "succeeded result carried no text block",
        ),
        VA_01_ID: _unscored(
            VA_01_ID, "faithfulness", "qa-va-01", "malformed verdict: not valid JSON"
        ),
    }


def _run4_requests() -> list[JudgeRequest]:
    return [
        _request(SEV_OK_ID, "severity_fidelity", "qa-sev-01"),
        _request(SEV_14_ID, "severity_fidelity", "qa-sev-14"),
        _request(VA_01_ID, "faithfulness", "qa-va-01"),
    ]


def _judge_answer(text: str) -> AnswerWithCitations:
    return AnswerWithCitations(
        text=text,
        citations=(),
        usage={"input_tokens": 1024, "output_tokens": 48},
    )


class TestSeverityRejudge:
    def test_one_unscored_severity_verdict_triggers_exactly_one_live_call(self) -> None:
        """Run 4's qa-sev-14: ONE targeted adapter.generate call, built
        from the original JudgeRequest (judge model, prompt, the batch
        path's max_tokens) — and the returned mapping carries the scored
        replacement WITH the call's usage (ledgered), the faithfulness
        fold-out untouched, the scored verdict untouched."""
        adapter = FakeAdapter(
            generate_results=[_judge_answer(json.dumps({"judged_lead": "emergency-level"}))]
        )
        rejudged = rejudge_gate_adjacent_severity(
            _run4_verdicts(), _run4_requests(), adapter, preflight=ALLOWED_PREFLIGHT
        )
        calls = adapter.calls_to("generate")
        assert len(calls) == 1, "ONE targeted live call, never a loop"
        (call,) = calls
        assert call.payload["config"]["model"] == JUDGE_MODEL
        assert call.payload["config"]["max_tokens"] == _JUDGE_MAX_TOKENS
        assert call.payload["messages"] == [
            {"role": "user", "content": "SYNTHETIC judge prompt for qa-sev-14"}
        ]

        replacement = rejudged[SEV_14_ID]
        assert replacement.scored is True
        assert replacement.verdict == {"judged_lead": "emergency-level"}
        assert replacement.usage == {"input_tokens": 1024, "output_tokens": 48}, (
            "the re-judge is paid: its usage rides the verdict for the ledger"
        )
        assert rejudged[VA_01_ID].scored is False, "faithfulness is never re-judged"
        assert rejudged[SEV_OK_ID] == _scored(SEV_OK_ID, "severity_fidelity", "qa-sev-01")

    def test_nothing_unscored_means_zero_live_calls(self) -> None:
        adapter = FakeAdapter()
        verdicts = {SEV_OK_ID: _scored(SEV_OK_ID, "severity_fidelity", "qa-sev-01")}
        rejudged = rejudge_gate_adjacent_severity(
            verdicts,
            [_request(SEV_OK_ID, "severity_fidelity", "qa-sev-01")],
            adapter,
            preflight=ALLOWED_PREFLIGHT,
        )
        assert adapter.calls_to("generate") == []
        assert rejudged == verdicts

    def test_unscored_non_severity_verdicts_are_never_rejudged(self) -> None:
        """qa-va-01 (faithfulness) feeds no gate: no spend on it."""
        adapter = FakeAdapter()
        verdicts = {
            VA_01_ID: _unscored(
                VA_01_ID, "faithfulness", "qa-va-01", "malformed verdict: not valid JSON"
            )
        }
        rejudged = rejudge_gate_adjacent_severity(
            verdicts,
            [_request(VA_01_ID, "faithfulness", "qa-va-01")],
            adapter,
            preflight=ALLOWED_PREFLIGHT,
        )
        assert adapter.calls_to("generate") == []
        assert rejudged[VA_01_ID].scored is False

    def test_rejudge_is_a_preflighted_spend_seam(self) -> None:
        """Identical discipline to submit_judge_batch: no pre-flight
        refuses, a failing pre-flight refuses — both with ZERO calls."""
        adapter = FakeAdapter()
        with pytest.raises(LiveRunRefusedError):
            rejudge_gate_adjacent_severity(_run4_verdicts(), _run4_requests(), adapter)
        assert adapter.calls_to("generate") == []
        with pytest.raises(BudgetExceededError):
            rejudge_gate_adjacent_severity(
                _run4_verdicts(), _run4_requests(), adapter, preflight=REFUSED_PREFLIGHT
            )
        assert adapter.calls_to("generate") == []

    def test_failed_rejudge_stays_unscored_single_attempt(self) -> None:
        """Fail-to-unscored, never fail-to-pass, never a retry loop: a
        malformed re-judge response leaves the verdict unscored after
        exactly one call."""
        adapter = FakeAdapter(generate_results=[_judge_answer("still not json {")])
        rejudged = rejudge_gate_adjacent_severity(
            _run4_verdicts(), _run4_requests(), adapter, preflight=ALLOWED_PREFLIGHT
        )
        assert len(adapter.calls_to("generate")) == 1, "single attempt, no retry"
        assert rejudged[SEV_14_ID].scored is False

    def test_fenced_rejudge_verdict_parses_like_the_batch_path(self) -> None:
        """The #324 fence tolerance applies to the re-judge too — one
        parse discipline, no second spelling."""
        fenced = '```json\n{"judged_lead": "serious"}\n```'
        adapter = FakeAdapter(generate_results=[_judge_answer(fenced)])
        rejudged = rejudge_gate_adjacent_severity(
            _run4_verdicts(), _run4_requests(), adapter, preflight=ALLOWED_PREFLIGHT
        )
        assert rejudged[SEV_14_ID].scored is True
        assert rejudged[SEV_14_ID].verdict == {"judged_lead": "serious"}

    def test_rejudged_severity_reaches_the_gate_records(self) -> None:
        """BEFORE the gate computes: the re-judged verdict flows into the
        severity gate's records as a scored item — closing the run-4
        gate-adjacency (an unscored gate item is never left to count
        against the gate when one live call could measure it)."""
        adapter = FakeAdapter(
            generate_results=[_judge_answer(json.dumps({"judged_lead": "serious"}))]
        )
        rejudged = rejudge_gate_adjacent_severity(
            _run4_verdicts(), _run4_requests(), adapter, preflight=ALLOWED_PREFLIGHT
        )
        gold_items = {
            "qa-sev-01": {"severity": {"expected_lead": "serious"}},
            "qa-sev-14": {"severity": {"expected_lead": "serious"}},
        }
        records = {
            record["item_id"]: record
            for record in severity_records_from_verdicts(rejudged, gold_items)
        }
        assert records["qa-sev-14"]["scored"] is True
        assert records["qa-sev-14"]["judged"] == "serious"
