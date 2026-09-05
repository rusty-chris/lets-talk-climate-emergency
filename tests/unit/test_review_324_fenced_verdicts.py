"""Judge verdict parsing tolerates ```json-fenced output
(review finding #324) — RED.

The verification smoke (PR #321, `data/verification-smoke/REPORT.md`)
lost every in-run severity verdict to a parse artifact: the haiku judge
wraps its verdict JSON in markdown code fences, and
``evals.judges._verdict_from_result``'s bare ``json.loads`` rejects the
fenced text — 0/11 scored in-run, while the same batch re-parsed
offline with the fences stripped scores 11/11 (10 exact + 1 adjacent,
zero two-level; no product regression — a pure measurement-integrity
defect).

The verbatim payloads below are the run's own, captured from batch
``msgbatch_01XzDRAKrzZ6HcRkDtve7PGg`` (e.g. custom_id
``claude-haiku-4-5__severity_fidelity__qa-sev-08``, ``stop_reason:
end_turn``, 17 output tokens):

    ```json\\n{"judged_lead": "serious"}\\n```

The pinned fix: before ``json.loads``, the collector strips a SINGLE
leading ````` ```json ````` / trailing ````` ``` ````` fence pair (the
fences at the very start and end of the text, surrounding whitespace
tolerated) — and nothing more:

- genuinely malformed output still degrades to unscored (fenced
  garbage, fenced non-objects, prose around a fence, doubly-nested
  fences);
- unfenced output parses exactly as before;
- fail-to-unscored never becomes fail-to-pass anywhere on the path.

All against the FakeBatchClient double; no network (judges are
Batches-only and the client is injected).
"""

from __future__ import annotations

import json

from evals.judges import (
    JudgeRequest,
    collect_judge_verdicts,
    severity_records_from_verdicts,
)
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    succeeded_batch_result,
)

#: The smoke run's verbatim fenced verdict payloads (one per severity
#: level the batch actually returned).
FENCED_SERIOUS = '```json\n{"judged_lead": "serious"}\n```'
FENCED_EMERGENCY = '```json\n{"judged_lead": "emergency-level"}\n```'
FENCED_REASSURING = '```json\n{"judged_lead": "reassuring"}\n```'

BATCH_ID = "synbatch_001"


def _severity_request(custom_id: str, item_id: str) -> JudgeRequest:
    return JudgeRequest(
        custom_id=custom_id,
        kind="severity_fidelity",
        item_id=item_id,
        judge_model="claude-sonnet-5",
        prompt="synthetic judge prompt",
        schema={"type": "object"},
    )


def _collect(payloads: dict[str, str]) -> dict:
    """Collect verdicts for one request per payload, keyed by custom_id."""
    requests = [_severity_request(custom_id, f"syn-{custom_id}") for custom_id in sorted(payloads)]
    client = FakeBatchClient(
        results=[succeeded_batch_result(custom_id, text) for custom_id, text in payloads.items()]
    )
    return collect_judge_verdicts(BATCH_ID, requests, client, waiter=lambda: None)


def _single(text: str):
    return _collect({"sev-1": text})["sev-1"]


# ---------------------------------------------------------------------------
# 1. The run's verbatim fenced payloads parse and score (RED)
# ---------------------------------------------------------------------------


class TestFencedVerdictsScore:
    def test_the_runs_verbatim_fenced_payload_scores(self):
        """THE #324 pin, with the smoke run's own bytes: a verdict
        wrapped in a single ```json fence pair parses and scores —
        under the bare json.loads it degrades to unscored and the whole
        paid batch measures nothing (0/11 in-run)."""
        verdict = _single(FENCED_SERIOUS)
        assert verdict.scored is True, verdict.failure_reason
        assert verdict.verdict == {"judged_lead": "serious"}

    def test_every_severity_level_the_batch_returned_scores(self):
        """All three verbatim payload variants the run's batch carried."""
        verdicts = _collect(
            {
                "sev-1": FENCED_SERIOUS,
                "sev-2": FENCED_EMERGENCY,
                "sev-3": FENCED_REASSURING,
            }
        )
        assert verdicts["sev-1"].verdict == {"judged_lead": "serious"}
        assert verdicts["sev-2"].verdict == {"judged_lead": "emergency-level"}
        assert verdicts["sev-3"].verdict == {"judged_lead": "reassuring"}
        for verdict in verdicts.values():
            assert verdict.scored is True, verdict.failure_reason

    def test_surrounding_whitespace_around_the_fences_is_tolerated(self):
        """A trailing newline after the closing fence (or leading
        whitespace before the opening one) must not defeat the strip —
        transports routinely append one."""
        verdict = _single(FENCED_SERIOUS + "\n")
        assert verdict.scored is True, verdict.failure_reason
        assert verdict.verdict == {"judged_lead": "serious"}

    def test_fenced_verdicts_feed_the_severity_gate_records(self):
        """The 0/11 -> 11/11 heal, end to end at the join: fenced
        verdicts flow through severity_records_from_verdicts as SCORED
        records joined to the audited gold labels — the severity gate
        measures again."""
        verdicts = _collect({"sev-1": FENCED_SERIOUS, "sev-2": FENCED_EMERGENCY})
        gold_items = {
            "syn-sev-1": {"severity": {"expected_lead": "serious"}},
            "syn-sev-2": {"severity": {"expected_lead": "emergency-level"}},
        }
        records = {r["item_id"]: r for r in severity_records_from_verdicts(verdicts, gold_items)}
        assert records["syn-sev-1"] == {
            "item_id": "syn-sev-1",
            "expected": "serious",
            "judged": "serious",
            "scored": True,
        }
        assert records["syn-sev-2"]["scored"] is True
        assert records["syn-sev-2"]["judged"] == "emergency-level"


# ---------------------------------------------------------------------------
# 2. Genuinely malformed output still degrades to unscored — the strip
#    is one fence pair, never a rescue heuristic
# ---------------------------------------------------------------------------


class TestMalformedStillRejects:
    def _assert_unscored(self, text: str) -> None:
        verdict = _single(text)
        assert verdict.scored is False, f"scored a malformed verdict: {text!r}"
        assert verdict.verdict is None
        assert verdict.failure_reason

    def test_fenced_garbage_is_still_unscored(self):
        """Fence-stripping exposes the content to the SAME strictness:
        non-JSON inside a fence pair stays malformed."""
        self._assert_unscored("```json\nthis is not json {\n```")

    def test_fenced_non_object_json_is_still_unscored(self):
        """A fenced JSON array parses but is not a JSON object — the
        existing not-an-object rejection applies unchanged behind the
        strip."""
        self._assert_unscored('```json\n["judged_lead", "serious"]\n```')

    def test_only_a_single_fence_pair_is_stripped(self):
        """The strip is one pair, not recursive: doubly-nested fences
        leave fenced text behind and degrade to unscored."""
        self._assert_unscored('```json\n```json\n{"judged_lead": "serious"}\n```\n```')

    def test_prose_around_a_fence_is_still_unscored(self):
        """Only a LEADING/trailing pair strips: a fence embedded after
        prose is not the payload envelope, and the collector must not
        go hunting for JSON inside arbitrary judge prose."""
        self._assert_unscored('The verdict is:\n```json\n{"judged_lead": "serious"}\n```')

    def test_unfenced_prose_is_still_unscored(self):
        self._assert_unscored("serious")

    def test_empty_fence_pair_is_still_unscored(self):
        self._assert_unscored("```json\n```")


# ---------------------------------------------------------------------------
# 3. Unfenced output is unchanged
# ---------------------------------------------------------------------------


class TestUnfencedBehaviourUnchanged:
    def test_bare_json_object_still_scores(self):
        """The pre-#324 happy path is untouched: an unfenced JSON
        object scores exactly as before."""
        verdict = _single(json.dumps({"judged_lead": "serious"}))
        assert verdict.scored is True
        assert verdict.verdict == {"judged_lead": "serious"}

    def test_bare_json_object_with_whitespace_still_scores(self):
        verdict = _single('  {"judged_lead": "reassuring"}\n')
        assert verdict.scored is True
        assert verdict.verdict == {"judged_lead": "reassuring"}
