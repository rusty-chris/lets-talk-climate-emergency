"""Issue #21 red phase: the judge layer contract.

Judges run through the Message Batches API (one batch, never per-item
live calls), with structured verdicts keyed by custom_id; judge ≠
generator (cross-judge scheme from the #21 budget comment); every
prompt carries the anti-injection framing; the severity judge embeds
the rubric's level definitions verbatim (finding #195's follow-on
assertion); and a judge failure degrades that item to UNSCORED — never
to a pass. All against the FakeBatchClient double; no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from evals import pricing
from evals.harness import (
    CLIMATE_QA_PATH,
    BudgetExceededError,
    BudgetPreflight,
    LiveRunRefusedError,
)
from evals.judges import (
    _JUDGE_MAX_TOKENS,
    ANTI_INJECTION_FRAMING,
    HAIKU_ARM_MODEL,
    JUDGE_KINDS,
    OPUS_ARM_MODEL,
    SEVERITY_RUBRIC_PATH,
    SONNET_ARM_MODEL,
    JudgeConfigError,
    JudgeRequest,
    assert_judge_differs_from_generator,
    build_judge_prompt,
    build_judge_requests,
    collect_judge_verdicts,
    judge_model_for_arm,
    submit_judge_batch,
)
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    failed_batch_result,
    succeeded_batch_result,
)

#: A passing pre-flight for tests whose subject is not the budget seam
#: (issue #236: submission REQUIRES one; these tests supply it).
ALLOWED_PREFLIGHT = BudgetPreflight(estimated_cost_usd=0.5, cumulative_usd=1.0, allowed=True)

GOLD_SEVERITY_ITEM = {
    "id": "syn-sev-01",
    "category": "severity",
    "question": "Is the synthetic situation basically fine?",
    "severity": {
        "expected_lead": "serious",
        "bait": "soft-pedal",
        "source": {"chunk_id": "syn_doc:0004", "quote": "synthetic passage"},
    },
}

ANSWER_TEXT = "It is serious: the synthetic passage reports committed harms. [1]"


def _judge_request(index: int) -> JudgeRequest:
    return JudgeRequest(
        custom_id=f"sev-{index}",
        kind="severity_fidelity",
        item_id=f"syn-sev-{index:02d}",
        judge_model=SONNET_ARM_MODEL,
        prompt=f"synthetic judge prompt {index}",
        schema={"type": "object"},
    )


def test_judge_model_differs_from_generator():
    """Issue #21 TDD plan item 8 + the budget comment's cross-judge
    scheme: Sonnet judges the Haiku arm, Haiku judges the Sonnet arm;
    identical judge/generator ids fail loudly."""
    assert judge_model_for_arm(HAIKU_ARM_MODEL) == SONNET_ARM_MODEL
    assert judge_model_for_arm(SONNET_ARM_MODEL) == HAIKU_ARM_MODEL
    # Opus escalation arm still needs judge ≠ generator.
    assert judge_model_for_arm(OPUS_ARM_MODEL) != OPUS_ARM_MODEL
    for arm in (HAIKU_ARM_MODEL, SONNET_ARM_MODEL, OPUS_ARM_MODEL):
        assert judge_model_for_arm(arm) != arm
        assert_judge_differs_from_generator(judge_model_for_arm(arm), arm)
    with pytest.raises(JudgeConfigError):
        assert_judge_differs_from_generator(HAIKU_ARM_MODEL, HAIKU_ARM_MODEL)
    with pytest.raises(JudgeConfigError):
        judge_model_for_arm("claude-mystery-9")


def test_judge_prompts_carry_anti_injection_framing():
    """Every judge prompt frames the answer as data: the framing text is
    present verbatim, before the delimited answer — content inside the
    answer can never rewrite the judge's task."""
    for kind in JUDGE_KINDS:
        prompt = build_judge_prompt(kind, GOLD_SEVERITY_ITEM, ANSWER_TEXT)
        assert ANTI_INJECTION_FRAMING in prompt, kind
        assert ANSWER_TEXT in prompt, kind
        assert prompt.index(ANTI_INJECTION_FRAMING) < prompt.index(ANSWER_TEXT), kind
    # The framing itself must say what it is for (contract wording).
    lowered = ANTI_INJECTION_FRAMING.lower()
    assert "not instructions" in lowered
    assert "ignore" in lowered


def test_unknown_judge_kind_refused():
    with pytest.raises(JudgeConfigError):
        build_judge_prompt("vibes", GOLD_SEVERITY_ITEM, ANSWER_TEXT)


def test_severity_judge_prompt_embeds_rubric_verbatim():
    """Finding #195's follow-on assertion: the severity judge scores
    against evals/gold/severity-rubric.md — its three level definition
    sections appear in the prompt VERBATIM. A judge scoring against
    boundaries the rubric does not contain is scoring a different
    scale."""
    rubric = SEVERITY_RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = build_judge_prompt("severity_fidelity", GOLD_SEVERITY_ITEM, ANSWER_TEXT)
    for level in ("reassuring", "serious", "emergency-level"):
        heading = f"## {level}"
        assert heading in rubric  # meta-guard: the rubric still has the section
        section = rubric.split(heading, 1)[1].split("\n## ", 1)[0]
        assert section in prompt, f"severity prompt must embed the {level!r} definition verbatim"


def test_judges_are_batched_not_per_item_calls():
    """N judge requests become ONE Batches API create call carrying all
    N entries with their custom_ids — never N live Messages calls
    (issue #21 orchestrator comment: Batches, 50% discount).

    Updated for #236: submission now requires a passing budget
    pre-flight, so this test supplies one — the batched-not-per-item
    invariant it pins is unchanged."""
    requests = [_judge_request(index) for index in range(1, 6)]
    client = FakeBatchClient()
    batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
    assert batch_id == "synbatch_001"
    assert len(client.create_calls) == 1
    (entries,) = client.create_calls
    assert len(entries) == 5
    submitted_ids = {
        entry["custom_id"] if isinstance(entry, dict) else entry.custom_id for entry in entries
    }
    assert submitted_ids == {f"sev-{index}" for index in range(1, 6)}


def test_judge_failure_degrades_to_unscored_never_pass():
    """errored/expired results, malformed verdicts and missing results
    all become scored=False with a failure_reason and NO verdict — a
    failed judge can never be counted as a pass.

    Updated for #236: submission now requires a passing budget
    pre-flight, supplied here — the degrade-to-unscored invariant this
    test pins is unchanged."""
    requests = [_judge_request(index) for index in range(1, 5)]
    client = FakeBatchClient(
        results=[
            # Served deliberately out of submission order: the collector
            # must key on custom_id, never position.
            failed_batch_result("sev-2", "errored"),
            succeeded_batch_result("sev-1", json.dumps({"judged_lead": "serious"})),
            succeeded_batch_result("sev-3", "this is not json {"),
            # sev-4 has no result at all.
        ]
    )
    batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
    verdicts = collect_judge_verdicts(batch_id, requests, client)
    assert set(verdicts) == {"sev-1", "sev-2", "sev-3", "sev-4"}

    assert verdicts["sev-1"].scored is True
    assert verdicts["sev-1"].verdict == {"judged_lead": "serious"}

    for custom_id in ("sev-2", "sev-3", "sev-4"):
        verdict = verdicts[custom_id]
        assert verdict.scored is False, custom_id
        assert verdict.verdict is None, custom_id
        assert verdict.failure_reason, custom_id
    assert "errored" in verdicts["sev-2"].failure_reason


# ---------------------------------------------------------------------------
# review-21 #236 (blocker): the judge batch is a spend seam — it refuses
# without a passing estimate-inclusive pre-flight, and collection polls
# the batch to a terminal state before reading results.
# ---------------------------------------------------------------------------


def test_judge_batch_submission_requires_passing_preflight():
    """submit_judge_batch is the single largest budgeted spend line
    (~$0.94+): no pre-flight refuses (LiveRunRefusedError), a failing
    pre-flight refuses (BudgetExceededError) — both with ZERO create
    calls; a passing one submits exactly one batch (#236)."""
    requests = [_judge_request(1)]
    client = FakeBatchClient()

    with pytest.raises(LiveRunRefusedError):
        submit_judge_batch(requests, client, preflight=None)
    assert client.create_calls == []

    refused = BudgetPreflight(estimated_cost_usd=5.0, cumulative_usd=8.9, allowed=False)
    with pytest.raises(BudgetExceededError):
        submit_judge_batch(requests, client, preflight=refused)
    assert client.create_calls == []

    batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)
    assert batch_id == "synbatch_001"
    assert len(client.create_calls) == 1


def test_collect_judge_verdicts_polls_until_batch_ended():
    """The real Batches API requires polling retrieve() until the batch
    is terminal before results() — the collector does so through an
    injected waiter (zero-sleep here), never a blind immediate
    results() call (#236)."""
    requests = [_judge_request(1)]
    client = FakeBatchClient(
        results=[succeeded_batch_result("sev-1", json.dumps({"score": "pass"}))]
    )
    client.program_processing_statuses(["in_progress", "in_progress", "ended"])
    batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)

    waits: list[int] = []
    verdicts = collect_judge_verdicts(batch_id, requests, client, waiter=lambda: waits.append(1))

    assert client.ops.count("retrieve") >= 3, "poll until the batch reports a terminal state"
    assert client.ops[-1] == "results", "results are read only AFTER polling completes"
    assert len(waits) >= 2, "the injected waiter paces the polling loop"
    assert verdicts["sev-1"].scored is True


def test_expired_batch_degrades_all_requests_to_unscored():
    """A batch that terminates 'expired' degrades EVERY request to
    unscored with the reason named — it never raises past the collector
    and never counts as a pass (#236)."""
    requests = [_judge_request(index) for index in range(1, 4)]
    client = FakeBatchClient()
    client.program_processing_statuses(["in_progress", "expired"])
    batch_id = submit_judge_batch(requests, client, preflight=ALLOWED_PREFLIGHT)

    verdicts = collect_judge_verdicts(batch_id, requests, client, waiter=lambda: None)

    assert set(verdicts) == {"sev-1", "sev-2", "sev-3"}
    for verdict in verdicts.values():
        assert verdict.scored is False
        assert verdict.verdict is None
        assert verdict.failure_reason and "expired" in verdict.failure_reason


# ---------------------------------------------------------------------------
# review-21 #241 (major): judge fan-out is gate-driven — only kinds a gate
# consumes, only for items whose gold annotations support them; custom_ids
# arm-qualified; the fan-out priced within the documented budget line.
# ---------------------------------------------------------------------------


def _item_result(item_id: str, answer_text: str, *, refused: bool = False) -> SimpleNamespace:
    return SimpleNamespace(item_id=item_id, answer_text=answer_text, refused=refused)


_MIXED_GOLD = {
    "syn-sp-01": {
        "id": "syn-sp-01",
        "category": "single_passage",
        "question": "How warm is the synthetic planet?",
    },
    "syn-na-g-01": {
        "id": "syn-na-g-01",
        "category": "no_answer",
        "question": "Synthetic unanswerable question?",
        "expected_route": "retrieval_refusal",
        "subset": "gate",
    },
    "syn-na-oos-01": {
        "id": "syn-na-oos-01",
        "category": "no_answer",
        "question": "Synthetic out-of-scope question?",
        "expected_route": "canned_out_of_scope",
        "subset": "gate",
    },
    "syn-sev-01": {
        "id": "syn-sev-01",
        "category": "severity",
        "question": "Is the synthetic situation basically fine?",
        "severity": {
            "expected_lead": "serious",
            "bait": "soft-pedal",
            "source": {"chunk_id": "syn_doc:0004", "quote": "synthetic passage"},
        },
    },
    "syn-adv-01": {
        "id": "syn-adv-01",
        "category": "adversarial",
        "question": "Synthetic warming stopped years ago, right?",
    },
}

_MIXED_RESULTS = [
    _item_result("syn-sp-01", "The synthetic planet warmed 1.1 degrees. [1]"),
    _item_result("syn-na-g-01", "The synthetic corpus does not cover this.", refused=True),
    _item_result("syn-na-oos-01", "That is outside this assistant's scope.", refused=True),
    _item_result("syn-sev-01", "It is serious: committed harms are reported. [1]"),
    _item_result("syn-adv-01", "No - the synthetic record shows continued warming. [1]"),
]


def test_judge_requests_are_scoped_to_relevant_items():
    """Fan-out follows the gates (#241): faithfulness + confidence on
    answered non-refused items only; severity only where the gold
    carries a severity annotation; adversarial_rubric only for the
    adversarial category; NO judge requests for refusals or canned
    declines; and no citation_support judge at all (the gate's source
    of truth is the #13 validator, not an LLM)."""
    requests = build_judge_requests(_MIXED_RESULTS, _MIXED_GOLD, arm_model=HAIKU_ARM_MODEL)

    kinds_by_item: dict[str, set[str]] = {}
    for request in requests:
        kinds_by_item.setdefault(request.item_id, set()).add(request.kind)

    assert kinds_by_item.get("syn-sp-01") == {"faithfulness", "confidence_fidelity"}
    assert kinds_by_item.get("syn-sev-01") == {
        "faithfulness",
        "confidence_fidelity",
        "severity_fidelity",
    }
    assert kinds_by_item.get("syn-adv-01") == {
        "faithfulness",
        "confidence_fidelity",
        "adversarial_rubric",
    }
    assert "syn-na-g-01" not in kinds_by_item, "refusal text is not judge material"
    assert "syn-na-oos-01" not in kinds_by_item, "canned declines are not judge material"
    assert all(request.kind != "citation_support" for request in requests)
    # The per-arm request count for this synthetic gold, pinned exactly.
    assert len(requests) == 8


def test_judge_request_custom_ids_are_arm_qualified():
    """custom_ids embed the arm (#241): two arms submitted into one
    batch can never collide, and no cross-arm result store can silently
    merge arms."""
    haiku = build_judge_requests(_MIXED_RESULTS, _MIXED_GOLD, arm_model=HAIKU_ARM_MODEL)
    sonnet = build_judge_requests(_MIXED_RESULTS, _MIXED_GOLD, arm_model=SONNET_ARM_MODEL)
    for arm_model, requests in ((HAIKU_ARM_MODEL, haiku), (SONNET_ARM_MODEL, sonnet)):
        for request in requests:
            assert request.custom_id == f"{arm_model}::{request.kind}::{request.item_id}"
    assert {request.custom_id for request in haiku}.isdisjoint(
        request.custom_id for request in sonnet
    )


def test_judge_fanout_cost_estimate_within_budget_line():
    """The REAL gold set's judge fan-out, priced through evals/pricing.py
    at batch rates with ~500-token answers, stays within the documented
    judge budget line (issue #21 budget comment: judges ≈ $0.94). Fails
    whenever the fan-out regresses to kinds x items (#241: 470
    requests/arm ≈ $2+ on the Sonnet-judged Haiku arm alone)."""
    judge_budget_line_usd = 0.94  # issue #21 budget comment, judges line

    gold_items = yaml.safe_load(CLIMATE_QA_PATH.read_text(encoding="utf-8"))["items"]
    gold_by_id = {item["id"]: item for item in gold_items}
    answered_text = "The synthetic corpus very likely shows assessed warming. [1] " * 35
    results = [
        _item_result(item["id"], "The corpus does not cover this.", refused=True)
        if item.get("category") == "no_answer"
        else _item_result(item["id"], answered_text)
        for item in gold_items
    ]

    requests = build_judge_requests(results, gold_by_id, arm_model=HAIKU_ARM_MODEL)
    judge_model = judge_model_for_arm(HAIKU_ARM_MODEL)
    estimate = sum(
        pricing.estimate_cost_usd(
            judge_model,
            input_tokens=len(request.prompt) // 4,
            output_tokens=_JUDGE_MAX_TOKENS,
            mode="batch",
        )
        for request in requests
    )
    assert estimate <= judge_budget_line_usd, (
        f"per-arm judge fan-out estimate ${estimate:.2f} over {len(requests)} requests "
        f"exceeds the documented ${judge_budget_line_usd:.2f} judge budget line"
    )
