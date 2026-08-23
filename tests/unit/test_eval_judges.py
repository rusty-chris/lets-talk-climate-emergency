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

import pytest

from evals.judges import (
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
    collect_judge_verdicts,
    judge_model_for_arm,
    submit_judge_batch,
)
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    failed_batch_result,
    succeeded_batch_result,
)

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
    (issue #21 orchestrator comment: Batches, 50% discount)."""
    requests = [_judge_request(index) for index in range(1, 6)]
    client = FakeBatchClient()
    batch_id = submit_judge_batch(requests, client)
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
    failed judge can never be counted as a pass."""
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
    batch_id = submit_judge_batch(requests, client)
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
