"""Live-release-run red phase (Fable): the two gaps between the merged
#303/#305/#308 battery and an honestly-measured LIVE run.

1. **The live severity feed.** With the owner severity audit complete
   (PR #308) the shared battery scores a supplied ``severity_records``
   feed — but ``run_release_eval`` still discards
   ``collect_judge_verdicts``' return value and passes NO feed, so every
   live run would report severity BLOCKED-unmeasured forever. Pinned
   here: ``evals.judges.severity_records_from_verdicts`` maps the
   severity-kind judge verdicts onto judged-severity records
   ({item_id, expected, judged, scored}) by joining each verdict to its
   gold item's ``severity.expected_lead``, and ``run_release_eval``
   feeds the mapped records into the shared battery — scored verdicts
   score the gate, judge-degraded verdicts stay unscored (never a
   pass), and a run with no judge batch stays BLOCKED (the #308
   absent-feed pin is not weakened). The severity judge prompt must
   also instruct the machine-readable verdict shape (``judged_lead``
   over the three rubric levels) — a judge free-styling prose degrades
   every item to unscored and burns the batch spend.

2. **Transcript fidelity (the #303 ratification note).** "The release
   recording tooling must feed the TRUE SSE transcript to validation,
   not the flat reconstruction." The runner's flat reconstruction (one
   whole-answer ``text`` event, every ``citation`` event trailing) makes
   the production segmenter — which attaches each citation event to the
   last text delivered before it — hang EVERY citation on the final
   sentence. Pinned here: the answer path drives the PRODUCTION
   streamed seam (``adapter.generate_stream`` on the production-built
   request) and hands validation the transcript built by the production
   translator (``rag.generation.answer_stream_to_sse``): text and
   citation events interleaved in transport order, usage and footer
   events present. An error-terminated stream (a delivery production
   would never validate) refuses loudly instead of scoring a truncated
   answer.

No test here touches the network (IMPLEMENTATION.md §4.4); synthetic
gold only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import harness, judges
from evals.harness import (
    AnswerPathDeps,
    GoldSets,
    HarnessError,
    RunJournal,
    load_and_validate_gold,
    run_answer_path,
)
from evals.judges import (
    JUDGE_KINDS,
    SEVERITY_LEVELS,
    JudgeVerdict,
    build_judge_prompt,
)
from rag.generation import (
    GENERATION_MAX_TOKENS_DEFAULT,
    GenerationConfig,
    GroundedAnswer,
    build_generation_request,
)
from rag.provider import FakeAdapter
from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    FakeBatchClient,
    production_passage_payload,
    succeeded_batch_result,
    write_synthetic_gold,
)

ARM_MODEL = "claude-haiku-4-5"

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic planet warming",
}

PASSAGES = RetrievedPassages(
    passages=(
        RerankedPassage(
            chunk_id="syn_doc:0001",
            rerank_score=0.9,
            clears_threshold=True,
            payload=production_passage_payload("syn_doc:0001"),
        ),
    )
)

REFUSAL = HonestRefusal(
    refusal_text="The synthetic corpus does not cover this.",
    covered_topics=("synthetic warming",),
    top_score=0.05,
    threshold=0.4,
)

FIRST_SENTENCE = "The synthetic planet warmed 1.1 degrees."
SECOND_SENTENCE = " It is very likely to continue."

#: A transport stream delivering TWO cited sentences with each citation
#: interleaved after the sentence it supports — the ordering a flat
#: reconstruction destroys.
INTERLEAVED_STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": FIRST_SENTENCE}},
    {
        "type": "content_block_delta",
        "delta": {
            "type": "citations_delta",
            "citation": {"cited_text": "synthetic passage", "document_index": 0},
        },
    },
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": SECOND_SENTENCE}},
    {
        "type": "content_block_delta",
        "delta": {
            "type": "citations_delta",
            "citation": {"cited_text": "synthetic passage", "document_index": 0},
        },
    },
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 80}},
    {"type": "message_stop"},
]

#: A stream that dies before message_stop — production emits a terminal
#: ``error`` event and never validates the truncated delivery.
TRUNCATED_STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": FIRST_SENTENCE}},
]


class _RecordingValidator:
    """Records every (result, sse_events) validation call and returns a
    minimal genuine-shaped outcome."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []

    def __call__(self, result: Any, sse_events: Any) -> Any:
        from rag.citation_validator import AnswerSentence, PairVerdict, ValidationOutcome

        self.calls.append((result, tuple(sse_events)))
        return ValidationOutcome(
            validated=True,
            sentences=(AnswerSentence(index=0, text=FIRST_SENTENCE, document_indices=(0,)),),
            verdicts=(
                PairVerdict(pair_index=0, sentence_index=0, document_index=0, supported=True),
            ),
            support_rate=1.0,
            model=ARM_MODEL,
        )


def _streamed_deps(
    stream: list[dict[str, Any]], validator: Any = None
) -> tuple[FakeAdapter, AnswerPathDeps]:
    adapter = FakeAdapter(
        generate_stream_results=[stream],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    return adapter, AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: PASSAGES, validate_exchange=validator
    )


# ---------------------------------------------------------------------------
# 2. Transcript fidelity: the production streamed path
# ---------------------------------------------------------------------------


def test_answer_path_generates_through_the_streamed_production_seam():
    """Cited generation goes through ``adapter.generate_stream`` on the
    production builder's request — never the folded ``generate`` call
    whose flat reconstruction the #303 note forbids."""
    adapter, deps = _streamed_deps(INTERLEAVED_STREAM)
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, mode="fake")

    assert adapter.calls_to("generate") == [], (
        "the folded generate call materialises no transcript: the #303 "
        "ratification note requires the TRUE SSE transcript, so the eval must "
        "drive the streamed production seam"
    )
    (call,) = adapter.calls_to("generate_stream")
    expected = build_generation_request(
        PASSAGES,
        ANSWERABLE_ITEM["question"],
        config=GenerationConfig(model=ARM_MODEL, max_tokens=GENERATION_MAX_TOKENS_DEFAULT),
    )
    assert call.payload == expected, (
        "the streamed request must still be the production builder's output "
        "field-for-field (#234) — streaming changes the transport, never the prompt"
    )

    assert result.refused is False
    assert result.answer_text == FIRST_SENTENCE + SECOND_SENTENCE
    assert result.retrieved_chunk_ids == ("syn_doc:0001",)
    assert len(result.citations) == 2
    assert result.citations[0]["chunk_id"] == "syn_doc:0001"
    # Usage merges message_start (input) with message_delta (output) — the
    # ledger row must not drop the input side.
    assert result.usage == {"input_tokens": 120, "output_tokens": 80}


def test_validator_receives_the_true_interleaved_sse_transcript():
    """validate_exchange receives the transcript the production
    translator (answer_stream_to_sse) builds from the transport events:
    text/citation events interleaved in ARRIVAL order — each citation
    event delivered before the next sentence's text, exactly where the
    segmenter's attaches-to-preceding-text rule needs it — plus the
    usage and footer events. A flat all-citations-trailing
    reconstruction fails this pin."""
    validator = _RecordingValidator()
    _, deps = _streamed_deps(INTERLEAVED_STREAM, validator)
    (result,) = run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, mode="fake")

    assert len(validator.calls) == 1
    grounded, sse_events = validator.calls[0]

    names = [event.get("event") for event in sse_events]
    assert names == ["text", "citation", "text", "citation", "usage", "footer"], (
        f"the delivered transcript must interleave citation events after the "
        f"text they cite (transport arrival order), then carry usage and the "
        f"footer — got {names}; a flat reconstruction (one text event, every "
        "citation trailing) hangs every citation on the final sentence"
    )
    assert sse_events[0]["data"]["text"] == FIRST_SENTENCE
    assert sse_events[2]["data"]["text"] == SECOND_SENTENCE
    for citation_event in (sse_events[1], sse_events[3]):
        assert citation_event["data"]["document_index"] == 0
        assert citation_event["data"]["chunk_id"] == "syn_doc:0001", (
            "citation events must carry the chunk_id resolved at the seam "
            "(answer_stream_to_sse resolution), like production delivery"
        )
    assert "Answers reflect sources as of" in sse_events[5]["data"]["text"]

    # The validated answer is production's reassembly of that transcript.
    assert isinstance(grounded, GroundedAnswer)
    assert grounded.text == FIRST_SENTENCE + SECOND_SENTENCE
    assert len(grounded.cited_passages) == 2
    assert grounded.cited_passages[0].payload["body"] == "synthetic passage"
    assert grounded.footer == sse_events[5]["data"]["text"]

    assert result.validation == {"validated": True, "supported": 1, "factual": 1}


def test_error_terminated_stream_refuses_loudly_and_never_validates(tmp_path: Path):
    """A stream production would terminate with an ``error`` event (here:
    ended before message_stop) is NOT scoreable evidence: the runner
    raises a HarnessError naming the item and the error type, validation
    never runs, and nothing is journalled — the item re-runs on resume
    instead of a truncated answer silently entering the gates."""

    def poisoned_validate(result, sse_events):
        raise AssertionError("an error-terminated delivery must never be validated")

    adapter = FakeAdapter(
        generate_stream_results=[TRUNCATED_STREAM],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter, retrieve=lambda decision: PASSAGES, validate_exchange=poisoned_validate
    )
    journal = RunJournal(tmp_path / "journal.jsonl")
    with pytest.raises(HarnessError) as excinfo:
        run_answer_path([ANSWERABLE_ITEM], deps, arm_model=ARM_MODEL, mode="fake", journal=journal)
    message = str(excinfo.value)
    assert "syn-sp-01" in message
    assert "incomplete_stream" in message
    assert not journal.completed_item_ids(), "a refused delivery must not be journalled as done"


# ---------------------------------------------------------------------------
# 1a. The severity verdict mapper
# ---------------------------------------------------------------------------


def _severity_verdict(
    item_id: str,
    *,
    scored: bool = True,
    verdict: dict[str, Any] | None = None,
    kind: str = "severity_fidelity",
) -> JudgeVerdict:
    return JudgeVerdict(
        custom_id=f"{ARM_MODEL}::{kind}::{item_id}",
        kind=kind,
        item_id=item_id,
        scored=scored,
        verdict=verdict,
        failure_reason=None if scored else "synthetic judge failure",
    )


GOLD_BY_ID = {
    "syn-sev-01": {
        "id": "syn-sev-01",
        "category": "severity",
        "severity": {"expected_lead": "serious"},
    },
    "syn-sev-02": {
        "id": "syn-sev-02",
        "category": "severity",
        "severity": {"expected_lead": "emergency-level"},
    },
}


def test_severity_records_mapper_joins_verdicts_to_gold_labels():
    """severity_records_from_verdicts derives one judged record per
    severity-kind verdict: expected from the gold ``severity.expected_lead``,
    judged from the verdict's ``judged_lead``; non-severity kinds are
    ignored (they feed no gate)."""
    mapper = getattr(judges, "severity_records_from_verdicts", None)
    assert mapper is not None, (
        "evals.judges.severity_records_from_verdicts must exist: the live "
        "severity gate's only honest feed is the severity judge's verdicts "
        "joined to the audited gold labels"
    )
    verdicts = {
        "a": _severity_verdict("syn-sev-01", verdict={"judged_lead": "serious"}),
        "b": _severity_verdict("syn-sev-02", verdict={"judged_lead": "reassuring"}),
        "c": _severity_verdict("syn-sev-01", kind="faithfulness", verdict={"verdict": "pass"}),
    }
    records = {record["item_id"]: record for record in mapper(verdicts, GOLD_BY_ID)}
    assert set(records) == {"syn-sev-01", "syn-sev-02"}, (
        "exactly one record per severity-kind verdict; other judge kinds feed no gate"
    )
    assert records["syn-sev-01"] == {
        "item_id": "syn-sev-01",
        "expected": "serious",
        "judged": "serious",
        "scored": True,
    }
    # The two-level candidate rides through untouched — the RATIFIED
    # exact/adjacent/two-level arithmetic lives in gates.severity_gate,
    # never a second implementation here.
    assert records["syn-sev-02"]["expected"] == "emergency-level"
    assert records["syn-sev-02"]["judged"] == "reassuring"
    assert records["syn-sev-02"]["scored"] is True


def test_severity_records_mapper_degrades_failures_to_unscored():
    """A judge-degraded verdict (batch failure) or a scored verdict with
    no parseable ``judged_lead`` maps to scored=False with judged=None —
    unscored is never agreement, so it counts AGAINST the ≥90% gate."""
    verdicts = {
        "a": _severity_verdict("syn-sev-01", scored=False, verdict=None),
        "b": _severity_verdict("syn-sev-02", verdict={"free_prose": "seems serious"}),
    }
    records = {
        record["item_id"]: record
        for record in judges.severity_records_from_verdicts(verdicts, GOLD_BY_ID)
    }
    for item_id in ("syn-sev-01", "syn-sev-02"):
        assert records[item_id]["scored"] is False
        assert records[item_id]["judged"] is None
    assert records["syn-sev-01"]["expected"] == "serious"


# ---------------------------------------------------------------------------
# 1b. The severity judge's machine-readable verdict instruction
# ---------------------------------------------------------------------------


def test_severity_judge_prompt_instructs_the_judged_lead_verdict_shape():
    """The severity prompt must name the exact verdict contract the
    collector parses — a JSON object carrying ``judged_lead`` over the
    three rubric levels. Without it a live judge free-styles prose,
    every item degrades to unscored, and the paid batch scores nothing."""
    prompt = build_judge_prompt("severity_fidelity", GOLD_BY_ID["syn-sev-01"], "Serious. [1]")
    assert "judged_lead" in prompt
    for level in SEVERITY_LEVELS:
        assert f'"{level}"' in prompt, f"the verdict instruction must name {level!r}"
    assert "JSON" in prompt


def test_every_judge_kind_instructs_a_json_object_verdict():
    """Every judge kind tells the model to answer with ONLY a JSON
    object — the collector treats non-JSON text as a malformed verdict
    (unscored), so an uninstructed prompt wastes the whole batch."""
    for kind in JUDGE_KINDS:
        prompt = build_judge_prompt(kind, GOLD_BY_ID["syn-sev-01"], "Serious. [1]")
        assert "JSON object" in prompt, f"{kind} prompt carries no verdict-format contract"


# ---------------------------------------------------------------------------
# 1c. run_release_eval feeds the mapped records into the shared battery
# ---------------------------------------------------------------------------


def _synthetic_gold(tmp_path: Path) -> GoldSets:
    qa_path, charts_path = write_synthetic_gold(tmp_path / "gold")
    return load_and_validate_gold(qa_path, charts_path)


def _release_deps_factory(gold: GoldSets, *, refuse_everything: bool = False):
    def deps_factory(arm_model: str) -> AnswerPathDeps:
        adapter = FakeAdapter(
            generate_stream_results=[INTERLEAVED_STREAM] * len(gold.qa_items),
            structured_results=[
                {"scope": "in_scope", "rewritten_query": item["question"]} for item in gold.qa_items
            ],
        )

        def retrieve(decision):
            query = decision.retrieval_query or ""
            if refuse_everything or "unanswerable" in query or "out-of-scope" in query:
                return REFUSAL
            return PASSAGES

        return AnswerPathDeps(
            adapter=adapter,
            retrieve=retrieve,
            validate_exchange=_RecordingValidator(),
        )

    return deps_factory


def _run_release_eval(gold: GoldSets, tmp_path: Path, batch_client: FakeBatchClient, **kwargs):
    return harness.run_release_eval(
        gold,
        arm_models=(ARM_MODEL,),
        deps_factory=_release_deps_factory(
            gold, refuse_everything=kwargs.pop("refuse_everything", False)
        ),
        plan_chart=lambda request: {
            "kind": "spec",
            "spec": {"chart_id": "syn-chart", "chart_type": "line"},
        },
        batch_client=batch_client,
        mode="fake",
        ledger_path=tmp_path / "spend-ledger.csv",
        journal_dir=None,
        session_id="live-wiring-fake",
        **kwargs,
    )


def _severity_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    (arm,) = payload["arms"]
    matches = [gate for gate in arm["gates"] if gate["name"] == "severity"]
    assert len(matches) == 1
    return matches[0]


def test_release_eval_scores_severity_from_the_judge_batch(tmp_path: Path):
    """A judge batch whose severity verdict agrees with the audited gold
    label yields a SCORED severity gate (passed, 1/1 on the synthetic
    gold) — the collected verdicts are finally fed to the battery
    instead of being discarded."""
    gold = _synthetic_gold(tmp_path)
    batch_client = FakeBatchClient(
        results=[
            succeeded_batch_result(
                f"{ARM_MODEL}::severity_fidelity::syn-sev-01",
                json.dumps({"judged_lead": "serious"}),
            )
        ]
    )
    payload = _run_release_eval(gold, tmp_path, batch_client)
    severity = _severity_gate_payload(payload)
    assert severity["status"] == "passed", (
        "the severity judge agreed with the audited gold label exactly: the "
        "gate must SCORE (the live feed), not stay blocked-unmeasured"
    )
    assert (severity["numerator"], severity["denominator"]) == (1, 1)


def test_release_eval_counts_unscored_severity_verdicts_against_the_gate(tmp_path: Path):
    """A batch that returns NO severity result (judge failure) leaves the
    severity item unscored — the gate scores 0/1 and FAILS; a judge
    failure is never a pass and never quietly re-blocks the gate."""
    gold = _synthetic_gold(tmp_path)
    payload = _run_release_eval(gold, tmp_path, FakeBatchClient(results=[]))
    severity = _severity_gate_payload(payload)
    assert severity["status"] == "failed"
    assert (severity["numerator"], severity["denominator"]) == (0, 1)
    evidence = {entry["item_id"]: entry for entry in severity["evidence"]}
    assert evidence["syn-sev-01"]["scored"] is False


def test_release_eval_without_judge_batch_keeps_severity_blocked(tmp_path: Path):
    """When no judge batch runs at all (every item refused → zero judge
    requests) the absent-feed contract holds: severity reports BLOCKED,
    exactly the #308 fail-closed pin."""
    gold = _synthetic_gold(tmp_path)
    payload = _run_release_eval(gold, tmp_path, FakeBatchClient(results=[]), refuse_everything=True)
    severity = _severity_gate_payload(payload)
    assert severity["status"] == "blocked"
    assert severity["reason"]


# ---------------------------------------------------------------------------
# 3. The classifier schema must be live-servable (found by this release run)
# ---------------------------------------------------------------------------


def test_classifier_schema_stays_inside_the_structured_outputs_subset():
    """The classifier's structured-output schema must be live-servable.

    The 2026-09-04 release run's first live classify call drew a 400
    (``output_config.format.schema: For 'anyOf', 'additionalProperties,
    properties, required, type' is not supported``, request id
    req_011Cej4Lwa6E4hD9x4i8RGUm): the live API rejects a node mixing
    ``anyOf`` with object keywords — the shape the finding-#86 steering
    block used. The unsafe→unsafe_subtype rule is ALREADY enforced at
    parse (parse_classifier_output raises, covered by the retry-once
    budget), so the schema steering is droppable without weakening any
    invariant; what is NOT acceptable is a unit-green schema that 400s
    on every live call (findings #203/#209/#262 — the exact failure
    class the shared subset lint exists to catch)."""
    from rag.query import build_query_processing_request
    from tests._schema_subset import assert_schema_within_structured_outputs_subset

    request = build_query_processing_request("How much has the planet warmed?")
    assert_schema_within_structured_outputs_subset(
        request["schema"], name="classifier processing schema"
    )


def test_schema_subset_lint_bans_anyof_mixed_with_object_keywords():
    """The shared lint itself must catch the newly-observed 400 shape for
    EVERY structured builder (finding #209's promotion rule): a node
    carrying ``anyOf`` alongside type/properties/required/
    additionalProperties is rejected, naming the offence."""
    from tests._schema_subset import assert_schema_within_structured_outputs_subset

    mixed = {
        "type": "object",
        "properties": {"scope": {"type": "string"}},
        "required": ["scope"],
        "additionalProperties": False,
        "anyOf": [{"required": ["scope"]}],
    }
    with pytest.raises(AssertionError, match="anyOf"):
        assert_schema_within_structured_outputs_subset(mixed, name="mixed-anyof")
