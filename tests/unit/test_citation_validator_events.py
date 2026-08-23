"""Post-stream badge events, stream composition, and the exchange
record (issue #13) — RED.

Unit tier, pure + FakeAdapter. The badge events extend the pinned #12
SSE vocabulary (text/citation/usage/footer) with ``badge`` and
``validation_degraded`` — both strictly POST-stream: no validator work
happens until the source stream's final event has been delivered, so
first-token latency never waits on validation (§3.3). Deliberately NOT
pinned (coordinator rider on findings #183/#184/#185): the exact
source-stream event ordering — the vocabulary is gaining
error/termination events on a parallel branch; these tests assert only
pass-through-unchanged and badges-strictly-after-the-final-source-event.
"""

from __future__ import annotations

import pytest

from rag.citation_validator import (
    BADGE_EVENT,
    UNVERIFIED_REASON_ENTAILMENT,
    UNVERIFIED_REASON_UNCITED,
    VALIDATION_DEGRADED_EVENT,
    ValidatorConfig,
    append_validation_events,
    exchange_log_record,
    validate_exchange,
    validation_sse_events,
)
from rag.provider import FakeAdapter, StructuredResult
from tests._citation_validator_fixtures import (
    corrupted_answer_events,
    make_grounded_answer,
    verdicts_output,
)

CONFIG = ValidatorConfig()

VALIDATOR_USAGE = {"input_tokens": 2500, "output_tokens": 300}


def outcome_for(fake: FakeAdapter):
    return validate_exchange(fake, make_grounded_answer(), corrupted_answer_events(), config=CONFIG)


def make_validate(fake: FakeAdapter):
    """The service-layer wiring: collected transcript -> outcome."""

    def validate(events):
        return validate_exchange(fake, make_grounded_answer(), events, config=CONFIG)

    return validate


class TestBadgeEvents:
    def test_badge_events_reference_sentence_and_citation(self):
        """The #18 UI contract: an entailment badge carries the sentence
        index AND the chip's document_index; an uncited badge carries
        the sentence index with document_index None (no chip exists)."""
        fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
        events = list(validation_sse_events(outcome_for(fake)))
        assert all(e["event"] == BADGE_EVENT for e in events)
        payloads = {
            (e["data"]["sentence_index"], e["data"]["document_index"], e["data"]["reason"])
            for e in events
        }
        assert payloads == {
            (2, 1, UNVERIFIED_REASON_ENTAILMENT),
            (3, None, UNVERIFIED_REASON_UNCITED),
        }

    def test_fully_supported_answer_emits_no_events(self):
        fake = FakeAdapter(structured_results=[verdicts_output(True, True)])
        answer = make_grounded_answer()
        # Strip the uncited sentence: cited sentences only, all supported.
        events = [
            e
            for e in corrupted_answer_events()
            if "Reservoir" not in str(e.get("data", {}).get("text", ""))
        ]
        outcome = validate_exchange(fake, answer, events, config=CONFIG)
        assert list(validation_sse_events(outcome)) == []

    def test_skipped_validation_emits_no_events(self):
        outcome = validate_exchange(FakeAdapter(), object(), (), config=CONFIG)
        assert list(validation_sse_events(outcome)) == []

    def test_degraded_validation_emits_one_typed_event(self):
        """Failure containment on the wire: a validator failure becomes
        exactly ONE typed degradation event — never a broken stream,
        never fake badges."""
        fake = FakeAdapter(structured_results=[RuntimeError("SYNTHETIC failure")])
        events = list(validation_sse_events(outcome_for(fake)))
        assert len(events) == 1
        assert events[0]["event"] == VALIDATION_DEGRADED_EVENT
        assert events[0]["data"]["reason"]


class TestStreamComposition:
    def test_badges_emitted_only_after_stream_complete(self):
        """Issue #13 acceptance (TDD plan step 5): stream start never
        waits on validation, and validation runs STRICTLY after the
        final source event is delivered. Instrumented with a lazy
        source and the FakeAdapter call log: zero adapter calls until
        every source event (footer included) has been pulled through;
        the first pull past the source triggers the one batched call
        and yields the first badge."""
        fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
        source_events = corrupted_answer_events()
        pulled: list[int] = []

        def lazy_source():
            for index, event in enumerate(source_events):
                pulled.append(index)
                yield event

        composed = append_validation_events(lazy_source(), make_validate(fake))

        first = next(composed)
        assert first == source_events[0]
        assert fake.calls == [], "first token must never wait on validation"

        delivered = [first]
        for _ in range(len(source_events) - 1):
            delivered.append(next(composed))
        assert delivered == source_events
        assert fake.calls == [], "validation must not start before the final source event"

        first_badge = next(composed)
        assert first_badge["event"] == BADGE_EVENT
        assert len(fake.calls_to("structured")) == 1

    def test_source_events_pass_through_unchanged_then_badges(self):
        """Pass-through fidelity + placement: the composed stream is the
        source events verbatim (whatever their ordering), followed by
        the badge events — no badge ever precedes a source event."""
        fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
        source_events = corrupted_answer_events()
        composed = list(append_validation_events(iter(source_events), make_validate(fake)))
        assert composed[: len(source_events)] == source_events
        trailing = composed[len(source_events) :]
        assert trailing, "the corrupted fixture must produce badges"
        assert {e["event"] for e in trailing} == {BADGE_EVENT}

    def test_validate_receives_the_full_transcript(self):
        seen: list = []

        def spy(events):
            seen.append(list(events))
            fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
            return validate_exchange(fake, make_grounded_answer(), events, config=CONFIG)

        source_events = corrupted_answer_events()
        list(append_validation_events(iter(source_events), spy))
        assert seen == [source_events]

    def test_error_terminated_stream_is_never_validated(self):
        """Coordinator rider (findings #183/#185): a stream that ended
        in an error event delivered an INCOMPLETE answer — the
        validator must not run and no badge or degradation event may
        fire after it. Source events pass through unchanged; the
        stream ends where it ended."""
        fake = FakeAdapter()
        source_events = [
            *corrupted_answer_events()[:-2],  # text/citation body, no clean close
            {"event": "error", "data": {"reason": "SYNTHETIC upstream failure"}},
        ]
        invoked: list = []

        def spy(events):
            invoked.append(events)
            return validate_exchange(fake, make_grounded_answer(), events, config=CONFIG)

        composed = list(append_validation_events(iter(source_events), spy))
        assert composed == source_events
        assert invoked == [], "an error-terminated stream must never be validated"
        assert fake.calls == []


class TestExchangeRecord:
    def test_aggregate_support_rate_logged(self):
        """Issue #13 acceptance (TDD plan step 6): the per-exchange
        aggregate support rate lands in the structured exchange record
        (the #22/#56 consumption seam), with usage and cost fields for
        the ledger — cost from evals.pricing (the single pricing
        source; hand-computed: 2500 in x $1/MTok + 300 out x $5/MTok =
        $0.0040 on Haiku, inside the §9 ~$0.003-stage envelope)."""
        from evals.pricing import estimate_cost_usd

        fake = FakeAdapter()
        fake.queue(
            "structured",
            StructuredResult(value=verdicts_output(True, False), usage=dict(VALIDATOR_USAGE)),
        )
        record = exchange_log_record(outcome_for(fake))
        assert record["citation_support_rate"] == pytest.approx(1 / 3)
        assert record["factual_sentence_count"] == 3
        assert sorted(record["unverified_sentence_indices"]) == [2, 3]
        assert record["validated"] is True
        assert record["skipped_reason"] is None
        assert record["degraded_reason"] is None
        assert record["model"] == CONFIG.model
        assert record["usage"] == VALIDATOR_USAGE
        assert record["cost_usd"] == pytest.approx(0.0040)
        assert record["cost_usd"] == pytest.approx(
            estimate_cost_usd(CONFIG.model, input_tokens=2500, output_tokens=300)
        )
        assert record["cost_usd"] < 0.005  # the §9 per-query validation budget

    def test_record_without_usage_carries_no_cost(self):
        """No usage -> cost None: the ledger never records a silent $0
        row (evals.pricing convention)."""
        fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
        record = exchange_log_record(outcome_for(fake))
        assert record["usage"] is None
        assert record["cost_usd"] is None

    def test_degraded_exchange_logs_unvalidated(self):
        fake = FakeAdapter(structured_results=[RuntimeError("SYNTHETIC failure")])
        record = exchange_log_record(outcome_for(fake))
        assert record["validated"] is False
        assert record["degraded_reason"]
        assert record["citation_support_rate"] is None

    def test_skipped_exchange_logs_its_reason(self):
        from rag.citation_validator import SKIPPED_NOT_VALIDATABLE

        outcome = validate_exchange(FakeAdapter(), object(), (), config=CONFIG)
        record = exchange_log_record(outcome)
        assert record["validated"] is False
        assert record["skipped_reason"] == SKIPPED_NOT_VALIDATABLE
        assert record["cost_usd"] is None
