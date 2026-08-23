"""Privacy-compliant structured logging (issue #22, DESIGN §9, UK GDPR).

Pins ``service.exchange_log``: the structured exchange record (question,
chunk ids, answer, citations, support rate — and NO identifiers), the
redaction scan over a real simulated exchange, the 90-day retention job,
the harvest flow (unsafe exclusion, irreversible detachment, the #56
feedback seam), and the disclosure line's presence on the response
surfaces.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from service.exchange_log import (
    EXCHANGE_LOG_RETENTION_DAYS,
    FORBIDDEN_IDENTIFIER_FIELDS,
    LOGGING_DISCLOSURE,
    ExchangeLog,
    build_exchange_record,
    detach_for_harvest,
    harvest_candidates,
)
from tests._service_fixtures import (
    T0,
    FrozenClock,
    classifier_output,
    events_named,
    make_harness,
    post_chat,
)


def make_record(**overrides):
    values = dict(
        question="What is warming the invented basin?",
        route="retrieval",
        answer_text="The basin has very likely warmed by one point nine degrees.",
        retrieved_chunk_ids=["syn-gen-doc::c0000", "syn-gen-doc::c0001"],
        citations=[{"chunk_id": "syn-gen-doc::c0000", "cited_text": "…"}],
        validation={"citation_support_rate": 0.5, "validated": True},
        usage_records=[
            {"model": "claude-haiku-4-5", "usage": {"input_tokens": 900, "output_tokens": 42}}
        ],
        exclude_from_harvest=False,
        timestamp=T0,
    )
    values.update(overrides)
    return build_exchange_record(**values)


class TestExchangeRecordStructure:
    def test_record_carries_the_designed_fields_exactly(self) -> None:
        record = make_record()
        assert set(record) == {
            "exchange_id",
            "timestamp",
            "question",
            "route",
            "answer_text",
            "retrieved_chunk_ids",
            "citations",
            "validation",
            "usage_records",
            "exclude_from_harvest",
            "feedback",
        }
        assert record["question"] == "What is warming the invented basin?"
        assert record["retrieved_chunk_ids"] == ["syn-gen-doc::c0000", "syn-gen-doc::c0001"]
        assert record["validation"]["citation_support_rate"] == 0.5
        assert record["timestamp"] == T0.isoformat()
        assert record["feedback"] is None  # the #56 seam, empty until #56

    def test_exchange_id_is_random_not_derived(self) -> None:
        """The #56 join key identifies the exchange, never the person:
        two identical exchanges get different ids (nothing about the
        content or client derives the id)."""
        one, two = make_record(), make_record()
        assert one["exchange_id"] != two["exchange_id"]
        uuid.UUID(hex=one["exchange_id"])  # well-formed random UUID hex

    def test_record_is_json_serialisable(self) -> None:
        json.dumps(make_record())


def test_logs_contain_no_raw_ips_or_identifiers(tmp_path) -> None:
    """Log-capture scan over a simulated exchange (the issue-named GATE
    test): a request arriving with every identifier a proxy could add
    produces a log line carrying NONE of them."""
    harness = make_harness(tmp_path)
    harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
    client = TestClient(harness.app)
    response = client.post(
        "/chat",
        json={"question": "a question about the invented basin"},
        headers={
            "x-forwarded-for": "203.0.113.77",
            "user-agent": "SyntheticBrowser/1.0 (privacy probe)",
            "cookie": "session=synthetic-cookie-value",
            "authorization": "Bearer synthetic-token",
        },
    )
    assert response.status_code == 200

    raw_log_text = harness.exchange_log.path.read_text(encoding="utf-8")
    assert raw_log_text.strip(), "the exchange must have been logged"
    for leaked in (
        "203.0.113.77",
        "SyntheticBrowser",
        "synthetic-cookie-value",
        "synthetic-token",
        "testclient",
    ):
        assert leaked not in raw_log_text, f"identifier {leaked!r} leaked into the exchange log"
    for record in harness.exchange_log.records():
        for field in FORBIDDEN_IDENTIFIER_FIELDS:
            assert field not in record


class TestRetention:
    def test_retention_job_deletes_over_90_days(self, tmp_path) -> None:
        clock = FrozenClock()
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=clock)
        old = make_record(timestamp=T0)
        log.append(old)
        clock.advance(timedelta(days=60))
        young = make_record(timestamp=clock())
        log.append(young)

        clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS - 60, seconds=1))
        removed = log.purge_expired()
        assert removed == 1
        remaining = log.records()
        assert [record["exchange_id"] for record in remaining] == [young["exchange_id"]]

    def test_purge_is_a_noop_inside_retention(self, tmp_path) -> None:
        clock = FrozenClock()
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=clock)
        log.append(make_record())
        clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS - 1))
        assert log.purge_expired() == 0
        assert len(log.records()) == 1


class TestHarvestFlow:
    def test_unsafe_exchanges_never_enter_harvest_queue(self) -> None:
        """Consumes #10's exclusion flag: unsafe (and unsafe-suspected,
        finding #86) records are excluded structurally, before any human
        sees a review queue."""
        safe = make_record()
        unsafe = make_record(
            question="an unsafe-classified question",
            route="canned",
            exclude_from_harvest=True,
        )
        queue = harvest_candidates([safe, unsafe])
        ids = [record["exchange_id"] for record in queue]
        assert safe["exchange_id"] in ids
        assert unsafe["exchange_id"] not in ids

    def test_detachment_is_irreversible(self) -> None:
        """Promotion strips every re-joinable field: no timestamp, no
        exchange id, no feedback, no usage — content only."""
        detached = detach_for_harvest(make_record())
        assert set(detached) == {
            "question",
            "retrieved_chunk_ids",
            "answer_text",
            "citations",
            "validation",
        }
        serialised = json.dumps(detached)
        assert T0.isoformat() not in serialised

    def test_detach_refuses_excluded_records(self) -> None:
        with pytest.raises(ValueError):
            detach_for_harvest(make_record(exclude_from_harvest=True))


class TestDisclosureSurfaced:
    def test_disclosure_line_matches_design_9(self) -> None:
        assert LOGGING_DISCLOSURE == (
            "Conversations are logged anonymously to improve the service — "
            "don't share personal details."
        )

    def test_chat_meta_event_carries_the_disclosure(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
        events = post_chat(TestClient(harness.app), "any question")
        meta = events_named(events, "meta")
        assert meta[0]["data"]["disclosure"] == LOGGING_DISCLOSURE

    def test_privacy_page_carries_disclosure_and_lawful_basis(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        client = TestClient(harness.app)
        privacy = client.get("/privacy")
        assert privacy.status_code == 200
        assert LOGGING_DISCLOSURE in privacy.text
        assert "legitimate interests" in privacy.text.lower()
        # /about links the privacy page.
        about = client.get("/about")
        assert "/privacy" in about.text


class TestChatExchangesAreLogged:
    def test_canned_exchange_logged_with_unsafe_exclusion(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue(
            "structured",
            classifier_output(scope="unsafe", unsafe_subtype="harassment"),
        )
        post_chat(TestClient(harness.app), "an abusive message")
        records = harness.exchange_log.records()
        assert len(records) == 1
        assert records[0]["route"] == "canned"
        assert records[0]["exclude_from_harvest"] is True
