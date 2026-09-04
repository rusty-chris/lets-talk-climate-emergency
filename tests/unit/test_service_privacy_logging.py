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
import logging
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path

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
            # Issue #57: the semantic-cache linkage — the source
            # exchange_id on a cached serving, None on every other route.
            "cached_from",
        }
        assert record["question"] == "What is warming the invented basin?"
        assert record["retrieved_chunk_ids"] == ["syn-gen-doc::c0000", "syn-gen-doc::c0001"]
        assert record["validation"]["citation_support_rate"] == 0.5
        assert record["timestamp"] == T0.isoformat()
        assert record["feedback"] is None  # the #56 seam, empty until #56
        assert record["cached_from"] is None  # the #57 seam, set only on cached servings

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


def test_no_captured_log_record_carries_the_client_ip(tmp_path, caplog) -> None:
    """Issue #212 companion guard (green by design at this tier): handling
    a request that arrives with a known forwarded IP produces NO Python
    log record — on the root logger or any propagating child, uvicorn's
    included — whose message carries that IP. The uvicorn access-log leak
    itself lives below the ASGI app and is pinned where it exists: the
    compose command-line assertions (test_service_config.py) and the
    smoke-tier log scan (tests/smoke/test_healthchecks.py). This test
    keeps the APP side of the guarantee from regressing if application
    logging is ever added."""
    import logging

    harness = make_harness(tmp_path)
    harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
    client = TestClient(harness.app)
    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/chat",
            json={"question": "a question from a known address"},
            headers={"x-forwarded-for": "203.0.113.77"},
        )
    assert response.status_code == 200
    for record in caplog.records:
        message = record.getMessage()
        assert "203.0.113.77" not in message, (
            f"a log record from logger {record.name!r} leaked the client IP: {message!r}"
        )


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


class TestRetentionActuallyOperates:
    """Issue #213: the purges exist but nothing invokes them. These pin
    the mechanisms that make the §9 bounds operational: a shipped
    operator runner for the file-backed exchange log, and (in
    test_service_rate_limit.py) the in-process scheduled purge that is
    the ONLY thing able to reach the in-memory rate-limit store."""

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def test_retention_runner_script_purges_exchange_log(self, tmp_path) -> None:
        """The shipped entrypoint exists and, given an old and a fresh
        record on disk, deletes only the old one."""
        assert (self.REPO_ROOT / "scripts" / "run_retention.py").is_file(), (
            "DEPLOYMENT.md §7 points the operator at a retention script — "
            "it must actually be shipped (issue #213)"
        )

        from service.retention import purge_exchange_log

        clock = FrozenClock()
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=clock)
        old = make_record(timestamp=clock())
        log.append(old)
        clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS + 1))
        fresh = make_record(timestamp=clock())
        log.append(fresh)

        removed = purge_exchange_log(tmp_path, clock=clock)
        assert removed == 1
        remaining = [record["exchange_id"] for record in log.records()]
        assert remaining == [fresh["exchange_id"]]

    def test_retention_pass_purges_both_stores(self) -> None:
        """One pass drives BOTH purge_expired seams and reports counts."""
        from service.retention import run_retention_pass

        class RecordingStore:
            def __init__(self, removed: int) -> None:
                self.removed = removed
                self.purge_calls = 0

            def purge_expired(self) -> int:
                self.purge_calls += 1
                return self.removed

        exchange_log, rate_limiter = RecordingStore(3), RecordingStore(2)
        counts = run_retention_pass(exchange_log, rate_limiter)
        assert exchange_log.purge_calls == 1
        assert rate_limiter.purge_calls == 1
        assert counts == {
            "exchange_records_removed": 3,
            "rate_limit_records_removed": 2,
        }

    def test_deployment_runbook_references_the_shipped_runner(self) -> None:
        """DEPLOYMENT.md §7 must describe the retention mechanism that
        exists: the shipped exchange-log runner script by name, and no
        pretence that an external job can purge the in-process
        rate-limit store (its purge is the app's own scheduled task)."""
        runbook = (self.REPO_ROOT / "service" / "DEPLOYMENT.md").read_text(encoding="utf-8")
        assert "run_retention" in runbook, (
            "DEPLOYMENT.md must point at the shipped retention runner "
            "(scripts/run_retention.py), not a hand-waved 'small script'"
        )


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


class TestAtomicRewritesAndTornLogs:
    """Review finding #265 RED — every full-file rewrite of
    ``exchanges.jsonl`` must be atomic (write to a temp file, fsync,
    ``os.replace``), and a torn/truncated TRAILING line — the residue an
    interrupted rewrite or append leaves — is quarantined loudly, never
    poison: ``records()``, ``purge_expired()``, ``record_feedback()``,
    the startup retention pass and the 6-hour in-process purge loop all
    keep working. Today a crash mid-``write_text`` can destroy up to 90
    days of records, and one torn line makes the service unbootable
    (the lifespan's startup pass raises) while silently killing the
    background retention task — the §9 90-day AND 7-day bounds stop.

    A MID-file decode failure is deliberately not pinned here: it
    signals something worse than a torn write and may still raise
    loudly (#265's stated policy). The quarantine destination (e.g. a
    ``<path>.corrupt`` sidecar) is left to the implementer; what is
    pinned is that the torn fragment leaves the live file, that valid
    records survive, and that the event is LOUD (a WARNING-or-worse log
    record), never a silent reparse."""

    TORN_TAIL = '{"exchange_id": "torn-tail", "timesta'

    @staticmethod
    def _torn_marker_logged(caplog) -> bool:
        return any(
            record.levelno >= logging.WARNING
            and any(
                word in record.getMessage().lower()
                for word in ("torn", "corrupt", "quarantin", "truncat")
            )
            for record in caplog.records
        )

    def _seeded_log(self, tmp_path, *, with_expired: bool = False):
        clock = FrozenClock()
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=clock)
        records = {}
        if with_expired:
            records["expired"] = make_record(
                timestamp=clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS + 1)
            )
            log.append(records["expired"])
        records["fresh"] = make_record(timestamp=clock())
        log.append(records["fresh"])
        return log, records

    def _append_torn_tail(self, log) -> None:
        with log.path.open("a", encoding="utf-8") as handle:
            handle.write(self.TORN_TAIL)  # no newline: a genuinely torn write

    def test_feedback_rewrite_never_clobbers_on_a_failed_write(self, tmp_path, monkeypatch) -> None:
        """Interrupt the atomic-replace step mid-``record_feedback``: the
        OLD file must be intact afterwards — the rewrite goes to a temp
        file first, never in place (a crash mid-write today leaves a
        truncated or torn file)."""
        log, _ = self._seeded_log(tmp_path)
        original = log.path.read_bytes()

        def crash(*args, **kwargs):
            raise OSError("simulated crash mid-rewrite (finding #265)")

        monkeypatch.setattr(os, "replace", crash)
        outcome = None
        try:
            outcome = log.record_feedback(log.records()[0]["exchange_id"], "down")
        except OSError:
            pass
        monkeypatch.undo()
        assert outcome is not True, (
            "record_feedback claimed success although the atomic replace "
            "of the rewritten file never happened (finding #265)"
        )
        assert log.path.read_bytes() == original, (
            "a failed rewrite altered exchanges.jsonl in place — the "
            "rewrite must land in a temp file and os.replace into place, "
            "so a crash mid-write leaves the OLD file intact (finding #265)"
        )

    def test_purge_rewrite_never_clobbers_on_a_failed_write(self, tmp_path, monkeypatch) -> None:
        """Same seam for ``purge_expired``: an interrupted purge rewrite
        loses NOTHING — the expired record simply survives until the
        next successful pass."""
        log, _ = self._seeded_log(tmp_path, with_expired=True)
        original = log.path.read_bytes()

        def crash(*args, **kwargs):
            raise OSError("simulated crash mid-rewrite (finding #265)")

        monkeypatch.setattr(os, "replace", crash)
        try:
            log.purge_expired()
        except OSError:
            pass
        monkeypatch.undo()
        assert log.path.read_bytes() == original, (
            "a failed purge rewrite altered exchanges.jsonl in place — "
            "up to 90 days of records are one ill-timed crash from gone "
            "(finding #265)"
        )

    def test_rewrites_fsync_before_replace(self) -> None:
        """Structural: the module flushes the temp file to disk (fsync)
        before the atomic replace — os.replace alone only orders the
        rename, not the data blocks, so a power loss could still
        install an empty file."""
        source = (Path(__file__).resolve().parents[2] / "service" / "exchange_log.py").read_text(
            encoding="utf-8"
        )
        assert "fsync" in source, (
            "service/exchange_log.py never fsyncs the rewritten temp file "
            "before os.replace — the atomic rename can install unflushed "
            "data after a power loss (finding #265)"
        )

    def test_records_tolerates_a_torn_trailing_line(self, tmp_path, caplog) -> None:
        log, records = self._seeded_log(tmp_path)
        self._append_torn_tail(log)
        with caplog.at_level(logging.WARNING):
            retained = log.records()
        assert [r["exchange_id"] for r in retained] == [records["fresh"]["exchange_id"]], (
            "records() must skip a torn trailing line and serve the valid "
            "records — today it raises JSONDecodeError and the whole "
            "triage/harvest read path is dead (finding #265)"
        )
        assert self._torn_marker_logged(caplog), (
            "the torn trailing line was swallowed silently — dropping "
            "operator data must be LOUD (a WARNING+ log record naming the "
            "torn/corrupt/quarantined line, finding #265)"
        )

    def test_purge_quarantines_a_torn_trailing_line_and_still_purges(
        self, tmp_path, caplog
    ) -> None:
        log, records = self._seeded_log(tmp_path, with_expired=True)
        self._append_torn_tail(log)
        with caplog.at_level(logging.WARNING):
            removed = log.purge_expired()
        assert removed >= 1, (
            "purge_expired must still enforce the 90-day bound over a log "
            "with a torn trailing line — today it raises and retention "
            "silently stops (finding #265)"
        )
        surviving_lines = [
            line for line in log.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        surviving = [json.loads(line) for line in surviving_lines]  # every line valid again
        assert [r["exchange_id"] for r in surviving] == [records["fresh"]["exchange_id"]], (
            "after the purge the live file must hold exactly the fresh "
            "record: the expired record purged, the torn fragment "
            "quarantined out of the live log (finding #265)"
        )
        assert self._torn_marker_logged(caplog)

    def test_feedback_lands_despite_a_torn_trailing_line(self, tmp_path, caplog) -> None:
        """A thumbs verdict on a valid record must not 500 because the
        file ends in a torn line — and the rewrite must not launder the
        torn fragment back into the live log as if it were data."""
        log, records = self._seeded_log(tmp_path)
        self._append_torn_tail(log)
        with caplog.at_level(logging.WARNING):
            recorded = log.record_feedback(records["fresh"]["exchange_id"], "down")
        assert recorded is True
        surviving = [
            json.loads(line)
            for line in log.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [r["exchange_id"] for r in surviving] == [records["fresh"]["exchange_id"]], (
            "the feedback rewrite copied the torn fragment back into the "
            "live file (or lost the record) — a torn tail is quarantined "
            "on the next read-modify-write, never silently rewritten "
            "(finding #265)"
        )
        assert surviving[0]["feedback"] == {"verdict": "down"}

    def test_startup_retention_pass_survives_a_torn_log(self, tmp_path) -> None:
        """One torn trailing line must not make the service unbootable:
        the lifespan's startup retention pass quarantines it, /health
        answers, and the 90-day bound is STILL enforced at boot."""
        clock = FrozenClock()
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        expired = make_record(timestamp=clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS + 1))
        fresh = make_record(timestamp=clock())
        (log_dir / "exchanges.jsonl").write_text(
            json.dumps(expired, ensure_ascii=False)
            + "\n"
            + json.dumps(fresh, ensure_ascii=False)
            + "\n"
            + self.TORN_TAIL,
            encoding="utf-8",
        )

        harness = make_harness(tmp_path, clock=clock)
        with TestClient(harness.app) as client:  # enters the lifespan: startup pass runs
            assert client.get("/health").status_code == 200
        retained = [record["exchange_id"] for record in harness.exchange_log.records()]
        assert fresh["exchange_id"] in retained
        assert expired["exchange_id"] not in retained, (
            "the startup pass served /health but did not enforce the "
            "90-day bound over the torn log — recovery must purge, not "
            "just boot (finding #265)"
        )

    def test_periodic_purge_survives_a_failing_pass(self, tmp_path, monkeypatch) -> None:
        """One raising retention pass must not kill the 6-hour background
        loop: the next interval still purges. Today the exception ends
        the asyncio task SILENTLY and both §9 bounds stop enforcing for
        the process lifetime."""
        from dataclasses import replace

        import service.app as service_app
        from service.app import create_app

        monkeypatch.setattr(service_app, "RETENTION_PURGE_INTERVAL", timedelta(milliseconds=20))

        class FlakyPurgeStore:
            """Succeeds at startup (pass 1), raises on the FIRST periodic
            pass (pass 2), then succeeds again."""

            def __init__(self) -> None:
                self.calls = 0

            def purge_expired(self) -> int:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated failing retention pass (finding #265)")
                return 0

        class QuietPurgeStore:
            def purge_expired(self) -> int:
                return 0

        harness = make_harness(tmp_path)
        flaky_log = FlakyPurgeStore()
        deps = replace(harness.deps, exchange_log=flaky_log, rate_limiter=QuietPurgeStore())
        app = create_app(harness.config, deps)

        with TestClient(app):
            deadline = time.monotonic() + 5
            while flaky_log.calls < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
        assert flaky_log.calls >= 3, (
            "after one failing pass the background retention loop never "
            "ran again — the task died silently and the 90-day and 7-day "
            "bounds stopped enforcing for the process lifetime (finding "
            "#265)"
        )
