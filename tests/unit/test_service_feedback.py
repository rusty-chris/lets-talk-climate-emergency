"""Thumbs feedback wired to the eval harvest (issue #56, SotA rec 4).

RED-phase suite — pins, against the composed app (TestClient, fake
seams) and the real ``service.exchange_log`` module:

- ``POST /feedback``: the closed up/down vocabulary (422 outside it),
  the 204-empty-body success, the uniform reveal-nothing 404 (unknown
  and purged ids byte-identical), the hashed-IP rate limit, both-modes
  service, zero adapter calls, and NO new identifier surface.
- The wire join: the meta event's ``exchange_id`` IS the logged
  record's — minted once, ridden to the client, posted back. No new
  SSE event name (``SSE_EVENT_NAMES`` untouched); no provider request
  changes.
- Storage (decision flagged in the red-phase report: in-record
  single-line rewrite, NOT a sidecar log): idempotent replacement,
  byte-identical unmatched lines, no new files, and feedback follows
  its exchange through the 90-day purge.
- Paused mode: the cached-starter path now logs a feedback-able
  exchange under the CANONICAL starter question (decision flagged);
  the paused non-starter furniture still logs nothing and carries
  ``exchange_id: None``.
- Harvest: thumbs-down exchanges surface FIRST in
  ``harvest_candidates``; the safety exclusion still beats any verdict;
  detachment strips feedback so a verdict never rides into a published
  eval set.
- Transparency: /privacy carries the pinned feedback sentence verbatim;
  /about mentions the feedback usage.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from service.app import (
    FEEDBACK_UNKNOWN_EXCHANGE_DETAIL,
    META_EVENT,
)
from service.budget import ServiceMode
from service.exchange_log import (
    EXCHANGE_LOG_RETENTION_DAYS,
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    FEEDBACK_VERDICTS,
    ExchangeLog,
    build_exchange_record,
    detach_for_harvest,
    harvest_candidates,
)
from service.rate_limit import RateLimiter, RotatingSaltProvider
from service.starter_cache import STARTER_QUESTIONS
from tests._generation_fixtures import transport_stream_events
from tests._service_fixtures import (
    T0,
    FrozenClock,
    classifier_output,
    events_named,
    make_harness,
    post_chat,
)


def make_record(**overrides):
    """One valid exchange record (the #22 schema) with overridable fields."""
    values = dict(
        question="What is warming the invented basin?",
        route="retrieval",
        answer_text="The basin has very likely warmed by one point nine degrees.",
        retrieved_chunk_ids=["syn-gen-doc::c0000"],
        citations=[{"chunk_id": "syn-gen-doc::c0000", "cited_text": "…"}],
        validation={"citation_support_rate": 0.5, "validated": True},
        usage_records=[],
        exclude_from_harvest=False,
        timestamp=T0,
    )
    values.update(overrides)
    return build_exchange_record(**values)


def canned_exchange(harness, question: str = "a question about the invented basin"):
    """Drive one canned exchange through /chat; return (events, record)."""
    harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
    events = post_chat(TestClient(harness.app), question)
    records = harness.exchange_log.records()
    assert records, "the canned exchange must have been logged"
    return events, records[-1]


def post_feedback(client, exchange_id: str, verdict: str, **headers):
    return client.post(
        "/feedback",
        json={"exchange_id": exchange_id, "verdict": verdict},
        headers=headers or None,
    )


# ---------------------------------------------------------------------------
# The wire join: meta.exchange_id IS the record's exchange_id.
# ---------------------------------------------------------------------------


class TestMetaCarriesTheExchangeId:
    def test_canned_meta_id_matches_the_logged_record(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        events, record = canned_exchange(harness)
        meta = events_named(events, META_EVENT)[0]["data"]
        assert "exchange_id" in meta, "the meta event must carry the #56 feedback join key"
        assert meta["exchange_id"] == record["exchange_id"], (
            "the id on the wire must BE the id on the logged record — one id, "
            "minted once, joining the visitor's thumbs click to the exchange"
        )
        uuid.UUID(hex=meta["exchange_id"])  # a well-formed random UUID hex

    def test_retrieval_meta_id_matches_the_logged_record(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue("generate_stream", transport_stream_events())
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        meta = events_named(events, META_EVENT)[0]["data"]
        record = harness.exchange_log.records()[-1]
        assert meta["exchange_id"] == record["exchange_id"]

    def test_two_exchanges_get_two_different_ids(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        first, _ = canned_exchange(harness)
        second, _ = canned_exchange(harness)
        first_id = events_named(first, META_EVENT)[0]["data"]["exchange_id"]
        second_id = events_named(second, META_EVENT)[0]["data"]["exchange_id"]
        assert first_id != second_id

    def test_no_new_sse_event_name_was_added(self) -> None:
        """The join key rides the EXISTING meta event: the emit-able
        vocabulary — and therefore the UI's handled/ignored parity and
        every replay request hash — is untouched by #56."""
        import service.app as service_app

        assert service_app.SSE_EVENT_NAMES == frozenset(
            {
                "meta",
                "answer",
                "chart",
                "sources",
                "text",
                "citation",
                "usage",
                "footer",
                "error",
                "badge",
                "validation_degraded",
            }
        )


class TestBuildExchangeRecordCarriesAProvidedId:
    def test_provided_exchange_id_is_carried_verbatim(self) -> None:
        provided = uuid.uuid4().hex
        record = make_record(exchange_id=provided)
        assert record["exchange_id"] == provided

    def test_provided_id_changes_nothing_else(self) -> None:
        provided = uuid.uuid4().hex
        record = make_record(exchange_id=provided)
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
            # Issue #57: the semantic-cache linkage key (None on this route).
            "cached_from",
        }
        assert record["feedback"] is None

    def test_omitted_id_still_mints_a_fresh_random_one(self) -> None:
        one, two = make_record(), make_record()
        assert one["exchange_id"] != two["exchange_id"]


# ---------------------------------------------------------------------------
# POST /feedback: the endpoint contract.
# ---------------------------------------------------------------------------


class TestFeedbackEndpoint:
    def test_verdict_lands_on_the_exchange_record(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        response = post_feedback(TestClient(harness.app), record["exchange_id"], FEEDBACK_UP)
        assert response.status_code == 204
        assert response.content == b"", (
            "success echoes NOTHING — not the id, not the verdict, not the exchange"
        )
        records = harness.exchange_log.records()
        assert len(records) == 1, "feedback must never append a record"
        assert records[0]["feedback"] == {"verdict": FEEDBACK_UP}

    def test_feedback_event_joins_log_record_without_identifiers(self, tmp_path) -> None:
        """The issue-named GATE test: a feedback request arriving with
        every identifier a proxy could add joins the record and stores
        NONE of them — the #56 surface adds no identifier (DESIGN §9)."""
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        response = TestClient(harness.app).post(
            "/feedback",
            json={"exchange_id": record["exchange_id"], "verdict": FEEDBACK_DOWN},
            headers={
                "x-forwarded-for": "203.0.113.99",
                "user-agent": "SyntheticBrowser/2.0 (feedback privacy probe)",
                "cookie": "session=synthetic-feedback-cookie",
                "authorization": "Bearer synthetic-feedback-token",
            },
        )
        assert response.status_code == 204

        updated = harness.exchange_log.records()[0]
        assert updated["feedback"] == {"verdict": FEEDBACK_DOWN}
        assert set(updated["feedback"]) == {"verdict"}, (
            "the feedback mapping carries the verdict and NOTHING else — no "
            "timestamp of its own, no client field, no join key beyond the "
            "record it already lives on"
        )
        raw_log_text = harness.exchange_log.path.read_text(encoding="utf-8")
        for leaked in (
            "203.0.113.99",
            "SyntheticBrowser",
            "synthetic-feedback-cookie",
            "synthetic-feedback-token",
            "testclient",
        ):
            assert leaked not in raw_log_text, f"identifier {leaked!r} leaked via feedback"

    def test_second_verdict_replaces_never_duplicates(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        client = TestClient(harness.app)
        assert post_feedback(client, record["exchange_id"], FEEDBACK_UP).status_code == 204
        assert post_feedback(client, record["exchange_id"], FEEDBACK_DOWN).status_code == 204
        records = harness.exchange_log.records()
        assert len(records) == 1
        assert records[0]["feedback"] == {"verdict": FEEDBACK_DOWN}, (
            "a changed mind REPLACES the verdict — up-then-down is one 'down', never two signals"
        )
        line_count = len(
            [
                line
                for line in harness.exchange_log.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        assert line_count == 1

    @pytest.mark.parametrize(
        "bad_verdict",
        ["sideways", "UP", "Down", "", "thumbs_up", 5, None, ["up"]],
    )
    def test_verdicts_outside_the_closed_vocabulary_are_422(self, tmp_path, bad_verdict) -> None:
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        response = post_feedback(TestClient(harness.app), record["exchange_id"], bad_verdict)
        assert response.status_code == 422, (
            f"verdict {bad_verdict!r} is outside {sorted(FEEDBACK_VERDICTS)} and must be refused"
        )
        assert harness.exchange_log.records()[0]["feedback"] is None, (
            "a refused verdict must write nothing"
        )

    def test_missing_fields_are_422(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        canned_exchange(harness)
        client = TestClient(harness.app)
        assert client.post("/feedback", json={"verdict": FEEDBACK_UP}).status_code == 422
        assert client.post("/feedback", json={"exchange_id": "abc"}).status_code == 422

    def test_unknown_exchange_id_is_a_reveal_nothing_404(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        canned_exchange(harness)
        never_issued = uuid.uuid4().hex
        response = post_feedback(TestClient(harness.app), never_issued, FEEDBACK_UP)
        assert response.status_code == 404
        assert response.json()["detail"] == FEEDBACK_UNKNOWN_EXCHANGE_DETAIL
        assert never_issued not in response.text, "the 404 must not echo the probed id"

    def test_purged_and_never_issued_ids_are_indistinguishable(self, tmp_path) -> None:
        """The 90-day purge must not be observable through the feedback
        endpoint: a 404 for an id that WAS once real is byte-identical
        to a 404 for an id that never existed — the endpoint reveals
        nothing about what was once logged."""
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        purged_id = record["exchange_id"]
        harness.clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS, seconds=1))
        assert harness.exchange_log.purge_expired() == 1

        client = TestClient(harness.app)
        purged_response = post_feedback(client, purged_id, FEEDBACK_DOWN)
        never_response = post_feedback(client, uuid.uuid4().hex, FEEDBACK_DOWN)
        assert purged_response.status_code == 404
        assert never_response.status_code == 404
        assert purged_response.content == never_response.content

    def test_feedback_is_rate_limited_by_the_hashed_ip_limiter(self, tmp_path) -> None:
        clock = FrozenClock()
        limiter = RateLimiter(clock=clock, salts=RotatingSaltProvider(clock), max_requests=2)
        harness = make_harness(tmp_path, clock=clock, limiter=limiter)
        _, record = canned_exchange(harness)
        client = TestClient(harness.app)
        # The canned /chat call above consumed one slot of the SHARED
        # limiter; the second feedback POST is the third request and must
        # be refused.
        first = post_feedback(client, record["exchange_id"], FEEDBACK_UP)
        assert first.status_code == 204
        refused = post_feedback(client, record["exchange_id"], FEEDBACK_DOWN)
        assert refused.status_code == 429
        assert record["exchange_id"] not in refused.text, "the 429 echoes nothing"
        assert "testclient" not in refused.text
        # Over-threshold feedback writes nothing.
        assert harness.exchange_log.records()[0]["feedback"] == {"verdict": FEEDBACK_UP}

    def test_limiter_store_stays_unjoinable_to_the_exchange_log(self, tmp_path) -> None:
        """The rate-limit store must not grow a query-side join key from
        the feedback path: no stored limiter record ever carries the
        exchange_id (the §9 structural-separation guarantee holds)."""
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        post_feedback(TestClient(harness.app), record["exchange_id"], FEEDBACK_UP)
        serialised = json.dumps(list(harness.limiter.stored_records()))
        assert record["exchange_id"] not in serialised

    def test_feedback_makes_zero_adapter_calls(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        record = make_record()
        harness.exchange_log.append(record)
        response = post_feedback(TestClient(harness.app), record["exchange_id"], FEEDBACK_UP)
        assert response.status_code == 204
        assert harness.adapter.calls == [], "feedback is $0: no adapter call of any kind"


# ---------------------------------------------------------------------------
# Paused mode: feedback still records; cached answers are valid signal.
# ---------------------------------------------------------------------------


def paused_harness(tmp_path):
    harness = make_harness(tmp_path)
    harness.tracker.record_usage(
        "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
    )
    assert harness.tracker.mode() is ServiceMode.PAUSED
    return harness


class TestFeedbackWhilePaused:
    def test_feedback_on_an_earlier_exchange_records_while_paused(self, tmp_path) -> None:
        """The budget cutting off mid-session must not silence the
        visitor: rating an already-answered exchange still lands."""
        harness = make_harness(tmp_path)
        _, record = canned_exchange(harness)
        harness.tracker.record_usage(
            "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
        )
        assert harness.tracker.mode() is ServiceMode.PAUSED
        calls_before = len(harness.adapter.calls)
        response = post_feedback(TestClient(harness.app), record["exchange_id"], FEEDBACK_DOWN)
        assert response.status_code == 204
        assert harness.exchange_log.records()[0]["feedback"] == {"verdict": FEEDBACK_DOWN}
        assert len(harness.adapter.calls) == calls_before, "still zero adapter calls while paused"

    def test_cached_starter_answer_is_feedbackable_while_paused(self, tmp_path) -> None:
        """DECISION (flagged in the red-phase report): the paused
        cached-starter path logs an exchange record — under the
        CANONICAL starter question, never the visitor's raw text — and
        mints the meta exchange_id for it, so feedback on a cached
        answer (valid signal about the cached entry) has a record to
        join."""
        harness = paused_harness(tmp_path)
        starter_question = STARTER_QUESTIONS[0]
        # Typed with stray whitespace: the cache lookup normalises; the
        # LOG must carry the canonical public starter text only.
        events = post_chat(TestClient(harness.app), f"  {starter_question}  ")
        meta = events_named(events, META_EVENT)[0]["data"]
        assert meta["exchange_id"], "a cached-starter answer must carry a feedback join key"
        uuid.UUID(hex=meta["exchange_id"])

        records = harness.exchange_log.records()
        assert len(records) == 1
        assert records[0]["exchange_id"] == meta["exchange_id"]
        assert records[0]["route"] == "cached_starter"
        assert records[0]["question"] == starter_question, (
            "the logged question is the CANONICAL public starter question, "
            "never the visitor's raw typed text"
        )
        assert records[0]["exclude_from_harvest"] is False

        response = post_feedback(TestClient(harness.app), meta["exchange_id"], FEEDBACK_DOWN)
        assert response.status_code == 204
        assert harness.exchange_log.records()[0]["feedback"] == {"verdict": FEEDBACK_DOWN}

    def test_paused_furniture_carries_no_exchange_id_and_logs_nothing(self, tmp_path) -> None:
        """The paused NON-starter path is unchanged by #56: not an
        exchange, nothing logged, nothing to rate — meta carries the
        key with value None (the key is always present on the wire)."""
        harness = paused_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "a personal question typed while paused")
        meta = events_named(events, META_EVENT)[0]["data"]
        assert "exchange_id" in meta
        assert meta["exchange_id"] is None
        assert harness.exchange_log.records() == []


# ---------------------------------------------------------------------------
# Storage: the in-record rewrite (decision flagged), purge interplay.
# ---------------------------------------------------------------------------


class TestFeedbackStorage:
    def test_record_feedback_rewrites_only_the_matched_line(self, tmp_path) -> None:
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=FrozenClock())
        first, second = make_record(), make_record(question="a second invented question")
        log.append(first)
        log.append(second)
        first_line_before = log.path.read_text(encoding="utf-8").splitlines()[0]

        assert log.record_feedback(second["exchange_id"], FEEDBACK_DOWN) is True

        lines = [line for line in log.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2, "feedback never appends a line"
        assert lines[0] == first_line_before, "unmatched lines stay byte-identical"
        records = log.records()
        assert [record["exchange_id"] for record in records] == [
            first["exchange_id"],
            second["exchange_id"],
        ], "order is preserved"
        assert records[0]["feedback"] is None
        assert records[1]["feedback"] == {"verdict": FEEDBACK_DOWN}

    def test_record_feedback_creates_no_sidecar_file(self, tmp_path) -> None:
        """The flagged storage decision, held structurally: feedback
        lives IN the exchange log — no second store whose retention,
        purge and detachment interplay would each need re-proving."""
        log_dir = tmp_path / "logs"
        log = ExchangeLog(log_dir / "exchanges.jsonl", clock=FrozenClock())
        record = make_record()
        log.append(record)
        files_before = sorted(path.name for path in log_dir.iterdir())
        log.record_feedback(record["exchange_id"], FEEDBACK_UP)
        assert sorted(path.name for path in log_dir.iterdir()) == files_before

    def test_unknown_exchange_id_returns_false_and_touches_nothing(self, tmp_path) -> None:
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=FrozenClock())
        log.append(make_record())
        bytes_before = log.path.read_bytes()
        assert log.record_feedback(uuid.uuid4().hex, FEEDBACK_UP) is False
        assert log.path.read_bytes() == bytes_before

    def test_record_feedback_rejects_verdicts_outside_the_vocabulary(self, tmp_path) -> None:
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=FrozenClock())
        record = make_record()
        log.append(record)
        with pytest.raises(ValueError):
            log.record_feedback(record["exchange_id"], "sideways")
        assert log.records()[0]["feedback"] is None

    def test_replacement_is_idempotent_at_the_store(self, tmp_path) -> None:
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=FrozenClock())
        record = make_record()
        log.append(record)
        log.record_feedback(record["exchange_id"], FEEDBACK_UP)
        log.record_feedback(record["exchange_id"], FEEDBACK_DOWN)
        log.record_feedback(record["exchange_id"], FEEDBACK_DOWN)
        records = log.records()
        assert len(records) == 1
        assert records[0]["feedback"] == {"verdict": FEEDBACK_DOWN}

    def test_feedback_follows_its_exchange_through_the_purge(self, tmp_path) -> None:
        """DESIGN §9's 90-day bound covers the verdict too: because the
        feedback lives ON the record, the purge deletes them together —
        no orphaned verdict, no residue of the join key."""
        clock = FrozenClock()
        log = ExchangeLog(tmp_path / "exchanges.jsonl", clock=clock)
        record = make_record(timestamp=clock())
        log.append(record)
        log.record_feedback(record["exchange_id"], FEEDBACK_DOWN)
        clock.advance(timedelta(days=EXCHANGE_LOG_RETENTION_DAYS, seconds=1))
        assert log.purge_expired() == 1
        assert log.records() == []
        remaining_text = log.path.read_text(encoding="utf-8")
        assert record["exchange_id"] not in remaining_text


# ---------------------------------------------------------------------------
# Harvest: thumbs-down first; detachment strips the verdict.
# ---------------------------------------------------------------------------


def rated(record, verdict):
    record = dict(record)
    record["feedback"] = {"verdict": verdict}
    return record


class TestTriageOrdering:
    def test_triage_lists_negative_exchanges_first(self) -> None:
        """The issue-named triage pin: thumbs-down exchanges surface
        FIRST in the harvest queue (a visitor said the answer failed
        them — the reviewer sees those before anything else), stably in
        original order, followed by the rest in original order."""
        upvoted = rated(make_record(question="an upvoted question"), FEEDBACK_UP)
        unrated = make_record(question="an unrated question")
        first_down = rated(make_record(question="the first downvoted question"), FEEDBACK_DOWN)
        second_down = rated(make_record(question="the second downvoted question"), FEEDBACK_DOWN)

        queue = harvest_candidates([upvoted, unrated, first_down, second_down])
        assert [record["exchange_id"] for record in queue] == [
            first_down["exchange_id"],
            second_down["exchange_id"],
            upvoted["exchange_id"],
            unrated["exchange_id"],
        ]

    def test_a_downvote_never_overrides_the_safety_exclusion(self) -> None:
        excluded_down = rated(
            make_record(question="an unsafe excluded question", exclude_from_harvest=True),
            FEEDBACK_DOWN,
        )
        safe = make_record()
        queue = harvest_candidates([excluded_down, safe])
        ids = [record["exchange_id"] for record in queue]
        assert excluded_down["exchange_id"] not in ids, (
            "the structural unsafe exclusion beats ANY verdict — a thumbs-down "
            "cannot pull an excluded exchange into a human review queue"
        )
        assert ids == [safe["exchange_id"]]


class TestPromotionStaysDetached:
    def test_promoted_exchange_enters_gold_set_detached(self) -> None:
        """The issue-named detachment pin, over a RATED record: the
        promoted eval case carries content only — the verdict and the
        join key must not survive detachment (feedback steered the
        triage; it never rides into the published set as an
        identifier)."""
        record = rated(make_record(), FEEDBACK_DOWN)
        detached = detach_for_harvest(record)
        assert set(detached) == {
            "question",
            "retrieved_chunk_ids",
            "answer_text",
            "citations",
            "validation",
        }
        serialised = json.dumps(detached)
        assert record["exchange_id"] not in serialised
        assert "feedback" not in serialised
        assert "verdict" not in serialised


# ---------------------------------------------------------------------------
# Transparency: the disclosure additions (issue scope: privacy + about).
# ---------------------------------------------------------------------------


class TestFeedbackDisclosure:
    def test_the_chat_one_liner_is_unchanged(self) -> None:
        """The DESIGN §9 chat disclosure line is pinned verbatim
        elsewhere and #56 does NOT extend it — feedback disclosure is a
        separate added sentence on /privacy."""
        from service.exchange_log import LOGGING_DISCLOSURE

        assert LOGGING_DISCLOSURE == (
            "Conversations are logged anonymously to improve the service — "
            "don't share personal details."
        )

    def test_privacy_page_carries_the_feedback_sentence_verbatim(self) -> None:
        from service.transparency import FEEDBACK_LOGGING_DISCLOSURE, render_privacy_page
        from tests._transparency_fixtures import page_text

        assert FEEDBACK_LOGGING_DISCLOSURE == (
            "If you rate an answer with the thumbs up/down buttons, the rating "
            "is stored on that conversation's anonymous log record — it adds "
            "no identifier, is deleted with the record, and is stripped out "
            "before any exchange is promoted into our published evaluation sets."
        )
        assert FEEDBACK_LOGGING_DISCLOSURE in page_text(render_privacy_page()), (
            "/privacy must disclose the feedback collection with the pinned sentence, verbatim"
        )

    def test_about_page_mentions_the_feedback_usage(self, tmp_path) -> None:
        from service.transparency import render_about_page
        from tests._transparency_fixtures import page_text

        about = page_text(
            render_about_page(
                eval_results_text="synthetic eval results",
                corpus_vintage="2026-08",
            )
        ).lower()
        assert "thumbs" in about, "/about must mention the thumbs feedback feature"
        assert "improve" in about, "/about must say what the feedback is used for"
