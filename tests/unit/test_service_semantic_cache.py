"""The chat pipeline serves the semantic cache honestly (issue #57).

RED suite, against the composed app (TestClient, FakeAdapter, fake
seams) with a REAL ``service.semantic_cache.SemanticCache`` over the
deterministic LOCAL hash embedder:

- A repeat first-turn question replays the stored answer VERBATIM —
  text, citations, badges, sources panel, footer — as ONE ``answer``
  event of kind ``cached`` carrying the ORIGINAL answer's
  ``generated_on`` date, with ZERO adapter calls, ZERO new spend, and
  the rate limiter still metering the request.
- Only clean exchanges populate the cache: refusals, canned, chart,
  degraded-validation and follow-up-turn exchanges never do.
- The serving logs its OWN exchange record (fresh ``exchange_id``,
  route ``cached``, ``cached_from`` = the source exchange, the SOURCE'S
  canonical question text, empty usage) and is feedback-able; a "down"
  verdict on the source OR any serving evicts the entry ("up" never
  does).
- Paused mode serves cache hits (the $0 path needs no budget), the
  semantic cache is consulted before the starter cache, and a miss
  leaves the paused behaviour unchanged.
- The CLIMATE_CHAT_SEMANTIC_CACHE switch: default ON, "0"/"false" off,
  junk refused; the composition root wires a SemanticCache only when
  enabled; the seeded smoke stacks pin it OFF (replay determinism).
- Retention: cached content follows the §9 90-day purge through
  ``run_retention_pass`` and the app lifespan.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from service.app import (
    ANSWER_EVENT,
    ANSWER_KIND_CACHED,
    ANSWER_KIND_CACHED_STARTER,
    ANSWER_KIND_PAUSED,
    ANSWER_KIND_REFUSAL,
    META_EVENT,
)
from service.budget import ServiceMode, SpendTracker
from service.config import (
    ENV_SEMANTIC_CACHE,
    ServiceConfigError,
    load_service_config,
)
from service.exchange_log import build_exchange_record, detach_for_harvest
from service.rate_limit import RateLimiter, RotatingSaltProvider
from service.retention import run_retention_pass
from service.semantic_cache import SemanticCache
from service.starter_cache import STARTER_QUESTIONS
from tests._generation_fixtures import make_refusal, transport_stream_events
from tests._indexing_fixtures import HashEmbeddingModel
from tests._semantic_cache_fixtures import store_kwargs
from tests._service_fixtures import (
    CORPUS_VERSION,
    T0,
    FakeRetrieve,
    FakeValidationOutcome,
    FakeValidationSeam,
    FrozenClock,
    apply_deploy_env,
    classifier_output,
    events_named,
    full_deploy_env,
    make_config,
    make_harness,
    post_chat,
)

QUESTION = "Why are scientists calling this an emergency?"
NEAR_MISS = "When are scientists calling this an emergency?"

#: A #13 badge the warm exchange earns — honesty demands it rides every
#: replay of that answer.
WARM_BADGE = {
    "event": "badge",
    "data": {"sentence_index": 1, "document_index": 0, "reason": "entailment_failed"},
}


def real_cache(clock: FrozenClock) -> SemanticCache:
    return SemanticCache(
        embedding_model=HashEmbeddingModel(),
        corpus_version=CORPUS_VERSION,
        clock=clock,
    )


def semantic_harness(tmp_path, *, badges=(), limiter=None, tracker=None, validation=None):
    clock = FrozenClock()
    harness = make_harness(
        tmp_path,
        clock=clock,
        limiter=limiter,
        tracker=tracker,
        validation=validation or FakeValidationSeam(badge_events=tuple(badges)),
        semantic_cache=real_cache(clock),
    )
    return harness


def program_live_exchange(harness) -> None:
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", transport_stream_events())


def warm(harness, question: str = QUESTION, **json_extra):
    """Run one full live grounded exchange and return its parsed events."""
    program_live_exchange(harness)
    return post_chat(TestClient(harness.app), question, **json_extra)


class TestCacheHitReplaysVerbatim:
    def test_hit_is_meta_plus_one_cached_answer_event(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path, badges=(WARM_BADGE,))
        warm(harness)
        events = post_chat(TestClient(harness.app), QUESTION)
        assert [event["event"] for event in events] == [META_EVENT, ANSWER_EVENT], (
            "a cache hit is exactly meta + one answer event — the stored "
            "sources/badges ride INSIDE the answer data (like cached_starter), "
            "never as fresh-looking stream events"
        )
        assert events[1]["data"]["kind"] == ANSWER_KIND_CACHED

    def test_hit_replays_the_stored_exchange_byte_identical(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path, badges=(WARM_BADGE,))
        original = warm(harness)
        original_text = "".join(e["data"]["text"] for e in events_named(original, "text"))
        original_citations = [e["data"] for e in events_named(original, "citation")]
        original_badges = [e["data"] for e in events_named(original, "badge")]
        original_sources = events_named(original, "sources")[0]["data"]["sources"]
        original_footer = events_named(original, "footer")[0]["data"]["text"]

        events = post_chat(TestClient(harness.app), QUESTION)
        data = events[1]["data"]
        assert data["text"] == original_text
        assert data["citations"] == original_citations
        assert data["badges"] == original_badges, (
            "the original answer earned an 'unverified' badge; the replay "
            "wears it too — caching never launders a verification mark"
        )
        assert data["sources"] == original_sources
        assert data["footer"] == original_footer

    def test_hit_is_dated_with_the_original_answers_date(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        warm(harness)
        harness.clock.advance(timedelta(days=2))
        events = post_chat(TestClient(harness.app), QUESTION)
        assert events[1]["data"]["generated_on"] == T0.date().isoformat(), (
            "the marker date is the ORIGINAL answer's day, never today's — a "
            "cached answer must never be presented as fresh"
        )

    def test_hit_makes_zero_adapter_calls_and_records_zero_spend(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        warm(harness)
        calls_after_warm = len(harness.adapter.calls)
        spent_after_warm = harness.tracker.spent_today()
        post_chat(TestClient(harness.app), QUESTION)
        assert len(harness.adapter.calls) == calls_after_warm, (
            "a cache hit costs $0 BECAUSE it makes no adapter call of any "
            "kind — not classifier, not generation, not validation"
        )
        assert harness.tracker.spent_today() == spent_after_warm

    def test_hit_meta_mints_a_fresh_exchange_id(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        source_meta = warm(harness)[0]["data"]
        hit_meta = post_chat(TestClient(harness.app), QUESTION)[0]["data"]
        assert hit_meta["mode"] == "live"
        assert hit_meta["preamble_note"] is None
        assert hit_meta["exchange_id"]
        assert hit_meta["exchange_id"] != source_meta["exchange_id"], (
            "each serving is its own feedback-able exchange (#56): a fresh "
            "join key per serving, never the source's reused"
        )

    def test_hits_still_meter_the_rate_limit(self, tmp_path) -> None:
        clock = FrozenClock()
        limiter = RateLimiter(clock=clock, salts=RotatingSaltProvider(clock), max_requests=2)
        harness = make_harness(
            tmp_path, clock=clock, limiter=limiter, semantic_cache=real_cache(clock)
        )
        program_live_exchange(harness)
        client = TestClient(harness.app)
        post_chat(client, QUESTION)  # 1: the warm exchange
        post_chat(client, QUESTION)  # 2: a cache hit — still counted
        response = client.post("/chat", json={"question": QUESTION, "history": []})
        assert response.status_code == 429, (
            "$0 to us is not $0 abuse-resistance: cache hits count against "
            "the per-IP limit exactly like live requests"
        )


class TestOnlyCleanExchangesPopulateTheCache:
    def test_near_miss_questions_go_live_not_cached(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        warm(harness)
        program_live_exchange(harness)
        events = post_chat(TestClient(harness.app), NEAR_MISS)
        assert events_named(events, "text"), "the near miss must stream a LIVE answer"
        assert len(harness.adapter.calls_to("generate_stream")) == 2, (
            f"{NEAR_MISS!r} is a different question from {QUESTION!r}: it must "
            "reach the live pipeline, never the cached answer"
        )

    def test_follow_up_turns_neither_consult_nor_populate(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        history = [
            {"role": "user", "content": "an earlier question"},
            {"role": "assistant", "content": "an earlier answer"},
        ]
        # A grounded exchange WITH history completes cleanly...
        warm(harness, QUESTION, history=history)
        # ...but must not have been cached: the identical first-turn
        # question still runs the live pipeline.
        program_live_exchange(harness)
        post_chat(TestClient(harness.app), QUESTION)
        assert len(harness.adapter.calls_to("generate_stream")) == 2
        # And with history attached, even a warmed cache is not consulted.
        program_live_exchange(harness)
        post_chat(TestClient(harness.app), QUESTION, history=history)
        assert len(harness.adapter.calls_to("generate_stream")) == 3

    def test_refusals_are_never_cached(self, tmp_path) -> None:
        clock = FrozenClock()
        harness = make_harness(
            tmp_path,
            clock=clock,
            retrieve=FakeRetrieve(result=make_refusal()),
            semantic_cache=real_cache(clock),
        )
        for _ in range(2):
            harness.adapter.queue("structured", classifier_output())
        client = TestClient(harness.app)
        first = post_chat(client, QUESTION)
        assert events_named(first, "answer")[0]["data"]["kind"] == ANSWER_KIND_REFUSAL
        second = post_chat(client, QUESTION)
        assert events_named(second, "answer")[0]["data"]["kind"] == ANSWER_KIND_REFUSAL
        assert len(harness.adapter.calls_to("structured")) == 2, (
            "an honest refusal is not an answer: the repeat question must "
            "re-run the pipeline, never replay the refusal from cache"
        )

    def test_canned_responses_are_never_cached(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        for _ in range(2):
            harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
        client = TestClient(harness.app)
        post_chat(client, "an out of scope question")
        post_chat(client, "an out of scope question")
        assert len(harness.adapter.calls_to("structured")) == 2, (
            "canned already has its own $0 path — the semantic cache must "
            "not shadow the classifier's routing"
        )

    def test_chart_responses_are_never_cached(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        for _ in range(2):
            harness.adapter.queue(
                "structured",
                classifier_output(scope="chart_request", rewritten="plot the basin anomaly"),
            )
        client = TestClient(harness.app)
        first = post_chat(client, "show me the basin anomaly chart")
        assert events_named(first, "chart"), "the control chart exchange must render"
        second = post_chat(client, "show me the basin anomaly chart")
        assert events_named(second, "chart")
        assert len(harness.adapter.calls_to("structured")) == 2, (
            "chart specs have permalinks — charts are never semantic-cached"
        )

    def test_degraded_validation_exchanges_are_never_cached(self, tmp_path) -> None:
        degraded = FakeValidationSeam(
            outcome=FakeValidationOutcome(
                validated=False,
                support_rate=None,
                degraded_reason="validator transport error",
            )
        )
        harness = semantic_harness(tmp_path, validation=degraded)
        warm(harness)
        program_live_exchange(harness)
        post_chat(TestClient(harness.app), QUESTION)
        assert len(harness.adapter.calls_to("generate_stream")) == 2, (
            "an answer whose validation degraded was never proved clean — it "
            "must not be replayed as a trusted cached answer"
        )


class TestServingExchangeRecord:
    def test_serving_logs_its_own_record_with_the_cached_linkage(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        source_meta = warm(harness)[0]["data"]
        # A near-exact variant (case + whitespace) that clears 0.95.
        variant = "  why are scientists calling this an emergency?  "
        hit_events = post_chat(TestClient(harness.app), variant)
        hit_meta = hit_events[0]["data"]

        records = harness.exchange_log.records()
        assert len(records) == 2
        serving = records[-1]
        assert serving["route"] == "cached"
        assert serving["exchange_id"] == hit_meta["exchange_id"]
        assert serving["cached_from"] == source_meta["exchange_id"], (
            "the FLAGGED linkage shape: the serving's record names its source "
            "exchange via the top-level cached_from key"
        )
        assert serving["question"] == QUESTION, (
            "the logged question is the SOURCE'S canonical text, never the "
            "visitor's raw variant — the cache adds no new query-text surface"
        )
        assert serving["answer_text"] == hit_events[1]["data"]["text"]
        assert serving["usage_records"] == []
        assert serving["validation"] == {}, (
            "validation belongs to the source exchange; the serving reports "
            "none of its own (the support metric is never double-counted)"
        )
        assert serving["feedback"] is None

    def test_build_exchange_record_carries_cached_from(self) -> None:
        record = build_exchange_record(
            question=QUESTION,
            route="cached",
            answer_text="a cached answer",
            retrieved_chunk_ids=[],
            citations=[],
            validation={},
            usage_records=[],
            exclude_from_harvest=False,
            timestamp=T0,
            cached_from="src-exchange-0001",
        )
        assert record["cached_from"] == "src-exchange-0001"

    def test_cached_from_defaults_to_none_on_every_other_route(self) -> None:
        record = build_exchange_record(
            question=QUESTION,
            route="retrieval",
            answer_text="a live answer",
            retrieved_chunk_ids=[],
            citations=[],
            validation={},
            usage_records=[],
            exclude_from_harvest=False,
            timestamp=T0,
        )
        assert record["cached_from"] is None

    def test_detachment_strips_the_cached_linkage(self) -> None:
        # Green guard: detach_for_harvest whitelists content fields, so
        # the cached_from join key can never ride into a published eval
        # case (same rule as exchange_id/feedback).
        record = {
            "exchange_id": "serving-1",
            "timestamp": T0.isoformat(),
            "question": QUESTION,
            "route": "cached",
            "answer_text": "a cached answer",
            "retrieved_chunk_ids": [],
            "citations": [],
            "validation": {},
            "usage_records": [],
            "exclude_from_harvest": False,
            "feedback": None,
            "cached_from": "src-exchange-0001",
        }
        assert "cached_from" not in detach_for_harvest(record)

    def test_a_serving_is_feedback_able_under_its_own_id(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        warm(harness)
        client = TestClient(harness.app)
        hit_meta = post_chat(client, QUESTION)[0]["data"]
        response = client.post(
            "/feedback", json={"exchange_id": hit_meta["exchange_id"], "verdict": "up"}
        )
        assert response.status_code == 204
        serving = harness.exchange_log.records()[-1]
        assert serving["feedback"] == {"verdict": "up"}


class TestThumbsDownEvictsThroughTheFeedbackRoute:
    def test_down_on_the_source_exchange_evicts_the_entry(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        source_meta = warm(harness)[0]["data"]
        client = TestClient(harness.app)
        response = client.post(
            "/feedback", json={"exchange_id": source_meta["exchange_id"], "verdict": "down"}
        )
        assert response.status_code == 204
        program_live_exchange(harness)
        post_chat(client, QUESTION)
        assert len(harness.adapter.calls_to("generate_stream")) == 2, (
            "a thumbs-down poisoned the cached answer: the repeat question "
            "must run live, never replay the downvoted answer"
        )

    def test_down_on_a_serving_evicts_the_source_entry(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        warm(harness)
        client = TestClient(harness.app)
        hit_meta = post_chat(client, QUESTION)[0]["data"]
        response = client.post(
            "/feedback", json={"exchange_id": hit_meta["exchange_id"], "verdict": "down"}
        )
        assert response.status_code == 204
        program_live_exchange(harness)
        post_chat(client, QUESTION)
        assert len(harness.adapter.calls_to("generate_stream")) == 2, (
            "a down verdict on ANY serving of the answer evicts it — the "
            "serving's fresh exchange_id resolves to its source entry"
        )

    def test_thumbs_up_never_evicts(self, tmp_path) -> None:
        harness = semantic_harness(tmp_path)
        source_meta = warm(harness)[0]["data"]
        client = TestClient(harness.app)
        response = client.post(
            "/feedback", json={"exchange_id": source_meta["exchange_id"], "verdict": "up"}
        )
        assert response.status_code == 204
        calls_before = len(harness.adapter.calls)
        events = post_chat(client, QUESTION)
        assert events[1]["data"]["kind"] == ANSWER_KIND_CACHED
        assert len(harness.adapter.calls) == calls_before


class TestPausedModeServing:
    def _paused_harness(self, tmp_path):
        clock = FrozenClock()
        tracker = SpendTracker(daily_budget_usd=0.0, opus_subcap_usd=0.0, clock=clock)
        cache = real_cache(clock)
        harness = make_harness(tmp_path, clock=clock, tracker=tracker, semantic_cache=cache)
        assert harness.tracker.mode() is ServiceMode.PAUSED
        return harness, cache

    def test_a_warm_entry_serves_while_paused(self, tmp_path) -> None:
        harness, cache = self._paused_harness(tmp_path)
        cache.store(**store_kwargs(QUESTION))
        events = post_chat(TestClient(harness.app), QUESTION)
        meta, answer = events[0]["data"], events[1]["data"]
        assert meta["mode"] == "paused"
        assert meta["exchange_id"], "a paused serving is still a feedback-able exchange"
        assert answer["kind"] == ANSWER_KIND_CACHED
        assert answer["text"] == store_kwargs(QUESTION)["answer_text"]
        assert answer["generated_on"] == store_kwargs(QUESTION)["generated_on"]
        assert harness.adapter.calls == [], "paused mode makes zero adapter calls — always"
        serving = harness.exchange_log.records()[-1]
        assert serving["route"] == "cached"
        assert serving["cached_from"] == store_kwargs(QUESTION)["source_exchange_id"]

    def test_semantic_cache_is_consulted_before_the_starter_cache(self, tmp_path) -> None:
        harness, cache = self._paused_harness(tmp_path)
        starter_question = STARTER_QUESTIONS[0]
        cache.store(**store_kwargs(starter_question))
        events = post_chat(TestClient(harness.app), starter_question)
        assert events[1]["data"]["kind"] == ANSWER_KIND_CACHED, (
            "FLAGGED decision 6: a runtime-validated semantic entry outranks "
            "the release-time starter entry for the same question"
        )

    def test_a_miss_leaves_the_paused_behaviour_unchanged(self, tmp_path) -> None:
        harness, _cache = self._paused_harness(tmp_path)
        client = TestClient(harness.app)
        starter = post_chat(client, STARTER_QUESTIONS[0])
        assert events_named(starter, "answer")[0]["data"]["kind"] == ANSWER_KIND_CACHED_STARTER
        furniture = post_chat(client, "a question with no cached answer")
        assert events_named(furniture, "answer")[0]["data"]["kind"] == ANSWER_KIND_PAUSED
        assert furniture[0]["data"]["exchange_id"] is None


class TestConfigSwitch:
    def _load(self, tmp_path, value: str | None):
        env = full_deploy_env(tmp_path)
        if value is not None:
            env[ENV_SEMANTIC_CACHE] = value
        return load_service_config(env)

    def test_default_is_on(self, tmp_path) -> None:
        assert self._load(tmp_path, None).semantic_cache_enabled is True, (
            "the ONE default-true flag: absent means enabled (live wants the "
            "$0 cache; the smoke stacks disable it explicitly)"
        )

    def test_explicit_values_parse(self, tmp_path) -> None:
        assert self._load(tmp_path, "0").semantic_cache_enabled is False
        assert self._load(tmp_path, "false").semantic_cache_enabled is False
        assert self._load(tmp_path, "1").semantic_cache_enabled is True
        assert self._load(tmp_path, "true").semantic_cache_enabled is True

    def test_junk_is_a_typed_refusal(self, tmp_path) -> None:
        with pytest.raises(ServiceConfigError) as excinfo:
            self._load(tmp_path, "maybe")
        assert ENV_SEMANTIC_CACHE in excinfo.value.invalid

    def test_composition_root_wires_the_cache_only_when_enabled(
        self, tmp_path, monkeypatch
    ) -> None:
        from service.main import build_service_deps

        env = full_deploy_env(tmp_path)
        apply_deploy_env(monkeypatch, env)
        cache_dir = tmp_path / "starter-cache"
        from tests._service_fixtures import write_starter_cache

        write_starter_cache(cache_dir)
        base = dict(starter_cache_dir=str(cache_dir), log_dir=str(tmp_path / "logs"))

        disabled = build_service_deps(make_config(semantic_cache_enabled=False, **base))
        assert disabled.semantic_cache is None

        # Enabled is the default — and wiring it must NOT load bge-m3
        # weights at construction (the embedder loads lazily, like every
        # other heavy dependency in the composition root).
        enabled = build_service_deps(make_config(**base))
        assert isinstance(enabled.semantic_cache, SemanticCache)
        assert enabled.semantic_cache.corpus_version == CORPUS_VERSION

    def test_disabled_cache_means_todays_behaviour(self, tmp_path) -> None:
        # semantic_cache=None (the disabled wiring): the identical repeat
        # question runs the FULL live pipeline again — deterministic for
        # the seeded replay smoke, unchanged for every existing suite.
        harness = make_harness(tmp_path)
        for _ in range(2):
            program_live_exchange(harness)
        client = TestClient(harness.app)
        post_chat(client, QUESTION)
        post_chat(client, QUESTION)
        assert len(harness.adapter.calls_to("generate_stream")) == 2


class TestSmokeDeterminismPins:
    def test_every_seeded_smoke_stack_disables_the_cache(self) -> None:
        # FLAGGED decision: the replay smoke's determinism rests on the
        # cache being OFF — a warmed in-process cache would answer repeat
        # questions differently run to run.
        from tests.smoke.test_cutoff_fails_closed import BREACH_ENV
        from tests.smoke.test_starter_live_replay import REPLAY_ENV
        from tests.smoke.test_starter_questions_end_to_end import PAUSED_ENV

        for name, env in (
            ("REPLAY_ENV", REPLAY_ENV),
            ("PAUSED_ENV", PAUSED_ENV),
            ("BREACH_ENV", BREACH_ENV),
        ):
            assert env.get("CLIMATE_CHAT_SEMANTIC_CACHE") == "0", (
                f"{name} must pin the semantic cache OFF for determinism"
            )

    def test_compose_passes_the_switch_through(self) -> None:
        from pathlib import Path

        compose_text = (Path(__file__).parents[2] / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        assert "CLIMATE_CHAT_SEMANTIC_CACHE" in compose_text, (
            "the api service must pass the switch through so the smoke env "
            "dicts can actually reach the composition root"
        )


class _RecordingPurgeCache:
    def __init__(self) -> None:
        self.purge_calls = 0

    def purge_expired(self) -> int:
        self.purge_calls += 1
        return 3


class TestRetentionWiring:
    def test_run_retention_pass_purges_the_semantic_cache(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        fake_cache = _RecordingPurgeCache()
        counts = run_retention_pass(
            harness.exchange_log, harness.limiter, semantic_cache=fake_cache
        )
        assert fake_cache.purge_calls == 1
        assert counts["semantic_cache_entries_removed"] == 3

    def test_run_retention_pass_without_a_cache_keeps_todays_shape(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        counts = run_retention_pass(harness.exchange_log, harness.limiter)
        assert set(counts) == {"exchange_records_removed", "rate_limit_records_removed"}

    def test_the_app_lifespan_purges_the_wired_cache(self, tmp_path) -> None:
        fake_cache = _RecordingPurgeCache()
        harness = make_harness(tmp_path, semantic_cache=fake_cache)
        with TestClient(harness.app):
            pass
        assert fake_cache.purge_calls >= 1, (
            "the startup retention pass must include the semantic cache — "
            "cached content follows the same 90-day bound as the log"
        )
