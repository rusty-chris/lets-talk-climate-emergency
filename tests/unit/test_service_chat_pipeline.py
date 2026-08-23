"""The chat endpoint wires the full merged pipeline (issue #22).

Pins, against the composed app (TestClient, FakeAdapter, fake seams):
query→route wiring (retrieval → generation stream → validation events |
chart → planner → renderer | canned direct), the SSE contract
passthrough (the #12 text/citation/usage/footer/error vocabulary and the
#13 badge events), preamble_note and disclosure as response-surface
furniture in the leading ``meta`` event, server-side spend accumulation
from the exchange's usage records, and the Opus best-mode wiring through
the #186-hardened gate with sub-cap fallback.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from charts.planner import ChartRefusal, CurationGap
from charts.spec import spec_hash
from rag.generation import GENERATION_MODEL_DEFAULT, OPUS_BEST_MODEL
from rag.query import ENGLISH_ANSWER_NOTE, SAMARITANS_PHONE, Route
from service.app import (
    ANSWER_EVENT,
    ANSWER_KIND_CANNED,
    ANSWER_KIND_REFUSAL,
    CHART_EVENT,
    META_EVENT,
)
from tests._generation_fixtures import make_refusal, transport_stream_events
from tests._service_fixtures import (
    SYNTHETIC_SPEC,
    FakePlanner,
    FakeRetrieve,
    classifier_output,
    events_named,
    make_config,
    make_harness,
    post_chat,
    stream_usage,
    usage_cost,
)


def retrieval_harness(tmp_path, **kwargs):
    """A harness programmed for one full retrieval-route exchange."""
    harness = make_harness(tmp_path, **kwargs)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", transport_stream_events())
    return harness


class TestRetrievalRoute:
    def test_full_pipeline_event_sequence(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]

        # meta first (furniture), then the #12 vocabulary in transport
        # order, footer as the generation terminal, badges strictly after.
        assert names[0] == META_EVENT
        assert "text" in names and "citation" in names and "usage" in names
        assert "footer" in names
        assert names.index("footer") > max(
            index for index, name in enumerate(names) if name in ("text", "citation", "usage")
        )
        meta = events[0]["data"]
        assert meta["mode"] == "live"
        assert meta["preamble_note"] is None
        assert meta["disclosure"]

        # The streamed text and resolved citation pass through unchanged.
        text = "".join(event["data"]["text"] for event in events_named(events, "text"))
        assert text == "The basin has very likely warmed by one point nine degrees."
        citation = events_named(events, "citation")[0]["data"]
        assert citation["chunk_id"] == "syn-gen-doc::c0000"
        assert citation["clears_threshold"] is True

    def test_exactly_one_structured_and_one_stream_call(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert len(harness.adapter.calls_to("structured")) == 1
        assert len(harness.adapter.calls_to("generate_stream")) == 1
        assert harness.adapter.calls_to("generate") == []

    def test_retrieve_receives_the_routed_decision(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert len(harness.retrieve.calls) == 1
        decision = harness.retrieve.calls[0]
        assert decision.route is Route.RETRIEVAL
        assert decision.retrieval_query == "synthetic rewritten query"

    def test_history_reaches_the_classifier_call(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)
        history = [
            {"role": "user", "content": "an earlier question"},
            {"role": "assistant", "content": "an earlier answer"},
        ]
        post_chat(TestClient(harness.app), "What about there?", history=history)
        payload = harness.adapter.calls_to("structured")[0].payload
        contents = [message["content"] for message in payload["messages"]]
        assert "an earlier question" in contents
        assert "an earlier answer" in contents

    def test_badge_events_appended_after_the_stream(self, tmp_path) -> None:
        badge = {"event": "badge", "data": {"sentence_index": 0, "document_index": 0}}
        from tests._service_fixtures import FakeValidationSeam

        seam = FakeValidationSeam(badge_events=(badge,))
        harness = retrieval_harness(tmp_path, validation=seam)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]
        assert "badge" in names
        assert names.index("badge") > names.index("footer")
        assert seam.validate_calls, "the validation seam was never driven"

    def test_spend_accumulates_from_the_stream_usage_record(self, tmp_path) -> None:
        """Server-side accounting: the generation stream's usage event is
        recorded against the daily cap via the single pricing source."""
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")
        expected = usage_cost(GENERATION_MODEL_DEFAULT, stream_usage())
        assert harness.tracker.spent_today() == pytest.approx(expected)

    def test_exchange_is_logged_with_pipeline_facts(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")
        records = harness.exchange_log.records()
        assert len(records) == 1
        record = records[0]
        assert record["question"] == "Why is the basin warming?"
        assert record["route"] == "retrieval"
        assert "syn-gen-doc::c0000" in record["retrieved_chunk_ids"]
        assert record["answer_text"].startswith("The basin has very likely warmed")
        assert record["validation"]["citation_support_rate"] == 0.75
        assert record["exclude_from_harvest"] is False


class TestErrorPassthrough:
    def test_error_terminated_stream_passes_through_without_footer(self, tmp_path) -> None:
        """Finding #184 honoured at the surface: a truncated transport
        stream surfaces the typed error event and NO footer, and the
        validation seam sees the error-terminated transcript (the #13
        contract then appends nothing)."""
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output())
        truncated = transport_stream_events()[:-1]  # drop message_stop
        harness.adapter.queue("generate_stream", truncated)

        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]
        assert "error" in names
        assert "footer" not in names
        assert "badge" not in names
        error = events_named(events, "error")[0]["data"]
        assert error["type"] == "incomplete_stream"


class TestPreambleNoteFurniture:
    def test_non_english_query_gets_the_note_in_meta(self, tmp_path) -> None:
        """The #12 orchestrator ratification: preamble_note is attached at
        the service surface, never in any prompt."""
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output(language="de"))
        harness.adapter.queue("generate_stream", transport_stream_events())
        events = post_chat(TestClient(harness.app), "Warum erwärmt sich das Becken?")
        assert events[0]["data"]["preamble_note"] == ENGLISH_ANSWER_NOTE
        # The note never rides into the generation request.
        stream_payload = harness.adapter.calls_to("generate_stream")[0].payload
        assert ENGLISH_ANSWER_NOTE not in str(stream_payload)


class TestCannedRoutes:
    def test_unsafe_self_harm_served_direct_with_zero_generation_calls(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue(
            "structured", classifier_output(scope="unsafe", unsafe_subtype="self_harm")
        )
        events = post_chat(TestClient(harness.app), "a self-harm message")
        answers = events_named(events, ANSWER_EVENT)
        assert len(answers) == 1
        assert answers[0]["data"]["kind"] == ANSWER_KIND_CANNED
        assert SAMARITANS_PHONE in answers[0]["data"]["text"]
        assert harness.adapter.calls_to("generate") == []
        assert harness.adapter.calls_to("generate_stream") == []
        assert harness.retrieve.calls == []

    def test_out_of_scope_served_direct(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
        events = post_chat(TestClient(harness.app), "who won the invented cup final?")
        answers = events_named(events, ANSWER_EVENT)
        assert answers[0]["data"]["kind"] == ANSWER_KIND_CANNED
        assert harness.adapter.calls_to("generate_stream") == []


class TestRefusalRoute:
    def test_honest_refusal_served_without_generation(self, tmp_path) -> None:
        harness = make_harness(tmp_path, retrieve=FakeRetrieve(result=make_refusal()))
        harness.adapter.queue("structured", classifier_output())
        events = post_chat(TestClient(harness.app), "something the corpus cannot answer")
        answers = events_named(events, ANSWER_EVENT)
        assert answers[0]["data"]["kind"] == ANSWER_KIND_REFUSAL
        assert "corpus does cover" in answers[0]["data"]["text"]
        assert harness.adapter.calls_to("generate_stream") == []


class TestChartRoute:
    def test_chart_request_plans_renders_stores_and_links(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue(
            "structured", classifier_output(scope="chart_request", rewritten="plot basin co2")
        )
        events = post_chat(TestClient(harness.app), "Show me basin CO2")
        charts = events_named(events, CHART_EVENT)
        assert len(charts) == 1
        data = charts[0]["data"]
        expected_hash = spec_hash(SYNTHETIC_SPEC)
        assert data["spec_hash"] == expected_hash
        assert data["permalink"] == f"/chart/{expected_hash}"
        assert data["alt_text"]

        # The planner got the rewritten request; the renderer got the spec;
        # the spec is stored so the permalink serves immediately.
        assert harness.planner.calls == ["plot basin co2"]
        assert harness.renderer.calls == [SYNTHETIC_SPEC]
        assert harness.spec_store.get(expected_hash) == SYNTHETIC_SPEC
        client = TestClient(harness.app)
        assert client.get(f"/chart/{expected_hash}").status_code == 200

    def test_chart_refusal_is_an_honest_answer(self, tmp_path) -> None:
        refusal = ChartRefusal(
            message="I can't chart that; nearest available: syn_annual_anomaly.",
            gap=CurationGap(
                chart_request="plot the invented seagrass index",
                requested_data="seagrass index",
                nearest_datasets=("syn_annual_anomaly",),
            ),
        )
        harness = make_harness(tmp_path, planner=FakePlanner(result=refusal))
        harness.adapter.queue("structured", classifier_output(scope="chart_request"))
        events = post_chat(TestClient(harness.app), "Plot the invented seagrass index")
        answers = events_named(events, ANSWER_EVENT)
        assert answers[0]["data"]["kind"] == ANSWER_KIND_REFUSAL
        assert "nearest available" in answers[0]["data"]["text"]
        assert harness.renderer.calls == []

    def test_planner_usage_feeds_the_spend_tracker(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output(scope="chart_request"))
        post_chat(TestClient(harness.app), "Show me basin CO2")
        # The FakePlanner reports {"input_tokens": 200} (a Haiku call).
        expected = usage_cost("claude-haiku-4-5", {"input_tokens": 200})
        assert harness.tracker.spent_today() == pytest.approx(expected)


class TestBestModeWiring:
    def opus_harness(self, tmp_path, *, exhaust_subcap: bool):
        config = make_config(
            starter_cache_dir=str(tmp_path / "starter-cache"),
            log_dir=str(tmp_path / "logs"),
            best_mode_enabled=True,
            daily_budget_usd=100.0,
            opus_subcap_usd=0.05,
        )
        harness = make_harness(tmp_path, config=config)
        if exhaust_subcap:
            harness.tracker.record_usage(
                OPUS_BEST_MODEL, {"input_tokens": 8_000, "output_tokens": 1_000}
            )
        harness.adapter.queue("structured", classifier_output())
        harness.adapter.queue("generate_stream", transport_stream_events())
        return harness

    def test_best_mode_streams_with_the_gated_model_and_a_guard(self, tmp_path) -> None:
        harness = self.opus_harness(tmp_path, exhaust_subcap=False)
        post_chat(TestClient(harness.app), "Why is the basin warming?")
        config = harness.adapter.calls_to("generate_stream")[0].payload["config"]
        assert config["model"] == OPUS_BEST_MODEL

    def test_subcap_breach_falls_back_to_the_default_model(self, tmp_path) -> None:
        """DECISION (flagged for ratification): when the Opus sub-cap is
        spent but the daily cap has room, the query proceeds on the
        default model — Haiku continues under the daily cap; the visitor
        gets an answer, not a refusal."""
        harness = self.opus_harness(tmp_path, exhaust_subcap=True)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        stream_calls = harness.adapter.calls_to("generate_stream")
        assert len(stream_calls) == 1
        assert stream_calls[0].payload["config"]["model"] == GENERATION_MODEL_DEFAULT
        assert "footer" in [event["event"] for event in events]
