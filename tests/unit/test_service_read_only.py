"""The fail-closed read-only paused state (issue #22 GATE, ADR-015 as amended).

Pins, against the composed app (TestClient, all adapters faked): breach
serves the read-only state — cached starter answers, chart permalinks
and the static surfaces stay up; chat/chart generation refuse with the
dated "paused for today" response; ZERO adapter calls while paused — and
the starter cache's pinned structure (release-time generation is a
release step; its output must load, be complete, and be clearly dated).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.app import (
    ANSWER_EVENT,
    ANSWER_KIND_CACHED_STARTER,
    ANSWER_KIND_PAUSED,
    META_EVENT,
)
from service.budget import ServiceMode
from service.starter_cache import (
    STARTER_QUESTIONS,
    StarterCacheError,
    load_starter_cache,
)
from tests._service_fixtures import (
    STARTER_GENERATED_ON,
    ServiceHarness,
    events_named,
    make_harness,
    post_chat,
    starter_cache_payload,
    write_starter_cache,
)


def paused_harness(tmp_path) -> ServiceHarness:
    """A harness whose tracker has already breached the daily cap."""
    harness = make_harness(tmp_path)
    harness.tracker.record_usage(
        "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
    )
    assert harness.tracker.mode() is ServiceMode.PAUSED
    return harness


def test_read_only_serves_cached_answers_and_static_surfaces(tmp_path) -> None:
    """The paused state is read-only, not dark (ADR-015 as amended)."""
    harness = paused_harness(tmp_path)
    client = TestClient(harness.app)

    # Static surfaces all serve.
    for path in ("/about", "/privacy", "/sources", "/voices"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} must serve while paused"
        assert response.text.strip(), f"{path} served an empty body"

    # A stored chart spec's permalink still serves ($0 — no LLM call).
    spec_hash = harness.spec_store.put({"spec_version": "1.0", "title": "flagship (invented)"})
    assert client.get(f"/chart/{spec_hash}").status_code == 200

    # A starter-topic question serves the cached answer, clearly dated.
    events = post_chat(client, STARTER_QUESTIONS[0])
    answers = events_named(events, ANSWER_EVENT)
    assert len(answers) == 1
    data = answers[0]["data"]
    assert data["kind"] == ANSWER_KIND_CACHED_STARTER
    assert data["text"].startswith("Cached synthetic starter answer")
    assert data["generated_on"] == STARTER_GENERATED_ON
    assert data["footer"]
    assert data["citations"]

    # Any other question gets the dated paused response.
    events = post_chat(client, "What is happening to the invented basin?")
    answers = events_named(events, ANSWER_EVENT)
    assert len(answers) == 1
    assert answers[0]["data"]["kind"] == ANSWER_KIND_PAUSED
    assert "paused for today" in answers[0]["data"]["text"]
    assert harness.clock().date().isoformat() in answers[0]["data"]["text"]


def test_meta_event_reports_paused_mode(tmp_path) -> None:
    harness = paused_harness(tmp_path)
    events = post_chat(TestClient(harness.app), "anything at all")
    meta = events_named(events, META_EVENT)
    assert meta and events[0]["event"] == META_EVENT
    assert meta[0]["data"]["mode"] == "paused"


def test_no_llm_calls_in_paused_state(tmp_path) -> None:
    """ZERO adapter calls while paused — not even the classifier: the
    read-only state costs $0 by construction, not by hope."""
    harness = paused_harness(tmp_path)
    client = TestClient(harness.app)

    post_chat(client, STARTER_QUESTIONS[0])
    post_chat(client, "Show me a chart of the invented basin")
    post_chat(client, "Why is the basin warming?")
    client.get("/about")
    spec_hash = harness.spec_store.put({"spec_version": "1.0", "title": "cached (invented)"})
    client.get(f"/chart/{spec_hash}")

    assert harness.adapter.calls == []
    assert harness.retrieve.calls == []
    assert harness.planner.calls == []


def test_chart_generation_refuses_while_paused(tmp_path) -> None:
    """The chart route is generation too: paused means the planner is
    never consulted and the paused answer comes back."""
    harness = paused_harness(tmp_path)
    events = post_chat(TestClient(harness.app), "Plot the basin anomaly since 1990")
    answers = events_named(events, ANSWER_EVENT)
    assert len(answers) == 1
    assert answers[0]["data"]["kind"] == ANSWER_KIND_PAUSED
    assert harness.planner.calls == []


def test_health_still_ok_while_paused(tmp_path) -> None:
    """A paused service is alive — monitoring must not read pause as
    outage (and the smoke healthcheck must keep passing)."""
    harness = paused_harness(tmp_path)
    response = TestClient(harness.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_paused_exchanges_are_not_logged_as_content(tmp_path) -> None:
    """A paused refusal is furniture, not an exchange: the question is
    NOT recorded (nothing was retrieved, answered, or spent), so the
    paused state cannot become a quiet full-text query log."""
    harness = paused_harness(tmp_path)
    post_chat(TestClient(harness.app), "a personal question typed while paused")
    serialised = json.dumps(harness.exchange_log.records())
    assert "a personal question typed while paused" not in serialised


class TestPausedStackServesItsPermalinks:
    """Issue #214: 'chart permalinks stay up while paused' (ADR-015) must
    be deliverable under the environment the runbook documents.

    Re-rendering a stored ~1 KB spec fundamentally needs the dataset
    manifest + landed chart pack, yet DEPLOYMENT.md §2 classes those as
    'Live-generation-only … not for the paused stack' — a runbook-
    following paused deploy 500s on every flagship permalink, exactly
    when the read-only state is the whole product.

    DECISION (flagged for ratification): the render inputs are REQUIRED
    deployment artifacts for any stack whose chart-spec store holds
    specs to serve (the paused/read-only stack included), validated
    loudly at BOOT — not persisted-rendered-artefact permalinks (the
    rejected alternative), and not a per-request 500. An empty store
    (the dev compose stub) requires nothing and 404s cleanly.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def test_boot_with_stored_permalinks_requires_the_render_inputs(
        self, monkeypatch, tmp_path
    ) -> None:
        """A deploy whose spec store holds permalinks refuses to boot
        without the manifest + pack, naming both variables at once."""
        import service.main
        from service.app import ServiceStartupError
        from service.chart_store import ChartSpecStore
        from tests._service_fixtures import SYNTHETIC_SPEC, apply_deploy_env, full_deploy_env

        store_dir = tmp_path / "chart-specs"
        ChartSpecStore(store_dir).put(SYNTHETIC_SPEC)
        env = full_deploy_env(tmp_path)
        env[service.main.ENV_CHART_STORE_DIR] = str(store_dir)
        apply_deploy_env(monkeypatch, env)
        # No network at the unit tier: the index reader seam reports the
        # legitimate no-index read-only start instead of touching qdrant.
        monkeypatch.setattr(service.main, "_make_index_version_reader", lambda config: lambda: None)

        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.create_service_app()
        message = str(excinfo.value)
        assert service.main.ENV_DATASET_MANIFEST in message
        assert service.main.ENV_CHART_PACK_DIR in message

    def test_boot_with_an_empty_spec_store_needs_no_render_inputs(
        self, monkeypatch, tmp_path
    ) -> None:
        """Companion guard: the dev compose stub (nothing stored, no
        datasets landed — ADR-023 keeps them out of git) must keep
        booting; there is nothing to serve, so nothing is required."""
        import service.main
        from tests._service_fixtures import apply_deploy_env, full_deploy_env

        env = full_deploy_env(tmp_path)
        env[service.main.ENV_CHART_STORE_DIR] = str(tmp_path / "empty-chart-specs")
        apply_deploy_env(monkeypatch, env)
        monkeypatch.setattr(service.main, "_make_index_version_reader", lambda config: lambda: None)

        app = service.main.create_service_app()
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get(f"/chart/{'f' * 64}").status_code == 404

    def test_paused_permalink_serving_with_the_render_inputs_present(
        self, monkeypatch, tmp_path
    ) -> None:
        """The delivered guarantee, end to end at this tier: a PAUSED
        app over the real composition-root wiring with the render
        inputs configured serves a stored spec's permalink (the fake
        renderer stands in for the #17 artefact path, which has its own
        integration suite)."""
        harness = paused_harness(tmp_path)
        spec_hash = harness.spec_store.put({"spec_version": "1.0", "title": "flagship (invented)"})
        response = TestClient(harness.app).get(f"/chart/{spec_hash}")
        assert response.status_code == 200
        assert response.json()["alt_text"]

    def test_runbook_lists_render_inputs_as_required_for_the_paused_stack(self) -> None:
        """DEPLOYMENT.md must stop classing the manifest + pack as
        needless for the paused stack: the block that says 'not for the
        paused stack' must not contain them."""
        runbook = (self.REPO_ROOT / "service" / "DEPLOYMENT.md").read_text(encoding="utf-8")
        assert "Live-generation-only" in runbook, (
            "DEPLOYMENT.md lost its optional-variables block entirely — "
            "update this test to the runbook's new structure"
        )
        optional_block_start = runbook.index("Live-generation-only")
        optional_block = runbook[optional_block_start:].split("\n\n")[0]
        for variable in ("CLIMATE_CHAT_DATASET_MANIFEST", "CLIMATE_CHAT_CHART_PACK_DIR"):
            assert variable not in optional_block, (
                f"DEPLOYMENT.md still classes {variable} as live-generation-"
                "only, but serving stored chart permalinks — a paused-stack "
                "guarantee (ADR-015) — requires it (issue #214)"
            )


class TestStarterCacheStructure:
    """The release-time cache artifact's pinned structure (content
    generation itself is a release step, deliberately untested here)."""

    def test_valid_cache_loads_with_entries_for_every_starter_question(self, tmp_path) -> None:
        write_starter_cache(tmp_path)
        cache = load_starter_cache(tmp_path)
        assert cache.generated_on == STARTER_GENERATED_ON
        for question in STARTER_QUESTIONS:
            entry = cache.lookup(question)
            assert entry is not None, f"no cache entry for starter question: {question!r}"
            assert entry.answer_text
            assert entry.citations
            assert entry.footer
            assert entry.generated_on == STARTER_GENERATED_ON

    def test_lookup_is_exact_never_fuzzy(self, tmp_path) -> None:
        write_starter_cache(tmp_path)
        cache = load_starter_cache(tmp_path)
        assert cache.lookup("Why is the sky blue?") is None
        # Whitespace-normalised exact match still resolves.
        assert cache.lookup(f"  {STARTER_QUESTIONS[0]}  ") is not None

    def test_missing_cache_file_refuses_loudly(self, tmp_path) -> None:
        with pytest.raises(StarterCacheError):
            load_starter_cache(tmp_path)

    def test_incomplete_cache_refuses_naming_the_missing_questions(self, tmp_path) -> None:
        payload = starter_cache_payload()
        dropped = payload["entries"].pop(0)
        write_starter_cache(tmp_path, payload)
        with pytest.raises(StarterCacheError) as excinfo:
            load_starter_cache(tmp_path)
        assert dropped["question"] in str(excinfo.value)

    def test_undated_cache_refuses(self, tmp_path) -> None:
        payload = starter_cache_payload()
        del payload["generated_on"]
        write_starter_cache(tmp_path, payload)
        with pytest.raises(StarterCacheError):
            load_starter_cache(tmp_path)

    def test_entry_missing_answer_or_footer_refuses(self, tmp_path) -> None:
        payload = starter_cache_payload()
        payload["entries"][3]["answer_text"] = ""
        write_starter_cache(tmp_path, payload)
        with pytest.raises(StarterCacheError):
            load_starter_cache(tmp_path)

    def test_starter_questions_match_design_7_1(self) -> None:
        """The §7.1 landing-page list, verbatim — the cache completeness
        check and #18's landing page share this single source of truth."""
        assert STARTER_QUESTIONS == (
            "Why are scientists calling this an emergency?",
            "How much has the planet warmed — and how fast is it accelerating?",
            "What happens at 1.5°C, 2°C, 3°C?",
            "Show me CO₂ and temperature over the last 10,000 years",
            "Hasn't the climate always changed?",
            "How sure are scientists it's human-caused?",
            "Didn't warming pause?",
            "What are tipping points and how close are we?",
            "How will this affect food, water and health?",
            "Are we on track to meet the Paris targets?",
            "Is it too late?",
            "What would an emergency response actually look like?",
            "Who is speaking up, and how do I get involved?",
        )
