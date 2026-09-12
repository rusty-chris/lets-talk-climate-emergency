"""Footprint feature RED — the /footprint route, wiring and the $0 rule.

App-level pins (TestClient over the composed app, all adapters faked):

- ``/footprint`` joined the transparency route vocabulary: parity with
  ``ui.footer`` (both constants move together), and the app serves the
  route in BOTH modes, ``text/html``, never rate-limited — the same
  serving contract as the four static pages, EXCEPT it renders per
  request through the injected ``ServiceDeps.footprint_page`` seam (its
  headline totals are live);
- with no seam the interim placeholder serves (the pre-#19 placeholder
  pattern), and even the placeholder carries the ADR-018 pair, the
  §4.11 disclaimer, and the explicit interim marker (#249 honesty);
- **$0 and anonymous**: page views make zero adapter calls, log
  nothing, and never touch the ledger;
- **the aggregate wiring**: one chat exchange records its usage token
  counts + one exchange into ``ServiceDeps.footprint_ledger`` (the same
  records the spend cap charges); a raising ledger NEVER breaks the
  exchange (the answer outranks the counter);
- **composition root**: ``service.main.build_service_deps`` wires a
  real ``FootprintLedger`` journalling beside the spend state, and a
  ``footprint_page`` callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import ui.footer
from charts.planner import ChartRefusal, CurationGap
from rag.provider import StructuredResult
from service.budget import ServiceMode
from service.footprint import (
    FOOTPRINT_ROUTE,
    FOOTPRINT_STATE_FILENAME,
    FootprintLedger,
    FootprintLedgerError,
)
from service.transparency import TRANSPARENCY_ROUTES
from tests._generation_fixtures import transport_stream_events
from tests._service_fixtures import (
    FakePlanner,
    FakeValidationOutcome,
    FakeValidationSeam,
    FrozenClock,
    classifier_output,
    make_config,
    make_harness,
    post_chat,
    stream_usage,
)


@dataclass
class RecordingLedger:
    """Duck-typed FootprintLedger double: records every call."""

    record_calls: list[tuple[list[dict[str, Any]], float]] = field(default_factory=list)
    raise_on_record: bool = False

    def record_exchange(self, usage_records, *, cpu_seconds: float = 0.0) -> None:
        if self.raise_on_record:
            raise FootprintLedgerError("synthetic journal failure")
        self.record_calls.append(([dict(r) for r in usage_records], cpu_seconds))

    def totals(self) -> Any:  # pragma: no cover - not consulted by routes
        raise AssertionError("routes must not read totals through the ledger double")


class CountingPage:
    """A footprint_page seam double proving per-request rendering."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return f"<html><body>FOOTPRINT-RENDER-{self.calls} (synthetic)</body></html>"


def retrieval_harness(tmp_path, **kwargs):
    harness = make_harness(tmp_path, **kwargs)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", transport_stream_events())
    return harness


class TestRouteVocabularyParity:
    """Adding /footprint is a route-vocabulary change: the pins move
    coherently on BOTH sides or not at all."""

    def test_footprint_is_a_transparency_route_on_both_sides(self) -> None:
        assert FOOTPRINT_ROUTE in TRANSPARENCY_ROUTES
        assert FOOTPRINT_ROUTE in ui.footer.TRANSPARENCY_ROUTES
        # The existing equality pin covers full-tuple parity; this pins
        # the membership so a partial move cannot pass either suite.
        assert TRANSPARENCY_ROUTES == ui.footer.TRANSPARENCY_ROUTES


class TestServingContract:
    def test_serves_the_injected_page_live(self, tmp_path) -> None:
        page = CountingPage()
        harness = make_harness(tmp_path, footprint_page=page)
        response = TestClient(harness.app).get(FOOTPRINT_ROUTE)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "FOOTPRINT-RENDER-1" in response.text

    def test_renders_per_request_not_once_at_startup(self, tmp_path) -> None:
        """The headline totals are live: every GET re-renders through
        the seam (unlike the four startup-built static pages)."""
        page = CountingPage()
        harness = make_harness(tmp_path, footprint_page=page)
        client = TestClient(harness.app)
        first = client.get(FOOTPRINT_ROUTE).text
        second = client.get(FOOTPRINT_ROUTE).text
        assert page.calls == 2
        assert first != second

    def test_serves_paused_read_only_not_dark(self, tmp_path) -> None:
        page = CountingPage()
        harness = make_harness(tmp_path, footprint_page=page)
        harness.tracker.record_usage(
            "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
        )
        assert harness.tracker.mode() is ServiceMode.PAUSED
        response = TestClient(harness.app).get(FOOTPRINT_ROUTE)
        assert response.status_code == 200
        assert "FOOTPRINT-RENDER-1" in response.text

    def test_placeholder_serves_without_the_seam_and_is_honest(self, tmp_path) -> None:
        # The pre-#19 interim-placeholder pattern, with the #249 honesty
        # bar: the ADR-018 pair, the §4.11 disclaimer, an explicit
        # interim marker.
        from service.transparency import (
            NON_AFFILIATION_DISCLAIMER,
            NONCOMMERCIAL_NOTE,
            STEWARD_CREDIT_TEXT,
        )

        harness = make_harness(tmp_path)
        text = TestClient(harness.app).get(FOOTPRINT_ROUTE).text
        assert STEWARD_CREDIT_TEXT in text
        assert NONCOMMERCIAL_NOTE in text
        assert NON_AFFILIATION_DISCLAIMER in text
        assert "pre-release placeholder page" in text

    def test_page_views_are_free_and_anonymous(self, tmp_path) -> None:
        ledger = RecordingLedger()
        harness = make_harness(tmp_path, footprint_page=CountingPage(), footprint_ledger=ledger)
        client = TestClient(harness.app)
        for _ in range(3):
            assert client.get(FOOTPRINT_ROUTE).status_code == 200
        # $0: no adapter calls; anonymous: nothing logged; and a page
        # view is NOT an exchange — the ledger records nothing.
        assert harness.adapter.calls == []
        assert harness.exchange_log.records() == []
        assert ledger.record_calls == []

    def test_never_rate_limited(self, tmp_path) -> None:
        harness = make_harness(
            tmp_path,
            footprint_page=CountingPage(),
            config=make_config(rate_limit_per_minute=1),
        )
        client = TestClient(harness.app)
        for _ in range(3):
            assert client.get(FOOTPRINT_ROUTE).status_code == 200


class TestLedgerWiring:
    """The application total counts what the spend cap charges."""

    def test_one_exchange_records_once_with_its_usage_tokens(self, tmp_path) -> None:
        ledger = RecordingLedger()
        harness = retrieval_harness(tmp_path, footprint_ledger=ledger)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert any(event["event"] == "footer" for event in events)
        assert len(ledger.record_calls) == 1, "exactly one record_exchange per logged exchange"
        usage_records, _cpu = ledger.record_calls[0]
        # The generation stream's charged usage reaches the ledger with
        # its token counts intact (the same records the spend cap saw).
        expected = stream_usage()
        recorded_outputs = sum(
            (record.get("usage") or {}).get("output_tokens", 0) or 0 for record in usage_records
        )
        assert recorded_outputs == expected["output_tokens"]
        recorded_cache_reads = sum(
            (record.get("usage") or {}).get("cache_read_input_tokens", 0) or 0
            for record in usage_records
        )
        assert recorded_cache_reads == expected["cache_read_input_tokens"]

    def test_declined_and_canned_exchanges_still_count(self, tmp_path) -> None:
        # §7: the totals include refused/declined exchanges — every
        # logged exchange records (possibly with zero usage), because
        # the classifier call was real spend.
        ledger = RecordingLedger()
        harness = make_harness(tmp_path, footprint_ledger=ledger)
        harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
        events = post_chat(TestClient(harness.app), "Who won the 1966 World Cup?")
        assert any(event["event"] == "answer" for event in events)
        assert len(ledger.record_calls) == 1

    def test_a_raising_ledger_never_breaks_the_exchange(self, tmp_path) -> None:
        # The answer outranks the counter: a journal failure must not
        # surface as a chat error.
        ledger = RecordingLedger(raise_on_record=True)
        harness = retrieval_harness(tmp_path, footprint_ledger=ledger)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]
        assert "error" not in names
        assert "footer" in names

    def test_no_ledger_means_no_recording_and_no_crash(self, tmp_path) -> None:
        harness = retrieval_harness(tmp_path)  # footprint_ledger=None
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert any(event["event"] == "footer" for event in events)


#: Nonzero server-side usage for the calls the FOOTER never sees but the
#: spend cap charges — the methodology (§1/§2/§7, BINDING) says the
#: application totals include every one of them.
CLASSIFIER_USAGE = {"input_tokens": 311, "output_tokens": 17}
VALIDATION_USAGE = {"input_tokens": 505, "output_tokens": 11}
PLANNER_REFUSAL_USAGE = {"input_tokens": 222, "output_tokens": 9}


def real_ledger(tmp_path) -> FootprintLedger:
    """A REAL FootprintLedger journalling under tmp_path (not the double):
    these pins read ``totals()`` back, end to end."""
    return FootprintLedger(state_dir=tmp_path / "footprint-ledger", clock=FrozenClock())


class TestFullScopeLedgerRecording:
    """Review finding #360 (HIGH) — the binding methodology's scope rule.

    docs/FOOTPRINT-METHODOLOGY.md §2 defines the per-exchange figure as
    the sum over ALL of the exchange's ``usage_records`` — classifier +
    generation + validation (or classifier + planner for chart queries) —
    and §7 promises the totals include "every runtime adapter call's
    tokens (the same records the spend cap charges — including
    refused/declined exchanges and validation calls)". The rule these
    pins enforce: WHATEVER SPEND CHARGES, THE LEDGER RECORDS. Today the
    classifier and validation calls are charged to spend but never reach
    ``usage_records``, so the flagship transparency number undercounts
    every live exchange by ~20–30% — in the flattering direction.
    """

    def test_grounded_exchange_records_classifier_generation_and_validation(self, tmp_path) -> None:
        ledger = real_ledger(tmp_path)
        validation = FakeValidationSeam(
            outcome=FakeValidationOutcome(usage=dict(VALIDATION_USAGE), model="claude-haiku-4-5")
        )
        harness = make_harness(tmp_path, footprint_ledger=ledger, validation=validation)
        harness.adapter.queue(
            "structured",
            StructuredResult(value=classifier_output(), usage=dict(CLASSIFIER_USAGE)),
        )
        harness.adapter.queue("generate_stream", transport_stream_events())
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert any(event["event"] == "footer" for event in events)

        generation = stream_usage()
        totals = ledger.totals()
        assert totals.exchanges == 1
        # §2: classifier + generation + validation, token for token — the
        # same records the spend cap charged for this exchange.
        assert totals.input_tokens == (
            CLASSIFIER_USAGE["input_tokens"]
            + generation["input_tokens"]
            + VALIDATION_USAGE["input_tokens"]
        ), "the ledger must record the classifier's and validator's charged input tokens"
        assert totals.output_tokens == (
            CLASSIFIER_USAGE["output_tokens"]
            + generation["output_tokens"]
            + VALIDATION_USAGE["output_tokens"]
        ), "the ledger must record the classifier's and validator's charged output tokens"
        assert totals.cache_read_input_tokens == generation["cache_read_input_tokens"]
        assert totals.cache_creation_input_tokens == generation["cache_creation_input_tokens"]

    def test_canned_exchange_records_the_classifiers_charged_usage(self, tmp_path) -> None:
        # §7: refused/declined exchanges are included — the classifier
        # call was real, charged spend, and must reach the totals (today
        # the canned route logs usage_records=[] and the tokens vanish).
        ledger = real_ledger(tmp_path)
        harness = make_harness(tmp_path, footprint_ledger=ledger)
        harness.adapter.queue(
            "structured",
            StructuredResult(
                value=classifier_output(scope="out_of_scope"),
                usage=dict(CLASSIFIER_USAGE),
            ),
        )
        events = post_chat(TestClient(harness.app), "Who won the 1966 World Cup?")
        assert any(event["event"] == "answer" for event in events)
        totals = ledger.totals()
        assert totals.exchanges == 1
        assert totals.input_tokens == CLASSIFIER_USAGE["input_tokens"]
        assert totals.output_tokens == CLASSIFIER_USAGE["output_tokens"]

    def test_chart_refusal_records_the_planners_charged_usage(self, tmp_path) -> None:
        # The chart-refusal branch charges the planner's usage to spend
        # but logs usage_records=[] — the charged tokens vanish from the
        # exchange record and the ledger alike (issue #360 reproduction).
        refusal = ChartRefusal(
            message="I can't chart that; nearest available: syn_annual_anomaly.",
            gap=CurationGap(
                chart_request="plot the invented seagrass index",
                requested_data="seagrass index",
                nearest_datasets=("syn_annual_anomaly",),
            ),
            usage=dict(PLANNER_REFUSAL_USAGE),
        )
        ledger = real_ledger(tmp_path)
        harness = make_harness(
            tmp_path, footprint_ledger=ledger, planner=FakePlanner(result=refusal)
        )
        harness.adapter.queue(
            "structured",
            StructuredResult(
                value=classifier_output(scope="chart_request"),
                usage=dict(CLASSIFIER_USAGE),
            ),
        )
        events = post_chat(TestClient(harness.app), "Plot the invented seagrass index")
        assert any(event["event"] == "answer" for event in events)
        totals = ledger.totals()
        assert totals.exchanges == 1
        assert totals.input_tokens == (
            CLASSIFIER_USAGE["input_tokens"] + PLANNER_REFUSAL_USAGE["input_tokens"]
        ), "the chart-refusal branch must record the planner's charged usage"
        assert totals.output_tokens == (
            CLASSIFIER_USAGE["output_tokens"] + PLANNER_REFUSAL_USAGE["output_tokens"]
        )


class TestWrongTypedJournalNeverGoesDark:
    """Review finding #365 — the one transparency page that must always
    serve, in both modes, must not 500 on a half-corrupt counter file.

    A journal that is valid JSON with wrong-typed values (a realistic
    crash-loop artifact for a file rewritten on every exchange) currently
    escapes ``FootprintLedger.totals()`` as a raw ``ValueError``/
    ``TypeError``; ``service.main._footprint_page`` catches only
    ``FootprintLedgerError``, so GET /footprint returns 500 instead of
    the honest unavailable notice.
    """

    def test_wrong_typed_journal_serves_the_unavailable_notice_not_500(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        import service.main as main
        from service.app import create_app
        from service.footprint import FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE
        from tests._service_fixtures import (
            apply_deploy_env,
            full_deploy_env,
            write_starter_cache,
        )
        from tests._transparency_fixtures import contains_verbatim

        apply_deploy_env(monkeypatch, full_deploy_env(tmp_path))
        cache_dir = tmp_path / "starter-cache"
        write_starter_cache(cache_dir)
        config = make_config(
            starter_cache_dir=str(cache_dir),
            log_dir=str(tmp_path / "logs"),
        )
        deps = main.build_service_deps(config)

        # A half-corrupt counter file: valid JSON, wrong-typed count.
        journal = deps.footprint_ledger.state_path
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(
            json.dumps(
                {
                    "since": "2026-09-01",
                    "exchanges": "many",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cpu_seconds": 0.0,
                }
            ),
            encoding="utf-8",
        )

        client = TestClient(create_app(config, deps), raise_server_exceptions=False)
        response = client.get(FOOTPRINT_ROUTE)
        assert response.status_code == 200, (
            "GET /footprint must serve the unavailable notice on a half-corrupt "
            f"journal, not go dark — got {response.status_code}"
        )
        assert contains_verbatim(response.text, FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE)


class TestCompositionRoot:
    def test_main_wires_ledger_and_page_beside_the_spend_state(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``build_service_deps`` composes the real seams: a
        FootprintLedger journalling BESIDE the spend journal (same
        state_dir under the volume-backed log dir — a redeploy can never
        reset the public count), and a per-request page callable."""
        import service.main as main
        from tests._service_fixtures import (
            apply_deploy_env,
            full_deploy_env,
            write_starter_cache,
        )

        apply_deploy_env(monkeypatch, full_deploy_env(tmp_path))
        cache_dir = tmp_path / "starter-cache"
        write_starter_cache(cache_dir)
        config = make_config(
            starter_cache_dir=str(cache_dir),
            log_dir=str(tmp_path / "logs"),
        )
        deps = main.build_service_deps(config)
        assert isinstance(deps.footprint_ledger, FootprintLedger)
        assert deps.footprint_ledger.state_path == (
            Path(config.log_dir) / "spend-state" / FOOTPRINT_STATE_FILENAME
        )
        assert callable(deps.footprint_page)
