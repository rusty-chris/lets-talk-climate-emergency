"""Footprint feature RED — the footer indicator's pure UI model.

The indicator is fed CLIENT-SIDE from the SSE ``usage`` events that
reach the browser per exchange (the #12 wire vocabulary — no new events,
no extra requests, $0 by construction):

- the fold builds ``AnswerView.footprint`` on COMPLETED exchanges:
  an ``estimated`` Wh range through ``service.footprint`` (the single
  source of truth — the UI can never drift from the published method);
- cached replays (semantic cache / cached starter) are the honest
  ``cached_zero`` state: ~zero new inference, never a fabricated range
  (the FLAGGED cached-exchange decision);
- kinds whose wire carries no usage (canned/refusal/paused) and
  errored/incomplete streams show NOTHING rather than an invented zero
  — their server-side metered spend lives in the /footprint totals;
- session accumulation is IDEMPOTENT by ``exchange_id`` (the #226
  machinery: a Streamlit rerun replays the cached exchange's events
  through the same fold — it must never double-count);
- the rendered line is the §9 template VERBATIM via
  ``service.footprint.format_footprint_footer`` — always a range,
  "est."/"estimates" present, the /footprint link, NO gCO2e;
- the shell renders through the pure helpers (structural pin, the
  finding-#233 no-wire-literals-in-the-shell discipline).
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from service.footprint import (
    WhRange,
    api_energy_wh,
    format_footprint_footer,
    format_footprint_footer_cached,
)
from tests._ui_fixtures import (
    answer_event,
    citation_event,
    error_event,
    footer_event,
    meta_event,
    text_event,
    usage_event,
)
from ui.render_model import (
    FOOTPRINT_STATUS_CACHED_ZERO,
    FOOTPRINT_STATUS_ESTIMATED,
    SESSION_FOOTPRINT_EMPTY,
    ErrorNotice,
    ExchangeFootprint,
    SessionFootprint,
    accumulate_session_footprint,
    exchange_footprint,
    fold_chat_stream,
    footprint_indicator_line,
    transport_failure_view,
)

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def meta_with_exchange_id(exchange_id: str | None = "ex-1") -> dict[str, Any]:
    event = meta_event()
    event["data"]["exchange_id"] = exchange_id
    return event


def grounded_stream(*, exchange_id: str | None = "ex-1") -> list[dict[str, Any]]:
    return [
        meta_with_exchange_id(exchange_id),
        text_event("The basin has very likely warmed."),
        citation_event(0, "chunk-1", "warmed by 1.9 degrees", "IPCC AR6"),
        usage_event(input_tokens=900, output_tokens=42),
        footer_event(),
    ]


class TestFoldBuildsTheFootprint:
    def test_completed_grounded_exchange_is_estimated_from_its_usage(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert view.footprint is not None
        assert view.footprint.status == FOOTPRINT_STATUS_ESTIMATED
        assert view.footprint.answer_wh == api_energy_wh(
            [{"input_tokens": 900, "output_tokens": 42}]
        )

    def test_multiple_usage_events_sum(self) -> None:
        events = grounded_stream()
        events.insert(3, usage_event(input_tokens=100, output_tokens=8))
        view = fold_chat_stream(events)
        assert view.footprint is not None
        assert view.footprint.answer_wh == api_energy_wh(
            [
                {"input_tokens": 100, "output_tokens": 8},
                {"input_tokens": 900, "output_tokens": 42},
            ]
        )

    def test_cache_token_counts_ride_into_the_estimate(self) -> None:
        # The §9 prompt-caching design makes cache reads a large share of
        # input tokens — the wire's cache keys must not be dropped.
        events = [
            meta_with_exchange_id(),
            text_event("answer"),
            {
                "event": "usage",
                "data": {
                    "input_tokens": 900,
                    "output_tokens": 42,
                    "cache_read_input_tokens": 4200,
                    "cache_creation_input_tokens": 0,
                },
            },
            footer_event(),
        ]
        view = fold_chat_stream(events)
        assert view.footprint is not None
        assert view.footprint.answer_wh == api_energy_wh(
            [
                {
                    "input_tokens": 900,
                    "output_tokens": 42,
                    "cache_read_input_tokens": 4200,
                    "cache_creation_input_tokens": 0,
                }
            ]
        )

    def test_error_terminated_stream_has_no_footprint(self) -> None:
        events = [
            meta_with_exchange_id(),
            text_event("partial"),
            usage_event(),
            error_event(),
        ]
        view = fold_chat_stream(events)
        assert view.footprint is None

    def test_incomplete_stream_has_no_footprint(self) -> None:
        # No footer = incomplete: an undelivered answer never wears a
        # cost estimate.
        events = [meta_with_exchange_id(), text_event("partial"), usage_event()]
        view = fold_chat_stream(events)
        assert view.footprint is None

    @pytest.mark.parametrize("kind", ["cached", "cached_starter"])
    def test_cached_replays_are_the_honest_cached_zero_state(self, kind: str) -> None:
        # The FLAGGED decision: a replay performs ~zero new inference —
        # say so; never fabricate an estimate range for it.
        events = [
            meta_with_exchange_id(),
            answer_event(
                kind,
                "cached answer text",
                generated_on="2026-09-01",
                footer="footer",
                citations=[],
            ),
        ]
        view = fold_chat_stream(events)
        assert view.footprint is not None
        assert view.footprint.status == FOOTPRINT_STATUS_CACHED_ZERO
        assert view.footprint.answer_wh is None

    @pytest.mark.parametrize("kind", ["canned", "refusal", "paused"])
    def test_kinds_without_wire_usage_show_nothing(self, kind: str) -> None:
        # No usage reached the client: nothing is shown rather than an
        # invented zero — the server-side metered spend for these lands
        # in the /footprint application totals instead.
        events = [meta_with_exchange_id(), answer_event(kind, "service text")]
        view = fold_chat_stream(events)
        assert view.footprint is None

    def test_standalone_helper_matches_the_fold(self) -> None:
        # One rule, one source of truth: exchange_footprint over the raw
        # events IS the fold's decision.
        events = grounded_stream()
        assert exchange_footprint(events) == fold_chat_stream(events).footprint


class TestSessionAccumulation:
    """#226 idempotency: reruns replay events — never double-count."""

    def estimated(self) -> ExchangeFootprint:
        return ExchangeFootprint(
            status=FOOTPRINT_STATUS_ESTIMATED,
            answer_wh=WhRange(low=0.1, central=0.5, high=1.5),
        )

    def test_a_new_exchange_adds_its_range(self) -> None:
        session = accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, "ex-1", self.estimated())
        assert session.total_wh == WhRange(low=0.1, central=0.5, high=1.5)
        assert "ex-1" in session.counted_exchange_ids

    def test_replaying_the_same_exchange_id_is_a_no_op(self) -> None:
        once = accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, "ex-1", self.estimated())
        twice = accumulate_session_footprint(once, "ex-1", self.estimated())
        assert twice == once, "a Streamlit rerun must never double-count an exchange"

    def test_distinct_exchanges_sum_elementwise(self) -> None:
        first = accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, "ex-1", self.estimated())
        second = accumulate_session_footprint(first, "ex-2", self.estimated())
        assert second.total_wh.low == pytest.approx(0.2)
        assert second.total_wh.central == pytest.approx(1.0)
        assert second.total_wh.high == pytest.approx(3.0)

    def test_cached_zero_adds_nothing_but_marks_the_exchange(self) -> None:
        cached = ExchangeFootprint(status=FOOTPRINT_STATUS_CACHED_ZERO, answer_wh=None)
        session = accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, "ex-9", cached)
        assert session.total_wh == SESSION_FOOTPRINT_EMPTY.total_wh
        assert "ex-9" in session.counted_exchange_ids

    def test_no_exchange_id_or_no_footprint_changes_nothing(self) -> None:
        assert (
            accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, None, self.estimated())
            == SESSION_FOOTPRINT_EMPTY
        )
        assert (
            accumulate_session_footprint(SESSION_FOOTPRINT_EMPTY, "ex-1", None)
            == SESSION_FOOTPRINT_EMPTY
        )

    def test_the_replay_scenario_end_to_end(self) -> None:
        """Fold the SAME cached events twice (a rerun) and accumulate:
        the session grows exactly once."""
        events = grounded_stream(exchange_id="ex-42")
        session = SESSION_FOOTPRINT_EMPTY
        for _rerun in range(3):
            view = fold_chat_stream(events)
            session = accumulate_session_footprint(session, view.exchange_id, view.footprint)
        assert session.total_wh == api_energy_wh([{"input_tokens": 900, "output_tokens": 42}])
        assert session.counted_exchange_ids == frozenset({"ex-42"})


class TestIndicatorLine:
    def session(self) -> SessionFootprint:
        return SessionFootprint(
            total_wh=WhRange(low=0.4, central=2.0, high=6.0),
            counted_exchange_ids=frozenset({"ex-1"}),
        )

    def test_estimated_view_renders_the_verbatim_template(self) -> None:
        view = fold_chat_stream(grounded_stream())
        line = footprint_indicator_line(view, self.session())
        assert view.footprint is not None and view.footprint.answer_wh is not None
        assert line == format_footprint_footer(view.footprint.answer_wh, self.session().total_wh)

    def test_cached_view_renders_the_cached_template(self) -> None:
        events = [
            meta_with_exchange_id(),
            answer_event("cached", "text", generated_on="2026-09-01", footer="f", citations=[]),
        ]
        view = fold_chat_stream(events)
        line = footprint_indicator_line(view, self.session())
        assert line == format_footprint_footer_cached(self.session().total_wh)

    def test_no_footprint_means_no_indicator(self) -> None:
        events = [meta_with_exchange_id(), answer_event("refusal", "no")]
        view = fold_chat_stream(events)
        assert footprint_indicator_line(view, self.session()) is None

    def test_errored_view_has_no_indicator(self) -> None:
        events = [meta_with_exchange_id(), text_event("x"), usage_event(), error_event()]
        view = fold_chat_stream(events)
        assert footprint_indicator_line(view, self.session()) is None


class TestTransportFailureWearsNoIndicator:
    """Review finding #366 — an undelivered answer never wears a cost
    estimate, on EVERY path that produces an errored/incomplete view.

    ``footprint_indicator_line``'s docstring pins "an incomplete/errored
    view → None", but the implementation checks only ``view.footprint``
    — it relies on the fold having yielded None. That holds for
    ``error``-event streams, not for a TRANSPORT failure after full
    delivery: ``transport_failure_view`` folds the teed events (usage +
    footer already present) and marks ``complete=False`` with a
    transport error, but does not clear ``footprint`` — so the shell
    renders "This answer is incomplete." AND a cost indicator under the
    same answer. DECISION pinned (flagged in the red-phase report): the
    failed exchange carries NO client-side footprint at all — no
    indicator, no session accumulation; its server-side metered spend
    lives in the /footprint application totals like every other
    undelivered path.
    """

    def transport_failed_after_delivery(self):
        # The connection dropped on the final read/close: usage AND
        # footer were already teed before the transport raised.
        events = [
            meta_with_exchange_id("ex-t"),
            text_event("The basin has very likely warmed."),
            usage_event(input_tokens=7000, output_tokens=700),
            footer_event(),
        ]
        return transport_failure_view(events, "connection lost")

    def test_transport_failure_view_carries_no_footprint(self) -> None:
        view = self.transport_failed_after_delivery()
        assert view.complete is False
        assert view.error is not None and view.error.error_type == "transport"
        assert view.footprint is None, (
            "a transport-failure view must not wear an estimated footprint"
        )

    def test_no_indicator_renders_for_the_transport_failure(self) -> None:
        view = self.transport_failed_after_delivery()
        session = SessionFootprint(
            total_wh=WhRange(low=0.4, central=2.0, high=6.0),
            counted_exchange_ids=frozenset({"ex-1"}),
        )
        assert footprint_indicator_line(view, session) is None, (
            "'This answer is incomplete.' and a cost indicator must never "
            "render under the same answer"
        )

    def test_session_accumulation_ignores_the_failed_exchange(self) -> None:
        view = self.transport_failed_after_delivery()
        session = accumulate_session_footprint(
            SESSION_FOOTPRINT_EMPTY, view.exchange_id, view.footprint
        )
        assert session == SESSION_FOOTPRINT_EMPTY, (
            "an errored delivery is not counted client-side (its metered "
            "spend lives in the /footprint application totals)"
        )

    def test_indicator_enforces_the_rule_where_it_is_stated(self) -> None:
        # Belt and braces: even if some OTHER path hands the indicator a
        # view that is errored/incomplete yet still carries a footprint,
        # the indicator itself must refuse — the docstring's rule is
        # enforced at the docstring's function, not two functions away.
        complete_view = fold_chat_stream(grounded_stream())
        assert complete_view.footprint is not None
        errored = replace(
            complete_view,
            complete=False,
            error=ErrorNotice(error_type="transport", message="connection lost"),
        )
        assert errored.footprint is not None  # the hostile precondition
        session = SessionFootprint(
            total_wh=WhRange(low=0.4, central=2.0, high=6.0),
            counted_exchange_ids=frozenset({"ex-1"}),
        )
        assert footprint_indicator_line(errored, session) is None


class TestShellWiring:
    """Structural (the shell-hygiene pattern): the Streamlit shell
    renders the indicator and accumulates the session through the pure
    helpers — no wire literals or arithmetic of its own."""

    def shell_names(self) -> set[str]:
        tree = ast.parse((UI_DIR / "app.py").read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        return names

    def test_shell_renders_the_indicator_through_the_pure_helper(self) -> None:
        assert "footprint_indicator_line" in self.shell_names(), (
            "ui/app.py must render the footer indicator via the pure "
            "footprint_indicator_line helper"
        )

    def test_shell_accumulates_the_session_through_the_pure_helper(self) -> None:
        assert "accumulate_session_footprint" in self.shell_names(), (
            "ui/app.py must accumulate the session footprint via the pure "
            "idempotent helper (the #226 rerun machinery)"
        )

    def test_no_factor_arithmetic_leaks_into_the_shell(self) -> None:
        source = (UI_DIR / "app.py").read_text(encoding="utf-8")
        for literal in ("E_OUT", "E_IN", "0.5", "CACHE_READ_FACTOR"):
            assert literal not in source, (
                f"ui/app.py carries footprint arithmetic ({literal!r}) — "
                "estimation lives in service.footprint only"
            )
