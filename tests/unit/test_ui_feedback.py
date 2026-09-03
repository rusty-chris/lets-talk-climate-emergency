"""Thumbs feedback on the chat surface (issue #56) — the UI side.

RED-phase suite — pins:

- The fold carries the meta event's ``exchange_id`` into the view
  verbatim (``None`` when the wire carried none — an older service, or
  the paused furniture that logged no exchange).
- ``feedback_widget_model``: the thumbs widget rides EVERY completed
  answer view that has a join key — grounded, chart, canned, refusal
  and cached-starter kinds alike — and never an errored, incomplete or
  key-less view.
- ``resolve_feedback_state``: the optimistic-display honesty rule — a
  failed POST shows the UNRECORDED state (no verdict selected, the
  pinned honest message), never a fake success.
- Wire-vocabulary parity: the UI's verdict constants ARE the service's
  (imported, not copied — drift is impossible by construction, and the
  pin proves the import stays).
- Structural guards (the shell-hygiene pattern): ui/app.py renders the
  widget through the pure model and the transport seam, with no
  verdict string literals of its own; the presenters facade exposes
  the whole feedback contract.
"""

from __future__ import annotations

import ast
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests._ui_fixtures import (
    DISCLOSURE,
    answer_event,
    chart_event,
    citation_event,
    error_event,
    footer_event,
    text_event,
)
from ui.render_model import (
    FEEDBACK_DOWN,
    FEEDBACK_NOT_RECORDED_MESSAGE,
    FEEDBACK_RECORDED_MESSAGE,
    FEEDBACK_STATE_RECORDED,
    FEEDBACK_STATE_UNRECORDED,
    FEEDBACK_UP,
    FeedbackWidget,
    feedback_widget_model,
    fold_chat_stream,
    resolve_feedback_state,
)

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

EXCHANGE_ID = uuid.uuid4().hex


def meta_event(
    exchange_id: Any = EXCHANGE_ID, mode: str = "live", *, omit_key: bool = False
) -> dict[str, Any]:
    """A meta event; ``omit_key=True`` fabricates an OLDER service's wire
    (pre-#56: no exchange_id field at all)."""
    data: dict[str, Any] = {"disclosure": DISCLOSURE, "preamble_note": None, "mode": mode}
    if not omit_key:
        data["exchange_id"] = exchange_id
    return {"event": "meta", "data": data}


def grounded_stream(**meta_kwargs: Any) -> list[dict[str, Any]]:
    return [
        meta_event(**meta_kwargs),
        text_event("The basin has very likely warmed. "),
        citation_event(0, "syn-doc::c0000", "an invented cited sentence"),
        footer_event(),
    ]


class TestFoldCarriesTheExchangeId:
    def test_meta_exchange_id_rides_into_the_view_verbatim(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert view.exchange_id == EXCHANGE_ID

    def test_wire_without_the_key_folds_to_none(self) -> None:
        """An older service's meta (no exchange_id field) must fold, not
        crash — and yield no join key."""
        view = fold_chat_stream(grounded_stream(omit_key=True))
        assert view.exchange_id is None

    def test_wire_with_a_null_key_folds_to_none(self) -> None:
        """The paused furniture path sends exchange_id: null (nothing was
        logged; there is no record to rate against)."""
        events = [
            meta_event(exchange_id=None, mode="paused"),
            answer_event("paused", "Paused for today."),
        ]
        view = fold_chat_stream(events)
        assert view.exchange_id is None


class TestFeedbackWidgetModel:
    def test_completed_grounded_answer_carries_the_widget(self) -> None:
        view = fold_chat_stream(grounded_stream())
        widget = feedback_widget_model(view)
        assert isinstance(widget, FeedbackWidget)
        assert widget.exchange_id == EXCHANGE_ID
        assert widget.up_verdict == FEEDBACK_UP
        assert widget.down_verdict == FEEDBACK_DOWN

    @pytest.mark.parametrize(
        "events",
        [
            pytest.param(
                [meta_event(), answer_event("canned", "A canned safety response.")],
                id="canned",
            ),
            pytest.param(
                [meta_event(), answer_event("refusal", "I can't answer that from my sources.")],
                id="refusal",
            ),
            pytest.param(
                [
                    meta_event(mode="paused"),
                    answer_event(
                        "cached_starter",
                        "A cached synthetic starter answer.",
                        generated_on="2026-08-15",
                        footer="Cached footer.",
                        citations=[],
                    ),
                ],
                id="cached_starter",
            ),
            pytest.param([meta_event(), chart_event()], id="chart"),
        ],
    )
    def test_every_completed_kind_carries_the_widget(self, events) -> None:
        """Feedback on refusals, canned answers and cached paused answers
        is exactly the triage signal #56 collects — the widget is not a
        grounded-only surface."""
        view = fold_chat_stream(events)
        assert view.complete is True, "fixture sanity: these views are completed answers"
        widget = feedback_widget_model(view)
        assert isinstance(widget, FeedbackWidget)
        assert widget.exchange_id == EXCHANGE_ID

    def test_no_widget_without_a_join_key(self) -> None:
        for view in (
            fold_chat_stream(grounded_stream(omit_key=True)),
            fold_chat_stream(
                [meta_event(exchange_id=None, mode="paused"), answer_event("paused", "Paused.")]
            ),
        ):
            assert feedback_widget_model(view) is None, (
                "no exchange record exists — a widget would POST into a guaranteed 404"
            )

    def test_no_widget_on_an_error_terminated_answer(self) -> None:
        events = [meta_event(), text_event("A partial "), error_event()]
        view = fold_chat_stream(events)
        assert feedback_widget_model(view) is None, (
            "an errored answer is not honestly rateable — the widget never "
            "dresses a failure up as a completed answer"
        )

    def test_no_widget_on_an_incomplete_stream(self) -> None:
        events = [meta_event(), text_event("A prefix that never finished")]
        view = fold_chat_stream(events)
        assert view.complete is False
        assert feedback_widget_model(view) is None


class TestFeedbackStateHonesty:
    def test_successful_post_shows_the_recorded_verdict(self) -> None:
        state = resolve_feedback_state(FEEDBACK_UP, True)
        assert state.status == FEEDBACK_STATE_RECORDED
        assert state.verdict == FEEDBACK_UP
        assert state.message == FEEDBACK_RECORDED_MESSAGE

    def test_failed_post_shows_unrecorded_never_fake_success(self) -> None:
        state = resolve_feedback_state(FEEDBACK_DOWN, False)
        assert state.status == FEEDBACK_STATE_UNRECORDED
        assert state.verdict is None, (
            "a failed POST must not display the thumb as registered — the "
            "optimistic click is rolled back honestly"
        )
        assert state.message == FEEDBACK_NOT_RECORDED_MESSAGE

    def test_the_pinned_messages_are_honest_prose(self) -> None:
        assert FEEDBACK_RECORDED_MESSAGE == "Thanks — your rating was recorded."
        assert FEEDBACK_NOT_RECORDED_MESSAGE == "Your rating wasn't recorded — please try again."

    @pytest.mark.parametrize("bad_verdict", ["sideways", "UP", "", "thumbs_up"])
    def test_unknown_verdicts_are_refused(self, bad_verdict: str) -> None:
        with pytest.raises(ValueError):
            resolve_feedback_state(bad_verdict, True)


class TestVerdictVocabularyParity:
    def test_ui_verdicts_are_the_service_verdicts(self) -> None:
        """The UI imports the vocabulary from service.exchange_log (a
        pure module the UI already depends on) — one closed vocabulary,
        no copy to drift. This pin fails if anyone ever forks it."""
        import service.exchange_log as exchange_log

        assert FEEDBACK_UP == exchange_log.FEEDBACK_UP
        assert FEEDBACK_DOWN == exchange_log.FEEDBACK_DOWN
        assert frozenset({FEEDBACK_UP, FEEDBACK_DOWN}) == exchange_log.FEEDBACK_VERDICTS


# ---------------------------------------------------------------------------
# Structural guards (the shell-hygiene pattern of test_ui_shell_hygiene.py).
# ---------------------------------------------------------------------------


def _app_tree() -> ast.Module:
    return ast.parse((UI_DIR / "app.py").read_text(encoding="utf-8"))


def _referenced_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


class TestShellFeedbackGuards:
    def test_shell_renders_the_widget_through_the_pure_model(self) -> None:
        referenced = _referenced_names(_app_tree())
        assert "feedback_widget_model" in referenced, (
            "ui/app.py must decide widget presence through the "
            "presenter-exported feedback_widget_model — never its own rule"
        )
        assert "resolve_feedback_state" in referenced, (
            "the post-click display state (recorded vs honestly unrecorded) "
            "is the pure resolve_feedback_state's decision, not the shell's"
        )

    def test_shell_posts_feedback_through_the_transport_seam(self) -> None:
        assert "http_feedback_transport" in referenced_app_names(), (
            "the feedback POST is network code and belongs to ui.transport's "
            "http_feedback_transport seam — the shell must not open its own"
        )

    def test_shell_has_no_verdict_string_literals(self) -> None:
        """The closed vocabulary reaches the shell only through the
        widget model's fields / the exported constants."""
        offending = [
            node.value
            for node in ast.walk(_app_tree())
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in {"up", "down", "recorded", "unrecorded"}
        ]
        assert not offending, (
            f"ui/app.py hardcodes feedback literals {sorted(set(offending))}; "
            "use the presenter-exported FEEDBACK_* constants"
        )

    def test_facade_exports_the_feedback_contract(self) -> None:
        import ui.presenters as presenters

        for name in (
            "FEEDBACK_UP",
            "FEEDBACK_DOWN",
            "FEEDBACK_UP_LABEL",
            "FEEDBACK_DOWN_LABEL",
            "FEEDBACK_STATE_RECORDED",
            "FEEDBACK_STATE_UNRECORDED",
            "FEEDBACK_RECORDED_MESSAGE",
            "FEEDBACK_NOT_RECORDED_MESSAGE",
            "FeedbackState",
            "FeedbackWidget",
            "FeedbackTransport",
            "feedback_widget_model",
            "resolve_feedback_state",
        ):
            assert hasattr(presenters, name), f"ui.presenters does not export {name}"


def referenced_app_names() -> set[str]:
    return _referenced_names(_app_tree())
