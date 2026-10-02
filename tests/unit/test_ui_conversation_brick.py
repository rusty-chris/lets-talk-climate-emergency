"""Hotfix RED — a chart turn must not poison the history, and a zero-event
turn must never brick the page (live blocker, climateemergency.chat, 2026-10).

The confirmed defect chain, pinned test by test:

1. The flagship chart starter streams ONLY meta + chart events — no text,
   no answer event (``service.app._cached_starter_events``). The shell
   therefore persists the turn with ``answer_text == ""`` (``st.write_stream``
   returns "" for a text-less stream and ``fold_chat_stream``'s ``view.text``
   is "" for a chart-only stream).
2. ``ui/app.py:_history_from_turns`` emits that turn as an assistant message
   with EMPTY content; nothing downstream filters it, so it reaches the
   provider request and the Anthropic Messages API rejects it with a 400 —
   the follow-up's classifier call dies BEFORE the meta event.
3. The aborted stream persists a turn with ``events == ()``; on the next
   Streamlit rerun the replay fold raises ``StreamContractError`` (only
   ``TransportError``/``SseProtocolError`` are caught), the page crashes,
   and chat input / starters / the ADR-018 footer all vanish — the whole
   conversation is bricked. The same brick fires for ANY zero-event turn
   (our own rate limiter's 429 mid-handshake, an api restart).

Two tiers of pin here, both hermetic ($0, no socket):

- Pure-ish shell behaviour on ``_history_from_turns`` (the function the
  shell-hygiene suite already requires ``ui/app.py`` to keep): no turn shape
  that can exist in session state may ever yield an empty-content message.
- Render-level behaviour through Streamlit's ``AppTest`` — the suite's first
  render tests (closing the #381 review nit: the helper-string tests prove a
  string's shape, not that the page survives rendering it). Every network
  seam (``ui.transport``) is monkeypatched; ``ui/app.py`` binds those names
  at script-exec time, so patching the module attribute reaches the AppTest
  run.

The sibling seam test (defence in depth at the service boundary) lives in
tests/unit/test_query_classifier.py; the no-public-traceback config pin in
tests/unit/test_ui_theme.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import ui.transport as ui_transport
from tests._ui_fixtures import chart_event, footer_event, meta_event, text_event, wire_lines
from ui.starter import free_text_submission, starter_submission

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "ui" / "app.py"

#: The canonical flagship chart starter question (the one everyone clicks
#: first — service/dev_starter_cache's chart entry) and its event stream as
#: the service actually emits it: meta + chart, NO text, NO answer event.
FLAGSHIP_QUESTION = "Show me CO₂ and temperature over the last 10,000 years"
FLAGSHIP_ALT_TEXT = "Line chart: CO2 and temperature over the last 10,000 years."


def _chart_turn() -> dict[str, Any]:
    """A chart turn EXACTLY as ui/app.py persists it today.

    ``answer_text`` is "" because ``st.write_stream`` returns "" for a
    stream with no text events and ``fold_chat_stream``'s ``view.text`` is
    "" for a chart-only stream — verified end-to-end by
    ``TestChartFollowUpEndToEnd``, which lets the REAL shell persist the
    turn. The history builder must stay honest for this shape regardless of
    any future persistence change: old session states live across deploys.
    """
    return {
        "question": FLAGSHIP_QUESTION,
        "events": (meta_event(), chart_event(alt_text=FLAGSHIP_ALT_TEXT)),
        "answer_text": "",
    }


def _failed_turn() -> dict[str, Any]:
    """A turn whose stream aborted before meta — exactly what the shell
    persists after a pre-meta failure (the 400'd follow-up, a 429 at the
    handshake, an api restart): no events, no text."""
    return {"question": "And what about methane?", "events": (), "answer_text": ""}


def _shell_namespace() -> dict[str, Any]:
    """Execute ui/app.py WITHOUT its module-trailer ``main()`` call.

    Streamlit runs the module top to bottom, so a plain import would render
    the whole shell (and open the budget fetch). Stripping the top-level
    expression-call statements leaves the imports, constants and function
    defs — the honest module surface — executable offline.
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
    ]
    namespace: dict[str, Any] = {}
    exec(compile(tree, str(APP_PATH), "exec"), namespace)  # noqa: S102 — our own source
    return namespace


class TestHistoryNeverCarriesEmptyContent:
    """``_history_from_turns`` output is provider-request material: the
    Anthropic Messages API 400s on ANY empty-content message, so one empty
    string in the history kills every follow-up in the conversation."""

    def test_chart_turn_contributes_a_non_empty_assistant_message(self) -> None:
        """The flagship chart answer must stay IN the history with meaningful
        content (its natural representation is the chart's alt text — the
        turn's events carry it) so "why does that matter?" can resolve
        "that"; it must never ride as ``content: ""``."""
        history = _shell_namespace()["_history_from_turns"]([_chart_turn()])
        for message in history:
            assert isinstance(message["content"], str) and message["content"].strip(), (
                f"history message {message!r} has empty content — the provider "
                "rejects it with a 400 and the follow-up dies before its meta event"
            )
        assistant_messages = [m for m in history if m["role"] == "assistant"]
        assert assistant_messages, (
            "the chart exchange must remain in the history (dropping it loses "
            "the thread the follow-up refers back to)"
        )

    def test_zero_event_turn_never_emits_an_empty_content_message(self) -> None:
        """A failed turn (events=(), answer_text="") is a legitimate session
        state — it is what the shell persists after any pre-meta stream
        failure. It may be dropped from the history or represented honestly,
        but it must NEVER surface as an empty-content message."""
        history = _shell_namespace()["_history_from_turns"]([_failed_turn()])
        for message in history:
            assert isinstance(message["content"], str) and message["content"].strip(), (
                f"history message {message!r} has empty content — one failed turn "
                "must not poison every subsequent question in the conversation"
            )


@pytest.fixture
def silenced_transport(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[dict[str, Any]]]]:
    """Silence every ``ui.transport`` seam for an AppTest run; returns the
    captured ``(question, history)`` of each POST /chat the shell opens.

    The fake chat transport answers like the live flagship starter: meta +
    chart for the flagship question, a plain grounded answer otherwise —
    written through the service's own ``format_sse_event`` (``wire_lines``)
    so the shell consumes the real wire shape, not an imitation.
    """
    captured: list[tuple[str, list[dict[str, Any]]]] = []

    def fake_http_chat_transport(base_url: str, **_: Any):
        def transport(question: str, history):
            captured.append((question, [dict(turn) for turn in history]))
            if question == FLAGSHIP_QUESTION:
                events = [meta_event(), chart_event(alt_text=FLAGSHIP_ALT_TEXT)]
            else:
                events = [meta_event(), text_event("A short grounded answer."), footer_event()]
            yield from wire_lines(events)

        return transport

    monkeypatch.setattr(ui_transport, "http_chat_transport", fake_http_chat_transport)
    # The svg/budget/feedback fetches degrade to None/False by contract; a
    # unit test never opens a socket (IMPLEMENTATION.md §3), not even to a
    # refused localhost port.
    monkeypatch.setattr(ui_transport, "fetch_chart_svg", lambda *a, **k: None)
    monkeypatch.setattr(ui_transport, "fetch_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        ui_transport, "http_feedback_transport", lambda *a, **k: lambda *_a, **_k: False
    )
    return captured


class TestZeroEventTurnNeverBricksTheRerun:
    """The render-level pin (the repo's first AppTest): replaying a failed
    turn must degrade honestly, never crash the page. Today
    ``fold_chat_stream(())`` raises ``StreamContractError``, the shell
    catches only ``TransportError``/``SseProtocolError``, and the WHOLE page
    dies — input, starters and ADR-018 footer gone, traceback public."""

    def test_rerun_with_a_failed_turn_renders_a_working_page(
        self, silenced_transport: list
    ) -> None:
        at = AppTest.from_file(str(APP_PATH))
        # Exactly what ui/app.py persists for a turn whose stream died
        # before meta (the brick's step 6).
        at.session_state["turns"] = [_failed_turn()]
        at.run(timeout=10)

        # 1) No uncaught exception reaches the page (today: StreamContractError).
        assert not at.exception, (
            "replaying a zero-event turn crashed the whole page: "
            f"{at.exception[0].value if at.exception else ''} — every rerun of "
            "this session is bricked (chat input, starters and footer all vanish)"
        )
        # 2) The conversation stays usable: the follow-up input still renders.
        assert len(at.chat_input) == 1, (
            "the chat input must survive a failed turn — the visitor can always ask again"
        )
        # 3) The failed turn shows an HONEST degraded view: its question is
        #    still on the page and some status element (st.warning/st.error/
        #    st.info — the answer_status_lines / finding-#224 vocabulary)
        #    says the answer did not complete.
        page_markdown = " ".join(block.value for block in at.markdown)
        assert _failed_turn()["question"] in page_markdown, (
            "the failed turn's question must stay visible in the thread"
        )
        status_texts = [el.value for el in (*at.warning, *at.error, *at.info)]
        assert status_texts, (
            "the failed turn must be shown as honestly degraded (a warning/"
            "error/info status), not silently blank"
        )
        # 4) And nothing on the page is a traceback (the no-public-traceback
        #    rule; the config-side pin lives in test_ui_theme.py).
        everything = page_markdown + " ".join(status_texts)
        for leak in ("Traceback", "StreamContractError"):
            assert leak not in everything, f"internal error text {leak!r} reached the page"
        # 5) Degrading the broken turn must not quietly re-POST it either —
        #    replay stays transport-free (the finding-#226 rule holds in the
        #    degraded path too).
        assert silenced_transport == [], (
            "rendering a stored (failed) turn must never open POST /chat"
        )


class TestChartFollowUpEndToEnd:
    """The whole defect end to end through the REAL shell: the first run
    streams the flagship chart answer and ui/app.py itself persists the
    turn; the second run asks a follow-up and the history it POSTs is
    captured at the transport seam. No message in that history may be
    empty — the live API 400s on it and the brick chain begins."""

    def test_follow_up_after_a_chart_answer_posts_no_empty_content_history(
        self, silenced_transport: list
    ) -> None:
        at = AppTest.from_file(str(APP_PATH))
        at.session_state["pending"] = starter_submission(FLAGSHIP_QUESTION)
        at.run(timeout=10)
        assert not at.exception, "the chart answer itself must render cleanly"

        at.session_state["pending"] = free_text_submission("Why does that matter?")
        at.run(timeout=10)
        assert not at.exception, "the follow-up must render cleanly"

        follow_up_calls = [
            (question, history)
            for question, history in silenced_transport
            if question != FLAGSHIP_QUESTION
        ]
        assert follow_up_calls, "the follow-up must reach the transport"
        _, history = follow_up_calls[-1]
        assert history, "the follow-up must carry the conversation so far"
        for message in history:
            assert isinstance(message["content"], str) and message["content"].strip(), (
                f"the follow-up POSTed an empty-content history message {message!r} "
                "— the provider 400s, the stream aborts pre-meta, and the next "
                "rerun bricks the page (the live climateemergency.chat blocker)"
            )
