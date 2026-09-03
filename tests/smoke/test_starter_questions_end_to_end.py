"""Starter-question acceptance against the paused stack, smoke tier.

Issue #18's step-9 criterion, extended by review finding #231 (plus the
smoke halves of findings #225 and #228): the original test drove ONE of
the 13 starter questions; the acceptance criterion is ALL of them — the
four-group landing partition is built on the full set, the loop reuses
the same running stack, and the dev cache guarantees a citation per
entry. This module keeps the original test's every invariant (dated
cached-starter view, chips with chunk_id/attribution/quote, deduped
sources) and widens coverage:

- every §7.1 starter question folds end to end through the ui pure
  modules (``starter_submission`` -> ``http_chat_transport`` ->
  ``stream_chat_events`` -> ``fold_chat_stream``) against the composed
  api's REAL SSE bytes;
- a free-text question composes end to end through the SAME path via
  ``free_text_submission`` (finding #225 — RED until the pure model and
  the service round-trip exist);
- the four footer transparency routes are servable at the SITE_URL
  origin in the paused stack (finding #228) — they are the read-only
  surfaces DESIGN §9 keeps up.

Paused, like tests/smoke/test_cutoff_fails_closed.py: daily cap 0 forces
the read-only state, the committed synthetic dev cache serves answers,
and there is no real API key — a single leaked live call would fail
loudly, never spend. The LIVE (replay-adapter) path is covered by
tests/smoke/test_starter_live_replay.py, a separate module so the two
stack environments never collide on host ports.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import httpx
import pytest

from service.starter_cache import STARTER_QUESTIONS
from tests.smoke._compose_stack import API, compose_stack
from ui.render_model import fold_chat_stream
from ui.sse_client import stream_chat_events
from ui.starter import free_text_submission, starter_submission
from ui.transport import http_chat_transport

pytestmark = pytest.mark.smoke

#: The paused (read-only) stack: cap 0 is a breach from the first request;
#: the committed synthetic dev cache is the default starter cache.
PAUSED_ENV = {
    "CLIMATE_CHAT_DAILY_BUDGET_USD": "0",
    "CLIMATE_CHAT_OPUS_SUBCAP_USD": "0",
    "CLIMATE_CHAT_CORPUS_VERSION": "corpus-smoke-0000",
    "CLIMATE_CHAT_CORPUS_VINTAGE": "2026-08-01",
    "CLIMATE_CHAT_SITE_URL": "http://localhost:8000",
    "CLIMATE_CHAT_QDRANT_URL": "http://qdrant:6333",
    "CLIMATE_CHAT_STARTER_CACHE_DIR": "/app/service/dev_starter_cache",
    "CLIMATE_CHAT_LOG_DIR": "/tmp/climate-chat-logs",
    # Presence-checked only; NOT a real key.
    "ANTHROPIC_API_KEY": "smoke-placeholder-not-a-real-key",
}


@pytest.fixture(scope="module")
def paused_stack() -> Iterator[dict[str, str]]:
    with compose_stack(PAUSED_ENV, context="paused") as env:
        yield env


def _fold_question(question: str, history: tuple[Mapping, ...] = ()):
    transport = http_chat_transport(API)
    events = list(stream_chat_events(transport, question, history))
    return fold_chat_stream(events)


def test_starter_questions_end_to_end(paused_stack) -> None:
    """ALL 13 §7.1 questions — finding #231: '1 of 13' was a fraction of
    the acceptance criterion; the loop is nearly free on the same stack.
    Every invariant of the original single-question test is preserved,
    now asserted per question."""
    for question in STARTER_QUESTIONS:
        # The click-equivalent: a starter topic tap IS an immediate,
        # verbatim submission — routed through the real transport and the
        # ui pure fold, exactly as the Streamlit shell composes them.
        submission = starter_submission(question)
        assert submission.question == question
        assert submission.history == ()

        view = _fold_question(submission.question, submission.history)

        # The paused stack served the dated cached starter answer, folded
        # through the ui pure modules.
        assert view.mode == "paused", question
        assert view.kind == "cached_starter", question
        assert view.generated_on == "2026-08-15", question
        assert view.text, f"the cached starter answer text must render: {question!r}"
        assert view.complete is True, question
        assert view.footer_text, f"the cached answer carries its dated footer: {question!r}"

        # Its citations became chips (the synthetic dev cache carries at
        # least one citation per starter answer) and a deduped source
        # list — the real cached-starter surface, end to end.
        assert view.chips, f"cached starter citations must render as chips: {question!r}"
        first = view.chips[0]
        assert first.chunk_id and first.attribution and first.quote
        assert view.sources and view.sources[0].chunk_id == first.chunk_id


def test_free_text_question_end_to_end(paused_stack) -> None:
    """Review finding #225 (smoke half) — the free-text path composes end
    to end: a non-starter question submitted through free_text_submission
    takes the SAME transport/fold path and receives the service's dated
    paused view (kind 'paused'; the service, not the UI, decides mode)."""
    submission = free_text_submission("Is it too late for my kids?")
    assert submission is not None
    assert submission.question == "Is it too late for my kids?"

    view = _fold_question(submission.question, submission.history)

    assert view.mode == "paused"
    assert view.kind == "paused"
    assert view.complete is True
    # The paused response is dated (paused_response_text carries today's
    # date) — honest about being the read-only state, fabricating nothing.
    assert view.text
    assert any(char.isdigit() for char in view.text), "the paused text must be dated"
    assert view.chips == ()


def test_transparency_routes_served_at_site_origin(paused_stack) -> None:
    """Review finding #228 (smoke half) — the footer's promises are
    servable in the paused stack: every transparency route the footer
    links must answer 200 at the SITE_URL origin (they are the read-only
    surfaces DESIGN §9 keeps up; /privacy is a legal surface)."""
    from ui.footer import TRANSPARENCY_ROUTES

    for route in TRANSPARENCY_ROUTES:
        response = httpx.get(f"{API}{route}", timeout=10, follow_redirects=True)
        assert response.status_code == 200, f"{route} must be servable at the site origin"
        assert response.text.strip(), f"{route} must carry content"
