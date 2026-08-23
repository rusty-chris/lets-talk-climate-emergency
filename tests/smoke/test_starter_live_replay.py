"""Live-path smoke over the ReplayAdapter-composed stack (finding #231) — RED.

Issue #18's TDD plan step 9 names the ReplayAdapter: *"all §7.1 questions
run against the composed stack with the ReplayAdapter; charts render
inline with download links."* What shipped runs paused-mode only — the
live retrieval SSE path (meta -> text/citation/usage/footer through the
REAL SSE writer and uvicorn framing into the real fold) and the flagship
chart's permalink/.csv/.svg seam have ZERO end-to-end coverage; the
composed stack has never once streamed a grounded answer through
``parse_sse_stream``.

RED contract (flagged decisions, pinned here for the implementer):

- ``service/config.py`` + ``service/main.py`` gain an explicit
  composition-root provider switch: ``CLIMATE_CHAT_PROVIDER=replay``
  plus ``CLIMATE_CHAT_REPLAY_DIR`` construct ``rag.provider.ReplayAdapter``
  over checked-in fixtures (default remains Anthropic; replay is never
  selected implicitly). Unit pins live in
  tests/unit/test_service_config.py::TestProviderSelection.
- docker-compose passes the two variables through to the api service.
- Replay fixtures are recorded for one grounded starter and the chart
  starter under tests/fixtures/replay/starter_smoke/ (the image already
  COPYies the repo, so the in-container path below exists once the
  fixtures are committed).

Zero live API calls (CI convention, finding #32): the adapter replays
recorded responses; the placeholder ANTHROPIC_API_KEY guarantees any
leaked live call fails loudly instead of spending.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from service.starter_cache import STARTER_QUESTIONS
from tests.smoke._compose_stack import API, compose_stack
from ui.render_model import fold_chat_stream
from ui.sse_client import stream_chat_events
from ui.starter import starter_submission
from ui.transport import http_chat_transport

pytestmark = pytest.mark.smoke

#: The §7.1 "wow moment" chart starter — pinned present in the one
#: source of truth by tests/unit/test_ui_starter.py.
CHART_STARTER = "Show me CO₂ and temperature over the last 10,000 years"

#: A grounded (retrieval-path) starter with recorded replay fixtures.
GROUNDED_STARTER = STARTER_QUESTIONS[0]

#: The replay-composed live stack: budget open, provider pinned to the
#: ReplayAdapter over the checked-in smoke fixtures (in-container path —
#: the shared image COPYies the repo to /app).
REPLAY_ENV = {
    "CLIMATE_CHAT_PROVIDER": "replay",
    "CLIMATE_CHAT_REPLAY_DIR": "/app/tests/fixtures/replay/starter_smoke",
    "CLIMATE_CHAT_DAILY_BUDGET_USD": "5.00",
    "CLIMATE_CHAT_OPUS_SUBCAP_USD": "1.00",
    "CLIMATE_CHAT_CORPUS_VERSION": "corpus-smoke-0000",
    "CLIMATE_CHAT_CORPUS_VINTAGE": "2026-08-01",
    "CLIMATE_CHAT_SITE_URL": "http://localhost:8000",
    "CLIMATE_CHAT_QDRANT_URL": "http://qdrant:6333",
    "CLIMATE_CHAT_STARTER_CACHE_DIR": "/app/service/dev_starter_cache",
    "CLIMATE_CHAT_LOG_DIR": "/tmp/climate-chat-logs",
    # Presence-checked only; the ReplayAdapter never uses it, and a leaked
    # live call fails auth loudly instead of spending.
    "ANTHROPIC_API_KEY": "smoke-placeholder-not-a-real-key",
}


@pytest.fixture(scope="module")
def replay_stack() -> Iterator[dict[str, str]]:
    with compose_stack(REPLAY_ENV, context="replay") as env:
        yield env


def _fold_starter(question: str):
    submission = starter_submission(question)
    transport = http_chat_transport(API)
    events = list(stream_chat_events(transport, submission.question, submission.history))
    return fold_chat_stream(events)


def test_starter_live_replay_end_to_end(replay_stack) -> None:
    """One grounded starter streams through the REAL wire: the service's
    SSE writer and uvicorn's chunking into parse_sse_stream and the fold
    — the seam the unit tier's synthetic in-memory events cannot cover
    (format_sse_event framing, line chunking, the citation event's field
    names as actually serialised)."""
    view = _fold_starter(GROUNDED_STARTER)

    assert view.mode == "live"
    assert view.kind == "grounded"
    assert view.complete is True, "the replayed grounded stream must reach its footer"
    assert view.text.strip(), "the grounded answer streams real prose"
    assert view.footer_text, "the §3.5 footer event must arrive over the real wire"

    # Chips keyed to sentences from the real citation events: the #13
    # segmentation over the real transcript, not fixture dicts.
    assert view.chips, "the grounded answer must carry citation chips"
    for chip in view.chips:
        assert chip.chunk_id and chip.quote and chip.attribution
    # Badge/validation events, when the recorded fixtures include them,
    # attach per the #13 contract: never on a degraded validation.
    if view.validation_degraded:
        assert all(chip.badges == () for chip in view.chips)


def test_chart_starter_renders_with_working_download_links(replay_stack) -> None:
    """The flagship chart end to end: the chart event folds to a
    ChartView whose DERIVED hrefs ({permalink}.csv / {permalink}.svg)
    agree with the service's REAL routes — exactly the suffix-vs-route
    seam a smoke exists to pin. Every 'View data & sources' link on a
    shared page breaks silently if these drift."""
    view = _fold_starter(CHART_STARTER)

    assert view.kind == "chart"
    assert view.chart is not None
    assert view.chart.alt_text.strip(), "the chart arrives with its §3.7 alt text"

    csv_response = httpx.get(view.chart.csv_href, timeout=30)
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    # Attribution header comments ride the data download (§3.7).
    assert csv_response.text.lstrip().startswith("#"), (
        "the .csv must open with attribution header comments"
    )

    svg_response = httpx.get(view.chart.svg_href, timeout=30)
    assert svg_response.status_code == 200
    assert svg_response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in svg_response.text

    # The permalink page itself resolves too — the copyable link works.
    permalink_response = httpx.get(view.chart.permalink, timeout=30)
    assert permalink_response.status_code == 200
