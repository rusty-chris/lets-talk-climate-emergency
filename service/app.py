"""The FastAPI service: chat SSE, permalinks, static surfaces (issue #22).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_service_*.py``,
``tests/integration/test_service_permalink_render.py`` and
``tests/smoke/test_cutoff_fails_closed.py`` pin the contract.

Routes are THIN (IMPLEMENTATION.md §1): every decision lives in the
injected dependencies (:class:`ServiceDeps`) or in the pure modules of
this package; the app factory wires them. ``service.main`` is the
composition root that reads the environment (``load_service_config``),
builds the real dependencies, and calls :func:`create_app`; tests build
apps from fakes and never touch the network.

## Route contract

- ``GET /health`` — ``{"status": "ok"}``, 200, in BOTH modes (a paused
  service is alive; monitoring must be able to tell pause from outage).
  Never rate-limited.
- ``POST /chat`` — body ``{"question": str, "history": [...]}``; SSE
  response (``text/event-stream``). Rate-limited per client IP; over the
  threshold → 429 with a body echoing nothing about the client.
- ``GET /chart/{spec_hash}`` — JSON ``{spec_hash, vega_lite, alt_text}``
  re-rendered from the STORED spec; ``GET /chart/{spec_hash}.csv`` —
  the attribution-headed CSV; ``GET /chart/{spec_hash}.svg`` — the SVG
  (vl-convert edge, integration-tier). Unknown or malformed hash → 404.
  NO fetch, NO LLM call — permalinks serve in both modes.
- ``GET /about``, ``GET /privacy``, ``GET /sources``, ``GET /voices`` —
  static surfaces, 200 in both modes. ``/privacy`` carries the
  :data:`service.exchange_log.LOGGING_DISCLOSURE` line and states the
  lawful basis ("legitimate interests"); ``/about`` links ``/privacy``.

## Chat SSE contract

Every chat response is one SSE stream. The FIRST event is always
:data:`META_EVENT` with data ``{"disclosure": LOGGING_DISCLOSURE,
"preamble_note": <str|None>, "mode": "live"|"paused"}`` — the #10
``preamble_note`` and the privacy disclosure are response-surface
furniture attached HERE (the #12 orchestrator ratification: they never
ride into any prompt). Then, by route:

- **RETRIEVAL (live):** the #12/#13 event vocabulary passed through
  unchanged and in order — ``text``/``citation``/``usage``/``footer``
  (complete answers) or a terminal ``error`` (no footer after an
  error), then the #13 ``badge``/``validation_degraded`` events via the
  injected ``append_validation_events``. Refusals
  (:class:`rag.retrieval.HonestRefusal`) become one :data:`ANSWER_EVENT`
  with ``kind == "refusal"`` — zero generation calls.
- **CHART (live):** planner → validator → renderer; success is one
  :data:`CHART_EVENT` with ``{"spec_hash", "permalink", "alt_text"}``
  (the spec stored so the permalink serves immediately); a
  :class:`charts.planner.ChartRefusal` is an :data:`ANSWER_EVENT` with
  ``kind == "refusal"``.
- **CANNED:** one :data:`ANSWER_EVENT` with ``kind == "canned"`` and
  the #10 canned text — zero generate/generate_stream calls, ever.
- **PAUSED (any route):** one :data:`ANSWER_EVENT` — ``kind ==
  "cached_starter"`` serving the dated cache entry when the question is
  a starter topic, else ``kind == "paused"`` with the dated
  ``paused_response_text`` — zero adapter calls of ANY kind.

Spend accounting: every adapter-reported usage mapping in the exchange
(classifier, generation stream ``usage`` event, planner, validation
outcome) is recorded into the spend tracker with its model id, and the
exchange is logged via ``build_exchange_record``/``ExchangeLog``.

Budget wiring: the generation config the chat route uses installs
``deps.spend_tracker.budget_guard`` as ``GenerationConfig.budget_guard``
(the #186-hardened gate). When the guard refuses Opus at its sub-cap
while the daily cap has room, the route falls back to the default model
for that query — Haiku continues under the daily cap (flagged for
ratification in the #22 red-phase notes).

## Startup contract

:func:`create_app` consults ``deps.index_corpus_version`` once at
construction: a recorded index version that MISMATCHES
``config.corpus_version`` raises :class:`ServiceStartupError` naming
both versions (a wrong deploy fails loudly, before traffic); a matching
version proceeds; ``None`` (no index recorded yet — e.g. the dev compose
stack before ingestion) starts the app with the read-only surfaces
serving.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI

from charts.planner import ChartRefusal, PlannedChart
from charts.render import ChartArtifact
from rag.provider import ProviderAdapter
from rag.query import QueryDecision
from rag.retrieval import HonestRefusal, RetrievedPassages
from service.budget import SpendTracker
from service.chart_store import ChartSpecStore
from service.config import ServiceConfig
from service.exchange_log import ExchangeLog
from service.rate_limit import RateLimiter
from service.starter_cache import StarterCache

__all__ = [
    "META_EVENT",
    "ANSWER_EVENT",
    "CHART_EVENT",
    "ANSWER_KIND_CANNED",
    "ANSWER_KIND_REFUSAL",
    "ANSWER_KIND_PAUSED",
    "ANSWER_KIND_CACHED_STARTER",
    "ServiceStartupError",
    "ServiceDeps",
    "format_sse_event",
    "create_app",
]

#: Service-level SSE vocabulary, extending the #12 (text/citation/usage/
#: footer/error) and #13 (badge/validation_degraded) events.
META_EVENT = "meta"
ANSWER_EVENT = "answer"
CHART_EVENT = "chart"

#: ``ANSWER_EVENT`` data ``kind`` values.
ANSWER_KIND_CANNED = "canned"
ANSWER_KIND_REFUSAL = "refusal"
ANSWER_KIND_PAUSED = "paused"
ANSWER_KIND_CACHED_STARTER = "cached_starter"


class ServiceStartupError(Exception):
    """The service refused to start (corpus/index version mismatch)."""


@dataclass(frozen=True)
class ServiceDeps:
    """Everything :func:`create_app` needs, injected (IMPLEMENTATION §1).

    The callables mirror the merged pipeline seams so tests fake them
    shape-for-shape:

    - ``retrieve``: ``QueryDecision -> RetrievedPassages | HonestRefusal``
      (``rag.retrieval.retrieve`` with its client/models/config bound).
    - ``plan_chart``: ``chart_request -> PlannedChart | ChartRefusal``
      (``charts.planner.plan_chart_request`` with adapter + manifest
      bound).
    - ``render_chart``: ``spec -> ChartArtifact``
      (``charts.render.render_chart`` with frames/manifest/site_url
      bound; pure — no fetch).
    - ``validate_exchange``: ``(result, sse_events) ->`` a #13
      ``ValidationOutcome``-shaped object; ``append_validation_events``:
      the #13 stream composer ``(events, validate) -> iterator``;
      ``exchange_log_record``: ``outcome -> mapping`` (the #13 seam —
      injected so this package never imports the in-flight #13 module).
    - ``index_corpus_version``: ``() -> str | None`` — the recorded
      index corpus version (``rag.indexing.get_index_corpus_version``
      bound), None when no index exists yet.
    - ``clock``: aware-UTC now; the ONLY time source the app uses.
    """

    adapter: ProviderAdapter
    retrieve: Callable[[QueryDecision], RetrievedPassages | HonestRefusal]
    plan_chart: Callable[[str], PlannedChart | ChartRefusal]
    render_chart: Callable[[Mapping[str, Any]], ChartArtifact]
    validate_exchange: Callable[[Any, Sequence[Mapping[str, Any]]], Any]
    append_validation_events: Callable[
        [Iterable[Mapping[str, Any]], Callable[[Sequence[Mapping[str, Any]]], Any]],
        Iterator[dict[str, Any]],
    ]
    exchange_log_record: Callable[[Any], Mapping[str, Any]]
    spend_tracker: SpendTracker
    rate_limiter: RateLimiter
    exchange_log: ExchangeLog
    starter_cache: StarterCache
    chart_spec_store: ChartSpecStore
    index_corpus_version: Callable[[], str | None]
    clock: Callable[[], datetime]


def format_sse_event(event: Mapping[str, Any]) -> str:
    """Pure: one ``{"event": name, "data": mapping}`` dict -> SSE wire text.

    ``event: <name>\\ndata: <compact JSON>\\n\\n`` — data is a single
    JSON line (JSON contains no raw newlines, so one ``data:`` field
    suffices and parsing stays trivial for #18 and the tests).
    """
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")


def create_app(config: ServiceConfig, deps: ServiceDeps) -> FastAPI:
    """The app factory: wire the routes over ``config`` + ``deps``.

    See the module docstring for the full pinned route/SSE/startup
    contract; the red suites under ``tests/unit/test_service_*.py`` are
    the source of truth.
    """
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")
