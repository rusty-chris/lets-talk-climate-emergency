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
  the transparency surfaces (issue #19; content contract in
  ``service.transparency``), 200 with ``text/html`` in BOTH modes,
  never rate-limited, zero adapter calls, nothing written to the
  exchange log. When ``deps.transparency`` (a
  :class:`service.transparency.TransparencyPages`) is provided, each
  route serves its page from ``TransparencyPages.as_route_map()``;
  ``None`` serves the interim pre-#19 placeholders. ``/privacy``
  carries the :data:`service.exchange_log.LOGGING_DISCLOSURE` line and
  states the lawful basis ("legitimate interests"); ``/about`` links
  ``/privacy``.

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

import asyncio
import contextlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from charts.planner import ChartRefusal, PlannedChart
from charts.render import ChartArtifact, render_svg
from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    CitedPassage,
    GenerationConfig,
    GroundedAnswer,
    stream_grounded_answer,
)
from rag.provider import ProviderAdapter
from rag.query import QueryDecision, Route, process_query
from rag.retrieval import HonestRefusal, RetrievedPassages
from service.budget import (
    OpusSubCapExceededError,
    ServiceMode,
    SpendTracker,
    paused_response_text,
)
from service.chart_store import ChartSpecStore
from service.config import ServiceConfig
from service.exchange_log import (
    LOGGING_DISCLOSURE,
    ExchangeLog,
    build_exchange_record,
)
from service.rate_limit import RateLimiter, resolve_client_ip
from service.retention import RETENTION_PURGE_INTERVAL, run_retention_pass
from service.starter_cache import StarterCache
from service.transparency import TransparencyPages

#: The single combined rewrite+classify call is a Haiku structured call
#: (rag.query); its usage is priced against this family.
CLASSIFIER_MODEL = GENERATION_MODEL_DEFAULT
#: The chart planner is a Haiku structured call (charts.planner).
PLANNER_MODEL = GENERATION_MODEL_DEFAULT

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
    #: The #19 transparency seam: the four pre-built pages the static
    #: routes serve (``service.main`` builds them at startup via
    #: ``service.transparency.build_transparency_pages``). ``None``
    #: serves the interim pre-#19 placeholders.
    transparency: TransparencyPages | None = None


def format_sse_event(event: Mapping[str, Any]) -> str:
    """Pure: one ``{"event": name, "data": mapping}`` dict -> SSE wire text.

    ``event: <name>\\ndata: <compact JSON>\\n\\n`` — data is a single
    JSON line (JSON contains no raw newlines, so one ``data:`` field
    suffices and parsing stays trivial for #18 and the tests).
    """
    data = json.dumps(event.get("data"), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event['event']}\ndata: {data}\n\n"


def _grounded_answer_from_sse(
    transcript: Sequence[Mapping[str, Any]],
    retrieved: RetrievedPassages,
    corpus_vintage: str,
) -> GroundedAnswer:
    """Reassemble a GroundedAnswer from a streamed SSE transcript.

    The #13 validator judges the answer's cited sentences against the
    passages it was built from; the streaming path never materialised a
    folded answer, so we rebuild one from the transcript's text and
    resolved citation events (each already carries its ``chunk_id`` and
    ``document_index`` from :func:`answer_stream_to_sse`).
    """
    text_parts: list[str] = []
    citations: list[CitedPassage] = []
    usage: Mapping[str, int] | None = None
    footer = ""
    for event in transcript:
        name = event.get("event")
        data = event.get("data") or {}
        if name == "text":
            text_parts.append(data.get("text", ""))
        elif name == "usage":
            usage = dict(data)
        elif name == "footer":
            footer = data.get("text", "")
        elif name == "citation":
            index = data.get("document_index")
            if isinstance(index, int) and 0 <= index < len(retrieved.passages):
                passage = retrieved.passages[index]
                citations.append(
                    CitedPassage(
                        chunk_id=data.get("chunk_id", passage.chunk_id),
                        document_index=index,
                        cited_text=data.get("cited_text", ""),
                        rerank_score=passage.rerank_score,
                        clears_threshold=bool(data.get("clears_threshold", True)),
                        degraded_fallback=bool(data.get("degraded_fallback", False)),
                        needs_hand_review=bool(data.get("needs_hand_review", False)),
                        payload=passage.payload,
                    )
                )
    return GroundedAnswer(
        text="".join(text_parts),
        cited_passages=tuple(citations),
        footer=footer,
        usage=usage,
    )


def create_app(config: ServiceConfig, deps: ServiceDeps) -> FastAPI:
    """The app factory: wire the routes over ``config`` + ``deps``.

    See the module docstring for the full pinned route/SSE/startup
    contract; the red suites under ``tests/unit/test_service_*.py`` are
    the source of truth.
    """
    # Startup contract: a recorded index version that MISMATCHES the
    # configured corpus is a wrong deploy — fail loudly, before traffic.
    # None (no index recorded yet, e.g. dev compose before ingestion) is
    # a legitimate read-only start.
    recorded_version = deps.index_corpus_version()
    if recorded_version is not None and recorded_version != config.corpus_version:
        raise ServiceStartupError(
            f"index corpus version {recorded_version!r} does not match the "
            f"configured corpus version {config.corpus_version!r}: refusing to "
            "start on a mismatched index (a wrong deploy fails loudly)"
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        # The retention purges exist but nothing in the running service
        # called them (#213); the rate-limit store is in-process memory, so
        # ONLY an in-process task can bound it. Run one pass at startup,
        # then every RETENTION_PURGE_INTERVAL for the process lifetime.
        run_retention_pass(deps.exchange_log, deps.rate_limiter)
        interval_s = RETENTION_PURGE_INTERVAL.total_seconds()

        async def _periodic_purge() -> None:
            while True:
                await asyncio.sleep(interval_s)
                run_retention_pass(deps.exchange_log, deps.rate_limiter)

        task = asyncio.create_task(_periodic_purge())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Let's Talk About the Climate Emergency — API", lifespan=lifespan)

    def record_usage_if(model: str | None, usage: Mapping[str, int] | None) -> None:
        """Record one adapter usage against the daily cap (no-op when absent)."""
        if model and usage:
            deps.spend_tracker.record_usage(model, usage)

    @app.get("/health")
    def health() -> dict[str, str]:
        # A paused service is still alive: /health answers 200 in both
        # modes and is NEVER rate-limited (monitoring must tell pause from
        # outage).
        return {"status": "ok"}

    @app.get("/about", response_class=HTMLResponse)
    def about() -> str:
        return _ABOUT_HTML

    @app.get("/privacy", response_class=HTMLResponse)
    def privacy() -> str:
        return _PRIVACY_HTML

    @app.get("/sources", response_class=HTMLResponse)
    def sources() -> str:
        return _SOURCES_HTML

    @app.get("/voices", response_class=HTMLResponse)
    def voices() -> str:
        return _VOICES_HTML

    def _load_spec_or_404(spec_hash: str) -> Mapping[str, Any]:
        spec = deps.chart_spec_store.get(spec_hash)
        if spec is None:
            # Unknown or malformed hash: a clean 404, never a 500 and never
            # a filesystem touch (the store validated the shape first).
            raise HTTPException(status_code=404, detail="unknown chart permalink")
        return spec

    # The suffixed permalink routes are declared BEFORE the bare one so
    # `<hash>.csv` / `<hash>.svg` never bind as a hash ending in the suffix.
    @app.get("/chart/{spec_hash}.csv")
    def chart_csv(spec_hash: str) -> Response:
        artifact = deps.render_chart(_load_spec_or_404(spec_hash))
        return Response(content=artifact.csv_text, media_type="text/csv")

    @app.get("/chart/{spec_hash}.svg")
    def chart_svg(spec_hash: str) -> Response:
        artifact = deps.render_chart(_load_spec_or_404(spec_hash))
        # The vl-convert edge (integration tier): pure spec -> SVG bytes.
        return Response(content=render_svg(artifact.vega_lite), media_type="image/svg+xml")

    @app.get("/chart/{spec_hash}")
    def chart(spec_hash: str) -> JSONResponse:
        artifact = deps.render_chart(_load_spec_or_404(spec_hash))
        # Re-rendered from the STORED spec — no fetch, no LLM call — so the
        # permalink serves in both live and paused modes.
        return JSONResponse(
            {
                "spec_hash": artifact.spec_hash,
                "vega_lite": artifact.vega_lite,
                "alt_text": artifact.alt_text,
            }
        )

    @app.post("/chat")
    def chat(payload: ChatRequest, request: Request) -> Response:
        # Rate limiting FIRST (before any adapter call): the (N+1)th
        # request over the window is refused with a body that echoes
        # nothing about the client.
        client_host = request.client.host if request.client else None
        forwarded_for = request.headers.get("x-forwarded-for")
        client_ip = resolve_client_ip(
            client_host, forwarded_for, trusted_proxy=config.trusted_proxy
        )
        if not deps.rate_limiter.allow(client_ip):
            return Response(
                content="rate limit exceeded — please slow down",
                status_code=429,
                media_type="text/plain",
            )

        question = payload.question
        history = [dict(turn) for turn in payload.history]
        stream = _chat_events(deps, config, question, history, record_usage_if)
        # Guarantee the generator's finalization (#211): on a client
        # disconnect Starlette (1.3.x, uvicorn ASGI spec 2.3) cancels the
        # stream task and leaves the sync generator to be closed by GC —
        # nondeterministic, so spend recording + exchange logging could be
        # skipped. It DOES await the response's background task afterwards,
        # even on disconnect, so closing the generator there deterministically
        # runs its try/finally (drain the charged usage, log the honest
        # partial). On normal completion the generator is already exhausted
        # and close() is a no-op.
        return StreamingResponse(
            (format_sse_event(event) for event in stream),
            media_type="text/event-stream",
            background=BackgroundTask(stream.close),
        )

    return app


class ChatRequest(BaseModel):
    """The POST /chat body: the latest question plus prior conversation turns."""

    question: str
    history: list[dict[str, Any]] = Field(default_factory=list)


def _meta_event(mode: ServiceMode, preamble_note: str | None) -> dict[str, Any]:
    return {
        "event": META_EVENT,
        "data": {
            "disclosure": LOGGING_DISCLOSURE,
            "preamble_note": preamble_note,
            "mode": mode.value,
        },
    }


def _answer_event(kind: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"event": ANSWER_EVENT, "data": {"kind": kind, "text": text, **extra}}


def _chat_events(
    deps: ServiceDeps,
    config: ServiceConfig,
    question: str,
    history: list[dict[str, Any]],
    record_usage_if: Callable[[str | None, Mapping[str, int] | None], None],
) -> Iterator[dict[str, Any]]:
    """The chat SSE event generator (meta first, then route-specific events)."""
    today = deps.clock().date()

    # PAUSED: the fail-closed read-only state. Zero adapter calls of ANY
    # kind; nothing is logged as content (a paused refusal is furniture,
    # not an exchange).
    if deps.spend_tracker.mode() is ServiceMode.PAUSED:
        yield _meta_event(ServiceMode.PAUSED, None)
        entry = deps.starter_cache.lookup(question)
        if entry is not None:
            yield _answer_event(
                ANSWER_KIND_CACHED_STARTER,
                entry.answer_text,
                generated_on=entry.generated_on,
                footer=entry.footer,
                citations=[dict(citation) for citation in entry.citations],
            )
        else:
            yield _answer_event(ANSWER_KIND_PAUSED, paused_response_text(today))
        return

    # LIVE: classify + route (exactly one structured call).
    decision = process_query(deps.adapter, question, history)
    record_usage_if(CLASSIFIER_MODEL, decision.classification.usage)
    yield _meta_event(ServiceMode.LIVE, decision.preamble_note)

    if decision.route is Route.CANNED:
        yield from _canned_events(deps, question, decision)
    elif decision.route is Route.CHART:
        yield from _chart_events(deps, question, decision, record_usage_if)
    else:
        yield from _retrieval_events(deps, config, question, decision, record_usage_if)


def _log_exchange(
    deps: ServiceDeps,
    *,
    question: str,
    route: str,
    answer_text: str,
    retrieved_chunk_ids: Sequence[str],
    citations: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    usage_records: Sequence[Mapping[str, Any]],
    exclude_from_harvest: bool,
) -> None:
    record = build_exchange_record(
        question=question,
        route=route,
        answer_text=answer_text,
        retrieved_chunk_ids=retrieved_chunk_ids,
        citations=citations,
        validation=validation,
        usage_records=usage_records,
        exclude_from_harvest=exclude_from_harvest,
        timestamp=deps.clock(),
    )
    deps.exchange_log.append(record)


def _canned_events(
    deps: ServiceDeps, question: str, decision: QueryDecision
) -> Iterator[dict[str, Any]]:
    text = decision.canned_response or ""
    # Log in a `finally` so a disconnect after the answer event still logs
    # the exchange (#211 — the canned window's finalization).
    try:
        yield _answer_event(ANSWER_KIND_CANNED, text)
    finally:
        _log_exchange(
            deps,
            question=question,
            route="canned",
            answer_text=text,
            retrieved_chunk_ids=[],
            citations=[],
            validation={},
            usage_records=[],
            exclude_from_harvest=decision.exclude_from_harvest,
        )


def _chart_events(
    deps: ServiceDeps,
    question: str,
    decision: QueryDecision,
    record_usage_if: Callable[[str | None, Mapping[str, int] | None], None],
) -> Iterator[dict[str, Any]]:
    result = deps.plan_chart(decision.chart_request or "")
    record_usage_if(PLANNER_MODEL, getattr(result, "usage", None))
    # Log in a `finally` (both branches) so the planner's charged usage is
    # always logged with the exchange, even on a mid-window disconnect (#211).
    if isinstance(result, ChartRefusal):
        try:
            yield _answer_event(ANSWER_KIND_REFUSAL, result.message)
        finally:
            _log_exchange(
                deps,
                question=question,
                route="chart",
                answer_text=result.message,
                retrieved_chunk_ids=[],
                citations=[],
                validation={},
                usage_records=[],
                exclude_from_harvest=decision.exclude_from_harvest,
            )
        return
    artifact = deps.render_chart(result.spec)
    stored_hash = deps.chart_spec_store.put(result.spec)
    try:
        yield {
            "event": CHART_EVENT,
            "data": {
                "spec_hash": stored_hash,
                "permalink": f"/chart/{stored_hash}",
                "alt_text": artifact.alt_text,
            },
        }
    finally:
        _log_exchange(
            deps,
            question=question,
            route="chart",
            answer_text=artifact.alt_text,
            retrieved_chunk_ids=[],
            citations=[],
            validation={},
            usage_records=[{"model": PLANNER_MODEL, "usage": getattr(result, "usage", None)}],
            exclude_from_harvest=decision.exclude_from_harvest,
        )


def _retrieval_events(
    deps: ServiceDeps,
    config: ServiceConfig,
    question: str,
    decision: QueryDecision,
    record_usage_if: Callable[[str | None, Mapping[str, int] | None], None],
) -> Iterator[dict[str, Any]]:
    retrieval_result = deps.retrieve(decision)
    if isinstance(retrieval_result, HonestRefusal):
        try:
            yield _answer_event(ANSWER_KIND_REFUSAL, retrieval_result.refusal_text)
        finally:
            _log_exchange(
                deps,
                question=question,
                route="retrieval",
                answer_text=retrieval_result.refusal_text,
                retrieved_chunk_ids=[],
                citations=[],
                validation={},
                usage_records=[],
                exclude_from_harvest=decision.exclude_from_harvest,
            )
        return

    # Best mode: try the gated Opus model behind its sub-cap guard; when the
    # sub-cap is spent but the daily cap has room, fall back to the default
    # model — the visitor gets an answer, not a refusal.
    gen_model = GENERATION_MODEL_DEFAULT

    def build_stream(model: str) -> Iterator[dict[str, Any]]:
        gen_config = GenerationConfig(
            model=model,
            best_mode_enabled=config.best_mode_enabled,
            budget_guard=deps.spend_tracker.budget_guard,
        )
        return stream_grounded_answer(
            deps.adapter,
            retrieval_result,
            question,
            config=gen_config,
            corpus_vintage=config.corpus_vintage,
        )

    if config.best_mode_enabled:
        try:
            sse_iter = build_stream(OPUS_BEST_MODEL)
            gen_model = OPUS_BEST_MODEL
        except OpusSubCapExceededError:
            sse_iter = build_stream(GENERATION_MODEL_DEFAULT)
            gen_model = GENERATION_MODEL_DEFAULT
    else:
        sse_iter = build_stream(GENERATION_MODEL_DEFAULT)

    outcome_holder: dict[str, Any] = {}

    def validate(transcript: Sequence[Mapping[str, Any]]) -> Any:
        answer = _grounded_answer_from_sse(transcript, retrieval_result, config.corpus_vintage)
        outcome = deps.validate_exchange(answer, transcript)
        outcome_holder["outcome"] = outcome
        return outcome

    # Spend recording + exchange logging must survive a client disconnect
    # (#211): StreamingResponse closes this generator (GeneratorExit at the
    # current yield) when the client drops the SSE connection, but the
    # provider was already sent the full generation request and BILLS the
    # tokens. So: record each usage event as it passes (not after the loop),
    # and run finalization in a `finally` — draining the remaining transport
    # events (bounded) to capture the terminal usage, then logging the
    # exchange with whatever was actually delivered (honest partial logging).
    usage_records: list[dict[str, Any]] = []
    delivered_text: list[str] = []
    citations: list[Mapping[str, Any]] = []
    usage_recorded = False
    events_iter = iter(deps.append_validation_events(sse_iter, validate))

    def absorb(event: Mapping[str, Any], *, delivered: bool) -> None:
        nonlocal usage_recorded
        name = event["event"]
        if name == "usage":
            # Meter the charged usage exactly once, the moment it is seen —
            # whether delivered to the client or drained in finalization.
            if not usage_recorded:
                record_usage_if(gen_model, event["data"])
                usage_records.append({"model": gen_model, "usage": event["data"]})
                usage_recorded = True
        elif delivered and name == "text":
            # answer_text logs only what the client actually received.
            delivered_text.append(event["data"].get("text", ""))
        elif delivered and name == "citation":
            citations.append(event["data"])

    try:
        for event in events_iter:
            absorb(event, delivered=True)
            yield dict(event)
    finally:
        # Drain any transport events not yet delivered (a disconnect leaves
        # the stream — and its terminal usage event — mid-flight); this runs
        # on normal completion too, where nothing is left. No yield here:
        # yielding during GeneratorExit is an error.
        with contextlib.suppress(Exception):
            for event in events_iter:
                absorb(event, delivered=False)

        outcome = outcome_holder.get("outcome")
        validation: Mapping[str, Any] = {}
        if outcome is not None:
            validation = deps.exchange_log_record(outcome)
            record_usage_if(getattr(outcome, "model", None), getattr(outcome, "usage", None))

        _log_exchange(
            deps,
            question=question,
            route="retrieval",
            answer_text="".join(delivered_text),
            retrieved_chunk_ids=[passage.chunk_id for passage in retrieval_result.passages],
            citations=citations,
            validation=validation,
            usage_records=usage_records,
            exclude_from_harvest=decision.exclude_from_harvest,
        )


_ABOUT_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>About — Let's Talk About the Climate Emergency</title></head><body>
<h1>About this briefing</h1>
<p>An evidence-grounded climate briefing that answers only from a named,
clearly-licensed corpus. Every answer cites the source text it draws on.</p>
<p>See our <a href="/privacy">privacy notice</a>, the
<a href="/sources">source library</a>, and the
<a href="/voices">voices of the climate movement</a>.</p>
</body></html>"""

_PRIVACY_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Privacy — Let's Talk About the Climate Emergency</title></head><body>
<h1>Privacy notice</h1>
<p>{LOGGING_DISCLOSURE}</p>
<p>Lawful basis: we process conversation logs under our legitimate interests
in operating and improving an anonymous public-education service. We store no
IP addresses, cookies, accounts or other identifiers alongside conversations;
hashed request counts used only for rate-limiting are held separately and for
no more than seven days.</p>
</body></html>"""

_SOURCES_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Sources — Let's Talk About the Climate Emergency</title></head><body>
<h1>Source library</h1>
<p>Every answer is grounded in this clearly-licensed corpus. Each cited
passage links back to its named source document.</p>
</body></html>"""

_VOICES_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Voices — Let's Talk About the Climate Emergency</title></head><body>
<h1>Voices of the climate movement</h1>
<p>First-party testimony from the climate movement, kept structurally
separate from the assessed scientific evidence.</p>
</body></html>"""
