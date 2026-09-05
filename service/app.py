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
- ``POST /feedback`` (issue #56) — body ``{"exchange_id": str,
  "verdict": "up"|"down"}`` (the closed
  ``service.exchange_log.FEEDBACK_VERDICTS`` vocabulary; anything else
  — miscased, blank, missing, non-string — → 422). Lands the verdict on
  the matched exchange record via ``ExchangeLog.record_feedback``:
  success is 204 with an EMPTY body (nothing echoed — not the id, not
  the verdict); an ``exchange_id`` matching no retained record → 404
  with a constant body that reveals NOTHING — a purged (90-day) id and
  a never-issued id are byte-identical 404s, so the endpoint can never
  be used to probe what was once logged. Rate-limited by the SAME
  hashed-IP limiter as ``/chat`` (429 echoes nothing about the
  client). Zero adapter calls, so it serves in BOTH modes — feedback
  on a cached answer while paused is valid signal. NO new identifier
  surface: nothing about the client is stored, joined, or logged by
  this route beyond the limiter's existing hashed count.
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
"preamble_note": <str|None>, "mode": "live"|"paused", "exchange_id":
<hex|None>}`` — the #10 ``preamble_note`` and the privacy disclosure are
response-surface furniture attached HERE (the #12 orchestrator
ratification: they never ride into any prompt).

``exchange_id`` (issue #56): the feedback join key, minted ONCE per
exchange (``uuid4().hex``) at stream start and passed into
``build_exchange_record`` so the id on the wire IS the id on the logged
record — the visitor's thumbs click posts back exactly the id their
exchange was logged under. It identifies the exchange, never the
person, and rides the EXISTING meta event: no new SSE event name, so
``SSE_EVENT_NAMES`` and the UI's handled/ignored parity are untouched,
and — being server-side furniture — no provider request or replay
request-hash changes. On every logged route (retrieval, chart, canned,
and the paused cached-starter path below) it is a fresh UUID hex; on
the paused non-starter furniture path — where NOTHING is logged and
there is no record to join — it is ``None`` (the key is always
present).

Then, by route:

- **RETRIEVAL (live):** immediately after ``meta`` and BEFORE the first
  ``text`` event, exactly one :data:`SOURCES_EVENT` (issue #220) built
  from the retrieval result by the pure ``build_sources_event`` seam —
  the §3.6/§7.2 sources-panel surface, with every excerpt bounded per
  the §2.1 licensing wall (see :func:`bounded_excerpt`; NO wire event
  ever carries full Tier-B text). Then the #12/#13 event vocabulary
  passed through unchanged and in order — ``text``/``citation``/
  ``usage``/``footer`` (complete answers) or a terminal ``error`` (no
  footer after an error), then the #13 ``badge``/
  ``validation_degraded`` events via the injected
  ``append_validation_events``. Refusals
  (:class:`rag.retrieval.HonestRefusal`) become one :data:`ANSWER_EVENT`
  with ``kind == "refusal"`` — zero generation calls and NO sources
  event (nothing was retrieved above threshold; nothing is dressed up
  as grounding). The sources event is server-side composition over the
  retrieval result: it changes NO provider request (replay fixtures'
  request hashes are unaffected) and adds NOTHING to the exchange log.
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
  ``paused_response_text`` — zero adapter calls of ANY kind. Issue #56
  amendment (decision FLAGGED in the #56 red-phase notes): the
  CACHED-STARTER path now logs an exchange record — route
  ``"cached_starter"``, question set to the cache entry's CANONICAL
  starter question text (one of the fixed public §7.1 questions —
  NEVER the visitor's raw typed text), the cached answer/citations,
  zero usage — and mints the meta ``exchange_id`` for it, so a cached
  answer can receive thumbs feedback while paused (valid signal about
  the cached entry). The paused NON-starter furniture path is
  unchanged: nothing logged, ``exchange_id: None`` (a paused refusal
  is furniture, not an exchange — the paused state still cannot become
  a quiet full-text query log).

- **SEMANTIC CACHE (issue #57, both modes):** when ``deps.
  semantic_cache`` is present and the request is FIRST-TURN (empty
  history), the cache is consulted after the rate limiter and BEFORE
  any adapter call. A hit is ``meta`` (current mode, ``preamble_note``
  None — no classifier ran — and a FRESH ``exchange_id``) plus exactly
  one :data:`ANSWER_EVENT` with ``kind == "cached"`` replaying the
  stored answer/citations/badges/sources/footer VERBATIM with the
  original answer's ``generated_on`` date: zero adapter calls, zero
  spend, still rate-limited. The serving is logged as its own exchange
  (route ``"cached"``, question = the SOURCE'S canonical question text
  — never the visitor's raw variant — ``cached_from`` = the source
  ``exchange_id``, empty ``usage_records``) and joined via
  ``record_serving`` so thumbs-down eviction reaches the source entry.
  In PAUSED mode the ratified decision-6 carve-out applies: an EXACT
  match of a starter question's canonical text still serves the curated
  editorial ``cached_starter`` answer (the editorial surface wins where
  it exists); the semantic cache is consulted for everything ELSE,
  before falling back to paused furniture. On the LIVE path (no starter
  cache serves there) the semantic cache is consulted first, and a clean
  completed retrieval exchange that passes
  ``semantic_cache.cacheable_exchange`` is stored after finalization.
  Contract pinned by ``tests/unit/test_service_semantic_cache.py``.

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
import itertools
import json
import logging
import uuid
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
    CITATION_EVENT,
    ERROR_EVENT,
    FOOTER_EVENT,
    GENERATION_DECLINE_MARKER,
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    TEXT_EVENT,
    USAGE_EVENT,
    CitedPassage,
    GenerationConfig,
    GroundedAnswer,
    classify_generation_decline,
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
    FEEDBACK_DOWN,
    FEEDBACK_VERDICTS,
    LOGGING_DISCLOSURE,
    ExchangeLog,
    build_exchange_record,
)
from service.rate_limit import IP_HASH_RETENTION_DAYS, RateLimiter, resolve_client_ip
from service.retention import RETENTION_PURGE_INTERVAL, run_retention_pass
from service.semantic_cache import SEMANTIC_CACHE_ROUTE, SemanticCache, cacheable_exchange
from service.starter_cache import StarterCache
from service.transparency import (
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    STEWARD_CREDIT_TEXT,
    TransparencyPages,
)

_LOGGER = logging.getLogger(__name__)

#: The single combined rewrite+classify call is a Haiku structured call
#: (rag.query); its usage is priced against this family.
CLASSIFIER_MODEL = GENERATION_MODEL_DEFAULT
#: The chart planner is a Haiku structured call (charts.planner).
PLANNER_MODEL = GENERATION_MODEL_DEFAULT

__all__ = [
    "META_EVENT",
    "ANSWER_EVENT",
    "CHART_EVENT",
    "SOURCES_EVENT",
    "SSE_EVENT_NAMES",
    "ANSWER_KIND_CANNED",
    "ANSWER_KIND_REFUSAL",
    "ANSWER_KIND_PAUSED",
    "ANSWER_KIND_CACHED_STARTER",
    "ANSWER_KIND_CACHED",
    "OPEN_EXCERPT_MAX_CHARS",
    "RESTRICTED_EXCERPT_MAX_CHARS",
    "EXCERPT_BOUNDS",
    "SOURCE_TIER_LABELS",
    "SOURCE_ENTRY_KEYS",
    "bounded_excerpt",
    "build_sources_event",
    "FEEDBACK_UNKNOWN_EXCHANGE_DETAIL",
    "ServiceStartupError",
    "ServiceDeps",
    "format_sse_event",
    "create_app",
]

#: Issue #56 — the ONE 404 detail the feedback route ever serves. A
#: constant, so a purged exchange_id and a never-issued exchange_id are
#: byte-identical refusals: the endpoint reveals nothing about whether an
#: exchange ever existed, and echoes neither the id nor any content.
FEEDBACK_UNKNOWN_EXCHANGE_DETAIL = "unknown exchange"

#: Service-level SSE vocabulary, extending the #12 (text/citation/usage/
#: footer/error) and #13 (badge/validation_degraded) events.
META_EVENT = "meta"
ANSWER_EVENT = "answer"
CHART_EVENT = "chart"
#: The #220 retrieved-passages surface: emitted once per grounded
#: exchange, after ``meta``, before the first ``text`` event.
SOURCES_EVENT = "sources"

#: The COMPLETE emit-able SSE vocabulary of the composed stream (review
#: finding #230): the #22 service names here, the #12 grounded-answer names
#: (imported from ``rag.generation`` — the producer), and the two #13
#: validator names. The UI's ``render_model.HANDLED_EVENTS | IGNORED_EVENTS``
#: is pinned SET-EQUAL to this both directions, so a service-side addition
#: fails the UI suite until the UI handles or explicitly ignores it — the
#: drift the one-directional pairwise guard used to miss. (The two validator
#: names are the strings ``rag.citation_validator`` owns; the UI's
#: set-equality guard catches a rename there, and ``service.*`` keeps its
#: single import of the validator in ``service.main``.)
SSE_EVENT_NAMES: frozenset[str] = frozenset(
    {
        META_EVENT,
        ANSWER_EVENT,
        CHART_EVENT,
        SOURCES_EVENT,
        TEXT_EVENT,
        CITATION_EVENT,
        USAGE_EVENT,
        FOOTER_EVENT,
        ERROR_EVENT,
        "badge",
        "validation_degraded",
    }
)

#: ``ANSWER_EVENT`` data ``kind`` values.
ANSWER_KIND_CANNED = "canned"
ANSWER_KIND_REFUSAL = "refusal"
ANSWER_KIND_PAUSED = "paused"
ANSWER_KIND_CACHED_STARTER = "cached_starter"
#: Issue #57: a semantic-cache hit replays a previously answered
#: grounded exchange VERBATIM as one ``answer`` event of this kind —
#: data carries the stored ``text``, ``footer``, ``citations``,
#: ``badges``, ``sources`` (all byte-identical to the original wire
#: events) and ``generated_on`` (the ORIGINAL answer's ISO date — the
#: honesty marker: a cached answer is never presented as fresh). Pinned
#: equal to ``service.semantic_cache.SEMANTIC_CACHE_ROUTE`` and the
#: UI's ``VIEW_KIND_CACHED``. No new SSE event NAME: the existing
#: ``answer`` event carries it, so ``SSE_EVENT_NAMES`` and the UI
#: handled/ignored parity are untouched.
ANSWER_KIND_CACHED = "cached"


# ---------------------------------------------------------------------------
# The #220 sources surface: the §2.1 licensing wall, pinned in pure code.
# ---------------------------------------------------------------------------

#: DECISION (flagged for ratification, #220 red notes): excerpt bounds are
#: CHARACTER counts over a verbatim prefix of the chunk body — deterministic,
#: language-agnostic, and unadapted (the Carbon Brief ND rule: excerpts are
#: displayed verbatim, never paraphrased; a mid-word cut is a display concern
#: the UI signals via ``excerpt_truncated``, never a rewording).
#:
#: ``open`` documents may carry the fuller (still bounded) excerpt.
OPEN_EXCERPT_MAX_CHARS = 600
#: Tier-B / permission-conditioned documents (``non-commercial-educational``
#: and ``permission-on-file``) carry a strictly tighter bound: a short
#: excerpt under the short-excerpt allowances, never approaching the whole
#: expressive work. DECISION flagged: 300 characters (~2 sentences).
RESTRICTED_EXCERPT_MAX_CHARS = 300

#: The licensing wall, keyed on the manifest's ``permitted_context`` (the
#: §2.1 invariant rule: enforcement keys on permitted_context, never on
#: tier labels). Keys are pinned equal to
#: ``ingestion.manifest.DOCUMENT_PERMITTED_CONTEXTS`` by the unit suite —
#: a new manifest context value fails tests until a bound is decided here.
#: Anything OUTSIDE this mapping fails CLOSED: no excerpt at all.
EXCERPT_BOUNDS: Mapping[str, int] = {
    "open": OPEN_EXCERPT_MAX_CHARS,
    "non-commercial-educational": RESTRICTED_EXCERPT_MAX_CHARS,
    "permission-on-file": RESTRICTED_EXCERPT_MAX_CHARS,
}

#: DECISION (flagged for ratification, #220 red notes): the wire's
#: ``source_tier`` is a DISPLAY label derived from ``permitted_context``
#: (open -> A, non-commercial-educational -> B, permission-on-file -> C).
#: The chunk payload's ``citation_metadata`` does not carry the manifest's
#: ``source_tier`` field today; deriving the display label from the same
#: permitted_context the wall keys on avoids an ingestion/reindex change in
#: this issue and can never disagree with the enforcement key. If the
#: manifest tier is later plumbed into the payload, this map goes away.
SOURCE_TIER_LABELS: Mapping[str, str] = {
    "open": "A",
    "non-commercial-educational": "B",
    "permission-on-file": "C",
}

#: The CLOSED per-passage wire shape of the sources event: every entry
#: carries exactly these keys, so no field can ever smuggle unbounded
#: source text past the excerpt wall.
SOURCE_ENTRY_KEYS: frozenset[str] = frozenset(
    {
        "doc_id",
        "chunk_id",
        "title",
        "attribution_text",
        "canonical_url",
        "source_type",
        "source_tier",
        "permitted_context",
        "excerpt",
        "excerpt_truncated",
    }
)


def bounded_excerpt(body: str, permitted_context: Any) -> str | None:
    """Pure licensing wall: the wire-safe excerpt of one chunk body.

    RED-phase contract stub (issue #220); the failing suite in
    ``tests/unit/test_service_sources_event.py`` pins the contract:

    - ``permitted_context`` keys :data:`EXCERPT_BOUNDS` EXACTLY (a
      string member of ``ingestion.manifest.DOCUMENT_PERMITTED_CONTEXTS``
      — case-sensitive, no normalisation): the excerpt is the verbatim
      prefix ``body[:bound]`` — equal to ``body`` when it fits, never
      padded, never paraphrased, never longer than the bound.
    - ANY other value — unknown, miscased, ``None``, missing — fails
      CLOSED: returns ``None`` (metadata-only entry, no excerpt on the
      wire). An unrecognised licensing context never leaks text.
    - Consequence pinned by the suite: NO sources event can ever carry
      full Tier-B text — a non-``open`` body longer than
      :data:`RESTRICTED_EXCERPT_MAX_CHARS` never appears whole in any
      serialised frame.
    """
    bound = EXCERPT_BOUNDS.get(permitted_context) if isinstance(permitted_context, str) else None
    if bound is None:
        return None
    return body[:bound]


def build_sources_event(retrieved: RetrievedPassages) -> dict[str, Any]:
    """Pure builder: the retrieval result -> the one #220 sources event.

    RED-phase contract stub (issue #220); the failing suite in
    ``tests/unit/test_service_sources_event.py`` pins the contract:

    - Returns ``{"event": SOURCES_EVENT, "data": {"sources": [...]}}``
      with one entry per retrieved passage, in retrieval order (best
      first — the same order the generation call's document indices use).
    - Each entry carries EXACTLY :data:`SOURCE_ENTRY_KEYS`: ``doc_id``,
      ``chunk_id``, ``title`` (the citation metadata's title, falling
      back to ``attribution_text``, falling back to ``doc_id``),
      ``attribution_text``, ``canonical_url`` (the §3.6 deep link),
      ``source_type`` (the §2.5 voices/evidence label, verbatim from the
      payload — the UI's "About the movement" separation keys on it),
      ``source_tier`` (:data:`SOURCE_TIER_LABELS` over
      ``permitted_context``; ``None`` when the context is unknown),
      ``permitted_context`` (verbatim), ``excerpt``
      (:func:`bounded_excerpt` over the chunk body — ``None`` is the
      fail-closed metadata-only state), and ``excerpt_truncated``
      (True iff a non-None excerpt is shorter than the body).
    - Pure over ``retrieved`` alone: no adapter, no manifest fetch, no
      clock — the provider request and its replay hash are untouched.
    """
    sources: list[dict[str, Any]] = []
    for passage in retrieved.passages:
        payload = passage.payload
        metadata = payload.get("citation_metadata") or {}
        doc_id = payload.get("doc_id")
        permitted_context = metadata.get("permitted_context")
        body = payload.get("body") or ""
        excerpt = bounded_excerpt(body, permitted_context)
        attribution_text = metadata.get("attribution_text")
        title = metadata.get("title") or attribution_text or doc_id
        source_tier = (
            SOURCE_TIER_LABELS.get(permitted_context)
            if isinstance(permitted_context, str)
            else None
        )
        sources.append(
            {
                "doc_id": doc_id,
                "chunk_id": passage.chunk_id,
                "title": title,
                "attribution_text": attribution_text,
                "canonical_url": metadata.get("canonical_url"),
                "source_type": payload.get("source_type"),
                "source_tier": source_tier,
                "permitted_context": permitted_context,
                "excerpt": excerpt,
                "excerpt_truncated": excerpt is not None and len(excerpt) < len(body),
            }
        )
    return {"event": SOURCES_EVENT, "data": {"sources": sources}}


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
    #: The #220 sources-surface seam: the pure builder the retrieval
    #: route calls ONCE per grounded exchange with the retrieval result;
    #: its returned event mapping is emitted verbatim (after ``meta``,
    #: before the first ``text``). ``None`` means the module default,
    #: :func:`build_sources_event`. Injected so tests pin the seam
    #: without monkeypatching.
    build_sources: Callable[[RetrievedPassages], Mapping[str, Any]] | None = None
    #: The #57 semantic response cache (``service.semantic_cache``), or
    #: ``None`` when the cache is disabled/absent — a None cache leaves
    #: every route's behaviour exactly as it is today. When present, the
    #: chat route consults it FIRST-TURN ONLY (empty history), after the
    #: rate limiter and before ANY adapter call, in BOTH modes; the
    #: /feedback route routes "down" verdicts into
    #: ``handle_thumbs_down`` after a successful 204. Contract pinned by
    #: ``tests/unit/test_service_semantic_cache.py``.
    semantic_cache: SemanticCache | None = None


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


def _rate_limit_or_429(
    request: Request, deps: ServiceDeps, config: ServiceConfig
) -> Response | None:
    """The shared per-IP rate-limit guard for /chat and /feedback (finding
    #298): resolve the client IP (honouring the trusted proxy) and consult
    the hashed-IP limiter. Returns the 429 ``Response`` — whose body echoes
    NOTHING about the client — when the request is over the window, else
    ``None`` so the route proceeds. Both routes call it FIRST, before any
    adapter call or exchange probe."""
    client_host = request.client.host if request.client else None
    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = resolve_client_ip(client_host, forwarded_for, trusted_proxy=config.trusted_proxy)
    if not deps.rate_limiter.allow(client_ip):
        return Response(
            content="rate limit exceeded — please slow down",
            status_code=429,
            media_type="text/plain",
        )
    return None


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
        run_retention_pass(deps.exchange_log, deps.rate_limiter, deps.semantic_cache)
        interval_s = RETENTION_PURGE_INTERVAL.total_seconds()

        async def _periodic_purge() -> None:
            while True:
                await asyncio.sleep(interval_s)
                # One raising pass must NOT kill the loop: a torn log, a
                # transient disk error or any purge failure would otherwise
                # end the task silently and stop enforcing both §9 bounds
                # for the process lifetime (finding #265). Log loudly and
                # let the next interval try again.
                try:
                    run_retention_pass(deps.exchange_log, deps.rate_limiter, deps.semantic_cache)
                except Exception:
                    _LOGGER.exception(
                        "a scheduled retention pass failed; the periodic "
                        "purge loop continues and will retry next interval "
                        "(finding #265)"
                    )

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

    # The four transparency surfaces (issue #19). When real pages are
    # injected (``service.main`` builds them at startup), each route serves
    # its built html; ``None`` keeps the interim pre-#19 placeholders so the
    # composed stack always serves in both modes. Never rate-limited, zero
    # adapter calls, nothing logged — a page view is not an exchange.
    @app.get("/about", response_class=HTMLResponse)
    def about() -> str:
        pages = deps.transparency
        return pages.about_html if pages is not None else _ABOUT_HTML

    @app.get("/privacy", response_class=HTMLResponse)
    def privacy() -> str:
        pages = deps.transparency
        return pages.privacy_html if pages is not None else _PRIVACY_HTML

    @app.get("/sources", response_class=HTMLResponse)
    def sources() -> str:
        pages = deps.transparency
        return pages.sources_html if pages is not None else _SOURCES_HTML

    @app.get("/voices", response_class=HTMLResponse)
    def voices() -> str:
        pages = deps.transparency
        return pages.voices_html if pages is not None else _VOICES_HTML

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
        if (limited := _rate_limit_or_429(request, deps, config)) is not None:
            return limited

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

    @app.post("/feedback")
    def feedback(payload: FeedbackRequest, request: Request) -> Response:
        """The #56 thumbs endpoint (module docstring, "Route contract").

        Rate-limit FIRST with the same hashed-IP limiter as /chat (429,
        nothing echoed); the closed verdict vocabulary is enforced (422);
        the verdict lands on the matched record via
        ``ExchangeLog.record_feedback`` (204 empty body on success, the
        constant :data:`FEEDBACK_UNKNOWN_EXCHANGE_DETAIL` 404 otherwise).
        Zero adapter calls, so it serves in both modes; nothing about the
        client is stored, joined, or logged beyond the limiter's own
        hashed count.
        """
        # Rate limiting FIRST (the shared hashed-IP limiter), with a body
        # that echoes nothing about the client or the probed exchange.
        if (limited := _rate_limit_or_429(request, deps, config)) is not None:
            return limited
        # The closed vocabulary (defence above record_feedback's own guard).
        if payload.verdict not in FEEDBACK_VERDICTS:
            raise HTTPException(status_code=422, detail="unknown verdict")
        # The join: False (unknown or purged id) becomes the uniform,
        # reveal-nothing 404; True is a 204 with an EMPTY body.
        if not deps.exchange_log.record_feedback(payload.exchange_id, payload.verdict):
            raise HTTPException(status_code=404, detail=FEEDBACK_UNKNOWN_EXCHANGE_DETAIL)
        # Issue #57: a "down" verdict landing on a cached serving OR its
        # source exchange poisons the cached answer — evict it so the repeat
        # question runs live rather than replaying the downvoted answer.
        # "Up" never evicts.
        if deps.semantic_cache is not None and payload.verdict == FEEDBACK_DOWN:
            deps.semantic_cache.handle_thumbs_down(payload.exchange_id)
        return Response(status_code=204)

    return app


class ChatRequest(BaseModel):
    """The POST /chat body: the latest question plus prior conversation turns."""

    question: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """The POST /feedback body (issue #56): the exchange join key + verdict.

    ``verdict`` must land in the closed
    :data:`service.exchange_log.FEEDBACK_VERDICTS` vocabulary — the
    route answers 422 for anything else (the red suite pins the
    behaviour, not the mechanism; ``FEEDBACK_VERDICTS`` is the source
    of truth either way).
    """

    exchange_id: str
    verdict: str


def _meta_event(
    mode: ServiceMode, preamble_note: str | None, exchange_id: str | None
) -> dict[str, Any]:
    return {
        "event": META_EVENT,
        "data": {
            "disclosure": LOGGING_DISCLOSURE,
            "preamble_note": preamble_note,
            "mode": mode.value,
            # The #56 feedback join key: the id the logged record carries
            # (a fresh hex on every logged route), or None on the paused
            # non-starter furniture path where nothing is logged. The key
            # is always present on the wire.
            "exchange_id": exchange_id,
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
    mode = deps.spend_tracker.mode()
    first_turn = not history

    # PAUSED: the fail-closed read-only state. Zero adapter calls of ANY
    # kind. The cached-starter path logs a feedback-able exchange (#56);
    # the non-starter furniture path logs nothing (a paused refusal is
    # furniture, not an exchange) and carries exchange_id: None.
    if mode is ServiceMode.PAUSED:
        entry = deps.starter_cache.lookup(question)
        # Issue #57 (ratified decision 6, with the exact-starter carve-out):
        # an EXACT match of a starter question's canonical text serves the
        # curated editorial starter answer (above); the semantic cache is
        # consulted for everything ELSE, before falling back to paused
        # furniture. A hit is $0 and serves while paused.
        if entry is None and deps.semantic_cache is not None and first_turn:
            hit = deps.semantic_cache.lookup(question)
            if hit is not None:
                yield from _cached_events(deps, ServiceMode.PAUSED, hit)
                return
        if entry is not None:
            exchange_id = uuid.uuid4().hex
            yield _meta_event(ServiceMode.PAUSED, None, exchange_id)
            citations = [dict(citation) for citation in entry.citations]
            # Log in a `finally` so a disconnect after the answer event still
            # logs the exchange (#211). The logged question is the CANONICAL
            # starter question (entry.question), NEVER the visitor's raw text.
            try:
                yield _answer_event(
                    ANSWER_KIND_CACHED_STARTER,
                    entry.answer_text,
                    generated_on=entry.generated_on,
                    footer=entry.footer,
                    citations=citations,
                )
            finally:
                _log_exchange(
                    deps,
                    question=entry.question,
                    route=ANSWER_KIND_CACHED_STARTER,
                    answer_text=entry.answer_text,
                    retrieved_chunk_ids=[],
                    citations=citations,
                    validation={},
                    usage_records=[],
                    exclude_from_harvest=False,
                    exchange_id=exchange_id,
                )
        else:
            yield _meta_event(ServiceMode.PAUSED, None, None)
            yield _answer_event(ANSWER_KIND_PAUSED, paused_response_text(today))
        return

    # LIVE. Issue #57: consult the semantic cache FIRST-TURN ONLY, before
    # ANY adapter call (the $0 replay path — no classifier, no generation).
    # A hit replays the stored grounded answer verbatim; a miss runs the
    # live pipeline unchanged.
    if deps.semantic_cache is not None and first_turn:
        hit = deps.semantic_cache.lookup(question)
        if hit is not None:
            yield from _cached_events(deps, ServiceMode.LIVE, hit)
            return

    # LIVE: classify + route (exactly one structured call). The feedback
    # join key is minted ONCE here and ridden to the client in meta, then
    # into build_exchange_record so the wire id IS the logged record's id.
    exchange_id = uuid.uuid4().hex
    decision = process_query(deps.adapter, question, history)
    record_usage_if(CLASSIFIER_MODEL, decision.classification.usage)
    yield _meta_event(ServiceMode.LIVE, decision.preamble_note, exchange_id)

    if decision.route is Route.CANNED:
        yield from _canned_events(deps, question, decision, exchange_id)
    elif decision.route is Route.CHART:
        yield from _chart_events(deps, question, decision, record_usage_if, exchange_id)
    else:
        yield from _retrieval_events(
            deps, config, question, decision, record_usage_if, exchange_id, history
        )


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
    exchange_id: str,
    cached_from: str | None = None,
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
        exchange_id=exchange_id,
        cached_from=cached_from,
    )
    deps.exchange_log.append(record)


def _cached_events(deps: ServiceDeps, mode: ServiceMode, hit: Any) -> Iterator[dict[str, Any]]:
    """Issue #57: replay one semantic-cache hit as meta + one ``cached`` answer.

    Zero adapter calls, zero spend. The answer event carries the stored
    text/footer/citations/badges/sources VERBATIM with the ORIGINAL
    answer's ``generated_on`` date (a cached answer is never presented as
    fresh). The serving logs its OWN exchange record (fresh
    ``exchange_id``, route ``cached``, ``cached_from`` = the source
    exchange, the SOURCE'S canonical question — never the visitor's raw
    variant, empty usage) and is joined via ``record_serving`` so a later
    thumbs-down on the serving evicts the source entry.
    """
    entry = hit.entry
    exchange_id = uuid.uuid4().hex
    yield _meta_event(mode, None, exchange_id)
    citations = [dict(citation) for citation in entry.citations]
    # Log + join in a `finally` so a disconnect after the answer event still
    # records the feedback-able serving (#211, matching every other route).
    try:
        yield _answer_event(
            ANSWER_KIND_CACHED,
            entry.answer_text,
            generated_on=entry.generated_on,
            footer=entry.footer,
            citations=citations,
            badges=[dict(badge) for badge in entry.badges],
            sources=[dict(source) for source in entry.sources],
        )
    finally:
        deps.semantic_cache.record_serving(exchange_id, entry.source_exchange_id)
        _log_exchange(
            deps,
            question=entry.question,
            route=SEMANTIC_CACHE_ROUTE,
            answer_text=entry.answer_text,
            retrieved_chunk_ids=[],
            citations=citations,
            validation={},
            usage_records=[],
            exclude_from_harvest=False,
            exchange_id=exchange_id,
            cached_from=entry.source_exchange_id,
        )


def _canned_events(
    deps: ServiceDeps, question: str, decision: QueryDecision, exchange_id: str
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
            exchange_id=exchange_id,
        )


def _chart_events(
    deps: ServiceDeps,
    question: str,
    decision: QueryDecision,
    record_usage_if: Callable[[str | None, Mapping[str, int] | None], None],
    exchange_id: str,
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
                exchange_id=exchange_id,
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
            exchange_id=exchange_id,
        )


def _decline_decision(accum_text: str) -> str | None:
    """Issue #313: is the accumulated generation text a structured decline?

    Returns ``"grounded"`` as soon as the first content line diverges from
    :data:`rag.generation.GENERATION_DECLINE_MARKER` (so a normal answer
    streams without buffering its whole first paragraph), ``"decline"`` once
    the completed first line IS the marker, or ``None`` while the first line
    is still an in-progress marker prefix (a marker split across transport
    deltas keeps buffering). First-line-only mirrors
    :func:`rag.generation.classify_generation_decline` — the injection guard.
    Clean completion (footer, no error) is confirmed by the caller; this
    decides the marker shape alone.
    """
    lead = accum_text.lstrip("\n")
    if "\n" in lead:
        first_line = lead.split("\n", 1)[0].strip()
        return "decline" if first_line == GENERATION_DECLINE_MARKER else "grounded"
    core = lead.strip()
    if core and not GENERATION_DECLINE_MARKER.startswith(core):
        return "grounded"
    return None


def _retrieval_events(
    deps: ServiceDeps,
    config: ServiceConfig,
    question: str,
    decision: QueryDecision,
    record_usage_if: Callable[[str | None, Mapping[str, int] | None], None],
    exchange_id: str,
    history: Sequence[Mapping[str, Any]],
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
                exchange_id=exchange_id,
            )
        return

    # Issue #220: exactly one sources event, HERE — after ``meta`` (yielded
    # by the caller) and before the first ``text`` — on grounded exchanges
    # only (the refusal path returned above). Pure server-side composition
    # over the retrieval result via the injectable seam: no provider request
    # is touched and nothing is added to the exchange log.
    build_sources = deps.build_sources or build_sources_event
    sources_event = dict(build_sources(retrieval_result))
    # The #57 cache stores the sources panel verbatim so a replay is
    # byte-identical; capture it here (already licence-bounded). The event is
    # YIELDED below inside the finalized try block: a client dropping right
    # after the sources event still commits the generation, so its charged
    # usage must reach finalization (#211) exactly as a later disconnect does.
    cache_sources = list((sources_event.get("data") or {}).get("sources", []))

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

    # Issue #313: classify a structured generation-level DECLINE before any
    # sources/text/footer/validation reaches the client. The decision is over
    # the ACCUMULATED answer text (so a marker split across transport deltas
    # still classifies) and only the FIRST line counts (a quoted or
    # passage-smuggled marker after it never flips the exchange — the
    # injection guard). A marked-but-error-terminated stream is an ERROR, not
    # a decline, so a decline is confirmed only on a cleanly-completed stream
    # (footer seen, no error). A normal answer diverges from the marker on its
    # first delta and streams on without buffering.
    head_events: list[Mapping[str, Any]] = []
    accum_text = ""
    saw_error = False
    saw_footer = False
    running_decision: str | None = None
    for event in sse_iter:
        head_events.append(event)
        name = event["event"]
        if name == TEXT_EVENT:
            accum_text += event["data"].get("text", "")
        elif name == ERROR_EVENT:
            saw_error = True
        elif name == FOOTER_EVENT:
            saw_footer = True
        if running_decision is None:
            running_decision = _decline_decision(accum_text)
        if running_decision == "grounded":
            break
        # A potential/undecided decline keeps draining to confirm completion.

    if saw_footer and not saw_error and classify_generation_decline(accum_text).is_decline:
        # A clean structured decline: ONE honest refusal answer — NO sources
        # (a refusal is never dressed up as grounding), NO factual-sentence
        # validation, NEVER admitted to the semantic cache; the generation
        # call WAS made, so its usage is metered and logged (§3.5's
        # refuse-without-spend goal belongs to the pre-filter alone now).
        decline_usage_records: list[dict[str, Any]] = []
        for event in head_events:
            if event["event"] == USAGE_EVENT:
                record_usage_if(gen_model, event["data"])
                decline_usage_records.append({"model": gen_model, "usage": event["data"]})
                break
        display_text = classify_generation_decline(accum_text).display_text
        try:
            yield _answer_event(ANSWER_KIND_REFUSAL, display_text)
        finally:
            _log_exchange(
                deps,
                question=question,
                route="retrieval",
                answer_text=display_text,
                retrieved_chunk_ids=[passage.chunk_id for passage in retrieval_result.passages],
                citations=[],
                validation={"generation_decline": True},
                usage_records=decline_usage_records,
                exclude_from_harvest=decision.exclude_from_harvest,
                exchange_id=exchange_id,
            )
        return

    # Not a decline (or a marked-but-incomplete stream, which is an error):
    # replay the peeked prefix ahead of the remaining transport events and
    # deliver the grounded exchange exactly as before.
    sse_iter = itertools.chain(head_events, sse_iter)

    outcome_holder: dict[str, Any] = {}

    def validate(transcript: Sequence[Mapping[str, Any]]) -> Any:
        answer = _grounded_answer_from_sse(transcript, retrieval_result)
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
    # #57: the footer/badges of a clean delivered answer, captured so a
    # cacheable exchange can be replayed byte-identical.
    cache_badges: list[Mapping[str, Any]] = []
    cache_footer: list[str] = []
    error_terminated = False
    usage_recorded = False
    # #57/#285: cache admission requires DELIVERY COMPLETENESS, not just a
    # clean drained transcript. The #211 finalization deliberately runs on a
    # client disconnect (StreamingResponse closes this generator; the drain
    # still completes validation over the FULL transcript), so a truncated
    # exchange looks ``validated`` with no ``error`` event. This flag is set
    # ONLY when the delivery loop exhausts normally — one flaky mobile
    # connection must never mint a permanently-served, citation-less answer.
    stream_completed = False
    events_iter = iter(deps.append_validation_events(sse_iter, validate))

    def absorb(event: Mapping[str, Any], *, delivered: bool) -> None:
        nonlocal usage_recorded, error_terminated
        name = event["event"]
        if name == "usage":
            # Meter the charged usage exactly once, the moment it is seen —
            # whether delivered to the client or drained in finalization.
            if not usage_recorded:
                record_usage_if(gen_model, event["data"])
                usage_records.append({"model": gen_model, "usage": event["data"]})
                usage_recorded = True
        elif name == "error":
            error_terminated = True
        elif delivered and name == "text":
            # answer_text logs only what the client actually received.
            delivered_text.append(event["data"].get("text", ""))
        elif delivered and name == "citation":
            citations.append(event["data"])
        elif delivered and name == FOOTER_EVENT:
            cache_footer.append(event["data"].get("text", ""))
        elif delivered and name == "badge":
            cache_badges.append(event["data"])

    try:
        # The #220 sources event opens the finalized region (see above): a
        # disconnect anywhere from here on drains + meters the generation.
        yield sources_event
        for event in events_iter:
            absorb(event, delivered=True)
            yield dict(event)
        # Reached only when every event was delivered to the client (a
        # disconnect raises GeneratorExit at a yield above and skips this).
        stream_completed = True
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
            exchange_id=exchange_id,
        )

        # Issue #57: admit only a provably clean, first-turn grounded
        # exchange into the semantic cache (fail-closed gate), storing the
        # delivered answer verbatim for a future $0 replay. The stored
        # date is the day it was answered — the honesty marker on replay.
        if (
            deps.semantic_cache is not None
            and stream_completed
            and cacheable_exchange(
                route="retrieval",
                history=history,
                validation=validation,
                error_terminated=error_terminated,
            )
        ):
            deps.semantic_cache.store(
                question=question,
                answer_text="".join(delivered_text),
                footer="".join(cache_footer),
                citations=citations,
                badges=cache_badges,
                sources=cache_sources,
                generated_on=deps.clock().date().isoformat(),
                source_exchange_id=exchange_id,
            )


# The interim transparency placeholders the un-ingested dev/compose-smoke
# stack legitimately serves until service.main builds the REAL #19 pages
# (a live deploy without published eval results refuses at boot — #249).
# Even these placeholders must not be materially dishonest: each carries
# the ADR-018 credit/non-commercial pair (adjacent), the §4.11
# non-affiliation disclaimer verbatim, and an explicit interim marker so
# the placeholder state is detectable from the outside (#249). The privacy
# placeholder interpolates the retention constant rather than hand-coding a
# figure that can silently diverge.
_PLACEHOLDER_FOOTER = (
    f"<footer><p>{STEWARD_CREDIT_TEXT} — {NONCOMMERCIAL_NOTE}</p>"
    f"<p>{NON_AFFILIATION_DISCLAIMER}</p>"
    "<p>Note: this is a pre-release placeholder page; the full transparency "
    "page is published once the service is deployed with released eval "
    "results.</p></footer>"
)

_ABOUT_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>About — Let's Talk About the Climate Emergency</title></head><body>
<h1>About this briefing</h1>
<p>An evidence-grounded climate briefing that answers only from a named,
clearly-licensed corpus. Every answer cites the source text it draws on.</p>
<p>See our <a href="/privacy">privacy notice</a>, the
<a href="/sources">source library</a>, and the
<a href="/voices">voices of the climate movement</a>.</p>
{_PLACEHOLDER_FOOTER}
</body></html>"""

_PRIVACY_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Privacy — Let's Talk About the Climate Emergency</title></head><body>
<h1>Privacy notice</h1>
<p>{LOGGING_DISCLOSURE}</p>
<p>Lawful basis: we process conversation logs under our legitimate interests
in operating and improving an anonymous public-education service. We store no
IP addresses, cookies, accounts or other identifiers alongside conversations;
hashed request counts used only for rate-limiting are held separately and for
no more than {IP_HASH_RETENTION_DAYS} days.</p>
{_PLACEHOLDER_FOOTER}
</body></html>"""

_SOURCES_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Sources — Let's Talk About the Climate Emergency</title></head><body>
<h1>Source library</h1>
<p>Every answer is grounded in this clearly-licensed corpus. Each cited
passage links back to its named source document.</p>
{_PLACEHOLDER_FOOTER}
</body></html>"""

_VOICES_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Voices — Let's Talk About the Climate Emergency</title></head><body>
<h1>Voices of the climate movement</h1>
<p>First-party testimony from the climate movement, kept structurally
separate from the assessed scientific evidence.</p>
{_PLACEHOLDER_FOOTER}
</body></html>"""
