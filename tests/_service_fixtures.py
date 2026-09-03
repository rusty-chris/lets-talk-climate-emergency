"""Shared builders for the issue-22 service red suite.

Everything here is synthetic (the Aurelian-Basin fixture universe of
#24/#9/#11/#12). The app-level tests compose the REAL service modules
(SpendTracker, RateLimiter, ExchangeLog, StarterCache, ChartSpecStore)
with fakes ONLY at the seams IMPLEMENTATION.md defines: the provider
adapter (FakeAdapter), retrieval, the chart planner/renderer bindings,
and the in-flight #13 validation seam (faked shape-for-shape against the
``rag/citation_validator.py`` stub contract on ``issue-13-citation-
validator``, so this suite never imports the unmerged module).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from charts.planner import PlannedChart
from charts.render import ChartArtifact
from charts.spec import spec_hash
from rag.provider import FakeAdapter
from service.app import ServiceDeps, create_app
from service.budget import SpendTracker
from service.chart_store import ChartSpecStore
from service.config import ServiceConfig
from service.exchange_log import ExchangeLog
from service.rate_limit import RateLimiter, RotatingSaltProvider
from service.starter_cache import (
    STARTER_CACHE_FILENAME,
    STARTER_QUESTIONS,
    load_starter_cache,
)
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    make_retrieved,
    transport_stream_events,
)

#: A fixed aware-UTC "now" the frozen clock starts from.
T0 = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)

CORPUS_VERSION = "corpus-2026-08-01"
SITE_URL = "https://climate-chat.example.test"


class FrozenClock:
    """An injectable clock: returns ``now`` until advanced/set by the test."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: Any) -> None:
        self.now = self.now + delta


def make_config(**overrides: Any) -> ServiceConfig:
    """A valid ServiceConfig for app tests; overrides applied on top."""
    values: dict[str, Any] = dict(
        daily_budget_usd=1.0,
        opus_subcap_usd=0.25,
        corpus_version=CORPUS_VERSION,
        corpus_vintage=CORPUS_VINTAGE,
        site_url=SITE_URL,
        qdrant_url="http://localhost:6333",
        starter_cache_dir="",  # tests point this at tmp_path
        log_dir="",
        rate_limit_per_minute=1000,  # generous: rate-limit tests tighten it
    )
    values.update(overrides)
    return ServiceConfig(**values)


def classifier_output(
    scope: str = "in_scope",
    rewritten: str = "synthetic rewritten query",
    *,
    unsafe_subtype: str | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """One structured-call result for the FakeAdapter's classifier queue."""
    output: dict[str, Any] = {
        "scope": scope,
        "rewritten_query": rewritten,
        "language": language,
    }
    if unsafe_subtype is not None:
        output["unsafe_subtype"] = unsafe_subtype
    return output


# ---------------------------------------------------------------------------
# The #13 validation seam, faked shape-for-shape against the red-phase
# contract stubs on origin/issue-13-citation-validator.
# ---------------------------------------------------------------------------


@dataclass
class FakeValidationOutcome:
    """Duck-typed stand-in for #13's ValidationOutcome (the seam's shape)."""

    validated: bool = True
    support_rate: float | None = 0.75
    usage: Mapping[str, int] | None = None
    model: str | None = "claude-haiku-4-5"
    skipped_reason: str | None = None
    degraded_reason: str | None = None


@dataclass
class FakeValidationSeam:
    """The three injected #13 callables, with call recording.

    ``append_validation_events`` mirrors the stub contract: every source
    event passes through unchanged and immediately; ``validate`` runs
    only after the source is exhausted, and ONLY when no ``error`` event
    terminated the stream; then the badge events (programmed on
    ``badge_events``) are appended.
    """

    outcome: FakeValidationOutcome = field(default_factory=FakeValidationOutcome)
    badge_events: tuple[dict[str, Any], ...] = ()
    validate_calls: list[list[Mapping[str, Any]]] = field(default_factory=list)
    record_calls: list[Any] = field(default_factory=list)

    def validate_exchange(self, result: Any, sse_events: Sequence[Mapping[str, Any]]) -> Any:
        self.validate_calls.append(list(sse_events))
        return self.outcome

    def append_validation_events(
        self,
        sse_events: Iterable[Mapping[str, Any]],
        validate: Callable[[Sequence[Mapping[str, Any]]], Any],
    ) -> Iterator[dict[str, Any]]:
        transcript: list[Mapping[str, Any]] = []
        errored = False
        for event in sse_events:
            transcript.append(event)
            if event.get("event") == "error":
                errored = True
            yield dict(event)
        if errored:
            return
        validate(transcript)
        yield from (dict(event) for event in self.badge_events)

    def exchange_log_record(self, outcome: Any) -> Mapping[str, Any]:
        self.record_calls.append(outcome)
        return {
            "citation_support_rate": getattr(outcome, "support_rate", None),
            "factual_sentence_count": 2,
            "unverified_sentence_indices": [],
            "validated": getattr(outcome, "validated", False),
            "skipped_reason": getattr(outcome, "skipped_reason", None),
            "degraded_reason": getattr(outcome, "degraded_reason", None),
            "model": getattr(outcome, "model", None),
            "usage": getattr(outcome, "usage", None),
            "cost_usd": None,
        }


# ---------------------------------------------------------------------------
# Chart-route fakes (planner/renderer bindings are deps; the units behind
# them have their own merged suites).
# ---------------------------------------------------------------------------

#: A tiny synthetic spec; content is irrelevant to the unit tier (the
#: fake renderer never validates it) but it hashes canonically.
SYNTHETIC_SPEC: dict[str, Any] = {
    "spec_version": "1.0",
    "chart_type": "line",
    "title": "Synthetic basin anomaly (invented)",
    "series": [{"dataset_id": "syn_annual_anomaly"}],
}


def make_artifact(spec: Mapping[str, Any]) -> ChartArtifact:
    """A canned ChartArtifact for the fake renderer."""
    return ChartArtifact(
        spec_hash=spec_hash(spec),
        vega_lite={"description": "synthetic vega-lite"},
        vega_lite_text='{"description": "synthetic vega-lite"}',
        alt_text="Synthetic line chart of an invented basin anomaly.",
        csv_text="# Synthetic attribution header\nyear,value\n1990,0.1\n",
    )


@dataclass
class FakeRenderer:
    """Recording fake for the bound ``render_chart`` dep."""

    calls: list[Mapping[str, Any]] = field(default_factory=list)

    def __call__(self, spec: Mapping[str, Any]) -> ChartArtifact:
        self.calls.append(dict(spec))
        return make_artifact(spec)


@dataclass
class FakePlanner:
    """Recording fake for the bound ``plan_chart`` dep."""

    result: Any = None
    calls: list[str] = field(default_factory=list)

    def __call__(self, chart_request: str) -> Any:
        self.calls.append(chart_request)
        if self.result is None:
            return PlannedChart(spec=dict(SYNTHETIC_SPEC), usage={"input_tokens": 200})
        return self.result


@dataclass
class FakeRetrieve:
    """Recording fake for the bound ``retrieve`` dep."""

    result: Any = None
    calls: list[Any] = field(default_factory=list)

    def __call__(self, decision: Any) -> Any:
        self.calls.append(decision)
        return self.result if self.result is not None else make_retrieved()


# ---------------------------------------------------------------------------
# Starter cache on disk
# ---------------------------------------------------------------------------

STARTER_GENERATED_ON = "2026-08-15"


def starter_cache_payload(**overrides: Any) -> dict[str, Any]:
    """A complete, valid starter_answers.json payload (every §7.1 question)."""
    entries = [
        {
            "question": question,
            "answer_text": f"Cached synthetic starter answer #{i} (invented).",
            "citations": [
                {
                    "chunk_id": f"syn-starter::c{i:04d}",
                    "attribution_text": "Meridian Climate Assessment Cycle 3 (invented)",
                    "cited_text": "an invented cited sentence",
                }
            ],
            "footer": f"Synthetic footer. Answers reflect sources as of {CORPUS_VINTAGE}.",
            "chart_spec_hash": None,
        }
        for i, question in enumerate(STARTER_QUESTIONS)
    ]
    payload: dict[str, Any] = {"generated_on": STARTER_GENERATED_ON, "entries": entries}
    payload.update(overrides)
    return payload


def write_starter_cache(cache_dir: Path, payload: Mapping[str, Any] | None = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / STARTER_CACHE_FILENAME
    path.write_text(
        json.dumps(payload if payload is not None else starter_cache_payload()),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Composition-root (service.main) deploy environments — issues #214/#215/#216
# ---------------------------------------------------------------------------


def full_deploy_env(tmp_path: Path) -> dict[str, str]:
    """A complete critical ``CLIMATE_CHAT_*`` environment over tmp dirs.

    Every ``CRITICAL_ENV_VARS`` entry is present and valid (a real
    starter cache is written; the log dir exists); the OPTIONAL artifact
    variables (threshold / dataset manifest / chart pack / chart store)
    are deliberately NOT included — the startup-validation tests add
    exactly the ones each scenario needs.
    """
    from service import config as service_config

    cache_dir = tmp_path / "deploy-starter-cache"
    write_starter_cache(cache_dir)
    log_dir = tmp_path / "deploy-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        service_config.ENV_DAILY_BUDGET_USD: "1.00",
        service_config.ENV_OPUS_SUBCAP_USD: "0.25",
        service_config.ENV_CORPUS_VERSION: CORPUS_VERSION,
        service_config.ENV_CORPUS_VINTAGE: CORPUS_VINTAGE,
        service_config.ENV_SITE_URL: SITE_URL,
        service_config.ENV_QDRANT_URL: "http://localhost:6333",
        service_config.ENV_STARTER_CACHE_DIR: str(cache_dir),
        service_config.ENV_LOG_DIR: str(log_dir),
        service_config.ENV_ANTHROPIC_API_KEY: "synthetic-test-key-not-real",
    }


def apply_deploy_env(monkeypatch: Any, env: Mapping[str, str]) -> None:
    """Reset the process env to EXACTLY ``env``'s deploy variables.

    Clears every ``CLIMATE_CHAT_*`` variable plus ``ANTHROPIC_API_KEY``
    first, so a developer's real environment can never leak into a
    composition-root test, then applies ``env``.
    """
    import os

    for name in list(os.environ):
        if name.startswith("CLIMATE_CHAT_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


# ---------------------------------------------------------------------------
# Deps + app assembly
# ---------------------------------------------------------------------------


@dataclass
class ServiceHarness:
    """One assembled app plus every seam the tests assert against."""

    app: Any
    config: ServiceConfig
    deps: ServiceDeps
    adapter: FakeAdapter
    clock: FrozenClock
    tracker: SpendTracker
    limiter: RateLimiter
    exchange_log: ExchangeLog
    spec_store: ChartSpecStore
    validation: FakeValidationSeam
    retrieve: FakeRetrieve
    planner: FakePlanner
    renderer: FakeRenderer
    #: The #57 semantic-cache seam handed into ServiceDeps (None = the
    #: pre-#57 behaviour — every existing suite runs cache-less).
    semantic_cache: Any = None


def make_harness(
    tmp_path: Path,
    *,
    adapter: FakeAdapter | None = None,
    config: ServiceConfig | None = None,
    clock: FrozenClock | None = None,
    tracker: SpendTracker | None = None,
    limiter: RateLimiter | None = None,
    retrieve: FakeRetrieve | None = None,
    planner: FakePlanner | None = None,
    validation: FakeValidationSeam | None = None,
    index_version: str | None = CORPUS_VERSION,
    transparency: Any = None,
    semantic_cache: Any = None,
) -> ServiceHarness:
    """Assemble a real-modules-fake-seams app for TestClient suites."""
    clock = clock or FrozenClock()
    adapter = adapter or FakeAdapter()
    cache_dir = tmp_path / "starter-cache"
    write_starter_cache(cache_dir)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config = config or make_config(starter_cache_dir=str(cache_dir), log_dir=str(log_dir))
    tracker = tracker or SpendTracker(
        daily_budget_usd=config.daily_budget_usd,
        opus_subcap_usd=config.opus_subcap_usd,
        clock=clock,
    )
    limiter = limiter or RateLimiter(
        clock=clock,
        salts=RotatingSaltProvider(clock),
        max_requests=config.rate_limit_per_minute,
    )
    exchange_log = ExchangeLog(log_dir / "exchanges.jsonl", clock=clock)
    spec_store = ChartSpecStore(tmp_path / "chart-specs")
    validation = validation or FakeValidationSeam()
    retrieve = retrieve or FakeRetrieve()
    planner = planner or FakePlanner()
    renderer = FakeRenderer()
    deps = ServiceDeps(
        adapter=adapter,
        retrieve=retrieve,
        plan_chart=planner,
        render_chart=renderer,
        validate_exchange=validation.validate_exchange,
        append_validation_events=validation.append_validation_events,
        exchange_log_record=validation.exchange_log_record,
        spend_tracker=tracker,
        rate_limiter=limiter,
        exchange_log=exchange_log,
        starter_cache=load_starter_cache(cache_dir),
        chart_spec_store=spec_store,
        index_corpus_version=lambda: index_version,
        clock=clock,
        # The #19 transparency seam: None serves the pre-#19 placeholders.
        transparency=transparency,
        # The #57 semantic-cache seam: None keeps every route exactly as
        # it is without the cache.
        semantic_cache=semantic_cache,
    )
    app = create_app(config, deps)
    return ServiceHarness(
        app=app,
        config=config,
        deps=deps,
        adapter=adapter,
        clock=clock,
        tracker=tracker,
        limiter=limiter,
        exchange_log=exchange_log,
        spec_store=spec_store,
        validation=validation,
        retrieve=retrieve,
        planner=planner,
        renderer=renderer,
        semantic_cache=semantic_cache,
    )


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse SSE wire text into [{"event": name, "data": mapping}, ...]."""
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        assert name is not None, f"SSE block without an event name: {block!r}"
        events.append({"event": name, "data": json.loads("\n".join(data_lines) or "null")})
    return events


def post_chat(client: Any, question: str, **json_extra: Any) -> list[dict[str, Any]]:
    """POST /chat and return the parsed SSE events (asserts 200 + SSE type)."""
    response = client.post("/chat", json={"question": question, **json_extra})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), (
        f"chat must answer as SSE, got {response.headers['content-type']!r}"
    )
    return parse_sse(response.text)


def events_named(events: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    return [dict(event) for event in events if event["event"] == name]


def usage_cost(model: str, usage: Mapping[str, int]) -> float:
    """The expected USD cost of one seam usage mapping (single source)."""
    from evals.pricing import estimate_cost_usd

    return estimate_cost_usd(
        model,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
    )


def stream_usage() -> dict[str, int]:
    """The usage mapping inside ``transport_stream_events()``'s message_delta."""
    for event in transport_stream_events():
        if event.get("type") == "message_delta":
            return dict(event["usage"])
    raise AssertionError("transport_stream_events lost its usage event")
