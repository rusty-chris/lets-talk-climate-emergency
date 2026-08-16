"""Query rewrite + scope classification (DESIGN.md §3.1, issue #10) — contract stubs.

RED phase: this module defines the public contract that the failing tests in
`tests/unit/test_query_classifier.py` and `tests/unit/test_labelled_query_set.py`
exercise — the enums, dataclasses, error type and function signatures are real;
every behaviour-bearing function raises ``NotImplementedError`` naming where
the implementation belongs. The implementer replaces the stub bodies here, in
`rag/query.py`, without touching the signatures the tests pin.

Seams (IMPLEMENTATION.md §1): all model calls go through the injected
``ProviderAdapter`` (`rag/provider.py`); request construction is a pure
builder; canned unsafe responses and routing are pure over the
``Classification``. DESIGN.md §3.3 makes rewrite + classify **one** small
structured Haiku call per query, and §3.4 forbids that call from ever carrying
citations configuration — the contract tests enforce both on the builder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rag.provider import ProviderAdapter

# UK-first deployment (DESIGN.md §3.1): the self-harm canned response
# signposts Samaritans on this number, always.
SAMARITANS_PHONE = "116 123"


class ScopeClass(StrEnum):
    """The six-class scope enum (DESIGN.md §3.1)."""

    IN_SCOPE = "in_scope"
    CHART_REQUEST = "chart_request"
    VOICES = "voices"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL_IN_SCOPE = "adversarial_in_scope"
    UNSAFE = "unsafe"


class UnsafeSubtype(StrEnum):
    """Unsafe subtypes with distinct canned responses (DESIGN.md §3.1).

    ``SELF_HARM`` gets the Samaritans signposting response; ``HARASSMENT``
    (abuse directed at the bot/site) gets a polite disengage.
    """

    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"


class Route(StrEnum):
    """Where a classified query goes next (DESIGN.md §3.1 routing)."""

    RETRIEVAL = "retrieval"  # in_scope | voices | adversarial_in_scope
    CHART = "chart"  # chart_request -> chart pipeline (#16)
    CANNED = "canned"  # unsafe | out_of_scope: canned response, no LLM call


class MalformedClassifierOutputError(Exception):
    """The classifier's structured output failed schema validation.

    The defined failure path (IMPLEMENTATION.md §4.3): malformed output is
    retried once through the adapter; a second malformed response raises this
    typed error — never a bare ``KeyError``/``ValueError`` crash.
    """


@dataclass(frozen=True)
class Classification:
    """Parsed result of the combined rewrite+classify structured call."""

    scope: ScopeClass
    rewritten_query: str
    unsafe_subtype: UnsafeSubtype | None = None
    # BCP-47-ish lowercase primary language subtag of the *user's query*
    # ("en", "de", "cy", ...). Defaults to "en" when the model omits it.
    language: str = "en"


@dataclass(frozen=True)
class QueryDecision:
    """The routing decision consumed by generation (#13), charts (#16), logs (#22)."""

    route: Route
    classification: Classification
    # The rewritten query that feeds retrieval (RETRIEVAL routes only).
    retrieval_query: str | None = None
    # The rewritten request handed to the chart planner (CHART route only).
    chart_request: str | None = None
    # voices classification biases retrieval toward the voices source (§3.2).
    voices_bias: bool = False
    # adversarial_in_scope routes to normal retrieval with a tone flag.
    tone_flag: bool = False
    # Non-English input: the one-line note (in English) explaining why the
    # answer is in English (DESIGN.md §3.1 MVP rule). None for English input.
    preamble_note: str | None = None
    # CANNED routes only: the full canned response text; no LLM call is made.
    canned_response: str | None = None
    # Unsafe exchanges are excluded from eval-case harvesting (#22 consumes
    # this flag from the log record; DESIGN.md §3.1/§8).
    exclude_from_harvest: bool = False


def build_query_processing_request(
    query: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Pure builder: the payload for the single combined rewrite+classify call.

    Returns ``{"messages": ..., "schema": ..., "config": ...}`` matching
    ``ProviderAdapter.structured``. Must carry the conversation ``history``
    (so references like "there" can be resolved) and must NEVER carry a
    ``documents`` key or any citations configuration (DESIGN.md §3.4).
    """
    raise NotImplementedError(
        "issue #10: implement build_query_processing_request in rag/query.py - "
        "a pure builder for the single combined rewrite+classify structured "
        "call (DESIGN.md 3.1/3.3), never carrying documents or citations "
        "config (3.4)"
    )


def parse_classifier_output(raw: Mapping[str, Any]) -> Classification:
    """Pure: validate a structured-output dict into a ``Classification``.

    Raises ``MalformedClassifierOutputError`` (naming the offending field) on
    anything outside the schema — unknown scope value, wrong types, missing
    required keys. ``language`` defaults to "en" and ``unsafe_subtype`` to
    None when absent.
    """
    raise NotImplementedError(
        "issue #10: implement parse_classifier_output in rag/query.py - pure "
        "schema validation into Classification, raising "
        "MalformedClassifierOutputError on malformed output (IMPLEMENTATION.md 4.3)"
    )


def classify_and_rewrite(
    adapter: ProviderAdapter,
    query: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> Classification:
    """One ``adapter.structured`` call; retry once on malformed output.

    A second malformed response raises ``MalformedClassifierOutputError``
    (exactly two adapter calls, never three; IMPLEMENTATION.md §4.3).
    """
    raise NotImplementedError(
        "issue #10: implement classify_and_rewrite in rag/query.py - one "
        "structured call through the adapter with retry-once-then-typed-error "
        "on malformed output"
    )


def canned_unsafe_response(subtype: UnsafeSubtype) -> str:
    """Pure: the canned response text per unsafe subtype (DESIGN.md §3.1).

    ``SELF_HARM`` must include Samaritans signposting (``SAMARITANS_PHONE``);
    ``HARASSMENT`` is a polite disengage with no crisis signposting.
    """
    raise NotImplementedError(
        "issue #10: implement canned_unsafe_response in rag/query.py - "
        "per-subtype canned text; self-harm signposts Samaritans 116 123"
    )


def route_classification(classification: Classification) -> QueryDecision:
    """Pure routing over the classification (DESIGN.md §3.1) — no adapter.

    in_scope -> RETRIEVAL; voices -> RETRIEVAL + voices_bias;
    adversarial_in_scope -> RETRIEVAL + tone_flag; chart_request -> CHART;
    out_of_scope -> CANNED polite redirect; unsafe -> CANNED per-subtype
    response + exclude_from_harvest. Non-English language sets the one-line
    ``preamble_note``.
    """
    raise NotImplementedError(
        "issue #10: implement route_classification in rag/query.py - pure "
        "routing over Classification per DESIGN.md 3.1"
    )


def process_query(
    adapter: ProviderAdapter,
    query: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> QueryDecision:
    """The query-processing entry point: classify_and_rewrite, then route.

    Exactly one ``structured`` adapter call per query (DESIGN.md §3.3);
    NEVER a ``generate`` or ``plan_chart`` call from this layer — unsafe and
    out-of-scope inputs get canned responses with no LLM generation call.
    """
    raise NotImplementedError(
        "issue #10: implement process_query in rag/query.py - "
        "classify_and_rewrite then route_classification; one structured call, "
        "zero generate/plan_chart calls"
    )
