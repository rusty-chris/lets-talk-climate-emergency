"""Query rewrite + scope classification (DESIGN.md §3.1, issue #10).

`process_query` is the entry point: one combined structured call
(`classify_and_rewrite`) rewrites the user's latest message into a
standalone query and classifies its scope into the six-class enum, then pure
routing (`route_classification`) decides what happens next — retrieval,
the chart pipeline, or a canned response with no LLM generation call at all.
Behaviour is pinned by `tests/unit/test_query_classifier.py` and
`tests/unit/test_labelled_query_set.py`.

Seams (IMPLEMENTATION.md §1): all model calls go through the injected
``ProviderAdapter`` (`rag/provider.py`); request construction
(`build_query_processing_request`) is a pure builder; canned unsafe
responses and routing are pure over the ``Classification``. DESIGN.md §3.3
makes rewrite + classify **one** small structured Haiku call per query, and
§3.4 forbids that call from ever carrying citations configuration — the
contract tests enforce both on the builder.
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

    ``exclude_from_harvest`` is the fail-safe for the failure path (finding
    #86): when the malformed output *suspected* unsafe content (e.g.
    ``scope: unsafe`` with no subtype), the flag is True so a service layer
    logging failed exchanges (#22) still honours DESIGN.md §3.1/§8 — unsafe-
    suspected content is never harvested, even when classification failed.
    """

    def __init__(self, message: str, *, exclude_from_harvest: bool = False) -> None:
        super().__init__(message)
        self.exclude_from_harvest = exclude_from_harvest


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


_PROCESSING_MODEL = "claude-haiku-4-5"
_PROCESSING_MAX_TOKENS = 256

# The instructions steering the single combined structured call. They travel
# as an ordinary message (the adapter surface has no separate system-prompt
# parameter, IMPLEMENTATION.md §1) so the seam stays a plain dict builder,
# testable without any client-shape assumptions.
_PROCESSING_INSTRUCTIONS = (
    "You are the query-processing stage of a climate-evidence chatbot. Given "
    "the conversation so far and the user's latest message, respond with one "
    "JSON object that does two things at once: (1) rewrite the latest "
    "message into a standalone query that resolves pronouns/references "
    "against the conversation and expands acronyms, keeping the user's own "
    "wording and language otherwise unchanged; (2) classify the message's "
    "scope as exactly one of: in_scope, chart_request, voices, "
    "out_of_scope, adversarial_in_scope, unsafe. Use chart_request only for "
    "an explicit request to plot/chart/graph data. Use voices for questions "
    "about the climate movement's own testimony, not scientific evidence. "
    "Use adversarial_in_scope for denialist-framed but evidence-answerable "
    "questions. Use unsafe for self-harm or harassment content, and set "
    "unsafe_subtype to 'self_harm' or 'harassment' accordingly. Always set "
    "language to the lowercase primary language subtag of the user's latest "
    "message (e.g. 'en', 'de', 'cy'), defaulting to 'en'."
)


def _processing_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scope": {"type": "string", "enum": [c.value for c in ScopeClass]},
            "rewritten_query": {"type": "string"},
            "unsafe_subtype": {"type": "string", "enum": [s.value for s in UnsafeSubtype]},
            "language": {"type": "string"},
        },
        "required": ["scope", "rewritten_query"],
        "additionalProperties": False,
        # Finding #86: scope=unsafe REQUIRES unsafe_subtype (the subtype
        # selects the canned response, DESIGN.md §3.1), so the constrained
        # decoder cannot legally emit the unroutable malformation at all.
        # parse_classifier_output enforces the same rule on whatever comes
        # back — a schema is steering, not validation.
        "anyOf": [
            {
                "properties": {"scope": {"const": ScopeClass.UNSAFE.value}},
                "required": ["unsafe_subtype"],
            },
            {
                "properties": {
                    "scope": {"enum": [c.value for c in ScopeClass if c is not ScopeClass.UNSAFE]}
                }
            },
        ],
    }


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
    messages: list[dict[str, Any]] = [{"role": "system", "content": _PROCESSING_INSTRUCTIONS}]
    messages.extend({"role": turn["role"], "content": turn["content"]} for turn in history)
    messages.append({"role": "user", "content": query})
    return {
        "messages": messages,
        "schema": _processing_schema(),
        "config": {"model": _PROCESSING_MODEL, "max_tokens": _PROCESSING_MAX_TOKENS},
    }


def parse_classifier_output(raw: Mapping[str, Any]) -> Classification:
    """Pure: validate a structured-output dict into a ``Classification``.

    Raises ``MalformedClassifierOutputError`` (naming the offending field) on
    anything outside the schema — unknown scope value, wrong types, missing
    required keys. ``language`` defaults to "en" and ``unsafe_subtype`` to
    None when absent.
    """
    if "scope" not in raw:
        raise MalformedClassifierOutputError("classifier output missing required field 'scope'")
    scope_raw = raw["scope"]
    try:
        scope = ScopeClass(scope_raw)
    except ValueError:
        raise MalformedClassifierOutputError(
            f"classifier output field 'scope' has invalid value {scope_raw!r}; "
            f"expected one of {[c.value for c in ScopeClass]}"
        ) from None

    if "rewritten_query" not in raw:
        raise MalformedClassifierOutputError(
            "classifier output missing required field 'rewritten_query'"
        )
    rewritten_query = raw["rewritten_query"]
    if not isinstance(rewritten_query, str):
        raise MalformedClassifierOutputError(
            "classifier output field 'rewritten_query' must be a string, got "
            f"{type(rewritten_query).__name__}"
        )

    unsafe_subtype: UnsafeSubtype | None = None
    unsafe_subtype_raw = raw.get("unsafe_subtype")
    if unsafe_subtype_raw is not None:
        try:
            unsafe_subtype = UnsafeSubtype(unsafe_subtype_raw)
        except ValueError:
            raise MalformedClassifierOutputError(
                f"classifier output field 'unsafe_subtype' has invalid value "
                f"{unsafe_subtype_raw!r}; expected one of "
                f"{[s.value for s in UnsafeSubtype]}",
                # The output SUSPECTED unsafe content even though the subtype
                # is unusable — fail-safe the harvest exclusion (finding #86).
                exclude_from_harvest=scope is ScopeClass.UNSAFE,
            ) from None

    # Finding #86: the subtype-required-when-unsafe rule must fail at PARSE so
    # classify_and_rewrite's retry-once covers it (a routing-stage failure
    # would fire only after the retry budget is gone). route_classification
    # keeps its own check as defence-in-depth.
    if scope is ScopeClass.UNSAFE and unsafe_subtype is None:
        raise MalformedClassifierOutputError(
            "classifier output field 'unsafe_subtype' is required when 'scope' "
            "is 'unsafe' (it selects the canned response, DESIGN.md 3.1)",
            exclude_from_harvest=True,
        )

    language = raw.get("language", "en")
    if not isinstance(language, str) or not language:
        raise MalformedClassifierOutputError(
            f"classifier output field 'language' must be a non-empty string, got {language!r}"
        )

    return Classification(
        scope=scope,
        rewritten_query=rewritten_query,
        unsafe_subtype=unsafe_subtype,
        language=language,
    )


def classify_and_rewrite(
    adapter: ProviderAdapter,
    query: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> Classification:
    """One ``adapter.structured`` call; retry once on malformed output.

    A second malformed response raises ``MalformedClassifierOutputError``
    (exactly two adapter calls, never three; IMPLEMENTATION.md §4.3). The
    raised error's ``exclude_from_harvest`` is sticky across both attempts
    (finding #86): if EITHER malformed output suspected unsafe content, the
    failure record stays excluded from eval harvesting.
    """
    request = build_query_processing_request(query, history)
    raw = adapter.structured(**request)
    try:
        return parse_classifier_output(raw)
    except MalformedClassifierOutputError as first_error:
        retry_raw = adapter.structured(**request)
        try:
            return parse_classifier_output(retry_raw)
        except MalformedClassifierOutputError as retry_error:
            retry_error.exclude_from_harvest = (
                retry_error.exclude_from_harvest or first_error.exclude_from_harvest
            )
            raise


_SELF_HARM_CANNED_RESPONSE = (
    "I'm really sorry you're feeling like this. You don't have to go through "
    "it alone — please reach out to Samaritans, free, any time, day or "
    "night, on 116 123 (UK), or visit samaritans.org. If you're in "
    "immediate danger, please contact emergency services. I'm a climate-"
    "evidence assistant and I'm not able to continue this conversation, but "
    "support is available right now."
)

_HARASSMENT_CANNED_RESPONSE = (
    "I'm not able to continue this conversation in this tone. I'm here to "
    "help with questions about climate science and the evidence behind it "
    "— I'm happy to help if you'd like to ask one."
)


def canned_unsafe_response(subtype: UnsafeSubtype) -> str:
    """Pure: the canned response text per unsafe subtype (DESIGN.md §3.1).

    ``SELF_HARM`` must include Samaritans signposting (``SAMARITANS_PHONE``);
    ``HARASSMENT`` is a polite disengage with no crisis signposting.
    """
    if subtype is UnsafeSubtype.SELF_HARM:
        return _SELF_HARM_CANNED_RESPONSE
    if subtype is UnsafeSubtype.HARASSMENT:
        return _HARASSMENT_CANNED_RESPONSE
    raise ValueError(f"no canned response defined for unsafe subtype {subtype!r}")


_OUT_OF_SCOPE_CANNED_RESPONSE = (
    "That's outside what I can help with — I only answer questions about "
    "climate science and the evidence behind it, drawn from this site's "
    "sourced corpus. Try asking about climate trends, impacts, or the "
    "assessed science instead."
)


def _english_answer_note(language: str) -> str:
    """The one-line note explaining a non-English query is answered in English."""
    return (
        f'Note: your message looked like it was written in "{language}", so '
        "I've answered in English, the only language this assistant "
        "currently supports."
    )


def route_classification(classification: Classification) -> QueryDecision:
    """Pure routing over the classification (DESIGN.md §3.1) — no adapter.

    in_scope -> RETRIEVAL; voices -> RETRIEVAL + voices_bias;
    adversarial_in_scope -> RETRIEVAL + tone_flag; chart_request -> CHART;
    out_of_scope -> CANNED polite redirect; unsafe -> CANNED per-subtype
    response + exclude_from_harvest. Non-English language sets the one-line
    ``preamble_note``.
    """
    preamble_note = (
        None if classification.language == "en" else _english_answer_note(classification.language)
    )
    rewritten = classification.rewritten_query
    scope = classification.scope

    if scope is ScopeClass.CHART_REQUEST:
        return QueryDecision(
            route=Route.CHART,
            classification=classification,
            chart_request=rewritten,
            preamble_note=preamble_note,
        )

    if scope is ScopeClass.OUT_OF_SCOPE:
        return QueryDecision(
            route=Route.CANNED,
            classification=classification,
            canned_response=_OUT_OF_SCOPE_CANNED_RESPONSE,
            preamble_note=preamble_note,
        )

    if scope is ScopeClass.UNSAFE:
        if classification.unsafe_subtype is None:
            raise MalformedClassifierOutputError(
                "classifier output field 'unsafe_subtype' is required when 'scope' is 'unsafe'"
            )
        return QueryDecision(
            route=Route.CANNED,
            classification=classification,
            canned_response=canned_unsafe_response(classification.unsafe_subtype),
            exclude_from_harvest=True,
            preamble_note=preamble_note,
        )

    if scope is ScopeClass.VOICES:
        return QueryDecision(
            route=Route.RETRIEVAL,
            classification=classification,
            retrieval_query=rewritten,
            voices_bias=True,
            preamble_note=preamble_note,
        )

    if scope is ScopeClass.ADVERSARIAL_IN_SCOPE:
        return QueryDecision(
            route=Route.RETRIEVAL,
            classification=classification,
            retrieval_query=rewritten,
            tone_flag=True,
            preamble_note=preamble_note,
        )

    # ScopeClass.IN_SCOPE
    return QueryDecision(
        route=Route.RETRIEVAL,
        classification=classification,
        retrieval_query=rewritten,
        preamble_note=preamble_note,
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
    classification = classify_and_rewrite(adapter, query, history)
    return route_classification(classification)
