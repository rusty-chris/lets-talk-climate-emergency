"""Provider adapter seam — the deterministic boundary around all LLM calls.

IMPLEMENTATION.md §4 / issue #24. Every LLM call in the system goes through
the `ProviderAdapter` protocol; nothing in the unit or integration test tiers
may touch the network. Three implementations are planned (IMPLEMENTATION.md
§1): `AnthropicAdapter` (live, built when the first consumer issue lands),
`FakeAdapter` (programmable, records every call) and `ReplayAdapter`
(checked-in recorded fixtures keyed by canonical request hash). Recording is
env-flag-gated via `RecordingAdapter`.

The protocol mirrors DESIGN.md §3.3–3.4/§5: cited generation
(`generate`) is a separate call from structured output (`structured`)
because Anthropic native citations are incompatible with
structured-output configuration — the §3.4 contract tests (issues #10/#13)
enforce that on the request builders.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Citation:
    """One citation attached to answer text (shape per Anthropic native citations)."""

    cited_text: str
    document_index: int
    document_title: str | None = None
    start_block_index: int | None = None
    end_block_index: int | None = None


@dataclass(frozen=True)
class AnswerWithCitations:
    """Return type of `ProviderAdapter.generate` (DESIGN.md §5).

    `usage` carries the transport's token accounting (input/output/cache
    tokens) when known — the §9 cost model and #21/#22 cost accounting read
    it from live and replayed responses alike (finding #64). None when the
    producer (e.g. a test's programmed fake) has no usage to report.
    """

    text: str
    citations: tuple[Citation, ...] = ()
    usage: Mapping[str, int] | None = None


class StructuredResult(Mapping[str, Any]):
    """Return type of `ProviderAdapter.structured` (finding #92).

    A Mapping view over the parsed structured output ``value`` — consumers
    index it exactly like the plain dict it replaces — that also carries
    ``usage``, mirroring ``AnswerWithCitations.usage`` (finding #64), so
    #21/#22 spend accounting can observe structured-call token usage. None
    when the producer (a test's programmed fake, a pre-#92 fixture) has no
    usage to report.

    Equality: equal to any Mapping with the same items (usage is metadata,
    not output); between two StructuredResults, usage must match too.
    """

    __slots__ = ("value", "usage")

    def __init__(
        self,
        value: Mapping[str, Any],
        usage: Mapping[str, int] | None = None,
    ) -> None:
        self.value: dict[str, Any] = dict(value)
        self.usage = usage

    def __getitem__(self, key: str) -> Any:
        return self.value[key]

    def __iter__(self):
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StructuredResult):
            return self.value == other.value and self.usage == other.usage
        if isinstance(other, Mapping):
            return self.value == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"StructuredResult(value={self.value!r}, usage={self.usage!r})"


def filter_int_usage(usage: Any) -> dict[str, int]:
    """The single canonical token-usage projection (finding #92).

    Keep only genuine integer usage fields, dropping bools (``bool`` is an
    ``int`` subclass that would otherwise poison a token count) and any
    non-integer value a transport might carry. Non-mapping input yields an
    empty dict. This is the shared primitive every usage merge/sum in the
    stack is built from, so the int/bool filter is defined exactly once.
    """
    if not isinstance(usage, Mapping):
        return {}
    return {
        key: value
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def merge_usage(*usages: Mapping[str, int] | None) -> dict[str, int] | None:
    """Sum token-usage mappings key-wise (finding #92); ``None`` is the identity.

    The one shared helper behind every retried-call usage total (the #10
    classifier retry, the #16 citation-validator retry, the chart planner
    retry): a retry's spend is always the sum of every charged attempt,
    never just the last one. Non-integer values are filtered
    (:func:`filter_int_usage`) — a strict robustness upgrade over the older
    unfiltered copies. Returns ``None`` when nothing summable was supplied.
    """
    total: dict[str, int] = {}
    for usage in usages:
        for key, value in filter_int_usage(usage).items():
            total[key] = total.get(key, 0) + value
    return total or None


@dataclass(frozen=True)
class RawProviderResponse:
    """An unparsed provider response: the raw API payload plus the streamed

    event sequence that produced it (finding #64). This is the typed vehicle
    for transport-level recordings — the future `AnthropicAdapter`'s parsing
    of citation deltas / streaming events into `AnswerWithCitations` is
    regression-pinned by replaying these through its parser (#13), using the
    same fixture machinery as the seam-level recordings. The folded typed
    methods (`generate`, `structured`) never return this type;
    `validate_response` rejects it at those seams. The streaming seam
    (`generate_stream`, finding #183) is the one consumer: a raw fixture's
    `events` replay through it in recorded order.
    """

    payload: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RecordedCall:
    """One adapter call as recorded by `FakeAdapter`: method name + full payload."""

    method: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class ProviderAdapter(Protocol):
    """The LLM seam (DESIGN.md §5, IMPLEMENTATION.md §1)."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        system: str | Sequence[Mapping[str, Any]] | None = None,
    ) -> AnswerWithCitations:
        """Grounded generation with native citations (DESIGN.md §3.3).

        Contract (finding #91, extended to generation by issue #12):
        ``system`` is the dedicated top-level channel for the system
        prompt and maps 1:1 onto the Anthropic Messages API's top-level
        ``system`` parameter — ``messages`` never carries a
        ``role: "system"`` entry (``validate_request`` enforces the ban
        at every adapter). Generation passes a SEQUENCE of text blocks
        rather than a bare string so the static-prefix-first ordering
        (the prompt-caching contract, issue #12) survives to the
        transport, where the AnthropicAdapter places the cache
        breakpoint on the last static block. ``system=None`` is omitted
        from recorded payloads, keeping pre-existing recorded request
        hashes valid — exactly the #91 rule for ``structured``.
        """
        ...

    def generate_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        system: str | Sequence[Mapping[str, Any]] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Grounded generation as a validated transport event stream (finding #183).

        The streaming twin of :meth:`generate` — same request payload,
        same seam validation (``validate_request`` runs BEFORE any event
        is served or any transport is touched), but the response is the
        provider's raw streaming event sequence (``message_start`` /
        ``content_block_delta`` / ``message_delta`` / ``message_stop`` /
        ``error``, each as a plain mapping) instead of the folded
        :class:`AnswerWithCitations`. ``rag.generation`` translates this
        vocabulary to the service's SSE events; #22 consumes it for the
        live streaming surface. ``generate`` is the folded convenience
        over exactly this stream, so the two paths cannot drift.
        """
        ...

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
        system: str | None = None,
    ) -> StructuredResult:
        """Structured-output call (rewriter, classifier, judges — never citations).

        Returns a ``StructuredResult``: a Mapping over the parsed output that
        also carries ``usage`` (finding #92), like ``generate`` does — the
        channel #21/#22 spend accounting reads.

        Contract (finding #91): ``system`` is the dedicated top-level channel
        for the system prompt and maps 1:1 onto the Anthropic Messages API's
        top-level ``system`` parameter. ``messages`` never carries a
        ``role: "system"`` entry — the live API rejects it on
        claude-haiku-4-5, and ``validate_request`` enforces the ban at every
        adapter, so no fake-backed green suite can hide a request that would
        400 live. ``system=None`` is omitted from recorded payloads, keeping
        pre-existing recorded request hashes valid.
        """
        ...


class ProviderContractError(ValueError):
    """A request or response violates the DESIGN §3.4 provider-seam contract.

    Raised by the shared seam validators (`validate_request` /
    `validate_response`) used by FakeAdapter, ReplayAdapter and
    RecordingAdapter alike — and by the live `AnthropicAdapter` when it
    lands. One validator, one contract: the fakes can never be laxer than
    live (review finding #62).
    """


def _generate_payload(
    messages: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    system: str | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """The canonical `generate` request payload, shared by every adapter.

    ``system`` is included only when given (the #91 rule, applied to
    generation by issue #12): a None system must hash identically to a
    pre-#12 request, or every recorded generate fixture made before the
    field existed would be silently invalidated.
    """
    payload: dict[str, Any] = {"messages": messages, "documents": documents, "config": config}
    if system is not None:
        payload["system"] = system
    return payload


def _structured_payload(
    messages: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    system: str | None,
) -> dict[str, Any]:
    """The canonical `structured` request payload, shared by every adapter.

    ``system`` is included only when given (finding #91): a None system must
    hash identically to a pre-#91 request, or every recorded fixture made
    before the field existed would be silently invalidated.
    """
    payload: dict[str, Any] = {"messages": messages, "schema": schema, "config": config}
    if system is not None:
        payload["system"] = system
    return payload


# DESIGN §3.4: "generation call documents bounded to reranked top-8".
MAX_GENERATE_DOCUMENTS = 8

# Cited generation is incompatible with structured-output/tool configuration
# (DESIGN §3.4) — the reason the protocol splits `generate` from `structured`.
_FORBIDDEN_GENERATE_CONFIG_KEYS = (
    "tools",
    "tool_choice",
    "output_schema",
    "response_format",
    "structured_output",
)


def validate_request(method: str, payload: Mapping[str, Any]) -> None:
    """Enforce the §3.4 request constraints at the provider seam.

    Called by every adapter before the request reaches a queue, a fixture
    lookup, or a (paid, live) transport — the seam-level backstop behind the
    §4.3 builder contract tests (#10/#13).
    """
    # Finding #91: the live Messages API rejects role "system" inside
    # `messages` (on claude-haiku-4-5, everywhere in the list). The system
    # prompt's only sanctioned channel is the dedicated top-level `system`
    # field, which the AnthropicAdapter passes through 1:1.
    for index, message in enumerate(payload.get("messages", ())):
        if isinstance(message, Mapping) and message.get("role") == "system":
            raise ProviderContractError(
                f"{method} messages[{index}] carries role 'system': the live "
                "Messages API rejects system-role messages - put the system "
                "prompt in the request's dedicated top-level 'system' field "
                "(finding #91)"
            )
    if method in ("generate", "generate_stream"):
        # Folded and streaming generation share ONE request contract
        # (finding #183): the streaming path is never laxer.
        documents = payload["documents"]
        if len(documents) > MAX_GENERATE_DOCUMENTS:
            raise ProviderContractError(
                f"{method} request carries {len(documents)} documents; DESIGN 3.4 bounds "
                f"the generation call to the reranked top-{MAX_GENERATE_DOCUMENTS}"
            )
        for index, document in enumerate(documents):
            citations = document.get("citations") if isinstance(document, Mapping) else None
            if not (isinstance(citations, Mapping) and citations.get("enabled") is True):
                raise ProviderContractError(
                    f"{method} document {index} lacks citations: {{enabled: true}}; "
                    "DESIGN 3.4 demands all-or-none citations - every document block "
                    "cited, no mixed cited/uncited blocks (the live API 400s on them)"
                )
        for key in _FORBIDDEN_GENERATE_CONFIG_KEYS:
            if key in payload["config"]:
                raise ProviderContractError(
                    f"{method} config carries {key!r}: cited generation is never combined "
                    "with structured-output/tool configuration (DESIGN 3.4) - use a "
                    "separate structured call"
                )
    elif method == "structured" and "citations" in payload["config"]:
        raise ProviderContractError(
            "structured config carries 'citations': structured-output calls never "
            "enable citations (IMPLEMENTATION 4.3 / DESIGN 3.4)"
        )


def validate_response(method: str, response: Any) -> None:
    """Enforce the per-method response type at the provider seam.

    A mis-programmed fake or a misfiled/mistyped replay fixture fails at the
    seam, not wherever downstream code first trips over the wrong type.
    """
    if method == "generate":
        if not isinstance(response, AnswerWithCitations):
            raise ProviderContractError(
                f"generate must return AnswerWithCitations, got {type(response).__name__}"
            )
    elif method == "generate_stream":
        # The streaming seam serves transport events (finding #183): a
        # RawProviderResponse (the recorded-events envelope) or a plain
        # sequence of event mappings. A folded AnswerWithCitations here is
        # a mistyped fixture/programming - there is no stream to serve.
        if isinstance(response, RawProviderResponse):
            events: Any = response.events
        else:
            events = response
        if (
            isinstance(events, (str, bytes))
            or isinstance(events, Mapping)
            or not isinstance(events, Sequence)
            or not all(isinstance(event, Mapping) for event in events)
        ):
            raise ProviderContractError(
                "generate_stream must serve a sequence of transport event mappings "
                "(directly or as RawProviderResponse.events), got "
                f"{type(response).__name__}"
            )
    elif method == "structured":
        # Finding #92: the structured seam has a usage channel, like
        # generate. Adapters return StructuredResult; the fakes wrap
        # programmed plain mappings before this check runs.
        if not isinstance(response, StructuredResult):
            raise ProviderContractError(
                f"structured must return StructuredResult, got {type(response).__name__}"
            )
    elif not isinstance(response, Mapping):
        raise ProviderContractError(
            f"{method} must return a mapping (structured output), got {type(response).__name__}"
        )


def _stream_events_from(response: Any) -> Iterator[dict[str, Any]]:
    """A validated generate_stream response -> the event iterator the seam serves.

    Accepts the two shapes ``validate_response("generate_stream", ...)``
    admits: a :class:`RawProviderResponse` (recorded-events envelope) or a
    plain sequence of event mappings. Events are served as deep copies so
    a consumer mutating one can never corrupt a programmed sequence or a
    cached fixture.
    """
    events = response.events if isinstance(response, RawProviderResponse) else response
    return iter([copy.deepcopy(dict(event)) for event in events])


class FakeAdapterExhaustedError(AssertionError):
    """A FakeAdapter method was called more times than responses were programmed.

    Subclasses AssertionError so an over-calling code path (e.g. a retry loop
    retrying twice where the design says once) fails the test loudly even if
    the test forgot to bound the call count itself.
    """


class _AdapterMethodsMixin:
    """Shared `ProviderAdapter` method signatures and payload construction
    (finding #109 / issue #109).

    `FakeAdapter`, `ReplayAdapter` and `RecordingAdapter` each want the same
    `generate`/`structured` surface and build the same payload shapes;
    only what happens to a built payload differs per adapter. This mixin
    owns the signatures and the payload construction —
    single-sourcing the shape that `canonical_request_hash` keys fixtures on
    — and forwards to `self._dispatch(method, payload)`, which each adapter
    implements.
    """

    def _dispatch(self, method: str, payload: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        system: str | Sequence[Mapping[str, Any]] | None = None,
    ) -> AnswerWithCitations:
        return self._dispatch(
            "generate",
            _generate_payload(messages, documents, config, system),
        )

    def generate_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        system: str | Sequence[Mapping[str, Any]] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        # Same payload shape as `generate` (one canonical request hash
        # discipline); the per-adapter _dispatch returns an event iterator
        # AFTER validate_request has run - never a lazily-validating
        # generator (finding #183: zero transport touches on violation).
        return self._dispatch(
            "generate_stream",
            _generate_payload(messages, documents, config, system),
        )

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
        system: str | None = None,
    ) -> StructuredResult:
        return self._dispatch(
            "structured",
            _structured_payload(messages, schema, config, system),
        )


class FakeAdapter(_AdapterMethodsMixin):
    """Programmable ProviderAdapter double (IMPLEMENTATION.md §4.1).

    Returns whatever the test programs and records every call (method name +
    full request payload) in `calls`, in call order. Responses are consumed
    strictly in sequence per method — retry-path tests (#16) program exactly
    the sequence they expect, and any extra call raises
    `FakeAdapterExhaustedError`. A programmed `BaseException` instance is
    raised instead of returned, for failure-path tests.
    """

    def __init__(
        self,
        generate_results: Sequence[Any] = (),
        structured_results: Sequence[Any] = (),
        generate_stream_results: Sequence[Any] = (),
    ) -> None:
        self.calls: list[RecordedCall] = []
        self._queues: dict[str, list[Any]] = {
            "generate": list(generate_results),
            "structured": list(structured_results),
            # Each programmed generate_stream result is one whole stream:
            # a sequence of transport event mappings, or a
            # RawProviderResponse whose `events` are served (finding #183).
            "generate_stream": list(generate_stream_results),
        }

    def queue(self, method: str, *results: Any) -> None:
        """Append programmed results for `method` after construction."""
        self._queues[method].extend(results)

    def calls_to(self, method: str) -> list[RecordedCall]:
        """The recorded calls to one method, in call order."""
        return [call for call in self.calls if call.method == method]

    def _dispatch(self, method: str, payload: Mapping[str, Any]) -> Any:
        # Validate before recording or consuming: a contract-violating call
        # never reaches the seam, mirroring the live path where the request
        # builder raises before the transport is touched (finding #62).
        validate_request(method, payload)
        # Record before consuming: the call log must stay truthful even for
        # the over-call that exhausts the queue. Deep copy so code under test
        # that mutates its message/document structures after the call (the
        # #16 retry pattern) cannot retroactively rewrite the log (#69).
        self.calls.append(RecordedCall(method=method, payload=copy.deepcopy(dict(payload))))
        queue = self._queues[method]
        if not queue:
            raise FakeAdapterExhaustedError(
                f"FakeAdapter.{method} called {len(self.calls_to(method))} time(s) but only "
                f"{len(self.calls_to(method)) - 1} response(s) were programmed"
            )
        result = queue.pop(0)
        if isinstance(result, BaseException):
            raise result
        # Programming convenience (finding #92): tests may queue plain
        # mappings for structured; the seam still RETURNS the contract type,
        # so consumers can always rely on `.usage` existing.
        if (
            method == "structured"
            and isinstance(result, Mapping)
            and not isinstance(result, StructuredResult)
        ):
            result = StructuredResult(value=result)
        validate_response(method, result)
        if method == "generate_stream":
            return _stream_events_from(result)
        return result


# The cache breakpoint the AnthropicAdapter request builder places on the
# LAST STATIC system block (issue #12): everything up to and including it
# is the cacheable prefix; every volatile block (tone instruction) and the
# per-query messages/documents come after, so two queries share the prefix
# byte-for-byte. TTL default (5 min) — the demo's traffic shape.
GENERATION_CACHE_CONTROL = {"type": "ephemeral"}


def _as_text_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise a seam message's content to a list of text-block dicts.

    The seam carries content either as a bare string or as a sequence of
    ``{"type": "text", "text": ...}`` blocks; the API request always uses
    the block form so document blocks and question text compose into one
    content list.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [dict(block) for block in content]


def build_anthropic_messages_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure: a seam `generate` payload -> Anthropic Messages API kwargs.

    The suite in ``tests/unit/test_generation_request_builders.py`` pins
    the mapping:

    - ``model`` / ``max_tokens`` from ``payload["config"]`` — nothing
      else from config leaks into the API request, and the builder can
      NEVER emit structured-output/tool configuration on a citations
      call (DESIGN §3.4; the live API 400s — spike-03 probe 1);
    - top-level ``system`` (finding #91) passed through in seam order,
      with :data:`GENERATION_CACHE_CONTROL` stamped on the last static
      block (block 0) and NO cache_control on any volatile block after
      it — the byte-stable prefix that must clear Haiku 4.5's
      4096-token cacheable-prefix floor;
    - ``messages``: one user turn whose content is the document blocks
      (spike-proven custom-content shape, ``citations: {enabled: true}``
      on all — all-or-none, ≤8) in seam order, followed by the question
      text block(s);
    - never a ``role: "system"`` message (the live API rejects it).

    Pure and transport-free: unit-testable without a key, and the §4.3
    contract tests run against it before any network call is possible.
    Even handed a corrupt seam payload it never CONSTRUCTS a violating
    request: the shared §3.4 validator runs first (finding #62).
    """
    validate_request("generate", payload)

    api_request: dict[str, Any] = {
        # Model id and output budget are the ONLY config fields that reach
        # the API request; no other key is read, so no structured-output or
        # tool configuration can ever be emitted here.
        "model": payload["config"]["model"],
        "max_tokens": payload["config"]["max_tokens"],
    }

    system = [dict(block) for block in payload.get("system") or ()]
    if system:
        # The cache breakpoint sits on the LAST STATIC block — block 0, the
        # committed system prompt. Everything after it (tone instruction) is
        # volatile and must stay unmarked, or per-query bytes would join the
        # "stable" prefix and silently zero the cache.
        system[0] = {**system[0], "cache_control": dict(GENERATION_CACHE_CONTROL)}
        api_request["system"] = system

    question_blocks: list[dict[str, Any]] = []
    for message in payload["messages"]:
        question_blocks.extend(_as_text_blocks(message["content"]))
    api_request["messages"] = [
        {
            "role": "user",
            # One user turn: document blocks in seam (reranked) order, so
            # document_index == rank, then the question text — all volatile,
            # all after the cached prefix.
            "content": [dict(block) for block in payload["documents"]] + question_blocks,
        }
    ]
    return api_request


class ProviderTransportError(RuntimeError):
    """A live provider transport failed (connect-time or mid-stream).

    The seam-typed wrapper for the vendor SDK's exception family (finding
    #184): ADR-012 scopes the provider seam as the model-agnostic
    interface, so raw ``anthropic.*`` exception types never leak through
    it. Carries what consumers (#22's HTTP/SSE mapping, retry policy)
    need — ``status_code`` (None for connect/timeout failures),
    ``error_type`` (the provider's error-type token when reported, e.g.
    ``overloaded_error``) and a ``retryable`` flag — while the raw SDK
    exception stays chained as ``__cause__`` for logs only.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable


def _seam_transport_error(exc: Exception) -> ProviderTransportError | None:
    """Map an anthropic SDK exception to the seam type; None for others.

    Retryability follows the transport semantics: connection/timeout
    failures and 408/429/5xx statuses are transient (the SDK's own retry
    classification); anything else — 4xx contract errors above all — is
    permanent and must surface as such, never be retried into spend.
    The seam message names the exception class, status and error type but
    NEVER echoes the SDK message text (it can carry internal detail; the
    chained ``__cause__`` keeps it available to logs).
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - SDK always present when live
        return None
    if not isinstance(exc, anthropic.APIError):
        return None
    status_code = getattr(exc, "status_code", None)
    error_type = None
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        reported = body.get("error")
        if isinstance(reported, Mapping):
            error_type = reported.get("type")
    if isinstance(exc, anthropic.APIConnectionError):
        # Includes APITimeoutError; no HTTP status ever arrived.
        retryable = True
    elif isinstance(status_code, int):
        retryable = status_code in (408, 429) or status_code >= 500
    else:
        retryable = False
    detail = type(exc).__name__
    if status_code is not None:
        detail += f", status {status_code}"
    if error_type:
        detail += f", {error_type}"
    return ProviderTransportError(
        f"provider transport failure ({detail}); {'retryable' if retryable else 'not retryable'}",
        status_code=status_code if isinstance(status_code, int) else None,
        error_type=error_type,
        retryable=retryable,
    )


class AnthropicKeyMissingError(RuntimeError):
    """A live AnthropicAdapter call was attempted with no API key available.

    Raised BEFORE the SDK client is constructed and before any network
    I/O: a keyless environment (the unit/integration tiers, keyless CI)
    must fail loudly and locally, never half-way into a transport.
    """


def accumulate_answer_from_stream_events(
    events: Iterable[Mapping[str, Any]],
) -> AnswerWithCitations:
    """Pure: an Anthropic streaming event sequence -> AnswerWithCitations.

    The transport-side twin of ``rag.generation.answer_stream_to_sse``
    (which translates the same vocabulary to SSE): text deltas
    concatenate in transport order; each ``citations_delta`` becomes a
    :class:`Citation` (custom-content ``content_block_location`` shape,
    spike-03); usage merges ``message_start``'s message usage (input +
    cache metadata) with ``message_delta``'s closing usage
    (output_tokens), keeping integer fields only — the shape
    ``AnswerWithCitations.usage`` consumers (#22 spend accounting, the
    cache smoke check) read.
    """
    text_parts: list[str] = []
    citations: list[Citation] = []
    usage: dict[str, int] = {}

    def absorb_usage(reported: Any) -> None:
        # message_start (input + cache metadata) and message_delta (closing
        # output_tokens) report disjoint fields; overwrite-merge folds them
        # into one usage view. Shares the canonical int/bool filter.
        usage.update(filter_int_usage(reported))

    for event in events:
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif delta_type == "citations_delta":
                citation = delta.get("citation") or {}
                citations.append(
                    Citation(
                        cited_text=citation.get("cited_text", ""),
                        document_index=citation["document_index"],
                        document_title=citation.get("document_title"),
                        start_block_index=citation.get("start_block_index"),
                        end_block_index=citation.get("end_block_index"),
                    )
                )
        elif event_type == "message_start":
            message = event.get("message") or {}
            absorb_usage(message.get("usage"))
        elif event_type == "message_delta":
            absorb_usage(event.get("usage"))

    return AnswerWithCitations(
        text="".join(text_parts),
        citations=tuple(citations),
        usage=usage or None,
    )


def build_anthropic_structured_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure: a seam ``structured`` payload -> Anthropic Messages API kwargs.

    Model + max_tokens are the only config fields that reach the request;
    the schema rides the ``output_config.format`` json-schema
    structured-output channel (spike/API reference: incompatible with
    citations — the live API 400s on the combination, consistent with
    §3.4 / IMPLEMENTATION §4.3, and the reason no citations config is ever
    emitted here). ``system`` (finding #91) passes through 1:1 to the
    top-level ``system`` field when given; ``messages`` never carries a
    ``role: "system"`` entry. The §3.4 seam validator runs first (finding
    #62), so even a corrupt payload never constructs a violating request.
    """
    validate_request("structured", payload)
    api_request: dict[str, Any] = {
        "model": payload["config"]["model"],
        "max_tokens": payload["config"]["max_tokens"],
        "output_config": {"format": {"type": "json_schema", "schema": dict(payload["schema"])}},
        "messages": [dict(message) for message in payload["messages"]],
    }
    system = payload.get("system")
    if system is not None:
        api_request["system"] = system
    return api_request


def _structured_result_from_message(message: Any) -> StructuredResult:
    """Fold a Messages API response into a StructuredResult.

    ``output_config.format`` guarantees the first text block is valid JSON
    for the schema; parse it into the value and carry the integer usage
    fields (finding #92) so spend accounting can observe structured-call
    tokens. A response that is not a JSON object despite the constraint is
    a transport-contract violation, surfaced as ``ProviderContractError`` —
    but that response was still fully CHARGED (e.g. output truncated at
    max_tokens), so its usage is attached to the error (``exc.usage``) for
    the ledger, never dropped (finding #205).
    """
    dumped = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    usage: dict[str, int] | None = filter_int_usage(dumped.get("usage")) or None
    text_parts = [
        block.get("text", "")
        for block in dumped.get("content", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    text = "".join(text_parts).strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        error = ProviderContractError(
            "structured output was not valid JSON despite output_config.format json_schema"
        )
        error.usage = usage
        raise error from exc
    if not isinstance(value, Mapping):
        error = ProviderContractError(
            f"structured output JSON was {type(value).__name__}, expected an object"
        )
        error.usage = usage
        raise error
    return StructuredResult(value=dict(value), usage=usage)


class AnthropicAdapter(_AdapterMethodsMixin):
    """The live Anthropic-backed ProviderAdapter (issue #12).

    Contract-validated from day one: `_dispatch` runs the same shared
    seam validator as FakeAdapter/ReplayAdapter/RecordingAdapter (finding
    #62 — one validator, one contract; a §3.4-violating request raises
    ``ProviderContractError`` here exactly as it would everywhere else,
    before the transport is touched). Request construction is the pure
    :func:`build_anthropic_messages_request`; the transport streams the
    response (SDK raw event iterator — timeout-safe for long cited
    answers) and folds the event sequence into
    :class:`AnswerWithCitations` via the pure
    :func:`accumulate_answer_from_stream_events`.

    Construction never needs a key (``evals/scripts`` builds the adapter
    before argparse-time key checks); the key is resolved lazily at call
    time — explicit ``api_key`` argument first, then
    :data:`LIVE_KEY_ENV_VAR` — and a keyless call raises
    :class:`AnthropicKeyMissingError` before the SDK client exists, so
    no keyless tier can ever reach the network through this class. The
    ``structured`` transport (issue #13) maps the seam payload onto the
    Messages API's ``output_config.format`` json-schema structured-output
    channel (incompatible with citations by construction — §3.4) and
    folds the JSON response into a :class:`StructuredResult`.
    """

    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        self._api_key = api_key
        self._client = client

    def _live_client(self) -> Any:
        if self._client is None:
            api_key = self._api_key or os.environ.get(LIVE_KEY_ENV_VAR)
            if not api_key:
                raise AnthropicKeyMissingError(
                    f"AnthropicAdapter has no API key: pass api_key= or set "
                    f"{LIVE_KEY_ENV_VAR}. Raised before any SDK client or network "
                    "I/O exists - the unit/integration tiers must never reach a "
                    "live transport (IMPLEMENTATION.md 3)"
                )
            # Lazy import: the anthropic SDK is a live-path dependency; the
            # keyless tiers never pay the import (repo convention, e.g.
            # FlagEmbedding in rag.indexing).
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _dispatch(self, method: str, payload: Mapping[str, Any]) -> Any:
        # The §3.4 backstop fires BEFORE any (paid, live) transport —
        # identical ordering to every other adapter (finding #62), on the
        # streaming path exactly as on the folded one (finding #183).
        validate_request(method, payload)
        if method == "structured":
            # Structured output (issue #13): a non-streaming Messages call
            # constrained by output_config.format json_schema; the response
            # is a small verdict object, so no streaming/timeout guard is
            # needed. Transport failures cross the seam as ProviderTransportError.
            return self._structured_call(payload)
        if method not in ("generate", "generate_stream"):
            raise NotImplementedError(
                f"AnthropicAdapter has no transport for provider-seam method {method!r}"
            )
        api_request = build_anthropic_messages_request(payload)
        # Key resolution is eager too: a keyless environment fails loudly
        # here, never lazily inside a half-consumed iterator.
        client = self._live_client()
        if method == "generate_stream":
            return self._transport_events(client, api_request)
        # The folded convenience is implemented OVER the stream (finding
        # #183): one transport path, one event vocabulary, no drift.
        response = accumulate_answer_from_stream_events(self._transport_events(client, api_request))
        validate_response(method, response)
        return response

    def _transport_events(
        self, client: Any, api_request: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Yield the SDK's streaming events as plain mappings, in SDK order.

        SDK exception types never cross the seam (finding #184): connect-time
        and mid-stream ``anthropic.*`` failures are re-raised as the seam's
        :class:`ProviderTransportError`, so consumers (#22) distinguish
        transient from permanent failure without importing the vendor SDK.
        """
        try:
            event_stream = client.messages.create(**api_request, stream=True)
            for event in event_stream:
                yield event.model_dump()
        except Exception as exc:
            seam_error = _seam_transport_error(exc)
            if seam_error is None:
                raise
            raise seam_error from exc

    def _structured_call(self, payload: Mapping[str, Any]) -> StructuredResult:
        """One non-streaming structured-output call -> StructuredResult.

        SDK exception types never cross the seam (finding #184): a live
        ``anthropic.*`` failure is re-raised as :class:`ProviderTransportError`,
        exactly as on the streaming generate path.
        """
        api_request = build_anthropic_structured_request(payload)
        client = self._live_client()
        try:
            message = client.messages.create(**api_request)
        except Exception as exc:
            seam_error = _seam_transport_error(exc)
            if seam_error is None:
                raise
            raise seam_error from exc
        result = _structured_result_from_message(message)
        validate_response("structured", result)
        return result


# The command a developer runs to (re-)record replay fixtures. Recording is
# env-flag-gated and needs a live key; it never happens implicitly in any
# CI-tier test run (IMPLEMENTATION.md §4.2).
RE_RECORD_COMMAND = "CLIMATE_CHAT_RECORD=1 uv run pytest -m live"


class CanonicalisationError(ValueError):
    """A payload cannot be given a canonical form (finding #68).

    Raised for non-string dict keys (json.dumps would silently coerce them,
    letting {1: x} and {"1": x} collide onto one fixture), for non-finite
    floats (not JSON; no real API request can carry them), and for types
    with no JSON equivalent. Refusal makes collisions impossible, matching
    the repo's fail-loudly style.
    """


def _canonicalise(obj: Any) -> Any:
    """Pre-pass normaliser pinning the canonical form (finding #68).

    Mapping -> dict (so MappingProxyType etc. hash like their dict
    equivalents), Sequence -> list (tuples like lists); dict keys must be
    str; floats must be finite, and int-valued floats normalise to int —
    JSON (and the API) treat 1 and 1.0 as the same number, so an int→float
    refactor at a call site must not invalidate every recording.
    """
    if isinstance(obj, Mapping):
        canonical: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalisationError(
                    f"payload dict key {key!r} is {type(key).__name__}, not str: "
                    "non-string keys have no canonical JSON form (json.dumps would "
                    "silently coerce them, colliding distinct payloads onto one fixture)"
                )
            canonical[key] = _canonicalise(value)
        return canonical
    if isinstance(obj, (str, bytes)):
        if isinstance(obj, bytes):
            raise CanonicalisationError("payload contains bytes; no canonical JSON form")
        return obj
    if isinstance(obj, Sequence):
        return [_canonicalise(item) for item in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CanonicalisationError(
                f"payload contains non-finite float {obj!r}: not JSON, and no real "
                "API request can carry it; a canonical hash must not exist for it"
            )
        return int(obj) if obj.is_integer() else obj
    if isinstance(obj, int):
        return obj
    raise CanonicalisationError(
        f"payload contains {type(obj).__name__}, which has no canonical JSON form"
    )


def canonical_request_hash(method: str, payload: Mapping[str, Any]) -> str:
    """The canonical hash keying replay fixtures.

    **The canonical form is the scrubbed form** (finding #63): credential
    keys/headers are stripped by `scrub_payload` before hashing, so a
    fixture recorded from a credentialed payload and the same semantic
    request built by keyless CI resolve to one key, and a fixture's
    filename is always recomputable from its committed (scrubbed) request.
    Headers/keys are transport concerns, never semantic request content —
    scrubbing is a no-op on credential-free payloads, so their hashes are
    unchanged.

    sha256 over the compact, key-sorted JSON of {method, payload} after
    scrubbing and the `_canonicalise` pre-pass (abstract mappings/sequences
    normalised, non-str keys and non-finite floats rejected, int-valued
    floats folded to int). Key order never matters; any byte of semantic
    content (prompt text, documents, schema, config, method) does — so a
    changed prompt invalidates its recordings by design
    (IMPLEMENTATION.md §4.2).
    """
    canonical = json.dumps(
        {"method": method, "payload": _canonicalise(scrub_payload(payload))},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# The on-disk replay-fixture format version (finding #64). Bump on any
# incompatible change to the envelope or response encodings; ReplayAdapter
# refuses fixtures declaring any other version, so old and new fixtures are
# told apart by declaration, never heuristically.
REPLAY_FIXTURE_FORMAT_VERSION = 1


class ReplayFormatError(ValueError):
    """A replay fixture declares an unknown (or no) format_version."""


def serialize_response(response: Any) -> dict[str, Any]:
    """Serialize an adapter response into the typed replay-fixture form."""
    if isinstance(response, AnswerWithCitations):
        serialized: dict[str, Any] = {
            "type": "answer_with_citations",
            "text": response.text,
            "citations": [asdict(citation) for citation in response.citations],
        }
        if response.usage is not None:
            serialized["usage"] = dict(response.usage)
        return serialized
    if isinstance(response, RawProviderResponse):
        return {
            "type": "raw",
            "payload": dict(response.payload),
            "events": [dict(event) for event in response.events],
        }
    if isinstance(response, StructuredResult):
        # Same "dict" encoding as before #92 with an optional usage key, so
        # pre-existing fixtures need no migration and old fixtures replay
        # with usage None.
        serialized = {"type": "dict", "value": dict(response.value)}
        if response.usage is not None:
            serialized["usage"] = dict(response.usage)
        return serialized
    if isinstance(response, Mapping):
        return {"type": "dict", "value": dict(response)}
    raise TypeError(f"cannot serialize adapter response of type {type(response).__name__}")


def deserialize_response(data: Mapping[str, Any]) -> Any:
    """Inverse of `serialize_response`."""
    kind = data.get("type")
    if kind == "answer_with_citations":
        return AnswerWithCitations(
            text=data["text"],
            citations=tuple(Citation(**citation) for citation in data.get("citations", [])),
            usage=data.get("usage"),
        )
    if kind == "raw":
        return RawProviderResponse(
            payload=data["payload"],
            events=tuple(data.get("events", [])),
        )
    if kind == "dict":
        return StructuredResult(value=data["value"], usage=data.get("usage"))
    raise ValueError(f"unknown replay response type: {kind!r}")


class ReplayFixtureMissingError(LookupError):
    """A replay lookup found no recorded fixture for the request."""


class ReplayAdapter(_AdapterMethodsMixin):
    """Replays checked-in recorded responses deterministically forever

    (IMPLEMENTATION.md §4.2). Fixtures live as `<canonical request hash>.json`
    files (see `serialize_response` for the response encoding). Any request
    without a recording raises `ReplayFixtureMissingError` naming the hash and
    the re-record command — replay never invents or approximates a response.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def _dispatch(self, method: str, payload: Mapping[str, Any]) -> Any:
        # Same seam validator as FakeAdapter/RecordingAdapter (finding #62):
        # an invalid request raises ProviderContractError naming the violated
        # constraint, never the misleading "no recorded fixture" error.
        validate_request(method, payload)
        request_hash = canonical_request_hash(method, payload)
        fixture_path = self.fixtures_dir / f"{request_hash}.json"
        if not fixture_path.is_file():
            raise ReplayFixtureMissingError(
                f"no recorded replay fixture for {method} request "
                f"(canonical request hash {request_hash}; expected {fixture_path}). "
                "The request payload has changed, or was never recorded - a changed "
                "prompt invalidates its recordings by design. Re-record with a live "
                f"ANTHROPIC_API_KEY via: {RE_RECORD_COMMAND}"
            )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        version = fixture.get("_meta", {}).get("format_version")
        if version != REPLAY_FIXTURE_FORMAT_VERSION:
            raise ReplayFormatError(
                f"replay fixture {fixture_path.name} declares format_version {version!r}, "
                f"but this code reads format_version {REPLAY_FIXTURE_FORMAT_VERSION} - "
                "replay never guesses at an on-disk format; migrate the fixture or "
                f"re-record via: {RE_RECORD_COMMAND}"
            )
        response = deserialize_response(fixture["response"])
        # A "dict" fixture replayed through generate (or vice versa) is a
        # mistyped or misfiled recording — reject it at the seam.
        validate_response(method, response)
        if method == "generate_stream":
            # A raw fixture's recorded event sequence replays through the
            # stream seam (finding #183) - the envelope built for this.
            return _stream_events_from(response)
        return response


# The explicit opt-in flag for live recording (IMPLEMENTATION.md §4.2):
# no accidental live calls from the test suite, ever.
RECORD_ENV_FLAG = "CLIMATE_CHAT_RECORD"
LIVE_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class RecordingDisabledError(RuntimeError):
    """Recording was attempted without the explicit env flag and a live key."""


class SecretLeakError(RuntimeError):
    """A secret pattern survived scrubbing; the fixture was NOT written."""


# Keys (normalised: lowercase, underscores as hyphens) that carry credentials
# or transport headers. Removed wholesale by `scrub_payload` — headers are
# transport concerns, never semantic request content, so nothing of value is
# lost by dropping the whole header mapping. Broadened per finding #65: the
# recorder wraps "any transport", so proxy/gateway and non-Anthropic
# credential names must be covered too.
_CREDENTIAL_KEYS = frozenset(
    {
        "api-key",
        "x-api-key",
        "anthropic-api-key",
        "authorization",
        "proxy-authorization",
        "auth-token",
        "bearer-token",
        "headers",
        "extra-headers",
        "apikey",
        "token",
        "access-token",
        "refresh-token",
        "proxy-token",
        "id-token",
        "secret",
        "client-secret",
        "aws-secret-access-key",
        "session-key",
        "password",
        "passwd",
        "cookie",
        "set-cookie",
        "credential",
        "credentials",
    }
)

# Hyphen-separated segments that mark a key as credential-bearing wherever
# they appear (e.g. "gateway-token", "webhook-secret"). Exact segments only —
# never substrings, so "max-tokens" (segment "tokens") stays untouched.
_CREDENTIAL_KEY_SEGMENTS = frozenset(
    {
        "token",
        "secret",
        "apikey",
        "password",
        "passwd",
        "cookie",
        "credential",
        "credentials",
        "authorization",
        "bearer",
        "auth",
    }
)

# Patterns that must never appear in a written fixture. If one survives
# scrubbing (e.g. a key leaked into prompt *content*), the recorder fails
# closed instead of redacting: an upstream leak must be loud, and a silently
# redacted fixture would hide it. Broadened per finding #65 beyond
# Anthropic/bearer shapes to the token shapes gitleaks (finding #35) showed
# matter for a repo slated to go public: AWS access keys, HuggingFace,
# GitHub, and generic sk- API keys.
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-"),
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"(?i)x-api-key"),
    re.compile(r"(?i)authorization"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
)


def _normalise_key(key: Any) -> str:
    return str(key).lower().replace("_", "-")


def _is_credential_key(key: Any) -> bool:
    """True when a (normalised) key is credential-bearing (finding #65).

    Exact membership in `_CREDENTIAL_KEYS`, a trailing "-key" segment
    (api-key, session-key, aws-secret-access-key...), or any exact
    credential segment (`_CREDENTIAL_KEY_SEGMENTS`). Deliberately biased
    towards over-scrubbing: a false positive drops a config key from a
    test fixture; a false negative commits a secret to a repo destined to
    go public.
    """
    normalised = _normalise_key(key)
    if normalised in _CREDENTIAL_KEYS:
        return True
    segments = normalised.split("-")
    if segments[-1] == "key":
        return True
    return any(segment in _CREDENTIAL_KEY_SEGMENTS for segment in segments)


def scrub_payload(obj: Any) -> Any:
    """Recursively strip credential-bearing keys from a payload tree."""
    if isinstance(obj, Mapping):
        return {
            key: scrub_payload(value) for key, value in obj.items() if not _is_credential_key(key)
        }
    if isinstance(obj, (list, tuple)):
        return [scrub_payload(item) for item in obj]
    return obj


def _locate_secret(obj: Any, pattern: re.Pattern[str], path: str) -> str | None:
    """The JSON path of the first node matching `pattern`, or None.

    A locating hint for `SecretLeakError` that never repeats the value
    (finding #66). A match inside a mapping *key* reports the parent path
    with a placeholder — the key text itself would be the secret.
    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and pattern.search(key):
                return f"{path or '<root>'}.<redacted mapping key>"
            key_path = f"{path}.{key}" if path else str(key)
            found = _locate_secret(value, pattern, key_path)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            found = _locate_secret(item, pattern, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(obj, str) and pattern.search(obj):
        return path
    return None


class RecordingAdapter(_AdapterMethodsMixin):
    """Env-flag-gated recorder wrapping any transport (IMPLEMENTATION.md §4.2).

    Delegates every call to `inner` (the live `AnthropicAdapter` when
    recording for real; a `FakeAdapter` in this issue's own tests — the
    recorder is buildable and testable with no API key on the machine) and
    writes each request/response pair as a replay fixture keyed by the
    canonical request hash.

    Refuses to construct unless the environment carries BOTH the explicit
    `CLIMATE_CHAT_RECORD=1` opt-in flag and a non-empty `ANTHROPIC_API_KEY`,
    so no test-suite run can record (or hit a live transport through this
    wrapper) by accident. Stored fixtures are scrubbed of credential keys and
    scanned for secret patterns; a surviving pattern aborts the write.
    """

    def __init__(
        self,
        inner: ProviderAdapter,
        fixtures_dir: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if env is None:
            env = os.environ
        flag = env.get(RECORD_ENV_FLAG, "").strip().lower()
        if flag not in {"1", "true"}:
            raise RecordingDisabledError(
                f"replay-fixture recording is disabled: set {RECORD_ENV_FLAG}=1 "
                "explicitly to opt in (no accidental live calls from the test suite)"
            )
        if not env.get(LIVE_KEY_ENV_VAR):
            raise RecordingDisabledError(
                f"replay-fixture recording requires a live {LIVE_KEY_ENV_VAR} in the "
                "environment; none is set"
            )
        self._inner = inner
        self.fixtures_dir = Path(fixtures_dir)

    def _record(self, method: str, payload: Mapping[str, Any], response: Any) -> None:
        # Defensive against a future live inner adapter: never serialize a
        # response that violates the per-method type contract (finding #62).
        validate_response(method, response)
        fixture = {
            "_meta": {
                "format_version": REPLAY_FIXTURE_FORMAT_VERSION,
                "scrubbed": True,
                "recorder": "rag.provider.RecordingAdapter",
                "provenance": (
                    "recorded through RecordingAdapter; request/response content must "
                    "derive from synthetic fixtures or licensed-open text only "
                    "(DESIGN.md 2.1 shipping invariant)"
                ),
                # Deliberately null (finding #67): the recorder cannot attest
                # what its content derives from. A committed fixture must have
                # a human fill in {who, date, note} — the provenance guard in
                # tests/unit/test_provider_adapter.py fails until then, making
                # commit-time sign-off an explicit human act, exactly like the
                # corpus manifest's human_signoff.
                "content_signoff": None,
            },
            "method": method,
            "request": scrub_payload(payload),
            "response": serialize_response(response),
        }
        text = json.dumps(fixture, indent=2, ensure_ascii=False)
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                # Never echo any fragment of the match (finding #66): this
                # exception lands in terminal scrollback and retained CI
                # logs of live recording jobs. Name the pattern and the JSON
                # path of the offending node — never its value.
                location = _locate_secret(fixture, pattern, "") or "<serialized fixture text>"
                raise SecretLeakError(
                    f"secret pattern {pattern.pattern!r} survived scrubbing in the "
                    f"{method} fixture (at {location}); the fixture was NOT written "
                    "- find and fix the upstream leak"
                )
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = self.fixtures_dir / f"{canonical_request_hash(method, payload)}.json"
        fixture_path.write_text(text + "\n", encoding="utf-8")

    def _dispatch(self, method: str, payload: Mapping[str, Any]) -> Any:
        validate_request(method, payload)
        if method == "generate_stream":
            # Tee the inner stream into a RawProviderResponse fixture
            # (finding #183): events are yielded onward untouched, and the
            # fixture is written only when the stream COMPLETES - a
            # truncated or errored stream never becomes a recording.
            return self._record_stream(payload, getattr(self._inner, method)(**payload))
        response = getattr(self._inner, method)(**payload)
        self._record(method, payload, response)
        return response

    def _record_stream(
        self, payload: Mapping[str, Any], inner_stream: Iterable[Mapping[str, Any]]
    ) -> Iterator[Mapping[str, Any]]:
        events: list[Mapping[str, Any]] = []
        for event in inner_stream:
            events.append(copy.deepcopy(dict(event)))
            yield event
        self._record(
            "generate_stream",
            payload,
            RawProviderResponse(payload={}, events=tuple(events)),
        )
