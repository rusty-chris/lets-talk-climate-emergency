"""Provider adapter seam — the deterministic boundary around all LLM calls.

IMPLEMENTATION.md §4 / issue #24. Every LLM call in the system goes through
the `ProviderAdapter` protocol; nothing in the unit or integration test tiers
may touch the network. Three implementations are planned (IMPLEMENTATION.md
§1): `AnthropicAdapter` (live, built when the first consumer issue lands),
`FakeAdapter` (programmable, records every call) and `ReplayAdapter`
(checked-in recorded fixtures keyed by canonical request hash). Recording is
env-flag-gated via `RecordingAdapter`.

The protocol mirrors DESIGN.md §3.3–3.4/§5: cited generation
(`generate`) is a separate call from structured output (`structured`,
`plan_chart`) because Anthropic native citations are incompatible with
structured-output configuration — the §3.4 contract tests (issues #10/#13)
enforce that on the request builders.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
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
    """Return type of `ProviderAdapter.generate` (DESIGN.md §5)."""

    text: str
    citations: tuple[Citation, ...] = ()


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
    ) -> AnswerWithCitations:
        """Grounded generation with native citations (DESIGN.md §3.3)."""
        ...

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Structured-output call (rewriter, classifier, judges — never citations)."""
        ...

    def plan_chart(
        self,
        request: str,
        catalog: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Chart-planner structured call; #15 narrows the return type to ChartSpec."""
        ...


class ProviderContractError(ValueError):
    """A request or response violates the DESIGN §3.4 provider-seam contract.

    Raised by the shared seam validators (`validate_request` /
    `validate_response`) used by FakeAdapter, ReplayAdapter and
    RecordingAdapter alike — and by the live `AnthropicAdapter` when it
    lands. One validator, one contract: the fakes can never be laxer than
    live (review finding #62).
    """


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
    if method == "generate":
        documents = payload["documents"]
        if len(documents) > MAX_GENERATE_DOCUMENTS:
            raise ProviderContractError(
                f"generate request carries {len(documents)} documents; DESIGN 3.4 bounds "
                f"the generation call to the reranked top-{MAX_GENERATE_DOCUMENTS}"
            )
        for index, document in enumerate(documents):
            citations = document.get("citations") if isinstance(document, Mapping) else None
            if not (isinstance(citations, Mapping) and citations.get("enabled") is True):
                raise ProviderContractError(
                    f"generate document {index} lacks citations: {{enabled: true}}; "
                    "DESIGN 3.4 demands all-or-none citations - every document block "
                    "cited, no mixed cited/uncited blocks (the live API 400s on them)"
                )
        for key in _FORBIDDEN_GENERATE_CONFIG_KEYS:
            if key in payload["config"]:
                raise ProviderContractError(
                    f"generate config carries {key!r}: cited generation is never combined "
                    "with structured-output/tool configuration (DESIGN 3.4) - use the "
                    "separate structured/plan_chart calls"
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
    elif not isinstance(response, Mapping):
        raise ProviderContractError(
            f"{method} must return a mapping (structured output), got {type(response).__name__}"
        )


class FakeAdapterExhaustedError(AssertionError):
    """A FakeAdapter method was called more times than responses were programmed.

    Subclasses AssertionError so an over-calling code path (e.g. a retry loop
    retrying twice where the design says once) fails the test loudly even if
    the test forgot to bound the call count itself.
    """


class FakeAdapter:
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
        plan_chart_results: Sequence[Any] = (),
    ) -> None:
        self.calls: list[RecordedCall] = []
        self._queues: dict[str, list[Any]] = {
            "generate": list(generate_results),
            "structured": list(structured_results),
            "plan_chart": list(plan_chart_results),
        }

    def queue(self, method: str, *results: Any) -> None:
        """Append programmed results for `method` after construction."""
        self._queues[method].extend(results)

    def calls_to(self, method: str) -> list[RecordedCall]:
        """The recorded calls to one method, in call order."""
        return [call for call in self.calls if call.method == method]

    def _next(self, method: str, payload: Mapping[str, Any]) -> Any:
        # Validate before recording or consuming: a contract-violating call
        # never reaches the seam, mirroring the live path where the request
        # builder raises before the transport is touched (finding #62).
        validate_request(method, payload)
        # Record before consuming: the call log must stay truthful even for
        # the over-call that exhausts the queue.
        self.calls.append(RecordedCall(method=method, payload=dict(payload)))
        queue = self._queues[method]
        if not queue:
            raise FakeAdapterExhaustedError(
                f"FakeAdapter.{method} called {len(self.calls_to(method))} time(s) but only "
                f"{len(self.calls_to(method)) - 1} response(s) were programmed"
            )
        result = queue.pop(0)
        if isinstance(result, BaseException):
            raise result
        validate_response(method, result)
        return result

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> AnswerWithCitations:
        return self._next(
            "generate",
            {"messages": messages, "documents": documents, "config": config},
        )

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._next(
            "structured",
            {"messages": messages, "schema": schema, "config": config},
        )

    def plan_chart(
        self,
        request: str,
        catalog: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._next("plan_chart", {"request": request, "catalog": catalog})


# The command a developer runs to (re-)record replay fixtures. Recording is
# env-flag-gated and needs a live key; it never happens implicitly in any
# CI-tier test run (IMPLEMENTATION.md §4.2).
RE_RECORD_COMMAND = "CLIMATE_CHAT_RECORD=1 uv run pytest -m live"


def canonical_request_hash(method: str, payload: Mapping[str, Any]) -> str:
    """The canonical hash keying replay fixtures.

    sha256 over the compact, key-sorted JSON of {method, payload}. Key order
    never matters; any byte of semantic content (prompt text, documents,
    schema, config, method) does — so a changed prompt invalidates its
    recordings by design (IMPLEMENTATION.md §4.2).
    """
    canonical = json.dumps(
        {"method": method, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialize_response(response: Any) -> dict[str, Any]:
    """Serialize an adapter response into the typed replay-fixture form."""
    if isinstance(response, AnswerWithCitations):
        return {
            "type": "answer_with_citations",
            "text": response.text,
            "citations": [asdict(citation) for citation in response.citations],
        }
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
        )
    if kind == "dict":
        return dict(data["value"])
    raise ValueError(f"unknown replay response type: {kind!r}")


class ReplayFixtureMissingError(LookupError):
    """A replay lookup found no recorded fixture for the request."""


class ReplayAdapter:
    """Replays checked-in recorded responses deterministically forever

    (IMPLEMENTATION.md §4.2). Fixtures live as `<canonical request hash>.json`
    files (see `serialize_response` for the response encoding). Any request
    without a recording raises `ReplayFixtureMissingError` naming the hash and
    the re-record command — replay never invents or approximates a response.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def _replay(self, method: str, payload: Mapping[str, Any]) -> Any:
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
        response = deserialize_response(fixture["response"])
        # A "dict" fixture replayed through generate (or vice versa) is a
        # mistyped or misfiled recording — reject it at the seam.
        validate_response(method, response)
        return response

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> AnswerWithCitations:
        return self._replay(
            "generate",
            {"messages": messages, "documents": documents, "config": config},
        )

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._replay(
            "structured",
            {"messages": messages, "schema": schema, "config": config},
        )

    def plan_chart(
        self,
        request: str,
        catalog: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._replay("plan_chart", {"request": request, "catalog": catalog})


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
# lost by dropping the whole header mapping.
_CREDENTIAL_KEYS = frozenset(
    {
        "api-key",
        "x-api-key",
        "anthropic-api-key",
        "authorization",
        "auth-token",
        "bearer-token",
        "headers",
        "extra-headers",
    }
)

# Patterns that must never appear in a written fixture. If one survives
# scrubbing (e.g. a key leaked into prompt *content*), the recorder fails
# closed instead of redacting: an upstream leak must be loud, and a silently
# redacted fixture would hide it.
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-"),
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"(?i)x-api-key"),
    re.compile(r"(?i)authorization"),
)


def _normalise_key(key: Any) -> str:
    return str(key).lower().replace("_", "-")


def scrub_payload(obj: Any) -> Any:
    """Recursively strip credential-bearing keys from a payload tree."""
    if isinstance(obj, Mapping):
        return {
            key: scrub_payload(value)
            for key, value in obj.items()
            if _normalise_key(key) not in _CREDENTIAL_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [scrub_payload(item) for item in obj]
    return obj


class RecordingAdapter:
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
                "scrubbed": True,
                "recorder": "rag.provider.RecordingAdapter",
                "provenance": (
                    "recorded through RecordingAdapter; request/response content must "
                    "derive from synthetic fixtures or licensed-open text only "
                    "(DESIGN.md 2.1 shipping invariant)"
                ),
            },
            "method": method,
            "request": scrub_payload(payload),
            "response": serialize_response(response),
        }
        text = json.dumps(fixture, indent=2, ensure_ascii=False)
        for pattern in _SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                raise SecretLeakError(
                    f"secret pattern {pattern.pattern!r} survived scrubbing in the "
                    f"{method} fixture (matched {match.group(0)[:12]!r}...); the fixture "
                    "was NOT written - find and fix the upstream leak"
                )
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = self.fixtures_dir / f"{canonical_request_hash(method, payload)}.json"
        fixture_path.write_text(text + "\n", encoding="utf-8")

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        documents: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> AnswerWithCitations:
        payload = {"messages": messages, "documents": documents, "config": config}
        validate_request("generate", payload)
        response = self._inner.generate(**payload)
        self._record("generate", payload, response)
        return response

    def structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {"messages": messages, "schema": schema, "config": config}
        validate_request("structured", payload)
        response = self._inner.structured(**payload)
        self._record("structured", payload, response)
        return response

    def plan_chart(
        self,
        request: str,
        catalog: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {"request": request, "catalog": catalog}
        validate_request("plan_chart", payload)
        response = self._inner.plan_chart(**payload)
        self._record("plan_chart", payload, response)
        return response
