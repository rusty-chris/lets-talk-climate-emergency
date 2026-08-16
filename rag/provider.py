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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
        # Record first: the call log must stay truthful even for the
        # over-call that exhausts the queue.
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
    """Canonical hash keying replay fixtures. Stub — issue #24."""
    raise NotImplementedError("issue #24: canonical_request_hash not implemented yet")


class ReplayFixtureMissingError(LookupError):
    """A replay lookup found no recorded fixture for the request."""


class ReplayAdapter:
    """Replays checked-in recorded responses (IMPLEMENTATION.md §4.2). Stub — issue #24."""

    def __init__(self, fixtures_dir: Path) -> None:
        raise NotImplementedError("issue #24: ReplayAdapter not implemented yet")


class RecordingAdapter:
    """Env-flag-gated live recorder (IMPLEMENTATION.md §4.2). Stub — issue #24."""

    def __init__(
        self,
        inner: ProviderAdapter,
        fixtures_dir: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        raise NotImplementedError("issue #24: RecordingAdapter not implemented yet")
