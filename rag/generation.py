"""Grounded generation with Claude native citations (issue #12, DESIGN §3.3/§3.4).

RED-phase contract stubs: every behaviour raises ``NotImplementedError``;
the failing suites in ``tests/unit/test_generation_*.py`` and
``tests/integration/test_generation_live.py`` pin the contract below.

This module is the orchestration layer between retrieval (#11) and the
provider seam (#24): it turns a :class:`rag.retrieval.RetrievedPassages`
into ONE ``ProviderAdapter.generate`` call and resolves the returned
native citations back onto the supplied passages. It never touches the
network itself — the AnthropicAdapter (``rag.provider``) owns transport.

Contract points the red suite pins:

- **Pure request builder.** :func:`build_generation_request` maps
  (RetrievedPassages, question, config) -> the seam payload for
  ``adapter.generate(**request)``: ``messages`` (the user question),
  ``documents`` (one custom-content block per passage, spike-03 shape,
  ``citations: {enabled: true}`` on every block), ``config`` (model +
  max_tokens, NEVER any structured-output/tool key) and ``system`` (the
  provider-agnostic system channel, finding #91: a sequence of text
  blocks with the STATIC system prompt first and every volatile block —
  the tone instruction — strictly after it, so the Anthropic builder can
  cache the stable prefix).
- **§3.4 enforced at the builder, not just the seam.** More than
  :data:`rag.retrieval.GENERATION_TOP_K` passages raises
  :class:`GenerationContractError` before any request object exists; the
  builder never constructs an uncited document block or a config carrying
  structured-output/tool configuration — the shared seam validator
  (``rag.provider.validate_request``) is the backstop, never the first
  line of defence.
- **System prompt is a committed artifact** (:data:`SYSTEM_PROMPT_PATH`)
  carrying DESIGN §3.3's eight rules; :func:`load_system_prompt` reads it
  verbatim (prompt text is data, IMPLEMENTATION.md §1). The prompt is
  deliberately sized: Haiku 4.5's minimum cacheable prefix is
  :data:`HAIKU_MIN_CACHEABLE_PREFIX_TOKENS` tokens and caching fails
  SILENTLY below it (orchestrator note on issue #12), so
  :func:`assert_cacheable_prefix` refuses loudly when the static prefix's
  conservative token lower bound (:func:`estimate_tokens_lower_bound`)
  is under the floor.
- **Model policy.** Default :data:`GENERATION_MODEL_DEFAULT`
  (claude-haiku-4-5); :data:`OPUS_BEST_MODEL` is the gated "best" mode
  (DESIGN §3.3/§9): requesting it without
  ``GenerationConfig.best_mode_enabled`` raises
  :class:`BestModeNotEnabledError`, and when enabled the
  ``GenerationConfig.budget_guard`` hook (the #22 sub-cap seam) is
  invoked with the model id before the request is built.
- **Flow.** :func:`generate_grounded_answer` makes exactly ONE
  ``generate`` call for a RetrievedPassages, resolves citations via
  :func:`resolve_citations` (a ``document_index`` outside the supplied
  document set raises — resolution never guesses), surfaces the #143
  degraded-parse flags on every :class:`CitedPassage`, and passes the
  adapter-reported ``usage`` (cache metadata included) through on the
  :class:`GroundedAnswer` for #22's cost accounting. A
  :class:`rag.retrieval.HonestRefusal` BYPASSES generation entirely —
  returned unchanged, zero adapter calls (§3.5: the refusal path is
  template-only).
- **Streaming.** :func:`answer_stream_to_sse` is a pure translation from
  the Anthropic streaming event sequence (``message_start`` /
  ``content_block_start`` / ``content_block_delta`` with ``text_delta``
  and ``citations_delta`` payloads / ``content_block_stop`` /
  ``message_delta`` (usage) / ``message_stop``) to the service's SSE
  events: ``text`` and ``citation`` events in transport order, a
  ``usage`` event carrying the token accounting (incl.
  ``cache_read_input_tokens``), and a final ``footer`` event carrying
  :func:`build_response_footer`'s verification note + corpus vintage
  (DESIGN §3.5/§10 cadence).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.provider import AnswerWithCitations, ProviderAdapter
from rag.retrieval import HonestRefusal, RetrievedPassages

__all__ = [
    "GENERATION_MODEL_DEFAULT",
    "OPUS_BEST_MODEL",
    "GENERATION_MAX_TOKENS_DEFAULT",
    "HAIKU_MIN_CACHEABLE_PREFIX_TOKENS",
    "SYSTEM_PROMPT_PATH",
    "GenerationContractError",
    "BestModeNotEnabledError",
    "GenerationConfig",
    "CitedPassage",
    "GroundedAnswer",
    "load_system_prompt",
    "estimate_tokens_lower_bound",
    "assert_cacheable_prefix",
    "build_generation_request",
    "resolve_citations",
    "generate_grounded_answer",
    "build_response_footer",
    "answer_stream_to_sse",
]

#: DESIGN §3.3: generation default. Model id is config, never hard-coded
#: at a call site.
GENERATION_MODEL_DEFAULT = "claude-haiku-4-5"

#: DESIGN §3.3/§9: the optional "best" mode, behind the budget sub-cap
#: (#22). Never the default.
OPUS_BEST_MODEL = "claude-opus-4-8"

#: Default output budget for a cited answer (§9 cost model: ~500 out).
GENERATION_MAX_TOKENS_DEFAULT = 1024

#: Haiku 4.5's minimum cacheable prompt prefix (orchestrator note on
#: issue #12, sourced from reviews/sota-portfolio-review-2026-08.md):
#: below this the API silently writes NO cache entry —
#: cache_creation_input_tokens: 0, no error — and the §9 "less with
#: prompt caching" assumption never fires. The static prefix must be
#: deliberately sized to clear it.
HAIKU_MIN_CACHEABLE_PREFIX_TOKENS = 4096

#: The committed system-prompt artifact carrying DESIGN §3.3's eight
#: rules. Prompt text is data (IMPLEMENTATION.md §1): the tests assert
#: its load-bearing invariants textually, not its phrasing.
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generation_system_prompt.md"


class GenerationContractError(ValueError):
    """A generation request or response violates the §3.3/§3.4 contract."""


class BestModeNotEnabledError(GenerationContractError):
    """Opus "best" mode was requested without the enabling config flag.

    DESIGN §3.3/§9: claude-opus-4-8 sits behind the budget sub-cap; a
    request for it with ``best_mode_enabled`` False is a configuration
    bug, refused loudly before any request is built.
    """


@dataclass(frozen=True)
class GenerationConfig:
    """Injected generation configuration (model id is config, §3.3).

    ``budget_guard`` is the #22 hook point: when best mode is enabled and
    the opus model requested, it is called with the model id BEFORE the
    request is built, so the daily budget sub-cap can refuse (by raising)
    without this module knowing anything about budgets. None means no
    guard is installed (unit tier); #22 installs the real one.
    """

    model: str = GENERATION_MODEL_DEFAULT
    max_tokens: int = GENERATION_MAX_TOKENS_DEFAULT
    best_mode_enabled: bool = False
    budget_guard: Callable[[str], None] | None = None


@dataclass(frozen=True)
class CitedPassage:
    """One supplied passage a returned citation resolved onto.

    Carries the passage's provenance intact — including the #143
    ``parse_backend`` / ``degraded_fallback`` / ``needs_hand_review``
    flags, so a degraded-parse source is visible on the answer surface,
    and ``clears_threshold`` so partial support (§3.5) stays namable.
    """

    chunk_id: str
    document_index: int
    cited_text: str
    rerank_score: float
    clears_threshold: bool
    degraded_fallback: bool
    needs_hand_review: bool
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedAnswer:
    """The resolved result of one grounded generation call.

    ``usage`` is the adapter-reported token accounting (finding #64),
    cache metadata included — the channel #22's structured logs and
    spend accounting read. ``footer`` is the §3.5 verification note +
    corpus vintage, built by :func:`build_response_footer`.
    """

    text: str
    cited_passages: tuple[CitedPassage, ...]
    footer: str
    usage: Mapping[str, int] | None = None
    partial_support: bool = False
    tone_flag: bool = False


def load_system_prompt() -> str:
    """Read the committed system-prompt artifact verbatim.

    Never assembled from code fragments: the artifact IS the prompt, so
    a prompt change is a reviewable diff of one file (and invalidates
    replay recordings by design).
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def estimate_tokens_lower_bound(text_value: str) -> int:
    """A conservative LOWER bound on the Anthropic token count of `text_value`.

    Used only for the cache-floor check: it must UNDER-estimate, so that
    ``estimate_tokens_lower_bound(prefix) >= 4096`` guarantees the real
    tokenised prefix clears Haiku 4.5's floor. (The spike-03 cost
    heuristic over-estimates by design — the opposite direction — so it
    cannot be reused here.)
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def assert_cacheable_prefix(system_blocks: Iterable[Mapping[str, Any]]) -> None:
    """Refuse loudly when the STATIC prefix cannot clear the 4096 floor.

    The static prefix is the leading run of stable system blocks (block
    0, the system prompt) — volatile blocks after it do not count toward
    the cacheable prefix. Raises :class:`GenerationContractError` naming
    the shortfall when the conservative lower bound is under
    :data:`HAIKU_MIN_CACHEABLE_PREFIX_TOKENS`; returns None otherwise.
    The live smoke check (`tests/integration/test_generation_live.py`)
    is the authoritative proof; this is the cheap always-on guard.
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def build_generation_request(
    retrieved: RetrievedPassages,
    question: str,
    *,
    config: GenerationConfig,
) -> dict[str, Any]:
    """Pure builder: the ``adapter.generate(**request)`` seam payload.

    Returns ``{"messages": ..., "documents": ..., "config": ...,
    "system": ...}``. See the module docstring for the pinned contract;
    ``tests/unit/test_generation_request_builders.py`` is the source of
    truth.
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def resolve_citations(
    answer: AnswerWithCitations,
    retrieved: RetrievedPassages,
) -> tuple[CitedPassage, ...]:
    """Map each returned citation's ``document_index`` onto the supplied passages.

    Guaranteed-vs-measured (DESIGN §3.3): this is the *guaranteed* half —
    every resolved citation provably points at text we supplied. An index
    outside the supplied document set raises
    :class:`GenerationContractError`; resolution never guesses.
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def generate_grounded_answer(
    adapter: ProviderAdapter,
    retrieval_result: RetrievedPassages | HonestRefusal,
    question: str,
    *,
    config: GenerationConfig,
    corpus_vintage: str,
) -> GroundedAnswer | HonestRefusal:
    """The answer path: RetrievedPassages -> one generate call -> GroundedAnswer.

    An :class:`HonestRefusal` input is returned unchanged with ZERO
    adapter calls (§3.5 — the refusal path never spends an LLM call).
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def build_response_footer(corpus_vintage: str) -> str:
    """Pure template: the §3.5 verification note + §10 corpus-vintage line.

    Must carry the vintage date itself ("… as of <corpus_vintage>") and
    the honest verification framing (citations resolve to retrieved
    text; entailment is measured, not guaranteed — DESIGN Appendix B).
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")


def answer_stream_to_sse(
    stream_events: Iterable[Mapping[str, Any]],
    *,
    corpus_vintage: str,
) -> Iterator[dict[str, Any]]:
    """Pure translation: Anthropic stream events -> the service's SSE events.

    Yields ``{"event": <name>, "data": <mapping>}`` dicts; the pinned
    vocabulary is ``text`` / ``citation`` / ``usage`` / ``footer``. See
    ``tests/unit/test_generation_streaming.py`` for the pinned ordering.
    """
    raise NotImplementedError("issue #12: red phase — implementer makes this pass")
