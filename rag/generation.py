"""Grounded generation with Claude native citations (issue #12, DESIGN §3.3/§3.4).

The suites in ``tests/unit/test_generation_*.py`` and
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
- **Model policy.** A CLOSED vocabulary
  (:data:`ALLOWED_GENERATION_MODEL_FAMILIES`, finding #186), matched by
  family so dated snapshot ids gate identically to their family; unknown
  ids refuse. Default :data:`GENERATION_MODEL_DEFAULT` (claude-haiku-4-5);
  any allowed non-default family is the gated "best" mode (DESIGN
  §3.3/§9): requesting it without ``GenerationConfig.best_mode_enabled``
  raises :class:`BestModeNotEnabledError`; best mode with no
  ``budget_guard`` installed refuses fail-closed (ADR-015); when both are
  present the guard (the #22 sub-cap seam) is invoked with the exact
  model id before the request is built.
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
  ``cache_read_input_tokens``), and exactly one terminal event — the
  ``footer`` (:func:`build_response_footer`'s verification note + corpus
  vintage, DESIGN §3.5/§10 cadence) ONLY under a complete answer, a
  typed ``error`` event for provider errors, premature stream ends and
  max_tokens truncation (finding #184: the trust footer is never stamped
  on a half-answer). :func:`stream_grounded_answer` is the streamed twin
  of :func:`generate_grounded_answer` (finding #183): the same pure
  builder's request through ``adapter.generate_stream``, so the seam
  validator runs identically on both paths.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag.provider import AnswerWithCitations, ProviderAdapter
from rag.retrieval import GENERATION_TOP_K, HonestRefusal, RetrievedPassages

__all__ = [
    "GENERATION_MODEL_DEFAULT",
    "OPUS_BEST_MODEL",
    "ALLOWED_GENERATION_MODEL_FAMILIES",
    "GENERATION_MAX_TOKENS_DEFAULT",
    "TEXT_EVENT",
    "CITATION_EVENT",
    "USAGE_EVENT",
    "FOOTER_EVENT",
    "ERROR_EVENT",
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
    "stream_grounded_answer",
]

#: The #12 grounded-answer SSE event names, pinned as named constants so
#: the UI's wire-vocabulary parity guard binds the PRODUCER, not a
#: duplicated string literal in a fixture (review finding #230). The #22
#: service (meta/answer/chart) and #13 validator (badge/validation_degraded)
#: own the rest of the vocabulary; ``service.app.SSE_EVENT_NAMES`` is the
#: complete declared set.
TEXT_EVENT = "text"
CITATION_EVENT = "citation"
USAGE_EVENT = "usage"
FOOTER_EVENT = "footer"
ERROR_EVENT = "error"

#: DESIGN §3.3: generation default. Model id is config, never hard-coded
#: at a call site.
GENERATION_MODEL_DEFAULT = "claude-haiku-4-5"

#: DESIGN §3.3/§9: the optional "best" mode, behind the budget sub-cap
#: (#22). Never the default.
OPUS_BEST_MODEL = "claude-opus-4-8"

#: The ratified release-eval bake-off arm (review-21 #234 RATIFICATION):
#: a sanctioned non-default family so the eval drives the production
#: builder for the Sonnet arm instead of forking a thinner request. Gated
#: exactly like Opus — best mode + a budget guard.
SONNET_BAKEOFF_MODEL = "claude-sonnet-5"

#: The CLOSED model vocabulary (finding #186): the only families a
#: generation request may name. A model id is allowed when it equals a
#: family or is a dated snapshot of one (``<family>-<suffix>``, the id
#: form Anthropic actually publishes); anything else refuses before any
#: request exists. Any allowed non-default family is the gated "best"
#: mode: the gate is "model != default", never "model == opus constant",
#: so a snapshot alias can never slip past the sub-cap.
ALLOWED_GENERATION_MODEL_FAMILIES = (
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    SONNET_BAKEOFF_MODEL,
)


def _generation_model_family(model: str) -> str | None:
    """The allowlisted family a model id belongs to, or None."""
    for family in ALLOWED_GENERATION_MODEL_FAMILIES:
        if model == family or model.startswith(family + "-"):
            return family
    return None


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
    a gated (non-default) model requested, it is called with the model id
    BEFORE the request is built, so the daily budget sub-cap can refuse
    (by raising) without this module knowing anything about budgets.
    None means no guard is installed — fine for default-model traffic,
    but best mode with a None guard REFUSES fail-closed (ADR-015,
    finding #186); a tier that genuinely wants no budget behaviour
    passes an explicit no-op with a name that says so. #22 installs the
    real one.
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


#: The one tone-instruction block adversarial routing (§3.1 tone_flag) adds
#: to the VOLATILE region of the system channel — position 1, directly after
#: the static prefix, so tone-flagged traffic still shares the cached prefix
#: byte-for-byte. Fixed text (§4: answer calmly from evidence; never echo
#: framing); it interpolates nothing.
TONE_INSTRUCTION = (
    "Tone note for this reply: the question was routed as adversarial or "
    "bad-faith-adjacent. Answer with particular calm: correct any false "
    "premise politely with a citation, never echo the question's framing, "
    "never mock or lecture, and keep every rule above unchanged - evidence "
    "only, qualifiers verbatim, severity exactly as the sources state it."
)


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Read the committed system-prompt artifact verbatim.

    Never assembled from code fragments: the artifact IS the prompt, so
    a prompt change is a reviewable diff of one file (and invalidates
    replay recordings by design).

    Cached per process (finding #297, sibling of #290): the artifact is
    committed and process-invariant, so the ~4k-token file is read once,
    not on every ``/chat`` request. Equality with a fresh ``read_text`` is
    unchanged within a process.
    """
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


#: A run of four or more of the SAME non-alphanumeric character: table
#: separator rows, dividers, padding — the non-prose regions that
#: tokenise far denser than 4 chars/token (a 20+-char hyphen run is 1–3
#: tokens) and would make ``len // 4`` OVER-estimate (finding #190).
_REPEATED_NON_PROSE_RUN = re.compile(r"([^0-9A-Za-z])\1{3,}")


@lru_cache(maxsize=32)
def estimate_tokens_lower_bound(text_value: str) -> int:
    """A conservative LOWER bound on the Anthropic token count of `text_value`.

    Used only for the cache-floor check: it must UNDER-estimate, so that
    ``estimate_tokens_lower_bound(prefix) >= 4096`` guarantees the real
    tokenised prefix clears Haiku 4.5's floor. (The spike-03 cost
    heuristic over-estimates by design — the opposite direction — so it
    cannot be reused here.)

    Two steps keep the bound honest for prose AND non-prose (finding
    #190): first every run of ≥4 identical non-alphanumeric characters
    (markdown table separators, dividers, padding — regions that
    tokenise at ~7–22 chars/token) collapses to a single character, so
    they contribute ~nothing; then one token per four characters over
    what remains — English prose tokenises at roughly 3.5–4 characters
    per token on Anthropic models (spike-03 observed ~3.5 on real corpus
    text), so ``len // 4`` under-counts it. The committed artifact must
    clear the floor under THIS discounted estimate; the live cache smoke
    check (`tests/integration/test_generation_live.py`) is the
    authoritative proof either way.
    """
    prose_only = _REPEATED_NON_PROSE_RUN.sub(r"\1", text_value)
    return len(prose_only) // 4


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
    blocks = list(system_blocks)
    if not blocks:
        raise GenerationContractError(
            "system channel carries no blocks at all: there is no static prefix "
            f"to cache, and Haiku 4.5 caches nothing below "
            f"{HAIKU_MIN_CACHEABLE_PREFIX_TOKENS} prefix tokens - silently"
        )
    static_prefix_text = str(blocks[0].get("text", ""))
    lower_bound = estimate_tokens_lower_bound(static_prefix_text)
    if lower_bound < HAIKU_MIN_CACHEABLE_PREFIX_TOKENS:
        raise GenerationContractError(
            f"static system prefix is under Haiku 4.5's "
            f"{HAIKU_MIN_CACHEABLE_PREFIX_TOKENS}-token minimum cacheable prefix: "
            f"conservative lower bound {lower_bound} tokens "
            f"({len(static_prefix_text)} chars). Below the floor the API writes "
            "NO cache entry - silently (cache_creation_input_tokens: 0, no "
            "error) - and the 9 'less with prompt caching' cost assumption "
            "never fires. Grow the committed prompt artifact; never pad it "
            "with volatile content."
        )
    return None


def _document_block(passage: Any) -> dict[str, Any]:
    """One custom-content document block (the spike-03-proven shape).

    The passage BODY is the sole citable text; the context header and
    all traceability (chunk id, #143 parse-provenance flags) ride the
    block's ``context`` field — model-visible, never cited (the spike-03
    'context header leaks into cited_text' lesson, made contractual)."""
    payload = passage.payload
    context_lines = [
        f"chunk_id: {passage.chunk_id}",
        f"section: {payload.get('context_header', '')}",
        f"source_type: {payload.get('source_type', '')}",
        f"consensus_position: {payload.get('consensus_position', '')}",
        f"parse_backend: {payload.get('parse_backend', '')}",
        f"degraded_fallback: {payload.get('degraded_fallback', False)}",
        f"needs_hand_review: {payload.get('needs_hand_review', False)}",
    ]
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": payload["body"]}],
        },
        "title": payload["citation_metadata"]["attribution_text"],
        "context": "\n".join(context_lines),
        # §3.4 all-or-none, satisfied at construction: this builder cannot
        # emit an uncited block, so a mixed set is unbuildable here.
        "citations": {"enabled": True},
    }


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
    # §3.4 bound, enforced HERE before any request object exists: the ≤8
    # is our design rule (the API imposes none - spike-03 probe 3), so
    # the builder is the first line of defence, the seam validator the
    # backstop.
    passage_count = len(retrieved.passages)
    if passage_count > GENERATION_TOP_K:
        raise GenerationContractError(
            f"generation request would carry {passage_count} passages; DESIGN 3.4 "
            f"bounds the generation call to the reranked top-{GENERATION_TOP_K} - "
            "refusing before any request object exists"
        )

    # Model policy (DESIGN 3.3/9, finding #186): a CLOSED vocabulary
    # matched by family - never an open string equality. Unknown ids
    # refuse before any request object exists; any allowed non-default
    # family is the gated 'best' mode behind the budget sub-cap, and
    # best mode with no guard installed refuses fail-closed (ADR-015).
    model_family = _generation_model_family(config.model)
    if model_family is None:
        raise GenerationContractError(
            f"model id {config.model!r} is outside the generation allowlist "
            f"{ALLOWED_GENERATION_MODEL_FAMILIES} (exact id or dated snapshot "
            "of an allowed family): model id is config (DESIGN 3.3) but config "
            "drawn from a closed, gated vocabulary - refusing before any "
            "request object exists"
        )
    if model_family != GENERATION_MODEL_DEFAULT:
        if not config.best_mode_enabled:
            raise BestModeNotEnabledError(
                f"{config.model} is the gated 'best' mode (DESIGN 3.3/9, "
                "behind the budget sub-cap); requesting it without "
                "best_mode_enabled is a configuration bug"
            )
        if config.budget_guard is None:
            raise GenerationContractError(
                f"best mode is enabled for {config.model} but no budget_guard "
                "is installed: the budget cut-off fails CLOSED (ADR-015) - "
                "install the #22 sub-cap guard, or pass an explicit no-op "
                "guard where a tier genuinely wants no budget behaviour"
            )
        # The guard hook fires BEFORE the request is built so #22's
        # sub-cap can refuse by raising - with the EXACT id that will
        # reach the API.
        config.budget_guard(config.model)

    system: list[dict[str, Any]] = [{"type": "text", "text": load_system_prompt()}]
    if retrieved.tone_flag:
        # The ONE volatile block adversarial routing may add - strictly
        # after the static prefix, so the cacheable prefix is untouched.
        system.append({"type": "text", "text": TONE_INSTRUCTION})
    # The cheap always-on floor guard (the live smoke check is the proof).
    assert_cacheable_prefix(system)

    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
        "documents": [_document_block(passage) for passage in retrieved.passages],
        # Only model + max_tokens ever leave config for the seam: no
        # structured-output/tool key can exist because none is constructed.
        "config": {"model": config.model, "max_tokens": config.max_tokens},
        "system": system,
    }


def _passage_for_document_index(index: Any, retrieved: RetrievedPassages) -> Any:
    """The supplied passage a citation's ``document_index`` points at.

    THE citation-resolution check, shared by the folded path
    (:func:`resolve_citations`) and the streaming path
    (:func:`answer_stream_to_sse`) so the two surfaces cannot drift
    (finding #185). An index outside the supplied document set raises
    :class:`GenerationContractError`; resolution never guesses.
    """
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not (0 <= index < len(retrieved.passages))
    ):
        raise GenerationContractError(
            f"citation document_index {index!r} is outside the supplied document "
            f"set (0..{len(retrieved.passages) - 1}): resolution never guesses "
            "- this response cannot be trusted onto the answer surface"
        )
    return retrieved.passages[index]


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
    resolved: list[CitedPassage] = []
    for citation in answer.citations:
        index = citation.document_index
        passage = _passage_for_document_index(index, retrieved)
        payload = passage.payload
        resolved.append(
            CitedPassage(
                chunk_id=passage.chunk_id,
                document_index=index,
                cited_text=citation.cited_text,
                rerank_score=passage.rerank_score,
                clears_threshold=passage.clears_threshold,
                degraded_fallback=bool(payload.get("degraded_fallback", False)),
                needs_hand_review=bool(payload.get("needs_hand_review", False)),
                payload=payload,
            )
        )
    return tuple(resolved)


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
    if isinstance(retrieval_result, HonestRefusal):
        return retrieval_result

    # The payload that leaves IS the pure builder's output - no drift
    # between the tested builder and the request on the wire. Contract
    # violations raise inside the builder, before the adapter is touched.
    request = build_generation_request(retrieval_result, question, config=config)
    answer = adapter.generate(**request)

    return GroundedAnswer(
        text=answer.text,
        cited_passages=resolve_citations(answer, retrieval_result),
        footer=build_response_footer(corpus_vintage),
        usage=answer.usage,
        partial_support=retrieval_result.partial_support,
        tone_flag=retrieval_result.tone_flag,
    )


def build_response_footer(corpus_vintage: str) -> str:
    """Pure template: the §3.5 verification note + §10 corpus-vintage line.

    Must carry the vintage date itself ("… as of <corpus_vintage>") and
    the honest verification framing (citations resolve to retrieved
    text; entailment is measured, not guaranteed — DESIGN Appendix B).
    """
    return (
        "Every citation links to source text retrieved from a named, "
        "clearly-licensed document; how well each sentence is supported by its "
        "cited text is verified after generation and published as a measured "
        f"rate, not guaranteed. Answers reflect sources as of {corpus_vintage}."
    )


def answer_stream_to_sse(
    stream_events: Iterable[Mapping[str, Any]],
    *,
    retrieved: RetrievedPassages,
    corpus_vintage: str,
) -> Iterator[dict[str, Any]]:
    """Pure translation: Anthropic stream events -> the service's SSE events.

    Yields ``{"event": <name>, "data": <mapping>}`` dicts; the pinned
    vocabulary is ``text`` / ``citation`` / ``usage`` / ``footer`` /
    ``error``. See ``tests/unit/test_generation_streaming.py`` for the
    pinned ordering: ``text`` and ``citation`` events in transport
    arrival order, the ``usage`` event where the transport reported
    usage (``message_delta``), and exactly one terminal event closing
    every stream.

    Citation resolution at the seam (finding #185): every
    ``citations_delta`` is resolved against ``retrieved`` — the same
    passages the request was built from — through the SAME logic as the
    folded path's :func:`resolve_citations`, before emission. The
    streamed citation data carries the transport fields plus the
    resolved ``chunk_id``, the #143 ``degraded_fallback`` /
    ``needs_hand_review`` provenance flags and ``clears_threshold``. An
    out-of-range ``document_index`` never becomes a citation event: the
    stream terminates with an ``error`` event (the folded path raises
    for the same response) and no footer.

    Termination honesty (finding #184): the ``footer`` is the product's
    trust statement and is stamped ONLY under a complete answer —
    ``message_stop`` observed and the answer not cut off by the output
    budget. Every other ending is a terminal ``error`` event instead:

    - a transport ``error`` event (the provider's error TYPE surfaces as
      client-safe signal; its raw message never does);
    - a stream that ends before ``message_stop`` (a truncated answer);
    - ``stop_reason: max_tokens`` (transport-complete, answer truncated
      mid-claim — ``usage`` still surfaces first for spend accounting).
    """
    message_stopped = False
    stop_reason: str | None = None
    for event in stream_events:
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                yield {"event": TEXT_EVENT, "data": {"text": delta.get("text", "")}}
            elif delta_type == "citations_delta":
                citation = dict(delta.get("citation") or {})
                try:
                    passage = _passage_for_document_index(citation.get("document_index"), retrieved)
                except GenerationContractError:
                    yield {
                        "event": ERROR_EVENT,
                        "data": {
                            "type": "citation_resolution",
                            "message": (
                                "A citation in this answer could not be resolved "
                                "to the retrieved passages; this response cannot "
                                "be trusted and was stopped."
                            ),
                        },
                    }
                    return
                payload = passage.payload
                citation.update(
                    {
                        "chunk_id": passage.chunk_id,
                        "clears_threshold": passage.clears_threshold,
                        "degraded_fallback": bool(payload.get("degraded_fallback", False)),
                        "needs_hand_review": bool(payload.get("needs_hand_review", False)),
                    }
                )
                yield {"event": CITATION_EVENT, "data": citation}
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            if event.get("usage"):
                yield {"event": USAGE_EVENT, "data": dict(event["usage"])}
        elif event_type == "message_stop":
            message_stopped = True
        elif event_type == "error":
            reported = event.get("error") or {}
            yield {
                "event": ERROR_EVENT,
                "data": {
                    "type": str(reported.get("type") or "provider_error"),
                    "message": (
                        "The provider reported an error before the answer "
                        "completed; this response is incomplete."
                    ),
                },
            }
            return
    if not message_stopped:
        yield {
            "event": ERROR_EVENT,
            "data": {
                "type": "incomplete_stream",
                "message": (
                    "The answer stream ended before completion; this response is truncated."
                ),
            },
        }
        return
    if stop_reason == "max_tokens":
        yield {
            "event": ERROR_EVENT,
            "data": {
                "type": "truncated",
                "message": (
                    "The answer hit its output limit and is truncated; it is "
                    "not a complete response."
                ),
            },
        }
        return
    yield {"event": FOOTER_EVENT, "data": {"text": build_response_footer(corpus_vintage)}}


def stream_grounded_answer(
    adapter: ProviderAdapter,
    retrieval_result: RetrievedPassages | HonestRefusal,
    question: str,
    *,
    config: GenerationConfig,
    corpus_vintage: str,
) -> Iterator[dict[str, Any]] | HonestRefusal:
    """The streamed twin of :func:`generate_grounded_answer` (finding #183).

    RetrievedPassages -> ONE ``adapter.generate_stream`` call -> the
    service's SSE events, via :func:`answer_stream_to_sse`. The request
    is the same pure builder's output as the folded path, so the seam
    validator and the §3.4/§3.3 contract run identically on both; #22's
    SSE endpoint consumes this and never touches the provider directly.

    An :class:`HonestRefusal` input is returned unchanged with ZERO
    adapter calls (§3.5 — the refusal path never spends an LLM call).
    Contract violations raise here, before the adapter is touched.
    """
    if isinstance(retrieval_result, HonestRefusal):
        return retrieval_result

    request = build_generation_request(retrieval_result, question, config=config)
    return answer_stream_to_sse(
        adapter.generate_stream(**request),
        retrieved=retrieval_result,
        corpus_vintage=corpus_vintage,
    )
