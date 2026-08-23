"""Runtime citation-support validator (issue #13, DESIGN §3.3).

RED-phase contract stubs: every behaviour raises ``NotImplementedError``;
the failing suites in ``tests/unit/test_citation_validator_*.py`` and
``tests/integration/test_citation_validator_*.py`` pin the contract below.

DESIGN §3.3 runtime/eval split, made code: after generation completes,
ONE batched Haiku call checks every factual-sentence -> cited-block pair
for entailment; sentences failing entailment get an "unverified" badge
applied AFTER the stream finishes; there is NO runtime regeneration and
no per-sentence call fan-out (the §9 cost model prices this stage at
~$0.003/query — a single batched call, or zero calls). Failure rates are
driven down offline by the eval harness (#21); this stage only *flags*.

Contract points the red suite pins:

- **Sentence segmentation over the delivered SSE transcript.**
  :func:`segment_answer_sentences` consumes the #12 SSE event sequence
  (``text`` / ``citation`` / ``usage`` / ``footer`` events, transport
  order — the exact artifact ``rag.generation.answer_stream_to_sse``
  produced and the service already holds) and yields ordered
  :class:`AnswerSentence` units. Sentence boundaries follow the
  ingestion tokenizer conventions (``ingestion.chunk.split_sentences``
  — the repo's one documented sentence-boundary rule; decimals and
  in-sentence numbers never split). Each ``citation`` event attaches to
  the sentence in progress when it arrived — the sentence containing
  the last non-whitespace text character emitted before the event,
  exactly how the transport interleaves citations after the text they
  cite. Non-factual furniture is excluded from validation
  (``factual=False``): interactional furniture (greetings, closings)
  and the §3.5 footer — the ``footer`` event never contributes sentence
  text, and the footer template text is never a factual sentence even
  if it appears in a text delta. A sentence carrying a citation is
  ALWAYS factual: the model attached evidence to it, so it gets
  verified.
- **Pairing.** :func:`build_entailment_pairs` maps every cited factual
  sentence onto its cited block(s) — one :class:`EntailmentPair` per
  (sentence, cited block), block text drawn from the resolved
  ``GroundedAnswer.cited_passages`` payload bodies (the same passages
  the generation call cited; ``document_index`` is the join key).
  Uncited factual sentences produce NO pair — they are auto-flagged
  ``uncited`` without spending entailment budget on them. Furniture
  produces nothing.
- **ONE batched structured call.** :func:`build_validation_request` is a
  pure builder producing a single ``ProviderAdapter.structured``
  payload carrying ALL pairs — never one call per sentence (the cost
  pin: exactly one adapter call regardless of sentence count, and the
  validator NEVER makes a ``generate`` call — no runtime
  regeneration). The request follows the seam conventions: instructions
  on the dedicated top-level ``system`` channel (finding #91), config
  carrying ONLY model + max_tokens (never citations — §3.4 /
  IMPLEMENTATION §4.3), deterministic and canonically hashable so
  replay fixtures can key on it. The schema demands exactly one
  verdict per pair (``verdicts``: array of ``{pair_index, supported}``,
  ``supported`` boolean, min/maxItems == pair count).
- **Verdict parsing with the malformed-retry-once convention**
  (#10/#16): :func:`parse_validation_output` raises
  :class:`MalformedValidationOutputError` on anything outside the
  schema (missing/extra/duplicate/unknown pair indices, non-boolean
  verdicts, missing ``verdicts``); :func:`validate_exchange` retries
  the IDENTICAL request once, and totals usage across both calls
  (finding #92) so spend accounting never under-reports a retried run.
- **Every factual sentence is accounted for** (review finding #191):
  the outcome partitions the factual sentences exactly — each one is
  either paired with its citations for entailment-checking or
  auto-flagged ``uncited``; none silently falls through. The
  whole-answer zero-citation case (a GroundedAnswer that cites
  nothing — the #191 gap) is validated like any other: zero adapter
  calls, every factual sentence flagged ``uncited``, support rate 0.0.
- **Support semantics.** A factual sentence is *supported* when at
  least one of its cited blocks entails it (any-entailment rule); a
  cited sentence failing ALL its pairs, and every uncited factual
  sentence, is *unverified*. Badges are chip-honest: EVERY failing
  pair yields a badge event referencing (sentence_index,
  document_index) even when a sibling block supports the sentence —
  the chip's citation does not support the claim, so the chip says so
  — while the aggregate rate counts the sentence as supported.
  ``support_rate = supported factual sentences / factual sentences``,
  None when there are no factual sentences.
- **Post-stream badge flow.** :func:`append_validation_events` composes
  the validator onto the delivered SSE stream: every source event is
  yielded through unchanged and immediately (first-token latency is
  NEVER blocked — the validator runs strictly after the final source
  event, footer included, has been delivered), then
  :func:`validation_sse_events` extends the pinned #12 vocabulary with
  ``badge`` events (``{"event": "badge", "data": {"sentence_index",
  "document_index", "reason"}}``; ``document_index`` None for uncited
  flags) — the #18 UI contract — or ONE typed
  ``validation_degraded`` event when validation failed. The badge flow
  composes with error-terminated streams (findings #183/#185: the SSE
  vocabulary is gaining error/termination events): a source stream
  carrying an ``error`` event delivered an incomplete answer, so the
  validator is NEVER invoked for it and no badge or degradation events
  are appended — the source events pass through unchanged and the
  stream ends where it ended. Beyond "badges strictly after the final
  source event", no exact event ordering of the source stream is
  assumed.
- **Failure containment.** The answer was already streamed; validator
  errors NEVER break it. :func:`validate_exchange` catches adapter and
  parse failures (transport errors, double-malformed output) and
  returns a degraded :class:`ValidationOutcome` (``validated=False``,
  ``degraded_reason`` naming the error) instead of raising; the
  exchange is logged as unvalidated and the delivered answer stands.
- **Skips (zero adapter calls).** Only a ``GroundedAnswer`` is ever
  validated: refusals (:class:`rag.retrieval.HonestRefusal`), canned
  responses and chart-route results (#10 ``QueryDecision``), and any
  other result type short-circuit to a skipped outcome
  (:data:`SKIPPED_NOT_VALIDATABLE`) with zero calls, as does a
  disabled validator (:data:`SKIPPED_DISABLED` — the enable flag is
  config). An answer with no entailment pairs (all-uncited, or
  furniture-only) also makes zero calls: uncited flags are pure.
- **The exchange record** (the #22/#56 consumption seam):
  :func:`exchange_log_record` is pure over the outcome and yields the
  structured-log fields — aggregate ``citation_support_rate`` (feeds
  the published metric), factual-sentence count, unverified indices,
  validated/skipped/degraded status, model, usage, and ``cost_usd``
  computed by :func:`evals.pricing.estimate_cost_usd` (the ledger's
  single pricing source; None without usage).
- **Perf evidence.** :func:`record_badge_latency` appends one CSV row
  per measured run (the #176 convention: a persistent home under
  ``evals/perf/``, default :data:`BADGE_LATENCY_LOG_FILENAME`), and the
  ~2 s badge budget (:data:`BADGE_LATENCY_BUDGET_SECONDS`) is asserted
  only on the demo hardware profile (issue #11 convention).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.provider import ProviderAdapter

__all__ = [
    "VALIDATOR_MODEL_DEFAULT",
    "VALIDATOR_MAX_TOKENS_DEFAULT",
    "BADGE_EVENT",
    "VALIDATION_DEGRADED_EVENT",
    "UNVERIFIED_REASON_ENTAILMENT",
    "UNVERIFIED_REASON_UNCITED",
    "SKIPPED_NOT_VALIDATABLE",
    "SKIPPED_DISABLED",
    "BADGE_LATENCY_BUDGET_SECONDS",
    "BADGE_LATENCY_LOG_FILENAME",
    "ValidatorConfig",
    "AnswerSentence",
    "EntailmentPair",
    "PairVerdict",
    "UnverifiedSentence",
    "ValidationOutcome",
    "MalformedValidationOutputError",
    "segment_answer_sentences",
    "build_entailment_pairs",
    "build_validation_request",
    "parse_validation_output",
    "validate_exchange",
    "validation_sse_events",
    "append_validation_events",
    "exchange_log_record",
    "record_badge_latency",
]

#: DESIGN §3.3: the batched entailment check is a Haiku call. Model id is
#: config, never hard-coded at a call site.
VALIDATOR_MODEL_DEFAULT = "claude-haiku-4-5"

#: Output budget for the batched verdict call (§9 / dev-cost-plan: ~300
#: verdict tokens for a typical answer; bounded well under the answer's
#: own budget — the validator emits verdicts, not prose).
VALIDATOR_MAX_TOKENS_DEFAULT = 512

#: The SSE event vocabulary extension (#12 pinned text/citation/usage/
#: footer; #13 adds these two, both strictly post-stream).
BADGE_EVENT = "badge"
VALIDATION_DEGRADED_EVENT = "validation_degraded"

#: Why a sentence is unverified (badge event ``reason`` values).
UNVERIFIED_REASON_ENTAILMENT = "entailment_failed"
UNVERIFIED_REASON_UNCITED = "uncited"

#: ``ValidationOutcome.skipped_reason`` values.
SKIPPED_NOT_VALIDATABLE = "not_validatable_result"
SKIPPED_DISABLED = "validator_disabled"

#: Issue #13 acceptance: badge application after stream end targets ~2 s
#: on the demo hardware profile; recorded every run, asserted only there.
BADGE_LATENCY_BUDGET_SECONDS = 2.0

#: The perf log's default basename under evals/perf/ (finding #176: a
#: persistent, documented home — never a temp dir a test deletes).
BADGE_LATENCY_LOG_FILENAME = "badge-latency.csv"


class MalformedValidationOutputError(Exception):
    """The validator's structured output failed schema validation.

    The defined failure path (IMPLEMENTATION.md §4.3, the #10/#16
    convention): malformed output is retried once through the adapter
    with the IDENTICAL request; a second malformed response surfaces as
    a degraded :class:`ValidationOutcome` — never a bare
    ``KeyError``/``ValueError`` crash, and never a broken answer (the
    answer was already streamed).
    """


@dataclass(frozen=True)
class ValidatorConfig:
    """Injected validator configuration (model id + enable flag are config)."""

    model: str = VALIDATOR_MODEL_DEFAULT
    max_tokens: int = VALIDATOR_MAX_TOKENS_DEFAULT
    enabled: bool = True


@dataclass(frozen=True)
class AnswerSentence:
    """One segmented sentence of the delivered answer.

    ``index`` is the sentence's 0-based position among ALL segmented
    sentences in delivery order (furniture included) — the stable
    reference badge events and the #18 UI use to locate the sentence.
    ``document_indices`` are the cited blocks whose citation events
    attached to this sentence, in arrival order; empty means uncited.
    """

    index: int
    text: str
    document_indices: tuple[int, ...] = ()
    factual: bool = True


@dataclass(frozen=True)
class EntailmentPair:
    """One (factual sentence, cited block) pair for the batched check.

    ``pair_index`` is the pair's 0-based position in the batched
    request — the join key the output schema's verdicts carry back.
    """

    pair_index: int
    sentence_index: int
    sentence_text: str
    document_index: int
    block_text: str


@dataclass(frozen=True)
class PairVerdict:
    """One parsed per-pair verdict from the batched structured output."""

    pair_index: int
    sentence_index: int
    document_index: int
    supported: bool


@dataclass(frozen=True)
class UnverifiedSentence:
    """One unverified flag: a failing pair, or an uncited factual sentence.

    ``document_index`` is the failing pair's cited block for
    ``entailment_failed`` flags, None for ``uncited`` flags (there is
    no chip to badge — the #18 UI renders a sentence-level marker).
    """

    sentence_index: int
    document_index: int | None
    reason: str


@dataclass(frozen=True)
class ValidationOutcome:
    """The result of running (or skipping) validation for one exchange.

    Exactly one of three states: validated (the batched call ran and
    parsed — or no pairs existed and the pure uncited flags stand),
    skipped (``skipped_reason`` set, zero calls), or degraded
    (``degraded_reason`` set — the validator failed; the answer
    stands, the exchange is logged as unvalidated).
    """

    validated: bool
    sentences: tuple[AnswerSentence, ...] = ()
    verdicts: tuple[PairVerdict, ...] = ()
    unverified: tuple[UnverifiedSentence, ...] = ()
    support_rate: float | None = None
    usage: Mapping[str, int] | None = None
    model: str | None = None
    skipped_reason: str | None = None
    degraded_reason: str | None = None


def segment_answer_sentences(
    sse_events: Sequence[Mapping[str, Any]],
) -> tuple[AnswerSentence, ...]:
    """Pure: the delivered #12 SSE transcript -> ordered AnswerSentences.

    Only ``text`` events contribute sentence text; ``citation`` events
    attach their ``document_index`` to the sentence in progress when
    they arrived; ``usage``/``footer`` events contribute nothing.
    Sentence boundaries follow ``ingestion.chunk.split_sentences``
    conventions; furniture (greetings/closings, the footer template
    text) is marked ``factual=False`` unless cited.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def build_entailment_pairs(
    sentences: Sequence[AnswerSentence],
    cited_passages: Sequence[Any],
) -> tuple[EntailmentPair, ...]:
    """Pure: cited factual sentences x their cited blocks -> pairs.

    ``cited_passages`` is ``GroundedAnswer.cited_passages`` (the
    resolved citations); each pair's ``block_text`` is the cited
    passage's stored ``payload["body"]``, joined on ``document_index``.
    Uncited factual sentences and furniture produce no pair.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def build_validation_request(
    pairs: Sequence[EntailmentPair],
    *,
    config: ValidatorConfig,
) -> dict[str, Any]:
    """Pure builder: ALL pairs -> ONE ``adapter.structured`` payload.

    Returns ``{"messages", "system", "schema", "config"}`` matching
    ``ProviderAdapter.structured``: instructions on the dedicated
    top-level ``system`` channel (finding #91), every pair's sentence
    and block text in ``messages`` (auditable), a schema demanding
    exactly one ``{pair_index, supported}`` verdict per pair, and
    ``config`` carrying ONLY ``{"model": config.model, "max_tokens":
    config.max_tokens}`` — never citations configuration (§3.4).
    Deterministic and canonically hashable (replay fixtures key on it).
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def parse_validation_output(
    raw: Mapping[str, Any],
    pairs: Sequence[EntailmentPair],
) -> tuple[PairVerdict, ...]:
    """Pure: validate the structured output into per-pair verdicts.

    Raises :class:`MalformedValidationOutputError` (naming the
    offending field) on missing ``verdicts``, wrong types, non-boolean
    ``supported``, or verdicts that fail to cover every pair exactly
    once (missing, extra, duplicate, or unknown ``pair_index``).
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def validate_exchange(
    adapter: ProviderAdapter,
    result: Any,
    sse_events: Sequence[Mapping[str, Any]] = (),
    *,
    config: ValidatorConfig,
) -> ValidationOutcome:
    """The validation entry point: one exchange -> one ValidationOutcome.

    Only a ``GroundedAnswer`` is validated; every other result type
    (HonestRefusal, canned/chart QueryDecisions, chart artefacts)
    returns a skipped outcome with ZERO adapter calls, as does
    ``config.enabled`` False. With pairs: exactly ONE ``structured``
    call (plus at most one identical retry on malformed output — the
    #10/#16 convention; usage totalled across both, finding #92) and
    NEVER a ``generate`` call. Without pairs: zero calls, pure uncited
    flags. Validator failures degrade — this function does not raise
    for adapter/parse errors; the answer already streamed and stands.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def validation_sse_events(outcome: ValidationOutcome) -> Iterator[dict[str, Any]]:
    """Pure: a ValidationOutcome -> the post-stream SSE events.

    One :data:`BADGE_EVENT` per unverified flag (``sentence_index``,
    ``document_index``, ``reason``), in sentence order; ONE
    :data:`VALIDATION_DEGRADED_EVENT` for a degraded outcome; nothing
    for a skipped or fully-supported outcome.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def append_validation_events(
    sse_events: Iterable[Mapping[str, Any]],
    validate: Callable[[Sequence[Mapping[str, Any]]], ValidationOutcome],
) -> Iterator[dict[str, Any]]:
    """Lazy composition: the delivered stream, then the validator's events.

    Yields every source event through unchanged and immediately —
    stream start NEVER waits on validation — and invokes ``validate``
    (with the collected transcript) only after the source iterator is
    exhausted, i.e. strictly after the final source event was
    delivered; then yields :func:`validation_sse_events` of the
    outcome. A source stream carrying an ``error`` event (the
    #183/#185 termination vocabulary) delivered an incomplete answer:
    ``validate`` is never invoked for it and nothing is appended.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def exchange_log_record(outcome: ValidationOutcome) -> dict[str, Any]:
    """Pure: the structured exchange-record fields (#22/#56 seam).

    Keys: ``citation_support_rate``, ``factual_sentence_count``,
    ``unverified_sentence_indices``, ``validated``, ``skipped_reason``,
    ``degraded_reason``, ``model``, ``usage`` and ``cost_usd``
    (computed by ``evals.pricing.estimate_cost_usd`` from the
    outcome's model + usage; None when usage is None — the ledger
    never records a silent $0 row).
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")


def record_badge_latency(
    log_path: Path | None = None,
    *,
    pair_count: int,
    wall_clock_seconds: float,
    hardware_profile: str,
) -> Mapping[str, Any]:
    """Append one badge-latency measurement to the perf log (CSV).

    The #176 convention (mirrors ``rag.retrieval.record_rerank_latency``):
    ``log_path`` defaults to a persistent home under ``evals/perf/``
    (:data:`BADGE_LATENCY_LOG_FILENAME`); creates the file with a
    header when absent, appends otherwise; every record carries at
    least ``pair_count``, ``wall_clock_seconds``, ``budget_seconds``
    (== :data:`BADGE_LATENCY_BUDGET_SECONDS`), ``within_budget`` and
    ``hardware_profile``; the written record is returned. The budget
    itself is asserted only on the demo hardware profile.
    """
    raise NotImplementedError("issue #13: red phase — implementer makes this pass")
