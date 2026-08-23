"""The chat answer view model (issue #18): pure fold over the #22 SSE stream.

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_ui_render_model.py`` pins the contract.

Everything the chat surface decides lives here, pure (IMPLEMENTATION.md
§1: Streamlit files stay thin). :func:`fold_chat_stream` reduces one
chat exchange's parsed SSE events (``ui.sse_client``) into one
:class:`AnswerView`; the shell only draws what the view says.

## The pinned fold contract

- **Meta first.** The service guarantees the FIRST event is ``meta``
  (``{"disclosure", "preamble_note", "mode"}``); a stream that opens
  with anything else raises :class:`StreamContractError`. The
  disclosure line and preamble note ride into the view verbatim — the
  one-line logging disclosure is part of the chat surface (issue #22
  privacy contract), never dropped.
- **Streaming text accumulates in order**; ``usage`` events are spend
  accounting, not prose — they NEVER leak into the view text.
- **Citation chips are chip-honest.** Chips are keyed by
  ``(sentence_index, document_index)`` using the SAME segmentation as
  the #13 validator (``rag.citation_validator.segment_answer_sentences``
  over the identical transcript — single source of truth, so badge
  events land on exactly the chip whose citation they judged; finding
  #207's dedupe applies: several same-document citation events inside
  one sentence are ONE chip). Each chip carries its quote popover text
  (the transport's verbatim ``cited_text``, first arrival wins), its
  attribution (``document_title``, falling back to ``chunk_id``), and
  the #143 provenance flags (``clears_threshold``,
  ``degraded_fallback``, ``needs_hand_review``) untouched.
- **Badges attach to the right chip.** A ``badge`` event
  ``{sentence_index, document_index, reason}`` attaches to the single
  matching chip; ``document_index: None`` (an uncited factual
  sentence) becomes a sentence-level :class:`UncitedFlag` in the view —
  there is no chip to badge (#13 contract).
- **Error honesty.** A terminal ``error`` event yields a degraded view:
  ``complete=False``, the delivered text prefix ONLY (never fabricated
  or padded), an :class:`ErrorNotice` carrying the event's type and
  message for display, no footer. Badges are NEVER attached on an
  error-terminated stream (the #13 composer appends none; any that
  appear anyway are a protocol breach and are not attached).
- **Chip honesty under degraded validation.** A ``validation_degraded``
  event sets ``validation_degraded=True`` and the view shows chips
  WITHOUT badges — an unvalidated answer never wears verification
  marks, and never loses its citations either.
- **Completeness = footer.** ``complete`` is True only when the
  ``footer`` event (the §3.5 verification note + corpus vintage) was
  observed; its text is the view's footer.
- **Answer kinds.** ``answer`` events map by ``kind``: ``refusal`` /
  ``canned`` / ``paused`` views carry the service text with NO chips
  and NO sources (nothing is fabricated); ``cached_starter`` (the
  paused starter path) yields a clearly-dated view — ``generated_on``
  surfaced, the cached footer, and chips built from the cached
  citations via :func:`chips_for_cached_citations`. A ``chart`` event
  yields ``kind == "chart"`` with a ``ui.charts.ChartView``.
- **Forward compatibility.** Unknown event names are ignored (the #12
  vocabulary may grow; an old UI must not crash on a new event).
- **Source list.** :func:`source_list` derives the "Sources (n)"
  surface from the chips: one entry per distinct ``chunk_id``, arrival
  order, carrying the attribution. (The SSE stream carries no
  retrieved-passages event, so the §3.6 full top-8 panel is NOT
  representable from this contract — flagged for ratification in the
  red-phase notes.)
- **Calibrated-term anchors.** :func:`calibrated_term_anchors` finds
  the calibrated likelihood vocabulary in answer text (case-insensitive,
  longest match wins, no overlapping anchors) so the shell can tooltip
  each occurrence to the likelihood legend (§7.2). The term list must
  match the assessment vocabulary pinned in
  ``rag/prompts/generation_system_prompt.md``.

Wire-vocabulary constants are defined HERE (not imported from
``service.app``, which would drag FastAPI into the UI image); the unit
suite asserts they equal the service's — drift fails tests, not users.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ui.charts import ChartView

__all__ = [
    "META_EVENT",
    "ANSWER_EVENT",
    "CHART_EVENT",
    "TEXT_EVENT",
    "CITATION_EVENT",
    "USAGE_EVENT",
    "FOOTER_EVENT",
    "ERROR_EVENT",
    "BADGE_EVENT",
    "VALIDATION_DEGRADED_EVENT",
    "VIEW_KIND_GROUNDED",
    "VIEW_KIND_CHART",
    "VIEW_KIND_REFUSAL",
    "VIEW_KIND_CANNED",
    "VIEW_KIND_PAUSED",
    "VIEW_KIND_CACHED_STARTER",
    "LIKELIHOOD_TERMS",
    "StreamContractError",
    "Badge",
    "UncitedFlag",
    "CitationChip",
    "SourceEntry",
    "ErrorNotice",
    "TermAnchor",
    "AnswerView",
    "ChatPage",
    "fold_chat_stream",
    "build_citation_chips",
    "chips_for_cached_citations",
    "source_list",
    "chat_page_model",
    "calibrated_term_anchors",
]

#: The service SSE vocabulary the fold consumes — event names pinned
#: equal to the service's own constants by the unit suite.
META_EVENT = "meta"
ANSWER_EVENT = "answer"
CHART_EVENT = "chart"
TEXT_EVENT = "text"
CITATION_EVENT = "citation"
USAGE_EVENT = "usage"
FOOTER_EVENT = "footer"
ERROR_EVENT = "error"
BADGE_EVENT = "badge"
VALIDATION_DEGRADED_EVENT = "validation_degraded"

#: ``AnswerView.kind`` values. The last four are pinned equal to the
#: service's ``ANSWER_KIND_*`` wire values.
VIEW_KIND_GROUNDED = "grounded"
VIEW_KIND_CHART = "chart"
VIEW_KIND_REFUSAL = "refusal"
VIEW_KIND_CANNED = "canned"
VIEW_KIND_PAUSED = "paused"
VIEW_KIND_CACHED_STARTER = "cached_starter"

#: The assessed likelihood vocabulary (§7.2 tooltips -> likelihood
#: legend), matching the table in rag/prompts/generation_system_prompt.md.
LIKELIHOOD_TERMS: tuple[str, ...] = (
    "virtually certain",
    "extremely likely",
    "very likely",
    "likely",
    "more likely than not",
    "about as likely as not",
    "unlikely",
    "very unlikely",
    "extremely unlikely",
    "exceptionally unlikely",
)


class StreamContractError(Exception):
    """The event stream broke the pinned service contract (e.g. no meta first)."""


@dataclass(frozen=True)
class Badge:
    """One applied unverified badge (#13): why this chip's claim is unverified."""

    sentence_index: int
    document_index: int
    reason: str


@dataclass(frozen=True)
class UncitedFlag:
    """A sentence-level unverified marker: an uncited factual sentence (#13)."""

    sentence_index: int
    reason: str


@dataclass(frozen=True)
class CitationChip:
    """One citation chip: a (sentence, cited document) binding with its quote."""

    sentence_index: int
    document_index: int
    chunk_id: str
    quote: str
    attribution: str
    clears_threshold: bool = True
    degraded_fallback: bool = False
    needs_hand_review: bool = False
    badges: tuple[Badge, ...] = ()


@dataclass(frozen=True)
class SourceEntry:
    """One row of the "Sources (n)" list: a distinct cited chunk."""

    chunk_id: str
    attribution: str


@dataclass(frozen=True)
class ErrorNotice:
    """The honest degraded-view notice for an error-terminated stream."""

    error_type: str
    message: str


@dataclass(frozen=True)
class TermAnchor:
    """One calibrated-term occurrence in answer text: [start, end) span."""

    term: str
    start: int
    end: int


@dataclass(frozen=True)
class AnswerView:
    """Everything the chat surface renders for one exchange (see module docs)."""

    mode: str
    kind: str
    disclosure: str
    preamble_note: str | None = None
    text: str = ""
    chips: tuple[CitationChip, ...] = ()
    uncited_flags: tuple[UncitedFlag, ...] = ()
    footer_text: str | None = None
    complete: bool = False
    error: ErrorNotice | None = None
    validation_degraded: bool = False
    generated_on: str | None = None
    chart: ChartView | None = None
    sources: tuple[SourceEntry, ...] = field(default=())


@dataclass(frozen=True)
class ChatPage:
    """The chat page chrome: the view plus the invariant furniture."""

    view: AnswerView
    disclosure: str
    footer: Any  # ui.footer.PageFooter — typed loosely to avoid an import cycle.


def fold_chat_stream(events: Iterable[Mapping[str, Any]]) -> AnswerView:
    """Reduce one exchange's parsed SSE events into an :class:`AnswerView`."""
    raise NotImplementedError("issue #18 red phase: fold_chat_stream is not implemented yet")


def build_citation_chips(
    transcript: Sequence[Mapping[str, Any]],
) -> tuple[CitationChip, ...]:
    """Chips keyed (sentence_index, document_index) via the #13 segmentation."""
    raise NotImplementedError("issue #18 red phase: build_citation_chips is not implemented yet")


def chips_for_cached_citations(
    citations: Sequence[Mapping[str, Any]],
) -> tuple[CitationChip, ...]:
    """Arrival-order chips from a cached starter entry's citation mappings."""
    raise NotImplementedError(
        "issue #18 red phase: chips_for_cached_citations is not implemented yet"
    )


def source_list(chips: Sequence[CitationChip]) -> tuple[SourceEntry, ...]:
    """The deduped (by chunk_id, arrival order) "Sources (n)" entries."""
    raise NotImplementedError("issue #18 red phase: source_list is not implemented yet")


def chat_page_model(view: AnswerView) -> ChatPage:
    """The chat page: the view plus the disclosure line and the ADR-018 footer."""
    raise NotImplementedError("issue #18 red phase: chat_page_model is not implemented yet")


def calibrated_term_anchors(text: str) -> tuple[TermAnchor, ...]:
    """Non-overlapping, longest-match-first calibrated-term spans in ``text``."""
    raise NotImplementedError("issue #18 red phase: calibrated_term_anchors is not implemented yet")
