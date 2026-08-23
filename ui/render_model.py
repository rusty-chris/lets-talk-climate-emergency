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

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from rag.citation_validator import segment_answer_sentences
from ui.charts import ChartView, chart_view_from_event
from ui.footer import build_page_footer

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
    "HANDLED_EVENTS",
    "IGNORED_EVENTS",
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
    "transport_failure_view",
    "answer_status_lines",
    "EXCHANGE_REPLAY",
    "EXCHANGE_STREAM",
    "ExchangeDecision",
    "resolve_exchange",
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

#: The COMPLETE event vocabulary this fold handles (review finding #230).
#: The unit suite pins ``HANDLED_EVENTS | IGNORED_EVENTS`` set-equal to the
#: service's declared emit-able vocabulary, BOTH directions — a service-side
#: event addition fails the UI suite until the UI decides handle-or-ignore.
HANDLED_EVENTS: frozenset[str] = frozenset(
    {
        META_EVENT,
        ANSWER_EVENT,
        CHART_EVENT,
        TEXT_EVENT,
        CITATION_EVENT,
        USAGE_EVENT,
        FOOTER_EVENT,
        ERROR_EVENT,
        BADGE_EVENT,
        VALIDATION_DEGRADED_EVENT,
    }
)

#: Service events the UI DELIBERATELY does not render — each membership is
#: a recorded decision, never an accident (finding #230). Empty today.
#: (Runtime forward-compat is separate: an old deployed UI still ignores
#: genuinely unknown names rather than crashing.)
IGNORED_EVENTS: frozenset[str] = frozenset()

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


def _sentence_index_of_citation(
    full_text: str, spans: Sequence[tuple[int, int]], emitted: int
) -> int | None:
    """Which sentence a citation arriving after ``emitted`` chars belongs to.

    Mirrors ``rag.citation_validator``'s assignment exactly (the transport
    emits a citation AFTER the text it cites, so it belongs to the sentence
    of the last non-whitespace character emitted before it) so a chip lands
    on the sentence the validator judged.
    """
    position = emitted - 1
    while position >= 0 and full_text[position].isspace():
        position -= 1
    if position < 0:
        return None
    fallback: int | None = None
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
        if start <= position:
            fallback = index
    return fallback


def build_citation_chips(
    transcript: Sequence[Mapping[str, Any]],
) -> tuple[CitationChip, ...]:
    """Chips keyed (sentence_index, document_index) via the #13 segmentation."""
    # The #13 validator is the single source of truth for sentence indices
    # and the (finding #207) per-sentence document dedupe: its output fixes
    # WHICH chips exist and in WHAT order.
    sentences = segment_answer_sentences(transcript)

    # Recover the sentence spans over the concatenated delivered text so each
    # citation event can be located exactly as the validator located it.
    full_text = "".join(
        str((event.get("data") or {}).get("text", ""))
        for event in transcript
        if event.get("event") == TEXT_EVENT
    )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        found = full_text.find(sentence.text, cursor)
        if found < 0:
            found = cursor
        spans.append((found, found + len(sentence.text)))
        cursor = found + len(sentence.text)

    # First arrival of each (sentence, document) wins its quote/attribution.
    metadata: dict[tuple[int, int], Mapping[str, Any]] = {}
    emitted = 0
    for event in transcript:
        name = event.get("event")
        data = event.get("data") or {}
        if name == TEXT_EVENT:
            emitted += len(str(data.get("text", "")))
        elif name == CITATION_EVENT:
            document_index = data.get("document_index")
            sentence_index = _sentence_index_of_citation(full_text, spans, emitted)
            if sentence_index is None:
                continue
            metadata.setdefault((sentence_index, document_index), data)

    # Emit chips in the validator's canonical order: sentences in order, each
    # sentence's documents in arrival order (already deduped by the validator).
    chips: list[CitationChip] = []
    for sentence in sentences:
        for document_index in sentence.document_indices:
            data = metadata.get((sentence.index, document_index))
            if data is None:
                continue
            chunk_id = data.get("chunk_id", "")
            title = data.get("document_title")
            chips.append(
                CitationChip(
                    sentence_index=sentence.index,
                    document_index=document_index,
                    chunk_id=chunk_id,
                    quote=data.get("cited_text", ""),
                    attribution=title or chunk_id,
                    clears_threshold=bool(data.get("clears_threshold", True)),
                    degraded_fallback=bool(data.get("degraded_fallback", False)),
                    needs_hand_review=bool(data.get("needs_hand_review", False)),
                )
            )
    return tuple(chips)


def chips_for_cached_citations(
    citations: Sequence[Mapping[str, Any]],
) -> tuple[CitationChip, ...]:
    """Arrival-order chips from a cached starter entry's citation mappings."""
    chips: list[CitationChip] = []
    for index, citation in enumerate(citations):
        chunk_id = citation.get("chunk_id", "")
        attribution = citation.get("attribution_text") or chunk_id
        chips.append(
            CitationChip(
                sentence_index=index,
                document_index=index,
                chunk_id=chunk_id,
                quote=citation.get("cited_text", ""),
                attribution=attribution,
            )
        )
    return tuple(chips)


def source_list(chips: Sequence[CitationChip]) -> tuple[SourceEntry, ...]:
    """The deduped (by chunk_id, arrival order) "Sources (n)" entries."""
    seen: set[str] = set()
    entries: list[SourceEntry] = []
    for chip in chips:
        if chip.chunk_id in seen:
            continue
        seen.add(chip.chunk_id)
        entries.append(SourceEntry(chunk_id=chip.chunk_id, attribution=chip.attribution))
    return tuple(entries)


def _apply_badges(
    chips: tuple[CitationChip, ...], badge_data: Sequence[Mapping[str, Any]]
) -> tuple[tuple[CitationChip, ...], tuple[UncitedFlag, ...]]:
    """Attach badges to their matching chips; None-document badges become flags."""
    by_chip: dict[tuple[int, int], list[Badge]] = defaultdict(list)
    flags: list[UncitedFlag] = []
    for data in badge_data:
        sentence_index = data.get("sentence_index")
        document_index = data.get("document_index")
        reason = data.get("reason", "")
        if document_index is None:
            # An uncited factual sentence (#13): no chip to badge — a
            # sentence-level marker instead.
            flags.append(UncitedFlag(sentence_index=sentence_index, reason=reason))
        else:
            by_chip[(sentence_index, document_index)].append(
                Badge(sentence_index=sentence_index, document_index=document_index, reason=reason)
            )
    badged = tuple(
        replace(
            chip,
            badges=tuple(by_chip.get((chip.sentence_index, chip.document_index), ())),
        )
        for chip in chips
    )
    return badged, tuple(flags)


def fold_chat_stream(events: Iterable[Mapping[str, Any]]) -> AnswerView:
    """Reduce one exchange's parsed SSE events into an :class:`AnswerView`."""
    events = list(events)
    if not events or events[0].get("event") != META_EVENT:
        raise StreamContractError("the chat stream must open with a meta event")

    meta = events[0].get("data") or {}
    mode = meta.get("mode", "live")
    disclosure = meta.get("disclosure", "")
    preamble_note = meta.get("preamble_note")

    kind = VIEW_KIND_GROUNDED
    text_parts: list[str] = []
    transcript: list[Mapping[str, Any]] = []
    badge_data: list[Mapping[str, Any]] = []
    footer_text: str | None = None
    complete = False
    error: ErrorNotice | None = None
    validation_degraded = False
    generated_on: str | None = None
    chart: ChartView | None = None
    cached_citations: Sequence[Mapping[str, Any]] = ()

    for event in events[1:]:
        if error is not None:
            # A terminal error ends the exchange: nothing after it is
            # attached (the #13 composer appends nothing post-error; a badge
            # that appears anyway is a protocol breach, never honoured).
            break
        name = event.get("event")
        data = event.get("data") or {}
        if name == TEXT_EVENT:
            text_parts.append(str(data.get("text", "")))
            transcript.append(event)
        elif name == CITATION_EVENT:
            transcript.append(event)
        elif name == USAGE_EVENT:
            # Spend accounting, never prose — it never leaks into the view.
            continue
        elif name == FOOTER_EVENT:
            footer_text = data.get("text")
            complete = True
        elif name == ERROR_EVENT:
            error = ErrorNotice(error_type=data.get("type", ""), message=data.get("message", ""))
            complete = False
            footer_text = None
        elif name == VALIDATION_DEGRADED_EVENT:
            validation_degraded = True
        elif name == BADGE_EVENT:
            badge_data.append(data)
        elif name == ANSWER_EVENT:
            kind = data.get("kind", VIEW_KIND_GROUNDED)
            text_parts = [str(data.get("text", ""))]
            # Terminal single-event answers are complete responses.
            complete = True
            if kind == VIEW_KIND_CACHED_STARTER:
                generated_on = data.get("generated_on")
                footer_text = data.get("footer")
                cached_citations = data.get("citations", ())
        elif name == CHART_EVENT:
            kind = VIEW_KIND_CHART
            chart = chart_view_from_event(data)
            complete = True
        # Unknown event names are ignored (forward compatibility).

    uncited_flags: tuple[UncitedFlag, ...] = ()
    if kind == VIEW_KIND_GROUNDED:
        chips = build_citation_chips(transcript)
        # An unvalidated or error-terminated answer wears NO verification
        # marks (and never loses its citations): badges/flags attach only
        # when validation ran cleanly to completion.
        if error is None and not validation_degraded:
            chips, uncited_flags = _apply_badges(chips, badge_data)
    elif kind == VIEW_KIND_CACHED_STARTER:
        chips = chips_for_cached_citations(cached_citations)
    else:
        chips = ()

    return AnswerView(
        mode=mode,
        kind=kind,
        disclosure=disclosure,
        preamble_note=preamble_note,
        text="".join(text_parts),
        chips=chips,
        uncited_flags=uncited_flags,
        footer_text=footer_text,
        complete=complete,
        error=error,
        validation_degraded=validation_degraded,
        generated_on=generated_on,
        chart=chart,
        sources=source_list(chips),
    )


def chat_page_model(view: AnswerView) -> ChatPage:
    """The chat page: the view plus the disclosure line and the ADR-018 footer."""
    return ChatPage(view=view, disclosure=view.disclosure, footer=build_page_footer())


def transport_failure_view(partial_events: Sequence[Mapping[str, Any]], message: str) -> AnswerView:
    """The honest view for a transport-level failure (review finding #224).

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_render_model.py::TestTransportFailure`` pin the
    contract:

    - Folds whatever teed events WERE delivered before the failure —
      the delivered text prefix is preserved verbatim, chips built from
      arrived citations, NO badges (same rule as an ``error`` event).
    - ``complete is False``, ``error.error_type == "transport"``, and
      the error message is the given human-honest ``message`` — never
      an exception repr, never a traceback.
    - Zero delivered events (connect refused, an immediate 429) yields
      a renderable error view, NOT :class:`StreamContractError`; its
      disclosure is the UI's own copy of the privacy line
      (``service.exchange_log.LOGGING_DISCLOSURE``) because the wire
      never delivered the meta event that normally carries it.
      DECISION flagged for ratification in the #224 red notes.
    """
    raise NotImplementedError


def answer_status_lines(view: AnswerView) -> tuple[str, ...]:
    """Pure honesty lines the shell renders unconditionally (finding #224).

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_render_model.py`` pin the contract:

    - ``complete=True`` and no error: ``()`` — nothing to flag.
    - ``complete=False`` and no error (a stream that simply ended
      early, no ``error`` event): exactly
      ``("This answer may be incomplete — the stream ended early.",)``.
    - An error view: the first line carries the error message for
      display, and the exact line ``"This answer is incomplete."``
      appears — so the shell has NO honesty decision of its own.
    """
    raise NotImplementedError


#: :attr:`ExchangeDecision.action` values (review finding #226).
EXCHANGE_REPLAY = "replay"
EXCHANGE_STREAM = "stream"


@dataclass(frozen=True)
class ExchangeDecision:
    """Replay the cached exchange, or open the transport (finding #226).

    ``action`` is :data:`EXCHANGE_REPLAY` (render from ``events``, NO
    transport call) or :data:`EXCHANGE_STREAM` (open ``POST /chat``).
    """

    action: str
    events: tuple[Mapping[str, Any], ...] = ()


def resolve_exchange(
    pending_question: str,
    cached: tuple[str, Sequence[Mapping[str, Any]]] | None,
) -> ExchangeDecision:
    """Pure replay-vs-stream decision for one Streamlit rerun (finding #226).

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_render_model.py::TestExchangeReplay`` pin the
    contract: Streamlit re-executes the script top-to-bottom on every
    rerun, and today each execution re-POSTs the pending question —
    re-spending the daily budget and silently replacing the rendered
    answer with a different generation. When ``cached`` holds
    ``(question, events)`` for the SAME pending question the decision is
    replay, carrying the cached events (folding them reproduces the
    original view exactly); a different question, or no cache, streams.
    """
    raise NotImplementedError


def calibrated_term_anchors(text: str) -> tuple[TermAnchor, ...]:
    """Non-overlapping, longest-match-first calibrated-term spans in ``text``."""
    lowered = text.lower()
    # Longest term first so "extremely unlikely" wins over "unlikely" and
    # "more likely than not" over "likely" at the same start.
    terms = sorted(LIKELIHOOD_TERMS, key=len, reverse=True)
    anchors: list[TermAnchor] = []
    position = 0
    length = len(lowered)
    while position < length:
        for term in terms:
            if lowered.startswith(term, position):
                anchors.append(TermAnchor(term=term, start=position, end=position + len(term)))
                position += len(term)  # non-overlapping: skip past the match
                break
        else:
            position += 1
    return tuple(anchors)
