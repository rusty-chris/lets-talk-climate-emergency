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
  order, carrying the attribution. (The honest #18 subset; the full
  §3.6 panel rides the #220 ``sources`` event below.)
- **Sources panel (issue #220).** A grounded stream's single
  ``sources`` event (after ``meta``, before the first ``text``) folds
  through :func:`build_sources_panel` into ``view.sources_panel`` — the
  §3.6/§7.2 retrieved-passages panel view model, grouped by
  ``source_type``: evidence entries in arrival order, ``voices``
  entries separated under the §2.1/§7.2 "About the movement" styling
  (:data:`VOICES_PANEL_HEADING`), each carrying its own
  ``attribution_text`` and the licence-bounded (possibly ``None`` —
  fail-closed, metadata-only) excerpt VERBATIM from the wire: the UI
  never fabricates, extends or paraphrases an excerpt. A stream with no
  sources event folds to ``sources_panel is None``; non-grounded kinds
  (refusal/canned/paused/cached) NEVER carry a panel, even if a
  protocol-breaching sources event appears.
- **Thumbs feedback (issue #56).** The meta event's ``exchange_id``
  (the feedback join key — it identifies the exchange, never the
  person) rides into ``view.exchange_id`` verbatim; a meta without the
  key (an older service) or with ``None`` (the paused non-starter
  furniture, which logs nothing) folds to ``None``.
  :func:`feedback_widget_model` decides — purely — which views carry
  the thumbs up/down widget: every COMPLETED answer view with a join
  key (grounded, chart, canned, refusal, cached starter — feedback on
  a cached answer is valid signal), never an errored or incomplete
  view, never a view without an ``exchange_id``.
  :func:`resolve_feedback_state` is the optimistic-display honesty
  rule: a successful POST shows the recorded verdict; a FAILED post
  shows the UNRECORDED state (no verdict displayed as selected, the
  pinned honest message) — never a fake success. The verdict
  vocabulary is imported from ``service.exchange_log`` (already a pure
  dependency of this module), so UI and service can never drift.
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
from pathlib import Path
from typing import Any

from rag.citation_validator import citation_sentence_assignments
from service.exchange_log import FEEDBACK_DOWN, FEEDBACK_UP, LOGGING_DISCLOSURE
from ui.charts import ChartAccessibilityError, ChartView, chart_view_from_event
from ui.footer import build_page_footer

#: The generation prompt's calibrated-vocabulary table is the single source
#: of truth for the likelihood legend's assessed ranges (finding #232); the
#: unit suite parses the SAME table so the legend cannot drift from it.
_GENERATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "rag" / "prompts" / "generation_system_prompt.md"
)

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
    "SOURCES_EVENT",
    "HANDLED_EVENTS",
    "IGNORED_EVENTS",
    "VOICES_PANEL_HEADING",
    "EVIDENCE_PANEL_HEADING",
    "SourcePanelEntry",
    "SourcesPanel",
    "build_sources_panel",
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
    "stream_text_delta",
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
    "LegendEntry",
    "likelihood_legend",
    "annotate_calibrated_terms",
    "FEEDBACK_UP",
    "FEEDBACK_DOWN",
    "FEEDBACK_UP_LABEL",
    "FEEDBACK_DOWN_LABEL",
    "FEEDBACK_STATE_RECORDED",
    "FEEDBACK_STATE_UNRECORDED",
    "FEEDBACK_RECORDED_MESSAGE",
    "FEEDBACK_NOT_RECORDED_MESSAGE",
    "FeedbackWidget",
    "FeedbackState",
    "feedback_widget_model",
    "resolve_feedback_state",
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
#: The #220 retrieved-passages surface (pinned equal to the service's
#: ``SOURCES_EVENT`` by the unit suite).
SOURCES_EVENT = "sources"

#: The COMPLETE event vocabulary this fold handles (review finding #230).
#: The unit suite pins ``HANDLED_EVENTS | IGNORED_EVENTS`` set-equal to the
#: service's declared emit-able vocabulary, BOTH directions — a service-side
#: event addition fails the UI suite until the UI decides handle-or-ignore.
#: ``sources`` (#220) is HANDLED: it folds into ``AnswerView.sources_panel``.
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
        SOURCES_EVENT,
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


#: §7.2: the heading the shell renders over voices passages — first-party
#: movement testimony styled apart from the assessed evidence (§2.1/§2.5).
VOICES_PANEL_HEADING = "About the movement"
#: The heading over the assessed-evidence group of the sources panel.
EVIDENCE_PANEL_HEADING = "Evidence"
#: The one ``source_type`` wire label that renders under the voices heading
#: (pinned equal to ``rag.retrieval.VOICES_SOURCE_TYPE``); any other label —
#: evidence or a forward-compat unknown — never masquerades as testimony.
VOICES_SOURCE_TYPE = "voices"


@dataclass(frozen=True)
class SourcePanelEntry:
    """One retrieved passage in the §3.6 sources panel (issue #220).

    Field-for-field the wire entry of the service's ``sources`` event:
    ``excerpt`` is the licence-bounded verbatim text (``None`` is the
    fail-closed metadata-only state — the UI shows attribution + deep
    link only, and NEVER fabricates or extends an excerpt);
    ``excerpt_truncated`` drives the display ellipsis; ``source_tier``
    is the display tier label (may be ``None`` on a fail-closed entry).
    """

    doc_id: str
    chunk_id: str
    title: str
    attribution: str
    canonical_url: str | None
    source_type: str
    source_tier: str | None
    permitted_context: str | None
    excerpt: str | None
    excerpt_truncated: bool


@dataclass(frozen=True)
class SourcesPanel:
    """The §3.6/§7.2 retrieved-passages panel view model (issue #220).

    ``evidence`` and ``voices`` partition the wire entries by
    ``source_type`` in arrival order: ``voices`` entries render under
    :data:`VOICES_PANEL_HEADING` (the §2.1 "About the movement"
    separation); every other entry — including a forward-compat unknown
    ``source_type``, which must never masquerade as movement testimony —
    renders under :data:`EVIDENCE_PANEL_HEADING`.
    """

    evidence: tuple[SourcePanelEntry, ...] = ()
    voices: tuple[SourcePanelEntry, ...] = ()


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
    #: The #220 §3.6 panel: built from the stream's single ``sources``
    #: event on grounded exchanges; ``None`` when no event arrived (an
    #: older service) and ALWAYS ``None`` on non-grounded kinds.
    sources_panel: SourcesPanel | None = None
    #: The #56 feedback join key from the meta event, verbatim: the id a
    #: thumbs click posts back. ``None`` when the wire carried none (an
    #: older service, or the paused non-starter furniture that logs no
    #: exchange) — with no key there is nothing to rate against.
    exchange_id: str | None = None


@dataclass(frozen=True)
class ChatPage:
    """The chat page chrome: the view plus the invariant furniture."""

    view: AnswerView
    disclosure: str
    footer: Any  # ui.footer.PageFooter — typed loosely to avoid an import cycle.


def build_citation_chips(
    transcript: Sequence[Mapping[str, Any]],
) -> tuple[CitationChip, ...]:
    """Chips keyed (sentence_index, document_index) via the #13 segmentation.

    The citation-to-sentence assignment is the validator's exported pairing
    (:func:`rag.citation_validator.citation_sentence_assignments`), NOT a
    hand-mirrored copy of its span-recovery rule (review finding #233): one
    rule, one source of truth, so a change to the validator's assignment can
    never silently drop a judged citation from the page. Each pair's
    quote/attribution/provenance ride the FIRST citation event carrying that
    document (chunk_id, title and the #143 flags are properties of the cited
    passage; the cited quote is first-arrival).
    """
    assignments = citation_sentence_assignments(transcript)

    # First-arrival citation-event data per document_index: the quote, the
    # attribution and the #143 provenance flags. The validator's pairing
    # decides WHICH (sentence, document) chips exist and in what order.
    metadata: dict[int, Mapping[str, Any]] = {}
    for event in transcript:
        if event.get("event") != CITATION_EVENT:
            continue
        data = event.get("data") or {}
        metadata.setdefault(data.get("document_index"), data)

    chips: list[CitationChip] = []
    for sentence_index, document_index in assignments:
        data = metadata.get(document_index)
        if data is None:
            continue
        chunk_id = data.get("chunk_id", "")
        title = data.get("document_title")
        chips.append(
            CitationChip(
                sentence_index=sentence_index,
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


def build_sources_panel(entries: Sequence[Mapping[str, Any]]) -> SourcesPanel:
    """Pure: the ``sources`` event's ``data["sources"]`` list -> the panel.

    RED-phase contract stub (issue #220); the failing suite in
    ``tests/unit/test_ui_sources_panel.py`` pins the contract:

    - One :class:`SourcePanelEntry` per wire entry, wire order preserved
      WITHIN each group; ``source_type == "voices"`` entries go to
      ``panel.voices`` (the §2.1 "About the movement" separation, each
      carrying its own voices ``attribution_text``); every other
      ``source_type`` — evidence today, any forward-compat unknown
      tomorrow — goes to ``panel.evidence`` (an unknown label must never
      be styled as movement testimony).
    - Field carriage is verbatim and honest: ``excerpt`` rides through
      untouched — ``None`` stays ``None`` (metadata-only display; the
      licensing wall is the SERVICE's; the UI never fabricates, extends
      or trims differently), ``excerpt_truncated`` rides through as a
      bool, ``attribution`` is the wire ``attribution_text`` and
      ``title`` the wire ``title`` (falling back to the attribution,
      then the chunk id, when blank).
    """
    evidence: list[SourcePanelEntry] = []
    voices: list[SourcePanelEntry] = []
    for entry in entries:
        attribution = entry.get("attribution_text")
        chunk_id = entry.get("chunk_id")
        title = entry.get("title") or attribution or chunk_id
        source_type = entry.get("source_type")
        panel_entry = SourcePanelEntry(
            doc_id=entry.get("doc_id"),
            chunk_id=chunk_id,
            title=title,
            attribution=attribution,
            canonical_url=entry.get("canonical_url"),
            source_type=source_type,
            source_tier=entry.get("source_tier"),
            permitted_context=entry.get("permitted_context"),
            excerpt=entry.get("excerpt"),
            excerpt_truncated=bool(entry.get("excerpt_truncated")),
        )
        # §2.1/§7.2: ONLY the exact "voices" label renders under "About the
        # movement"; any other (or unknown, forward-compat) label is evidence,
        # never dressed up as first-party movement testimony.
        (voices if source_type == VOICES_SOURCE_TYPE else evidence).append(panel_entry)
    return SourcesPanel(evidence=tuple(evidence), voices=tuple(voices))


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


def fold_chat_stream(
    events: Iterable[Mapping[str, Any]], *, chart_base_url: str = ""
) -> AnswerView:
    """Reduce one exchange's parsed SSE events into an :class:`AnswerView`.

    ``chart_base_url`` is plumbed straight into
    :func:`ui.charts.chart_view_from_event` so ``view.chart`` is the ONE
    renderable chart with absolute (off-Streamlit-origin) permalink/.csv/
    .svg/embed targets — the shell renders ``view.chart`` and never
    re-derives a divergent copy from the raw events (review finding #229).
    A chart event whose alt text is missing/blank folds to an honest error
    view (``chart is None``, ``complete is False``) instead of crashing the
    page mid-render; :func:`chart_view_from_event`'s own contract is
    unchanged (it still raises).
    """
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
    sources_entries: Sequence[Mapping[str, Any]] | None = None

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
        elif name == SOURCES_EVENT:
            # #220: the §3.6 retrieved-passages surface. Captured verbatim
            # here; it folds into a panel ONLY on a grounded exchange (below)
            # — a protocol-breaching sources event on a non-grounded kind is
            # never dressed up as grounding.
            sources_entries = data.get("sources", [])
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
            try:
                chart = chart_view_from_event(data, base_url=chart_base_url)
            except ChartAccessibilityError as exc:
                # A mute chart is an upstream bug; the fold degrades honestly
                # rather than crashing the public page mid-render (#229).
                error = ErrorNotice(error_type="chart_accessibility", message=str(exc))
                chart = None
                complete = False
            else:
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

    # #220: the panel rides ONLY grounded exchanges (never a refusal/canned/
    # paused/cached kind, even if a sources event breached the protocol); with
    # no sources event (an older service) it stays None rather than fabricated.
    sources_panel = (
        build_sources_panel(sources_entries)
        if kind == VIEW_KIND_GROUNDED and sources_entries is not None
        else None
    )

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
        sources_panel=sources_panel,
    )


def stream_text_delta(event: Mapping[str, Any]) -> str:
    """The live-streamable prose token from one parsed SSE event, else ``""``.

    Keeps the "which events carry displayable prose, and where" decision in
    the pure core (finding #233): the Streamlit shell tees the raw event
    stream through this instead of hardcoding the ``text`` event name and
    its data-field key. Only ``text`` events contribute prose; usage,
    citation, footer, badge and the rest never leak into the streamed body.
    """
    if event.get("event") != TEXT_EVENT:
        return ""
    return str((event.get("data") or {}).get("text", ""))


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
    notice = ErrorNotice(error_type="transport", message=message)
    events = list(partial_events)
    if events and events[0].get("event") == META_EVENT:
        # Fold whatever WAS delivered (the text prefix verbatim, chips from
        # arrived citations, NO badges — no badge events arrived), then mark
        # it honestly incomplete with the transport notice.
        base = fold_chat_stream(events)
        return replace(base, complete=False, error=notice, footer_text=None)
    # Zero delivered events (connect refused, an immediate 429): the wire
    # never delivered the meta event that normally carries the disclosure,
    # so synthesize it from the UI's own copy of the one-line notice
    # (ratified decision 1) rather than raise StreamContractError.
    return AnswerView(
        mode="live",
        kind=VIEW_KIND_GROUNDED,
        disclosure=LOGGING_DISCLOSURE,
        error=notice,
        complete=False,
    )


#: The pinned honesty lines (finding #224, ratified decision 2).
_INCOMPLETE_STREAM_LINE = "This answer may be incomplete — the stream ended early."
_ERROR_INCOMPLETE_LINE = "This answer is incomplete."


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
    if view.error is not None:
        # The error message for display, plus the exact incompleteness line
        # so the shell branches on nothing.
        return (view.error.message, _ERROR_INCOMPLETE_LINE)
    if not view.complete:
        return (_INCOMPLETE_STREAM_LINE,)
    return ()


# ---------------------------------------------------------------------------
# Thumbs feedback (issue #56): the widget model + the optimistic-display
# honesty rule. The verdict vocabulary is service.exchange_log's own
# (imported above) — one closed vocabulary, no UI-side copy to drift.
# ---------------------------------------------------------------------------

#: The visible thumb labels (wording DECISION flagged in the #56
#: red-phase notes): plain, honest, no free-text field in MVP.
FEEDBACK_UP_LABEL = "Helpful"
FEEDBACK_DOWN_LABEL = "Not helpful"

#: :attr:`FeedbackState.status` values.
FEEDBACK_STATE_RECORDED = "recorded"
FEEDBACK_STATE_UNRECORDED = "unrecorded"

#: The pinned honesty lines for the post-click state: a recorded rating
#: says so; a FAILED post says the rating was NOT recorded — the UI never
#: fakes a success it did not get.
FEEDBACK_RECORDED_MESSAGE = "Thanks — your rating was recorded."
FEEDBACK_NOT_RECORDED_MESSAGE = "Your rating wasn't recorded — please try again."


@dataclass(frozen=True)
class FeedbackWidget:
    """The thumbs up/down widget for one completed answer view.

    Carries the exchange's join key and the closed verdict vocabulary
    the shell posts back — the shell draws two buttons and POSTs
    ``(exchange_id, verdict)`` through the transport seam; every
    decision (which views get a widget, which verdicts exist, what the
    labels say) lives HERE, pure.
    """

    exchange_id: str
    up_verdict: str = FEEDBACK_UP
    down_verdict: str = FEEDBACK_DOWN
    up_label: str = FEEDBACK_UP_LABEL
    down_label: str = FEEDBACK_DOWN_LABEL


@dataclass(frozen=True)
class FeedbackState:
    """The post-click display state: what the visitor is told happened.

    ``verdict`` is the verdict to display as selected — ``None`` in the
    unrecorded state (a failed POST never shows a thumb as registered).
    """

    status: str
    verdict: str | None
    message: str


def feedback_widget_model(view: AnswerView) -> FeedbackWidget | None:
    """Pure: the thumbs widget for ``view``, or ``None`` (no widget).

    RED-phase contract stub (issue #56); the failing suite in
    ``tests/unit/test_ui_feedback.py`` pins the contract:

    - EVERY completed answer view with a join key carries the widget:
      grounded, chart, canned, refusal, and cached-starter kinds alike
      (a thumbs-down on a refusal or a cached paused answer is exactly
      the triage signal #56 exists to collect).
    - ``None`` when ``view.exchange_id`` is ``None`` (an older service,
      or the paused furniture that logged no exchange — nothing to
      rate against), when ``view.error`` is set, or when
      ``view.complete`` is False (an answer the visitor never fully
      received is not honestly rateable).
    - The widget's ``exchange_id`` is the view's, verbatim.
    """
    raise NotImplementedError


def resolve_feedback_state(verdict: str, post_succeeded: bool) -> FeedbackState:
    """Pure: the optimistic-display honesty rule for one thumbs click.

    RED-phase contract stub (issue #56); the failing suite in
    ``tests/unit/test_ui_feedback.py`` pins the contract:

    - ``verdict`` outside the service vocabulary (``FEEDBACK_UP`` /
      ``FEEDBACK_DOWN``) raises ``ValueError`` — the closed vocabulary
      holds on the UI side too.
    - ``post_succeeded`` True → ``(FEEDBACK_STATE_RECORDED, verdict,
      FEEDBACK_RECORDED_MESSAGE)``: the click is shown as registered.
    - ``post_succeeded`` False → ``(FEEDBACK_STATE_UNRECORDED, None,
      FEEDBACK_NOT_RECORDED_MESSAGE)``: the UNRECORDED state — no
      verdict displayed as selected, the honest message shown. A failed
      POST (404 on an expired exchange, 429, a dead network) is NEVER
      dressed up as a recorded rating.
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
    if cached is not None and cached[0] == pending_question:
        return ExchangeDecision(action=EXCHANGE_REPLAY, events=tuple(cached[1]))
    return ExchangeDecision(action=EXCHANGE_STREAM)


@dataclass(frozen=True)
class LegendEntry:
    """One likelihood-legend row: a calibrated term and its assessed range.

    ``assessed_probability`` is the probability wording from the table in
    ``rag/prompts/generation_system_prompt.md`` (e.g. ``"90–100%"``) —
    the prompt table is the single source of truth; the unit suite parses
    it so the legend cannot drift (review finding #232).
    """

    term: str
    assessed_probability: str


def likelihood_legend() -> tuple[LegendEntry, ...]:
    """The likelihood-scale legend model (§7.2/§7.3, review finding #232).

    RED-phase contract stub (review-18 fix wave): one entry per
    :data:`LIKELIHOOD_TERMS` member, in the vocabulary's order, each
    carrying the assessed probability wording from the generation
    prompt's calibrated-vocabulary table. This is the legend the
    calibrated-term markers reference — a reader who does not know
    "very likely" means >=90% gets told.
    """
    table = _parse_prompt_likelihood_table()
    return tuple(
        LegendEntry(term=term, assessed_probability=table[term]) for term in LIKELIHOOD_TERMS
    )


def _parse_prompt_likelihood_table() -> dict[str, str]:
    """Parse the calibrated-vocabulary table from the generation prompt.

    The prompt table is the single source of truth for each term's assessed
    probability wording (finding #232); this uses the SAME parse the unit
    suite uses, so the legend cannot drift from the prompt.
    """
    prompt = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")
    table: dict[str, str] = {}
    for line in prompt.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0] and "%" in cells[1]:
            table[cells[0]] = cells[1]
    return table


def annotate_calibrated_terms(text: str, anchors: Sequence[TermAnchor]) -> str:
    """Answer text with each anchored span visibly marked (finding #232).

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_render_model.py::TestCalibratedMarkup`` pin the
    contract: each anchored span is wrapped in the pinned markdown-bold
    marker (``**…**`` — DECISION flagged for ratification: bold is the
    Streamlit-renderable highlight the legend expander pairs with), every
    anchor exactly once, and ALL non-anchor text is byte-identical —
    markdown metacharacters in the surrounding answer are never escaped
    or corrupted by the annotation pass.
    """
    if not anchors:
        return text
    ordered = sorted(anchors, key=lambda anchor: anchor.start)
    parts: list[str] = []
    cursor = 0
    for anchor in ordered:
        # Non-anchor text is copied byte-for-byte; only the anchored span is
        # wrapped in the pinned markdown-bold marker (ratified decision 7).
        parts.append(text[cursor : anchor.start])
        parts.append(f"**{text[anchor.start : anchor.end]}**")
        cursor = anchor.end
    parts.append(text[cursor:])
    return "".join(parts)


def calibrated_term_anchors(text: str) -> tuple[TermAnchor, ...]:
    """Non-overlapping, longest-match-first calibrated-term spans in ``text``.

    Matches only at WORD BOUNDARIES (review finding #232, ratified decision
    7): a term abutting a letter on either side does not anchor
    ("blikely"/"unlikelyish" yield nothing), while punctuation-adjacent
    occurrences ("Very likely.", "(very likely)") still anchor — marking a
    mid-word match would label non-calibrated text as calibrated vocabulary.
    """
    lowered = text.lower()
    # Longest term first so "extremely unlikely" wins over "unlikely" and
    # "more likely than not" over "likely" at the same start.
    terms = sorted(LIKELIHOOD_TERMS, key=len, reverse=True)
    anchors: list[TermAnchor] = []
    position = 0
    length = len(lowered)
    while position < length:
        matched = False
        for term in terms:
            if not lowered.startswith(term, position):
                continue
            end = position + len(term)
            # Letter-adjacency on either side is not a word boundary.
            before_is_letter = position > 0 and lowered[position - 1].isalpha()
            after_is_letter = end < length and lowered[end].isalpha()
            if before_is_letter or after_is_letter:
                continue
            anchors.append(TermAnchor(term=term, start=position, end=end))
            position = end  # non-overlapping: skip past the match
            matched = True
            break
        if not matched:
            position += 1
    return tuple(anchors)
