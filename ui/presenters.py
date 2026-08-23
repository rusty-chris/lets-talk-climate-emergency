"""The pure presenter surface (IMPLEMENTATION.md §1: "decisions live in
``ui/presenters.py``").

Issue #18 grew the presenter layer into cohesive pure modules —
``ui.sse_client`` (wire parsing + the transport seam), ``ui.render_model``
(the answer-view fold), ``ui.starter`` (landing page), ``ui.footer``
(ADR-018 chrome), ``ui.charts`` (inline chart views) — and this facade
keeps IMPLEMENTATION.md's single import surface true: the Streamlit
shell imports its decisions from HERE, and the shell-hygiene unit suite
pins that this surface exposes the whole pure contract. No logic lives
in this file.
"""

from __future__ import annotations

from ui.charts import ChartAccessibilityError, ChartView, chart_view_from_event
from ui.footer import (
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    STEWARD_CREDIT_TEXT,
    TRANSPARENCY_ROUTES,
    FooterInvariantError,
    PageFooter,
    StewardCredit,
    build_page_footer,
    render_footer_lines,
)
from ui.render_model import (
    LIKELIHOOD_TERMS,
    AnswerView,
    Badge,
    ChatPage,
    CitationChip,
    ErrorNotice,
    SourceEntry,
    StreamContractError,
    TermAnchor,
    UncitedFlag,
    build_citation_chips,
    calibrated_term_anchors,
    chat_page_model,
    chips_for_cached_citations,
    fold_chat_stream,
    source_list,
)
from ui.sse_client import (
    ChatTransport,
    SseProtocolError,
    parse_sse_stream,
    stream_chat_events,
)
from ui.starter import (
    SITE_NAME,
    SITE_NAME_SHORT,
    STARTER_GROUP_HEADINGS,
    STARTER_QUESTIONS,
    TAGLINE,
    ChatSubmission,
    LandingPage,
    StarterGroup,
    landing_page_model,
    starter_groups,
    starter_submission,
)

__all__ = [
    # sse_client
    "ChatTransport",
    "SseProtocolError",
    "parse_sse_stream",
    "stream_chat_events",
    # render_model
    "LIKELIHOOD_TERMS",
    "AnswerView",
    "Badge",
    "ChatPage",
    "CitationChip",
    "ErrorNotice",
    "SourceEntry",
    "StreamContractError",
    "TermAnchor",
    "UncitedFlag",
    "build_citation_chips",
    "calibrated_term_anchors",
    "chat_page_model",
    "chips_for_cached_citations",
    "fold_chat_stream",
    "source_list",
    # starter
    "SITE_NAME",
    "SITE_NAME_SHORT",
    "STARTER_GROUP_HEADINGS",
    "STARTER_QUESTIONS",
    "TAGLINE",
    "ChatSubmission",
    "LandingPage",
    "StarterGroup",
    "landing_page_model",
    "starter_groups",
    "starter_submission",
    # footer
    "NON_AFFILIATION_DISCLAIMER",
    "NONCOMMERCIAL_NOTE",
    "STEWARD_CREDIT_TEXT",
    "TRANSPARENCY_ROUTES",
    "FooterInvariantError",
    "PageFooter",
    "StewardCredit",
    "build_page_footer",
    "render_footer_lines",
    # charts
    "ChartAccessibilityError",
    "ChartView",
    "chart_view_from_event",
]
