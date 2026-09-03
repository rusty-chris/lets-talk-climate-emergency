"""The cached-answer view is honest and verbatim (issue #57, UI side).

RED suite over ``ui.render_model``: folding the service's single
``answer`` event of kind ``cached`` rebuilds EXACTLY the view the
original grounded stream produced — same chips with the same badges
(verification marks are never laundered by caching), same sources
panel, same footer — plus the honesty furniture: ``generated_on`` set
to the ORIGINAL answer's date and the pinned "Cached answer" notice.
The serving's fresh ``exchange_id`` rides the view so the #56 thumbs
widget works on cached answers too.
"""

from __future__ import annotations

import service.app as service_app
from tests._ui_fixtures import (
    answer_event,
    badge_event,
    citation_event,
    footer_event,
    meta_event,
    text_event,
    usage_event,
)
from ui.render_model import (
    VIEW_KIND_CACHED,
    VIEW_KIND_GROUNDED,
    cached_answer_notice,
    feedback_widget_model,
    fold_chat_stream,
)

GENERATED_ON = "2026-08-20"
FOOTER_TEXT = "Every citation links to source text. Sources as of 2026-08."


def wire_source_entry(index: int, source_type: str = "evidence") -> dict:
    """One #220 sources-event entry (the exact SOURCE_ENTRY_KEYS shape)."""
    return {
        "doc_id": f"syn-doc-{index}",
        "chunk_id": f"syn-doc-{index}::c0000",
        "title": f"Synthetic Source {index}",
        "attribution_text": f"Synthetic Source {index} (2026)",
        "canonical_url": f"https://example.test/doc-{index}",
        "source_type": source_type,
        "source_tier": "A",
        "permitted_context": "open",
        "excerpt": "A licence-bounded verbatim excerpt.",
        "excerpt_truncated": False,
    }


def with_exchange_id(event: dict, exchange_id: str | None) -> dict:
    event = {"event": event["event"], "data": dict(event["data"])}
    event["data"]["exchange_id"] = exchange_id
    return event


SOURCES_ENTRIES = [wire_source_entry(0), wire_source_entry(1, source_type="voices")]

#: The original grounded stream the cached serving replays. Sentence 0
#: cites document 0 (badged unverified); sentence 1 is uncited (flagged).
GROUNDED_TRANSCRIPT = [
    with_exchange_id(meta_event(), "source-exchange-1"),
    {"event": "sources", "data": {"sources": SOURCES_ENTRIES}},
    text_event("The basin has very likely warmed. "),
    text_event("Nights warm faster than days."),
    citation_event(0, "syn-doc-0::c0000", "very likely warmed", "Synthetic Source 0"),
    usage_event(),
    footer_event(FOOTER_TEXT),
    badge_event(0, 0, "entailment_failed"),
    badge_event(1, None, "uncited"),
]


def cached_wire_events(exchange_id: str = "serving-exchange-9") -> list[dict]:
    """The #57 serving: meta + ONE answer event replaying the stored data."""
    grounded = fold_chat_stream(GROUNDED_TRANSCRIPT)
    return [
        with_exchange_id(meta_event(), exchange_id),
        answer_event(
            "cached",
            grounded.text,
            generated_on=GENERATED_ON,
            footer=FOOTER_TEXT,
            citations=[
                dict(event["data"]) for event in GROUNDED_TRANSCRIPT if event["event"] == "citation"
            ],
            badges=[
                dict(event["data"]) for event in GROUNDED_TRANSCRIPT if event["event"] == "badge"
            ],
            sources=[dict(entry) for entry in SOURCES_ENTRIES],
        ),
    ]


class TestKindParity:
    def test_view_kind_matches_the_service_wire_kind(self) -> None:
        assert VIEW_KIND_CACHED == service_app.ANSWER_KIND_CACHED == "cached"


class TestCachedViewReplaysTheOriginalVerbatim:
    def test_view_kind_and_completeness(self) -> None:
        view = fold_chat_stream(cached_wire_events())
        assert view.kind == VIEW_KIND_CACHED
        assert view.kind != VIEW_KIND_GROUNDED, (
            "a cached answer never masquerades as a fresh grounded one"
        )
        assert view.complete is True
        assert view.error is None

    def test_text_and_footer_replay_byte_identical(self) -> None:
        grounded = fold_chat_stream(GROUNDED_TRANSCRIPT)
        cached = fold_chat_stream(cached_wire_events())
        assert cached.text == grounded.text
        assert cached.footer_text == grounded.footer_text == FOOTER_TEXT

    def test_chips_and_badges_replay_identically(self) -> None:
        grounded = fold_chat_stream(GROUNDED_TRANSCRIPT)
        cached = fold_chat_stream(cached_wire_events())
        assert grounded.chips, "the fixture must produce at least one chip to compare"
        assert cached.chips == grounded.chips, (
            "the replayed chips are the ORIGINAL (sentence, document) "
            "bindings with the ORIGINAL badges attached — same pairing rule, "
            "same quotes, same provenance flags"
        )
        assert any(chip.badges for chip in cached.chips), (
            "the original answer's 'unverified' badge must be worn by every "
            "replay — caching never launders a verification mark"
        )
        assert cached.uncited_flags == grounded.uncited_flags
        assert cached.uncited_flags, "the uncited-sentence flag rides the replay too"

    def test_sources_panel_replays_with_the_voices_partition(self) -> None:
        grounded = fold_chat_stream(GROUNDED_TRANSCRIPT)
        cached = fold_chat_stream(cached_wire_events())
        assert cached.sources_panel is not None
        assert cached.sources_panel == grounded.sources_panel, (
            "the §3.6 panel — including the voices/evidence separation — is "
            "part of the answer surface and replays verbatim"
        )
        assert cached.sources == grounded.sources

    def test_generated_on_carries_the_original_answers_date(self) -> None:
        view = fold_chat_stream(cached_wire_events())
        assert view.generated_on == GENERATED_ON


class TestCachedAnswerNotice:
    def test_notice_names_the_cached_state_and_the_date(self) -> None:
        notice = cached_answer_notice(GENERATED_ON)
        assert "Cached answer" in notice, (
            "the visible marker is pinned wording: a visitor must be able to "
            "tell a cached answer from a fresh one at a glance"
        )
        assert GENERATED_ON in notice, "the marker carries the ORIGINAL answer's ISO date"

    def test_notice_is_pure_and_deterministic(self) -> None:
        assert cached_answer_notice(GENERATED_ON) == cached_answer_notice(GENERATED_ON)


class TestFeedbackOnCachedViews:
    def test_the_serving_exchange_id_rides_the_view_verbatim(self) -> None:
        view = fold_chat_stream(cached_wire_events("serving-exchange-9"))
        assert view.exchange_id == "serving-exchange-9"

    def test_the_thumbs_widget_rides_a_cached_view(self) -> None:
        view = fold_chat_stream(cached_wire_events("serving-exchange-9"))
        widget = feedback_widget_model(view)
        assert widget is not None, (
            "feedback on a cached serving is exactly the #56/#57 eviction "
            "signal — the widget must ride the cached view"
        )
        assert widget.exchange_id == "serving-exchange-9"

    def test_a_key_less_cached_view_carries_no_widget(self) -> None:
        view = fold_chat_stream(cached_wire_events(None))
        assert feedback_widget_model(view) is None


class TestNoticeExportSurface:
    def test_presenters_export_the_cached_constants(self) -> None:
        import ui.presenters as presenters

        assert presenters.VIEW_KIND_CACHED == VIEW_KIND_CACHED
        assert presenters.cached_answer_notice is cached_answer_notice
