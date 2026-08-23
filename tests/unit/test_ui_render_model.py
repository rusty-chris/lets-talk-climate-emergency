"""Issue #18 RED — the chat answer view model (pure fold over #22 SSE events).

Pins ``ui.render_model``: meta-first, streaming text accumulation,
chip-honest citation chips keyed by the #13 segmentation, badge
attachment, uncited flags, error honesty (degraded view, never
fabricated text, never post-error badges), validation_degraded chip
honesty, paused/cached-starter/refusal/canned/chart answer kinds, the
source list, the disclosure line, and calibrated-term tooltip anchors.

Two sentences used throughout segment deterministically under
``rag.citation_validator.segment_answer_sentences`` (the shared
single source of truth for sentence indices):

    sentence 0: "Warming has reached 1.2C."   (cited: doc 0)
    sentence 1: "The rate is accelerating."   (cited: doc 1)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._ui_fixtures import (
    DISCLOSURE,
    answer_event,
    badge_event,
    chart_event,
    citation_event,
    error_event,
    footer_event,
    meta_event,
    text_event,
    usage_event,
    validation_degraded_event,
)
from ui.render_model import (
    LIKELIHOOD_TERMS,
    CitationChip,
    SourceEntry,
    StreamContractError,
    UncitedFlag,
    answer_status_lines,
    calibrated_term_anchors,
    chat_page_model,
    chips_for_cached_citations,
    fold_chat_stream,
    resolve_exchange,
    source_list,
    transport_failure_view,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def grounded_stream(*, extra_tail: list | None = None) -> list:
    """meta -> two cited sentences -> usage -> footer [-> extra events]."""
    return [
        meta_event(preamble_note="Answering the underlying question."),
        text_event("Warming has reached 1.2C. "),
        citation_event(0, "chunk-a", "warming of about 1.2C", "Meridian Assessment"),
        text_event("The rate is accelerating."),
        citation_event(1, "chunk-b", "the rate has accelerated", "Meridian Ocean Report"),
        usage_event(),
        footer_event("Every citation links to source text. Sources as of 2026-08."),
        *(extra_tail or []),
    ]


class TestFoldBasics:
    def test_meta_event_must_come_first(self) -> None:
        with pytest.raises(StreamContractError):
            fold_chat_stream([text_event("hello"), meta_event()])

    def test_streamed_text_accumulates_in_order(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert view.text == "Warming has reached 1.2C. The rate is accelerating."
        assert view.kind == "grounded"
        assert view.mode == "live"

    def test_disclosure_and_preamble_note_carried_verbatim(self) -> None:
        """The one-line logging disclosure is part of the chat surface."""
        view = fold_chat_stream(grounded_stream())
        assert view.disclosure == DISCLOSURE
        assert view.preamble_note == "Answering the underlying question."

    def test_usage_events_never_leak_into_view_text(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert "input_tokens" not in view.text
        assert "900" not in view.text

    def test_complete_only_when_footer_observed(self) -> None:
        complete = fold_chat_stream(grounded_stream())
        assert complete.complete is True
        assert complete.footer_text == (
            "Every citation links to source text. Sources as of 2026-08."
        )

        no_footer = fold_chat_stream([meta_event(), text_event("Warming has reached 1.2C.")])
        assert no_footer.complete is False
        assert no_footer.footer_text is None

    def test_unknown_event_names_are_ignored_forward_compatibly(self) -> None:
        events = grounded_stream(extra_tail=[{"event": "telemetry", "data": {"x": 1}}])
        view = fold_chat_stream(events)
        assert view.text == "Warming has reached 1.2C. The rate is accelerating."
        assert view.complete is True


class TestCitationChips:
    def test_chips_bound_to_sentence_and_document_with_quote_and_attribution(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert [
            (chip.sentence_index, chip.document_index, chip.chunk_id) for chip in view.chips
        ] == [(0, 0, "chunk-a"), (1, 1, "chunk-b")]
        assert view.chips[0].quote == "warming of about 1.2C"
        assert view.chips[0].attribution == "Meridian Assessment"

    def test_same_document_cited_twice_in_one_sentence_is_one_chip(self) -> None:
        """Finding #207: adjacent same-document citation spans, one chip;
        the quote popover keeps the FIRST arrival's cited text."""
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C. "),
            citation_event(0, "chunk-a", "first cited span", "Meridian Assessment"),
            citation_event(0, "chunk-a", "second cited span", "Meridian Assessment"),
            footer_event(),
        ]
        view = fold_chat_stream(events)
        assert len(view.chips) == 1
        assert view.chips[0].quote == "first cited span"

    def test_attribution_falls_back_to_chunk_id_without_document_title(self) -> None:
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C."),
            citation_event(0, "chunk-a", "warming of about 1.2C", None),
            footer_event(),
        ]
        view = fold_chat_stream(events)
        assert view.chips[0].attribution == "chunk-a"

    def test_provenance_flags_carried_untouched(self) -> None:
        """#143 honesty flags ride the chip; the UI never launders them."""
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C."),
            citation_event(
                0,
                "chunk-a",
                "warming of about 1.2C",
                "Meridian Assessment",
                clears_threshold=False,
                degraded_fallback=True,
                needs_hand_review=True,
            ),
            footer_event(),
        ]
        chip = fold_chat_stream(events).chips[0]
        assert chip.clears_threshold is False
        assert chip.degraded_fallback is True
        assert chip.needs_hand_review is True


class TestBadges:
    def test_badge_attaches_to_the_matching_chip_only(self) -> None:
        view = fold_chat_stream(
            grounded_stream(extra_tail=[badge_event(1, 1, "entailment_failed")])
        )
        unbadged, badged = view.chips
        assert badged.sentence_index == 1 and badged.document_index == 1
        assert [b.reason for b in badged.badges] == ["entailment_failed"]
        assert unbadged.badges == ()

    def test_uncited_flag_becomes_sentence_level_marker_not_a_chip_badge(self) -> None:
        """document_index None = an uncited factual sentence (#13): there is
        no chip to badge — the view carries a sentence-level marker."""
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C. "),
            citation_event(0, "chunk-a", "warming of about 1.2C", "Meridian Assessment"),
            text_event("Nothing cites this claim."),
            footer_event(),
            badge_event(1, None, "uncited"),
        ]
        view = fold_chat_stream(events)
        assert view.uncited_flags == (UncitedFlag(sentence_index=1, reason="uncited"),)
        assert all(chip.badges == () for chip in view.chips)

    def test_validation_degraded_shows_chips_without_badges(self) -> None:
        """Chip honesty: an unvalidated answer keeps its citations and
        wears NO verification marks."""
        view = fold_chat_stream(grounded_stream(extra_tail=[validation_degraded_event()]))
        assert view.validation_degraded is True
        assert len(view.chips) == 2
        assert all(chip.badges == () for chip in view.chips)
        assert view.uncited_flags == ()


class TestErrorHonesty:
    def test_error_event_yields_degraded_view_never_fabricated_text(self) -> None:
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C. The rate"),
            error_event("truncated", "The answer hit its output limit and is truncated."),
        ]
        view = fold_chat_stream(events)
        assert view.complete is False
        assert view.footer_text is None
        assert view.text == "Warming has reached 1.2C. The rate"
        assert view.error is not None
        assert view.error.error_type == "truncated"
        assert "truncated" in view.error.message

    def test_badges_never_attach_after_an_error_terminated_stream(self) -> None:
        """The #13 composer appends nothing after an error; a badge that
        appears anyway is a protocol breach and is never attached."""
        events = [
            meta_event(),
            text_event("Warming has reached 1.2C. "),
            citation_event(0, "chunk-a", "warming of about 1.2C", "Meridian Assessment"),
            error_event(),
            badge_event(0, 0, "entailment_failed"),
        ]
        view = fold_chat_stream(events)
        assert all(chip.badges == () for chip in view.chips)
        assert view.uncited_flags == ()


class TestTransportFailure:
    """Review finding #224 RED — transport failures fold to an honest view.

    A routine 429, an api restart mid-stream, or a malformed frame must
    NEVER surface a Python traceback on the public page: the shell hands
    the teed partial events plus a human-honest message to
    ``transport_failure_view`` and renders the resulting degraded view —
    the decision is pure, so the Streamlit shell has none.
    """

    def partial_events(self) -> list:
        return [
            meta_event(),
            text_event("Warming has reached 1.2C. "),
            citation_event(0, "chunk-a", "warming of about 1.2C", "Meridian Assessment"),
        ]

    def test_transport_failure_view_is_incomplete_with_error_notice(self) -> None:
        message = "The connection to the answer service was interrupted."
        view = transport_failure_view(self.partial_events(), message)

        assert view.complete is False
        assert view.error is not None
        assert view.error.error_type == "transport"
        assert message in view.error.message
        # The delivered prefix is preserved VERBATIM — never padded,
        # never dropped, never replaced by the error notice.
        assert view.text == "Warming has reached 1.2C. "
        # Same rule as an `error` event: no badges attach on a stream
        # that did not complete validation.
        assert all(chip.badges == () for chip in view.chips)
        assert view.uncited_flags == ()

    def test_transport_failure_before_meta_yields_error_view_not_contract_error(self) -> None:
        """Zero delivered events (connect refused / an immediate 429) is a
        NORMAL production failure: it must yield a renderable error view,
        never raise StreamContractError at the render boundary."""
        message = "Too many questions right now — please try again in a minute."
        view = transport_failure_view([], message)

        assert view.complete is False
        assert view.error is not None
        assert view.error.error_type == "transport"
        assert message in view.error.message
        assert view.text == ""
        assert view.chips == ()

    def test_transport_failure_message_never_carries_traceback_text(self) -> None:
        """The error notice is user-facing prose: no exception reprs, no
        module internals, no traceback markers ever ride the page model."""
        view = transport_failure_view(self.partial_events(), "The stream was interrupted.")
        for banned in ("Traceback", "httpx", "Exception("):
            assert banned not in view.error.message

    def test_error_view_still_carries_disclosure_line(self) -> None:
        """The privacy disclosure is part of the chat surface on EVERY
        path (issue #22 privacy contract) — including error pages, and
        including a failure before the meta event ever delivered it
        (transport_failure_view synthesizes it from the UI's own copy of
        the one-line notice; decision flagged in the #224 red notes).
        The error page also carries the ADR-018 footer pair."""
        with_meta = transport_failure_view(self.partial_events(), "interrupted")
        assert with_meta.disclosure == DISCLOSURE

        no_meta = transport_failure_view([], "connect refused")
        assert no_meta.disclosure == DISCLOSURE

        page = chat_page_model(no_meta)
        assert page.disclosure == DISCLOSURE
        assert page.footer.credit.credit_text == "Built by Rusty Data"
        assert page.footer.credit.noncommercial_note


class TestAnswerStatusLines:
    """Review finding #224 RED — the honesty flag reaches the page.

    ``fold_chat_stream`` computes ``complete=False`` for a truncated
    stream and the shell drops it at the render boundary; these pin a
    pure ``answer_status_lines`` the shell renders unconditionally, so a
    partial answer can never render complete-looking.
    """

    def test_status_lines_flag_incomplete_stream_without_error_event(self) -> None:
        truncated = fold_chat_stream([meta_event(), text_event("Warming has reached 1.2C.")])
        assert truncated.complete is False and truncated.error is None
        assert answer_status_lines(truncated) == (
            "This answer may be incomplete — the stream ended early.",
        )

    def test_complete_view_has_no_status_lines(self) -> None:
        complete = fold_chat_stream(grounded_stream())
        assert complete.complete is True
        assert answer_status_lines(complete) == ()

    def test_error_view_status_lines_carry_the_error_and_the_incomplete_line(self) -> None:
        errored = fold_chat_stream(
            [
                meta_event(),
                text_event("Warming has reached 1.2C. The rate"),
                error_event("truncated", "The answer hit its output limit and is truncated."),
            ]
        )
        lines = answer_status_lines(errored)
        assert lines, "an error view must produce visible status lines"
        assert "The answer hit its output limit and is truncated." in lines[0]
        assert "This answer is incomplete." in lines

    def test_transport_failure_view_status_lines_include_the_error(self) -> None:
        view = transport_failure_view([meta_event()], "The stream was interrupted.")
        lines = answer_status_lines(view)
        assert any("The stream was interrupted." in line for line in lines)
        assert "This answer is incomplete." in lines


class TestAnswerKinds:
    def test_refusal_view_carries_text_and_fabricates_nothing(self) -> None:
        events = [
            meta_event(),
            answer_event("refusal", "The corpus does not cover that; it does cover..."),
        ]
        view = fold_chat_stream(events)
        assert view.kind == "refusal"
        assert view.text.startswith("The corpus does not cover")
        assert view.chips == ()
        assert view.sources == ()
        assert view.complete is True

    def test_canned_view(self) -> None:
        view = fold_chat_stream([meta_event(), answer_event("canned", "I can help with...")])
        assert view.kind == "canned"
        assert view.text == "I can help with..."
        assert view.chips == ()
        assert view.complete is True

    def test_paused_without_cache_shows_dated_paused_text(self) -> None:
        events = [
            meta_event(mode="paused"),
            answer_event("paused", "Paused for today (2026-08-23); back tomorrow."),
        ]
        view = fold_chat_stream(events)
        assert view.mode == "paused"
        assert view.kind == "paused"
        assert "2026-08-23" in view.text
        assert view.chips == ()
        assert view.complete is True

    def test_paused_cached_starter_is_clearly_dated_with_chips_and_footer(self) -> None:
        cached_citations = [
            {
                "chunk_id": "chunk-a",
                "attribution_text": "Meridian Assessment",
                "cited_text": "warming of about 1.2C",
            },
            {
                "chunk_id": "chunk-b",
                "attribution_text": "Meridian Ocean Report",
                "cited_text": "the rate has accelerated",
            },
        ]
        events = [
            meta_event(mode="paused"),
            answer_event(
                "cached_starter",
                "Scientists call it an emergency because...",
                generated_on="2026-08-20",
                footer="Every citation links to source text. Sources as of 2026-08.",
                citations=cached_citations,
            ),
        ]
        view = fold_chat_stream(events)
        assert view.mode == "paused"
        assert view.kind == "cached_starter"
        assert view.generated_on == "2026-08-20"
        assert view.footer_text == ("Every citation links to source text. Sources as of 2026-08.")
        assert view.complete is True
        assert [(chip.chunk_id, chip.quote, chip.attribution) for chip in view.chips] == [
            ("chunk-a", "warming of about 1.2C", "Meridian Assessment"),
            ("chunk-b", "the rate has accelerated", "Meridian Ocean Report"),
        ]

    def test_chips_for_cached_citations_preserve_arrival_order(self) -> None:
        chips = chips_for_cached_citations(
            [
                {"chunk_id": "c2", "attribution_text": "B", "cited_text": "q2"},
                {"chunk_id": "c1", "attribution_text": "A", "cited_text": "q1"},
            ]
        )
        assert [(c.chunk_id, c.attribution, c.quote) for c in chips] == [
            ("c2", "B", "q2"),
            ("c1", "A", "q1"),
        ]

    def test_chart_event_yields_chart_view_with_alt_text_passed_through(self) -> None:
        alt = "Line chart: CO2 and temperature over the last 10,000 years."
        view = fold_chat_stream([meta_event(), chart_event("cafe0123beef", alt)])
        assert view.kind == "chart"
        assert view.chart is not None
        assert view.chart.spec_hash == "cafe0123beef"
        assert view.chart.permalink == "/chart/cafe0123beef"
        assert view.chart.alt_text == alt

    def test_fold_builds_chart_view_against_the_provided_base_url(self) -> None:
        """Review finding #229 RED — base_url belongs IN the fold.

        The shell re-derives the chart from raw events today only because
        the fold produces relative hrefs it cannot use; plumbing
        chart_base_url through makes view.chart the one renderable chart
        and deletes the shell's divergent copy. (Red today as a TypeError:
        the missing parameter IS the defect.)"""
        view = fold_chat_stream(
            [meta_event(), chart_event("cafe0123beef")],
            chart_base_url="https://site.example",
        )
        assert view.chart is not None
        assert view.chart.permalink == "https://site.example/chart/cafe0123beef"
        assert view.chart.csv_href == "https://site.example/chart/cafe0123beef.csv"
        assert view.chart.svg_href == "https://site.example/chart/cafe0123beef.svg"
        assert "https://site.example/chart/cafe0123beef.svg" in view.chart.embed_snippet

    def test_repeated_chart_events_pin_one_winner(self) -> None:
        """Review finding #229 — GREEN regression pin, marked as such.

        The fold already keeps the LAST chart event; the shell renders
        the FIRST (its raw-event next() re-derivation). This pins the
        fold's last-wins rule as the ONE rule, so the shell's divergent
        copy has no tested behaviour to hide behind once deleted."""
        view = fold_chat_stream(
            [
                meta_event(),
                chart_event("first1111111", "First chart alt text."),
                chart_event("second222222", "Second chart alt text."),
            ]
        )
        assert view.kind == "chart"
        assert view.chart is not None
        assert view.chart.spec_hash == "second222222"
        assert view.chart.alt_text == "Second chart alt text."

    def test_chart_event_without_alt_text_folds_to_error_view_not_exception(self) -> None:
        """Review finding #229 RED — the fold degrades honestly.

        chart_view_from_event keeps raising ChartAccessibilityError (its
        contract is unchanged — pinned in test_ui_charts.py); the FOLD
        must convert that into an honest error view, never crash the
        public page mid-render."""
        blank_alt = dict(chart_event("cafe0123beef")["data"], alt_text="   ")
        view = fold_chat_stream([meta_event(), {"event": "chart", "data": blank_alt}])

        assert view.error is not None, "a mute chart must fold to an error view"
        assert "accessib" in (view.error.error_type + view.error.message).lower()
        assert view.chart is None
        assert view.complete is False


class TestExchangeReplay:
    """Review finding #226 RED — a rerun must never re-POST the question.

    Streamlit re-executes the script on every rerun (the 'R' key, the
    menu, an app deploy): today each execution re-opens POST /chat,
    re-spending the daily budget and replacing the rendered answer with
    a different generation. The replay-vs-stream decision is made pure
    (``resolve_exchange``) so it is testable — the shell only obeys it.
    """

    def test_answered_question_replays_cached_events_without_streaming(self) -> None:
        question = "How bad is it?"
        events = tuple(grounded_stream())
        decision = resolve_exchange(question, (question, events))

        assert decision.action == "replay"
        # The decision carries everything the render needs: no transport
        # call is required or permitted on this branch (the shell-side
        # guard is pinned in test_ui_shell_hygiene.py).
        assert decision.events == events
        assert fold_chat_stream(decision.events) == fold_chat_stream(events)

    def test_new_question_streams_and_is_cached(self) -> None:
        cached = ("How bad is it?", tuple(grounded_stream()))

        fresh = resolve_exchange("What can we do?", cached)
        assert fresh.action == "stream"

        # After the fresh stream completes, the recorded (question,
        # events) pair round-trips through the replay path on the next
        # rerun — the same exchange is never streamed twice.
        recorded = ("What can we do?", tuple(grounded_stream()))
        rerun = resolve_exchange("What can we do?", recorded)
        assert rerun.action == "replay"
        assert rerun.events == recorded[1]

    def test_no_cache_streams(self) -> None:
        decision = resolve_exchange("How bad is it?", None)
        assert decision.action == "stream"


class TestSourceList:
    def test_sources_deduped_by_chunk_id_in_arrival_order(self) -> None:
        chips = (
            CitationChip(0, 0, "chunk-a", "q1", "Meridian Assessment"),
            CitationChip(1, 1, "chunk-b", "q2", "Meridian Ocean Report"),
            CitationChip(2, 0, "chunk-a", "q3", "Meridian Assessment"),
        )
        assert source_list(chips) == (
            SourceEntry("chunk-a", "Meridian Assessment"),
            SourceEntry("chunk-b", "Meridian Ocean Report"),
        )

    def test_fold_populates_sources_from_chips(self) -> None:
        view = fold_chat_stream(grounded_stream())
        assert view.sources == (
            SourceEntry("chunk-a", "Meridian Assessment"),
            SourceEntry("chunk-b", "Meridian Ocean Report"),
        )


class TestCalibratedTermAnchors:
    def test_terms_anchor_case_insensitively_with_correct_spans(self) -> None:
        text = "It is Very likely that warming will continue."
        anchors = calibrated_term_anchors(text)
        assert [a.term for a in anchors] == ["very likely"]
        (anchor,) = anchors
        assert text[anchor.start : anchor.end].lower() == "very likely"

    def test_longest_match_wins_and_anchors_never_overlap(self) -> None:
        text = "Collapse this century is extremely unlikely but not impossible."
        anchors = calibrated_term_anchors(text)
        assert [a.term for a in anchors] == ["extremely unlikely"]

        text2 = "It is more likely than not that the threshold is crossed."
        anchors2 = calibrated_term_anchors(text2)
        assert [a.term for a in anchors2] == ["more likely than not"]

    def test_plain_text_yields_no_anchors(self) -> None:
        assert calibrated_term_anchors("The planet is warming.") == ()

    def test_vocabulary_matches_the_generation_prompt_table(self) -> None:
        """Single source of truth: every UI legend term appears in the
        calibrated-vocabulary table pinned in the generation prompt."""
        prompt = (REPO_ROOT / "rag" / "prompts" / "generation_system_prompt.md").read_text(
            encoding="utf-8"
        )
        for term in LIKELIHOOD_TERMS:
            assert term in prompt, f"likelihood term {term!r} missing from the prompt table"


class TestChatPage:
    def test_chat_page_carries_disclosure_and_the_adr018_footer_pair(self) -> None:
        view = fold_chat_stream(grounded_stream())
        page = chat_page_model(view)
        assert page.disclosure == DISCLOSURE
        assert page.footer.credit.credit_text == "Built by Rusty Data"
        assert page.footer.credit.noncommercial_note  # never the credit alone
