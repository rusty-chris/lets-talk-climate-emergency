"""The Streamlit shell (issue #18, DESIGN §7): thin views, pure decisions.

Every decision — SSE parsing, the answer-view fold, chip/badge mapping,
the landing model, the ADR-018 footer, chart views, the replay-vs-stream
rerun decision, the honest transport-failure view — lives in the pure
core behind ``ui.presenters`` (IMPLEMENTATION.md §1). This file only
draws what those models say and owns the one imperative concern the pure
core deliberately cannot: talking to the ``POST /chat`` SSE endpoint,
through the injected ``ui.transport`` seam.

The landing page shows the §7.1 starter topics as clickable buttons and a
free-text ``st.chat_input`` (both submit the exact question through the
same pure path). The chat view replays a cached exchange or streams a
fresh answer (never re-POSTing on a Streamlit rerun — finding #226),
guards the stream against transport failures (an honest degraded view,
never a public traceback — finding #224), and renders the annotated
answer body (calibrated-term markers + the likelihood legend — finding
#232), citation chips, uncited-sentence flags, the "Sources (n)" list,
status/error honesty, the paused / cached-starter surfaces, and inline
chart answers straight from ``view.chart`` (finding #229). The ADR-018
footer and the real transparency-route links (finding #228) sit on every
page.
"""

from __future__ import annotations

import os

import streamlit as st

from ui.presenters import (
    EVIDENCE_PANEL_HEADING,
    EXCHANGE_REPLAY,
    VIEW_KIND_GROUNDED,
    VOICES_PANEL_HEADING,
    AnswerView,
    ChartView,
    SseProtocolError,
    TransportError,
    annotate_calibrated_terms,
    answer_status_lines,
    build_page_footer,
    calibrated_term_anchors,
    chat_input_model,
    fold_chat_stream,
    footer_link_line,
    free_text_submission,
    landing_page_model,
    likelihood_legend,
    render_footer_lines,
    resolve_exchange,
    starter_submission,
    stream_chat_events,
    stream_text_delta,
    transport_failure_view,
)
from ui.transport import http_chat_transport

#: Where the shell reaches the #22 service. In compose the api service is
#: reachable at http://api:8000; a local dev run overrides to localhost.
API_URL = os.environ.get("CLIMATE_CHAT_API_URL", "http://localhost:8000")

#: The public origin used to render chart permalinks and the transparency
#: links absolutely (copy / embed / off-site targets); relative when unknown.
SITE_URL = os.environ.get("CLIMATE_CHAT_SITE_URL", "")


def _chart_base_url() -> str:
    """The origin chart permalinks/.csv/.svg must resolve off (the api, not
    this Streamlit host)."""
    return SITE_URL or API_URL


def _render_footer() -> None:
    """The ADR-018 steward credit + non-affiliation + REAL transparency links."""
    st.divider()
    footer = build_page_footer()
    # The credit pair and the non-affiliation line stay as captions; the
    # transparency routes become real, absolute markdown links on the
    # api/site origin (captions don't render markdown links — finding #228).
    for line in render_footer_lines(footer)[:2]:
        st.caption(line)
    st.markdown(footer_link_line(footer, _chart_base_url()))


def _render_chips(view: AnswerView) -> None:
    """Citation chips with verbatim quote popovers and unverified badges."""
    if not view.chips:
        return
    st.markdown("**Citations**")
    for chip in view.chips:
        label = f"[{chip.sentence_index + 1}] {chip.attribution}"
        if chip.badges:
            label += " ⚠"
        with st.popover(label):
            st.write(chip.quote)
            for badge in chip.badges:
                st.warning(f"Unverified: {badge.reason}")
            if not chip.clears_threshold:
                st.caption("Below the citation-support threshold.")
            if chip.needs_hand_review:
                st.caption("Flagged for hand review.")


def _render_sources(view: AnswerView) -> None:
    """The §7.2 "Sources (n)" surface, rendered from the fold's own list."""
    sources = view.sources
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for entry in sources:
            st.write(f"- {entry.attribution}")


def _render_source_panel_entry(entry) -> None:
    """One retrieved passage: attribution, tier, deep link, bounded excerpt."""
    tier = f" · Tier {entry.source_tier}" if entry.source_tier else ""
    st.markdown(f"**{entry.title}**{tier}")
    st.caption(entry.attribution)
    if entry.excerpt is not None:
        # The excerpt is the SERVICE's licence-bounded verbatim text; the UI
        # never fabricates or extends it, and signals a mid-passage cut with
        # the wire's own excerpt_truncated flag (no invented ellipsis prose).
        suffix = "…" if entry.excerpt_truncated else ""
        st.write(f"> {entry.excerpt}{suffix}")
    if entry.canonical_url:
        st.markdown(f"[Read the source]({entry.canonical_url})")


def _render_sources_panel(view: AnswerView) -> None:
    """The §3.6/§7.2 retrieved-passages panel: assessed evidence, and the
    first-party movement voices separated under "About the movement"."""
    panel = view.sources_panel
    if panel is None or (not panel.evidence and not panel.voices):
        return
    with st.expander("Sources & passages"):
        if panel.evidence:
            st.markdown(f"**{EVIDENCE_PANEL_HEADING}**")
            for entry in panel.evidence:
                _render_source_panel_entry(entry)
        if panel.voices:
            st.markdown(f"**{VOICES_PANEL_HEADING}**")
            for entry in panel.voices:
                _render_source_panel_entry(entry)


def _render_chart(chart: ChartView) -> None:
    """An inline chart answer: alt text + permalink · data · svg · embed."""
    st.caption(chart.alt_text)
    st.markdown(f"[Permalink]({chart.permalink})")
    st.markdown(f"[View data & sources]({chart.csv_href}) · [Download SVG]({chart.svg_href})")
    st.code(chart.embed_snippet, language="html")


def _render_answer_prose(view: AnswerView) -> None:
    """The answer body with calibrated-term markers (finding #232).

    Used for the post-stream re-render (replay, non-grounded kinds, error
    views); during a live grounded stream the plain streamed tokens are
    already on screen, so this is skipped there.
    """
    anchors = calibrated_term_anchors(view.text)
    st.markdown(annotate_calibrated_terms(view.text, anchors))


def _render_likelihood_legend() -> None:
    """The likelihood-scale legend the calibrated-term markers reference."""
    with st.expander("What do 'very likely' and friends mean?"):
        for entry in likelihood_legend():
            st.markdown(f"**{entry.term}** — {entry.assessed_probability}")


def _render_chat_input() -> None:
    """The free-text question input, on every page (§7.1 'Ask anything')."""
    model = chat_input_model()
    st.caption(model.disclosure)
    typed = st.chat_input(model.placeholder)
    if typed:
        submission = free_text_submission(typed)
        if submission is not None:
            st.session_state["pending"] = submission
            st.session_state.pop("exchange", None)
            st.rerun()


def _submit(question: str) -> None:
    st.session_state["pending"] = starter_submission(question)
    # A fresh question invalidates any cached exchange (finding #226).
    st.session_state.pop("exchange", None)


def _render_landing() -> None:
    page = landing_page_model()
    st.title(page.name)
    st.subheader(page.tagline)
    for group in page.groups:
        st.markdown(f"**{group.heading}**")
        for question in group.questions:
            st.button(
                question,
                key=f"starter::{question}",
                on_click=_submit,
                args=(question,),
                use_container_width=True,
            )


def _render_chat(question: str) -> None:
    if st.button("← Back", key="back"):
        st.session_state.pop("pending", None)
        st.session_state.pop("exchange", None)
        st.rerun()

    st.markdown(f"**You asked:** {question}")

    # The replay-vs-stream decision is pure (finding #226): a Streamlit
    # rerun replays the cached exchange instead of re-POSTing the question.
    decision = resolve_exchange(question, st.session_state.get("exchange"))
    base_url = _chart_base_url()

    with st.chat_message("assistant"):
        if decision.action == EXCHANGE_REPLAY:
            view = fold_chat_stream(list(decision.events), chart_base_url=base_url)
            _render_answer_prose(view)
        else:
            transport = http_chat_transport(API_URL)
            events: list[dict] = []

            def _text_stream():
                # st.write_stream renders text tokens live; we tee the raw
                # events so the fold can decide chips/badges/footer once the
                # stream completes. The "which event carries prose" decision
                # is pure (stream_text_delta), so the shell has no wire
                # literals of its own (finding #233).
                for event in stream_chat_events(transport, question):
                    events.append(event)
                    yield stream_text_delta(event)

            try:
                st.write_stream(_text_stream)
                view = fold_chat_stream(events, chart_base_url=base_url)
            except (TransportError, SseProtocolError) as exc:
                # A routine 429, an api restart mid-stream, or a malformed
                # frame folds the teed partial events into an honest view —
                # never a public Python traceback (finding #224).
                view = transport_failure_view(events, str(exc))
                _render_answer_prose(view)
            else:
                # Cache the completed exchange so a rerun replays it instead
                # of re-POSTing (finding #226).
                st.session_state["exchange"] = (question, tuple(events))
                if view.kind != VIEW_KIND_GROUNDED:
                    # Non-grounded kinds carry no text events to stream.
                    _render_answer_prose(view)

        if view.chart is not None:
            _render_chart(view.chart)
        _render_answer_tail(view)
        _render_likelihood_legend()


def _render_answer_tail(view: AnswerView) -> None:
    """Everything after the answer prose: status honesty, chips, flags, sources."""
    if view.generated_on:
        st.caption(f"Cached answer, generated on {view.generated_on}.")

    if view.preamble_note:
        st.info(view.preamble_note)

    # The honesty lines are pure and rendered UNCONDITIONALLY (finding
    # #224): an incomplete or errored answer is never presented complete.
    for line in answer_status_lines(view):
        st.warning(line)

    # Verification marks attach only on a cleanly-completed validation path;
    # an error view carries none (the pure fold already enforced this).
    if view.error is None:
        _render_chips(view)
        for flag in view.uncited_flags:
            st.warning(f"Sentence {flag.sentence_index + 1}: {flag.reason}")
        if view.validation_degraded:
            st.caption("Citation validation was unavailable; badges are not shown.")
        _render_sources(view)

    # The §3.6 retrieved-passages panel rides the grounded exchange's own
    # sources event (issue #220); it arrives before the first token, so an
    # error-terminated answer keeps whatever honestly landed.
    _render_sources_panel(view)

    if view.footer_text:
        st.caption(view.footer_text)
    # The privacy disclosure is part of the chat surface on EVERY path,
    # error pages included (issue #22 privacy contract).
    st.caption(view.disclosure)


def main() -> None:
    st.set_page_config(page_title="Let's Talk About the Climate Emergency")
    pending = st.session_state.get("pending")
    if pending is None:
        _render_landing()
    else:
        _render_chat(pending.question)
    _render_chat_input()
    _render_footer()


# Streamlit executes the module top to bottom on every rerun.
main()
