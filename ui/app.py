"""The Streamlit shell (issue #18, DESIGN §7): thin views, pure decisions.

Every decision — SSE parsing, the answer-view fold, chip/badge mapping,
the landing model, the ADR-018 footer, chart views — lives in the pure
core behind ``ui.presenters`` (IMPLEMENTATION.md §1). This file only
draws what those models say and owns the one imperative concern the pure
core deliberately cannot: talking to the ``POST /chat`` SSE endpoint,
through the injected ``ui.transport`` seam.

The landing page shows the §7.1 starter topics as clickable buttons; a
click submits the exact question. The chat view streams the answer via
``st.write_stream`` over ``stream_chat_events``, then renders citation
chips (quote popovers + unverified badges), uncited-sentence flags, the
"Sources (n)" list, error/degraded honesty, the paused / cached-starter
surfaces, and inline chart answers (permalink · .csv · .svg · embed
snippet · alt text). The ADR-018 footer and the transparency-route links
sit on every page.
"""

from __future__ import annotations

import os

import streamlit as st

from ui.presenters import (
    AnswerView,
    ChartView,
    build_page_footer,
    calibrated_term_anchors,
    chart_view_from_event,
    fold_chat_stream,
    landing_page_model,
    render_footer_lines,
    source_list,
    starter_submission,
    stream_chat_events,
)
from ui.transport import http_chat_transport

#: Where the shell reaches the #22 service. In compose the api service is
#: reachable at http://api:8000; a local dev run overrides to localhost.
API_URL = os.environ.get("CLIMATE_CHAT_API_URL", "http://localhost:8000")

#: The public origin used to render chart permalinks absolutely (copy /
#: embed targets must resolve off-site); relative when unknown.
SITE_URL = os.environ.get("CLIMATE_CHAT_SITE_URL", "")


def _render_footer() -> None:
    """The ADR-018 steward credit + non-affiliation + transparency routes."""
    st.divider()
    footer = build_page_footer()
    for line in render_footer_lines(footer):
        st.caption(line)


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
    """The §7.2 "Sources (n)" surface, derived from the chips."""
    sources = view.sources or source_list(view.chips)
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for entry in sources:
            st.write(f"- {entry.attribution}")


def _render_chart(chart: ChartView) -> None:
    """An inline chart answer: alt text + permalink · data · svg · embed."""
    st.caption(chart.alt_text)
    st.markdown(f"[Permalink]({chart.permalink})")
    st.markdown(f"[View data & sources]({chart.csv_href}) · [Download SVG]({chart.svg_href})")
    st.code(chart.embed_snippet, language="html")


def _submit(question: str) -> None:
    st.session_state["pending"] = starter_submission(question)


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
        st.rerun()

    st.markdown(f"**You asked:** {question}")

    transport = http_chat_transport(API_URL)
    events: list[dict] = []

    def _text_stream():
        # st.write_stream renders text tokens live; we tee the raw events so
        # the fold can decide chips/badges/footer once the stream completes.
        for event in stream_chat_events(transport, question):
            events.append(event)
            if event.get("event") == "text":
                yield event["data"].get("text", "")

    with st.chat_message("assistant"):
        st.write_stream(_text_stream)
        view = fold_chat_stream(events)
        # Re-render the answer body from the fold for the non-grounded kinds
        # (answers/charts carry no text events to stream) and to attach the
        # citations, badges and sources the stream could not.
        if view.kind != "grounded":
            st.write(view.text)
        if view.kind == "chart":
            # Rebuild the chart view against the public origin so the
            # permalink / .csv / .svg / embed targets resolve OFF this
            # Streamlit host (they are served by the api, not the UI).
            data = next((e["data"] for e in events if e.get("event") == "chart"), None)
            if data is not None:
                _render_chart(chart_view_from_event(data, base_url=SITE_URL or API_URL))
        _render_answer_tail(view)


def _render_answer_tail(view: AnswerView) -> None:
    """Everything after the streamed prose: chips, flags, sources, notes."""
    if view.generated_on:
        st.caption(f"Cached answer, generated on {view.generated_on}.")

    if view.preamble_note:
        st.info(view.preamble_note)

    if view.error is not None:
        st.error(f"{view.error.error_type}: {view.error.message}")
        st.caption("This answer is incomplete.")
        return

    _render_chips(view)
    for flag in view.uncited_flags:
        st.warning(f"Sentence {flag.sentence_index + 1}: {flag.reason}")
    if view.validation_degraded:
        st.caption("Citation validation was unavailable; badges are not shown.")
    _render_sources(view)
    if calibrated_term_anchors(view.text):
        st.caption(
            "Highlighted phrases (e.g. 'very likely') follow the calibrated likelihood scale."
        )
    if view.footer_text:
        st.caption(view.footer_text)
    st.caption(view.disclosure)


def main() -> None:
    st.set_page_config(page_title="Let's Talk About the Climate Emergency")
    pending = st.session_state.get("pending")
    if pending is None:
        _render_landing()
    else:
        _render_chat(pending.question)
    _render_footer()


# Streamlit executes the module top to bottom on every rerun.
main()
