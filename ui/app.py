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

import base64
import html
import os

import streamlit as st

from ui.presenters import (
    EVIDENCE_PANEL_HEADING,
    FEEDBACK_STATE_RECORDED,
    SESSION_FOOTPRINT_CAPTION,
    SESSION_FOOTPRINT_EMPTY,
    SESSION_FOOTPRINT_EMPTY_LINE,
    SESSION_FOOTPRINT_HEADING,
    VIEW_KIND_GROUNDED,
    VOICES_PANEL_HEADING,
    AnswerView,
    ChartView,
    SessionFootprint,
    SseProtocolError,
    TransportError,
    accumulate_session_footprint,
    answer_status_lines,
    build_page_footer,
    cached_answer_notice,
    chat_input_model,
    feedback_widget_model,
    fold_chat_stream,
    footer_link_line,
    footprint_indicator_line,
    free_text_submission,
    landing_page_model,
    likelihood_legend,
    render_footer_lines,
    render_inline_answer,
    resolve_feedback_state,
    session_footprint_display,
    starter_submission,
    steward_mark_img_tag,
    stream_chat_events,
    stream_text_delta,
    transport_failure_view,
    uncited_sentence_note,
    unverified_badge_note,
)
from ui.theme import (
    globe_loader_placeholder,
    inject_theme,
    render_hero,
    render_top_bar,
)
from ui.transport import (
    fetch_budget,
    fetch_chart_svg,
    http_chat_transport,
    http_feedback_transport,
)

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


#: The transparency menu shown in the branded top bar (owner ask: use the
#: previously blank header for a menu). These are the real service routes
#: (finding #228); they resolve off the public origin when known.
_TRANSPARENCY_NAV = (
    ("About", "/about"),
    ("Sources", "/sources"),
    ("Voices", "/voices"),
    ("Footprint", "/footprint"),
    ("Privacy", "/privacy"),
)


def _top_nav_items() -> list[tuple[str, str]]:
    """(label, absolute-href) pairs for the top-bar transparency menu."""
    base = _chart_base_url().rstrip("/")
    return [(label, f"{base}{path}") for label, path in _TRANSPARENCY_NAV]


def _render_footer() -> None:
    """The ADR-018 steward credit (mark + live rustydata.ai link) +
    non-affiliation + REAL transparency links."""
    st.divider()
    footer = build_page_footer()
    lines = render_footer_lines(footer)
    # Rusty Data branding: the footer-scale mark inline with the ADR-018
    # credit line, one line, via st.markdown so the rustydata.ai link is
    # LIVE (captions render markdown links unreliably, the finding-#228
    # lesson). The mark is a base64 data-URI <img> — NOT st.image, which
    # cannot render a local .svg under Streamlit 1.38+ (it raises
    # MediaFileStorageError; the crash on the 2026-09-13 first page load).
    st.markdown(f"{steward_mark_img_tag()} {lines[0]}", unsafe_allow_html=True)
    # The non-affiliation disclaimer stays a caption; the transparency
    # routes are real, absolute markdown links on the api/site origin.
    st.caption(lines[1])
    st.markdown(footer_link_line(footer, _chart_base_url()))


def _render_chips(view: AnswerView) -> None:
    """Citation chips with verbatim quote popovers and unverified badges."""
    if not view.chips:
        return
    st.markdown("**Citations**")
    for chip in view.chips:
        label = f"[{chip.sentence_index + 1}] {chip.attribution}"
        if chip.badges:
            # A neutral info affordance (ⓘ), not a ⚠ error glyph (#401): an
            # "unverified" badge is measured, honest nuance — the runtime
            # support check found the cited source may not fully entail this
            # sentence — so it reads as transparency, not failure.
            label += " ⓘ"
        with st.popover(label):
            # The quote is verbatim ND-constrained source text: render it
            # through st.text, which interprets no markdown/KaTeX/HTML, so a
            # "$…$" span or a "*"/"_"/"#" is shown as written, never adapted
            # (finding #267).
            st.text(chip.quote)
            for badge in chip.badges:
                # Informational, not alarming (#401): st.info's neutral tone
                # (theme-aware in light AND dark) plus the pure core's plain
                # language, so the honestly-published support gap reads as
                # nuance we disclose, never a hard error.
                st.info(unverified_badge_note(badge.reason))
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
        # Rendered through st.text — the non-interpreting surface — so
        # markdown/KaTeX metacharacters (e.g. two "$" figures) reach the
        # reader exactly as written, an ND-verbatim rendering, not an adapted
        # one (finding #267). The only added signal is the wire-driven "…".
        suffix = "…" if entry.excerpt_truncated else ""
        st.text(f"{entry.excerpt}{suffix}")
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
    """An inline chart answer — the GRAPH itself, then its affordances.

    Owner ask (2026-09-15): "show me X" must produce a visible graph. The SVG
    is fetched off the internal api and inlined as a base64 data-URI ``<img>``
    on a light card (st.image cannot render an SVG under Streamlit 1.38+, the
    finding-#229/footer lesson; the light card keeps the white chart legible on
    the dark theme). If the fetch fails the answer degrades to the permalink /
    data / download links rather than showing nothing.
    """
    st.caption(chart.alt_text)
    svg = fetch_chart_svg(API_URL, chart.spec_hash)
    if svg:
        encoded = base64.b64encode(svg).decode("ascii")
        st.markdown(
            f'<img class="climate-chart" src="data:image/svg+xml;base64,{encoded}" '
            f'alt="{html.escape(chart.alt_text, quote=True)}" />',
            unsafe_allow_html=True,
        )
    st.markdown(f"[Permalink]({chart.permalink})")
    st.markdown(f"[View data & sources]({chart.csv_href}) · [Download SVG]({chart.svg_href})")
    with st.expander("Embed this chart"):
        st.code(chart.embed_snippet, language="html")


def _feedback_state_key(exchange_id: str) -> str:
    return f"feedback::{exchange_id}"


def _post_feedback(exchange_id: str, verdict: str) -> None:
    """Post one thumbs verdict and store the HONEST resolved state.

    Network belongs to ui.transport's seam; whether the click is shown as
    recorded (204) or honestly unrecorded (any other outcome) is the pure
    resolve_feedback_state's decision, never the shell's.
    """
    succeeded = http_feedback_transport(API_URL)(exchange_id, verdict)
    st.session_state[_feedback_state_key(exchange_id)] = resolve_feedback_state(verdict, succeeded)


def _render_feedback(view: AnswerView) -> None:
    """The thumbs up/down widget on a completed, rateable answer (#56).

    Widget presence is the pure feedback_widget_model's decision (never the
    shell's own rule); the closed verdict vocabulary and labels reach here
    only through the widget's fields.
    """
    widget = feedback_widget_model(view)
    if widget is None:
        return
    state = st.session_state.get(_feedback_state_key(widget.exchange_id))
    # Once recorded, the rating stands — show its confirmation, no re-vote.
    if state is not None and state.status == FEEDBACK_STATE_RECORDED:
        st.caption(state.message)
        return
    up_col, down_col = st.columns(2)
    up_col.button(
        widget.up_label,
        key=f"feedback::{widget.exchange_id}::{widget.up_verdict}",
        on_click=_post_feedback,
        args=(widget.exchange_id, widget.up_verdict),
    )
    down_col.button(
        widget.down_label,
        key=f"feedback::{widget.exchange_id}::{widget.down_verdict}",
        on_click=_post_feedback,
        args=(widget.exchange_id, widget.down_verdict),
    )
    # A prior FAILED post shows the honest unrecorded message (never a fake
    # success); the visitor can try again with the buttons still above.
    if state is not None:
        st.caption(state.message)


def _render_answer_prose(view: AnswerView) -> None:
    """The answer body with calibrated-term markers AND inline citation marks.

    Used for the post-stream re-render (replay, non-grounded kinds, error
    views); during a live grounded stream the plain streamed tokens are
    already on screen, so this is skipped there. The inline ``⁽N⁾`` marks
    (issue #399) trail each cited sentence, echoing the ``[N]`` chip below —
    the reader sees which statement each citation backs. Both annotations are
    merged in one pure pass (render_inline_answer) so their offsets never
    fight; the result is markdown/Unicode only, so a single ``st.markdown``
    renders it (no unsafe HTML over model prose).
    """
    st.markdown(render_inline_answer(view.text, view.chips))


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
    # Issue #398: the Earth-from-space hero (planet motif + gradient wordmark)
    # replaces the bare st.title/subheader so the landing page opens on the
    # planet theme, not a default Streamlit heading. Self-contained CSS/SVG.
    render_hero(page.name, page.tagline)
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


#: Cap the conversation history sent to the backend (owner ask 2026-09-15:
#: the bot must remember the conversation). Bounds tokens on long chats; the
#: most recent turns carry the thread the model needs to follow up.
_MAX_HISTORY_TURNS = 10


def _history_from_turns(turns: list[dict]) -> list[dict[str, str]]:
    """The (role, content) history the backend expects, oldest first.

    One user turn then one assistant turn per completed exchange, capped to
    the most recent :data:`_MAX_HISTORY_TURNS`. The service owns query
    processing and passes this VERBATIM (finding: sse_client), so the cap
    lives here."""
    history: list[dict[str, str]] = []
    for turn in turns[-_MAX_HISTORY_TURNS:]:
        history.append({"role": "user", "content": turn["question"]})
        history.append({"role": "assistant", "content": turn["answer_text"]})
    return history


def _render_completed_turn(turn: dict, base_url: str):
    """Replay one finished turn (question + answer) — no re-POST (finding #226)."""
    st.markdown(f"**You asked:** {turn['question']}")
    with st.chat_message("assistant"):
        view = fold_chat_stream(list(turn["events"]), chart_base_url=base_url)
        _render_answer_prose(view)
        if view.chart is not None:
            _render_chart(view.chart)
    return view


def _render_chat() -> None:
    """The multi-turn chat (owner ask 2026-09-15: the bot must remember the
    conversation): prior turns are REPLAYED from their stored events — never
    re-POSTed on a Streamlit rerun (finding #226) — and only the pending
    question opens POST /chat, carrying the conversation so far as history."""
    if st.button("← New conversation", key="reset"):
        for key in ("turns", "pending", "session_footprint"):
            st.session_state.pop(key, None)
        st.rerun()

    base_url = _chart_base_url()
    turns: list[dict] = st.session_state.get("turns", [])
    session = SESSION_FOOTPRINT_EMPTY
    last_view = None

    # 1) Every completed turn, replayed from its stored events — no transport,
    #    so a Streamlit rerun never re-POSTs (finding #226).
    for turn in turns:
        view = _render_completed_turn(turn, base_url)
        session = accumulate_session_footprint(session, view.exchange_id, view.footprint)
        last_view = view

    # 2) The pending (in-flight) question is the ONLY thing that opens the
    #    transport — conditionally, on this stream branch (finding #226).
    pending = st.session_state.get("pending")
    if pending is not None:
        question = pending.question
        history = _history_from_turns(turns)
        st.markdown(f"**You asked:** {question}")
        with st.chat_message("assistant"):
            transport = http_chat_transport(API_URL)
            events: list[dict] = []

            def _text_stream():
                # Prior turns travel as history so the model can follow up
                # (owner ask). "Which event carries prose" stays pure (#233).
                for event in stream_chat_events(transport, question, history):
                    events.append(event)
                    yield stream_text_delta(event)

            loader = globe_loader_placeholder()
            answer_text = ""
            try:
                answer_text = st.write_stream(_text_stream) or ""
                view = fold_chat_stream(events, chart_base_url=base_url)
            except (TransportError, SseProtocolError) as exc:
                # A 429, an api restart mid-stream, or a malformed frame folds
                # the teed partial events into an honest view (finding #224).
                view = transport_failure_view(events, str(exc))
                _render_answer_prose(view)
            else:
                if view.kind != VIEW_KIND_GROUNDED:
                    # Non-grounded kinds carry no text events to stream.
                    _render_answer_prose(view)
            finally:
                loader.empty()
            if view.chart is not None:
                _render_chart(view.chart)
        # Persist the finished turn and clear pending: a later rerun replays it
        # from these events (no re-POST) and the NEXT question carries it as
        # history. write_stream returns the streamed prose; the folded view's
        # text is the fallback for non-grounded/errored answers.
        st.session_state["turns"] = [
            *turns,
            {
                "question": question,
                "events": tuple(events),
                "answer_text": answer_text or view.text or "",
            },
        ]
        st.session_state.pop("pending", None)
        session = accumulate_session_footprint(session, view.exchange_id, view.footprint)
        last_view = view

    # The session cumulative is idempotent by exchange_id (finding #226): the
    # replay each rerun never double-counts.
    st.session_state["session_footprint"] = session
    _render_session_footprint(session)
    if last_view is not None:
        _render_answer_tail(last_view, session)
    _render_likelihood_legend()


def _render_session_footprint(session: SessionFootprint) -> None:
    """The prominent, VISUAL main-page footprint panel (issue #402).

    Draws the live per-session estimate as a gauge + a headline RANGE + an
    everyday-equivalent anchor, straight from the pure
    ``session_footprint_display`` model — the shell holds no figure,
    threshold or arithmetic of its own (the finding-#233 discipline).

    Privacy: this is a pure function of the in-memory ``SessionFootprint``
    the UI already accumulates per exchange (issue #402 / #226). It reads
    NO identifier and writes NO storage — the figure is scoped to THIS
    visit and vanishes when the session ends. There is deliberately NO
    per-IP/per-day figure here: the exchange log stores no ``ip_hash`` and
    the rate-limit store no usage, so a per-IP total cannot be built
    without joining the two deliberately-separated stores — which the
    privacy design (and the owner) declined.
    """
    display = session_footprint_display(session)
    with st.container(border=True):
        st.markdown(f"**{SESSION_FOOTPRINT_HEADING}**")
        if display.is_empty:
            # Before the first exchange: the honest zero start, never an
            # invented figure or an empty 0-of-a-mug gauge.
            st.caption(SESSION_FOOTPRINT_EMPTY_LINE)
            return
        # The RANGE is the headline figure (§9: never a bare central point).
        st.metric(
            "Estimated energy this visit",
            f"{display.total_wh_low}–{display.total_wh_high} Wh",
        )
        # A rough visual gauge (mugs-of-tea fraction); the range beside it
        # carries the honesty, so the meter is explicitly a central estimate.
        st.progress(display.meter_fraction)
        st.caption(f"Central estimate ~{display.total_wh_central} Wh · {display.equivalent_line}")
        st.caption(SESSION_FOOTPRINT_CAPTION)


#: Display-only USD→GBP conversion for the operator spend readout. The caps
#: are enforced server-side in USD; this just renders them in the owner's
#: currency. Set the USD caps to £target / this rate for round £ figures.
GBP_PER_USD = 0.79


def _render_operator_spend() -> None:
    """The owner's live compute-cost readout (owner ask 2026-09-15).

    Gated SERVER-SIDE by ``CLIMATE_CHAT_SHOW_SPEND``: when off, ``/budget``
    reports ``enabled: false`` and this renders nothing (cost never leaks
    publicly). When on, shows today's and this week's spend against the caps
    in GBP, plus a paused banner if a cap has been reached. Silent on any
    fetch failure — a missing readout must never break the page.
    """
    data = fetch_budget(API_URL)
    if not data or not data.get("enabled"):
        return

    def gbp(usd: float | None) -> float | None:
        return None if usd is None else usd * GBP_PER_USD

    spent_day = gbp(data.get("spent_today_usd"))
    cap_day = gbp(data.get("daily_cap_usd"))
    spent_week = gbp(data.get("spent_week_usd"))
    cap_week = gbp(data.get("weekly_cap_usd"))
    with st.container(border=True):
        st.markdown("**Compute cost (live · operator view)**")
        if data.get("cap_reached"):
            st.warning(
                "Paused — a spend cap was reached; serving cached answers only until it resets."
            )
        if spent_day is not None and cap_day:
            st.progress(min(1.0, max(0.0, spent_day / cap_day)) if cap_day else 0.0)
            st.caption(f"Today: £{spent_day:.2f} / £{cap_day:.0f}")
        if spent_week is not None and cap_week:
            st.caption(f"This week: £{spent_week:.2f} / £{cap_week:.0f}")
        st.caption(
            "Live generation runs on Opus; the service pauses to cached answers at either cap."
        )


def _render_answer_tail(view: AnswerView, session: SessionFootprint) -> None:
    """Everything after the answer prose: status honesty, chips, flags, sources."""
    if view.generated_on:
        # The honesty line comes from the pure core so the rendered caption
        # and its pin can never diverge (finding #289); the shell holds no
        # cached-answer copy of its own.
        st.caption(cached_answer_notice(view.generated_on))

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
            # An uncited factual sentence is honest disclosure, not an error
            # (#401): the same neutral, theme-aware informational tone as the
            # unverified badge, never the yellow warning box.
            st.info(uncited_sentence_note(flag.sentence_index, flag.reason))
        if view.validation_degraded:
            st.caption("Citation validation was unavailable; badges are not shown.")
        _render_sources(view)

    # The §3.6 retrieved-passages panel rides the grounded exchange's own
    # sources event (issue #220); it arrives before the first token, so an
    # error-terminated answer keeps whatever honestly landed.
    _render_sources_panel(view)

    if view.footer_text:
        st.caption(view.footer_text)
    # The footprint indicator (docs/FOOTPRINT-METHODOLOGY.md): the §9 template
    # verbatim via the pure helper — always a range, "est." present, the
    # /footprint link, no gCO2e. None on views that carry no honest figure.
    indicator = footprint_indicator_line(view, session)
    if indicator is not None:
        st.caption(indicator)
    # The thumbs up/down widget rides every completed, rateable answer (#56);
    # the pure model returns None on errored/incomplete/key-less views.
    _render_feedback(view)
    # The privacy disclosure is part of the chat surface on EVERY path,
    # error pages included (issue #22 privacy contract).
    st.caption(view.disclosure)


def main() -> None:
    st.set_page_config(
        page_title="Let's Talk About the Climate Emergency",
        page_icon="🌍",
        layout="centered",
    )
    # Issue #398: inject the planet-Earth theme once per rerun, before any
    # widget draws, so the whole shell (hero, buttons, panels, globe loader)
    # picks it up. Self-contained CSS — no external stylesheet/font/image.
    inject_theme()
    # The branded top bar uses the previously blank header (owner ask): the
    # Rusty Data steward mark + the transparency menu on every page, and — once
    # a chat has started — the app title compactly in the header (so the chat
    # view drops the big landing hero but keeps the title visible up top).
    # In a conversation once a question is pending OR any turn has completed
    # (multi-turn memory, owner ask 2026-09-15): the running thread persists so
    # follow-ups carry history. The landing page shows only before the first Q.
    pending = st.session_state.get("pending")
    in_conversation = pending is not None or bool(st.session_state.get("turns"))
    page_title = landing_page_model().name if in_conversation else None
    render_top_bar(steward_mark_img_tag(), _top_nav_items(), page_title=page_title)
    if not in_conversation:
        # §7.1 / issue #403: the free-text "Ask anything" input is the first
        # interactive element on the landing page, with the starter groups
        # (and their headings) intact below it — the chat box invites typing
        # before it offers the 13 canned prompts.
        _render_chat_input()
        _render_landing()
        # The main-page footprint panel: on a first load it shows the honest
        # zero-start invitation (issue #402).
        _render_session_footprint(
            st.session_state.get("session_footprint", SESSION_FOOTPRINT_EMPTY)
        )
    else:
        # The multi-turn thread (prior turns replayed + the pending one
        # streamed with history), then the "ask a follow-up" input below it.
        _render_chat()
        _render_chat_input()
    # The owner's live compute-cost readout, beside the footprint panel
    # (owner ask 2026-09-15). Renders only when CLIMATE_CHAT_SHOW_SPEND is on
    # server-side; otherwise silent, so operating cost never leaks publicly.
    _render_operator_spend()
    _render_footer()


# Streamlit executes the module top to bottom on every rerun.
main()
