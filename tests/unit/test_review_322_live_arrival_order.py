"""Span stamping against the LIVE citations_delta arrival order
(review finding #322, blocker) — RED.

The verification smoke run (PR #321, `data/verification-smoke/REPORT.md`)
measured **202 of 202** live citation SSE events carrying a zero-width
answer span (`answer_block_start == answer_block_end`). The #310 span
stamping in ``rag.generation.answer_stream_to_sse`` assumes the
``citations_delta`` arrives AFTER its block's text deltas and stamps
``answer_block_end`` from the running cursor at citation arrival. On the
live stream the ``citations_delta`` arrives at block OPEN — the smoke
worktree's journalled transcripts are the arrival-order ground truth
(item qa-sp-01: citation events at offsets 0/96/234/419, each with
``start == end``, each delivered BEFORE its block's text deltas) — so
every stamped span is ``[block_start, block_start)``: empty. The #13
validator's strict-overlap rule attaches an empty span to NO sentence,
which regressed citation_support BELOW the legacy rule the #310 fix
replaced (48/334 = 14.4% vs 27.5%).

The pinned fix (mechanism FLAGGED for the orchestrator, decided here so
the contract is coherent): a citation's span is finalized at block CLOSE
(``content_block_stop``), and the citation SSE event is **BUFFERED until
its block closes** — resolved against the retrieved passages at ARRIVAL
(fail-fast on a poisoned ``document_index`` is unchanged), emitted at
``content_block_stop`` in arrival order, stamped with the block's full
``[start, end)`` extent. Buffering was chosen over
emit-immediately-plus-span-patch because every downstream consumer is
already position-independent for span-carrying citations — the #13
fold (``segment_answer_sentences``) attaches by span overlap, the #18
chips consume the fold's exported rule (finding #233), and the #22
service/permalink path only accumulates citation payloads — so moving
the event to block close needs NO new wire vocabulary and no
patch-event ordering rules, while a span-patch event would have added a
second citation-shaped event name (breaking the #230 parity guard) and
forced every consumer to handle late mutation. The #12 "transport
order" contract reading: citation events stay in transport order OF
THEIR BLOCK CLOSES, with arrival order preserved within a block.

Corollaries pinned here:

- **No zero-width span is ever emitted for a block that delivered
  text** — the smoke run's 202/202 signature becomes structurally
  impossible.
- A stream that ends before an open block closes (premature end) still
  terminates with the honest ``error`` event (finding #184) and never
  delivers a zero-width-span citation on the way out.
- A ``citations_delta`` for a block that was never opened (legacy /
  partial streams) keeps its existing behaviour: emitted immediately,
  span-less, legacy last-text-char attachment downstream.

Everything here is synthetic (the Aurelian-Basin universe): the
fixtures reproduce the journalled ARRIVAL ORDER, never the release
corpus text (DESIGN §2.1 shipping invariant).
"""

from __future__ import annotations

from typing import Any

from rag.citation_validator import segment_answer_sentences
from rag.generation import answer_stream_to_sse
from tests._citation_validator_fixtures import (
    text_event,
    transcript,
)
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    make_retrieved,
)

# ---------------------------------------------------------------------------
# The live-arrival-order fixture: two answer blocks, each opening with its
# citations_delta BEFORE any of its text deltas — the journalled qa-sp-01
# shape (citation at offset 0, then the block's text; next block's citation
# at the next block boundary, then its text).
# ---------------------------------------------------------------------------

FIRST_BLOCK_SENTENCES = (
    "Surface temperatures across the Aurelian Basin have very likely risen "
    "by one point nine degrees since the fictional baseline period.",
    "Reservoir inflows across the basin declined in twenty one of the last "
    "twenty five invented water years.",
    "Heat events across the basin have become more frequent since the "
    "fictional baseline period (high confidence).",
)

#: Block 0's delivered text: three sentences, streamed as four text deltas
#: (the first sentence split mid-word, as the live transport does).
FIRST_BLOCK_TEXT = " ".join(FIRST_BLOCK_SENTENCES)

SECOND_BLOCK_SENTENCE = "The basin's snowpack season has shortened by three invented weeks."

#: Block 1's delivered text (leading space: block concatenation forms the
#: answer text).
SECOND_BLOCK_TEXT = " " + SECOND_BLOCK_SENTENCE


def _block_start(index: int) -> dict[str, Any]:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }


def _block_stop(index: int) -> dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def _text_delta(index: int, text: str) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def _citations_delta(index: int, document_index: int, cited_text: str) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "citations_delta",
            "citation": {
                "type": "content_block_location",
                "document_index": document_index,
                "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
                "start_block_index": 0,
                "end_block_index": 1,
                "cited_text": cited_text,
            },
        },
    }


def _message_close() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 900,
                "output_tokens": 84,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 4200,
            },
        },
        {"type": "message_stop"},
    ]


def live_arrival_transport_stream() -> list[dict[str, Any]]:
    """The REAL journalled arrival order: each block's citations_delta
    lands at block OPEN, before any of the block's text deltas."""
    return [
        {"type": "message_start", "message": {"model": "claude-haiku-4-5", "role": "assistant"}},
        _block_start(0),
        _citations_delta(0, 0, FIRST_BLOCK_TEXT),  # citation FIRST — the live order
        _text_delta(0, FIRST_BLOCK_SENTENCES[0][:40]),
        _text_delta(0, FIRST_BLOCK_SENTENCES[0][40:] + " "),
        _text_delta(0, FIRST_BLOCK_SENTENCES[1] + " "),
        _text_delta(0, FIRST_BLOCK_SENTENCES[2]),
        _block_stop(0),
        _block_start(1),
        _citations_delta(1, 1, SECOND_BLOCK_SENTENCE),  # again at block open
        _text_delta(1, SECOND_BLOCK_TEXT),
        _block_stop(1),
        *_message_close(),
    ]


def _sse(events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return list(
        answer_stream_to_sse(
            events if events is not None else live_arrival_transport_stream(),
            retrieved=make_retrieved(3),
            corpus_vintage=CORPUS_VINTAGE,
        )
    )


def _citation_data(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["data"] for e in events if e["event"] == "citation"]


# ---------------------------------------------------------------------------
# 1. THE #322 pins: block-open citations get the block's FULL extent
# ---------------------------------------------------------------------------


class TestBlockOpenCitationSpans:
    def test_block_open_citation_spans_the_full_block_extent(self):
        """THE #322 pin: a citations_delta arriving at block OPEN is
        stamped with its block's full [start, end) char extent —
        finalized at content_block_stop, never at citation arrival.
        Under the #310 code this stamps [block_start, block_start):
        the smoke run's 202/202 zero-width signature."""
        citations = _citation_data(_sse())
        assert len(citations) == 2
        first_block_len = len(FIRST_BLOCK_TEXT)
        assert citations[0]["answer_block_start"] == 0
        assert citations[0]["answer_block_end"] == first_block_len
        assert citations[1]["answer_block_start"] == first_block_len
        assert citations[1]["answer_block_end"] == first_block_len + len(SECOND_BLOCK_TEXT)

    def test_no_zero_width_span_is_ever_emitted_for_a_block_with_text(self):
        """The structural invariant the smoke run falsified 202/202
        times: every delivered citation event whose block delivered text
        carries a strictly non-empty [start, end) span."""
        for citation in _citation_data(_sse()):
            assert citation["answer_block_end"] > citation["answer_block_start"], (
                "zero-width answer span delivered for a block that streamed "
                f"text: {citation['answer_block_start']!r} == "
                f"{citation['answer_block_end']!r} — the exact live defect "
                "the verification smoke measured (issue #322)"
            )

    def test_citation_events_are_delivered_at_block_close_not_block_open(self):
        """The FLAGGED emission-order decision, pinned: the citation
        event is BUFFERED until its block's content_block_stop, so on
        the delivered stream each block's text events precede its
        citation event(s) — no span-patch event, no new wire vocabulary
        (the #230 parity guard's set is untouched)."""
        events = _sse()
        assert [e["event"] for e in events] == [
            "text",
            "text",
            "text",
            "text",
            "citation",
            "text",
            "citation",
            "usage",
            "footer",
        ]

    def test_stamped_spans_slice_the_delivered_text_to_their_block(self):
        """Offsets index the concatenation of delivered text events:
        slicing by each citation's span recovers exactly its block's
        text — the property that makes the #13 overlap attachment
        honest."""
        events = _sse()
        full_text = "".join(e["data"]["text"] for e in events if e["event"] == "text")
        citations = _citation_data(events)
        assert (
            full_text[citations[0]["answer_block_start"] : citations[0]["answer_block_end"]]
            == FIRST_BLOCK_TEXT
        )
        assert (
            full_text[citations[1]["answer_block_start"] : citations[1]["answer_block_end"]]
            == SECOND_BLOCK_TEXT
        )

    def test_same_block_citations_flush_in_arrival_order_with_the_same_extent(self):
        """Two citations_delta events inside ONE block (both at open,
        as the live API may batch them): both buffered, both emitted at
        the block's close in arrival order, both spanning the block."""
        stream = [
            {"type": "message_start", "message": {"model": "claude-haiku-4-5"}},
            _block_start(0),
            _citations_delta(0, 2, FIRST_BLOCK_TEXT),
            _citations_delta(0, 0, FIRST_BLOCK_TEXT),
            _text_delta(0, FIRST_BLOCK_TEXT),
            _block_stop(0),
            *_message_close(),
        ]
        citations = _citation_data(_sse(stream))
        assert [c["document_index"] for c in citations] == [2, 0]
        for citation in citations:
            assert (citation["answer_block_start"], citation["answer_block_end"]) == (
                0,
                len(FIRST_BLOCK_TEXT),
            )

    def test_live_order_stream_attaches_every_covered_sentence_end_to_end(self):
        """Seam-to-validator round trip on the LIVE arrival order — the
        exact production path the smoke run measured at 14.4%: the SSE
        output feeds segment_answer_sentences and every sentence of the
        cited block carries its citation. Under the zero-width defect
        NO sentence attaches at all."""
        events = _sse()
        body = [e for e in events if e["event"] in ("text", "citation")]
        sentences = segment_answer_sentences(transcript(*body))
        assert [s.text for s in sentences] == [*FIRST_BLOCK_SENTENCES, SECOND_BLOCK_SENTENCE]
        assert [s.document_indices for s in sentences] == [(0,), (0,), (0,), (1,)]


# ---------------------------------------------------------------------------
# 2. Premature termination: honesty holds, zero-width never leaks
# ---------------------------------------------------------------------------


class TestPrematureEndWithBufferedCitation:
    def test_truncated_stream_never_delivers_a_zero_width_citation(self):
        """A stream that dies with a block still open (its citation
        arrived at open, its extent unknowable) terminates with the
        honest `error` event (finding #184) and never delivers a
        zero-width-span citation on the way out. The implementer may
        drop the buffered citation or stamp the truncated extent —
        either way `start == end` never reaches the client. Under the
        #310 code the zero-width citation is emitted before the error."""
        truncated = [
            {"type": "message_start", "message": {"model": "claude-haiku-4-5"}},
            _block_start(0),
            _citations_delta(0, 0, FIRST_BLOCK_TEXT),
            _text_delta(0, FIRST_BLOCK_SENTENCES[0]),
            # stream ends: no block stop, no message_stop
        ]
        events = _sse(truncated)
        assert events[-1]["event"] == "error"
        assert all(e["event"] != "footer" for e in events)
        for citation in _citation_data(events):
            assert citation.get("answer_block_start") != citation.get("answer_block_end"), (
                "a zero-width-span citation leaked from a truncated stream"
            )


# ---------------------------------------------------------------------------
# 3. The ordering contracts the buffering mechanism relies on
#    (green today — they document WHY moving the citation event to block
#    close is safe for every consumer of the delivered stream)
# ---------------------------------------------------------------------------


class TestFoldIsPositionIndependentForSpanCitations:
    def test_validator_attachment_follows_the_span_not_arrival_position(self):
        """The #13 fold attaches span-carrying citations by [start, end)
        overlap regardless of WHERE the event sits in the delivered
        transcript — a citation-before-text transcript attaches
        identically to citation-after-text. This is the property that
        makes buffering (and any historical event position in cached
        journals) equivalent to the fold."""
        span_citation = {
            "event": "citation",
            "data": {
                "type": "content_block_location",
                "document_index": 0,
                "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
                "start_block_index": 0,
                "end_block_index": 1,
                "cited_text": FIRST_BLOCK_TEXT,
                "answer_block_start": 0,
                "answer_block_end": len(FIRST_BLOCK_TEXT),
            },
        }
        citation_first = transcript(span_citation, text_event(FIRST_BLOCK_TEXT))
        citation_last = transcript(text_event(FIRST_BLOCK_TEXT), span_citation)
        first = [s.document_indices for s in segment_answer_sentences(citation_first)]
        last = [s.document_indices for s in segment_answer_sentences(citation_last)]
        assert first == last == [(0,), (0,), (0,)]

    def test_ui_chips_follow_spans_not_arrival_position(self):
        """The #18 chips consume the fold's exported pairing (finding
        #233), so they inherit the same position independence — the page
        shows the citation on every covered sentence no matter where the
        event arrived."""
        from ui.render_model import build_citation_chips

        span_citation = {
            "event": "citation",
            "data": {
                "type": "content_block_location",
                "document_index": 0,
                "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
                "start_block_index": 0,
                "end_block_index": 1,
                "cited_text": FIRST_BLOCK_TEXT,
                "answer_block_start": 0,
                "answer_block_end": len(FIRST_BLOCK_TEXT),
            },
        }
        chips = build_citation_chips(transcript(span_citation, text_event(FIRST_BLOCK_TEXT)))
        assert [(c.sentence_index, c.document_index) for c in chips] == [(0, 0), (1, 0), (2, 0)]


class TestBufferingChangesNeitherFailFastNorLegacyPaths:
    def test_block_open_poison_citation_still_fails_fast(self):
        """Resolution happens at ARRIVAL even though emission is
        buffered: a citations_delta at block open with an out-of-range
        document_index terminates the stream with the `error` event
        before any of the block's text streams — buffering must never
        delay the guaranteed-resolution check (finding #185)."""
        stream = [
            {"type": "message_start", "message": {"model": "claude-haiku-4-5"}},
            _block_start(0),
            _citations_delta(0, 99, FIRST_BLOCK_TEXT),
            _text_delta(0, FIRST_BLOCK_TEXT),
            _block_stop(0),
            *_message_close(),
        ]
        events = _sse(stream)
        assert events[-1]["event"] == "error"
        assert all(e["event"] != "footer" for e in events)
        assert all(e["event"] != "citation" for e in events)

    def test_unopened_block_citation_keeps_the_spanless_legacy_emission(self):
        """A citations_delta for a block that was never opened via
        content_block_start (legacy / partial streams) has no block
        close to wait for: it is emitted immediately, WITHOUT span
        fields, and falls back to the legacy last-text-char attachment
        downstream — exactly the existing pinned behaviour."""
        stream = [
            {"type": "message_start", "message": {"model": "claude-haiku-4-5"}},
            _citations_delta(0, 0, FIRST_BLOCK_TEXT),
            _text_delta(0, FIRST_BLOCK_TEXT),
            *_message_close(),
        ]
        events = _sse(stream)
        assert [e["event"] for e in events] == ["citation", "text", "usage", "footer"]
        (citation,) = _citation_data(events)
        assert "answer_block_start" not in citation
        assert "answer_block_end" not in citation
