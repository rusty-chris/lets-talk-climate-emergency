"""Span-aware citation attachment (review finding #310, blocker) — RED.

The release run's citation_support gate failed at 27.5% because the
citation-to-sentence attachment rule structurally discards citations.
The Claude citations API is BLOCK-scoped: with citations enabled the
answer content splits into text blocks, a cited block's text is the
whole answer span the citation supports, and in streaming the
``citations_delta`` arrives after the block's text deltas. The current
rule attaches each ``citation`` SSE event to exactly ONE sentence (the
last text before the event), so a cited block spanning N sentences has
N-1 sentences scored ``uncited`` — the qa-sp-06 signature (7 citation
events, 4+ verbatim-cited sentences, scored 2/10).

Mechanism, decided from the wire ground truth (FLAGGED for the
orchestrator): a ``citations_delta`` carries ``cited_text`` /
``document_index`` / ``document_title`` and a SOURCE location
(``content_block_location``: ``start_block_index``/``end_block_index``
into the cited document) — it carries NO answer-text span. The answer
span of a citation is the extent of the ANSWER content block it belongs
to, delimited by the ``content_block_start``/``content_block_stop``
transport events that ``rag.generation.answer_stream_to_sse`` currently
drops. The SSE wire vocabulary therefore cannot express block extents
today, so the fix EXTENDS the citation event payload:

- ``answer_stream_to_sse`` stamps every ``citation`` SSE event with
  ``answer_block_start`` / ``answer_block_end`` — the ``[start, end)``
  character offsets of the citation's answer content block within the
  concatenation of all delivered ``text`` event texts. No new SSE event
  names (the #230 wire-vocabulary parity guard and the pinned
  text/citation/usage/footer envelope are untouched); the request on
  the provider seam is unchanged, so ``canonical_request_hash`` replay
  keys are unaffected.
- ``segment_answer_sentences`` attaches a span-carrying citation to
  EVERY sentence overlapping ``[answer_block_start, answer_block_end)``
  — pairing, the pooled support denominator, badges and the #18 UI
  chips (via ``citation_sentence_assignments`` /
  ``build_citation_chips``) all follow, because they are derived from
  the one exported rule (finding #233).
- A citation event WITHOUT the span fields (older transcripts, cached
  journals) attaches by the legacy last-text-character rule — the
  existing suite pins that path and stays green.

Everything here is synthetic (the Aurelian-Basin universe); the
qa-sp-06-shaped case reproduces the SHAPE of the defect — a
multi-sentence verbatim-cited block with one citation event — never the
release corpus text (DESIGN §2.1 shipping invariant).
"""

from __future__ import annotations

from typing import Any

import pytest

from rag.citation_validator import (
    UNVERIFIED_REASON_UNCITED,
    ValidatorConfig,
    build_entailment_pairs,
    citation_sentence_assignments,
    segment_answer_sentences,
    validate_exchange,
    validation_sse_events,
)
from rag.generation import (
    GenerationConfig,
    GroundedAnswer,
    answer_stream_to_sse,
    build_generation_request,
    build_response_footer,
    resolve_citations,
)
from rag.provider import (
    AnswerWithCitations,
    Citation,
    FakeAdapter,
    ReplayAdapter,
    StructuredResult,
    canonical_request_hash,
)
from rag.retrieval import RetrievedPassages
from tests._citation_validator_fixtures import (
    text_event,
    transcript,
    verdicts_output,
)
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    make_passage,
    make_retrieved,
)
from tests.conftest import FIXTURES_ROOT

REPLAY_FIXTURES_DIR = FIXTURES_ROOT / "replay"

# ---------------------------------------------------------------------------
# The qa-sp-06-shaped fixture: ONE cited answer block spanning THREE
# sentences, every sentence verbatim in the cited chunk body, ONE
# citation event — plus a trailing uncited factual sentence so the
# uncited path is visible alongside.
# ---------------------------------------------------------------------------

BLOCK_SENTENCES = (
    "Surface temperatures across the Aurelian Basin have very likely risen "
    "by one point nine degrees since the fictional baseline period.",
    "Reservoir inflows across the basin declined in twenty one of the last "
    "twenty five invented water years.",
    "Heat events across the basin have become more frequent since the "
    "fictional baseline period (high confidence).",
)

#: The cited answer block's delivered text: the three sentences.
CITED_BLOCK_TEXT = " ".join(BLOCK_SENTENCES)

TRAILING_UNCITED_SENTENCE = "The basin's snowpack season has shortened by three invented weeks."

#: The cited chunk body carries every block sentence verbatim — the
#: qa-sp-06 signature: an honest judge over correctly-attached pairs
#: entails all of them.
CITED_CHUNK_BODY = CITED_BLOCK_TEXT


def spanned_citation_sse_event(
    document_index: int,
    *,
    answer_block_start: int,
    answer_block_end: int,
    cited_text: str = CITED_CHUNK_BODY,
) -> dict[str, Any]:
    """One delivered ``citation`` SSE event carrying its answer-block span.

    The #310 payload extension: alongside the resolved transport fields,
    ``answer_block_start``/``answer_block_end`` give the ``[start, end)``
    char offsets of the citation's answer content block within the
    concatenated delivered ``text`` events.
    """
    return {
        "event": "citation",
        "data": {
            "type": "content_block_location",
            "document_index": document_index,
            "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
            "start_block_index": 0,
            "end_block_index": 1,
            "cited_text": cited_text,
            "answer_block_start": answer_block_start,
            "answer_block_end": answer_block_end,
        },
    }


def qa_sp_06_shaped_transcript() -> list[dict[str, Any]]:
    """Delivered SSE transcript: 3-sentence cited block, 1 citation event,
    then an uncited factual sentence."""
    return transcript(
        text_event(CITED_BLOCK_TEXT),
        spanned_citation_sse_event(
            0,
            answer_block_start=0,
            answer_block_end=len(CITED_BLOCK_TEXT),
        ),
        text_event(" " + TRAILING_UNCITED_SENTENCE),
    )


def qa_sp_06_shaped_answer() -> GroundedAnswer:
    """A GroundedAnswer whose single cited passage body carries the
    block's three sentences verbatim (resolution via the production
    ``resolve_citations``, so payload shapes match the real seam)."""
    passage = make_passage(0, body=CITED_CHUNK_BODY)
    retrieved = RetrievedPassages(passages=(passage,))
    answer = AnswerWithCitations(
        text=CITED_BLOCK_TEXT + " " + TRAILING_UNCITED_SENTENCE,
        citations=(
            Citation(
                cited_text=CITED_CHUNK_BODY,
                document_index=0,
                document_title="Meridian Climate Assessment Cycle 3 (invented)",
                start_block_index=0,
                end_block_index=1,
            ),
        ),
    )
    return GroundedAnswer(
        text=answer.text,
        cited_passages=resolve_citations(answer, retrieved),
        footer=build_response_footer(CORPUS_VINTAGE),
    )


# ---------------------------------------------------------------------------
# 1. Validator: span-aware attachment covers every overlapped sentence
# ---------------------------------------------------------------------------


class TestSpanAwareAttachment:
    def test_spanning_citation_attaches_to_all_covered_sentences(self):
        """THE #310 pin: one citation whose answer-block span covers N
        sentences attaches to ALL N — never only the last text before
        the event."""
        sentences = segment_answer_sentences(qa_sp_06_shaped_transcript())
        assert [s.text for s in sentences] == [
            *BLOCK_SENTENCES,
            TRAILING_UNCITED_SENTENCE,
        ]
        for covered in sentences[:3]:
            assert covered.document_indices == (0,), (
                f"sentence {covered.index} is inside the cited block's span "
                "and must carry its citation"
            )
        assert sentences[3].document_indices == ()

    def test_sentences_outside_the_span_never_attach(self):
        """A block-1 citation must not bleed onto block-0 text: only
        sentences overlapping [answer_block_start, answer_block_end)
        attach."""
        first = BLOCK_SENTENCES[0]
        second = BLOCK_SENTENCES[1]
        full = first + " " + second
        events = transcript(
            text_event(full),
            spanned_citation_sse_event(
                1,
                answer_block_start=len(first),
                answer_block_end=len(full),
            ),
        )
        sentences = segment_answer_sentences(events)
        assert sentences[0].document_indices == ()
        assert sentences[1].document_indices == (1,)

    def test_span_attachment_dedupes_repeated_same_document_citations(self):
        """Finding #207 composes with spans: two same-document citation
        events over one block still yield ONE attachment per covered
        sentence — chips, pairs and judge spend never double."""
        events = transcript(
            text_event(CITED_BLOCK_TEXT),
            spanned_citation_sse_event(
                0, answer_block_start=0, answer_block_end=len(CITED_BLOCK_TEXT)
            ),
            spanned_citation_sse_event(
                0, answer_block_start=0, answer_block_end=len(CITED_BLOCK_TEXT)
            ),
        )
        sentences = segment_answer_sentences(events)
        for sentence in sentences:
            assert sentence.document_indices == (0,)

    def test_spanless_citation_keeps_the_legacy_attachment_rule(self):
        """Back-compat pin, kept DELIBERATE: a citation event carrying no
        answer-span fields (older transcripts, cached journals) attaches
        by the legacy last-non-whitespace-text rule — one sentence."""
        from tests._citation_validator_fixtures import citation_sse_event

        events = transcript(
            text_event(BLOCK_SENTENCES[0] + " " + BLOCK_SENTENCES[1]),
            citation_sse_event(0),
        )
        sentences = segment_answer_sentences(events)
        assert sentences[0].document_indices == ()
        assert sentences[1].document_indices == (0,)


# ---------------------------------------------------------------------------
# 2. Pairing, pooled denominator, badges and the UI chip path all follow
# ---------------------------------------------------------------------------


class TestSpanAttachmentDerivedSurfaces:
    def test_pairing_yields_one_pair_per_covered_sentence(self):
        """Issue #310 required test 1: a 3-sentence cited block with a
        single trailing citation event -> THREE entailment pairs, each
        carrying the cited block's body."""
        answer = qa_sp_06_shaped_answer()
        sentences = segment_answer_sentences(qa_sp_06_shaped_transcript())
        pairs = build_entailment_pairs(sentences, answer.cited_passages)
        assert len(pairs) == 3
        assert [p.sentence_index for p in pairs] == [0, 1, 2]
        assert {p.document_index for p in pairs} == {0}
        for pair, sentence_text in zip(pairs, BLOCK_SENTENCES, strict=True):
            assert pair.sentence_text == sentence_text
            assert pair.block_text == CITED_CHUNK_BODY

    def test_citation_sentence_assignments_export_covers_the_span(self):
        """The #18 UI chip join consumes the SAME exported rule (finding
        #233): the spanning citation appears once per covered sentence,
        in delivery order."""
        assignments = citation_sentence_assignments(qa_sp_06_shaped_transcript())
        assert list(assignments) == [(0, 0), (1, 0), (2, 0)]

    def test_ui_chips_follow_the_span_rule(self):
        """One chip per covered (sentence, document): the page shows the
        citation on every sentence the model actually cited."""
        from ui.render_model import build_citation_chips

        chips = build_citation_chips(qa_sp_06_shaped_transcript())
        assert [(c.sentence_index, c.document_index) for c in chips] == [
            (0, 0),
            (1, 0),
            (2, 0),
        ]
        for chip in chips:
            assert chip.quote == CITED_CHUNK_BODY

    def test_qa_sp_06_shaped_exchange_scores_every_covered_sentence_supported(self):
        """Issue #310 required test 4, the regression that pins the gate:
        with the judge entailing every pair (the sentences ARE verbatim
        in the cited body), every covered sentence counts supported —
        support_rate 3/4 with ONLY the genuinely-uncited trailing
        sentence flagged. Under the block-collapse defect this exchange
        scored 1/4 with two false 'uncited' flags."""
        adapter = FakeAdapter()
        adapter.queue(
            "structured",
            StructuredResult(
                value=verdicts_output(True, True, True),
                usage={"input_tokens": 500, "output_tokens": 30},
            ),
        )
        outcome = validate_exchange(
            adapter,
            qa_sp_06_shaped_answer(),
            qa_sp_06_shaped_transcript(),
            config=ValidatorConfig(),
        )
        assert outcome.validated, outcome.degraded_reason
        assert outcome.support_rate == pytest.approx(3 / 4)
        assert [(f.sentence_index, f.document_index, f.reason) for f in outcome.unverified] == [
            (3, None, UNVERIFIED_REASON_UNCITED)
        ]
        # Exactly one batched call: span attachment changes the pair set,
        # never the one-call cost model.
        assert len(adapter.calls_to("structured")) == 1

        # Badge coherence: the only badge is the honest uncited flag —
        # no false 'uncited' badge on a covered sentence.
        badges = list(validation_sse_events(outcome))
        assert [b["data"]["sentence_index"] for b in badges] == [3]
        assert badges[0]["data"]["reason"] == UNVERIFIED_REASON_UNCITED


# ---------------------------------------------------------------------------
# 3. The streaming seam: answer_stream_to_sse preserves block extents
# ---------------------------------------------------------------------------


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


#: Block 1's delivered text (leading space: block concatenation forms
#: the answer text).
SECOND_BLOCK_TEXT = " " + TRAILING_UNCITED_SENTENCE


def multi_block_transport_stream() -> list[dict[str, Any]]:
    """The block-scoped shape the citations API actually streams: two
    answer text blocks, each closed by its own citations_delta after its
    text deltas, with explicit content_block_start/stop boundaries."""
    return [
        {"type": "message_start", "message": {"model": "claude-haiku-4-5", "role": "assistant"}},
        _block_start(0),
        _text_delta(0, BLOCK_SENTENCES[0] + " "),
        _text_delta(0, BLOCK_SENTENCES[1] + " "),
        _text_delta(0, BLOCK_SENTENCES[2]),
        _citations_delta(0, 0, CITED_CHUNK_BODY),
        _block_stop(0),
        _block_start(1),
        _text_delta(1, SECOND_BLOCK_TEXT),
        _citations_delta(1, 1, TRAILING_UNCITED_SENTENCE),
        _block_stop(1),
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


class TestStreamPreservesBlockExtents:
    def _sse(self) -> list[dict[str, Any]]:
        return list(
            answer_stream_to_sse(
                multi_block_transport_stream(),
                retrieved=make_retrieved(3),
                corpus_vintage=CORPUS_VINTAGE,
            )
        )

    def test_citation_events_carry_their_answer_block_char_span(self):
        """Issue #310 required test 2: block extents survive the SSE
        translation — each citation event is stamped with the [start,
        end) char offsets of ITS answer block within the concatenated
        delivered text."""
        events = self._sse()
        citations = [e["data"] for e in events if e["event"] == "citation"]
        assert len(citations) == 2
        first_block_len = len(CITED_BLOCK_TEXT)
        assert citations[0]["answer_block_start"] == 0
        assert citations[0]["answer_block_end"] == first_block_len
        assert citations[1]["answer_block_start"] == first_block_len
        assert citations[1]["answer_block_end"] == first_block_len + len(SECOND_BLOCK_TEXT)

    def test_span_extension_adds_no_new_event_names(self):
        """The mechanism is a citation-payload extension, not new wire
        vocabulary: the pinned text/citation/usage/footer envelope holds
        for the multi-block stream (the #230 parity guard's set is
        untouched)."""
        events = self._sse()
        assert [e["event"] for e in events] == [
            "text",
            "text",
            "text",
            "citation",
            "text",
            "citation",
            "usage",
            "footer",
        ]

    def test_stamped_spans_delimit_the_delivered_text_they_cover(self):
        """The spans are offsets into the concatenation of the delivered
        text events — slicing the concatenated text by each citation's
        span recovers exactly its block's text."""
        events = self._sse()
        full_text = "".join(e["data"]["text"] for e in events if e["event"] == "text")
        citations = [e["data"] for e in events if e["event"] == "citation"]
        assert (
            full_text[citations[0]["answer_block_start"] : citations[0]["answer_block_end"]]
            == CITED_BLOCK_TEXT
        )
        assert (
            full_text[citations[1]["answer_block_start"] : citations[1]["answer_block_end"]]
            == SECOND_BLOCK_TEXT
        )

    def test_streamed_multi_block_answer_attaches_end_to_end(self):
        """Seam-to-validator round trip: the SSE output of the
        multi-block stream feeds segment_answer_sentences and every
        sentence of the cited block carries the citation — the
        production path the release run exercised, healed."""
        events = self._sse()
        body = [e for e in events if e["event"] in ("text", "citation")]
        sentences = segment_answer_sentences(transcript(*body))
        assert [s.document_indices for s in sentences] == [(0,), (0,), (0,), (1,)]


# ---------------------------------------------------------------------------
# 4. Replay pin against a REAL recorded stream (issue #310 required test 3)
# ---------------------------------------------------------------------------

QUESTION = "How much has the invented basin warmed?"


def _generation_stream_recorded() -> bool:
    """True when a genuine recorded generate_stream exchange exists for
    the canonical fixture request (the #162 rule: hand-authored replay
    fixtures are never a substitute for a recording). No recording
    exists today — issue #310 requires a session to pin the block
    granularity Haiku actually emits."""
    request = build_generation_request(
        make_retrieved(3),
        QUESTION,
        config=GenerationConfig(),
    )
    fixture = REPLAY_FIXTURES_DIR / f"{canonical_request_hash('generate_stream', request)}.json"
    return fixture.is_file()


@pytest.mark.skipif(
    not _generation_stream_recorded(),
    reason="awaiting a generation recording session (issue #310 required test 3) — "
    "record via CLIMATE_CHAT_RECORD=1 with a live key; the recording must pin "
    "the real block granularity Haiku emits with citations enabled",
)
def test_recorded_stream_block_extents_survive_the_sse_translation():
    """Against the recorded transport stream: every SSE citation event's
    stamped answer-block span matches the extent of the recorded content
    block its citations_delta belonged to."""
    request = build_generation_request(make_retrieved(3), QUESTION, config=GenerationConfig())
    recorded = list(ReplayAdapter(REPLAY_FIXTURES_DIR).generate_stream(**request))

    # Reconstruct each block's char extent from the recorded transport.
    block_spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    block_start: dict[int, int] = {}
    for event in recorded:
        if event.get("type") == "content_block_start":
            block_start[event["index"]] = cursor
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                cursor += len(delta.get("text", ""))
        elif event.get("type") == "content_block_stop":
            index = event["index"]
            block_spans[index] = (block_start.get(index, 0), cursor)

    citation_block_indices = [
        e["index"]
        for e in recorded
        if e.get("type") == "content_block_delta"
        and (e.get("delta") or {}).get("type") == "citations_delta"
    ]
    assert citation_block_indices, "recorded exchange carries no citations"

    sse = list(
        answer_stream_to_sse(recorded, retrieved=make_retrieved(3), corpus_vintage=CORPUS_VINTAGE)
    )
    citations = [e["data"] for e in sse if e["event"] == "citation"]
    for data, block_index in zip(citations, citation_block_indices, strict=True):
        assert (data["answer_block_start"], data["answer_block_end"]) == block_spans[block_index]
