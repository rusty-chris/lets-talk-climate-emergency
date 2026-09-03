"""Sentence segmentation + sentence<->cited-block pairing (issue #13) — RED.

Unit tier, pure: ``segment_answer_sentences`` and
``build_entailment_pairs`` are functions over the delivered #12 SSE
transcript and the resolved ``GroundedAnswer.cited_passages`` — no
adapter, no network. The segmentation deliberately REUSES the ingestion
sentence-tokenizer conventions (``ingestion.chunk.split_sentences``):
the repo has exactly one documented sentence-boundary rule, and the
validator's sentence units must match it, catalogued failure modes and
all.
"""

from __future__ import annotations

from ingestion.chunk import split_sentences
from rag.citation_validator import (
    build_entailment_pairs,
    segment_answer_sentences,
)
from rag.generation import build_response_footer
from tests._citation_validator_fixtures import (
    CORPUS_VINTAGE,
    CORRUPTED_SENTENCE,
    GREETING_SENTENCE,
    SUPPORTED_SENTENCE,
    UNCITED_FACTUAL_SENTENCE,
    citation_sse_event,
    corrupted_answer_events,
    make_grounded_answer,
    text_event,
    transcript,
)


class TestSegmentation:
    def test_segmentation_matches_ingestion_tokenizer_conventions(self):
        """The validator's sentence boundaries ARE the ingestion
        tokenizer's boundaries: for a furniture-free answer delivered
        as arbitrary text-delta chunking, the segmented texts equal
        ``split_sentences`` over the concatenated text — including the
        decimal-number non-split ("1.9 degrees" stays one sentence)."""
        full_text = (
            "Surface temperatures rose by 1.9 degrees across the basin. "
            "Reservoir inflows declined in 21 of the last 25 invented water years. "
            "Attribution studies compare the observed decline against counterfactuals."
        )
        # Deliver in awkward chunks: sentence boundaries never align with
        # delta boundaries, exactly like a real token stream.
        events = transcript(
            text_event(full_text[:37]),
            text_event(full_text[37:100]),
            text_event(full_text[100:]),
        )
        sentences = segment_answer_sentences(events)
        assert [s.text for s in sentences] == split_sentences(full_text)
        assert len(sentences) == 3

    def test_sentence_indices_are_delivery_order_positions(self):
        sentences = segment_answer_sentences(corrupted_answer_events())
        assert [s.index for s in sentences] == list(range(len(sentences)))
        assert [s.text for s in sentences] == [
            GREETING_SENTENCE,
            SUPPORTED_SENTENCE,
            CORRUPTED_SENTENCE,
            UNCITED_FACTUAL_SENTENCE,
        ]

    def test_citation_attaches_to_the_sentence_in_progress(self):
        """Each citation event attaches to the sentence being emitted
        when it arrived — including the arrived-just-after-the-period
        edge (the transport emits citations_delta AFTER the text it
        cites, so a citation between sentences belongs to the sentence
        just completed, never the one about to start)."""
        sentences = segment_answer_sentences(corrupted_answer_events())
        by_text = {s.text: s for s in sentences}
        assert by_text[SUPPORTED_SENTENCE].document_indices == (0,)
        assert by_text[CORRUPTED_SENTENCE].document_indices == (1,)
        assert by_text[GREETING_SENTENCE].document_indices == ()
        assert by_text[UNCITED_FACTUAL_SENTENCE].document_indices == ()

    def test_two_citations_on_one_sentence_both_attach(self):
        events = transcript(
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(0),
            citation_sse_event(1),
        )
        (sentence,) = segment_answer_sentences(events)
        assert sentence.document_indices == (0, 1)

    def test_repeated_same_document_citation_attaches_once(self):
        """Review finding #207: native citations routinely cite the SAME
        document for multiple adjacent spans of one sentence — two
        citation events, one chip. The duplicate attachment must dedupe
        (order-preserving, first arrival kept): (sentence, document) is
        how the #18 UI identifies a chip, so a duplicate index buys a
        duplicate entailment pair, duplicate judge spend and duplicate
        badge events downstream. Genuinely-distinct multi-document
        sentences (the chip-honesty case, pinned above) are untouched."""
        events = transcript(
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(0),
            citation_sse_event(0),
        )
        (sentence,) = segment_answer_sentences(events)
        assert sentence.document_indices == (0,)

        # Order-preserving first occurrence when distinct chips interleave.
        events = transcript(
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(1),
            citation_sse_event(0),
            citation_sse_event(1),
        )
        (sentence,) = segment_answer_sentences(events)
        assert sentence.document_indices == (1, 0)

    def test_greeting_furniture_is_not_factual(self):
        """Pinned furniture exclusions: interactional sentences carry no
        checkable claim and are never validated."""
        events = transcript(
            text_event("Hello! "),
            text_event("Thanks for asking. "),
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(0),
            text_event(" I hope this helps!"),
        )
        sentences = segment_answer_sentences(events)
        factual_flags = {s.text: s.factual for s in sentences}
        assert factual_flags["Hello!"] is False
        assert factual_flags["Thanks for asking."] is False
        assert factual_flags["I hope this helps!"] is False
        assert factual_flags[SUPPORTED_SENTENCE] is True

    def test_footer_event_contributes_no_sentence(self):
        """The §3.5 footer arrives as its own ``footer`` event and never
        becomes answer text — the transcript's footer must not appear
        among the segmented sentences."""
        sentences = segment_answer_sentences(corrupted_answer_events())
        footer_text = build_response_footer(CORPUS_VINTAGE)
        assert all(footer_text not in s.text for s in sentences)

    def test_footer_text_in_a_text_delta_is_never_factual(self):
        """Belt and braces: even if the footer template text leaks into
        the answer's own text deltas, it is furniture, not a factual
        sentence to validate."""
        events = transcript(
            text_event(SUPPORTED_SENTENCE + " "),
            citation_sse_event(0),
            text_event(build_response_footer(CORPUS_VINTAGE)),
        )
        sentences = segment_answer_sentences(events)
        footer_sentences = [s for s in sentences if "measured" in s.text and "rate" in s.text]
        assert footer_sentences, "footer text should still segment into sentences"
        assert all(s.factual is False for s in footer_sentences)

    def test_cited_sentence_is_always_factual(self):
        """A citation attached means the model asserted something from a
        source — even furniture-looking text gets verified once cited."""
        events = transcript(
            text_event("Thanks for asking."),
            citation_sse_event(0),
        )
        (sentence,) = segment_answer_sentences(events)
        assert sentence.document_indices == (0,)
        assert sentence.factual is True

    def test_uncited_factual_sentence_is_factual(self):
        sentences = segment_answer_sentences(corrupted_answer_events())
        by_text = {s.text: s for s in sentences}
        assert by_text[UNCITED_FACTUAL_SENTENCE].factual is True


class TestPairing:
    def test_sentence_pairing_covers_every_factual_sentence(self):
        """Issue #13 TDD plan step 1: each CITED factual sentence maps to
        its cited block(s) — one pair per (sentence, block), every cited
        sentence covered; non-factual sentences excluded; uncited
        factual sentences contribute no pair (they are auto-flagged
        without an entailment call instead)."""
        answer = make_grounded_answer()
        sentences = segment_answer_sentences(corrupted_answer_events())
        pairs = build_entailment_pairs(sentences, answer.cited_passages)

        cited_factual = [s for s in sentences if s.factual and s.document_indices]
        expected_pair_keys = {
            (s.index, document_index)
            for s in cited_factual
            for document_index in s.document_indices
        }
        assert {(p.sentence_index, p.document_index) for p in pairs} == expected_pair_keys
        # Coverage: every cited factual sentence appears in the pair set.
        assert {p.sentence_index for p in pairs} == {s.index for s in cited_factual}
        # Exclusions: furniture and uncited sentences never pair.
        furniture_or_uncited = {
            s.index for s in sentences if not s.factual or not s.document_indices
        }
        assert not ({p.sentence_index for p in pairs} & furniture_or_uncited)

    def test_pair_indices_are_stable_request_positions(self):
        answer = make_grounded_answer()
        sentences = segment_answer_sentences(corrupted_answer_events())
        pairs = build_entailment_pairs(sentences, answer.cited_passages)
        assert [p.pair_index for p in pairs] == list(range(len(pairs)))

    def test_pairs_carry_the_cited_block_bodies(self):
        """The block side of each pair is the cited passage's stored
        body (``payload["body"]``) joined on document_index — the text
        the entailment judge reads."""
        answer = make_grounded_answer()
        sentences = segment_answer_sentences(corrupted_answer_events())
        pairs = build_entailment_pairs(sentences, answer.cited_passages)
        bodies = {p.document_index: p.payload["body"] for p in answer.cited_passages}
        assert pairs, "the corrupted-answer fixture must produce pairs"
        for pair in pairs:
            assert pair.block_text == bodies[pair.document_index]
            assert pair.sentence_text in (SUPPORTED_SENTENCE, CORRUPTED_SENTENCE)

    def test_two_citations_one_sentence_yield_two_pairs(self):
        events = transcript(
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(0),
            citation_sse_event(1),
        )
        answer = make_grounded_answer(document_indices=(0, 1))
        sentences = segment_answer_sentences(events)
        pairs = build_entailment_pairs(sentences, answer.cited_passages)
        assert len(pairs) == 2
        assert {p.document_index for p in pairs} == {0, 1}
        assert len({p.sentence_index for p in pairs}) == 1

    def test_repeated_same_document_citation_yields_one_pair(self):
        """Review finding #207, the spend half: two same-document spans on
        one sentence must produce ONE entailment pair — the block body is
        the dominant token cost of the batched call, and the judge gains
        nothing from answering the same (sentence, block) question twice
        (§9 ~$0.003/query). One pair also forecloses the split-verdict
        case (the same chip counting supported AND badged)."""
        events = transcript(
            text_event(SUPPORTED_SENTENCE),
            citation_sse_event(0),
            citation_sse_event(0),
        )
        answer = make_grounded_answer(document_indices=(0,))
        sentences = segment_answer_sentences(events)
        pairs = build_entailment_pairs(sentences, answer.cited_passages)
        assert len(pairs) == 1
        (pair,) = pairs
        assert (pair.sentence_index, pair.document_index) == (0, 0)
        assert pair.sentence_text == SUPPORTED_SENTENCE


class TestCitationAssignmentExport:
    """Review finding #233 RED — the assignment rule is exported, not mirrored.

    ui/render_model.py hand-copies the validator's citation-to-sentence
    assignment (~40 lines, line-for-line _sentence_spans plus the
    last-non-whitespace-char rule). If the validator's rule changes again
    (it already did once - finding #207) the UI copy drifts silently and
    a judged citation vanishes from the page without a trace. The
    validator must export the pairing as public API - pinned HERE once,
    consumed by both the validator and the UI.
    """

    def test_segmentation_exposes_citation_assignment_in_arrival_order(self):
        # Imported inside the test: the export does not exist yet (RED);
        # a module-level import would fail collection for the whole file.
        from rag.citation_validator import citation_sentence_assignments

        events = transcript(
            text_event(SUPPORTED_SENTENCE + " "),
            citation_sse_event(0),
            citation_sse_event(0),  # finding #207 duplicate: deduped, once
            text_event(CORRUPTED_SENTENCE),
            citation_sse_event(1),
        )
        assignments = citation_sentence_assignments(events)

        # Arrival-ordered (sentence_index, document_index) pairs, deduped
        # per sentence exactly as segment_answer_sentences dedupes.
        assert list(assignments) == [(0, 0), (1, 1)]

    def test_assignment_export_agrees_with_segmentation_for_shared_input(self):
        """Identity pin: for the same transcript, the exported pairing is
        exactly the (index, document_indices) structure the segmentation
        reports - one rule, two consumers, zero drift."""
        from rag.citation_validator import citation_sentence_assignments

        events = corrupted_answer_events()
        assignments = citation_sentence_assignments(events)
        sentences = segment_answer_sentences(events)

        expected = [
            (sentence.index, document_index)
            for sentence in sentences
            for document_index in sentence.document_indices
        ]
        # Same pairs; arrival order for a stream whose citations arrive
        # in sentence order IS sentence order.
        assert list(assignments) == expected
