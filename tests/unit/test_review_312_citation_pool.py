"""Review #312 red phase (Fable): the citation_support denominator pools
sentences that carry no checkable claim against the corpus.

Two poison classes from the 2026-09-04/05 live release run (issue #312):

1. **Honest generation-level declines pooled in.** 12 answered ``qa-na-*``
   items produced zero-citation honest declines and contributed 64
   factual-counted sentences with 0 supported (12.2% of the 524-sentence
   pool). Those sentences are meta-statements about the retrieval set and
   referral advice — unentailable by corpus chunks *by construction* —
   and the same items are already failed by the refusal gate, so pooling
   them double-counts one product failure across two gates. Pinned here:
   a generation-level decline contributes ZERO factual sentences to the
   citation_support pool (visible in the evidence, excluded from the
   denominator), and the runner marks the decline on the validation
   record.

2. **Furniture classifier too narrow.** ``_FURNITURE_PHRASES`` covers
   greetings/closings only, so passage-meta, referral and bare
   transition sentences in NORMAL answers are counted factual-uncited
   (~17% of factual-counted sentences in the run's 10-item deep sample).

FLAGGED RULE (test-author decision, from the issue's live examples — all
sentence texts below are verbatim from the run journal's answer_text):
an uncited sentence is non-factual when it is
  (a) **passage-meta** — its subject is the supplied material or the
      answer's own discourse ("The passages I was given…", "The sources
      listed in the panel…", "The passages illustrate…"), not the world;
  (b) **referral** — advice to consult sources outside the corpus
      ("you'd want to look for…", "…would be a good starting point");
  (c) **bare transition/commentary** — evaluative rhetoric carrying no
      verifiable specific content ("The human toll is tangible.").
The ratified "cited ⇒ always factual" rule is preserved unchanged, and
world-claims (including ones mentioning 'sources' as a noun) stay
factual. The mechanism is the implementer's; the sentence set is pinned.

FLAGGED CONTRACT (decline detection): the validation record carries
``generation_decline: True`` at minimum for an answered (non-refused)
exchange on a ``no_answer``-category gold item that delivered ZERO
citations — the shape of all 12 live cases. A zero-citation answer on an
ANSWERABLE item is NOT a decline: it stays pooled fail-closed.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from rag.citation_validator import ValidationOutcome, segment_answer_sentences
from rag.provider import AnswerWithCitations, FakeAdapter
from rag.retrieval import RerankedPassage, RetrievedPassages
from tests._eval_harness_fixtures import (
    production_passage_payload,
    transport_stream_for_answer,
)

ARM_MODEL = "claude-haiku-4-5"

#: The actual qa-na-g-05 answer from the live run's journal
#: (data/release-run/journals/claude-haiku-4-5-answers.jsonl) — the
#: pinned regression the issue requires: 3 sentences, 0 citations,
#: journalled as validation {validated: true, supported: 0, factual: 3}.
QA_NA_G_05_ANSWER = (
    "The passages I was given don't answer that question. They cover climate "
    "trends in the United States, extreme weather events, and climate tipping "
    "points — but they don't address earthquakes or any relationship between "
    "climate change and earthquake strength.\n\n"
    "For a scientific answer to this question, you'd want to look for sources "
    "specifically focused on seismology and the mechanics of earthquakes, "
    "rather than climate assessment literature."
)

#: Verbatim uncited sentences the live run counted factual-with-0-supported
#: (issue #312's evidence: qa-na-g-05, qa-sp-06 S1/S7, qa-tg-03 S2/S4/S5,
#: qa-va-02 S0/S2/S3). Each must classify non-factual when uncited.
NON_FACTUAL_WHEN_UNCITED = (
    # qa-na-g-05 (decline opener + retrieval-set meta + referral)
    "The passages I was given don't answer that question.",
    "They cover climate trends in the United States, extreme weather events, "
    "and climate tipping points — but they don't address earthquakes or any "
    "relationship between climate change and earthquake strength.",
    "For a scientific answer to this question, you'd want to look for sources "
    "specifically focused on seismology and the mechanics of earthquakes, "
    "rather than climate assessment literature.",
    # qa-sp-06 S1/S7 (bare transitions/commentary)
    "The evidence is detailed and spans different hazard types.",
    "The human toll is tangible.",
    # qa-tg-03 S2/S4/S5 (passage-meta + decline + source-panel referral)
    "The passages illustrate how this works in practice.",
    "However, the passages I was given don't explain the detailed methodology "
    "behind how scientists carry out these calculations.",
    "The full sources listed in the panel — particularly Chapter 3 of the "
    "Fifth National Climate Assessment, which is referenced for attribution "
    "science details — would provide that technical explanation.",
    # qa-va-02 S0/S3 (decline variant + panel/Wikipedia referral)
    "The passages I was given don't answer that question in the way you're asking it.",
    "The sources listed in the panel may have that kind of material; his "
    "Wikipedia entry, linked in the passage, would be a good starting point.",
)

#: World-claims that must STAY factual when uncited — the rule must not
#: over-reach (several sit right next to pinned non-factual sentences in
#: the same live answers).
FACTUAL_WHEN_UNCITED = (
    "Heatwaves have become more common and severe in the West since the 1980s (high confidence).",
    "Event attribution now allows us to assign a quantifiable fraction of "
    "attributable risk to climate change.",
    "In the 1980s, the country experienced, on average, one billion-dollar "
    "weather disaster every four months.",
    "Climate change made the record-breaking Pacific Northwest heatwave of June 2021 hotter.",
    # 'sources' as a world noun, not the answer's source panel.
    "Greenhouse gas emissions from human sources continue to rise.",
)


def _segment_text(text: str):
    return segment_answer_sentences([{"event": "text", "data": {"text": text}}])


# ---------------------------------------------------------------------------
# The classifier: passage-meta / referral / transition sentences are
# non-factual unless cited (issue #312 required test 2).
# ---------------------------------------------------------------------------


def test_passage_meta_referral_and_transition_sentences_are_not_factual():
    """Every pinned live-run sentence classifies factual=False when
    uncited — none carries a claim a corpus chunk could entail."""
    for text in NON_FACTUAL_WHEN_UNCITED:
        (sentence,) = _segment_text(text)
        assert sentence.factual is False, (
            f"uncited passage-meta/referral/transition sentence counted factual "
            f"(it can never be supported, so it poisons the 0.95 pool): {text!r}"
        )


def test_world_claim_sentences_stay_factual_when_uncited():
    """The broadened rule must not over-reach: uncited world-claims stay
    factual (and keep failing the gate when unsupported — fail-closed)."""
    for text in FACTUAL_WHEN_UNCITED:
        (sentence,) = _segment_text(text)
        assert sentence.factual is True, (
            f"world-claim sentence wrongly classified non-factual: {text!r}"
        )


def test_cited_meta_sentence_is_always_factual():
    """The ratified 'cited ⇒ always factual' rule survives the broadening:
    a citation attached to a meta-shaped sentence means the model asserted
    something from a source, and it gets verified regardless."""
    events = [
        {
            "event": "text",
            "data": {"text": "The passages illustrate how this works in practice."},
        },
        {"event": "citation", "data": {"document_index": 0}},
    ]
    (sentence,) = segment_answer_sentences(events)
    assert sentence.document_indices == (0,)
    assert sentence.factual is True


def test_qa_na_g_05_regression_contributes_zero_factual_sentences():
    """The pinned regression (issue #312 required test 3): the actual
    qa-na-g-05 answer text segments to its three delivered sentences and
    contributes ZERO factual sentences — not the 3/0-supported the live
    run pooled."""
    sentences = _segment_text(QA_NA_G_05_ANSWER)
    assert len(sentences) == 3, "the delivered answer is three sentences"
    factual = [sentence for sentence in sentences if sentence.factual]
    assert factual == [], (
        f"the honest-decline answer must contribute nothing to the citation "
        f"pool; counted factual: {[sentence.text for sentence in factual]}"
    )


# ---------------------------------------------------------------------------
# The gate: generation-level declines contribute zero to the pool
# (issue #312 required test 1).
# ---------------------------------------------------------------------------


def test_generation_decline_records_are_excluded_from_the_pool():
    """A validation record flagged ``generation_decline`` contributes ZERO
    factual sentences to the citation_support denominator — the item is
    the refusal gate's evidence, never double-counted here — while
    staying visible in the gate evidence. The ratified 0.95 target then
    holds over the CLEANED pool: 19/20 (=0.95 exactly) passes."""
    from evals.gates import GATE_FAILED, GATE_PASSED, citation_support_gate

    decline_record = {
        "item_id": "qa-na-g-05",
        "validated": True,
        "supported": 0,
        "factual": 3,
        "generation_decline": True,
    }
    records = [
        {"item_id": "syn-ok-01", "validated": True, "supported": 19, "factual": 20},
        decline_record,
    ]
    result = citation_support_gate(records, threshold=0.95)
    assert (result.numerator, result.denominator) == (19, 20), (
        "the decline's 3 unentailable sentences must not poison the pool "
        "(19/23 would fail a genuinely-0.95 answer set)"
    )
    assert result.status == GATE_PASSED

    entry = next(
        (entry for entry in result.evidence if entry.get("item_id") == "qa-na-g-05"),
        None,
    )
    assert entry is not None, "the excluded decline stays VISIBLE in the evidence"
    assert entry.get("generation_decline") is True

    # The cleaned pool still enforces the ratified strict target: one more
    # unsupported sentence (18/20 = 0.90) fails.
    weaker = [
        {"item_id": "syn-ok-01", "validated": True, "supported": 18, "factual": 20},
        dict(decline_record),
    ]
    weaker_result = citation_support_gate(weaker, threshold=0.95)
    assert (weaker_result.numerator, weaker_result.denominator) == (18, 20)
    assert weaker_result.status == GATE_FAILED


# ---------------------------------------------------------------------------
# The runner: a generation-level honest decline is marked on the
# validation record it journals.
# ---------------------------------------------------------------------------

CLASSIFICATION_IN_SCOPE = {
    "scope": "in_scope",
    "rewritten_query": "synthetic rewritten query",
}

PASSAGES = RetrievedPassages(
    passages=(
        RerankedPassage(
            chunk_id="syn_doc:0001",
            rerank_score=0.9,
            clears_threshold=True,
            payload=production_passage_payload("syn_doc:0001"),
        ),
    )
)

NO_ANSWER_ITEM = {
    "id": "qa-na-g-05",
    "category": "no_answer",
    "question": "Are earthquakes getting stronger because of climate change?",
    "expected_behaviour": "refusal",
    "subset": "gate",
    "expected_route": "retrieval_refusal",
}

ANSWERABLE_ITEM = {
    "id": "syn-sp-01",
    "category": "single_passage",
    "question": "How warm is the synthetic planet?",
    "expected_behaviour": "answer",
    "gold_chunk_ids": ["syn_doc:0001"],
}


def _segmenting_validator(grounded, sse_events):
    """A genuine-shaped validator over the REAL production segmentation:
    validated, zero entailment pairs (nothing was cited)."""
    return ValidationOutcome(
        validated=True,
        sentences=segment_answer_sentences(sse_events),
        verdicts=(),
    )


def _run_single_item(item, answer: AnswerWithCitations):
    from evals.harness import AnswerPathDeps, run_answer_path

    adapter = FakeAdapter(
        generate_stream_results=[transport_stream_for_answer(answer)],
        structured_results=[CLASSIFICATION_IN_SCOPE],
    )
    deps = AnswerPathDeps(
        adapter=adapter,
        retrieve=lambda decision: PASSAGES,
        validate_exchange=_segmenting_validator,
    )
    (result,) = run_answer_path([item], deps, arm_model=ARM_MODEL, mode="fake")
    return result


def test_runner_marks_generation_level_decline_on_the_validation_record():
    """Retrieval did NOT refuse (the live failure shape) but generation
    answered the no_answer item with the zero-citation honest decline:
    the runner marks ``generation_decline`` on the validation record so
    the gate can exclude the exchange from the pool. The item stays
    refused=False — it is (correctly) FAILED by the refusal gate."""
    from evals.gates import citation_support_gate

    result = _run_single_item(
        NO_ANSWER_ITEM,
        AnswerWithCitations(
            text=QA_NA_G_05_ANSWER,
            citations=(),
            usage={"input_tokens": 2400, "output_tokens": 96},
        ),
    )
    assert result.refused is False, "a generation-level decline is not a retrieval refusal"
    assert result.citations == ()
    assert result.validation is not None
    assert result.validation.get("generation_decline") is True, (
        "an answered, zero-citation exchange on a no_answer gold item is a "
        "generation-level honest decline: the validation record must say so "
        "(issue #312 — it is the refusal gate's evidence, not this pool's)"
    )

    # And the pooled arithmetic over the run records ignores the decline.
    gate = citation_support_gate(
        [
            {"item_id": result.item_id, **dict(result.validation)},
            {"item_id": "syn-ok-01", "validated": True, "supported": 19, "factual": 20},
        ],
        threshold=0.95,
    )
    assert (gate.numerator, gate.denominator) == (19, 20)


def test_zero_citation_answer_on_answerable_item_is_not_a_decline():
    """Fail-closed is NOT weakened: an uncited factual answer on an
    ANSWERABLE item carries no decline flag and its factual sentences
    stay pooled with zero supported."""
    from evals.gates import citation_support_gate

    result = _run_single_item(
        ANSWERABLE_ITEM,
        AnswerWithCitations(
            text="Global surface temperature rose 1.1C between 1850 and 2020.",
            citations=(),
            usage={"input_tokens": 2400, "output_tokens": 24},
        ),
    )
    assert result.refused is False
    assert result.validation is not None
    assert not result.validation.get("generation_decline"), (
        "an answerable item's uncited answer is a support FAILURE, never an "
        "excluded decline — the exclusion must not become a leak"
    )
    gate = citation_support_gate(
        [{"item_id": result.item_id, **dict(result.validation)}], threshold=0.95
    )
    assert (gate.numerator, gate.denominator) == (0, 1)
