"""Issue #328 red phase (Fable): rhetorical-stance / discourse sentences
leak past ``_carries_no_checkable_claim`` into the citation pool.

The owner-commissioned citation-gate review (the #325 decision's evidence
base) sampled the resmoke run's uncited-factual pool and found ~29% of it
is discourse furniture misclassified as factual: rhetorical STANCE and
DISCOURSE sentences — the model agreeing, evaluating, or talking about
the interlocutor's framing — which no corpus chunk can entail by
construction. Excluding the leak moves measured support ~64.5% -> ~71%
with ZERO behaviour change (the answers, citations and entailment
verdicts are untouched; only the denominator stops counting furniture).

Pinned contract (#312 precedent — the sentence set is the contract, the
mechanism is the implementer's):
- the review's sampled sentences, verbatim from the issue, classify
  factual=False when uncited (the issue quotes three samples; the
  review's fuller sample was not committed, so three verbatim pins plus
  the class definition are the red contract — FLAGGED in the red-phase
  report);
- the ratified "cited => always factual" override is preserved;
- the #312 world-claim controls PLUS new stance-shaped world-claim
  controls stay factual — a sentence that takes a stance AND carries a
  checkable claim is still a claim.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from typing import Any

from rag.citation_validator import segment_answer_sentences
from tests.unit.test_review_312_citation_pool import FACTUAL_WHEN_UNCITED

#: Verbatim stance/discourse samples from the review (issue #328) — each
#: must classify non-factual when uncited.
STANCE_DISCOURSE_NON_FACTUAL = (
    "Yes, absolutely.",
    "The concern is substantial.",
    "That premise doesn't match what the evidence shows.",
)

#: NEW factual controls (test-author, per the issue: "world-claims must
#: stay factual — the #312 controls plus new ones"). Each sits close to a
#: pinned stance sentence in shape — stance opener, copula+abstract noun,
#: or evidence-talk — but carries a checkable world-claim.
NEW_FACTUAL_CONTROLS = (
    # Stance opener + claim: the "Yes," must not sweep the claim out.
    "Yes, the Arctic is warming faster than the global average.",
    # "the evidence shows" + claim: evidence-talk with checkable content.
    "The evidence shows warming of about 1.1C since the late 1800s.",
    # Copula + "concern" + specific content: not bare stance.
    "The concern is that 2C of warming would double the frequency of heat extremes.",
    # "absolutely" as an intensifier inside a world-claim.
    "Scientists are absolutely certain that human activity drives recent warming.",
)


def _segment_text(text: str):
    return segment_answer_sentences([{"event": "text", "data": {"text": text}}])


# ---------------------------------------------------------------------------
# The classifier: the review's stance/discourse samples are non-factual.
# ---------------------------------------------------------------------------


def test_review_sampled_stance_sentences_are_not_factual_when_uncited():
    for text in STANCE_DISCOURSE_NON_FACTUAL:
        (sentence,) = _segment_text(text)
        assert sentence.factual is False, (
            f"uncited stance/discourse sentence counted factual (it can never "
            f"be entailed by a corpus chunk, so it poisons the pooled "
            f"denominator — ~29% of the resmoke uncited pool): {text!r}"
        )


def test_312_world_claim_controls_still_factual():
    """The #312 controls are re-asserted through the broadened rule —
    the stance extension must not regress the earlier boundary."""
    for text in FACTUAL_WHEN_UNCITED:
        (sentence,) = _segment_text(text)
        assert sentence.factual is True, (
            f"#312 factual control wrongly reclassified non-factual: {text!r}"
        )


def test_new_stance_shaped_world_claims_stay_factual():
    for text in NEW_FACTUAL_CONTROLS:
        (sentence,) = _segment_text(text)
        assert sentence.factual is True, (
            f"stance-shaped world-claim wrongly classified non-factual "
            f"(the stance rule is over-reaching): {text!r}"
        )


def test_cited_stance_sentence_is_always_factual():
    """The ratified 'cited => always factual' override survives the
    broadening, exactly as it survived #312."""
    for text in STANCE_DISCOURSE_NON_FACTUAL:
        events = [
            {"event": "text", "data": {"text": text}},
            {"event": "citation", "data": {"document_index": 0}},
        ]
        (sentence,) = segment_answer_sentences(events)
        assert sentence.document_indices == (0,)
        assert sentence.factual is True, f"cited sentence must stay factual: {text!r}"


# ---------------------------------------------------------------------------
# Gate-arithmetic interaction (fixture-level): excluding the furniture
# raises measured support with no behaviour change.
# ---------------------------------------------------------------------------

#: A resmoke-shaped answer interleaving the review's verbatim stance
#: sentences with world-claims: 3 factual sentences (2 cited+entailed,
#: 1 uncited) and 3 stance sentences.
STANCE_HEAVY_ANSWER = (
    "Yes, absolutely.\n\n"
    "Global surface temperature rose about 1.1C between 1850 and 2020. "
    "Sea level rose about 20cm over the same period. "
    "The concern is substantial.\n\n"
    "That premise doesn't match what the evidence shows. "
    "Warming has continued in every decade since the 1970s."
)


def _sentence_records(sentences, attached_indices: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": sentence.index,
            "paragraph": sentence.paragraph,
            "factual": sentence.factual,
            "attached": sentence.index in attached_indices,
        }
        for sentence in sentences
    ]


def test_excluding_stance_furniture_raises_measured_support_without_behaviour_change():
    """Same answer, same citations, same entailment verdicts — the ONLY
    difference is whether the three stance sentences pool as factual.

    With the leak (stance pooled): 4 uncited of 6 "factual" = 0.667 —
    the uncited_factual_rate ceiling (0.35) FAILS a genuinely well-cited
    answer. With the hygiene fix: 1 uncited of 3 factual = 0.333 —
    PASSES. Entailment precision is 2/2 either way: nothing about the
    answer's behaviour moved, only the measurement stopped counting
    furniture (the review's ~64.5% -> ~71% arithmetic, in miniature)."""
    from evals.gates import (
        GATE_FAILED,
        GATE_PASSED,
        citation_entailment_precision_gate,
        uncited_factual_rate_gate,
    )

    sentences = _segment_text(STANCE_HEAVY_ANSWER)
    assert [sentence.factual for sentence in sentences] == [
        False,  # "Yes, absolutely."
        True,  # temperature claim
        True,  # sea-level claim
        False,  # "The concern is substantial."
        False,  # "That premise doesn't match what the evidence shows."
        True,  # decadal-warming claim
    ], "red pre-condition: the classifier separates stance from claims"

    attached = {1, 2}
    verdicts = [
        {"pair_index": 0, "sentence_index": 1, "document_index": 0, "supported": True},
        {"pair_index": 1, "sentence_index": 2, "document_index": 0, "supported": True},
    ]
    clean_record = {
        "item_id": "syn-stance-01",
        "validated": True,
        "supported": 2,
        "factual": 3,
        "sentences": _sentence_records(sentences, attached),
        "verdicts": verdicts,
    }
    # The counterfactual leak: identical delivery, stance pooled factual.
    leaky_record = dict(clean_record)
    leaky_record["factual"] = 6
    leaky_record["sentences"] = [{**entry, "factual": True} for entry in clean_record["sentences"]]

    clean_gate = uncited_factual_rate_gate([clean_record])
    assert (clean_gate.numerator, clean_gate.denominator) == (1, 3)
    assert clean_gate.status == GATE_PASSED

    leaky_gate = uncited_factual_rate_gate([leaky_record])
    assert (leaky_gate.numerator, leaky_gate.denominator) == (4, 6)
    assert leaky_gate.status == GATE_FAILED

    for record in (clean_record, leaky_record):
        precision = citation_entailment_precision_gate([record])
        assert (precision.numerator, precision.denominator) == (2, 2), (
            "entailment precision is untouched by the hygiene fix — the fix is "
            "measurement-only, never a behaviour change"
        )
