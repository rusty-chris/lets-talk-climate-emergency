"""Issue #303 red phase (Fable): one sentence-boundary rule, the
production one.

Verified divergence, not a hypothetical: ``rag/citation_validator.py``
segments the delivered answer with ``ingestion.chunk.split_sentences``
— the issue #2 SPIKE splitter (its own docstring: "Deliberately simple;
its failure modes ... are catalogued ... as risks for #7") — while the
production chunker uses ``ingestion.pipeline.split_sentences`` ("Real
sentence segmentation (finding 7 — not the spike regex)"). They
disagree on realistic climate-answer text:

- ``"... (high confidence). 2023 was the warmest year on record."`` —
  the spike regex splits before the digit-initial ``2023``; the
  production rule (uppercase-only boundary) does not;
- ``"... (see e.g. The Sixth Assessment). It is ..."`` — the spike
  regex breaks inside ``e.g. The``; production's protected-abbreviation
  list re-joins it;
- ``"... (IPCC, 2021). (This includes methane.)"`` — the spike regex
  splits before the parenthesised continuation; production does not.

Every downstream number moves with the boundary rule: the pooled
factual-sentence denominator of the RATIFIED 0.95 citation-support
release gate, the badge events' ``sentence_index``, and the #18 UI
citation chips (``ui.render_model`` consumes
``citation_sentence_assignments``). These tests pin the validator + UI
chip path onto the SINGLE production implementation. They pin
behavioural equality with ``ingestion.pipeline.split_sentences`` — the
implementer is free to re-home the function, as long as one production
rule remains the single source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ingestion.pipeline import split_sentences as production_split_sentences
from rag.citation_validator import citation_sentence_assignments, segment_answer_sentences

#: Realistic answer texts on which the spike and production splitters
#: verifiably disagree (see module docstring).
DIVERGENT_ANSWER_TEXTS = (
    "Global mean sea level rose about 0.20 m between 1901 and 2018 "
    "(high confidence). 2023 was the warmest year on record.",
    "Warming drove heavier rainfall (see e.g. The Sixth Assessment). "
    "It is very likely to continue.",
    "Emissions rose sharply (IPCC, 2021). (This includes methane.)",
)


def _text_events(*chunks: str) -> list[dict[str, Any]]:
    return [{"event": "text", "data": {"text": chunk}} for chunk in chunks]


def _segmented_texts(events: Sequence[dict[str, Any]]) -> list[str]:
    return [sentence.text for sentence in segment_answer_sentences(events)]


def test_validator_segments_with_the_production_sentence_rule():
    """segment_answer_sentences must follow ingestion.pipeline.split_sentences
    — the repo's ONE production sentence-boundary rule — byte-for-byte on
    the texts where the spike splitter diverges."""
    for text in DIVERGENT_ANSWER_TEXTS:
        assert _segmented_texts(_text_events(text)) == production_split_sentences(text), (
            f"validator segmentation diverged from the production rule on {text!r}: "
            "the citation validator (and everything downstream of it) must use "
            "ingestion.pipeline.split_sentences semantics, never the issue #2 "
            "spike regex (issue #303 item: one sentence-boundary rule)"
        )


def test_factual_sentence_denominator_follows_production_boundaries():
    """The pooled citation-support denominator counts PRODUCTION
    sentences: '... (high confidence). 2023 was ...' is ONE factual
    sentence under the production rule — the spike regex would count two
    and silently shift the ratified 0.95 gate's arithmetic."""
    text = DIVERGENT_ANSWER_TEXTS[0]
    sentences = segment_answer_sentences(_text_events(text))
    factual = [sentence for sentence in sentences if sentence.factual]
    assert len(factual) == 1, (
        f"expected one production-rule factual sentence, got "
        f"{[sentence.text for sentence in factual]} — a denominator that moves "
        "with the spike regex corrupts the release gate's pooled arithmetic"
    )


def test_citation_chip_assignment_follows_production_boundaries():
    """The #18 UI chip join (citation_sentence_assignments — consumed by
    ui.render_model) indexes sentences by the production rule: a citation
    arriving after the second delivered sentence attaches to sentence
    index 1, not the spike regex's index 2 — a chip pointing at the
    wrong sentence is a user-visible integrity bug."""
    events = [
        *_text_events(
            "Warming drove heavier rainfall (see e.g. The Sixth Assessment). ",
            "It is very likely to continue.",
        ),
        {"event": "citation", "data": {"document_index": 0}},
    ]
    assert citation_sentence_assignments(events) == ((1, 0),)
