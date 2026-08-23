"""Shared builders for the issue-13 citation-validator red suite.

Everything here is synthetic (the fictional Aurelian-Basin universe of
the #24/#9/#11/#12 fixtures). The transcripts are built in the #12 SSE
vocabulary (``text`` / ``citation`` / ``usage`` / ``footer`` events —
exactly the ``rag.generation.answer_stream_to_sse`` output shape the
service delivers), because that transcript is the validator's input:
citation-to-sentence attachment is transport order, which the flattened
``AnswerWithCitations`` alone cannot express.

``corrupted_answer_events`` is the canonical corrupted-answer scenario
(issue #13 acceptance): a REAL citation carrying a NON-ENTAILING
sentence — the cited block says the basin warmed; the sentence claims it
cooled. The FakeAdapter suite programs verdicts over it now; the
ReplayAdapter acceptance tests replay the model's recorded verdicts on
the same pairs once a recording session lands (the #162 rule:
hand-authored replay fixtures are never an acceptable substitute for
recordings).
"""

from __future__ import annotations

from typing import Any

from rag.generation import GroundedAnswer, build_response_footer, resolve_citations
from rag.provider import AnswerWithCitations, Citation
from tests._generation_fixtures import CORPUS_VINTAGE, make_payload, make_retrieved

#: Interactional furniture the model may open with — never a factual
#: claim, never validated (issue #13 scope: non-factual furniture
#: excluded from the factual-sentence units).
GREETING_SENTENCE = "Thanks for asking."

#: The supported factual sentence: restates cited block 0's body claim.
SUPPORTED_SENTENCE = (
    "Surface temperatures across the Aurelian Basin have very likely "
    "risen by one point nine degrees since the fictional baseline period."
)

#: The corrupted factual sentence: cites block 1 (which says the basin
#: WARMED) while claiming the opposite — a real citation that does not
#: entail its sentence.
CORRUPTED_SENTENCE = (
    "Surface temperatures across the Aurelian Basin have very likely "
    "cooled by four degrees since the fictional baseline period."
)

#: An uncited factual sentence: a checkable claim with no citation
#: attached — auto-flagged without an entailment call.
UNCITED_FACTUAL_SENTENCE = (
    "Reservoir inflows across the basin declined in twenty one of the "
    "last twenty five invented water years."
)

#: The usage the generation stream reported (irrelevant to validation,
#: present so transcripts carry the full pinned vocabulary).
GENERATION_USAGE = {
    "input_tokens": 900,
    "output_tokens": 42,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 4200,
}


def text_event(text: str) -> dict[str, Any]:
    """One delivered ``text`` SSE event (the #12 vocabulary)."""
    return {"event": "text", "data": {"text": text}}


def citation_sse_event(document_index: int, cited_text: str | None = None) -> dict[str, Any]:
    """One delivered ``citation`` SSE event (custom-content shape)."""
    return {
        "event": "citation",
        "data": {
            "type": "content_block_location",
            "document_index": document_index,
            "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
            "start_block_index": 0,
            "end_block_index": 1,
            "cited_text": cited_text or make_payload(document_index)["body"],
        },
    }


def usage_event() -> dict[str, Any]:
    return {"event": "usage", "data": dict(GENERATION_USAGE)}


def footer_event() -> dict[str, Any]:
    return {"event": "footer", "data": {"text": build_response_footer(CORPUS_VINTAGE)}}


def transcript(*body_events: dict[str, Any]) -> list[dict[str, Any]]:
    """A full delivered transcript: body events + the pinned close."""
    return [*body_events, usage_event(), footer_event()]


def corrupted_answer_events() -> list[dict[str, Any]]:
    """The canonical corrupted-answer transcript (issue #13 acceptance).

    Sentence 0: furniture. Sentence 1: supported, cites block 0.
    Sentence 2: corrupted, REALLY cites block 1 (non-entailing).
    Sentence 3: uncited factual.
    """
    return transcript(
        text_event(GREETING_SENTENCE + " "),
        text_event(SUPPORTED_SENTENCE),
        citation_sse_event(0),
        text_event(" " + CORRUPTED_SENTENCE),
        citation_sse_event(1),
        text_event(" " + UNCITED_FACTUAL_SENTENCE),
    )


def corrupted_answer_text() -> str:
    return " ".join(
        [
            GREETING_SENTENCE,
            SUPPORTED_SENTENCE,
            CORRUPTED_SENTENCE,
            UNCITED_FACTUAL_SENTENCE,
        ]
    )


def make_grounded_answer(
    *,
    document_indices: tuple[int, ...] = (0, 1),
    passage_count: int = 2,
    text: str | None = None,
    usage: dict[str, int] | None = None,
) -> GroundedAnswer:
    """A resolved GroundedAnswer whose cited_passages carry real bodies.

    ``document_indices`` lists the blocks the answer cited (in citation
    order); resolution runs through the real ``resolve_citations`` so
    the payload shapes match what production hands the validator.
    """
    retrieved = make_retrieved(passage_count)
    answer = AnswerWithCitations(
        text=text if text is not None else corrupted_answer_text(),
        citations=tuple(
            Citation(
                cited_text=make_payload(index)["body"],
                document_index=index,
                document_title="Meridian Climate Assessment Cycle 3 (invented)",
                start_block_index=0,
                end_block_index=1,
            )
            for index in document_indices
        ),
        usage=usage or dict(GENERATION_USAGE),
    )
    return GroundedAnswer(
        text=answer.text,
        cited_passages=resolve_citations(answer, retrieved),
        footer=build_response_footer(CORPUS_VINTAGE),
        usage=answer.usage,
    )


def verdicts_output(*supported: bool) -> dict[str, Any]:
    """A well-formed structured verdict output over pairs 0..n-1."""
    return {
        "verdicts": [
            {"pair_index": index, "supported": flag} for index, flag in enumerate(supported)
        ]
    }
