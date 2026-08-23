"""Generation flow: RetrievedPassages -> one generate call ->
GroundedAnswer; HonestRefusal bypasses generation (issue #12) — RED.

Unit tier: everything runs through FakeAdapter (IMPLEMENTATION.md §4.1)
— our behaviour around the model, never the model. The FakeAdapter call
log is the instrument: exactly-one-call, zero-call and
raises-before-the-seam invariants are all observed behaviour, never
trusted report fields.
"""

from __future__ import annotations

import pytest

from rag.generation import (
    GenerationConfig,
    GenerationContractError,
    GroundedAnswer,
    build_generation_request,
    build_response_footer,
    generate_grounded_answer,
    resolve_citations,
)
from rag.provider import AnswerWithCitations, Citation, FakeAdapter
from rag.retrieval import GENERATION_TOP_K, HonestRefusal
from tests._generation_fixtures import (
    CORPUS_VINTAGE,
    make_refusal,
    make_retrieved,
)

QUESTION = "How much has the invented basin warmed?"


def _answer_citing(*document_indices: int, usage=None) -> AnswerWithCitations:
    return AnswerWithCitations(
        text="The basin has very likely warmed by one point nine degrees.",
        citations=tuple(
            Citation(
                cited_text=f"cited text for document {i}",
                document_index=i,
                document_title="Meridian Climate Assessment Cycle 3 (invented)",
                start_block_index=0,
                end_block_index=1,
            )
            for i in document_indices
        ),
        usage=usage,
    )


def _generate(adapter, retrieved, *, config=None):
    return generate_grounded_answer(
        adapter,
        retrieved,
        QUESTION,
        config=config or GenerationConfig(),
        corpus_vintage=CORPUS_VINTAGE,
    )


class TestGenerateCall:
    def test_exactly_one_generate_call_with_the_built_request(self):
        """The flow makes ONE generate call (§3.3 runtime budget), and
        its payload IS the pure builder's output — no drift between the
        tested builder and the request that actually leaves."""
        retrieved = make_retrieved(3)
        fake = FakeAdapter(generate_results=[_answer_citing(0)])
        _generate(fake, retrieved)
        calls = fake.calls_to("generate")
        assert len(calls) == 1
        expected = build_generation_request(retrieved, QUESTION, config=GenerationConfig())
        recorded = calls[0].payload
        assert recorded["documents"] == list(expected["documents"])
        assert recorded["config"] == expected["config"]
        assert recorded["messages"] == list(expected["messages"])
        assert recorded["system"] == list(expected["system"])

    def test_citations_resolve_to_supplied_blocks(self):
        """FakeAdapter round-trip: each returned document_index maps back
        to the supplied passage — the 'guaranteed' half of
        guaranteed-vs-measured (§3.3)."""
        retrieved = make_retrieved(4)
        fake = FakeAdapter(generate_results=[_answer_citing(0, 3)])
        answer = _generate(fake, retrieved)
        assert isinstance(answer, GroundedAnswer)
        assert [p.chunk_id for p in answer.cited_passages] == [
            retrieved.passages[0].chunk_id,
            retrieved.passages[3].chunk_id,
        ]
        assert [p.document_index for p in answer.cited_passages] == [0, 3]
        for cited in answer.cited_passages:
            assert cited.cited_text.startswith("cited text for document")

    def test_out_of_range_document_index_raises_never_guesses(self):
        retrieved = make_retrieved(2)
        fake = FakeAdapter(generate_results=[_answer_citing(5)])
        with pytest.raises(GenerationContractError):
            _generate(fake, retrieved)

    def test_resolve_citations_is_pure_and_reusable(self):
        """The resolution seam is a pure function over (answer,
        retrieved) — the citation-support validator (#13/§3.3) reuses it
        without re-running generation."""
        retrieved = make_retrieved(2)
        resolved = resolve_citations(_answer_citing(1), retrieved)
        assert len(resolved) == 1
        assert resolved[0].chunk_id == retrieved.passages[1].chunk_id

    def test_degraded_doc_flags_surface_on_cited_passages(self):
        """#143: a degraded-parse source stays visible on the answer
        surface — the flags ride the resolved citation, straight from
        the stored payload."""
        retrieved = make_retrieved(3, degraded_indices=(1,))
        fake = FakeAdapter(generate_results=[_answer_citing(0, 1)])
        answer = _generate(fake, retrieved)
        clean, degraded = answer.cited_passages
        assert clean.degraded_fallback is False
        assert clean.needs_hand_review is False
        assert degraded.degraded_fallback is True
        assert degraded.needs_hand_review is True

    def test_usage_with_cache_metadata_rides_the_answer(self):
        """The adapter-reported usage — cache_read_input_tokens included
        — lands on the GroundedAnswer intact: the channel #22's
        structured logs and spend accounting read (finding #64)."""
        usage = {
            "input_tokens": 900,
            "output_tokens": 42,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 4200,
        }
        fake = FakeAdapter(generate_results=[_answer_citing(0, usage=usage)])
        answer = _generate(fake, make_retrieved(2))
        assert answer.usage == usage

    def test_partial_support_and_tone_flag_carried_onto_the_answer(self):
        fake = FakeAdapter(generate_results=[_answer_citing(0)])
        answer = _generate(fake, make_retrieved(3, tone_flag=True, partial_support=True))
        assert answer.tone_flag is True
        assert answer.partial_support is True


class TestRefusalBypass:
    def test_honest_refusal_bypasses_generation_entirely(self):
        """§3.5: the refusal path is template-only — the refusal comes
        back unchanged and the adapter records ZERO calls of any kind."""
        fake = FakeAdapter()  # unprogrammed: ANY call would raise
        refusal = make_refusal(tone_flag=True)
        result = _generate(fake, refusal)
        assert isinstance(result, HonestRefusal)
        assert result == refusal
        assert fake.calls == []


class TestContractShortCircuit:
    def test_over_bound_passages_raise_before_the_adapter_is_touched(self):
        """§4.3 item 4: the violation short-circuits pre-network — the
        FakeAdapter transport records zero calls."""
        fake = FakeAdapter()
        oversized = make_retrieved(GENERATION_TOP_K + 1)
        with pytest.raises(GenerationContractError):
            _generate(fake, oversized)
        assert fake.calls == []


class TestFooter:
    def test_footer_carries_verification_note_and_corpus_vintage(self):
        """§3.5 footer + §10 cadence: the vintage date appears verbatim
        ('as of <date>') alongside the honest verification framing."""
        footer = build_response_footer(CORPUS_VINTAGE)
        assert CORPUS_VINTAGE in footer
        assert "as of" in footer.lower()
        assert "verif" in footer.lower()

    def test_grounded_answer_carries_the_footer(self):
        fake = FakeAdapter(generate_results=[_answer_citing(0)])
        answer = _generate(fake, make_retrieved(2))
        assert answer.footer == build_response_footer(CORPUS_VINTAGE)
