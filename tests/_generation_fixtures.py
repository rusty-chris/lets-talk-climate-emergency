"""Shared builders for the issue-12 grounded-generation red suite.

Everything here is synthetic (the fictional Aurelian-Basin universe of
the #24/#9/#11 fixtures). The passage builders mirror the exact stored
payload shape `rag.indexing.build_index` writes (§2.4 metadata plus the
#143 parse-provenance flags), so the generation tests exercise the same
shapes production hands the generation layer.

`transport_stream_events` is the hand-built Anthropic streaming event
sequence (shape per the spike-03 findings and the recorded-response
conventions of #24's RawProviderResponse): text deltas, a citations
delta, final usage on message_delta. The pure SSE-translation tests run
against it now; the replay-pinned variant runs once a genuine recording
session lands (the #162 rule: hand-authored replay fixtures are never an
acceptable substitute for recordings).
"""

from __future__ import annotations

from typing import Any

from rag.retrieval import HonestRefusal, RerankedPassage, RetrievedPassages

#: The corpus vintage every generation test threads through (the §10
#: cadence line: "Answers reflect sources as of <corpus version date>").
CORPUS_VINTAGE = "2026-08-01"

#: A refusal threshold consistent with the scores below.
THRESHOLD = 0.5


def make_payload(
    index: int,
    *,
    body: str | None = None,
    source_type: str = "evidence",
    consensus_position: str = "assessed",
    degraded: bool = False,
) -> dict[str, Any]:
    """One stored chunk payload, exactly the fields `build_index` writes."""
    chunk_id = f"syn-gen-doc::c{index:04d}"
    return {
        "chunk_id": chunk_id,
        "doc_id": "syn-gen-doc",
        "section_path": ["1 Observed warming"],
        "context_header": "Meridian Climate Assessment (invented) > 1 Observed warming",
        "body": body
        or (
            f"Synthetic passage {index}: surface temperatures across the Aurelian "
            "Basin have very likely risen by one point nine degrees since the "
            "fictional baseline period."
        ),
        "token_count": 42,
        "confidence_markers": ["very likely"],
        "block_types": ["text"],
        "consensus_position": consensus_position,
        "source_type": source_type,
        "citation_metadata": {
            "licence": "CC-BY-4.0",
            "attribution_text": "Meridian Climate Assessment Cycle 3 (invented)",
            "canonical_url": "https://example.invalid/meridian/cycle-3",
            "permitted_context": "public-noncommercial",
        },
        "parse_backend": "pymupdf" if degraded else "docling",
        "degraded_fallback": degraded,
        "needs_hand_review": degraded,
        "content_hash": f"synthetic-hash-{index:04d}",
    }


def make_passage(
    index: int,
    *,
    rerank_score: float = 0.9,
    clears_threshold: bool = True,
    degraded: bool = False,
    **payload_overrides: Any,
) -> RerankedPassage:
    payload = make_payload(index, degraded=degraded, **payload_overrides)
    return RerankedPassage(
        chunk_id=payload["chunk_id"],
        rerank_score=rerank_score,
        clears_threshold=clears_threshold,
        payload=payload,
    )


def make_retrieved(
    count: int = 3,
    *,
    tone_flag: bool = False,
    partial_support: bool = False,
    degraded_indices: tuple[int, ...] = (),
) -> RetrievedPassages:
    """A RetrievedPassages of `count` synthetic passages, best first."""
    passages = tuple(
        make_passage(
            i,
            rerank_score=0.95 - i * 0.05,
            clears_threshold=not (partial_support and i == count - 1),
            degraded=i in degraded_indices,
        )
        for i in range(count)
    )
    return RetrievedPassages(
        passages=passages,
        tone_flag=tone_flag,
        partial_support=partial_support,
    )


def make_refusal(*, tone_flag: bool = False) -> HonestRefusal:
    return HonestRefusal(
        refusal_text=(
            "I can't answer that from the evidence in this corpus.\n"
            "Here is what the corpus does cover:\n- invented basin warming"
        ),
        covered_topics=("invented basin warming",),
        top_score=0.12,
        threshold=THRESHOLD,
        tone_flag=tone_flag,
    )


def citation_event(document_index: int = 0, cited_text: str = "") -> dict[str, Any]:
    """One citations_delta stream event (custom-content location shape)."""
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "citations_delta",
            "citation": {
                "type": "content_block_location",
                "document_index": document_index,
                "document_title": "Meridian Climate Assessment Cycle 3 (invented)",
                "start_block_index": 0,
                "end_block_index": 1,
                "cited_text": cited_text or make_payload(document_index)["body"],
            },
        },
    }


def transport_stream_events() -> list[dict[str, Any]]:
    """A full Anthropic streaming sequence for one short cited answer.

    Two text deltas, then a citation attached to the block, then the
    final usage (carrying cache-read metadata) — the shape #24's
    RawProviderResponse records and the AnthropicAdapter parses.
    """
    return [
        {"type": "message_start", "message": {"model": "claude-haiku-4-5", "role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "The basin has very likely warmed "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "by one point nine degrees."},
        },
        citation_event(document_index=0),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 900,
                "output_tokens": 42,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 4200,
            },
        },
        {"type": "message_stop"},
    ]
