"""Custom-content citation blocks (issue #7 TDD plan 11; DESIGN §2.4/§3.3;
reviews/spike-03-probe-findings.md) — RED. Unit tier, pure.

The payload shape is the one the spike-03 probe proved live; the two
contractual lessons from the spike are pinned here: the citable content
is the chunk BODY only (the context header travels in `context`, so
cited_text quotes pure evidence), and citations are enabled on every
block (§3.4 all-or-none).
"""

from __future__ import annotations

import json

from ingestion.blocks import build_citation_blocks
from ingestion.pipeline import ChunkRecord


def _chunk(index: int, **overrides) -> ChunkRecord:
    fields = {
        "chunk_id": f"syn-doc::{index:04d}",
        "doc_id": "syn-doc",
        "section_path": ("1 Invented section",),
        "context_header": "syn-doc (invented) → 1 Invented section",
        "body": f"Invented evidence sentence number {index} for the block tests.",
        "token_count": 12,
        "confidence_markers": (),
        "consensus_position": "assessed",
        "source_type": "evidence",
        "citation_metadata": {
            "title": "syn-doc (invented)",
            "licence": "CC BY 4.0 (invented)",
            "attribution_text": "Block Fixture Author (fictional)",
            "canonical_url": "https://mca.example.invalid/syn-doc",
            "permitted_context": "open",
        },
    }
    fields.update(overrides)
    return ChunkRecord(**fields)


def test_one_block_per_citable_unit():
    """TDD plan 11: block count matches chunk count, order preserved,
    each block a custom-content document whose text is the chunk body."""
    chunks = [_chunk(i) for i in range(3)]
    blocks = build_citation_blocks(chunks)
    assert len(blocks) == 3
    for chunk, block in zip(chunks, blocks, strict=True):
        assert block["type"] == "document"
        assert block["source"]["type"] == "content"
        content = block["source"]["content"]
        assert [part["type"] for part in content] == ["text"] * len(content)
        assert "".join(part["text"] for part in content) == chunk.body


def test_block_context_carries_header_and_chunk_id_not_the_citable_text():
    """Spike-03: the context header leaked into cited_text when embedded
    in the block text. Production: header + chunk id ride in `context`
    (passed to the model, never cited); the citable content excludes
    them."""
    chunk = _chunk(0)
    (block,) = build_citation_blocks([chunk])
    context = block["context"]
    assert chunk.chunk_id in context
    assert chunk.context_header in context
    cited_text = "".join(part["text"] for part in block["source"]["content"])
    assert chunk.context_header not in cited_text


def test_citations_enabled_on_every_block():
    """§3.4 all-or-none, set at emission (the #12 request builders assert
    it again at request level)."""
    blocks = build_citation_blocks([_chunk(i) for i in range(4)])
    for block in blocks:
        assert block["citations"] == {"enabled": True}


def test_blocks_carry_hand_review_and_backend_flags():
    """Review finding #143: a degraded-parse chunk's block payload must
    carry the hand-review flag and parse backend (alongside
    consensus_position in `context`), so the downstream indexer/service
    can quarantine without a join against run state."""
    chunk = _chunk(0, parse_backend="pymupdf", needs_hand_review=True)
    (block,) = build_citation_blocks([chunk])
    context = block["context"]
    assert "needs_hand_review=True" in context, context
    assert "parse_backend=pymupdf" in context, context


def test_blocks_carry_full_citation_metadata():
    """TDD plan 11: every block carries the chunk's citation metadata —
    attribution, canonical URL, licence, permitted_context,
    consensus_position and source_type are all recoverable from the
    payload (the exact field layout is the implementer's, e.g. folded
    into `context`/`title`; presence is the contract)."""
    chunk = _chunk(
        0,
        consensus_position="beyond-assessed-range",
        source_type="voices",
    )
    (block,) = build_citation_blocks([chunk])
    serialised = json.dumps(block)
    for value in (
        "Block Fixture Author (fictional)",
        "https://mca.example.invalid/syn-doc",
        "CC BY 4.0 (invented)",
        "open",
        "beyond-assessed-range",
        "voices",
    ):
        assert value in serialised, f"citation metadata {value!r} missing from block payload"
