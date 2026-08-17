"""Shared builders for the issue-7 ingestion RED suite.

Everything constructed here is synthetic — invented sentences over the
fictional Aurelian-Basin universe of the #24 fixture corpus, .invalid
URLs, invented sign-offs. No real Tier B/C text (DESIGN §2.1).

Tests build :class:`ingestion.parse.StructuredDoc` inputs directly: the
StructuredDoc/Block types are the parse seam (IMPLEMENTATION.md §1) — the
production chunker is tested on parse *output* fixtures, never through
the heavy Docling install (unit tier, IMPLEMENTATION.md §3).
"""

from __future__ import annotations

from typing import Any

from ingestion.parse import Block, BlockType, StructuredDoc
from ingestion.pipeline import IngestConfig

#: Atomic block-type values (never split by cap logic, exempt from the
#: tiny-chunk floor) as they appear in ChunkRecord.block_types.
ATOMIC_BLOCK_TYPES = frozenset({"table", "figure", "footnote"})


def word_tokens(text: str) -> int:
    """Deterministic injectable token counter for the unit tier
    (IngestConfig.token_counter): whitespace word count. The production
    default is the embedding tokenizer; unit tests must not load it.
    """
    return len(text.split())


def config(**overrides: Any) -> IngestConfig:
    """An IngestConfig with the deterministic counter injected."""
    overrides.setdefault("token_counter", word_tokens)
    return IngestConfig(**overrides)


def heading(text: str, level: int = 1) -> Block:
    return Block(BlockType.HEADING, text, level=level)


def text(body: str) -> Block:
    return Block(BlockType.TEXT, body)


def footnote(body: str) -> Block:
    return Block(BlockType.FOOTNOTE, body)


def figure(caption: str | None = None) -> Block:
    return Block(BlockType.FIGURE, "[FIGURE]", caption=caption)


def caption(body: str) -> Block:
    return Block(BlockType.CAPTION, body)


def doc(doc_id: str, blocks: list[Block], title: str | None = None) -> StructuredDoc:
    return StructuredDoc(
        doc_id=doc_id,
        title=title or f"{doc_id} (invented)",
        blocks=blocks,
        backend="test-fixture",
    )


def sentence(n_words: int, tag: str) -> str:
    """A synthetic sentence of exactly ``n_words`` whitespace words,
    ending in a period, uniquely identifiable by ``tag``."""
    assert n_words >= 2
    words = [f"{tag}w{i}" for i in range(n_words - 1)]
    return " ".join(words) + f" {tag}end."


def manifest_entry(doc_id: str = "syn-doc", **overrides: Any) -> dict[str, Any]:
    """A fully valid §2.1 document entry (invented), matching the schema
    ingestion.manifest.validate_document enforces."""
    entry: dict[str, Any] = {
        "id": doc_id,
        "title": f"{doc_id} (invented)",
        "licence": "CC BY 4.0 (invented)",
        "licence_evidence": "Invented licence statement captured 2026-08-16",
        "attribution_text": f"{doc_id} Fixture Author (fictional)",
        "canonical_url": f"https://mca.example.invalid/{doc_id}",
        "redistributable": True,
        "permitted_context": "open",
        "permission_evidence": None,
        "consensus_position": "assessed",
        "source_type": "evidence",
        "source_tier": "A",
        "sha256": "ab" * 32,
        "retrieved_at": "2026-08-16",
        "human_signoff": {
            "who": "issue-7 test author",
            "date": "2026-08-17",
            "note": "synthetic red-suite entry",
        },
    }
    entry.update(overrides)
    return entry


def minimal_pdf_bytes(body_text: str = "SYNTHETIC FIXTURE - invented PDF body text") -> bytes:
    """A minimal valid single-page PDF, generated from scratch in-test —
    entirely synthetic bytes, no external source, no pymupdf needed."""
    stream = f"BT /F1 12 Tf 72 720 Td ({body_text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n% SYNTHETIC FIXTURE - authored for this project's tests\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)
