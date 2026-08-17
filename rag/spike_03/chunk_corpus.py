"""PROTOTYPE (issue #3 spike) — build the chunk corpus for the RAG probe.

Reuses issue #2's committed prototype parser (``ingestion.parse``, Docling) and
chunker (``ingestion.chunk``) verbatim — this is the "rerun spike #2's chunker on
the spike documents" step. spike_run.py drops chunk *text* from its JSON dump; the
probe needs the text (to embed and to build custom-content citation blocks), so
this driver writes a richer corpus.

Run once (Docling is slow on CPU; OCR is on by default in the #2 parser)::

    uv run python -m rag.spike_03.chunk_corpus

Output (gitignored, under ``data/spike/``):
* ``spike03_chunks.jsonl`` — one row per chunk: chunk_id, doc_id, title,
  section_path, block_types, token_estimate, and the full chunk ``text``
  (context header + body, exactly as the #2 chunker emits it).
"""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.chunk import ChunkConfig, chunk
from ingestion.parse import parse_document

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "spike"

# Same two documents as spike #2 (findings note records URLs + sha256).
DOCS = [
    ("nca5_ch2", "NCA5_Ch2_Climate-Trends.pdf", "NCA5 Chapter 2: Climate Trends"),
    (
        "esd_tipping_review",
        "esd-15-41-2024_tipping-cascades-review.pdf",
        "Climate tipping point interactions and cascades: a review",
    ),
]


def build() -> list[dict]:
    config = ChunkConfig()
    rows: list[dict] = []
    for doc_id, fname, title in DOCS:
        path = DATA_DIR / fname
        if not path.exists():
            raise SystemExit(f"MISSING: {path} (fetch per spike-02 findings note)")
        print(f"Parsing {doc_id} (Docling) ...", flush=True)
        doc = parse_document(path, doc_id=doc_id, title=title, backend="docling")
        chunks = chunk(doc, config)
        print(f"  {doc_id}: backend={doc.backend} blocks={len(doc.blocks)} chunks={len(chunks)}")
        for c in chunks:
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": doc.title,
                    "section_path": list(c.section_path),
                    "block_types": list(c.block_types),
                    "token_estimate": c.token_estimate,
                    "pages": list(c.pages),
                    "text": c.text,
                }
            )
    return rows


def main() -> int:
    rows = build()
    out = DATA_DIR / "spike03_chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
