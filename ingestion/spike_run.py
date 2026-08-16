"""PROTOTYPE (issue #2 spike) — driver: parse the two source PDFs and chunk them.

Not production code and not imported by anything shippable. Run it to reproduce
the hand-review material behind ``reviews/spike-02-parsing-findings.md``::

    uv run python -m ingestion.spike_run

Outputs (all under the gitignored ``data/spike/``):
* ``<doc>.blocks.txt``  — parsed blocks in reading order (backend sanity check)
* ``<doc>.chunks.txt``  — human-readable chunk dump for boundary hand-review
* ``<doc>.chunks.json`` — machine-readable chunk list (seeds #7 goldens)
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from ingestion.chunk import Chunk, ChunkConfig, chunk
from ingestion.parse import StructuredDoc, parse_document

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "spike"

DOCS = [
    ("nca5_ch2", "NCA5_Ch2_Climate-Trends.pdf", "NCA5 Chapter 2: Climate Trends"),
    (
        "esd_tipping_review",
        "esd-15-41-2024_tipping-cascades-review.pdf",
        "Climate tipping point interactions and cascades: a review",
    ),
]


def dump_blocks(doc: StructuredDoc) -> str:
    out = [f"# {doc.doc_id}  (backend={doc.backend}, title={doc.title!r})", ""]
    for i, b in enumerate(doc.blocks):
        lvl = f" L{b.level}" if b.level is not None else ""
        cap = f"  caption={b.caption!r}" if b.caption else ""
        pg = f" p{b.page}" if b.page is not None else ""
        out.append(f"[{i:04d}] {b.type.value}{lvl}{pg}: {b.text[:120]!r}{cap}")
    return "\n".join(out)


def dump_chunks(doc: StructuredDoc, chunks: list[Chunk]) -> str:
    out = [f"# {doc.doc_id}: {len(chunks)} chunks (backend={doc.backend})", ""]
    for c in chunks:
        out.append("=" * 88)
        out.append(f"id={c.chunk_id}  tokens~{c.token_estimate}  pages={c.pages}")
        out.append(f"section_path={c.section_path}")
        out.append(f"block_types={c.block_types}")
        out.append(f"context_header={c.context_header!r}")
        out.append("-" * 88)
        out.append(c.text)
        out.append("")
    return "\n".join(out)


def chunks_to_jsonable(chunks: list[Chunk]) -> list[dict]:
    rows = []
    for c in chunks:
        d = dataclasses.asdict(c)
        # drop the full text (with header) to keep the recorded list small;
        # keep everything needed to characterise boundaries.
        d.pop("text")
        d["section_path"] = list(c.section_path)
        d["block_types"] = list(c.block_types)
        d["pages"] = list(c.pages)
        rows.append(d)
    return rows


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "docling"
    config = ChunkConfig()
    summary = []
    for doc_id, fname, title in DOCS:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"MISSING: {path} (fetch per findings note); skipping")
            continue
        print(f"Parsing {doc_id} with backend={backend} ...", flush=True)
        doc = parse_document(path, doc_id=doc_id, title=title, backend=backend)
        chunks = chunk(doc, config)
        (DATA_DIR / f"{doc_id}.blocks.txt").write_text(dump_blocks(doc))
        (DATA_DIR / f"{doc_id}.chunks.txt").write_text(dump_chunks(doc, chunks))
        (DATA_DIR / f"{doc_id}.chunks.json").write_text(
            json.dumps(chunks_to_jsonable(chunks), indent=2)
        )
        n_over = sum(1 for c in chunks if c.token_estimate > config.max_tokens)
        n_fig = sum(1 for c in chunks if "figure" in c.block_types)
        n_tab = sum(1 for c in chunks if "table" in c.block_types)
        print(
            f"  {doc.doc_id}: backend={doc.backend} blocks={len(doc.blocks)} "
            f"chunks={len(chunks)} oversized={n_over} "
            f"chunks_with_figure={n_fig} chunks_with_table={n_tab}"
        )
        summary.append((doc.doc_id, doc.backend, len(doc.blocks), len(chunks), n_over))
    print("\nSUMMARY")
    for row in summary:
        print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
