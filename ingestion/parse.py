"""PROTOTYPE (issue #2 spike) — structure-aware PDF parsing.

*** This is spike / de-risking code, NOT the production parser. ***
The production parsing pipeline is issue #7 and starts from a red test suite
(IMPLEMENTATION.md §2 item 6). This module exists only to (a) exercise Docling
and the PyMuPDF fallback on two real documents and (b) feed the prototype
chunker (``ingestion.chunk``) so chunk boundaries can be hand-reviewed. Findings
live in ``reviews/spike-02-parsing-findings.md``.

Design seam it prototypes (IMPLEMENTATION.md §1, DESIGN §2.4):
    parse_document(path) -> StructuredDoc
with Docling and PyMuPDF as two implementations behind the one interface. Tests
in #7 will assert on ``StructuredDoc``, never on Docling internals.

Heavy dependencies (``docling``, ``pymupdf``) are imported lazily inside the
backend functions so that importing this module — and therefore
``ingestion.chunk`` and the committed characterisation tests — needs neither the
libraries nor the (gitignored) source PDFs. Install for the spike with::

    uv pip install docling pymupdf   # docling 2.120.1, pymupdf 1.28.2 used here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class BlockType(StrEnum):
    """Structural role of a parsed block. Deliberately small for the spike."""

    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_FURNITURE = "page_furniture"  # running headers/footers/page numbers


HEADING_TYPES = frozenset({BlockType.TITLE, BlockType.HEADING})


@dataclass
class Block:
    """One structural unit in reading order.

    ``level`` is the heading depth (1 = top section) for HEADING/TITLE blocks,
    otherwise ``None``. Figure/table blocks carry a placeholder in ``text`` and
    the human-readable caption (the citable text, DESIGN §2.4) in ``caption``.
    """

    type: BlockType
    text: str
    level: int | None = None
    caption: str | None = None
    page: int | None = None


@dataclass
class StructuredDoc:
    """Backend-independent parsed document — the seam the chunker consumes."""

    doc_id: str
    title: str
    blocks: list[Block] = field(default_factory=list)
    backend: str = "unknown"


# --------------------------------------------------------------------------- #
# Docling backend
# --------------------------------------------------------------------------- #
def parse_with_docling(path: str | Path, doc_id: str, title: str | None = None) -> StructuredDoc:
    """Parse a PDF with Docling into a StructuredDoc (prototype mapping)."""
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import (
        DocItemLabel,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    result = DocumentConverter().convert(str(path))
    ddoc = result.document

    blocks: list[Block] = []
    resolved_title = title

    for item, _level in ddoc.iterate_items():
        page = _first_page(item)

        if isinstance(item, TitleItem):
            text = (item.text or "").strip()
            if text:
                resolved_title = resolved_title or text
                blocks.append(Block(BlockType.TITLE, text, level=0, page=page))
            continue

        if isinstance(item, SectionHeaderItem):
            text = (item.text or "").strip()
            if text:
                lvl = getattr(item, "level", None) or 1
                blocks.append(Block(BlockType.HEADING, text, level=int(lvl), page=page))
            continue

        if isinstance(item, TableItem):
            cap = _caption_text(item, ddoc)
            blocks.append(Block(BlockType.TABLE, "[TABLE]", caption=cap, page=page))
            continue

        if isinstance(item, PictureItem):
            cap = _caption_text(item, ddoc)
            blocks.append(Block(BlockType.FIGURE, "[FIGURE]", caption=cap, page=page))
            continue

        if isinstance(item, ListItem):
            text = (item.text or "").strip()
            if text:
                blocks.append(Block(BlockType.LIST_ITEM, text, page=page))
            continue

        if isinstance(item, TextItem):
            text = (item.text or "").strip()
            if not text:
                continue
            label = getattr(item, "label", None)
            if label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                btype = BlockType.PAGE_FURNITURE
            elif label == DocItemLabel.CAPTION:
                btype = BlockType.CAPTION
            elif label == DocItemLabel.FOOTNOTE:
                btype = BlockType.FOOTNOTE
            else:
                btype = BlockType.TEXT
            blocks.append(Block(btype, text, page=page))
            continue

    return StructuredDoc(
        doc_id=doc_id,
        title=(resolved_title or doc_id),
        blocks=blocks,
        backend="docling",
    )


def _first_page(item) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        try:
            return int(prov[0].page_no)
        except Exception:  # noqa: BLE001 - prototype, best effort
            return None
    return None


def _caption_text(item, ddoc) -> str | None:
    try:
        cap = item.caption_text(ddoc)
    except Exception:  # noqa: BLE001 - prototype, best effort
        cap = None
    cap = (cap or "").strip()
    return cap or None


# --------------------------------------------------------------------------- #
# PyMuPDF fallback backend
# --------------------------------------------------------------------------- #
def parse_with_pymupdf(path: str | Path, doc_id: str, title: str | None = None) -> StructuredDoc:
    """Fallback parse with PyMuPDF.

    PyMuPDF has no document model, so headings are guessed from font size: a line
    whose dominant span is meaningfully larger than the modal body-text size is
    treated as a heading. This is deliberately crude — its weaknesses are exactly
    what the findings note is about (it cannot see tables, figure captions, or
    footnotes as such).
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    # First pass: find the modal (body) font size.
    sizes: dict[int, int] = {}
    lines: list[tuple[int, float, str]] = []  # (page, size, text)
    for pno, page in enumerate(doc, start=1):
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bucket = round(size)
                sizes[bucket] = sizes.get(bucket, 0) + 1
                lines.append((pno, size, text))
    body_size = max(sizes, key=sizes.get) if sizes else 10

    blocks: list[Block] = []
    for pno, size, text in lines:
        if round(size) > body_size + 1 and len(text) < 200:
            level = 1 if round(size) >= body_size + 4 else 2
            blocks.append(Block(BlockType.HEADING, text, level=level, page=pno))
        else:
            blocks.append(Block(BlockType.TEXT, text, page=pno))
    doc.close()
    return StructuredDoc(
        doc_id=doc_id,
        title=(title or doc_id),
        blocks=blocks,
        backend="pymupdf",
    )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
def parse_document(
    path: str | Path,
    doc_id: str,
    title: str | None = None,
    backend: str = "docling",
) -> StructuredDoc:
    """Parse ``path`` into a StructuredDoc via the requested backend.

    ``backend="docling"`` falls back to PyMuPDF if Docling raises, mirroring the
    DESIGN §2.4 "Docling with PyMuPDF fallback" wiring.
    """
    if backend == "pymupdf":
        return parse_with_pymupdf(path, doc_id, title)
    try:
        return parse_with_docling(path, doc_id, title)
    except Exception as exc:  # noqa: BLE001 - spike: prove the fallback path
        print(f"[parse] Docling failed ({exc!r}); falling back to PyMuPDF")
        return parse_with_pymupdf(path, doc_id, title)
