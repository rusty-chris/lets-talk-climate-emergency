"""Structure-aware document parsing — the production parse seam (DESIGN §2.4).

Design seam (IMPLEMENTATION.md §1):
    parse_document(path, doc_id) -> StructuredDoc
with Docling (structure-aware primary) and PyMuPDF (a *degraded, loud*
fallback) as two implementations behind the one interface. Callers assert on
``StructuredDoc``, never on Docling internals; the production pipeline
(``ingestion.pipeline``) flags any ``backend == "pymupdf"`` document for hand
review (spike-02 finding: the font-size heuristic is unusable for journal
articles, so a PyMuPDF result is "no reliable structure", never trusted
silently). Issue #7 productionised the #2 spike:

* OCR is disabled for born-digital PDFs (spike-02 cost note: RapidOCR dominated
  runtime and is unnecessary when a text layer is present); model/OCR downloads
  stay out of git.
* the Docling→PyMuPDF fallback logs a loud warning rather than failing silent.

Heavy dependencies (``docling``, ``pymupdf``) are imported lazily inside the
backend functions so that importing this module — and the pure
``StructuredDoc``/``Block`` types the unit tier builds directly — needs neither
the libraries nor any source PDF. They live in the optional ``parse`` extras
group (``docling`` pulls torch/transformers — multi-GB) rather than the base
dependencies, so the api/ui serving image never carries them (issue #125):
dev/CI/the ingestion pipeline install them with ``uv sync --extra parse`` (or
``--all-extras``). When a backend is invoked without the extra installed, the
lazy import raises :class:`ParseBackendMissingError` — a typed,
``except ImportError``-compatible error naming the install hint — instead of
a bare ``ModuleNotFoundError`` surfacing deep inside the backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

_log = logging.getLogger(__name__)

#: Install hint shared by every "backend not installed" error message
#: (issue #125): names both the uv extras form and the pip extras form.
_PARSE_EXTRA_HINT = "install it with `uv sync --extra parse` (or `pip install .[parse]`)"


class ParseBackendMissingError(ModuleNotFoundError):
    """A parse backend (docling/pymupdf) was invoked without the ``parse``
    extra installed.

    Subclasses ``ModuleNotFoundError`` so existing ``except ImportError``
    handling still catches it, while giving an actionable message instead of
    a bare import error surfacing deep inside the backend (#125).
    """


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
def _docling_converter():
    """A Docling converter with OCR disabled (born-digital PDFs, spike-02
    cost note): the layout model recovers structure; OCR only adds runtime
    and a model download that stays out of git."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_with_docling(path: str | Path, doc_id: str, title: str | None = None) -> StructuredDoc:
    """Parse a PDF with Docling into a StructuredDoc (structure-aware primary)."""
    try:
        from docling_core.types.doc.document import (
            DocItemLabel,
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
            TitleItem,
        )

        result = _docling_converter().convert(str(path))
    except ModuleNotFoundError as exc:
        raise ParseBackendMissingError(
            f"Docling is not installed; {_PARSE_EXTRA_HINT} to parse with the "
            "structure-aware primary backend.",
            name=exc.name,
        ) from exc
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
            # Keep table *data* citable (spike-02 finding 3c): export the cells
            # as markdown so numeric tables contribute their numbers, not a bare
            # placeholder. Best-effort — fall back to the placeholder on failure.
            try:
                cells = (item.export_to_markdown(ddoc) or "").strip()
            except Exception:  # noqa: BLE001 - export is best-effort
                cells = ""
            blocks.append(Block(BlockType.TABLE, cells or "[TABLE]", caption=cap, page=page))
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
    try:
        import pymupdf
    except ModuleNotFoundError as exc:
        raise ParseBackendMissingError(
            f"PyMuPDF is not installed; {_PARSE_EXTRA_HINT} to parse with the "
            "degraded fallback backend.",
            name=exc.name,
        ) from exc

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
    except Exception as exc:  # noqa: BLE001 - any Docling failure must degrade, loudly
        # LOUD fallback (DESIGN §2.4, amended): never silent. The pipeline reads
        # ``backend == "pymupdf"`` off the returned StructuredDoc and flags the
        # document for hand review before indexing.
        _log.warning(
            "Docling failed to parse %s (%r); falling back to the DEGRADED PyMuPDF "
            "backend — its structure is unreliable and the document must be hand-reviewed "
            "before indexing",
            doc_id,
            exc,
        )
        return parse_with_pymupdf(path, doc_id, title)
