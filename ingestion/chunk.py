"""PROTOTYPE (issue #2 spike) — structure-aware section chunker.

*** Spike / de-risking code, NOT the production chunker. ***
The production chunker is issue #7 (DESIGN §2.4, IMPLEMENTATION.md §1: "the
showcase bespoke code; the most heavily unit-tested module in the repo"). This
prototype exists to make chunk boundaries concrete enough to hand-review on two
real documents, and to seed #7's golden-output tests.

Pure function, no I/O and no heavy deps: ``chunk(StructuredDoc, ChunkConfig) ->
list[Chunk]``. The rules it prototypes (from the issue / DESIGN §2.4):

* split on headings — a chunk never spans a heading boundary;
* target <= ~500 tokens per chunk, splitting long sections at sentence bounds;
* one-sentence overlap between consecutive chunks *within the same section*;
* every chunk gets a prepended context header "document title -> section path";
* figure/table placeholders are retained with their captions as citable text.

Token counting here is a whitespace-word estimate, not a real tokenizer — good
enough to observe boundary behaviour; #7 will use the embedding model's
tokenizer. The estimate is documented so goldens are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.parse import HEADING_TYPES, Block, BlockType, StructuredDoc

# A block that carries citable prose. Headings structure the tree; page
# furniture (running heads/feet, page numbers) is dropped before chunking.
_CONTENT_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.FIGURE,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
    }
)


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 500
    overlap_sentences: int = 1
    context_arrow: str = " → "  # "→" joins document title and section path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]
    context_header: str
    text: str
    token_estimate: int
    block_types: tuple[str, ...]
    pages: tuple[int, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Token / sentence helpers (crude but documented, so goldens are reproducible)
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Whitespace-word count. A stand-in for a real tokenizer (see module doc)."""
    return len(text.split())


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Naive sentence split on ``. ! ?`` followed by a capital/quote/number.

    Deliberately simple; its failure modes (abbreviations, "e.g.", decimals,
    citations like "(IPCC, 2021).") are catalogued in the findings note as risks
    for #7.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# Section grouping
# --------------------------------------------------------------------------- #
@dataclass
class _Section:
    path: tuple[str, ...]
    blocks: list[Block]


def group_sections(doc: StructuredDoc) -> list[_Section]:
    """Walk blocks in reading order, maintaining a heading stack, and emit one
    ``_Section`` per run of content under a distinct heading path.

    A heading opens a new section context; content blocks accumulate into the
    current section. Because a new heading always starts a new ``_Section``, no
    downstream chunk can span a heading — the invariant is enforced structurally
    here, not by the splitter.
    """
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current: _Section | None = None

    def current_path() -> tuple[str, ...]:
        return tuple(text for _, text in heading_stack)

    for block in doc.blocks:
        if block.type in HEADING_TYPES:
            level = block.level if block.level and block.level > 0 else 1
            # Pop deeper-or-equal headings, then push this one.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text))
            current = None  # force a fresh section for content under this heading
            continue

        if block.type not in _CONTENT_TYPES:
            continue  # drop page furniture etc.

        if current is None or current.path != current_path():
            current = _Section(path=current_path(), blocks=[])
            sections.append(current)
        current.blocks.append(block)

    return sections


# --------------------------------------------------------------------------- #
# Rendering a block to citable text
# --------------------------------------------------------------------------- #
def block_text(block: Block) -> str:
    """Render a block to the text that goes into a chunk.

    Figure/table placeholders are kept and annotated with their caption so the
    unit stays citable (DESIGN §2.4) even though the visual is gone.
    """
    if block.type in (BlockType.FIGURE, BlockType.TABLE):
        label = "Figure" if block.type is BlockType.FIGURE else "Table"
        if block.caption:
            return f"[{label.upper()}] {block.caption}"
        return f"[{label.upper()} — no caption extracted]"
    return block.text


# --------------------------------------------------------------------------- #
# The chunker
# --------------------------------------------------------------------------- #
def chunk(doc: StructuredDoc, config: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a StructuredDoc into <=~max_tokens section-scoped chunks."""
    config = config or ChunkConfig()
    chunks: list[Chunk] = []

    for section in group_sections(doc):
        section_units = _section_units(section)
        if not section_units:
            continue
        for piece in _pack_units(section_units, config):
            idx = len(chunks)
            path = section.path or ("(document root)",)
            header = _context_header(doc.title, section.path, config)
            body = piece.text
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{idx:04d}",
                    doc_id=doc.doc_id,
                    section_path=path if section.path else (),
                    context_header=header,
                    text=f"{header}\n\n{body}",
                    token_estimate=estimate_tokens(body),
                    block_types=piece.block_types,
                    pages=piece.pages,
                )
            )
    return chunks


def _context_header(doc_title: str, section_path: tuple[str, ...], config: ChunkConfig) -> str:
    parts = [doc_title, *section_path] if section_path else [doc_title]
    return config.context_arrow.join(parts)


@dataclass
class _Unit:
    """A sentence-or-atomic unit tagged with its source block type and page."""

    text: str
    block_type: BlockType
    page: int | None
    atomic: bool  # figure/table/footnote units are never split further


def _section_units(section: _Section) -> list[_Unit]:
    units: list[_Unit] = []
    for block in section.blocks:
        rendered = block_text(block)
        if block.type in (BlockType.FIGURE, BlockType.TABLE, BlockType.FOOTNOTE):
            units.append(_Unit(rendered, block.type, block.page, atomic=True))
        else:
            for sent in split_sentences(rendered) or [rendered]:
                units.append(_Unit(sent, block.type, block.page, atomic=False))
    return units


@dataclass
class _Piece:
    text: str
    block_types: tuple[str, ...]
    pages: tuple[int, ...]


def _pack_units(units: list[_Unit], config: ChunkConfig) -> list[_Piece]:
    """Greedily pack units into <=max_tokens pieces, with sentence overlap.

    Overlap only carries *non-atomic* trailing sentences into the next piece, and
    only within this section (the caller never mixes sections). A single unit that
    alone exceeds max_tokens becomes its own oversized piece (recorded as a risk).
    """
    pieces: list[_Piece] = []
    cur: list[_Unit] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        text = " ".join(u.text for u in cur).strip()
        btypes = tuple(dict.fromkeys(u.block_type.value for u in cur))
        pages = tuple(dict.fromkeys(u.page for u in cur if u.page is not None))
        pieces.append(_Piece(text=text, block_types=btypes, pages=pages))

    for unit in units:
        utok = estimate_tokens(unit.text)
        if cur and cur_tokens + utok > config.max_tokens:
            flush()
            # Seed the next piece with the overlap tail (non-atomic sentences).
            overlap = _overlap_tail(cur, config.overlap_sentences)
            cur = list(overlap)
            cur_tokens = sum(estimate_tokens(u.text) for u in cur)
        cur.append(unit)
        cur_tokens += utok
    flush()
    return pieces


def _overlap_tail(units: list[_Unit], n_sentences: int) -> list[_Unit]:
    if n_sentences <= 0:
        return []
    tail: list[_Unit] = []
    for unit in reversed(units):
        if unit.atomic:
            break  # don't carry a figure/table/footnote across as "overlap"
        tail.append(unit)
        if len(tail) >= n_sentences:
            break
    return list(reversed(tail))
