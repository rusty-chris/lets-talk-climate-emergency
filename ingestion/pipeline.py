"""Production ingestion pipeline (issue #7): manifest-driven fetch → verify
→ parse → chunk → citation metadata → custom-content citation blocks.

The requirements come from:

- DESIGN.md §2.1–§2.4 (as amended: PyMuPDF is a *degraded, loud* fallback
  with hand-review flagging; the assessed-range statements are a launch
  dependency; IPCC headline statements sit behind a feature flag,
  default OFF, capped ≤10 per SPM);
- ``reviews/spike-02-parsing-findings.md`` — the eight recorded parsing
  failure modes, plus the #41 cap-semantics amendment: the ≤max_tokens
  cap covers the chunk's full embedded text *including* the context
  header; overlap seeding re-checks the cap; an oversized atomic unit is
  split at sentence boundaries rather than passed through;
- ``reviews/spike-03-probe-findings.md`` — the custom-content block shape
  the generation call consumes, and the rule that the context header is
  carried in the block's ``context`` field so citations quote only the
  evidence body;
- issue #100 — the manifest ``sha256`` pins the *ingest artefact* at
  ``source_url``; this pipeline re-fetch-verifies exactly those bytes.

Licensing gate: this module CONSUMES :mod:`ingestion.manifest` (issue #5)
— it never re-implements an invariant. Any invariant-violating document
refuses the whole run (DESIGN §2.1) *before* any fetch or parse happens.

Seams (IMPLEMENTATION.md §1): parsing is injected —
``parser(path, doc_id) -> StructuredDoc`` — so no unit or integration
test needs the heavy Docling install; Docling and PyMuPDF are two
implementations behind the seam. Transport is injected for fetches
(tests use ``file://``). Token counting is injected via
:class:`IngestConfig` so the unit tier stays free of model weights;
the production default is the embedding model's tokenizer
(spike finding 7).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from ingestion.manifest import load_corpus_manifest, verify_fetched_sha256
from ingestion.parse import Block, BlockType, StructuredDoc, parse_document

__all__ = [
    "IngestConfig",
    "IngestError",
    "ChunkRecord",
    "DocumentIngestRecord",
    "IngestResult",
    "parse_html",
    "reclassify_furniture_headings",
    "reconstruct_section_hierarchy",
    "strip_front_matter",
    "associate_captions",
    "segregate_references",
    "normalise_text",
    "strip_note_markers",
    "split_sentences",
    "count_tokens",
    "extract_confidence_markers",
    "chunk_document",
    "check_assessed_range_statements_present",
    "ingest_corpus",
]


class IngestError(RuntimeError):
    """A pipeline refusal that is not a licensing-invariant violation.

    Raised for: the headline-statements cap (>10 statements per SPM fails
    the ingest, DESIGN §2.1 Tier C), a missing assessed-range statements
    source (DESIGN §2.3 launch dependency), and any other condition where
    continuing would emit chunks the design forbids. Licensing refusals
    keep raising :class:`ingestion.manifest.ManifestError` (and sha256
    drift its :class:`ingestion.manifest.Sha256MismatchError` subclass) —
    callers distinguish the classes.
    """


@dataclass(frozen=True)
class IngestConfig:
    """Tunable ingest behaviour. Defaults are the DESIGN values.

    ``token_counter``: injectable token-count function used for the chunk
    cap and floor. ``None`` means the production embedding tokenizer
    (spike finding 7 — never the whitespace estimate the spike used).
    Unit tests inject a deterministic counter so the unit tier needs no
    model weights (IMPLEMENTATION.md §3).

    ``headline_statements_enabled`` **defaults to False** (DESIGN §2.1
    Tier C: the IPCC curated headline-statements set is behind a feature
    flag until the legal check clears it; a refusal from the IPCC is then
    a config change, not a scramble).
    """

    max_tokens: int = 500
    min_tokens: int = 20
    overlap_sentences: int = 1
    context_arrow: str = " → "
    headline_statements_enabled: bool = False
    headline_statements_cap: int = 10
    enforce_assessed_ranges: bool = True
    token_counter: Callable[[str], int] | None = None


@dataclass(frozen=True)
class ChunkRecord:
    """One production chunk with its full citation metadata (DESIGN §2.4).

    Contract points the tests pin:

    - ``body`` is the citable evidence text and carries NO context header
      (spike-03: the header leaks into ``cited_text`` otherwise; it moves
      to the citation block's ``context`` field).
    - ``embedding_text`` (header + blank line + body) is what gets
      embedded; ``token_count`` is the configured counter applied to
      ``embedding_text`` — i.e. the ≤max_tokens cap INCLUDES the context
      header (#41 amendment).
    - ``chunk_id`` is a pure function of the chunk's content and
      provenance: identical input → identical id across reruns; a change
      to this chunk's text changes the id; changes to *other* documents
      never do (the idempotent incremental re-embedding hook,
      DESIGN §2.4 "embedding idempotent/incremental"). Ids are pairwise
      unique within a run (#138): a repeated identical
      ``(section_path, body)`` pair folds its occurrence ordinal into the
      hash, deterministically.
    - ``confidence_markers`` are the calibrated-language phrases found in
      ``body`` (see :func:`extract_confidence_markers`).
    - ``consensus_position`` and ``source_type`` propagate verbatim from
      the document's manifest entry (§2.3 severity-skew guardrail; §2.5
      voices labelling).
    - ``citation_metadata`` carries the §2.4 per-chunk schema from the
      manifest entry: at least ``licence``, ``attribution_text``,
      ``canonical_url``, ``permitted_context``, plus the document title.
    """

    chunk_id: str
    doc_id: str
    section_path: tuple[str, ...]
    context_header: str
    body: str
    token_count: int
    confidence_markers: tuple[str, ...]
    consensus_position: str
    source_type: str
    citation_metadata: Mapping[str, Any]
    block_types: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    #: Review finding #139: the one deliberate cap exception. True only
    #: when the chunk is a single unsplittable unit (a table row never
    #: split mid-cell, a token-dense single word) whose token count alone
    #: exceeds the cap — emitted alone and flagged so an indexer can see,
    #: log or quarantine it. Never set on ordinary chunks, which all
    #: respect ``token_count <= max_tokens``.
    oversized_atomic: bool = False

    @property
    def embedding_text(self) -> str:
        """Header + body — the text the embedder sees and the cap covers."""
        return f"{self.context_header}\n\n{self.body}"


@dataclass(frozen=True)
class DocumentIngestRecord:
    """Per-document provenance recorded by the run (never silent).

    ``degraded_fallback`` / ``needs_hand_review`` / ``warnings`` implement
    the amended DESIGN §2.4 rule: a PyMuPDF-parsed document is a LOUD
    degraded result — a per-document warning is recorded here (and in the
    written ingest manifest) and the document is flagged for hand review
    before indexing. ``skipped``/``skip_reason`` record feature-flag skips
    (e.g. headline statements while the flag is off).
    """

    doc_id: str
    parse_backend: str
    degraded_fallback: bool = False
    needs_hand_review: bool = False
    warnings: tuple[str, ...] = ()
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class IngestResult:
    """The full output of one ingest run.

    ``blocks`` are the custom-content citation block payloads, one per
    citable unit (chunk), built by :func:`ingestion.blocks.build_citation_blocks`.
    ``documents`` maps doc_id → :class:`DocumentIngestRecord`.
    """

    chunks: tuple[ChunkRecord, ...] = ()
    blocks: tuple[Mapping[str, Any], ...] = ()
    documents: Mapping[str, DocumentIngestRecord] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Block-type values whose chunks are deliberately atomic — never merged
#: with prose, exempt from the tiny-chunk floor (finding 8), split only at
#: sentence boundaries when they exceed the cap (#41 path b).
_ATOMIC_TYPES = frozenset({"table", "figure", "footnote"})

#: Content blocks that carry citable evidence into chunks. Headings shape
#: the section tree; titles, captions (folded into their object by
#: :func:`associate_captions`) and page furniture never chunk directly.
_CONTENT_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.FIGURE,
        BlockType.FOOTNOTE,
    }
)

#: A numeric section prefix like ``2`` / ``2.2`` / ``2.2.1`` — the depth is
#: the dot-count + 1 (finding 1: Docling flattens every heading to level 1).
_NUM_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\s|$)")

#: Front-matter / boilerplate heading labels (finding 2; extended by
#: review finding #140 with the bare labels the real NCA5 chapter emits):
#: author-role, affiliation, table-of-contents and recommended-citation
#: headings never chunk as evidence.
_FRONT_MATTER_RE = re.compile(
    r"(?i)\b("
    r"lead authors?|contributing authors?|coordinating authors?|corresponding authors?"
    r"|chapter authors?|recommended citation|cover art|acknowledge?ments?"
    r"|affiliations?|about the authors?|editors?|reviewers?"
    r"|authors?|table of contents|contents"
    r")\b"
)

#: Affiliation-shaped text (#140): institution keywords, superscript
#: digit-comma runs ("Nico Wunderling 1,2,3") and enumerated address
#: lines. Used only in the TITLE → first-heading span, where real papers
#: put the author/affiliation wall (never applied to ordinary prose).
_AFFILIATION_KEYWORD_RE = re.compile(
    r"(?i)\b(universit|institut|laborato|department|centre|center|college"
    r"|school of|observatory|academy|faculty)"
)
_SUPERSCRIPT_RUN_RE = re.compile(r"\b\d{1,2}\s*(?:[,;]\s*\d{1,2})+\b")
_ENUMERATED_ADDRESS_RE = re.compile(r"(?:^|[.;]\s+)\d{1,2}\s+[A-Z]")


def _looks_like_affiliation(text: str) -> bool:
    """Heuristic for author/affiliation walls (#140): digit-comma
    superscript runs, or institution keywords combined with enumeration
    digits / dense commas. Abstract-like prose (few digits, no
    institution keywords) stays False and is kept."""
    if _SUPERSCRIPT_RUN_RE.search(text):
        return True
    if not _AFFILIATION_KEYWORD_RE.search(text):
        return False
    digit_tokens = sum(1 for word in text.split() if any(ch.isdigit() for ch in word))
    if digit_tokens >= 2 or len(_ENUMERATED_ADDRESS_RE.findall(text)) >= 2:
        return True
    return text.count(",") >= 3 and digit_tokens >= 1


#: Reference / bibliography section headings (finding 5): segregated out of
#: the evidence index (they were 20–30% of spike chunks, low citable value).
_REFERENCE_RE = re.compile(
    r"(?i)^\s*(?:\d+(?:\.\d+)*\s+)?(references|bibliography|works cited|reference list)\s*$"
)

#: PDF ligatures (finding 6) → their plain-letter equivalents.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

#: Line-break hyphenation (finding 6): ``tempera-\nture`` → ``temperature``.
#: Only across a newline, so real hyphenated compounds survive.
_HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\n(?=\w)")

#: Superscript note/citation markers glued to a sentence end (finding 4):
#: a letter, then sentence punctuation, then 1–3 digits before whitespace.
#: The letter lookbehind is what protects decimals ("1.9") from stripping.
_NOTE_MARKER_RE = re.compile(r"(?<=[A-Za-z])([.!?])(\d{1,3})(?=\s|$)")

#: Real sentence boundary: sentence punctuation, whitespace, an uppercase
#: start (finding 7). Decimals and "Fig. 1"/"e.g." don't match (a digit or
#: lowercase follows); a short protected-abbreviation list re-joins the
#: rare uppercase-after-abbreviation case.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PROTECTED_ABBREV = frozenset(
    {
        "fig",
        "eq",
        "no",
        "al",
        "cf",
        "ca",
        "approx",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "vs",
        "st",
        "sr",
        "jr",
        "e.g",
        "i.e",
        "etc",
    }
)

#: Calibrated-language markers (DESIGN §2.4), longest phrasing first so the
#: alternation captures "very likely" before the bare "likely".
_CONFIDENCE_MARKERS = [
    "virtually certain",
    "more likely than not",
    "about as likely as not",
    "extremely likely",
    "extremely unlikely",
    "very likely",
    "very unlikely",
    "unlikely",
    "likely",
    "very high confidence",
    "high confidence",
    "medium confidence",
    "low confidence",
]
_CONFIDENCE_RE = re.compile(
    r"\b(" + "|".join(re.escape(marker) for marker in _CONFIDENCE_MARKERS) + r")\b",
    re.IGNORECASE,
)
#: Positive likelihood markers that a preceding "not" negates (the trap).
_POSITIVE_LIKELIHOOD = frozenset(
    {"likely", "very likely", "extremely likely", "more likely than not"}
)


# ---------------------------------------------------------------------------
# Parsing (HTML direct; PDFs arrive through the injected parser seam)
# ---------------------------------------------------------------------------


class _HTMLBuilder(HTMLParser):
    """Stream an explainer page into structural blocks (DESIGN §2.4).

    ``h1`` becomes the TITLE; ``h2``–``h6`` become HEADING blocks whose
    level reflects the markup depth (so nesting survives without Docling);
    ``<p>``/``<li>`` become TEXT/LIST_ITEM; ``<table>`` keeps its cell text
    citable; ``<figure>`` pairs with its ``<figcaption>``. Content inside
    ``<script>``/``<style>``/``<nav>``/``<header>``/``<footer>`` (page
    furniture) is dropped.
    """

    _SKIP = frozenset({"script", "style", "nav", "header", "footer"})
    _HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title: str | None = None
        self._skip_depth = 0
        self._mode: str | None = None
        self._buf: list[str] = []
        self._heading_tag: str | None = None
        self._in_table = 0
        self._in_table_caption = False
        self._table_cells: list[str] = []
        self._table_caption: list[str] = []
        self._in_figure = 0
        self._in_figcaption = False
        self._fig_caption: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._in_table += 1
            self._table_cells = []
            self._table_caption = []
            return
        if self._in_table:
            if tag == "caption":
                self._in_table_caption = True
            return
        if tag == "figure":
            self._in_figure += 1
            self._fig_caption = []
            return
        if self._in_figure:
            if tag == "figcaption":
                self._in_figcaption = True
            return
        if tag in self._HEADINGS:
            self._mode = "heading"
            self._heading_tag = tag
            self._buf = []
        elif tag == "p":
            self._mode = "text"
            self._buf = []
        elif tag == "li":
            self._mode = "listitem"
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_table:
            (self._table_caption if self._in_table_caption else self._table_cells).append(data)
            return
        if self._in_figure:
            if self._in_figcaption:
                self._fig_caption.append(data)
            return
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._in_table:
            if tag == "caption":
                self._in_table_caption = False
            elif tag == "table":
                self._in_table -= 1
                cells = " ".join(t.strip() for t in self._table_cells if t.strip())
                caption = " ".join(t.strip() for t in self._table_caption if t.strip()) or None
                self.blocks.append(Block(BlockType.TABLE, cells or "[TABLE]", caption=caption))
            return
        if self._in_figure:
            if tag == "figcaption":
                self._in_figcaption = False
            elif tag == "figure":
                self._in_figure -= 1
                caption = " ".join(t.strip() for t in self._fig_caption if t.strip()) or None
                self.blocks.append(Block(BlockType.FIGURE, "[FIGURE]", caption=caption))
            return
        text = " ".join("".join(self._buf).split())
        if tag in self._HEADINGS and self._mode == "heading":
            if tag == "h1":
                self.title = self.title or text
                if text:
                    self.blocks.append(Block(BlockType.TITLE, text, level=0))
            elif text:
                self.blocks.append(Block(BlockType.HEADING, text, level=int(tag[1]) - 1))
            self._mode = None
            self._buf = []
        elif tag == "p" and self._mode == "text":
            if text:
                self.blocks.append(Block(BlockType.TEXT, text))
            self._mode = None
            self._buf = []
        elif tag == "li" and self._mode == "listitem":
            if text:
                self.blocks.append(Block(BlockType.LIST_ITEM, text))
            self._mode = None
            self._buf = []


def parse_html(html: str, doc_id: str, title: str | None = None) -> StructuredDoc:
    """Parse HTML directly into a StructuredDoc (DESIGN §2.4: HTML direct).

    Contract: ``h1``–``h6`` become TITLE/HEADING blocks with their true
    nesting level; body prose becomes TEXT blocks under the enclosing
    heading; ``<table>`` cell content is retained citable in the TABLE
    block's text (spike finding 3c — never a bare placeholder);
    ``<figcaption>``/caption markup attaches to its figure; page furniture
    (nav, header, footer, script, style) is excluded.
    """
    builder = _HTMLBuilder()
    builder.feed(html)
    builder.close()
    return StructuredDoc(
        doc_id=doc_id,
        title=title or builder.title or doc_id,
        blocks=builder.blocks,
        backend="html",
    )


def _heading_depth(text: str, fallback_level: int | None) -> int:
    match = _NUM_PREFIX_RE.match(text)
    if match:
        return match.group(1).count(".") + 1
    if fallback_level and fallback_level > 0:
        return fallback_level
    return 1


def reconstruct_section_hierarchy(doc: StructuredDoc) -> StructuredDoc:
    """Recover heading nesting that the PDF parser flattened (finding 1).

    Docling returns every heading at level 1; the hierarchy lives only in
    the heading text (numeric prefixes such as ``2`` / ``2.2`` / ``2.2.1``,
    or Key-Message structure). Contract: after reconstruction, a block
    under a ``2.2.1``-style heading resolves to the nested section path
    ``('2 …', '2.2 …', '2.2.1 …')``, not a single flat element.
    """
    blocks: list[Block] = []
    for block in doc.blocks:
        if block.type is BlockType.HEADING:
            blocks.append(
                Block(
                    block.type,
                    block.text,
                    level=_heading_depth(block.text, block.level),
                    caption=block.caption,
                    page=block.page,
                )
            )
        else:
            blocks.append(block)
    return StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=blocks, backend=doc.backend)


#: A heading whose exact text recurs at least this many times in one
#: document is page furniture (a running head re-typed as a heading by
#: the parser, #141), not structure. Real section headings do not repeat
#: verbatim; running heads repeat once per page.
_FURNITURE_RECURRENCE_THRESHOLD = 3


def reclassify_furniture_headings(doc: StructuredDoc) -> StructuredDoc:
    """Reclassify running-head page furniture that arrived typed as a
    heading (review finding #141).

    On the real NCA5 chapter Docling emits the running head 'Fifth
    National Climate Assessment' as a SectionHeaderItem (not a
    PAGE_HEADER TextItem), which terminated References segregation
    mid-bibliography and opened pseudo-sections across the chapter. A
    HEADING whose text equals the document title, or whose exact text
    recurs ``>= _FURNITURE_RECURRENCE_THRESHOLD`` times, becomes
    PAGE_FURNITURE — run before front-matter stripping, reference
    segregation and section walking so none of them ever see it as
    structure.
    """
    heading_counts = Counter(
        block.text.strip().lower() for block in doc.blocks if block.type is BlockType.HEADING
    )
    title_norm = (doc.title or "").strip().lower()
    blocks: list[Block] = []
    for block in doc.blocks:
        if block.type is BlockType.HEADING:
            norm = block.text.strip().lower()
            if norm == title_norm or heading_counts[norm] >= _FURNITURE_RECURRENCE_THRESHOLD:
                blocks.append(Block(BlockType.PAGE_FURNITURE, block.text, page=block.page))
                continue
        blocks.append(block)
    return StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=blocks, backend=doc.backend)


def strip_front_matter(doc: StructuredDoc) -> StructuredDoc:
    """Drop front-matter/boilerplate noise before chunking (finding 2;
    hardened by review finding #140).

    Author/affiliation lists, role lines ("Chapter Lead Author", "Cover
    Art"), bare "Authors"/"Table of Contents" sections and
    recommended-citation blocks must not reach the chunk output — neither
    as sections nor as body text. Leading prose before the first
    structural head is dropped; between the document TITLE and the first
    genuine heading (where real papers put the author/affiliation wall,
    AFTER the first head — the #140 gap) affiliation-shaped text is
    dropped while abstract-like prose is kept.
    """
    out: list[Block] = []
    head_seen = False
    title_front_zone = False
    skip_until_head = False
    for block in doc.blocks:
        if block.type is BlockType.TITLE:
            head_seen = True
            title_front_zone = True
            skip_until_head = False
            out.append(block)
            continue
        if block.type is BlockType.HEADING:
            head_seen = True
            title_front_zone = False
            if _FRONT_MATTER_RE.search(block.text):
                skip_until_head = True
                continue
            skip_until_head = False
            out.append(block)
            continue
        if not head_seen:
            # Leading affiliation/boilerplate prose before any head — drop.
            if block.type in (BlockType.TEXT, BlockType.LIST_ITEM, BlockType.PAGE_FURNITURE):
                continue
            out.append(block)
            continue
        if skip_until_head:
            continue
        if (
            title_front_zone
            and block.type in (BlockType.TEXT, BlockType.LIST_ITEM)
            and _looks_like_affiliation(block.text)
        ):
            # The affiliation wall between the paper's TITLE and its first
            # real heading (#140) — never citable evidence.
            continue
        out.append(block)
    return StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=out, backend=doc.backend)


def associate_captions(doc: StructuredDoc) -> StructuredDoc:
    """Pair each figure/table with its caption, de-duplicated (finding 3;
    hardened by review finding #138).

    A caption emitted as a separate adjacent block attaches to its
    figure/table — searched in BOTH directions ("caption sits in the very
    next block" is not the general case on real Docling output: 10 of 16
    real NCA5 figure chunks came out captionless under next-block-only
    pairing). Caption text appears exactly once in the output (no
    attached-plus-standalone duplicates); an orphan caption attached to
    nothing is dropped (never evidence).
    """
    blocks = doc.blocks
    consumed: set[int] = set()
    resolved: dict[int, str | None] = {}
    for index, block in enumerate(blocks):
        if block.type not in (BlockType.FIGURE, BlockType.TABLE):
            continue
        caption = block.caption
        nxt = blocks[index + 1] if index + 1 < len(blocks) else None
        prev = blocks[index - 1] if index > 0 else None
        if caption is None:
            # Nearest-CAPTION search, next block first, then the previous
            # one (#138) — each caption pairs with at most one object.
            if nxt is not None and nxt.type is BlockType.CAPTION and index + 1 not in consumed:
                caption = nxt.text
                consumed.add(index + 1)
            elif prev is not None and prev.type is BlockType.CAPTION and index - 1 not in consumed:
                caption = prev.text
                consumed.add(index - 1)
        else:
            # Already attached (e.g. Docling caption refs / <figcaption>):
            # consume an adjacent duplicate so the text appears once.
            for neighbour_index, neighbour in ((index + 1, nxt), (index - 1, prev)):
                if (
                    neighbour is not None
                    and neighbour_index not in consumed
                    and neighbour.type in (BlockType.CAPTION, BlockType.TEXT)
                    and neighbour.text.strip() == caption.strip()
                ):
                    consumed.add(neighbour_index)
                    break
        resolved[index] = caption
    out: list[Block] = []
    for index, block in enumerate(blocks):
        if index in consumed:
            continue
        if block.type in (BlockType.FIGURE, BlockType.TABLE):
            out.append(
                Block(
                    block.type,
                    block.text,
                    level=block.level,
                    caption=resolved.get(index),
                    page=block.page,
                )
            )
        elif block.type is BlockType.CAPTION:
            # Orphan caption not attached to any object — drop (never evidence).
            continue
        else:
            out.append(block)
    return StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=out, backend=doc.backend)


def segregate_references(doc: StructuredDoc) -> tuple[StructuredDoc, tuple[Block, ...]]:
    """Split the reference/bibliography section out of the evidence stream
    (finding 5).

    Returns ``(evidence_doc, reference_blocks)``: the References/
    Bibliography section is never emitted as evidence chunks (it was
    20–30% of all spike chunks and dilutes retrieval); its blocks are
    returned separately for optional structured storage.
    """
    out: list[Block] = []
    refs: list[Block] = []
    in_references = False
    reference_level = 0
    for block in doc.blocks:
        if block.type is BlockType.HEADING:
            if _REFERENCE_RE.match(block.text):
                in_references = True
                reference_level = block.level or 1
                refs.append(block)
                continue
            if in_references and (block.level or 1) <= reference_level:
                in_references = False
        if in_references:
            refs.append(block)
            continue
        out.append(block)
    return (
        StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=out, backend=doc.backend),
        tuple(refs),
    )


# ---------------------------------------------------------------------------
# Text normalisation (findings 4, 6, 7)
# ---------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Normalise PDF extraction artefacts (finding 6).

    Contract: line-break hyphenation is rejoined (``tempera-`` +
    linebreak + ``ture`` becomes ``temperature``); unicode ligatures
    (ﬁ, ﬂ, …) map to their plain-letter equivalents; decimals and real
    hyphenated compounds already present in the prose are left intact.
    """
    text = _HYPHEN_BREAK_RE.sub("", text)
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    return text


def strip_note_markers(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove superscript note/citation markers flattened into prose
    (finding 4), returning ``(clean_text, markers)``.

    NCA5-style extraction glues superscript reference numbers to sentence
    ends ("…considerably more.67 In just three decades…"): the stray
    numeral is stripped from the text and captured in ``markers``.
    Genuine numbers — decimals ("1.9 °C") and in-sentence values
    ("rose by 67 mm") — are never touched.
    """
    markers: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        markers.append(match.group(2))
        return match.group(1)

    clean = _NOTE_MARKER_RE.sub(_capture, text)
    return clean, tuple(markers)


def split_sentences(text: str) -> list[str]:
    """Real sentence segmentation (finding 7 — not the spike regex).

    Contract: abbreviations ("e.g.", "Fig. 1", "et al."), decimals and
    parenthesised citations do not produce false sentence breaks; ordinary
    ``.``/``!``/``?`` boundaries do.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    merged: list[str] = []
    for part in parts:
        if merged:
            last_word = re.split(r"\s+", merged[-1].rstrip())[-1].rstrip(".").lower()
            if last_word in _PROTECTED_ABBREV:
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return [part.strip() for part in merged if part.strip()]


def count_tokens(text: str) -> int:
    """Production token counter — the embedding model's tokenizer
    (finding 7), used when :attr:`IngestConfig.token_counter` is None.

    Unit tests never call this (model weights live outside the unit
    tier); they inject a deterministic counter instead. Until the RAG
    embedding model is pinned (#3/#8) this is a deterministic
    word-plus-punctuation stand-in so ``make corpus`` runs without model
    weights; production wires the real tokenizer through
    ``IngestConfig.token_counter``.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


# ---------------------------------------------------------------------------
# Citation metadata
# ---------------------------------------------------------------------------


def extract_confidence_markers(text: str) -> tuple[str, ...]:
    """Extract calibrated-language markers from evidence text (DESIGN §2.4).

    Captures IPCC-style likelihood and confidence phrasing ("virtually
    certain", "very likely", "likely", "unlikely", "high/medium/low
    confidence", …) as they appear. Negation trap (issue #7 TDD plan 8):
    "not likely" must NOT be recorded as "likely", and "unlikely" must be
    captured as "unlikely" — the bare positive marker never appears for a
    negated or prefixed occurrence.
    """
    found: list[str] = []
    for match in _CONFIDENCE_RE.finditer(text):
        marker = match.group(1).lower()
        if marker in _POSITIVE_LIKELIHOOD:
            preceding = text[: match.start()].rstrip().lower()
            if preceding.endswith("not"):
                continue
        if marker not in found:
            found.append(marker)
    return tuple(found)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _context_header(title: str, section_path: tuple[str, ...], config: IngestConfig) -> str:
    parts = [title, *section_path] if section_path else [title]
    return config.context_arrow.join(parts)


def _chunk_id(doc_id: str, section_path: tuple[str, ...], body: str, occurrence: int = 0) -> str:
    # Content + provenance hash: same input → same id; a body change moves
    # only this chunk's id; another section's edit never touches it.
    # ``occurrence`` (#138) disambiguates repeated identical
    # (section_path, body) pairs within one document: the first occurrence
    # hashes exactly as before (ids stable across this change), later
    # occurrences fold their ordinal into the hash. The ordinal counts
    # only same-section duplicates in document order, so it is
    # deterministic across reruns and untouched by edits elsewhere.
    raw = "\x1f".join([doc_id, "\x1e".join(section_path), body])
    if occurrence:
        raw = f"{raw}\x1f#{occurrence}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{digest}"


def _embed_tokens(header: str, body: str, counter: Callable[[str], int]) -> int:
    return counter(f"{header}\n\n{body}")


#: A markdown table separator row (``|----|----|`` and ``|:---:|`` forms):
#: pure alignment furniture with no citable content, yet a
#: punctuation-aware token counter prices a wide one at hundreds of
#: tokens (#139 — the mechanism behind the real 1742-token table chunk).
#: Stripped before token counting and never emitted in a citable body.
#: Data rows are safe: any digit or letter fails the character class.
_TABLE_SEPARATOR_ROW_RE = re.compile(r"^[\s|:-]*-[\s|:-]*$")


def _strip_separator_rows(cells: str) -> str:
    return "\n".join(line for line in cells.splitlines() if not _TABLE_SEPARATOR_ROW_RE.match(line))


def _render_block_body(block: Block) -> str:
    """Render an atomic block to its citable chunk text (finding 3c)."""
    if block.type is BlockType.FIGURE:
        parts = ["[FIGURE]"]
        if block.caption:
            parts.append(block.caption.strip())
        return " ".join(parts)
    if block.type is BlockType.TABLE:
        parts: list[str] = []
        cells = _strip_separator_rows((block.text or "").strip()).strip()
        if cells and cells != "[TABLE]":
            parts.append(cells)
        if block.caption:
            parts.append(block.caption.strip())
        return " ".join(parts) if parts else "[TABLE]"
    # FOOTNOTE / other atomic prose.
    return normalise_text(block.text)


def _split_oversized(sentence: str, header: str, config: IngestConfig) -> list[tuple[str, bool]]:
    """Split a single unit that exceeds the cap even alone, at word bounds.

    Returns ``(piece, oversized_atomic)`` pairs. A single word whose token
    count alone exceeds the cap (a long DOI/URL under a tight cap) is the
    #139 policy case: it is emitted alone and FLAGGED — never silently
    passed through inside a larger chunk, never truncated (cutting citable
    evidence would corrupt exactly what a citation quotes).
    """
    counter = config.token_counter or count_tokens
    words = sentence.split()
    pieces: list[tuple[str, bool]] = []
    current: list[str] = []
    for word in words:
        if _embed_tokens(header, " ".join([*current, word]), counter) <= config.max_tokens:
            current.append(word)
            continue
        if current:
            pieces.append((" ".join(current), False))
            current = []
        if _embed_tokens(header, word, counter) <= config.max_tokens:
            current = [word]
        else:
            pieces.append((word, True))
    if current:
        pieces.append((" ".join(current), False))
    return pieces or [(sentence, False)]


def _split_atomic(body: str, header: str, config: IngestConfig) -> list[tuple[str, bool]]:
    """Keep an atomic unit whole if it fits; else split at sentence bounds
    (#41 path b — never pass an oversized atomic chunk through)."""
    counter = config.token_counter or count_tokens
    if _embed_tokens(header, body, counter) <= config.max_tokens:
        return [(body, False)]
    out: list[tuple[str, bool]] = []
    current: list[str] = []
    for sentence in split_sentences(body) or [body]:
        with_next = _embed_tokens(header, " ".join([*current, sentence]), counter)
        if current and with_next > config.max_tokens:
            out.append((" ".join(current), False))
            current = []
        if not current and _embed_tokens(header, sentence, counter) > config.max_tokens:
            out.extend(_split_oversized(sentence, header, config))
        else:
            current.append(sentence)
    if current:
        out.append((" ".join(current), False))
    return out


def _split_table(
    cells: str, caption: str | None, header: str, config: IngestConfig
) -> list[tuple[str, bool]]:
    """Split an oversized table at ROW boundaries, never mid-cell (#139).

    Every piece re-carries the header row so cells keep their labels (the
    qualifier-from-claim separation §2.4 chunking exists to prevent). A
    single row that exceeds the cap even alone becomes its own piece,
    header attached, flagged ``oversized_atomic`` — the documented cap
    exception, loud rather than silently passed through or cut mid-cell.
    The caption rides with the first piece when it fits, else as its own
    trailing piece.
    """
    counter = config.token_counter or count_tokens
    rows = [line for line in cells.splitlines() if line.strip()]
    header_row, data_rows = rows[0], rows[1:]
    pieces: list[tuple[str, bool]] = []
    current: list[str] = [header_row]
    for row in data_rows:
        if _embed_tokens(header, "\n".join([*current, row]), counter) <= config.max_tokens:
            current.append(row)
            continue
        if len(current) > 1:
            pieces.append(("\n".join(current), False))
        fresh = [header_row, row]
        if _embed_tokens(header, "\n".join(fresh), counter) <= config.max_tokens:
            current = fresh
        else:
            pieces.append(("\n".join(fresh), True))
            current = [header_row]
    if len(current) > 1:
        pieces.append(("\n".join(current), False))
    if not pieces:
        # Degenerate: a header row alone that exceeds the cap.
        pieces = [(header_row, _embed_tokens(header, header_row, counter) > config.max_tokens)]
    if caption:
        first_body, first_flag = pieces[0]
        with_caption = f"{first_body}\n{caption}"
        if first_flag or _embed_tokens(header, with_caption, counter) <= config.max_tokens:
            pieces[0] = (with_caption, first_flag)
        else:
            pieces.append((caption, False))
    return pieces


def _table_bodies(block: Block, header: str, config: IngestConfig) -> list[tuple[str, bool]]:
    """Render a TABLE block to citable ``(body, oversized)`` pieces."""
    counter = config.token_counter or count_tokens
    cells = _strip_separator_rows((block.text or "").strip()).strip()
    if cells == "[TABLE]":
        cells = ""
    caption = (block.caption or "").strip() or None
    if not cells and not caption:
        # Review finding #138: nothing citable — never a placeholder chunk.
        return []
    rendered = " ".join(part for part in (cells, caption) if part)
    if _embed_tokens(header, rendered, counter) <= config.max_tokens:
        return [(rendered, False)]
    if "\n" in cells:
        return _split_table(cells, caption, header, config)
    return _split_atomic(rendered, header, config)


def _pack_sentences(
    sentences: list[str], header: str, config: IngestConfig
) -> list[tuple[str, bool]]:
    """Greedily pack prose sentences into ≤cap bodies with one-sentence
    overlap, returning ``(body, oversized_atomic)`` pairs. The cap (#41)
    is re-checked after overlap seeding; a seed that would crowd out the
    next new sentence is dropped so no sentence is lost; a single
    oversized sentence is split at word bounds (an unsplittable
    token-dense word arrives flagged from :func:`_split_oversized`)."""
    counter = config.token_counter or count_tokens
    bodies: list[tuple[str, bool]] = []
    previous_last: str | None = None
    index = 0
    total = len(sentences)
    while index < total:
        current: list[str] = []
        if (
            config.overlap_sentences
            and previous_last is not None
            and _embed_tokens(header, " ".join([previous_last, sentences[index]]), counter)
            <= config.max_tokens
        ):
            current = [previous_last]
        start = index
        while index < total and (
            _embed_tokens(header, " ".join([*current, sentences[index]]), counter)
            <= config.max_tokens
        ):
            current.append(sentences[index])
            index += 1
        if index == start:
            # The next sentence did not fit. A blocking overlap seed is
            # dropped and the sentence retried solo; if oversized alone, split.
            solo = sentences[index]
            if _embed_tokens(header, solo, counter) <= config.max_tokens:
                bodies.append((solo, False))
                previous_last = solo
            else:
                pieces = _split_oversized(solo, header, config)
                bodies.extend(pieces)
                previous_last = pieces[-1][0]
            index += 1
            continue
        bodies.append((" ".join(current), False))
        previous_last = current[-1]
    return bodies


def _pack_section(
    blocks: list[Block], header: str, config: IngestConfig
) -> list[tuple[str, tuple[str, ...], bool]]:
    """Pack one section's content blocks into ``(body, block_types,
    oversized_atomic)`` triples.

    Prose is packed with overlap; figures/tables/footnotes are atomic (each
    its own chunk; tables split at row boundaries, other atomics at
    sentence bounds, when oversized — #139). Overlap never carries across
    an atomic unit — the flushed prose run ends there and the next prose
    run starts fresh (the #41-recorded boundary rule)."""
    results: list[tuple[str, tuple[str, ...], bool]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        nonlocal prose
        if prose:
            for body, oversized in _pack_sentences(prose, header, config):
                results.append((body, ("text",), oversized))
            prose = []

    for block in blocks:
        if block.type in (BlockType.FIGURE, BlockType.TABLE, BlockType.FOOTNOTE):
            flush_prose()
            atomic_type = block.type.value
            if block.type is BlockType.TABLE:
                bodies = _table_bodies(block, header, config)
            else:
                rendered = _render_block_body(block)
                if rendered == "[FIGURE]":
                    # Review finding #138: a captionless figure carries no
                    # citable evidence — never emit a zero-content
                    # placeholder as a citable unit.
                    continue
                bodies = _split_atomic(rendered, header, config)
            for body, oversized in bodies:
                results.append((body, (atomic_type,), oversized))
        elif block.type in (BlockType.TEXT, BlockType.LIST_ITEM):
            clean, _ = strip_note_markers(normalise_text(block.text))
            for sentence in split_sentences(clean) or ([clean.strip()] if clean.strip() else []):
                prose.append(sentence)
    flush_prose()
    return results


def _walk_sections(doc: StructuredDoc) -> list[tuple[tuple[str, ...], list[Block]]]:
    """Group content blocks under their reconstructed nested section path.

    A heading always opens a fresh section, so no chunk can span a heading
    boundary — the invariant is structural here, not in the splitter."""
    sections: list[tuple[tuple[str, ...], list[Block]]] = []
    stack: list[str] = []
    current_path: tuple[str, ...] = ()
    current_blocks: list[Block] = []

    def flush() -> None:
        nonlocal current_blocks
        if current_blocks:
            sections.append((current_path, current_blocks))
            current_blocks = []

    for block in doc.blocks:
        if block.type is BlockType.HEADING:
            flush()
            depth = block.level if block.level and block.level > 0 else 1
            del stack[depth - 1 :]
            while len(stack) < depth - 1:
                stack.append("")
            stack.append(block.text)
            current_path = tuple(part for part in stack if part)
        elif block.type in _CONTENT_TYPES:
            current_blocks.append(block)
        # TITLE / CAPTION / PAGE_FURNITURE are not section content.
    flush()
    return sections


def _make_record(
    doc: StructuredDoc,
    section_path: tuple[str, ...],
    header: str,
    body: str,
    token_count: int,
    block_types: tuple[str, ...],
    consensus_position: str,
    source_type: str,
    citation_metadata: Mapping[str, Any],
    occurrence: int = 0,
    oversized_atomic: bool = False,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=_chunk_id(doc.doc_id, section_path, body, occurrence),
        doc_id=doc.doc_id,
        section_path=section_path,
        context_header=header,
        body=body,
        token_count=token_count,
        confidence_markers=extract_confidence_markers(body),
        consensus_position=consensus_position,
        source_type=source_type,
        citation_metadata=dict(citation_metadata),
        block_types=block_types,
        pages=(),
        oversized_atomic=oversized_atomic,
    )


def _citation_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": entry.get("title"),
        "licence": entry.get("licence"),
        "attribution_text": entry.get("attribution_text"),
        "canonical_url": entry.get("canonical_url"),
        "permitted_context": entry.get("permitted_context"),
    }


def _chunk_headline_statements(
    doc: StructuredDoc,
    config: IngestConfig,
    consensus_position: str,
    source_type: str,
    citation_metadata: Mapping[str, Any],
) -> list[ChunkRecord]:
    if not config.headline_statements_enabled:
        return []
    counter = config.token_counter or count_tokens
    statements = [block.text for block in doc.blocks if block.type is BlockType.TEXT]
    if len(statements) > config.headline_statements_cap:
        raise IngestError(
            f"headline-statements ingest for {doc.doc_id!r}: {len(statements)} curated "
            f"statements exceed the <= {config.headline_statements_cap}-per-SPM substantiality "
            "cap (DESIGN §2.1 Tier C); ingest refused rather than silently truncated"
        )
    heading = next(
        (b.text for b in doc.blocks if b.type in (BlockType.HEADING, BlockType.TITLE)),
        "headline statements",
    )
    section_path = (heading,)
    header = _context_header(doc.title, section_path, config)
    records: list[ChunkRecord] = []
    occurrences: Counter[str] = Counter()
    for statement in statements:
        # One statement = one chunk, carried verbatim (never normalised: the
        # curated text is the citable unit and its "S.1"-style numbering must
        # survive intact).
        token_count = _embed_tokens(header, statement, counter)
        records.append(
            _make_record(
                doc,
                section_path,
                header,
                statement,
                token_count,
                ("headline",),
                consensus_position,
                source_type,
                citation_metadata,
                occurrence=occurrences[statement],
            )
        )
        occurrences[statement] += 1
    return records


def chunk_document(
    doc: StructuredDoc,
    manifest_entry: Mapping[str, Any],
    config: IngestConfig | None = None,
) -> list[ChunkRecord]:
    """The production structure-aware chunker (DESIGN §2.4; the most
    unit-tested module in the repo, IMPLEMENTATION.md §1).

    Invariants the red tests pin:

    - section-bounded: no chunk spans a heading boundary; section paths
      are the reconstructed nested hierarchy (finding 1);
    - cap (#41 amendment; #139): every chunk's ``token_count`` — the
      configured counter over ``embedding_text``, header INCLUDED — is
      ≤ ``config.max_tokens``; overlap seeding re-checks the cap; an
      atomic unit longer than the cap is split — TABLEs at row
      boundaries with the header row re-carried per piece (never
      mid-cell), other atomics at sentence boundaries — never passed
      through oversized. The one documented exception: a single
      unsplittable unit (one table row, one token-dense word) over the
      cap alone is emitted as its own chunk with
      ``oversized_atomic=True`` — loud, never silent;
    - one-sentence overlap between adjacent chunks of one section;
      overlap is intentionally skipped after a chunk ending in an atomic
      unit (the #41-recorded rule, now explicit);
    - tiny-chunk floor (finding 8): no chunk below ``config.min_tokens``
      except deliberately atomic units;
    - headline-statements profile (manifest ``ingest_profile:
      headline-statements``): one statement = one chunk; more than
      ``config.headline_statements_cap`` statements raises
      :class:`IngestError` (cap violation fails ingest); the profile is
      only ingested at all when ``config.headline_statements_enabled``;
    - metadata: ``confidence_markers`` extracted per chunk;
      ``consensus_position`` and ``source_type`` propagate from
      ``manifest_entry`` to every chunk; ``citation_metadata`` carries
      the §2.4 schema fields.
    """
    config = config or IngestConfig()
    counter = config.token_counter or count_tokens
    consensus_position = manifest_entry.get("consensus_position") or "assessed"
    source_type = manifest_entry.get("source_type") or "evidence"
    citation_metadata = _citation_metadata(manifest_entry)

    if manifest_entry.get("ingest_profile") == "headline-statements":
        return _chunk_headline_statements(
            doc, config, consensus_position, source_type, citation_metadata
        )

    work = reclassify_furniture_headings(doc)
    work = reconstruct_section_hierarchy(work)
    work = strip_front_matter(work)
    work = associate_captions(work)
    work, _references = segregate_references(work)

    records: list[ChunkRecord] = []
    occurrences: Counter[tuple[tuple[str, ...], str]] = Counter()
    for section_path, blocks in _walk_sections(work):
        header = _context_header(doc.title, section_path, config)
        for body, block_types, oversized_atomic in _pack_section(blocks, header, config):
            token_count = _embed_tokens(header, body, counter)
            atomic = bool(block_types) and set(block_types) <= _ATOMIC_TYPES
            if not atomic and token_count < config.min_tokens and counter(body) <= 3:
                # Degenerate fragment (finding 8: a lone heading-follower
                # scrap, an axis label like "45°N") — suppress, never index.
                continue
            records.append(
                _make_record(
                    doc,
                    section_path,
                    header,
                    body,
                    token_count,
                    block_types,
                    consensus_position,
                    source_type,
                    citation_metadata,
                    occurrence=occurrences[(section_path, body)],
                    oversized_atomic=oversized_atomic,
                )
            )
            occurrences[(section_path, body)] += 1
    return records


# ---------------------------------------------------------------------------
# Launch-dependency check (DESIGN §2.3)
# ---------------------------------------------------------------------------


def _will_be_ingested(entry: Mapping[str, Any], config: IngestConfig) -> bool:
    if entry.get("ingest_profile") == "headline-statements":
        return config.headline_statements_enabled
    return True


def check_assessed_range_statements_present(
    entries: Iterable[Mapping[str, Any]],
    config: IngestConfig | None = None,
) -> None:
    """The severity-skew guardrail's corpus-side check (DESIGN §2.3).

    The corpus manifest must contain at least one document that (a)
    declares ``provides_assessed_ranges: true`` — the assessed-range
    statements for sensitivity / committed warming / warming levels, via
    the IPCC curated set or NCA5 equivalents — and (b) will actually be
    ingested under ``config`` (a headline-statements document does not
    satisfy the check while the feature flag is off). Raises
    :class:`IngestError` naming the missing dependency otherwise; the
    check pins presence of the source, never its content.
    """
    config = config or IngestConfig()
    for entry in entries:
        if entry.get("provides_assessed_ranges") and _will_be_ingested(entry, config):
            return
    raise IngestError(
        "no ingested document declares provides_assessed_ranges: the assessed-range statements "
        "(equilibrium sensitivity / committed warming / warming levels) required as a launch "
        "dependency (DESIGN §2.3) are absent — refusing to ship 'Hansen in, assessed ranges out'"
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _is_html(path: str | None) -> bool:
    return bool(path) and path.lower().endswith((".html", ".htm"))


def ingest_corpus(
    manifest_path: Path,
    corpus_dir: Path,
    config: IngestConfig | None = None,
    parser: Callable[..., StructuredDoc] | None = None,
    transport: Callable[[str], bytes] | None = None,
) -> IngestResult:
    r"""Run the whole manifest-driven pipeline (DESIGN §2.4)::

        fetch (verify sha256) -> parse -> chunk -> citation metadata
              -> custom-content citation blocks
              \-> per-document ingest records (incl. fallback warnings)

    Contract points the tests pin:

    - **Gate at entry**: the manifest loads through
      :func:`ingestion.manifest.load_corpus_manifest` FIRST; any
      invariant-violating document refuses the whole run
      (:class:`~ingestion.manifest.ManifestError`) before any fetch,
      parse or chunk happens — the injected transport/parser are never
      called on a refused run, and no partial output is produced.
    - **Fetch + verify**: each document's bytes come from its
      ``source_url`` (issue #100: the manifest ``sha256`` pins the ingest
      artefact at ``source_url``) via the injected transport (tests use
      ``file://``), verified with
      :func:`ingestion.manifest.verify_fetched_sha256`; a mismatch
      refuses the run.
    - **Assessed-range launch dependency**: when
      ``config.enforce_assessed_ranges``,
      :func:`check_assessed_range_statements_present` runs at entry.
    - **Loud fallback**: a document parsed by the PyMuPDF fallback gets a
      warning + ``needs_hand_review`` on its
      :class:`DocumentIngestRecord`; never a silent code path.
    - **Determinism**: two runs over identical inputs produce identical
      chunks (ids included) and identical block payloads — the
      idempotent re-embedding hook.
    """
    from ingestion.blocks import build_citation_blocks

    config = config or IngestConfig()
    corpus_dir = Path(corpus_dir)
    manifest_path = Path(manifest_path)

    # 1. Gate at entry: validate every document BEFORE any fetch. A single
    #    invariant violation refuses the whole run (DESIGN §2.1).
    manifest = load_corpus_manifest(manifest_path)

    # Raw entries carry the ingest-only keys (title, source_type,
    # ingest_profile, provides_assessed_ranges) the chunker consumes.
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw_entries: list[Mapping[str, Any]] = list(raw.get("documents") or [])
    raw_by_id = {entry.get("id"): entry for entry in raw_entries}

    # 2. Launch-dependency check at entry (before any fetch).
    if config.enforce_assessed_ranges:
        check_assessed_range_statements_present(raw_entries, config)

    chunks: list[ChunkRecord] = []
    documents: dict[str, DocumentIngestRecord] = {}

    for record in manifest.documents:
        entry = raw_by_id.get(record.id, {})

        # Feature-flag skip (recorded, never silent): a headline-statements
        # document withheld while the flag is off.
        if (
            entry.get("ingest_profile") == "headline-statements"
            and not config.headline_statements_enabled
        ):
            documents[record.id] = DocumentIngestRecord(
                doc_id=record.id,
                parse_backend="skipped",
                skipped=True,
                skip_reason="headline-statements feature flag off (DESIGN §2.1 Tier C)",
            )
            continue

        # Manifest-only entries (non-open contexts ship no in-repo artefact)
        # carry no fetchable source — record the skip rather than fetch null.
        if not record.source_url:
            documents[record.id] = DocumentIngestRecord(
                doc_id=record.id,
                parse_backend="skipped",
                skipped=True,
                skip_reason="manifest-only entry (no source_url to fetch)",
            )
            continue

        if transport is None:
            raise IngestError(f"{record.id}: no transport provided to fetch {record.source_url}")

        # 3. Fetch the pinned bytes, write the artefact, verify the sha256
        #    (#100: the pin is the ingest artefact at source_url).
        data = transport(record.source_url)
        artefact = corpus_dir / (record.path or f"{record.id}.bin")
        artefact.parent.mkdir(parents=True, exist_ok=True)
        artefact.write_bytes(data)
        verify_fetched_sha256(record.id, artefact, record.sha256)

        # 4. Parse: HTML direct; PDFs through the injected Docling seam
        #    (or the production parser when none is injected).
        if _is_html(record.path):
            sdoc = parse_html(data.decode("utf-8"), record.id, title=entry.get("title"))
            backend = "html"
        else:
            entry_title = entry.get("title")
            parse = parser or (
                lambda p, d, _title=entry_title, **_: parse_document(p, d, title=_title)
            )
            sdoc = parse(artefact, record.id)
            backend = sdoc.backend

        degraded = backend == "pymupdf"
        warnings: tuple[str, ...] = ()
        if degraded:
            warnings = (
                f"{record.id}: parsed with the PyMuPDF fallback — degraded structure "
                "(no reliable headings/tables/captions); flagged for hand review before "
                "indexing (DESIGN §2.4, amended)",
            )

        documents[record.id] = DocumentIngestRecord(
            doc_id=record.id,
            parse_backend=backend,
            degraded_fallback=degraded,
            needs_hand_review=degraded,
            warnings=warnings,
        )

        # 5. Chunk + citation metadata (per-chunk §2.4 schema).
        chunks.extend(chunk_document(sdoc, entry, config))

    # 6. Emit one custom-content citation block per chunk.
    blocks = build_citation_blocks(chunks)
    return IngestResult(chunks=tuple(chunks), blocks=tuple(blocks), documents=documents)
