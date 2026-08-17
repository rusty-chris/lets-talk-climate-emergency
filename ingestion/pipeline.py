"""Production ingestion pipeline (issue #7): manifest-driven fetch → verify
→ parse → chunk → citation metadata → custom-content citation blocks.

RED phase: every function below is a contract stub raising
:class:`NotImplementedError`; the failing tests under
``tests/unit/test_ingestion_*.py`` and
``tests/integration/test_ingest_pipeline.py`` define the behaviour
(IMPLEMENTATION.md §2). The requirements come from:

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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.parse import Block, StructuredDoc

__all__ = [
    "IngestConfig",
    "IngestError",
    "ChunkRecord",
    "DocumentIngestRecord",
    "IngestResult",
    "parse_html",
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
      DESIGN §2.4 "embedding idempotent/incremental").
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
# Parsing (HTML direct; PDFs arrive through the injected parser seam)
# ---------------------------------------------------------------------------


def parse_html(html: str, doc_id: str, title: str | None = None) -> StructuredDoc:
    """Parse HTML directly into a StructuredDoc (DESIGN §2.4: HTML direct).

    Contract: ``h1``–``h6`` become TITLE/HEADING blocks with their true
    nesting level; body prose becomes TEXT blocks under the enclosing
    heading; ``<table>`` cell content is retained citable in the TABLE
    block's text (spike finding 3c — never a bare placeholder);
    ``<figcaption>``/caption markup attaches to its figure; page furniture
    (nav, header, footer, script, style) is excluded.
    """
    raise NotImplementedError("issue #7: production HTML parser not implemented yet")


def reconstruct_section_hierarchy(doc: StructuredDoc) -> StructuredDoc:
    """Recover heading nesting that the PDF parser flattened (finding 1).

    Docling returns every heading at level 1; the hierarchy lives only in
    the heading text (numeric prefixes such as ``2`` / ``2.2`` / ``2.2.1``,
    or Key-Message structure). Contract: after reconstruction, a block
    under a ``2.2.1``-style heading resolves to the nested section path
    ``('2 …', '2.2 …', '2.2.1 …')``, not a single flat element.
    """
    raise NotImplementedError("issue #7: heading-nesting reconstruction not implemented yet")


def strip_front_matter(doc: StructuredDoc) -> StructuredDoc:
    """Drop front-matter/boilerplate noise before chunking (finding 2).

    Author/affiliation lists, role lines ("Chapter Lead Author", "Cover
    Art"), recommended-citation blocks and publisher boilerplate must not
    reach the chunk output — neither as sections nor as body text.
    """
    raise NotImplementedError("issue #7: front-matter stripper not implemented yet")


def associate_captions(doc: StructuredDoc) -> StructuredDoc:
    """Pair each figure/table with its caption, de-duplicated (finding 3).

    A caption emitted as a separate adjacent block attaches to its
    figure/table (parent/caption refs or adjacency); caption text appears
    exactly once in the output (no attached-plus-standalone duplicates).
    """
    raise NotImplementedError("issue #7: caption association not implemented yet")


def segregate_references(doc: StructuredDoc) -> tuple[StructuredDoc, tuple[Block, ...]]:
    """Split the reference/bibliography section out of the evidence stream
    (finding 5).

    Returns ``(evidence_doc, reference_blocks)``: the References/
    Bibliography section is never emitted as evidence chunks (it was
    20–30% of all spike chunks and dilutes retrieval); its blocks are
    returned separately for optional structured storage.
    """
    raise NotImplementedError("issue #7: reference-list segregation not implemented yet")


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
    raise NotImplementedError("issue #7: text normalisation not implemented yet")


def strip_note_markers(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove superscript note/citation markers flattened into prose
    (finding 4), returning ``(clean_text, markers)``.

    NCA5-style extraction glues superscript reference numbers to sentence
    ends ("…considerably more.67 In just three decades…"): the stray
    numeral is stripped from the text and captured in ``markers``.
    Genuine numbers — decimals ("1.9 °C") and in-sentence values
    ("rose by 67 mm") — are never touched.
    """
    raise NotImplementedError("issue #7: note-marker stripping not implemented yet")


def split_sentences(text: str) -> list[str]:
    """Real sentence segmentation (finding 7 — not the spike regex).

    Contract: abbreviations ("e.g.", "Fig. 1", "et al."), decimals and
    parenthesised citations do not produce false sentence breaks; ordinary
    ``.``/``!``/``?`` boundaries do.
    """
    raise NotImplementedError("issue #7: production sentence tokenizer not implemented yet")


def count_tokens(text: str) -> int:
    """Production token counter — the embedding model's tokenizer
    (finding 7), used when :attr:`IngestConfig.token_counter` is None.

    Unit tests never call this (model weights live outside the unit
    tier); they inject a deterministic counter instead.
    """
    raise NotImplementedError("issue #7: embedding-tokenizer token counting not implemented yet")


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
    raise NotImplementedError("issue #7: confidence-marker extraction not implemented yet")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


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
    - cap (#41 amendment): every chunk's ``token_count`` — the configured
      counter over ``embedding_text``, header INCLUDED — is
      ≤ ``config.max_tokens``; overlap seeding re-checks the cap; an
      atomic unit longer than the cap is split at sentence boundaries,
      never passed through oversized;
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
    raise NotImplementedError("issue #7: production chunker not implemented yet")


# ---------------------------------------------------------------------------
# Launch-dependency check (DESIGN §2.3)
# ---------------------------------------------------------------------------


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
    raise NotImplementedError("issue #7: assessed-range launch-dependency check not implemented")


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def ingest_corpus(
    manifest_path: Path,
    corpus_dir: Path,
    config: IngestConfig | None = None,
    parser: Callable[..., StructuredDoc] | None = None,
    transport: Callable[[str], bytes] | None = None,
) -> IngestResult:
    """Run the whole manifest-driven pipeline (DESIGN §2.4)::

        fetch (verify sha256) -> parse -> chunk -> citation metadata
              -> custom-content citation blocks
              \\-> per-document ingest records (incl. fallback warnings)

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
    raise NotImplementedError("issue #7: ingest_corpus not implemented yet")
