# SYNTHETIC FIXTURE — authored for this project's tests
"""Synthetic StructuredDoc for the issue #2 chunker characterisation tests.

No real Tier B/C source text (DESIGN §2.1 / IMPLEMENTATION.md §5): every string
below is invented for this repo. The document is structurally realistic —
nested headings (H1 → H2), a table and a figure with captions, a footnote, a
calibrated-language sentence, and a deliberately long section that forces the
<=~500-token splitter to fire — so the committed characterisation tests exercise
the real boundary rules, not a toy.

The first line of this file is the mandated synthetic-fixture marker; a meta-test
asserts its presence.
"""

from __future__ import annotations

from ingestion.parse import Block, BlockType, StructuredDoc

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE — authored for this project's tests"

# A long lorem-style paragraph (invented) used to force a multi-chunk split.
_LONG = " ".join(
    f"Sentence number {i} describes an invented observation about the model region {i}."
    for i in range(1, 61)
)


def build_synthetic_doc() -> StructuredDoc:
    """Return the synthetic StructuredDoc exercised by the characterisation tests."""
    blocks = [
        Block(BlockType.TITLE, "Synthetic Climate Report", level=0, page=1),
        Block(BlockType.PAGE_FURNITURE, "Synthetic Report — page 1", page=1),
        # --- Section 1 (H1) ---
        Block(BlockType.HEADING, "Observed Trends", level=1, page=1),
        Block(
            BlockType.TEXT,
            "Global mean temperature has risen over the invented record. "
            "This warming is very likely driven by the fictional forcing agent.",
            page=1,
        ),
        # --- Subsection 1.1 (H2) ---
        Block(BlockType.HEADING, "Regional Detail", level=2, page=2),
        Block(
            BlockType.TEXT,
            "The northern synthetic region warms faster than the southern one.",
            page=2,
        ),
        Block(
            BlockType.TABLE,
            "[TABLE]",
            caption="Table 1. Invented decadal temperature anomalies by synthetic region.",
            page=2,
        ),
        Block(
            BlockType.FIGURE,
            "[FIGURE]",
            caption="Figure 1. Fictional time series of the regional warming index.",
            page=2,
        ),
        # --- Section 2 (H1) — long, forces a split, and has a footnote ---
        Block(BlockType.HEADING, "Projections", level=1, page=3),
        Block(BlockType.TEXT, _LONG, page=3),
        Block(
            BlockType.FOOTNOTE,
            "1. All projections here are fabricated for testing and cite nothing real.",
            page=3,
        ),
    ]
    return StructuredDoc(
        doc_id="synthetic",
        title="Synthetic Climate Report",
        blocks=blocks,
        backend="synthetic",
    )


# The document outline the chunker's section_paths must reproduce, in order.
EXPECTED_OUTLINE: list[tuple[str, ...]] = [
    ("Observed Trends",),
    ("Observed Trends", "Regional Detail"),
    ("Projections",),
]
