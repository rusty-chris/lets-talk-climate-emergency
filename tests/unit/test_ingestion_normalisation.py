"""The eight spike-2 parsing failure modes as named production
requirements (issue #7; reviews/spike-02-parsing-findings.md findings
1–8) — RED. Unit tier, synthetic inputs only.

Finding 8 (tiny-chunk floor) is pinned in test_ingestion_chunker.py; the
HTML-direct arm of finding 3c (table cells) in test_ingestion_html.py.
"""

from __future__ import annotations

from ingestion.parse import Block, BlockType
from ingestion.pipeline import (
    chunk_document,
    normalise_text,
    split_sentences,
    strip_note_markers,
)
from tests._ingestion_fixtures import (
    caption,
    config,
    doc,
    figure,
    heading,
    manifest_entry,
    sentence,
    text,
)

# --------------------------------------------------------------------------
# Finding 1 — heading nesting reconstruction
# --------------------------------------------------------------------------


def test_heading_nesting_reconstructed_from_numeric_prefixes():
    """Docling flattens every heading to level 1; the hierarchy survives
    only in the numeric prefixes. Production must reconstruct the nested
    section path ('2 …', '2.2 …', '2.2.1 …') — not a flat single label."""
    flat = doc(
        "syn-nesting",
        [
            heading("1 Introduction", level=1),
            text(sentence(12, "intro")),
            heading("2 Results", level=1),
            text(sentence(12, "results")),
            heading("2.2 Interactions", level=1),
            text(sentence(12, "interactions")),
            heading("2.2.1 Deep cascades", level=1),
            text(sentence(12, "cascades")),
        ],
    )
    chunks = chunk_document(flat, manifest_entry(), config())
    deep = [c for c in chunks if "cascades" in c.body]
    assert deep, "the 2.2.1 content must be chunked"
    for chunk in deep:
        assert chunk.section_path == (
            "2 Results",
            "2.2 Interactions",
            "2.2.1 Deep cascades",
        ), f"flat section path not reconstructed: {chunk.section_path}"


# --------------------------------------------------------------------------
# Finding 2 — front-matter / boilerplate exclusion
# --------------------------------------------------------------------------


def test_front_matter_noise_excluded_from_chunks():
    """Author roles, affiliation walls and recommended-citation blocks
    (31 spurious 'headings' on one NCA5 page) never reach chunk output —
    neither as section paths nor as body text."""
    noisy = doc(
        "syn-frontmatter",
        [
            text("1 Meridian Institute of Imaginary Studies, Aurelia (invented affiliation)"),
            heading("Chapter Lead Author"),
            text("Imara Quell, Meridian Institute (invented)"),
            heading("Recommended Citation"),
            text("Quell, I. (invented citation string for front-matter tests)."),
            heading("1 Introduction"),
            text(sentence(20, "realbody")),
        ],
    )
    chunks = chunk_document(noisy, manifest_entry(), config())
    assert any("realbody" in c.body for c in chunks), "real content must survive"
    for chunk in chunks:
        for noise in (
            "Chapter Lead Author",
            "Recommended Citation",
            "Imara Quell",
            "Institute of Imaginary Studies",
        ):
            assert noise not in chunk.body and noise not in " ".join(chunk.section_path), (
                f"front-matter noise {noise!r} leaked into {chunk.chunk_id}"
            )


# --------------------------------------------------------------------------
# Finding 3 — caption association + de-duplication
# --------------------------------------------------------------------------


def test_caption_associated_with_adjacent_figure():
    """A caption emitted as a separate adjacent block attaches to its
    figure: the placeholder chunk carries the caption text as citable
    text (DESIGN §2.4), never '[FIGURE — no caption extracted]'."""
    caption_text = "Figure 2. Invented changes between the two baseline periods."
    with_figure = doc(
        "syn-caption",
        [
            heading("1 Figures"),
            text(sentence(12, "prefig")),
            figure(caption=None),
            caption(caption_text),
        ],
    )
    chunks = chunk_document(with_figure, manifest_entry(), config())
    figure_chunks = [c for c in chunks if "[FIGURE" in c.body]
    assert figure_chunks, "the figure placeholder must be retained citable"
    assert any(caption_text in c.body for c in figure_chunks), (
        "the adjacent caption must be paired with its figure in the same chunk"
    )


def test_caption_text_never_duplicated():
    """Spike finding 3: some captions arrive both attached and as a
    standalone text block — the output must carry the caption exactly
    once."""
    caption_text = "Figure 3. Invented lowland trend, duplicated by the parser."
    duplicated = doc(
        "syn-caption-dupe",
        [
            heading("1 Figures"),
            figure(caption=caption_text),
            text(caption_text),
            text(sentence(12, "postfig")),
        ],
    )
    chunks = chunk_document(duplicated, manifest_entry(), config())
    occurrences = sum(c.body.count(caption_text) for c in chunks)
    assert occurrences == 1, f"caption appears {occurrences} times; must be de-duplicated to 1"


def test_table_cell_content_retained_citable():
    """Finding 3c: table *data* must stay citable — a parsed table block
    carrying cell content contributes its numbers to the chunk text, not
    a bare placeholder."""
    table_block = Block(
        BlockType.TABLE,
        "| Indicator | 2024 value |\n| Basin-mean temperature | 13.1 °C |",
        caption="Table 1. Invented indicator values.",
    )
    with_table = doc(
        "syn-table",
        [heading("1 Indicators"), text(sentence(12, "pretable")), table_block],
    )
    chunks = chunk_document(with_table, manifest_entry(), config())
    assert any("13.1" in c.body for c in chunks), (
        "table cell values must survive into chunk text (a data table with only a "
        "placeholder contributes no citable numbers)"
    )


# --------------------------------------------------------------------------
# Finding 4 — superscript note/citation markers
# --------------------------------------------------------------------------


def test_footnote_markers_stripped_from_prose():
    """NCA5-style glued superscript markers ('…more.67 In just…') are
    stripped from the text and captured as markers."""
    glued = (
        "Sea level along the invented coastline rose by about 11 units, which is "
        "considerably more than the basin average.67 In just the last three decades "
        "the invented rate doubled.68"
    )
    clean, markers = strip_note_markers(glued)
    assert markers == ("67", "68")
    assert ".67" not in clean and ".68" not in clean
    assert "67" not in clean and "68" not in clean
    assert "considerably more than the basin average." in clean
    assert "the invented rate doubled." in clean


def test_genuine_numbers_survive_marker_stripping():
    """Decimals and in-sentence values are never mistaken for markers."""
    prose = "Warming of 1.9 °C is recorded, and the shelf rose by 67 mm in 2024."
    clean, markers = strip_note_markers(prose)
    assert markers == ()
    assert clean == prose


# --------------------------------------------------------------------------
# Finding 5 — reference-list segregation
# --------------------------------------------------------------------------


def test_reference_list_segregated_from_evidence_index():
    """Bibliographies were 20–30% of spike chunks and dilute retrieval:
    the References section is never emitted as evidence chunks."""
    reference_string = (
        "Quenneville, R. and Ash, T. (invented), Fingerprinting regional drying, "
        "Journal of Imaginary Climatology, 12, 34-56."
    )
    with_references = doc(
        "syn-references",
        [
            heading("1 Synthesis"),
            text(sentence(20, "synthesis")),
            heading("References"),
            text(reference_string),
            text("Pellier, A. and Vance, B. (invented), Committed warming revisited, ibid."),
        ],
    )
    chunks = chunk_document(with_references, manifest_entry(), config())
    assert any("synthesis" in c.body for c in chunks)
    for chunk in chunks:
        assert "References" not in chunk.section_path, (
            f"{chunk.chunk_id}: References section emitted as evidence"
        )
        assert reference_string not in chunk.body, (
            f"{chunk.chunk_id}: bibliography text emitted as evidence"
        )


# --------------------------------------------------------------------------
# Finding 6 — hyphenation / ligature normalisation
# --------------------------------------------------------------------------


def test_linebreak_hyphenation_rejoined():
    assert (
        normalise_text("The tempera-\nture record of the invented basin is complete.")
        == "The temperature record of the invented basin is complete."
    )


def test_pdf_ligatures_normalised_to_plain_text():
    assert normalise_text("A ﬁne ﬂow of invented data") == "A fine flow of invented data"


def test_normalisation_preserves_real_compounds_and_decimals():
    prose = "Well-mixed gases warmed the basin by 1.9 °C over the invented record."
    assert normalise_text(prose) == prose


# --------------------------------------------------------------------------
# Finding 7 — real sentence segmentation
# --------------------------------------------------------------------------


def test_sentence_splitter_does_not_break_on_abbreviations():
    prose = (
        "The decline (see Fig. 1), e.g. across the southern terraces, continued unabated. "
        "A second invented study agreed."
    )
    sentences = split_sentences(prose)
    assert len(sentences) == 2, f"abbreviations produced false breaks: {sentences}"
    assert "Fig. 1" in sentences[0]
    assert sentences[1] == "A second invented study agreed."


def test_sentence_splitter_does_not_break_on_decimals():
    prose = "Values fell by 0.5 % in the invented record. Later they rose again."
    sentences = split_sentences(prose)
    assert len(sentences) == 2, f"decimals produced false breaks: {sentences}"
    assert "0.5 %" in sentences[0]
