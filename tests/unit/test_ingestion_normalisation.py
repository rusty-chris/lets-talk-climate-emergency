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
            text(sentence(25, "intro")),
            heading("2 Results", level=1),
            text(sentence(25, "results")),
            heading("2.2 Interactions", level=1),
            text(sentence(25, "interactions")),
            heading("2.2.1 Deep cascades", level=1),
            text(sentence(25, "cascades")),
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


def test_key_message_hierarchy_reconstructed():
    """Review finding #151: finding 1's NCA5 arm — 'Key Message N.M'
    headings are depth-1 parents and the prose-headline sub-headings
    that follow are their children, until the next Key Message. All 113
    real NCA5 chunks previously carried flat single-element paths. A
    References heading is never swallowed as a Key-Message child."""
    reference_string = "Quell, I. (invented), Basin trends, Journal of Imaginary Climatology."
    nca_shaped = doc(
        "syn-key-messages",
        [
            Block(BlockType.TITLE, "Synthetic Assessment Chapter 2", level=0),
            heading("Key Message 2.1: The Basin Is Warming", level=1),
            text(sentence(25, "kmoneintro")),
            heading("The Basin Warmed Faster Than The Shelf", level=1),
            text(sentence(25, "kmonesub")),
            heading("Key Message 2.2: Water Cycles Are Shifting", level=1),
            heading("Rainfall Moved South", level=1),
            text(sentence(25, "kmtwosub")),
            heading("References", level=1),
            text(reference_string),
        ],
        title="Synthetic Assessment Chapter 2",
    )
    chunks = chunk_document(nca_shaped, manifest_entry(), config())
    kmone_sub = [c for c in chunks if "kmonesub" in c.body]
    assert kmone_sub, "the first Key Message's sub-heading content must chunk"
    for chunk in kmone_sub:
        assert chunk.section_path == (
            "Key Message 2.1: The Basin Is Warming",
            "The Basin Warmed Faster Than The Shelf",
        ), f"Key-Message hierarchy not reconstructed: {chunk.section_path}"
    kmone_intro = [c for c in chunks if "kmoneintro" in c.body]
    assert kmone_intro and all(
        c.section_path == ("Key Message 2.1: The Basin Is Warming",) for c in kmone_intro
    ), "prose directly under a Key Message files under the Key Message itself"
    kmtwo_sub = [c for c in chunks if "kmtwosub" in c.body]
    assert kmtwo_sub and all(
        c.section_path == ("Key Message 2.2: Water Cycles Are Shifting", "Rainfall Moved South")
        for c in kmtwo_sub
    ), "the second Key Message must start a fresh parent"
    assert not any(reference_string in c.body for c in chunks), (
        "References must stay segregated, never swallowed as a Key-Message child"
    )


def test_html_markup_nesting_survives_chunking_for_numeric_headings():
    """Review finding #148: reconstruct_section_hierarchy let a numeric
    prefix OVERRIDE trusted HTML markup levels — an h3 titled '2024 was
    the warmest year on record' was forced to top level, discarding its
    real parent. Composed through chunk_document, markup nesting must
    survive."""
    from ingestion.pipeline import parse_html

    prose_a = sentence(25, "warmyear")
    prose_b = sentence(25, "factsa")
    html = (
        "<html><body><main>"
        "<h1>Invented Year In Review</h1>"
        "<h2>Evidence</h2>"
        f"<h3>2024 was the warmest year on record</h3><p>{prose_a}</p>"
        f"<h3>10 facts about warming</h3><p>{prose_b}</p>"
        "</main></body></html>"
    )
    parsed = parse_html(html, "syn-html-numeric")
    chunks = chunk_document(parsed, manifest_entry(), config())
    warmest = [c for c in chunks if "warmyear" in c.body]
    facts = [c for c in chunks if "factsa" in c.body]
    assert warmest and facts, "both h3 sections must chunk"
    for chunk in warmest:
        assert chunk.section_path == ("Evidence", "2024 was the warmest year on record"), (
            f"markup nesting destroyed by numeric-prefix override: {chunk.section_path}"
        )
    for chunk in facts:
        assert chunk.section_path == ("Evidence", "10 facts about warming"), chunk.section_path


def test_numeric_prefix_reconstruction_only_applies_to_flattened_parses():
    """Review finding #148, second arm: a heading arriving with a trusted
    non-flattened level (e.g. the PyMuPDF font heuristic's level 2) keeps
    it — '2024 Update' under '3 Results' stays nested, because a bare
    leading integer with no dotted-prefix or sibling-sequence signal is
    prose, not a section number."""
    flat = doc(
        "syn-trusted-levels",
        [
            heading("3 Results", level=1),
            text(sentence(25, "results")),
            heading("2024 Update", level=2),
            text(sentence(25, "update")),
        ],
    )
    chunks = chunk_document(flat, manifest_entry(), config())
    update = [c for c in chunks if "update" in c.body]
    assert update, "the level-2 section must chunk"
    for chunk in update:
        assert chunk.section_path == ("3 Results", "2024 Update"), (
            f"trusted level discarded / bare year treated as a section number: {chunk.section_path}"
        )


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


def test_bare_authors_and_toc_headings_stripped():
    """Review finding #140: on the real NCA5 chapter a bare 'Authors'
    section (109 tokens of author/affiliation lines) and SEVEN 'Table of
    Contents' chunks (~3,300 tokens of dot-leader lines — a retrieval
    magnet for exactly the headline phrases) reached the evidence index.
    The bare labels must strip with their sections."""
    noisy = doc(
        "syn-bare-labels",
        [
            heading("Authors"),
            text("Imara Quell, Meridian Institute of Imaginary Studies, Aurelia (invented)"),
            heading("Table of Contents"),
            text("Why the basin matters...................5 | | Invented Emissions Long Ago"),
            heading("Contents"),
            text("Overview.............2 Findings.............7 (invented dot-leader run)"),
            heading("1 Introduction"),
            text(sentence(20, "realintro")),
        ],
    )
    chunks = chunk_document(noisy, manifest_entry(), config())
    assert any("realintro" in c.body for c in chunks), "real content must survive"
    for chunk in chunks:
        path = " ".join(chunk.section_path)
        for label in ("Authors", "Table of Contents", "Contents"):
            assert label not in path, f"front-matter section {label!r} chunked: {chunk.chunk_id}"
        for noise in ("Imara Quell", "...................5", "dot-leader run"):
            assert noise not in chunk.body, (
                f"front-matter body text leaked into {chunk.chunk_id}: {noise!r}"
            )


def test_affiliation_wall_after_title_stripped():
    """Review finding #140: Docling emits the paper's TITLE first, so the
    ESD-style author/affiliation wall arrives AFTER the first head and
    the old before-first-head rule kept it (~950 tokens of citable
    'evidence' on the real review). Affiliation-shaped text between the
    TITLE and the first genuine heading must drop; abstract-like prose in
    the same span must survive."""
    abstract = (
        "Abstract. Interactions between invented tipping elements may either "
        "stabilise or destabilise the wider basin system. We review the "
        "invented literature and identify knowledge gaps across scales."
    )
    walled = doc(
        "syn-affiliation-wall",
        [
            Block(BlockType.TITLE, "Invented tipping interactions: a review", level=0),
            text("Nira Vollan 1,2,3 ; * , Anso der Velt 4,5 ; * , Tovin Marsh 6 , Imara Quell 2"),
            text(
                "1 Meridian Institute of Imaginary Studies, Aurelia. 2 Laboratoire des "
                "Etudes Imaginaires (LEI), Aurelia. 3 University of the Basin, Department "
                "of Fictional Climatology. 4 Institute for Invented Dynamics, Aurelia."
            ),
            text(abstract),
            heading("1 Introduction"),
            text(sentence(20, "realbody")),
        ],
    )
    chunks = chunk_document(walled, manifest_entry(), config())
    assert any("realbody" in c.body for c in chunks), "real content must survive"
    assert any("stabilise or destabilise" in c.body for c in chunks), (
        "abstract-like prose between the title and the first heading must be KEPT"
    )
    for chunk in chunks:
        for noise in (
            "Nira Vollan",
            "Meridian Institute",
            "Laboratoire",
            "University of the Basin",
        ):
            assert noise not in chunk.body, (
                f"affiliation wall leaked into {chunk.chunk_id}: {noise!r}"
            )


def test_front_matter_match_requires_label_heading():
    """Review finding #149: the label regex was applied with search()
    over the whole heading, so a genuine numbered section like
    '2 Feedbacks and reviewers of the draft assessment' was silently
    DELETED from the evidence index. A label must be the heading (bare
    label, or a short qualifier prefix like 'Chapter Lead Author');
    an explicitly numbered heading is never front matter."""
    mixed = doc(
        "syn-anchored-labels",
        [
            heading("2 Feedbacks and reviewers of the draft assessment"),
            text(sentence(25, "feedbacksection")),
            heading("Community reviewers and expert elicitation"),
            text(sentence(25, "communitysection")),
            heading("Reviewers"),
            text("Imara Quell, Meridian Institute (invented reviewer line)."),
            heading("3 Conclusions"),
            text(sentence(25, "conclusions")),
        ],
    )
    chunks = chunk_document(mixed, manifest_entry(), config())
    assert any("feedbacksection" in c.body for c in chunks), (
        "a genuine numbered section containing a label word was silently deleted (#149)"
    )
    assert any("communitysection" in c.body for c in chunks), (
        "a heading merely containing 'reviewers' mid-phrase must not strip its section"
    )
    assert not any("Imara Quell" in c.body for c in chunks), (
        "a bare 'Reviewers' heading must still strip with its section"
    )


def test_front_matter_drops_are_recorded():
    """Review finding #149: each stripped front-matter section must be
    auditable — chunk_document records every dropped heading into the
    caller's warnings sink (ingest_corpus persists them onto the
    document's ingest record), so a hand audit is not the only way to
    notice deleted evidence."""
    sink: list[str] = []
    noisy = doc(
        "syn-recorded-drops",
        [
            heading("Reviewers"),
            text("Imara Quell, Meridian Institute (invented reviewer line)."),
            heading("Table of Contents"),
            text("Overview.............2 (invented dot-leader line)"),
            heading("1 Introduction"),
            text(sentence(25, "realintro")),
        ],
    )
    chunks = chunk_document(noisy, manifest_entry(), config(), warnings_sink=sink)
    assert any("realintro" in c.body for c in chunks)
    assert any("Reviewers" in w for w in sink), f"dropped heading not recorded: {sink}"
    assert any("Table of Contents" in w for w in sink), f"dropped heading not recorded: {sink}"


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


def test_caption_preceding_its_figure_attaches():
    """Review finding #138: 'caption sits in the very next block' is not
    the general case on real Docling output (10 of 16 NCA5 figure chunks
    came out captionless). A CAPTION block immediately BEFORE its figure
    must pair with it — and appear exactly once."""
    caption_text = "Figure 5. Invented caption emitted before its figure object."
    with_leading_caption = doc(
        "syn-caption-before",
        [
            heading("1 Figures"),
            text(sentence(12, "prefig")),
            caption(caption_text),
            figure(caption=None),
        ],
    )
    chunks = chunk_document(with_leading_caption, manifest_entry(), config())
    figure_chunks = [c for c in chunks if "[FIGURE" in c.body]
    assert figure_chunks, "the captioned figure must be retained citable"
    assert any(caption_text in c.body for c in figure_chunks), (
        "a caption emitted immediately before its figure must pair with it"
    )
    occurrences = sum(c.body.count(caption_text) for c in chunks)
    assert occurrences == 1, f"caption appears {occurrences} times; must appear exactly once"


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


def test_space_separated_note_markers_stripped():
    """Review finding #150: on the real NCA5 chapter Docling emits the
    superscript markers SPACE-separated ('…at the national scale. 262
    However, uncertainties…') and they survived into citable bodies. The
    spaced form — sentence punctuation, whitespace, 1-3 digits, then a
    capitalised sentence start — is stripped and captured."""
    prose = (
        "Invented emissions can be traced into removal estimates at the national "
        "scale. 262  However, uncertainties in the invented inventories remain large."
    )
    clean, markers = strip_note_markers(prose)
    assert "262" in markers, f"space-separated marker not captured: {markers}"
    assert "262" not in clean, f"stray numeral left in citable prose: {clean!r}"
    assert "However, uncertainties in the invented inventories" in clean


def test_marker_after_closing_parenthesis_stripped():
    """Review finding #150: the after-parenthesis form ('…(Chs. 14,
    15). 23 These…') passed through because the pattern required a
    LETTER before the punctuation. The counter-case pins the chosen
    disambiguation: a genuine sentence-initial number followed by
    lowercase prose ('…in 2023. 24 stations reported…') is never
    stripped."""
    prose = (
        "Invented smoke degraded air quality across the western basin "
        "(Chs. 14, 15). 23 These extreme events occur more often now."
    )
    clean, markers = strip_note_markers(prose)
    assert "23" in markers, f"after-parenthesis marker not captured: {markers}"
    assert " 23 " not in clean and not clean.rstrip().endswith("23"), clean
    assert "These extreme events occur" in clean

    counter_case = "The invented network expanded further. 24 stations reported record warmth."
    unchanged, none_markers = strip_note_markers(counter_case)
    assert none_markers == ()
    assert unchanged == counter_case, (
        "a genuine sentence-initial number before lowercase prose must survive"
    )


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


def test_repeated_running_head_heading_does_not_terminate_references():
    """Review finding #141: on the real NCA5 chapter Docling emits the
    running head 'Fifth National Climate Assessment' as a heading INSIDE
    the reference list, flipping segregation off — ~11 chunks of pure
    bibliography (DOIs included) indexed as citable evidence under a
    section that does not exist. A heading equal to the document title /
    recurring page furniture must not end the References section."""
    running_head = "Fifth Synthetic Basin Assessment"
    reference_string = (
        "Quenneville, R. and Ash, T. (invented), Fingerprinting regional drying, "
        "Journal of Imaginary Climatology, 12, 34-56. https://doi.invalid/10.9999/syn.1"
    )
    with_furniture = doc(
        "syn-running-head-refs",
        [
            heading(running_head),
            heading("1 Synthesis"),
            text(sentence(20, "synthesis")),
            heading(running_head),
            heading("References"),
            text(reference_string),
            heading(running_head),
            text("Pellier, A. and Vance, B. (invented), Committed warming revisited, ibid."),
        ],
        title=running_head,
    )
    chunks = chunk_document(with_furniture, manifest_entry(), config())
    assert any("synthesis" in c.body for c in chunks), "real content must survive"
    for chunk in chunks:
        assert "Committed warming revisited" not in chunk.body, (
            f"{chunk.chunk_id}: bibliography after the running-head heading leaked as evidence"
        )
        assert reference_string not in chunk.body
        assert chunk.section_path != (running_head,), (
            f"{chunk.chunk_id}: chunk filed under the running-head pseudo-section"
        )


def test_running_head_heading_does_not_open_a_section():
    """Review finding #141: a recurring page-furniture heading must not
    open a new 'section' — prose after a page break keeps the preceding
    real section path (6 occurrences corrupted paths through the real
    NCA5 chunk stream)."""
    furniture = "Synthetic Assessment Running Head"
    corrupted = doc(
        "syn-running-head-paths",
        [
            heading("1 Observed changes"),
            text(sentence(20, "beforebreak")),
            heading(furniture),
            text(sentence(20, "afterbreak")),
            heading(furniture),
            text(sentence(20, "later")),
            heading(furniture),
            text(sentence(20, "lastrun")),
        ],
        title="Synthetic Basin Report",
    )
    chunks = chunk_document(corrupted, manifest_entry(), config())
    after = [c for c in chunks if "afterbreak" in c.body]
    assert after, "prose after the furniture heading must still chunk"
    for chunk in after:
        assert chunk.section_path == ("1 Observed changes",), (
            f"{chunk.chunk_id}: filed under {chunk.section_path} instead of the real section"
        )
    assert not any(furniture in " ".join(c.section_path) for c in chunks), (
        "the recurring running head must never appear in any section path"
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
