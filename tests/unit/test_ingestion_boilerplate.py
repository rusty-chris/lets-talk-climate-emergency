"""HTML boilerplate filtering before indexing (issue #331) — RED.

The corpus-expansion packet (corpus/EXPANSION-SIGNOFF.md) found the first
HTML sources chunk cleanly but emit nav/footer boilerplate as ordinary
chunks — breadcrumbs, menus, "Follow NASA", related-content teasers, the
climate.gov archive banner — roughly a third to half of chunks per page.
This suite pins :func:`ingestion.boilerplate.filter_html_boilerplate`, a
pure classification step over parsed blocks (post-parse, pre-chunk):

- the packet's boilerplate shapes are dropped, each drop recorded in the
  audit with a reason from the closed vocabulary;
- substantive content survives byte-identical (drop-only — kept blocks
  are an in-order subsequence of the input, never rewritten);
- fail-open honesty: ambiguous blocks are KEPT (marker phrases inside
  prose, substantive list items, qualified headings, short factual
  paragraphs) — the hand audit catches residue, a lost chunk does not;
- HTML-source scoping: any non-``html`` backend passes through
  unchanged with an empty audit (PDFs bypass the filter entirely);
- determinism: identical input → identical output and audit.

Fixture: tests/fixtures/ingestion/synthetic_boilerplate_page.html —
invented Aurelian-Basin content around the boilerplate shapes, carried in
plain div/ul markup (NOT semantic nav/footer, which parse_html already
drops). Two verbatim real marker strings are deliberate, per the packet:
the "Follow NASA" social label and the NOAA climate.gov archive-banner
sentence (a US-government site notice) quoted in EXPANSION-SIGNOFF.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.boilerplate import (
    BOILERPLATE_REASONS,
    BoilerplateAudit,
    filter_html_boilerplate,
)
from ingestion.parse import Block, BlockType, StructuredDoc
from ingestion.pipeline import parse_html

FIXTURES_INGESTION = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
BOILERPLATE_PAGE = FIXTURES_INGESTION / "synthetic_boilerplate_page.html"

#: The climate.gov archive banner, verbatim as the sign-off packet quotes
#: it (whitespace-normalised to one line, as parse_html emits blocks).
ARCHIVE_BANNER = (
    "This website is an ARCHIVED version of NOAA Climate.gov as of June 25, "
    "2025. Content is not being updated or maintained, and some links may no "
    "longer work."
)

#: Whole-block boilerplate texts that must be dropped, keyed by the pinned
#: reason code.
EXPECTED_DROPS: dict[str, set[str]] = {
    "banner": {ARCHIVE_BANNER},
    "cookie": {
        "We use cookies to measure visits to this invented site. Accept all "
        "cookies or manage your preferences."
    },
    "menu": {
        "Missions",
        "Galleries",
        "News & Events",
        "About the Observatory",
        "Privacy Policy",
        "Terms of Use",
        "Accessibility",
        "Contact the Observatory",
    },
    "breadcrumb": {"Home", "News & Features", "Understanding Climate"},
    "social": {
        "Follow NASA",
        "NASA on Facebook",
        "NASA on Instagram",
        "NASA on X",
        "NASA on YouTube",
    },
    "related": {
        "Related",
        "Ten invented facts about the southern terrace",
        "What is the basin's carbon budget? (invented teaser)",
        "Vital signs of the basin: shelf sea level (invented teaser)",
    },
}

#: Substantive controls that must survive byte-identical (each is the
#: full text of one parsed block).
SUBSTANTIVE_KEEPS = {
    # Section prose with claims and calibrated language.
    "Basin-mean surface temperature has risen by 1.9 °C since the 1861–1880 "
    "baseline. It is very likely that warming will continue through "
    "mid-century under every pathway assessed for the basin.",
    # Fail-open: prose CONTAINING a marker phrase is not a social block.
    "Researchers follow NASA-style calibration practices when homogenising "
    "the station record, so the observed trend is robust to instrument moves "
    "and re-sitings.",
    # Substantive list items (claims with numbers) are not menus.
    "Terrace glaciers lost 14% of their mapped area between 1990 and 2024, "
    "the fastest loss in the invented record.",
    "Southern-terrace rainfall declined from 640 mm to 550 mm per year over the same period.",
    # Ambiguous short factual paragraph — fail-open keeps it.
    "Data are updated quarterly by the observatory.",
    # Q&A pair: the question heading and its answer.
    "Weren't there warnings of basin cooling years ago?",
    "A handful of 1970s articles speculated about a coming basin cooling, "
    "but the assessed literature of that decade already leaned toward "
    "warming, and the observations gathered since have settled the question.",
    # A genuine heading merely CONTAINING a label word is not a teaser rail.
    "Related warming feedbacks",
    "Loss of terrace snow lowers the basin's albedo, which amplifies the "
    "warming that caused the loss — an assessed feedback carried in every "
    "basin projection.",
}


def _parse() -> StructuredDoc:
    return parse_html(BOILERPLATE_PAGE.read_text(encoding="utf-8"), doc_id="syn-boilerplate")


def _filter() -> tuple[StructuredDoc, BoilerplateAudit]:
    return filter_html_boilerplate(_parse())


def _block_key(block: Block) -> tuple[str, str, int | None, str | None]:
    return (block.type.value, block.text, block.level, block.caption)


def test_fixture_parses_with_boilerplate_present():
    """Honesty precondition: the fixture's boilerplate really does arrive
    as ordinary parsed blocks (the #331 finding) — the filter has real
    work to do, and the drop tests below cannot pass vacuously."""
    parsed = _parse()
    texts = {b.text for b in parsed.blocks}
    for reason, expected in EXPECTED_DROPS.items():
        missing = expected - texts
        assert not missing, f"{reason} fixture blocks failed to parse: {missing!r}"
    missing_keeps = SUBSTANTIVE_KEEPS - texts
    assert not missing_keeps, f"substantive fixture blocks failed to parse: {missing_keeps!r}"


def test_filter_returns_document_and_audit():
    filtered, audit = _filter()
    assert isinstance(filtered, StructuredDoc)
    assert isinstance(audit, BoilerplateAudit)
    assert filtered.doc_id == "syn-boilerplate"
    assert audit.doc_id == "syn-boilerplate"
    assert filtered.backend == "html", "the filter must not relabel the parse backend"
    assert filtered.title == _parse().title, "the filter must not retitle the document"


@pytest.mark.parametrize("reason", sorted(EXPECTED_DROPS))
def test_boilerplate_shape_dropped_and_recorded(reason: str):
    """Each packet shape — banner, cookie notice, menus, breadcrumbs, the
    'Follow NASA' social block (heading AND items), the 'Related' teaser
    rail (heading AND teasers) — is dropped from the block stream and
    every drop is recorded in the audit under this reason."""
    filtered, audit = _filter()
    kept_texts = {b.text for b in filtered.blocks}
    leaked = EXPECTED_DROPS[reason] & kept_texts
    assert not leaked, f"{reason} boilerplate leaked into kept blocks: {leaked!r}"
    recorded = {d.text for d in audit.dropped if d.reason == reason}
    missing = EXPECTED_DROPS[reason] - recorded
    assert not missing, f"drops missing from the {reason} audit trail: {missing!r}"


def test_substantive_content_survives_byte_identical():
    """Every substantive control block survives with its text unchanged,
    byte for byte — the filter drops, it never rewrites."""
    filtered, _audit = _filter()
    kept_texts = {b.text for b in filtered.blocks}
    missing = SUBSTANTIVE_KEEPS - kept_texts
    assert not missing, f"substantive content lost or altered by the filter: {missing!r}"


def test_kept_blocks_are_an_in_order_subsequence_of_input():
    """Drop-only invariant: the output blocks are the input blocks minus
    the dropped ones — same order, same type/text/level/caption. No
    merging, trimming, reordering or rewriting."""
    parsed = _parse()
    filtered, audit = _filter()
    input_keys = [_block_key(b) for b in parsed.blocks]
    position = 0
    for block in filtered.blocks:
        key = _block_key(block)
        while position < len(input_keys) and input_keys[position] != key:
            position += 1
        assert position < len(input_keys), (
            f"kept block not found in input order (rewritten or reordered): {key!r}"
        )
        position += 1
    assert len(filtered.blocks) + audit.dropped_count == len(parsed.blocks), (
        "every input block must be either kept or audited as dropped — never silently lost"
    )


def test_fail_open_marker_phrases_inside_prose_are_kept():
    """A prose paragraph that merely CONTAINS marker vocabulary — 'follow
    NASA-style calibration', a sentence listing 'breadcrumbs, menus, a
    social-follow block' — is ambiguous at worst and must be KEPT."""
    filtered, _audit = _filter()
    kept = " \x00 ".join(b.text for b in filtered.blocks)
    assert "follow NASA-style calibration practices" in kept
    assert "imitates the boilerplate shapes of an agency explainer page" in kept


def test_fail_open_ambiguity_is_kept_never_silently_dropped():
    """The pinned honesty rule: nothing outside the audited drops may go
    missing. Kept + audited == parsed, and every audited drop is one of
    the known boilerplate shapes — an ambiguous block the rules cannot
    classify stays in the stream for the hand audit to judge."""
    parsed = _parse()
    filtered, audit = _filter()
    all_expected_drops = set().union(*EXPECTED_DROPS.values())
    for dropped in audit.dropped:
        assert dropped.text in all_expected_drops, (
            f"over-eager drop — ambiguous block removed: {dropped.text!r} ({dropped.reason})"
        )
    assert len(filtered.blocks) == len(parsed.blocks) - len(all_expected_drops)


def test_audit_reasons_are_closed_vocabulary_and_lines_render():
    filtered, audit = _filter()
    assert audit.dropped_count == len(audit.dropped) > 0
    for dropped in audit.dropped:
        assert dropped.reason in BOILERPLATE_REASONS, f"unknown reason {dropped.reason!r}"
        assert dropped.text, "an audit entry must identify the dropped text"
        assert dropped.block_type in {t.value for t in BlockType}
    lines = audit.report_lines()
    assert len(lines) == audit.dropped_count
    assert f"banner: {ARCHIVE_BANNER}" in lines
    assert "social: Follow NASA" in lines


@pytest.mark.parametrize("backend", ["docling", "pymupdf", "test-fixture"])
def test_non_html_backends_bypass_the_filter_entirely(backend: str):
    """HTML-source scoping: a document from any other backend passes
    through with its block list unchanged and an empty audit — even when
    its text LOOKS like boilerplate (a PDF quoting the banner is
    evidence, not furniture)."""
    doc = StructuredDoc(
        doc_id="syn-pdf-like",
        title="syn-pdf-like (invented)",
        blocks=[
            Block(BlockType.HEADING, "1 Invented section", level=1),
            Block(BlockType.TEXT, ARCHIVE_BANNER),
            Block(BlockType.TEXT, "Follow NASA"),
            Block(BlockType.LIST_ITEM, "Home"),
        ],
        backend=backend,
    )
    filtered, audit = filter_html_boilerplate(doc)
    assert filtered.blocks == doc.blocks, f"{backend} document must bypass the filter"
    assert filtered.doc_id == doc.doc_id and filtered.backend == backend
    assert audit.dropped == ()
    assert audit.dropped_count == 0


def test_filter_is_deterministic():
    first_doc, first_audit = _filter()
    second_doc, second_audit = _filter()
    assert [_block_key(b) for b in first_doc.blocks] == [_block_key(b) for b in second_doc.blocks]
    assert first_audit == second_audit
