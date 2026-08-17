"""HTML-direct parsing (issue #7; DESIGN §2.4 'HTML direct') — RED.

Over the fresh-authored synthetic explainer fixture
tests/fixtures/ingestion/synthetic_explainer.html (invented content,
.invalid links). The HTML path needs no Docling: headings carry their
true nesting in the markup, so the parser must surface it, exclude page
furniture, and keep table cells and figure captions citable.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.parse import HEADING_TYPES, BlockType
from ingestion.pipeline import parse_html

FIXTURES_INGESTION = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
EXPLAINER = FIXTURES_INGESTION / "synthetic_explainer.html"

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE — authored for this project's tests"


def _parse():
    return parse_html(EXPLAINER.read_text(encoding="utf-8"), doc_id="syn-explainer")


def test_ingestion_fixture_files_carry_synthetic_marker():
    """The #24 shipping guard extended to this suite's fixture directory:
    every committed file under tests/fixtures/ingestion/ declares itself
    synthetic in its first line (DESIGN §2.1 — no real Tier B/C text)."""
    files = [p for p in sorted(FIXTURES_INGESTION.rglob("*")) if p.is_file()]
    assert files, "fixture directory unexpectedly empty"
    for path in files:
        first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert SYNTHETIC_MARKER in first_line, f"{path.name}: missing synthetic marker"


def test_html_headings_carry_true_nesting_levels():
    """h2/h3 depth must survive into heading levels: 'The lowland record'
    (h3) sits one level below 'Observed warming' (h2)."""
    parsed = _parse()
    headings = {b.text: b for b in parsed.blocks if b.type in HEADING_TYPES}
    assert "Observed warming" in headings, "h2 heading missing from parse output"
    assert "The lowland record" in headings, "h3 heading missing from parse output"
    h2 = headings["Observed warming"]
    h3 = headings["The lowland record"]
    assert h2.level is not None and h3.level is not None
    assert h3.level == h2.level + 1, (
        f"h3 must nest one level below h2 (got h2={h2.level}, h3={h3.level})"
    )


def test_html_page_furniture_excluded():
    """nav/header/footer/script/style content never becomes blocks."""
    parsed = _parse()
    all_text = " ".join(b.text for b in parsed.blocks) + " ".join(
        b.caption or "" for b in parsed.blocks
    )
    for furniture in (
        "Home",
        "Site navigation",
        "footer furniture",
        "fixture script noise",
        "font-family",
    ):
        assert furniture not in all_text, f"page furniture leaked into blocks: {furniture!r}"


def test_html_table_cells_retained_citable():
    """Finding 3c on the HTML path: the indicator table's cell values are
    present in the parsed blocks — never reduced to a bare placeholder."""
    parsed = _parse()
    table_blocks = [b for b in parsed.blocks if b.type is BlockType.TABLE]
    assert table_blocks, "the fixture's table must parse to a TABLE block"
    table_text = " ".join(b.text + " " + (b.caption or "") for b in table_blocks)
    for cell in ("13.1", "550", "Basin-mean temperature"):
        assert cell in table_text, f"table cell {cell!r} lost in parsing"


def test_html_figure_caption_attached():
    """The figcaption pairs with its figure block (finding 3)."""
    parsed = _parse()
    figure_blocks = [b for b in parsed.blocks if b.type is BlockType.FIGURE]
    assert figure_blocks, "the fixture's figure must parse to a FIGURE block"
    captions = " ".join(b.caption or "" for b in figure_blocks)
    assert "Figure 1. Invented lowland temperature series" in captions


def test_html_prose_lands_under_its_heading():
    """Body prose attaches to the enclosing section in reading order."""
    parsed = _parse()
    texts = [b.text for b in parsed.blocks if b.type is BlockType.TEXT]
    assert any("1.9 °C since the" in t for t in texts), "section prose missing"
    assert any("not likely that station moves" in t for t in texts), (
        "the negation-trap sentence must survive parsing for the metadata tests"
    )
