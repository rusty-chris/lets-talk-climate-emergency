"""Pipeline wiring of the HTML boilerplate filter (issue #331) — RED.

Integration tier (auto-marked by this directory's conftest), mirroring
test_ingest_pipeline.py's file:// + injected-parser harness. Pins the
production seam: :func:`ingestion.pipeline.ingest_corpus` applies
:func:`ingestion.boilerplate.filter_html_boilerplate` between the HTML
parse and ``chunk_document`` —

- boilerplate never reaches chunk output for HTML documents, while the
  surviving chunks are exactly the pure-function composition
  ``chunk_document(filter_html_boilerplate(parse_html(...)))`` (one
  seam, no divergent second implementation, survivors byte-identical);
- PDF documents bypass the filter entirely — their chunks keep flowing
  even when their text looks like boilerplate, and their ingest record
  shows zero boilerplate drops;
- the per-document audit trail (count + reasons) is surfaced on
  :class:`~ingestion.pipeline.DocumentIngestRecord` AND persisted in the
  written ingest report ``corpus/ingest_run.json``;
- reruns over identical inputs stay byte-identical (determinism).

Fixtures are synthetic: the committed boilerplate page (see
tests/unit/test_ingestion_boilerplate.py's docstring for the two
deliberate verbatim marker strings) and an in-test minimal PDF.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ingestion.boilerplate import BOILERPLATE_REASONS, filter_html_boilerplate
from ingestion.parse import Block, BlockType, StructuredDoc
from ingestion.pipeline import chunk_document, ingest_corpus, parse_html
from tests._ingestion_fixtures import config, manifest_entry, minimal_pdf_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BOILERPLATE_PAGE = FIXTURES / "ingestion" / "synthetic_boilerplate_page.html"

HTML_DOC_ID = "syn-boiler-explainer"
PDF_DOC_ID = "syn-boiler-notes"

#: Boilerplate strings that must never appear in any HTML chunk body.
BOILERPLATE_CHUNK_POISON = (
    "ARCHIVED version of NOAA Climate.gov",
    "Accept all cookies",
    "Understanding Climate",  # breadcrumb crumb
    "Ten invented facts about the southern terrace",  # related-content teaser
    "NASA on Facebook",
    "Privacy Policy",
)


def _fetch_file_url(url: str) -> bytes:
    from urllib.request import urlopen

    with urlopen(url) as handle:  # noqa: S310 - file:// fixtures only
        return handle.read()


#: PDF prose that LOOKS like boilerplate ("Follow NASA" verbatim) but is
#: parsed evidence in a PDF document — the filter must never touch it.
#: Long enough to clear the min_tokens floor under the word counter.
PDF_MARKER_PROSE = (
    "Follow NASA and the invented shelf agencies publish the basin record "
    "every month; this sentence lives inside a PDF document, so the HTML "
    "boilerplate filter must never touch, drop or audit it in any way."
)


def _pdf_parser(path, doc_id, **_kwargs) -> StructuredDoc:
    return StructuredDoc(
        doc_id=doc_id,
        title=f"{doc_id} (invented)",
        blocks=[
            Block(BlockType.HEADING, "1 Invented PDF section", level=1),
            Block(BlockType.TEXT, PDF_MARKER_PROSE),
        ],
        backend="docling",
    )


def _build_corpus(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    html_bytes = BOILERPLATE_PAGE.read_bytes()
    html_source = sources / f"{HTML_DOC_ID}.src.html"
    html_source.write_bytes(html_bytes)
    html_entry = manifest_entry(
        HTML_DOC_ID,
        path=f"{HTML_DOC_ID}.html",
        source_url=html_source.as_uri(),
        sha256=hashlib.sha256(html_bytes).hexdigest(),
        provides_assessed_ranges=True,
    )

    pdf_bytes = minimal_pdf_bytes()
    pdf_source = sources / f"{PDF_DOC_ID}.src.pdf"
    pdf_source.write_bytes(pdf_bytes)
    pdf_entry = manifest_entry(
        PDF_DOC_ID,
        path=f"{PDF_DOC_ID}.pdf",
        source_url=pdf_source.as_uri(),
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
    )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump({"documents": [html_entry, pdf_entry]}, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest_path, corpus_dir, {"html": html_entry, "pdf": pdf_entry}


def _run(manifest_path: Path, corpus_dir: Path):
    return ingest_corpus(
        manifest_path,
        corpus_dir,
        config=config(),
        parser=_pdf_parser,
        transport=_fetch_file_url,
    )


def test_html_boilerplate_never_reaches_chunk_output(tmp_path):
    """No banner/cookie/breadcrumb/social/related/menu text survives into
    any chunk body or section path of the HTML document."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)
    html_chunks = [c for c in result.chunks if c.doc_id == HTML_DOC_ID]
    assert html_chunks, "the HTML explainer must still produce substantive chunks"
    for chunk in html_chunks:
        for poison in BOILERPLATE_CHUNK_POISON:
            assert poison not in chunk.body, (
                f"boilerplate leaked into a chunk body: {poison!r} in {chunk.chunk_id}"
            )
        for element in chunk.section_path:
            assert element not in ("Follow NASA", "Related"), (
                f"a boilerplate section survived into section_path: {chunk.section_path!r}"
            )


def test_substantive_html_chunks_survive_with_section_paths(tmp_path):
    """The substantive controls keep chunking under their real headings:
    claims, the Q&A pair, and the qualified 'Related warming feedbacks'
    section all reach the chunk output."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)
    html_chunks = [c for c in result.chunks if c.doc_id == HTML_DOC_ID]
    bodies = " \x00 ".join(c.body for c in html_chunks)
    assert "risen by 1.9 °C since the 1861–1880 baseline" in bodies
    assert "follow NASA-style calibration practices" in bodies
    assert "Terrace glaciers lost 14% of their mapped area" in bodies
    assert "speculated about a coming basin cooling" in bodies
    assert "amplifies the warming that caused the loss" in bodies
    section_paths = {c.section_path for c in html_chunks}
    assert ("How do we know the basin is warming?",) in section_paths
    assert ("Weren't there warnings of basin cooling years ago?",) in section_paths
    assert ("Related warming feedbacks",) in section_paths, (
        "a genuine section merely containing 'Related' must survive (fail-open)"
    )


def test_pipeline_chunks_equal_pure_filter_composition(tmp_path):
    """One seam, byte-identical survivors: the pipeline's HTML chunks are
    exactly chunk_document over filter_html_boilerplate over parse_html —
    same ids, section paths and bodies. The filter must live at the
    audited block seam, not as a second implementation (and never as a
    silent parse_html skip-list extension, which would leave the audit
    empty and fail the report tests below)."""
    manifest_path, corpus_dir, entries = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)
    html_chunks = [c for c in result.chunks if c.doc_id == HTML_DOC_ID]

    parsed = parse_html(
        BOILERPLATE_PAGE.read_text(encoding="utf-8"),
        HTML_DOC_ID,
        title=entries["html"]["title"],
    )
    filtered, _audit = filter_html_boilerplate(parsed)
    expected = chunk_document(filtered, entries["html"], config())
    assert [(c.chunk_id, c.section_path, c.body) for c in html_chunks] == [
        (c.chunk_id, c.section_path, c.body) for c in expected
    ]


def test_pdf_documents_bypass_the_filter_entirely(tmp_path):
    """A PDF document's chunks flow byte-identical even when its prose
    contains 'Follow NASA' verbatim, and its ingest record shows zero
    boilerplate drops — the filter is HTML-source-scoped."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)
    pdf_chunks = [c for c in result.chunks if c.doc_id == PDF_DOC_ID]
    assert pdf_chunks, "the PDF document must produce chunks"
    assert any("Follow NASA" in c.body for c in pdf_chunks), (
        "PDF evidence containing boilerplate-looking text must survive untouched"
    )
    record = result.documents[PDF_DOC_ID]
    assert record.boilerplate_dropped == 0
    assert record.boilerplate_reasons == ()


def test_ingest_report_surfaces_the_boilerplate_audit(tmp_path):
    """Count + reasons per document, on the in-memory record AND in the
    persisted corpus/ingest_run.json — the removal is auditable, never
    silent."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)
    record = result.documents[HTML_DOC_ID]
    assert record.boilerplate_dropped > 0, "the HTML document's drops must be counted"
    assert record.boilerplate_dropped == len(record.boilerplate_reasons)
    for line in record.boilerplate_reasons:
        reason = line.split(":", 1)[0]
        assert reason in BOILERPLATE_REASONS, f"unknown reason in report line {line!r}"
    assert "social: Follow NASA" in record.boilerplate_reasons
    assert any(
        line.startswith("banner: This website is an ARCHIVED")
        for line in record.boilerplate_reasons
    )

    run_record = json.loads((corpus_dir / "ingest_run.json").read_text(encoding="utf-8"))
    by_id = {entry["doc_id"]: entry for entry in run_record["documents"]}
    html_entry = by_id[HTML_DOC_ID]
    assert html_entry["boilerplate_dropped"] == record.boilerplate_dropped
    assert tuple(html_entry["boilerplate_reasons"]) == record.boilerplate_reasons
    pdf_entry = by_id[PDF_DOC_ID]
    assert pdf_entry["boilerplate_dropped"] == 0
    assert tuple(pdf_entry["boilerplate_reasons"]) == ()


def test_filtered_ingest_is_deterministic_across_reruns(tmp_path):
    """Guard pin: with the filter in the path, two runs over identical
    inputs still produce identical chunks (ids included) and identical
    per-document records — the idempotent re-embedding hook survives."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    first = _run(manifest_path, corpus_dir)
    second = _run(manifest_path, corpus_dir)
    assert [(c.chunk_id, c.section_path, c.body) for c in first.chunks] == [
        (c.chunk_id, c.section_path, c.body) for c in second.chunks
    ]
    assert first.documents == second.documents
