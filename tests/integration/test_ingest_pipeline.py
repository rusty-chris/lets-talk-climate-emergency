"""End-to-end ingest of a synthetic fixture corpus (issue #7 TDD plan
12–13) — RED. Integration tier (auto-marked by this directory's
conftest): manifest-driven fetch from file:// fixtures → sha256 verify →
parse → chunk → citation metadata → custom-content blocks; run twice,
identical output. PDFs come through the injected parser seam so no
heavy Docling install is needed here either; the HTML document takes
the production HTML-direct path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ingestion.manifest import Sha256MismatchError
from ingestion.parse import Block, BlockType, StructuredDoc
from ingestion.pipeline import (
    IngestError,
    check_assessed_range_statements_present,
    ingest_corpus,
)
from tests._ingestion_fixtures import config, manifest_entry, minimal_pdf_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPLAINER_HTML = FIXTURES / "ingestion" / "synthetic_explainer.html"


def _fetch_file_url(url: str) -> bytes:
    from urllib.request import urlopen

    with urlopen(url) as handle:  # noqa: S310 - file:// fixtures only
        return handle.read()


def _pdf_parser(backend: str = "docling"):
    """Injected PDF parser (the Docling seam): deterministic synthetic
    StructuredDoc, backend label as requested."""

    def parser(path, doc_id, **_kwargs) -> StructuredDoc:
        return StructuredDoc(
            doc_id=doc_id,
            title=f"{doc_id} (invented)",
            blocks=[
                Block(BlockType.HEADING, "1 Invented PDF section", level=1),
                Block(
                    BlockType.TEXT,
                    "Invented parsed PDF prose, long enough to clear the tiny-chunk "
                    "floor, carrying one very likely calibrated phrase for the "
                    "metadata path.",
                ),
            ],
            backend=backend,
        )

    return parser


def _build_corpus(tmp_path: Path):
    """A two-document synthetic corpus: the committed HTML explainer and
    an in-test-generated minimal PDF, both fetched via file://."""
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    html_bytes = EXPLAINER_HTML.read_bytes()
    html_source = sources / "syn-e2e-explainer.src.html"
    html_source.write_bytes(html_bytes)
    html_entry = manifest_entry(
        "syn-e2e-explainer",
        path="syn-e2e-explainer.html",
        source_url=html_source.as_uri(),
        sha256=hashlib.sha256(html_bytes).hexdigest(),
        provides_assessed_ranges=True,
    )

    pdf_bytes = minimal_pdf_bytes()
    pdf_source = sources / "syn-e2e-notes.src.pdf"
    pdf_source.write_bytes(pdf_bytes)
    pdf_entry = manifest_entry(
        "syn-e2e-notes",
        path="syn-e2e-notes.pdf",
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


def _run(manifest_path: Path, corpus_dir: Path, pdf_backend: str = "docling"):
    return ingest_corpus(
        manifest_path,
        corpus_dir,
        config=config(),
        parser=_pdf_parser(pdf_backend),
        transport=_fetch_file_url,
    )


def test_ingest_fixture_corpus_end_to_end(tmp_path):
    """TDD plan 12: the whole §2.4 flow over file:// fixtures — both
    documents fetched, verified, parsed, chunked with metadata, and one
    citation block emitted per chunk."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir)

    doc_ids = {chunk.doc_id for chunk in result.chunks}
    assert doc_ids == {"syn-e2e-explainer", "syn-e2e-notes"}
    assert len(result.blocks) == len(result.chunks), "one block per citable unit"
    assert set(result.documents) == {"syn-e2e-explainer", "syn-e2e-notes"}

    html_chunks = [c for c in result.chunks if c.doc_id == "syn-e2e-explainer"]
    assert any("13.1" in c.body for c in html_chunks), (
        "the explainer's table values must arrive citable end-to-end"
    )
    for chunk in result.chunks:
        assert chunk.citation_metadata["canonical_url"].endswith(chunk.doc_id)
        assert chunk.token_count > 0


def test_ingest_runs_are_deterministic(tmp_path):
    """TDD plan 12: run twice over identical inputs — identical chunk
    records (ids included) and byte-identical block payloads. This is the
    hash-stable, idempotent re-embedding contract."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    first = _run(manifest_path, corpus_dir)
    second = _run(manifest_path, corpus_dir)
    assert first.chunks == second.chunks
    assert [json.dumps(b, sort_keys=True) for b in first.blocks] == [
        json.dumps(b, sort_keys=True) for b in second.blocks
    ]


def test_e2e_sha256_mismatch_refuses_whole_run(tmp_path):
    """ADR-023/#100 at pipeline level: a tampered pin refuses the run."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    documents = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["documents"]
    documents[0]["sha256"] = "1" * 64
    manifest_path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump({"documents": documents}, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(Sha256MismatchError, match="syn-e2e-explainer"):
        _run(manifest_path, corpus_dir)


def test_e2e_pymupdf_fallback_recorded_in_run_output(tmp_path):
    """Amended §2.4 end-to-end: when the PDF arrives via the fallback
    backend, the run's document record carries the warning and the
    hand-review flag — visible in the pipeline's own output, not just a
    log line."""
    manifest_path, corpus_dir, _ = _build_corpus(tmp_path)
    result = _run(manifest_path, corpus_dir, pdf_backend="pymupdf")
    record = result.documents["syn-e2e-notes"]
    assert record.degraded_fallback is True
    assert record.needs_hand_review is True
    assert record.warnings


def test_assessed_range_statements_present():
    """TDD plan 13, the launch-dependency check over the committed
    fixture corpus manifest (the named test the issue's acceptance
    criteria re-run against the real MVP corpus at release): the
    manifest must contain the assessed-statements source; stripping the
    declaration must fail the check. Presence is pinned, never content."""
    documents = yaml.safe_load((FIXTURES / "corpus" / "manifest.yaml").read_text(encoding="utf-8"))[
        "documents"
    ]
    assert check_assessed_range_statements_present(documents, config()) is None

    stripped = [
        {k: v for k, v in entry.items() if k != "provides_assessed_ranges"} for entry in documents
    ]
    with pytest.raises(IngestError, match="(?i)assessed"):
        check_assessed_range_statements_present(stripped, config())
