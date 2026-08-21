"""Pipeline entry behaviour (issue #7; DESIGN §2.1/§2.3/§2.4) — RED.

Unit tier: file:// sources in tmp_path, injected transport and parser,
no subprocess, no network. Pins: the licensing gate refuses the WHOLE
run at entry (before any fetch); sha256 verifies the ingest artefact
(#100 semantics); the PyMuPDF fallback is loud (warning + hand-review
flag, never silent); the assessed-range launch dependency (§2.3) is
checked; the spike's heavy parse deps are productionised.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from ingestion.manifest import ManifestError, Sha256MismatchError
from ingestion.parse import Block, BlockType, StructuredDoc
from ingestion.pipeline import check_assessed_range_statements_present, ingest_corpus
from tests._ingestion_fixtures import config, manifest_entry, minimal_pdf_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]

_HTML_BODY = (
    "<!-- SYNTHETIC FIXTURE — authored for this project's tests -->\n"
    "<html><body><main><h1>Invented gate-test page</h1>"
    "<h2>1 Invented section</h2>"
    "<p>Invented evidence prose for the pipeline gate tests. A second invented "
    "sentence keeps the chunk above any floor.</p>"
    "</main></body></html>\n"
)


def _write_manifest(path: Path, documents: list[dict]) -> None:
    path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump({"documents": documents}, allow_unicode=True),
        encoding="utf-8",
    )


def _html_entry(tmp_path: Path, doc_id: str = "syn-gate-doc", **overrides) -> dict:
    source = tmp_path / "sources" / f"{doc_id}.src.html"
    source.parent.mkdir(exist_ok=True)
    source.write_text(_HTML_BODY, encoding="utf-8")
    # Overrides win over the defaults (e.g. the mismatch test supplies a
    # deliberately wrong ``sha256``); merging before the call avoids a
    # "multiple values for keyword argument" TypeError.
    fields = dict(
        path=f"{doc_id}.html",
        source_url=source.as_uri(),
        sha256=hashlib.sha256(_HTML_BODY.encode("utf-8")).hexdigest(),
        provides_assessed_ranges=True,
    )
    fields.update(overrides)
    return manifest_entry(doc_id, **fields)


class SpyTransport:
    """Injected transport recording every fetch."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        from urllib.request import urlopen

        with urlopen(url) as handle:  # noqa: S310 - file:// only, test fixture
            return handle.read()


def test_invariant_violating_doc_refuses_whole_run_before_any_fetch(tmp_path):
    """DESIGN §2.1: one bad document refuses the WHOLE run — and the
    refusal happens at entry, before the transport is ever touched, so
    no bytes of any document are fetched under a refused manifest."""
    good = _html_entry(tmp_path, "syn-gate-good")
    bad = manifest_entry("syn-gate-bad", licence_evidence="")  # unbacked licence claim
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [good, bad])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    transport = SpyTransport()
    with pytest.raises(ManifestError, match="syn-gate-bad"):
        ingest_corpus(manifest, corpus_dir, config=config(), transport=transport)
    assert transport.calls == [], (
        f"the gate must refuse before any fetch — the transport was called for {transport.calls}"
    )


def test_sha256_mismatch_refuses_the_run(tmp_path):
    """#100 semantics: the manifest sha256 pins the ingest artefact at
    source_url; fetched bytes that do not match refuse the run with the
    drift class (Sha256MismatchError), naming the document."""
    entry = _html_entry(tmp_path, "syn-gate-drift", sha256="0" * 64)
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    with pytest.raises(Sha256MismatchError, match="syn-gate-drift"):
        ingest_corpus(manifest, corpus_dir, config=config(), transport=SpyTransport())


def test_assessed_range_source_required_at_entry(tmp_path):
    """DESIGN §2.3 launch dependency: a manifest with no document
    declaring provides_assessed_ranges refuses the ingest (IngestError
    naming the dependency) — 'Hansen in, assessed ranges out' must be
    impossible to ship silently."""
    from ingestion.pipeline import IngestError

    entry = _html_entry(tmp_path, "syn-gate-lone")
    del entry["provides_assessed_ranges"]
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    with pytest.raises(IngestError, match="(?i)assessed"):
        ingest_corpus(manifest, corpus_dir, config=config(), transport=SpyTransport())


def test_headline_only_assessed_source_does_not_satisfy_check_while_flag_off():
    """The two amendments interact: a headline-statements document cannot
    carry the launch dependency while its feature flag is off — the check
    counts only documents that will actually be ingested."""
    from ingestion.pipeline import IngestError

    headline_only = [
        manifest_entry(
            "syn-headlines",
            ingest_profile="headline-statements",
            provides_assessed_ranges=True,
        )
    ]
    with pytest.raises(IngestError, match="(?i)assessed"):
        check_assessed_range_statements_present(headline_only, config())
    assert (
        check_assessed_range_statements_present(
            headline_only, config(headline_statements_enabled=True)
        )
        is None
    )


def _pdf_entry(tmp_path: Path, doc_id: str) -> dict:
    pdf_bytes = minimal_pdf_bytes()
    source = tmp_path / "sources" / f"{doc_id}.src.pdf"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(pdf_bytes)
    return manifest_entry(
        doc_id,
        path=f"{doc_id}.pdf",
        source_url=source.as_uri(),
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        provides_assessed_ranges=True,
    )


def _injected_parser(backend: str):
    def parser(path, doc_id, **_kwargs) -> StructuredDoc:
        return StructuredDoc(
            doc_id=doc_id,
            title=f"{doc_id} (invented)",
            blocks=[
                Block(BlockType.HEADING, "1 Invented section", level=1),
                Block(
                    BlockType.TEXT,
                    "Invented parsed prose long enough to clear any tiny-chunk floor "
                    "in the fallback tests of the production pipeline.",
                ),
            ],
            backend=backend,
        )

    return parser


def test_pymupdf_fallback_is_loud_and_flags_hand_review(tmp_path):
    """Amended DESIGN §2.4: a PyMuPDF-parsed document is a DEGRADED
    result — the run records a per-document warning naming the fallback
    and flags the document for hand review. Never a silent code path."""
    entry = _pdf_entry(tmp_path, "syn-gate-fallback")
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    result = ingest_corpus(
        manifest,
        corpus_dir,
        config=config(),
        parser=_injected_parser("pymupdf"),
        transport=SpyTransport(),
    )
    record = result.documents["syn-gate-fallback"]
    assert record.parse_backend == "pymupdf"
    assert record.degraded_fallback is True
    assert record.needs_hand_review is True, "fallback-parsed documents require hand review"
    assert record.warnings, "the fallback must record a per-document warning"
    assert any("pymupdf" in w.lower() or "fallback" in w.lower() for w in record.warnings)


def test_docling_parse_is_not_flagged_degraded(tmp_path):
    """The primary path carries no degraded/hand-review flags."""
    entry = _pdf_entry(tmp_path, "syn-gate-primary")
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    result = ingest_corpus(
        manifest,
        corpus_dir,
        config=config(),
        parser=_injected_parser("docling"),
        transport=SpyTransport(),
    )
    record = result.documents["syn-gate-primary"]
    assert record.degraded_fallback is False
    assert record.needs_hand_review is False


def test_headline_doc_skip_is_recorded_not_silent(tmp_path):
    """With the flag off (the default), a headline-statements document is
    skipped and the skip is RECORDED on its ingest record — an operator
    can see what the flag withheld."""
    evidence = _html_entry(tmp_path, "syn-gate-evidence")
    headline = _html_entry(tmp_path, "syn-gate-headlines")
    del headline["provides_assessed_ranges"]
    headline["ingest_profile"] = "headline-statements"
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [evidence, headline])
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    result = ingest_corpus(manifest, corpus_dir, config=config(), transport=SpyTransport())
    record = result.documents["syn-gate-headlines"]
    assert record.skipped is True
    assert record.skip_reason, "the skip must say why (feature flag off)"
    assert not any(c.doc_id == "syn-gate-headlines" for c in result.chunks)
    assert any(c.doc_id == "syn-gate-evidence" for c in result.chunks)


def test_parse_dependencies_are_productionised():
    """Spike deviation 3 / #7 ownership: docling and pymupdf were
    spike-only installs; the production pipeline pins them as real
    dependencies so `make corpus` works from a clean environment."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    import tomllib

    dependencies = " ".join(tomllib.loads(pyproject)["project"]["dependencies"])
    assert "docling" in dependencies, "docling must be a pinned production dependency (#7)"
    assert "pymupdf" in dependencies.lower(), (
        "pymupdf (the loud fallback) must be a pinned production dependency (#7)"
    )


def test_sha_mismatch_leaves_no_artefact_at_the_indexed_path(tmp_path):
    """Review finding #144 (the #80 defect re-introduced): ingest_corpus
    wrote fetched bytes to the final artefact path BEFORE verifying and
    left them there on refusal. After a Sha256MismatchError run, nothing
    — neither the artefact nor a temp file — may sit at or beside the
    path an indexing step would read."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    entry = _html_entry(tmp_path, "syn-mismatch-doc", sha256="2" * 64)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, [entry])

    with pytest.raises(Sha256MismatchError, match="syn-mismatch-doc"):
        ingest_corpus(
            manifest_path,
            corpus_dir,
            config=config(),
            transport=SpyTransport(),
        )
    assert not (corpus_dir / "syn-mismatch-doc.html").exists(), (
        "unverified fetched bytes left at the indexed path after a sha256 refusal (#80/#144)"
    )
    leftovers = [p for p in corpus_dir.rglob("*") if p.is_file() and p.name != "ingest_run.json"]
    assert leftovers == [], f"no partial/temp artefacts may survive a refused fetch: {leftovers}"


def test_nonopen_documents_are_not_written_into_corpus_dir(tmp_path):
    """Review finding #144: §2.1 — only permitted_context 'open' text may
    land under the repo-shipped corpus dir. A Tier B
    (non-commercial-educational) document with a source_url must be
    fetched to a workspace OUTSIDE corpus_dir (transiently, for parsing
    only), it must still be chunked, and the prepared-text ship check
    must pass over the corpus tree after the run."""
    from ingestion.manifest import check_prepared_text_shipping

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    open_entry = _html_entry(tmp_path, "syn-open-doc")
    tier_b = _html_entry(
        tmp_path,
        "syn-tier-b-doc",
        permitted_context="non-commercial-educational",
        redistributable=False,
        provides_assessed_ranges=False,
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, [open_entry, tier_b])

    result = ingest_corpus(
        manifest_path,
        corpus_dir,
        config=config(),
        transport=SpyTransport(),
    )
    assert any(c.doc_id == "syn-tier-b-doc" for c in result.chunks), (
        "the Tier B document must still be ingested (via the workspace)"
    )
    assert not (corpus_dir / "syn-tier-b-doc.html").exists(), (
        "non-open prepared text written under the repo-shipped corpus dir (§2.1)"
    )
    assert (corpus_dir / "syn-open-doc.html").exists(), (
        "open text still lands at its declared corpus path"
    )
    documents = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["documents"]
    assert (
        check_prepared_text_shipping(
            (
                {
                    "id": d["id"],
                    "path": d.get("path"),
                    "permitted_context": d["permitted_context"],
                }
                for d in documents
            ),
            corpus_dir,
        )
        is None
    )
