"""The production ingestion entrypoint (issue #7 acceptance criterion 1;
review finding #145) — integration.

#7 closed without a production caller for ingest_corpus: `make corpus`
runs the #5 invariant/fetch gate only, `ingest_corpus` had no transport
implementation and no manifest to run over. The split is deliberate and
recorded (Makefile + corpus/README.md): `make corpus` stays the fast
licensing gate; `make ingest` drives the full pipeline
(manifest gate -> verified fetch -> parse -> chunk -> citation blocks)
via scripts/ingest_corpus.py with the production urllib transport
(file:// in these tests, https:// live).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPLAINER_HTML = FIXTURES / "ingestion" / "synthetic_explainer.html"


def _entry(doc_id: str, source: Path, sha256: str, **overrides) -> dict:
    fields = {
        "id": doc_id,
        "title": f"{doc_id} (invented)",
        "licence": "CC BY 4.0 (invented)",
        "licence_evidence": "Invented licence statement for the entrypoint tests",
        "attribution_text": f"{doc_id} Fixture Author (fictional)",
        "canonical_url": f"https://mca.example.invalid/{doc_id}",
        "redistributable": True,
        "permitted_context": "open",
        "consensus_position": "assessed",
        "source_type": "evidence",
        "source_tier": "A",
        "sha256": sha256,
        "retrieved_at": "2026-08-21",
        "path": f"{doc_id}.html",
        "source_url": source.as_uri(),
        "provides_assessed_ranges": True,
        "human_signoff": {
            "who": "review-7 test author",
            "date": "2026-08-21",
            "note": "synthetic entrypoint-test entry",
        },
    }
    fields.update(overrides)
    return fields


def _write_manifest(path: Path, documents: list[dict]) -> None:
    path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump({"documents": documents}, allow_unicode=True),
        encoding="utf-8",
    )


def _build_fixture_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    out_dir = tmp_path / "out"

    html_bytes = EXPLAINER_HTML.read_bytes()
    source = sources / "syn-entry-doc.src.html"
    source.write_bytes(html_bytes)
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest,
        [_entry("syn-entry-doc", source, hashlib.sha256(html_bytes).hexdigest())],
    )
    return manifest, corpus_dir, out_dir


def _make_ingest(manifest: Path, corpus_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "make",
            "ingest",
            f"INGEST_MANIFEST={manifest}",
            f"INGEST_CORPUS_DIR={corpus_dir}",
            f"INGEST_OUT_DIR={out_dir}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def _run_ingest_script(
    manifest: Path, corpus_dir: Path, out_dir: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/ingest_corpus.py",
            "--manifest",
            str(manifest),
            "--corpus-dir",
            str(corpus_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def test_make_ingest_runs_the_ingestion_pipeline(tmp_path):
    """Review #145: a make-reachable entrypoint drives manifest ->
    verified fetch -> parse -> chunk -> blocks over the production
    urllib transport (file:// here), persisting the ingest run record
    and the chunk/block payloads."""
    manifest, corpus_dir, out_dir = _build_fixture_corpus(tmp_path)
    result = _make_ingest(manifest, corpus_dir, out_dir)
    assert result.returncode == 0, (
        f"make ingest failed on a clean manifest:\n{result.stdout}\n{result.stderr}"
    )

    run_record = json.loads((corpus_dir / "ingest_run.json").read_text(encoding="utf-8"))
    documents = {entry["doc_id"]: entry for entry in run_record["documents"]}
    assert documents["syn-entry-doc"]["chunk_count"] > 0, "the document must actually chunk"

    chunks_path = out_dir / "chunks.jsonl"
    assert chunks_path.is_file(), "chunk payloads must be persisted for the indexer (#9)"
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert chunks and all(c["doc_id"] == "syn-entry-doc" for c in chunks)
    assert all(c["chunk_id"] for c in chunks)
    blocks_path = out_dir / "blocks.jsonl"
    assert blocks_path.is_file(), "citation block payloads must be persisted"
    blocks = [json.loads(line) for line in blocks_path.read_text(encoding="utf-8").splitlines()]
    assert len(blocks) == len(chunks), "one block per citable unit"


def test_ingest_entrypoint_is_reproducible(tmp_path):
    """Acceptance criterion 1: end-to-end reproducibly — two runs over
    identical inputs produce byte-identical chunk payloads (hash-stable
    chunk ids)."""
    manifest, corpus_dir, out_dir = _build_fixture_corpus(tmp_path)
    first = _run_ingest_script(manifest, corpus_dir, out_dir)
    assert first.returncode == 0, first.stderr
    first_bytes = (out_dir / "chunks.jsonl").read_bytes()
    second = _run_ingest_script(manifest, corpus_dir, out_dir)
    assert second.returncode == 0, second.stderr
    assert (out_dir / "chunks.jsonl").read_bytes() == first_bytes


def test_ingest_entrypoint_exit_codes_fail_closed(tmp_path):
    """The #81 exit-code taxonomy carries over: a sha256 mismatch exits
    3 (drift), leaving no artefact at the indexed path; an invariant
    violation exits 1."""
    manifest, corpus_dir, out_dir = _build_fixture_corpus(tmp_path)
    documents = yaml.safe_load(manifest.read_text(encoding="utf-8"))["documents"]
    documents[0]["sha256"] = "3" * 64
    _write_manifest(manifest, documents)
    drift = _run_ingest_script(manifest, corpus_dir, out_dir)
    assert drift.returncode == 3, f"sha mismatch must exit 3, got {drift.returncode}"
    assert not (corpus_dir / "syn-entry-doc.html").exists(), (
        "refused bytes left at the indexed path (#144)"
    )

    documents[0]["licence_evidence"] = ""
    _write_manifest(manifest, documents)
    refusal = _run_ingest_script(manifest, corpus_dir, out_dir)
    assert refusal.returncode == 1, f"invariant refusal must exit 1, got {refusal.returncode}"


def test_real_corpus_manifest_validates_with_pins_and_assessed_ranges():
    """Review #145: the real MVP corpus manifest exists, passes the full
    #5 schema (real sha256 pins, licence evidence, sign-offs), contains
    the two pin-verified spike documents, and declares the §2.3
    assessed-ranges launch dependency. Manifest-only skeleton sources
    wait as recorded comments until their pins/licensing land — nothing
    fake validates here."""
    from ingestion.manifest import load_corpus_manifest
    from ingestion.pipeline import check_assessed_range_statements_present

    manifest_path = REPO_ROOT / "corpus" / "manifest.yaml"
    assert manifest_path.is_file(), "corpus/manifest.yaml (the real MVP manifest) is missing"
    manifest = load_corpus_manifest(manifest_path)
    ids = {d.id for d in manifest.documents}
    assert {"nca5_ch2", "esd_tipping_review"} <= ids, (
        "the two spike-pinned documents must be real manifest entries"
    )
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert check_assessed_range_statements_present(raw["documents"]) is None


def test_hand_audit_checklist_is_committed():
    """Review #145 / #7 acceptance criterion 2: the chunk-quality
    hand-audit checklist exists per source family."""
    checklist = REPO_ROOT / "corpus" / "hand-audit-checklist.md"
    assert checklist.is_file(), "corpus/hand-audit-checklist.md is missing"
    content = checklist.read_text(encoding="utf-8")
    for family_marker in ("NCA5", "esd_tipping_review", "HTML explainer", "Tier B"):
        assert family_marker in content, f"checklist must cover the {family_marker!r} family"
