"""`make corpus` end-to-end (issue #5, TDD plan item 11) — integration.

RED phase. The issue's amended contract (ADR-023 + the #5 scope comment):
`make corpus` re-fetches the manifest's open-tier text, verifies every
fetched file against its manifest-pinned sha256, and runs all licensing
invariants — failing loudly, naming the offending document, on any
violation. The repo pins which bytes without hosting them; the public
archives are the data store.

Pinned interface (the Makefile does not exist yet; this is its contract):

    make corpus CORPUS_MANIFEST=<manifest.yaml> CORPUS_DIR=<dir>

with the variables defaulting to corpus/manifest.yaml and corpus/. The
target must be runnable from the repo root and exit non-zero on any
invariant or hash failure, with the offending document id in its output
(stdout or stderr) so CI logs identify the document without re-running.
Implementation note: the repo has a corpus/ directory, so without a
`.PHONY: corpus` declaration make reports "Nothing to be done" and exits
zero — the target must be phony.

These tests drive the target against temp manifests built from invented
entries whose `source_url` uses file:// URIs into tmp_path — no network,
no real text; Docker-free but subprocess-driven, hence integration tier
(auto-marked by this directory's conftest).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_SYNTHETIC_TEXT = (
    "# SYNTHETIC FIXTURE — authored for this project's tests\n"
    "\n"
    "## Invented section\n"
    "\n"
    "Runtime-only body for the make-corpus integration tests.\n"
)


def _open_entry(doc_id: str, source_file: Path, sha256: str) -> dict:
    """A fully valid `open` document entry fetching from a file:// source.

    All metadata invented (.invalid URLs, fictional signoff).
    """
    return {
        "id": doc_id,
        "title": f"{doc_id} (invented)",
        "path": f"{doc_id}.md",
        "source_url": source_file.as_uri(),
        "source_tier": "A",
        "source_type": "evidence",
        "licence": "CC BY 4.0 (invented)",
        "licence_evidence": "Invented licence statement captured 2026-08-16",
        "attribution_text": "Make Corpus Fixture (fictional)",
        "canonical_url": f"https://mca.example.invalid/{doc_id}",
        "redistributable": True,
        "permitted_context": "open",
        "permission_evidence": None,
        "consensus_position": "assessed",
        "sha256": sha256,
        "retrieved_at": "2026-08-16",
        "human_signoff": {"who": "fixture", "date": "2026-08-16", "note": "make corpus test"},
    }


def _write_manifest(path: Path, documents: list[dict]) -> None:
    path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump({"documents": documents}, allow_unicode=True),
        encoding="utf-8",
    )


def _make_corpus(manifest: Path, corpus_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["make", "corpus", f"CORPUS_MANIFEST={manifest}", f"CORPUS_DIR={corpus_dir}"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def test_make_corpus_fetches_verifies_and_passes_clean_manifest(tmp_path):
    """TDD plan item 11, happy path: against a clean manifest, `make

    corpus` fetches each open document from its source_url into
    CORPUS_DIR, the fetched bytes verify against the pinned sha256, all
    invariants run, and the target exits zero.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    documents = []
    for doc_id in ("syn-make-a", "syn-make-b"):
        source = sources / f"{doc_id}.src.md"
        text = _SYNTHETIC_TEXT + f"\nDocument {doc_id}.\n"
        source.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        documents.append(_open_entry(doc_id, source, digest))
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, documents)

    result = _make_corpus(manifest, corpus_dir)
    assert result.returncode == 0, (
        f"make corpus failed on a clean manifest:\n{result.stdout}\n{result.stderr}"
    )
    for doc in documents:
        fetched = corpus_dir / doc["path"]
        assert fetched.is_file(), f"{doc['id']}: not fetched to {fetched}"
        assert hashlib.sha256(fetched.read_bytes()).hexdigest() == doc["sha256"]


def test_make_corpus_fails_on_sha256_mismatch(tmp_path):
    """TDD plan item 11, hash path: a manifest pinning the wrong sha256

    for its source makes the target fail, naming the document and the
    sha256 field — fetched bytes that do not match the pin never pass
    silently (ADR-023).
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    source = sources / "syn-make-drift.src.md"
    source.write_text(_SYNTHETIC_TEXT, encoding="utf-8")
    entry = _open_entry("syn-make-drift", source, "0" * 64)  # deliberately wrong pin
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])

    result = _make_corpus(manifest, corpus_dir)
    output = result.stdout + result.stderr
    assert result.returncode == 3, (
        "a sha256 mismatch (upstream drift, needs re-pinning review) must exit with "
        f"its own code 3, distinct from invariant refusal (1) and fetch failure (2); "
        f"got {result.returncode}"
    )
    assert "syn-make-drift" in output, "failure output must name the offending document"
    assert "sha256" in output, "failure output must name the violated field"
    # Fail closed (review #80): the semantic pinned here is that nothing
    # lands at the indexed path on a mismatch — the refused bytes are
    # deleted, not quarantined, so a later step trusting corpus_dir can
    # never ingest bytes the gate refused (the validation/indexing TOCTOU).
    destination = corpus_dir / entry["path"]
    assert not destination.exists(), (
        "unverified fetched bytes must never persist at the indexed path after a "
        "sha256 mismatch (review #80)"
    )
    leftovers = [p.name for p in corpus_dir.rglob("*") if p.is_file()]
    assert leftovers == [], f"no partial/temp artefacts may survive a refused fetch: {leftovers}"


def test_make_corpus_verifies_committed_open_text(tmp_path):
    """Review #80: an `open` document whose prepared text is committed
    in-repo (path set, no source_url — the shape §2.1 permits and the
    fixture corpus uses) is verified against its sha256 pin too; the pin
    says *which bytes* (ADR-023) whether or not the bytes were fetched.
    Wrong pin -> failure naming the document and sha256; correct pin ->
    the target passes.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    committed = corpus_dir / "syn-make-committed.md"
    committed.write_text(_SYNTHETIC_TEXT, encoding="utf-8")
    digest = hashlib.sha256(_SYNTHETIC_TEXT.encode("utf-8")).hexdigest()

    entry = _open_entry("syn-make-committed", committed, digest)
    entry["path"] = "syn-make-committed.md"
    entry["source_url"] = None
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])

    result = _make_corpus(manifest, corpus_dir)
    assert result.returncode == 0, (
        f"correctly pinned committed open text must pass:\n{result.stdout}\n{result.stderr}"
    )

    tampered = dict(entry, sha256="1" * 64)  # the committed bytes no longer match the pin
    _write_manifest(manifest, [tampered])
    result = _make_corpus(manifest, corpus_dir)
    output = result.stdout + result.stderr
    assert result.returncode == 3, "a tampered/stale committed file is the hash-drift class (3)"
    assert "syn-make-committed" in output, "failure output must name the offending document"
    assert "sha256" in output, "failure output must name the violated field"


def test_make_corpus_names_document_on_fetch_failure(tmp_path):
    """Review #81: a fetch failure (here file:// to a nonexistent path —
    the unreachable-origin case ADR-023 predicts as the *common* failure)
    names the document id and the source URL in the output, and exits
    with the fetch/environment code 2, distinct from the licensing
    refusal code 1 — so a CI retry wrapper can retry flaky origins
    without ever retrying a genuine licensing refusal.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    missing_source = tmp_path / "sources" / "never-written.src.md"
    entry = _open_entry("syn-make-unreachable", missing_source, "2" * 64)
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [entry])

    result = _make_corpus(manifest, corpus_dir)
    output = result.stdout + result.stderr
    assert result.returncode == 2, (
        f"fetch failures are retryable environment failures, exit 2; got {result.returncode}"
    )
    assert "syn-make-unreachable" in output, "fetch failures must name the offending document"
    assert missing_source.as_uri() in output, "fetch failures must name the source URL"


def test_make_corpus_exit_codes_are_documented_constants():
    """The exit-code taxonomy is part of the target's contract (review
    #81): 1 = licensing/invariant refusal, 2 = fetch/environment failure,
    3 = sha256 mismatch. Pinned as importable constants so CI wrappers
    and these tests share one source of truth.
    """
    import scripts.make_corpus as make_corpus

    assert make_corpus.EXIT_INVARIANT == 1
    assert make_corpus.EXIT_FETCH == 2
    assert make_corpus.EXIT_HASH_MISMATCH == 3


def test_make_corpus_fails_on_mistiered_document(tmp_path):
    """TDD plan item 11, invariant path: the deliberately mis-tiered case —

    a non-commercial-educational document with prepared text present in
    the corpus directory — makes the target fail, naming the document.
    The invariants run as part of `make corpus`, not only in pytest.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # A valid open document so the manifest is otherwise healthy.
    source = sources / "syn-make-ok.src.md"
    source.write_text(_SYNTHETIC_TEXT, encoding="utf-8")
    digest = hashlib.sha256(_SYNTHETIC_TEXT.encode("utf-8")).hexdigest()
    good = _open_entry("syn-make-ok", source, digest)

    # The mis-tiered document: non-open, yet its prepared text sits in the
    # corpus directory (the DESIGN §2.1 ship rule violation).
    prepared = corpus_dir / "syn-make-mistiered.md"
    prepared.write_text(_SYNTHETIC_TEXT, encoding="utf-8")
    bad = _open_entry("syn-make-mistiered", prepared, digest)
    bad["permitted_context"] = "non-commercial-educational"
    bad["redistributable"] = False
    bad["source_url"] = None
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [good, bad])

    result = _make_corpus(manifest, corpus_dir)
    output = result.stdout + result.stderr
    assert result.returncode == 1, (
        "a licensing-invariant violation is the refusal class: exit 1 (review #81), "
        f"got {result.returncode}"
    )
    assert "syn-make-mistiered" in output, "failure output must name the offending document"
