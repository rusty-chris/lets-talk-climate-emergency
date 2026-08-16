"""Meta-tests for the synthetic fixture corpus (issue #24, IMPLEMENTATION.md §5).

The corpus under tests/fixtures/corpus/ (documents + manifest) and the tiny
CSVs under tests/fixtures/charts/ are the only document/data fixtures the
ingestion, retrieval and chart issues (#7, #9, #11, #15, #16) may build on.
Everything here is authored fresh for this repo: fictional but structurally
realistic. These tests make that safety property load-bearing:

- every fixture file carries the SYNTHETIC FIXTURE first-line marker, so no
  unmarked (i.e. potentially real Tier B/C) text can sit in the corpus —
  DESIGN.md §2.1's shipping invariant, and the guard review finding #51
  showed is needed (real data rows had been committed as "synthetic");
- the corpus actually contains the document structures the chunker/parser
  tests need (nested headings, a table, footnotes, calibrated language
  including the "not likely" negation trap);
- the chart CSVs stay tiny and parseable.

The walk excludes interpreter cache artefacts, extending the convention
PR #54 established for the spike fixtures.
"""

from __future__ import annotations

import csv
from pathlib import Path

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE — authored for this project's tests"

# Interpreter cache artefacts (e.g. __pycache__ written at collection time)
# are not fixtures — same exclusion as tests/unit/test_spike_parsers.py (#44).
_BYTECODE_SUFFIXES = {".pyc", ".pyo"}


def _is_cache_artefact(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in _BYTECODE_SUFFIXES


def _fixture_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not _is_cache_artefact(path.relative_to(directory))
    )


def _read_docs(fixture_corpus_dir: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in _fixture_files(fixture_corpus_dir)
        if path.suffix == ".md"
    }


def test_all_fixture_docs_carry_synthetic_marker(
    fixture_corpus_dir, chart_fixtures_dir, fixture_dataset_manifest_path
):
    """TDD plan item 7 / acceptance criterion: the first line of every fixture

    file in the corpus, chart-CSV and dataset-manifest directories is the
    SYNTHETIC FIXTURE marker, so no unmarked (potentially real Tier B/C)
    text can ever sit there (DESIGN.md §2.1).
    """
    corpus_files = _fixture_files(fixture_corpus_dir)
    chart_files = _fixture_files(chart_fixtures_dir)
    dataset_files = _fixture_files(fixture_dataset_manifest_path.parent)
    assert corpus_files, "synthetic corpus directory is empty"
    assert chart_files, "chart fixtures directory is empty"
    assert dataset_files, "synthetic dataset-manifest directory is empty"
    for path in corpus_files + chart_files + dataset_files:
        text = path.read_text(encoding="utf-8-sig")
        first_line = text.splitlines()[0] if text else ""
        assert SYNTHETIC_MARKER in first_line, (
            f"{path.name}: first line lacks the marker {SYNTHETIC_MARKER!r} - "
            "either add it or remove the file; unmarked text is treated as "
            "potentially real source text and may not be committed"
        )


def test_corpus_has_four_to_six_documents(fixture_corpus_dir):
    """IMPLEMENTATION.md §5: 4-6 short fresh-authored documents."""
    docs = _read_docs(fixture_corpus_dir)
    assert 4 <= len(docs) <= 6, f"expected 4-6 corpus documents, found {sorted(docs)}"


def test_corpus_contains_required_document_structures(fixture_corpus_dir):
    """The structural features the parser/chunker tests need all exist:

    nested headings, a table, and footnotes (issues/24.md scope).
    """
    docs = _read_docs(fixture_corpus_dir)
    all_text = "\n".join(docs.values())

    nested = [
        name
        for name, text in docs.items()
        if any(line.startswith("## ") for line in text.splitlines())
        and any(line.startswith("### ") for line in text.splitlines())
    ]
    assert nested, "no document with nested (##/###) headings"

    tables = [
        name
        for name, text in docs.items()
        if any(line.startswith("|") and line.rstrip().endswith("|") for line in text.splitlines())
    ]
    assert tables, "no document containing a table"

    assert "[^1]" in all_text, "no document containing footnotes"


def test_corpus_contains_calibrated_language_and_negation_trap(fixture_corpus_dir):
    """Calibrated-language sentences exist, including the 'not likely'

    negation trap - the regex-proxy calibration checks (#12, #21) must be
    exercised against a qualifier that flips meaning under naive matching.
    """
    all_text = "\n".join(_read_docs(fixture_corpus_dir).values()).lower()
    for phrase in ("very likely", "high confidence", "not likely"):
        assert phrase in all_text, f"calibrated-language phrase {phrase!r} missing from corpus"
    # The trap must appear as a genuine negation, not only inside 'very likely'.
    assert "not likely" in all_text


def test_chart_csvs_are_tiny_and_parseable(chart_fixtures_dir):
    """Chart-transform CSVs: comment-marked first line, small enough to read

    in review (IMPLEMENTATION.md §5), and parseable as numeric rows.
    """
    csv_paths = [p for p in _fixture_files(chart_fixtures_dir) if p.suffix == ".csv"]
    assert len(csv_paths) >= 2, "expected at least two synthetic chart CSVs"
    for path in csv_paths:
        assert path.stat().st_size < 8192, f"{path.name}: not tiny ({path.stat().st_size} bytes)"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("#"), f"{path.name}: marker line must be a # comment"
        rows = list(csv.reader(lines[1:]))
        header, data = rows[0], rows[1:]
        assert len(header) >= 2, f"{path.name}: expected at least two columns"
        assert len(data) >= 5, f"{path.name}: expected at least five data rows"
        for row in data:
            assert len(row) == len(header), f"{path.name}: ragged row {row}"
            for value in row[1:]:
                float(value)  # numeric data columns


# ---------------------------------------------------------------------------
# The fixture manifest (TDD plan item 8)
# ---------------------------------------------------------------------------

PERMITTED_CONTEXTS = {"open", "non-commercial-educational", "permission-on-file"}

# One deliberate violation entry per licensing invariant enforced in code
# (DESIGN.md §2.1 + issue #5's TDD plan, as amended by ADR-023 and review
# findings #45/#46). #5's validator tests feed these entries to the real
# validator and assert each refusal path fires.
EXPECTED_VIOLATION_INVARIANTS = {
    "unset_permitted_context",
    "unknown_permitted_context_value",
    "empty_human_signoff",
    "permission_on_file_without_evidence",
    "nonopen_prepared_text_in_repo",
    "nonopen_dataset_in_chart_pack",
    "splice_pair_without_alignment_periods",
    # Added for issue #5's amended invariants:
    "licence_claim_without_evidence",  # reviews #45/#46: claims need evidence
    "open_dataset_without_licence_evidence",  # review #45 (Bereiter) at schema level
    "provisional_dataset_in_chart_pack",  # pack is confirmed-open only
    "provenance_segment_without_licence_evidence",  # review #46 (Mauna Loa/Scripps)
    "attribution_missing_segment_credit",  # review #46: caption credit coverage
}

REQUIRED_DOCUMENT_FIELDS = {
    "id",
    "licence",
    "licence_evidence",
    "attribution_text",
    "canonical_url",
    "redistributable",
    "permitted_context",
    "consensus_position",
    "sha256",
    "retrieved_at",
    "source_tier",
    "human_signoff",
}


def _load_manifest(fixture_manifest_path: Path) -> dict:
    assert fixture_manifest_path.is_file(), f"fixture manifest missing: {fixture_manifest_path}"
    import yaml

    manifest = yaml.safe_load(fixture_manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), "manifest must parse to a mapping"
    return manifest


def test_fixture_manifest_covers_every_permitted_context(fixture_manifest_path):
    """TDD plan item 8: open, non-commercial-educational and permission-on-file

    are all represented among the *valid* entries, and every valid entry
    carries the full DESIGN.md §2.1 field set.
    """
    manifest = _load_manifest(fixture_manifest_path)
    documents = manifest["documents"]
    contexts = {doc["permitted_context"] for doc in documents}
    assert contexts == PERMITTED_CONTEXTS, (
        f"valid manifest entries must cover exactly the permitted_context values "
        f"{sorted(PERMITTED_CONTEXTS)}; found {sorted(contexts)}"
    )
    for doc in documents:
        missing = REQUIRED_DOCUMENT_FIELDS - doc.keys()
        assert not missing, f"{doc.get('id')}: missing required fields {sorted(missing)}"
        signoff = doc["human_signoff"]
        assert {"who", "date", "note"} <= signoff.keys(), f"{doc['id']}: incomplete human_signoff"
        if doc["permitted_context"] == "permission-on-file":
            assert doc.get("permission_evidence"), (
                f"{doc['id']}: permission-on-file requires permission_evidence"
            )


def test_fixture_manifest_valid_entries_are_internally_consistent(
    fixture_manifest_path, fixture_corpus_dir
):
    """Valid entries obey the invariants they exist to demonstrate:

    only `open` documents carry in-repo prepared text (DESIGN.md §2.1
    ship rule); referenced files exist and their sha256 matches, so editing
    a fixture doc without updating the manifest fails loudly.
    """
    import hashlib

    manifest = _load_manifest(fixture_manifest_path)
    paths_seen = []
    for doc in manifest["documents"]:
        path = doc.get("path")
        if doc["permitted_context"] != "open":
            assert path is None, (
                f"{doc['id']}: non-open entries must not ship prepared text in-repo"
            )
            continue
        assert path, f"{doc['id']}: open entries in this corpus reference a committed file"
        doc_file = fixture_corpus_dir / path
        assert doc_file.is_file(), f"{doc['id']}: {path} not found in corpus"
        digest = hashlib.sha256(doc_file.read_bytes()).hexdigest()
        assert digest == doc["sha256"], (
            f"{doc['id']}: sha256 mismatch for {path} - fixture edited without "
            f"updating the manifest (expected {doc['sha256']}, got {digest})"
        )
        paths_seen.append(path)
    # Every committed corpus doc is under manifest control.
    committed = {p.name for p in fixture_corpus_dir.glob("*.md")}
    assert committed == set(paths_seen), (
        f"corpus files and manifest disagree: only in corpus {committed - set(paths_seen)}, "
        f"only in manifest {set(paths_seen) - committed}"
    )


def test_fixture_manifest_flags_special_documents(fixture_manifest_path):
    """The corpus's special documents are flagged the way DESIGN.md §2.1/§2.5

    require: one beyond-assessed-range entry, one voices entry, and
    consensus_position defaulting semantics spelled out explicitly.
    """
    manifest = _load_manifest(fixture_manifest_path)
    documents = manifest["documents"]
    beyond = [d for d in documents if d["consensus_position"] == "beyond-assessed-range"]
    assert len(beyond) == 1, "exactly one beyond-assessed-range document expected"
    assert beyond[0]["path"] == "synthetic_beyond_assessed.md"
    voices = [d for d in documents if d.get("source_type") == "voices"]
    assert len(voices) == 1, "exactly one voices document expected"
    assert voices[0]["path"] == "synthetic_voices.md"


def test_fixture_manifest_has_violation_entry_per_invariant(fixture_manifest_path):
    """TDD plan item 8: one deliberate violation entry per §2.1 invariant,

    each naming its invariant and actually exhibiting the violation, so #5's
    validator tests can assert every refusal path against real entries.
    """
    manifest = _load_manifest(fixture_manifest_path)
    violations = manifest["violations"]
    by_invariant = {entry["violates"]: entry for entry in violations}
    assert set(by_invariant) == EXPECTED_VIOLATION_INVARIANTS, (
        f"violation entries must cover exactly {sorted(EXPECTED_VIOLATION_INVARIANTS)}; "
        f"found {sorted(by_invariant)}"
    )

    doc = by_invariant["unset_permitted_context"]["document"]
    assert "permitted_context" not in doc

    doc = by_invariant["unknown_permitted_context_value"]["document"]
    assert doc["permitted_context"] not in PERMITTED_CONTEXTS

    doc = by_invariant["empty_human_signoff"]["document"]
    assert not doc.get("human_signoff")

    doc = by_invariant["permission_on_file_without_evidence"]["document"]
    assert doc["permitted_context"] == "permission-on-file"
    assert not doc.get("permission_evidence")

    doc = by_invariant["nonopen_prepared_text_in_repo"]["document"]
    assert doc["permitted_context"] != "open"
    assert doc.get("path"), "this violation is about in-repo prepared text"

    dataset = by_invariant["nonopen_dataset_in_chart_pack"]["dataset"]
    assert dataset["in_chart_pack"] is True
    assert dataset["permitted_context"] != "open"

    # Restructured for #5: the splice/rebaseline decision lives on the
    # pair (matching datasets/manifest.yaml and ADR-020), and the
    # violation is the decision being entirely unrecorded — neither an
    # explicit `rebaseline: null` nor a block with alignment periods.
    pair = by_invariant["splice_pair_without_alignment_periods"]["splice_pair"]
    assert pair.get("paleo") and pair.get("instrumental")
    assert "rebaseline" not in pair

    doc = by_invariant["licence_claim_without_evidence"]["document"]
    assert doc.get("licence")
    assert not doc.get("licence_evidence")

    dataset = by_invariant["open_dataset_without_licence_evidence"]["dataset"]
    assert dataset["permitted_context"] == "open"
    assert not dataset.get("licence_evidence")

    dataset = by_invariant["provisional_dataset_in_chart_pack"]["dataset"]
    assert dataset["permitted_context"] == "open-provisional"
    assert dataset["in_chart_pack"] is True
    assert dataset.get("licence_note"), "the violation must be pack membership, not a missing note"

    dataset = by_invariant["provenance_segment_without_licence_evidence"]["dataset"]
    segments = dataset["provenance"]
    assert len(segments) >= 2, "multi-origin means at least two provenance segments"
    assert any(not seg.get("licence_evidence") for seg in segments)
    assert any(seg.get("licence_evidence") for seg in segments), (
        "exactly the segment-level gap is the violation, not a blanket omission"
    )

    dataset = by_invariant["attribution_missing_segment_credit"]["dataset"]
    segments = dataset["provenance"]
    assert all(seg.get("credit") and seg.get("licence_evidence") for seg in segments)
    missing = [s for s in segments if s["credit"] not in dataset["attribution_text"]]
    assert missing, "attribution_text must omit at least one segment's credit"
