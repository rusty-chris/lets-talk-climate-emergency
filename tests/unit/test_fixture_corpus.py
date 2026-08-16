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


def test_all_fixture_docs_carry_synthetic_marker(fixture_corpus_dir, chart_fixtures_dir):
    """TDD plan item 7 / acceptance criterion: the first line of every fixture

    file in the corpus and chart-CSV directories is the SYNTHETIC FIXTURE
    marker, so no unmarked (potentially real Tier B/C) text can ever sit
    there (DESIGN.md §2.1).
    """
    corpus_files = _fixture_files(fixture_corpus_dir)
    chart_files = _fixture_files(chart_fixtures_dir)
    assert corpus_files, "synthetic corpus directory is empty"
    assert chart_files, "chart fixtures directory is empty"
    for path in corpus_files + chart_files:
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
