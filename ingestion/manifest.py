"""Corpus/dataset manifest schema + licensing invariants (issue #5) — STUBS.

RED phase: every callable below raises NotImplementedError so the #5 test
suite is importable-but-failing. The implementer replaces these bodies (and
adds the record types) in this module — IMPLEMENTATION.md §1 places the
§2.1 invariants in `ingestion/manifest.py`, pure over mappings/paths passed
in, never assumed. The tests in tests/unit/test_manifest_schema.py,
tests/unit/test_licensing_invariants.py and
tests/integration/test_make_corpus.py pin the behaviour; their docstrings
are the contract.

Design sources: DESIGN.md §2.1 (invariants, as amended), §2.4; ADR-023
(no dataset files in git + sha256 verification of fetched files); review
findings #45/#46 (licence claims require evidence; multi-origin datasets
carry per-segment provenance and credit).

Conventions the tests rely on:

- Refusals raise :class:`ManifestError`. A refusal message names the
  offending document/dataset/pair id and every violated field — report all
  violations found for an entry, not just the first.
- Loaded records expose the manifest fields as attributes (dataclasses),
  typed: bools are bools, dates are ``datetime.date``, ``human_signoff``
  is a record with ``who``/``date``/``note`` attributes, provenance
  segments are records with ``origin``/``period``/``licence``/
  ``licence_evidence``/``credit`` attributes.
- Loaders tolerate unknown top-level keys (``version``, ``access_date``,
  and the fixtures-only ``violations`` section are not entries).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Values a *corpus document* may carry (DESIGN.md §2.1 — exactly three).
DOCUMENT_PERMITTED_CONTEXTS = frozenset(
    {"open", "non-commercial-educational", "permission-on-file"}
)

#: Values a *dataset* may carry. Datasets additionally admit
#: ``open-provisional`` (ADR-023 / review #45): an unconfirmed open verdict
#: that requires a ``licence_note`` recording the evidence trail and is
#: never allowed in the chart data pack.
DATASET_PERMITTED_CONTEXTS = frozenset(DOCUMENT_PERMITTED_CONTEXTS | {"open-provisional"})

#: First-line marker that exempts a committed data-like file from the
#: ADR-023 no-dataset-files-in-git check (same marker the #24 fixture
#: corpus uses; match on this substring, tolerating a BOM and either dash).
SYNTHETIC_FIXTURE_MARKER = "SYNTHETIC FIXTURE"


class ManifestError(ValueError):
    """A licensing-invariant violation. The message names the offending

    document/dataset/pair id and every violated field (issue #5 acceptance
    criterion: refusal messages name the offending document and field).
    """


def load_corpus_manifest(path: Path) -> Any:
    """Load + validate a corpus manifest; return an object with ``.documents``.

    ``.documents`` is a list of typed document records (attribute access).
    Any entry violating a §2.1 invariant refuses the whole load with
    :class:`ManifestError`.
    """
    raise NotImplementedError("issue #5: implement load_corpus_manifest in ingestion/manifest.py")


def validate_document(entry: Mapping[str, Any]) -> Any:
    """Validate one corpus-manifest document entry; return its typed record.

    Pure over the mapping. Raises :class:`ManifestError` naming the entry
    id and every violated field. This is the build gate: indexing calls
    this (via load_corpus_manifest) before any document is ingested.
    """
    raise NotImplementedError("issue #5: implement validate_document in ingestion/manifest.py")


def load_dataset_manifest(path: Path) -> Any:
    """Load + validate a dataset manifest; return an object with

    ``.datasets`` (mapping id -> typed record) and ``.splice_pairs``
    (list of typed pair records). Refuses invalid entries/pairs with
    :class:`ManifestError`.
    """
    raise NotImplementedError("issue #5: implement load_dataset_manifest in ingestion/manifest.py")


def validate_dataset(entry: Mapping[str, Any]) -> Any:
    """Validate one dataset entry (id included in the mapping); return its

    typed record. Pure. Raises :class:`ManifestError` naming the dataset
    id and every violated field.
    """
    raise NotImplementedError("issue #5: implement validate_dataset in ingestion/manifest.py")


def validate_splice_pair(pair: Mapping[str, Any]) -> Any:
    """Validate one splice-pair entry; return its typed record.

    ADR-020: the rebaseline decision is fixed in the manifest — a pair
    must carry either an explicit ``rebaseline: null`` or a rebaseline
    block with ``alignment_period_ce``; an absent ``rebaseline`` key
    refuses with :class:`ManifestError`.
    """
    raise NotImplementedError("issue #5: implement validate_splice_pair in ingestion/manifest.py")


def check_prepared_text_shipping(documents: Iterable[Mapping[str, Any]], corpus_dir: Path) -> None:
    """The repo-shipping check for corpus text (DESIGN §2.1, runs in CI's

    unit stage): raise :class:`ManifestError` if prepared text exists under
    ``corpus_dir`` for any document whose ``permitted_context`` is not
    ``open``; return None when clean.
    """
    raise NotImplementedError(
        "issue #5: implement check_prepared_text_shipping in ingestion/manifest.py"
    )


def find_committed_data_files(repo_root: Path, tracked_files: Iterable[str]) -> list[str]:
    """The ADR-023 no-dataset-files-in-git check (runs in CI's unit stage).

    Given the repo root and git-tracked relative paths, return the tracked
    data-like files (at minimum: .csv/.tsv/.txt/.nc/.parquet) whose first
    line does not contain :data:`SYNTHETIC_FIXTURE_MARKER`. An empty list
    means the tree is clean.
    """
    raise NotImplementedError(
        "issue #5: implement find_committed_data_files in ingestion/manifest.py"
    )


def verify_fetched_sha256(entry_id: str, path: Path, expected_sha256: str) -> None:
    """Verify a fetched file against its manifest-pinned hash (ADR-023).

    Return None on a match; raise :class:`ManifestError` naming
    ``entry_id`` and ``sha256`` on a mismatch or missing file.
    """
    raise NotImplementedError("issue #5: implement verify_fetched_sha256 in ingestion/manifest.py")
