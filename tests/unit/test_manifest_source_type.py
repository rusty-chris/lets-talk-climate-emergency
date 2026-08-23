"""Ingestion-side ``source_type`` enum enforcement (issue #158, reopened
ingestion half).

RED phase. PR #170 landed the indexer/query half: ``rag.indexing``
refuses any chunk or filter value outside the closed vocabulary
``{"evidence", "voices"}``, exact and case-sensitive. This suite pins the
remaining half: the same closed-enum check at **manifest load** —
``ingestion.manifest.validate_document`` / ``load_corpus_manifest``, the
exact choke point that already enforces ``human_signoff`` and
``permitted_context`` — so a bad value fails at authoring time and never
reaches the pipeline or the index.

Contract pinned here:

- ``SOURCE_TYPES`` has ONE declaration for the whole repo, now homed in
  ``ingestion.manifest`` (dependency-light; ``rag.indexing`` already sits
  downstream of ``ingestion``), and ``rag.indexing.SOURCE_TYPES`` is the
  SAME object — a re-export, never a second copy (the same
  non-duplication pin the UI red phase put on STARTER_QUESTIONS).
- A document whose ``source_type`` is missing, null, non-string, unknown,
  miscased, or whitespace-padded REFUSES with :class:`ManifestError` —
  never normalised, never defaulted (the §2.1 fail-closed philosophy: the
  #11 voices filter is an exact include-list over this field, so any
  value that isn't an exact member would silently escape it). NOTE: this
  is deliberately stricter than the #170 comment's "missing may default
  to evidence" — every fixture and both real corpus entries already carry
  the field explicitly, so the default has no remaining legitimate user
  at the manifest layer.
- Refusal messages name the offending document id, the field, the
  offending value, and the full allowed vocabulary; violations aggregate
  with other §2.1 violations per the module's convention.
- ``chunk_document`` — which consumes raw entries directly in unit use —
  double-checks explicit values the same way ``ingest_profile`` is
  double-checked (review #142 pattern).
- Both ACTIVE entries of the real ``corpus/manifest.yaml`` pass: the
  real manifest stays loadable, with every record's ``source_type`` a
  validated member of the vocabulary.

All inputs are the shared synthetic fixtures or invented inline metadata;
no document text, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import rag.indexing
from ingestion.manifest import (
    SOURCE_TYPES,
    ManifestError,
    load_corpus_manifest,
    validate_document,
)
from ingestion.pipeline import IngestError, chunk_document
from tests._ingestion_fixtures import doc, heading, manifest_entry, sentence, text

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CORPUS_MANIFEST = REPO_ROOT / "corpus" / "manifest.yaml"


def _assert_refusal_names(entry, *fragments):
    """``validate_document(entry)`` must raise :class:`ManifestError`

    whose message contains every fragment — checked individually so a
    failure names exactly which one is missing (the shared
    assert_refusal_names convention of test_licensing_invariants.py).
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, f"refusal must name {fragment!r}; got: {message}"


# ---------------------------------------------------------------------------
# Single source of truth (non-duplication pin)
# ---------------------------------------------------------------------------


def test_source_type_vocabulary_has_a_single_declaration():
    """``rag.indexing.SOURCE_TYPES`` must be the very object declared in

    ``ingestion.manifest`` — identity, not equality. Two frozensets that
    happen to be equal today are exactly how the vocabularies drift apart
    tomorrow (one layer extended, the other silently letting the new
    value through or refusing it — either way the #158 defect reopens).
    """
    assert rag.indexing.SOURCE_TYPES is SOURCE_TYPES, (
        "SOURCE_TYPES is declared twice — rag.indexing must re-export "
        "ingestion.manifest.SOURCE_TYPES, never carry its own copy"
    )


def test_source_type_vocabulary_is_exactly_evidence_and_voices():
    """DESIGN §2.5/§3.2: exactly two source layers. Widening the enum is a

    deliberate act (with a matching #11 filter policy), never a drive-by.
    """
    assert SOURCE_TYPES == frozenset({"evidence", "voices"})


# ---------------------------------------------------------------------------
# The happy path: valid members load, typed, onto the record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", sorted(SOURCE_TYPES))
def test_valid_source_type_loads_onto_the_typed_record(value):
    """A valid member passes validation and lands on the returned

    :class:`DocumentRecord` as the validated string — downstream layers
    read the typed record, never re-fish the raw YAML.
    """
    record = validate_document(manifest_entry("syn-source-ok", source_type=value))
    assert record.source_type == value


# ---------------------------------------------------------------------------
# Refusal paths: missing / null / non-string / unknown / miscased
# ---------------------------------------------------------------------------


def test_missing_source_type_refuses():
    """No silent layer assignment: an entry that never says which layer it

    belongs to refuses by name, listing the allowed vocabulary. (A voices
    document omitting the field is exactly the DESIGN §2.5 failure — it
    would otherwise default into the evidence corpus past the #11
    exclusion.)
    """
    entry = manifest_entry("syn-source-missing")
    del entry["source_type"]
    _assert_refusal_names(entry, "syn-source-missing", "source_type", "evidence", "voices")


def test_null_source_type_refuses():
    """An explicit ``source_type: null`` is not a decision either."""
    entry = manifest_entry("syn-source-null", source_type=None)
    _assert_refusal_names(entry, "syn-source-null", "source_type", "evidence", "voices")


@pytest.mark.parametrize(
    "value",
    [
        ["voices", "evidence"],  # a YAML list — one document, one layer
        True,
        7,
        {"layer": "voices"},
    ],
    ids=["list", "bool", "int", "mapping"],
)
def test_non_string_source_type_refuses(value):
    """The field is a single string, never a collection or a scalar of

    another type — refuse by name (the review #82 convention: a
    wrong-typed field is a naming refusal, never a traceback), naming the
    offending value.
    """
    entry = manifest_entry("syn-source-nonstring", source_type=value)
    _assert_refusal_names(
        entry, "syn-source-nonstring", "source_type", repr(value), "evidence", "voices"
    )


@pytest.mark.parametrize(
    "value",
    ["opinions", "", "evidence corpus", "voice"],
    ids=["unknown-word", "empty-string", "two-words", "near-miss"],
)
def test_unknown_source_type_refuses(value):
    """Any value outside the closed vocabulary refuses — including the

    explicit empty string and near-misses. Arbitrary manifest strings
    never extend the enum.
    """
    entry = manifest_entry("syn-source-unknown", source_type=value)
    _assert_refusal_names(
        entry, "syn-source-unknown", "source_type", repr(value), "evidence", "voices"
    )


@pytest.mark.parametrize(
    "value",
    ["Voices", "EVIDENCE", "Evidence", " voices ", "voices\n"],
    ids=["titlecase", "uppercase", "capitalised", "padded", "trailing-newline"],
)
def test_case_or_whitespace_variant_refuses_never_normalises(value):
    """Case variants and padded values are REJECTED, not normalised —

    matching the indexer/query half exactly. Normalising here would be
    actively dangerous: ``chunk_document`` reads the raw entry, so a
    manifest that quietly accepted ``" voices "`` would still propagate
    the raw padded value onto every chunk payload, where the exact-match
    #11 include-list — and the #170 indexer refusal — see a foreign
    string. One law, both layers: exact members only.
    """
    entry = manifest_entry("syn-source-miscased", source_type=value)
    _assert_refusal_names(
        entry, "syn-source-miscased", "source_type", repr(value), "evidence", "voices"
    )


def test_bad_source_type_aggregates_with_other_violations():
    """Module convention (issue #5): ALL violations for an entry are

    aggregated into one refusal, not just the first found — a bad
    ``source_type`` must not mask, or be masked by, a licensing
    violation.
    """
    entry = manifest_entry("syn-source-aggregate", source_type="opinions")
    del entry["licence"]
    _assert_refusal_names(entry, "syn-source-aggregate", "source_type", "'opinions'", "licence")


# ---------------------------------------------------------------------------
# The choke point: load_corpus_manifest refuses the whole load
# ---------------------------------------------------------------------------


def test_load_corpus_manifest_refuses_whole_load_on_bad_source_type(tmp_path):
    """One bad entry refuses the WHOLE manifest load — the same choke

    point that enforces ``human_signoff``: the pipeline consumes
    :func:`load_corpus_manifest` before any fetch, so nothing downstream
    ever sees the bad value.
    """
    manifest = {
        "documents": [
            manifest_entry("syn-load-good"),
            manifest_entry("syn-load-bad", source_type="Voices"),
        ]
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        load_corpus_manifest(path)
    message = str(excinfo.value)
    assert "syn-load-bad" in message
    assert "source_type" in message
    assert "'Voices'" in message


# ---------------------------------------------------------------------------
# Chunker double-check (the review #142 pattern for raw-entry unit use)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["Voices", "opinions", ["voices"]], ids=["case", "unknown", "list"]
)
def test_chunker_refuses_explicit_unknown_source_type(value):
    """``chunk_document`` consumes raw manifest entries directly in unit

    use, so — exactly like ``ingest_profile`` (review #142) — an explicit
    value outside the vocabulary refuses fail-closed here too, before a
    single chunk carries it toward the index.
    """
    two_block_doc = doc(
        "syn-chunk-badtype",
        [heading("1 Observed changes"), text(sentence(12, "srctype"))],
    )
    entry = manifest_entry("syn-chunk-badtype", source_type=value)
    with pytest.raises(IngestError) as excinfo:
        chunk_document(two_block_doc, entry)
    message = str(excinfo.value)
    assert "syn-chunk-badtype" in message
    assert "source_type" in message


# ---------------------------------------------------------------------------
# The real manifest stays loadable
# ---------------------------------------------------------------------------


def test_real_corpus_manifest_passes_with_explicit_source_types():
    """Every ACTIVE entry of the repo's real corpus manifest carries an

    explicit, valid ``source_type`` and the manifest loads under the new
    check — enforcement must land without breaking the real corpus.
    """
    raw = yaml.safe_load(REAL_CORPUS_MANIFEST.read_text(encoding="utf-8"))
    for entry in raw["documents"]:
        assert "source_type" in entry, (
            f"real corpus entry {entry.get('id')!r} must declare source_type explicitly"
        )

    manifest = load_corpus_manifest(REAL_CORPUS_MANIFEST)
    assert len(manifest.documents) >= 2
    for record in manifest.documents:
        assert record.source_type in SOURCE_TYPES, (
            f"{record.id}: loaded record must carry a validated source_type, "
            f"got {record.source_type!r}"
        )
