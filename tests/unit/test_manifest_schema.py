"""Manifest schema: parsing + typing (issue #5, TDD plan items 1 and 10).

RED phase for `ingestion/manifest.py` (IMPLEMENTATION.md §1: pure
load/validate of the corpus and dataset manifests, paths passed in, never
assumed). These tests pin the *shape* contract — a fully populated entry
loads with every DESIGN.md §2.1 field typed, defaults apply, and the
dataset schema supports the structures review findings #45/#46 require
(open-provisional with licence_note; per-segment provenance for
multi-origin records like Mauna Loa). The refusal paths live in
tests/unit/test_licensing_invariants.py.

Everything runs against the synthetic fixture manifests
(tests/fixtures/corpus/manifest.yaml, tests/fixtures/datasets/manifest.yaml)
plus the repo's real datasets/manifest.yaml — all invented or
already-committed metadata, no document text, no network.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from ingestion.manifest import (
    load_corpus_manifest,
    load_dataset_manifest,
    validate_document,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _by_id(records):
    return {record.id: record for record in records}


# ---------------------------------------------------------------------------
# Corpus manifest (TDD plan item 1)
# ---------------------------------------------------------------------------


def test_manifest_parses_minimal_valid_entry(fixture_manifest_path):
    """TDD plan item 1: a fully populated `open` entry loads with all §2.1

    fields typed — strings are strings, `redistributable` a real bool,
    dates real dates, `human_signoff` a record with who/date/note.
    """
    manifest = load_corpus_manifest(fixture_manifest_path)
    doc = _by_id(manifest.documents)["syn-basin-assessment"]

    assert doc.permitted_context == "open"
    assert doc.licence.startswith("Public domain")
    assert doc.licence_evidence and isinstance(doc.licence_evidence, str)
    assert doc.attribution_text and isinstance(doc.attribution_text, str)
    assert doc.canonical_url.startswith("https://")
    assert doc.redistributable is True
    assert doc.consensus_position == "assessed"
    assert isinstance(doc.sha256, str) and len(doc.sha256) == 64
    assert int(doc.sha256, 16) is not None  # hex-parseable
    assert doc.retrieved_at == datetime.date(2026, 8, 16)
    assert doc.source_tier == "A"
    assert doc.human_signoff.who == "issue #24 fixture author"
    assert doc.human_signoff.date == datetime.date(2026, 8, 16)
    assert doc.human_signoff.note


def test_manifest_loads_every_valid_document(fixture_manifest_path):
    """The loader returns one typed record per `documents` entry (the

    fixtures-only `violations` section and unknown top-level keys are not
    documents), preserving ids — including the manifest-only non-open
    entries that ship no prepared text.
    """
    manifest = load_corpus_manifest(fixture_manifest_path)
    ids = set(_by_id(manifest.documents))
    assert {"syn-basin-assessment", "syn-nc-gap-report", "syn-permitted-synthesis"} <= ids
    assert not any(i.startswith("bad-") for i in ids), (
        "violation entries must never be loaded as documents"
    )


def test_manifest_only_entries_carry_no_path(fixture_manifest_path):
    """Non-open entries are manifest-only records: `path` is None and the

    loader accepts them (their text is fetched at ingest time, never
    committed — DESIGN.md §2.1 ship rule).
    """
    manifest = load_corpus_manifest(fixture_manifest_path)
    docs = _by_id(manifest.documents)
    assert docs["syn-nc-gap-report"].permitted_context == "non-commercial-educational"
    assert docs["syn-nc-gap-report"].path is None
    assert docs["syn-permitted-synthesis"].permitted_context == "permission-on-file"
    assert docs["syn-permitted-synthesis"].path is None


def test_consensus_position_defaults_to_assessed():
    """TDD plan item 10: an entry omitting `consensus_position` validates

    and defaults to "assessed" (DESIGN.md §2.1).
    """
    entry = {
        # Inline synthetic entry — invented metadata only, no document text.
        "id": "syn-inline-default",
        "licence": "CC BY 4.0 (invented)",
        "licence_evidence": "Invented licence statement captured 2026-08-16",
        "attribution_text": "Inline Fixture Author (fictional)",
        "canonical_url": "https://doi.example.invalid/10.0000/inline.1",
        "redistributable": True,
        "permitted_context": "open",
        "permission_evidence": None,
        "sha256": "ab" * 32,
        "retrieved_at": "2026-08-16",
        "source_tier": "A",
        "human_signoff": {"who": "fixture", "date": "2026-08-16", "note": "inline"},
    }
    record = validate_document(entry)
    assert record.consensus_position == "assessed"


def test_beyond_assessed_range_flag_round_trips(fixture_manifest_path):
    """`consensus_position: beyond-assessed-range` survives loading — the

    severity-skew guardrail (DESIGN.md §2.3) reads this flag downstream.
    """
    manifest = load_corpus_manifest(fixture_manifest_path)
    doc = _by_id(manifest.documents)["syn-beyond-assessed"]
    assert doc.consensus_position == "beyond-assessed-range"


# ---------------------------------------------------------------------------
# Dataset manifest (same schema discipline — issue #5 scope; reviews #45/#46)
# ---------------------------------------------------------------------------


def test_dataset_manifest_parses_typed_records(fixture_dataset_manifest_path):
    """The dataset loader returns typed records keyed by id, with the

    schema-discipline fields: licence + evidence, permitted_context,
    in_chart_pack as a real bool, sha256, attribution_text, human_signoff.
    """
    manifest = load_dataset_manifest(fixture_dataset_manifest_path)
    ds = manifest.datasets["syn-instrumental-co2"]
    assert ds.permitted_context == "open"
    assert ds.in_chart_pack is True
    assert ds.licence and ds.licence_evidence
    assert ds.attribution_text == "Windvane Observatory (fictional)"
    assert isinstance(ds.sha256, str) and len(ds.sha256) == 64
    assert ds.url.startswith("https://")
    assert ds.human_signoff.who == "issue #5 fixture author"
    assert set(manifest.datasets) == {
        "syn-instrumental-co2",
        "syn-paleo-co2",
        "syn-instrumental-temp",
        "syn-provisional-reconstruction",
        "syn-multi-origin-co2",
    }


def test_dataset_schema_accepts_open_provisional_with_licence_note(
    fixture_dataset_manifest_path,
):
    """`open-provisional` (the Kaufman/Bereiter verdict shape, review #45)

    is a valid dataset context when a licence_note records the evidence
    trail and the dataset stays out of the chart pack.
    """
    manifest = load_dataset_manifest(fixture_dataset_manifest_path)
    ds = manifest.datasets["syn-provisional-reconstruction"]
    assert ds.permitted_context == "open-provisional"
    assert ds.in_chart_pack is False
    assert "written confirmation" in ds.licence_note.lower()


def test_dataset_manifest_supports_per_segment_provenance(fixture_dataset_manifest_path):
    """Review #46: a multi-origin dataset (the Mauna Loa/Scripps shape)

    carries per-segment provenance records — each segment typed with
    origin, period, licence, licence_evidence and credit — and its
    attribution_text credits every segment.
    """
    manifest = load_dataset_manifest(fixture_dataset_manifest_path)
    ds = manifest.datasets["syn-multi-origin-co2"]
    assert len(ds.provenance) == 2
    for segment in ds.provenance:
        assert segment.origin
        assert segment.period
        assert segment.licence
        assert segment.licence_evidence
        assert segment.credit
        assert segment.credit in ds.attribution_text
    origins = {segment.origin for segment in ds.provenance}
    assert origins == {
        "Quill Institute Gas Program (fictional)",
        "Windvane Observatory (fictional)",
    }


def test_single_origin_dataset_needs_no_provenance(fixture_dataset_manifest_path):
    """Provenance segments are for multi-origin records; a plain

    single-origin dataset validates without any and exposes an empty
    segment list (not None-crashes downstream).
    """
    manifest = load_dataset_manifest(fixture_dataset_manifest_path)
    ds = manifest.datasets["syn-instrumental-temp"]
    assert list(ds.provenance) == []


def test_dataset_manifest_parses_both_splice_pair_forms(fixture_dataset_manifest_path):
    """ADR-020: both legal curation decisions load — an explicit

    `rebaseline: null` with rationale (absolute-scale case) and a
    rebaseline block whose alignment_period_ce is fixed in the manifest.
    """
    manifest = load_dataset_manifest(fixture_dataset_manifest_path)
    pairs = {pair.id: pair for pair in manifest.splice_pairs}

    co2 = pairs["syn-co2-splice"]
    assert co2.paleo == "syn-paleo-co2"
    assert co2.instrumental == "syn-instrumental-co2"
    assert co2.splice_year_ce == 1959
    assert co2.rebaseline is None
    assert co2.rationale

    temp = pairs["syn-temp-splice"]
    assert temp.rebaseline is not None
    assert temp.rebaseline.apply_to == "syn-instrumental-temp"
    assert list(temp.rebaseline.alignment_period_ce) == [1880, 1900]


def test_repo_dataset_manifest_passes_schema_validation():
    """The repo's real datasets/manifest.yaml validates under the schema.

    This is the enforcement that makes the review-#45/#46 sloppiness
    impossible in the shipped manifest: the current spike draft does NOT
    satisfy the schema (entries carry licence claims with no
    licence_evidence; Bereiter claims `open` though its archive metadata
    has no licence grant; the Mauna Loa entry records no per-segment
    provenance or Scripps credit; entries lack in_chart_pack,
    attribution_text and human_signoff). Making this pass means amending
    datasets/manifest.yaml per the evidence already recorded in findings
    #45/#46 and reviews/spike-04-chart-findings.md — record the facts
    those reviews established; do not invent new licence facts.
    """
    manifest = load_dataset_manifest(REPO_ROOT / "datasets" / "manifest.yaml")
    assert manifest.datasets, "the real dataset manifest must load non-empty"
    mlo = manifest.datasets["noaa_gml_co2_mlo"]
    # Review #46: the pre-May-1974 segment is Scripps/Keeling data with its
    # own terms — the manifest must say so, per segment, and credit it.
    assert len(mlo.provenance) >= 2, "Mauna Loa is multi-origin (NOAA GML + Scripps/Keeling)"
    assert any("scripps" in segment.origin.lower() for segment in mlo.provenance)
    assert any("scripps" in segment.credit.lower() for segment in mlo.provenance)
    scripps = next(s for s in mlo.provenance if "scripps" in s.origin.lower())
    assert "government" not in scripps.licence.lower(), (
        "review #46: the licence field must not claim US Government work "
        "for the Scripps-obtained segment"
    )
    # Review #45: Bereiter's open claim has no grant on file — it cannot
    # remain `open`; the Kaufman entry already models the provisional shape.
    bereiter = manifest.datasets["bereiter2015_co2"]
    assert bereiter.permitted_context == "open-provisional"
    assert bereiter.licence_note
