"""Licensing-invariant refusal paths (issue #5, TDD plan items 2-9).

RED phase for `ingestion/manifest.py` — the legal load-bearing wall
(DESIGN.md §2.1 as amended by ADR-023 and review findings #45/#46). Every
refusal path of the build gate has a named test asserting build failure
with a message naming the offending document/dataset id and the violated
field(s); refusals aggregate all violations found for an entry, so the
message-content assertions here hold however the implementer orders the
checks.

The violation inputs come from the deliberate per-invariant entries in
tests/fixtures/corpus/manifest.yaml (issue #24 infrastructure, extended on
this branch) — the validator is fed the same shared fixtures the meta-tests
guarantee, never improvised real-looking data. Inline entries below are
invented metadata only.

Ship-side checks (also here, unit tier per the issue's acceptance
criteria): the no-committed-text check keyed on permitted_context, the
ADR-023 no-dataset-files-in-git check, and sha256 verification of fetched
files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from ingestion.manifest import (
    PROJECT_OPERATIONAL_DATA_MARKER,
    SYNTHETIC_FIXTURE_MARKER,
    ManifestError,
    check_prepared_text_shipping,
    find_committed_data_files,
    load_corpus_manifest,
    load_dataset_manifest,
    validate_dataset,
    validate_document,
    validate_splice_pair,
    verify_fetched_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus" / "manifest.yaml"

# Loaded at import time so the refusal-message tests can parametrize over
# the shared violation entries (their presence/shape is guaranteed by
# tests/unit/test_fixture_corpus.py).
_FIXTURE_RAW = yaml.safe_load(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
_VIOLATIONS = {entry["violates"]: entry for entry in _FIXTURE_RAW["violations"]}

_DATASET_FIXTURE_RAW = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "datasets" / "manifest.yaml").read_text(encoding="utf-8")
)


def _valid_document_entry(entry_id: str) -> dict:
    """A copy of the fixture manifest's first valid open document, re-id'd
    for inline mutation (invented metadata only)."""
    return dict(_FIXTURE_RAW["documents"][0], id=entry_id)


def _valid_dataset_entry(entry_id: str) -> dict:
    """A copy of a valid fixture dataset entry, re-id'd for inline
    mutation (invented metadata only)."""
    return dict(_DATASET_FIXTURE_RAW["datasets"]["syn-instrumental-co2"], id=entry_id)


def _violation_document(invariant: str) -> dict:
    return _VIOLATIONS[invariant]["document"]


def _violation_dataset(invariant: str) -> dict:
    return _VIOLATIONS[invariant]["dataset"]


def assert_refusal_names(validator, entry, *fragments):
    """Assert ``validator(entry)`` raises :class:`ManifestError` whose

    message contains every fragment in ``*fragments`` (refactor #111,
    audit finding 2a) — the repeated four-line
    ``pytest.raises``/``str(excinfo.value)``/per-fragment ``assert in``
    tail collapsed to one call, with the same assertion strength: each
    fragment is still checked individually so a failure names exactly
    which one is missing.
    """
    with pytest.raises(ManifestError) as excinfo:
        validator(entry)
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, f"refusal must name {fragment!r}"


# ---------------------------------------------------------------------------
# The build gate: document refusal paths (TDD plan items 2-6)
# ---------------------------------------------------------------------------


def test_build_refuses_unset_permitted_context():
    """TDD plan item 2: no `permitted_context`, no indexing — the central

    re-keying of every invariant (DESIGN §2.1).
    """
    with pytest.raises(ManifestError, match="bad-unset-context"):
        validate_document(_violation_document("unset_permitted_context"))


def test_build_refuses_unknown_permitted_context_value():
    """TDD plan item 3: values outside open | non-commercial-educational |

    permission-on-file are rejected, not coerced or ignored.
    """
    with pytest.raises(ManifestError, match="bad-unknown-context"):
        validate_document(_violation_document("unknown_permitted_context_value"))


def test_build_refuses_empty_human_signoff():
    """TDD plan item 4 (absent case): no document is indexed without a

    named human sign-off (DESIGN §2.2 step 3).
    """
    with pytest.raises(ManifestError, match="bad-no-signoff"):
        validate_document(_violation_document("empty_human_signoff"))


def test_build_refuses_incomplete_human_signoff():
    """TDD plan item 4 (incomplete case, fixture added per review #70):

    a signoff of {who} without date/note is not a sign-off.
    """
    with pytest.raises(ManifestError, match="bad-partial-signoff"):
        validate_document(_violation_document("incomplete_human_signoff"))


def test_build_refuses_permission_on_file_without_evidence():
    """TDD plan item 5: `permission-on-file` is a claim about a letter on

    file — empty `permission_evidence` refuses.
    """
    with pytest.raises(ManifestError, match="bad-permission-no-evidence"):
        validate_document(_violation_document("permission_on_file_without_evidence"))


def test_build_refuses_missing_required_fields():
    """§2.1 field carriage (review #70): an otherwise-valid entry missing

    required fields (here licence, sha256, attribution_text, retrieved_at)
    refuses, naming every absent field.
    """
    assert_refusal_names(
        validate_document,
        _violation_document("missing_required_field"),
        "bad-missing-fields",
        "sha256",
        "attribution_text",
        "retrieved_at",
    )


def test_build_refuses_unknown_consensus_position_value():
    """Review #79: `consensus_position` is a closed enum (assessed |
    beyond-assessed-range, DESIGN §2.1). A typo'd value must refuse,
    naming the document, the field and the valid choices — fail-open
    coercion here silently disarms the §2.3 severity-skew guardrail for
    exactly the Hansen-style documents it exists for.
    """
    assert_refusal_names(
        validate_document,
        _violation_document("unknown_consensus_position"),
        "bad-unknown-consensus",
        "consensus_position",
        "assessed",
        "beyond-assessed-range",
    )


def test_build_refuses_empty_string_consensus_position():
    """Review #79: an explicit empty string is not 'unset' — it refuses
    rather than silently defaulting to assessed. (Inline entry: invented
    metadata only, modelled on the fixture's valid open entry.)
    """
    entry = dict(_VIOLATIONS["unknown_consensus_position"]["document"])
    entry["id"] = "syn-inline-empty-consensus"
    entry["consensus_position"] = ""
    assert_refusal_names(
        validate_document, entry, "syn-inline-empty-consensus", "consensus_position"
    )


def test_build_refuses_unknown_ingest_profile_value():
    """Review #142 — the #79 defect class re-introduced on a new field:
    the entire Tier C headline-statements protection keys on the exact
    string 'headline-statements' in ``ingest_profile``, and the failure
    direction of a typo is OPEN (the document ingests as ordinary
    evidence with the feature flag ignored). The field is a closed enum:
    absent | 'headline-statements'; anything else refuses naming the
    document, the field and the valid values. (Inline entry: invented
    metadata only.)
    """
    from tests._ingestion_fixtures import manifest_entry

    entry = manifest_entry("syn-bad-profile", ingest_profile="headline_statements")
    assert_refusal_names(
        validate_document,
        entry,
        "syn-bad-profile",
        "ingest_profile",
        "headline-statements",
    )


def test_valid_ingest_profile_round_trips_and_absent_is_none():
    """The closed enum's positive arm: 'headline-statements' loads onto
    the typed record (so the pipeline can key skips off validated data,
    not raw YAML), and an absent field reads None."""
    from tests._ingestion_fixtures import manifest_entry

    with_profile = validate_document(
        manifest_entry("syn-good-profile", ingest_profile="headline-statements")
    )
    assert with_profile.ingest_profile == "headline-statements"
    without_profile = validate_document(manifest_entry("syn-no-profile"))
    assert without_profile.ingest_profile is None


def test_build_refuses_licence_claim_without_evidence():
    """Reviews #45/#46 at corpus level: a licence claim requires a

    non-empty `licence_evidence` naming its source; an empty string is as
    refused as an absent field.
    """
    with pytest.raises(ManifestError, match="bad-licence-no-evidence"):
        validate_document(_violation_document("licence_claim_without_evidence"))


_DOCUMENT_REFUSAL_FIELDS = [
    ("unset_permitted_context", "bad-unset-context", "permitted_context"),
    ("unknown_permitted_context_value", "bad-unknown-context", "permitted_context"),
    ("empty_human_signoff", "bad-no-signoff", "human_signoff"),
    ("incomplete_human_signoff", "bad-partial-signoff", "human_signoff"),
    ("permission_on_file_without_evidence", "bad-permission-no-evidence", "permission_evidence"),
    ("licence_claim_without_evidence", "bad-licence-no-evidence", "licence_evidence"),
    ("unknown_consensus_position", "bad-unknown-consensus", "consensus_position"),
]


@pytest.mark.parametrize(
    ("invariant", "doc_id", "field"),
    _DOCUMENT_REFUSAL_FIELDS,
    ids=[row[0] for row in _DOCUMENT_REFUSAL_FIELDS],
)
def test_refusal_messages_name_document_and_field(invariant, doc_id, field):
    """TDD plan item 6 / acceptance criterion: every refusal names the

    offending document id and the violated field in its message.
    """
    assert_refusal_names(validate_document, _violation_document(invariant), doc_id, field)


def test_loading_manifest_with_violating_entry_refuses(tmp_path):
    """The gate holds at manifest level, not only per entry: a manifest

    containing one bad document among valid ones refuses to load, naming
    the bad one. Indexing consumes load_corpus_manifest, so this is what
    'the build refuses to index' means in code.
    """
    good = yaml.safe_load(FIXTURE_MANIFEST.read_text(encoding="utf-8"))["documents"][0]
    bad = _violation_document("unset_permitted_context")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump({"documents": [good, bad]}, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="bad-unset-context"):
        load_corpus_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Lexical strictness (review #82): the gate's guarantees must be semantic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["licence", "licence_evidence", "attribution_text", "canonical_url", "source_tier"]
)
def test_document_whitespace_only_value_refuses(field):
    """Review #82: a whitespace-only string satisfies no evidence or
    carriage invariant — ' ' is as refused as '' or absent, for every
    required document string field.
    """
    entry = _valid_document_entry("syn-inline-ws-doc")
    entry[field] = "   "
    assert_refusal_names(validate_document, entry, "syn-inline-ws-doc", field)


def test_document_whitespace_only_permission_evidence_refuses():
    """Review #82 x Tier-C: a single space is not a letter on file."""
    entry = dict(
        next(
            d for d in _FIXTURE_RAW["documents"] if d["permitted_context"] == "permission-on-file"
        ),
        id="syn-inline-ws-permission",
    )
    entry["permission_evidence"] = " "
    assert_refusal_names(
        validate_document, entry, "syn-inline-ws-permission", "permission_evidence"
    )


@pytest.mark.parametrize("subfield", ["who", "note"])
def test_document_whitespace_only_signoff_subfield_refuses(subfield):
    """Review #82: a sign-off whose who/note is whitespace names nobody."""
    entry = _valid_document_entry("syn-inline-ws-signoff")
    entry["human_signoff"] = dict(entry["human_signoff"], **{subfield: " \t"})
    assert_refusal_names(validate_document, entry, "syn-inline-ws-signoff", "human_signoff")


@pytest.mark.parametrize("field", ["licence", "licence_evidence", "attribution_text", "url"])
def test_dataset_whitespace_only_value_refuses(field):
    """Review #82, dataset side: the same strip-before-check discipline."""
    entry = _valid_dataset_entry("syn-inline-ws-dataset")
    entry[field] = " "
    assert_refusal_names(validate_dataset, entry, "syn-inline-ws-dataset", field)


@pytest.mark.parametrize("bad_sha", ["TBD", "0" * 63, "G" * 64, "0" * 64 + "0", "0X" * 32])
def test_sha256_must_be_64_hex(bad_sha):
    """Review #82: the pin says *which bytes* (ADR-023) — a placeholder
    like 'TBD' pins nothing and must refuse at validation time, not at
    (possibly never-reached) fetch time.
    """
    entry = _valid_document_entry("syn-inline-bad-sha")
    entry["sha256"] = bad_sha
    assert_refusal_names(validate_document, entry, "syn-inline-bad-sha", "sha256")


def test_document_id_required():
    """Review #82: an entry without an id cannot be named by any refusal
    and two anonymous records could coexist — id is required.
    """
    entry = _valid_document_entry("syn-inline-anonymous")
    del entry["id"]
    with pytest.raises(ManifestError, match="id"):
        validate_document(entry)


def test_dataset_id_required():
    """Dataset side of review #82's id requirement."""
    entry = _valid_dataset_entry("syn-inline-anonymous")
    del entry["id"]
    with pytest.raises(ManifestError, match="id"):
        validate_dataset(entry)


def test_document_redistributable_must_be_boolean():
    """Review #82: redistributable: 'false' (a quoted string) coerced to
    True under bool() — the exact wrong-way flip for a legal flag. Bools
    must be real bools.
    """
    entry = _valid_document_entry("syn-inline-stringly-bool")
    entry["redistributable"] = "false"
    assert_refusal_names(validate_document, entry, "syn-inline-stringly-bool", "redistributable")


def test_dataset_in_chart_pack_must_be_boolean():
    """The dataset-side boolean, same discipline: 'true' as a string is a
    type error, not pack membership.
    """
    entry = _valid_dataset_entry("syn-inline-stringly-pack")
    entry["in_chart_pack"] = "false"
    assert_refusal_names(validate_dataset, entry, "syn-inline-stringly-pack", "in_chart_pack")


def test_malformed_signoff_type_refuses_with_manifest_error():
    """Review #82: human_signoff: 'signed' must produce a ManifestError
    naming the document and field — never an AttributeError traceback.
    """
    entry = _valid_document_entry("syn-inline-signoff-string")
    entry["human_signoff"] = "signed"
    assert_refusal_names(validate_document, entry, "syn-inline-signoff-string", "human_signoff")


def test_malformed_provenance_type_refuses_with_manifest_error():
    """A provenance given as a string (or with string segments) is a
    naming refusal, not an AttributeError.
    """
    entry = _valid_dataset_entry("syn-inline-provenance-string")
    entry["provenance"] = "all fine, trust me"
    assert_refusal_names(validate_dataset, entry, "syn-inline-provenance-string", "provenance")

    entry = _valid_dataset_entry("syn-inline-segment-string")
    entry["provenance"] = ["all fine, trust me"]
    assert_refusal_names(validate_dataset, entry, "syn-inline-segment-string", "provenance")


def test_malformed_rebaseline_type_refuses_with_manifest_error():
    """A rebaseline given as the *string* 'null' (a classic YAML slip) is
    a naming refusal, not an AttributeError.
    """
    pair = {
        "id": "syn-inline-rebaseline-string",
        "paleo": "syn-paleo-co2",
        "instrumental": "syn-instrumental-co2",
        "splice_year_ce": 1959,
        "rationale": "Invented.",
        "rebaseline": "null",
    }
    assert_refusal_names(validate_splice_pair, pair, "syn-inline-rebaseline-string", "rebaseline")


# ---------------------------------------------------------------------------
# Repo-shipping check: no committed text for non-open documents (item 7)
# ---------------------------------------------------------------------------


def test_ship_check_fails_on_nonopen_prepared_text(fixture_corpus_dir):
    """TDD plan item 7 / acceptance criterion: the mis-tiered fixture — a

    non-commercial-educational document whose `path` points at prepared
    text committed in-repo — fails the ship check, naming document and
    file. Runs in CI's unit stage.
    """
    bad = _violation_document("nonopen_prepared_text_in_repo")
    with pytest.raises(ManifestError) as excinfo:
        check_prepared_text_shipping([bad], fixture_corpus_dir)
    message = str(excinfo.value)
    assert "bad-nc-prepared-text" in message
    assert "synthetic_indicator_table.md" in message


def test_ship_check_passes_open_prepared_text(fixture_corpus_dir, fixture_manifest_path):
    """Control for item 7: the valid fixture documents — open entries with

    committed text, non-open entries manifest-only (path None) — pass the
    ship check.
    """
    documents = yaml.safe_load(fixture_manifest_path.read_text(encoding="utf-8"))["documents"]
    assert check_prepared_text_shipping(documents, fixture_corpus_dir) is None


# ---------------------------------------------------------------------------
# Dataset invariants (items 8-9, amended by ADR-023 and reviews #45/#46)
# ---------------------------------------------------------------------------


def test_dataset_pack_rejects_nonopen_dataset():
    """TDD plan item 8: a non-open dataset flagged for the chart pack

    refuses — exported chart images travel into arbitrary (including
    commercial) contexts (DESIGN §2.1).
    """
    with pytest.raises(ManifestError, match="bad-nc-pack-dataset"):
        validate_dataset(_violation_dataset("nonopen_dataset_in_chart_pack"))


def test_dataset_pack_rejects_open_provisional_dataset():
    """The pack requires *confirmed* open: an open-provisional verdict

    (the Kaufman/Bereiter shape, review #45) with `in_chart_pack: true`
    refuses even though its licence_note is in order.
    """
    with pytest.raises(ManifestError, match="bad-provisional-in-pack"):
        validate_dataset(_violation_dataset("provisional_dataset_in_chart_pack"))


def test_open_dataset_requires_licence_evidence():
    """Review #45 (Bereiter): a dataset cannot claim `open` with no

    licence evidence on file — the schema refuses the unbacked claim.
    """
    with pytest.raises(ManifestError, match="bad-open-no-evidence"):
        validate_dataset(_violation_dataset("open_dataset_without_licence_evidence"))


def test_open_provisional_requires_licence_note():
    """`open-provisional` without a licence_note recording the evidence

    trail is just an unbacked claim with a softer name — refused. (Single
    mutation of the valid fixture entry: invented metadata only.)
    """
    entry = _valid_dataset_entry("syn-inline-provisional-bare")
    entry["permitted_context"] = "open-provisional"
    entry["in_chart_pack"] = False
    assert_refusal_names(validate_dataset, entry, "syn-inline-provisional-bare", "licence_note")


def test_dataset_permission_on_file_requires_permission_evidence():
    """Review #78: the Tier-C rule holds for datasets exactly as for
    documents — `permission-on-file` is a claim about a letter on file,
    and a dataset entry making it with empty `permission_evidence`
    refuses, naming the dataset and the field.
    """
    assert_refusal_names(
        validate_dataset,
        _violation_dataset("dataset_permission_on_file_without_evidence"),
        "bad-dataset-permission-no-evidence",
        "permission_evidence",
    )


def test_dataset_requires_retrieved_at():
    """Review #78 / DESIGN §3: datasets carry the §2.1 manifest fields;
    with ADR-023's fetch-at-build model, `retrieved_at` records when the
    pinned bytes were captured and is required. (Single mutation of the
    valid fixture entry: invented metadata only.)
    """
    entry = _valid_dataset_entry("syn-inline-no-retrieved-at")
    del entry["retrieved_at"]
    assert_refusal_names(validate_dataset, entry, "syn-inline-no-retrieved-at", "retrieved_at")


def test_dataset_rejects_unknown_permitted_context():
    """The dataset enum is open | open-provisional |

    non-commercial-educational | permission-on-file; anything else
    refuses. (Single mutation of the valid fixture entry: invented
    metadata only.)
    """
    entry = _valid_dataset_entry("syn-inline-bad-context")
    entry["permitted_context"] = "basically-fine"
    entry["in_chart_pack"] = False
    assert_refusal_names(validate_dataset, entry, "syn-inline-bad-context", "permitted_context")


def test_dataset_manifest_requires_alignment_periods_for_splice_pairs():
    """TDD plan item 9 (as amended): a splice pair with no rebaseline

    decision recorded at all — neither an explicit `rebaseline: null` nor
    a block with alignment_period_ce — refuses; the alignment decision is
    curation-time and manifest-fixed (ADR-020), never left open for the
    LLM or the renderer to improvise.
    """
    pair = _VIOLATIONS["splice_pair_without_alignment_periods"]["splice_pair"]
    with pytest.raises(ManifestError) as excinfo:
        validate_splice_pair(pair)
    message = str(excinfo.value)
    assert "bad-splice-no-alignment" in message
    assert "rebaseline" in message


def _dataset_manifest_with_pairs(tmp_path, pairs: list[dict]) -> Path:
    """A temp dataset manifest reusing the fixture's valid datasets with
    the given splice pairs (invented metadata only)."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump(
            {"datasets": _DATASET_FIXTURE_RAW["datasets"], "splice_pairs": pairs},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_splice_pair_refuses_unknown_dataset_reference(tmp_path):
    """Review #84: a pair naming dataset ids absent from the manifest must
    refuse at load time, naming the pair and the dangling reference —
    otherwise a renamed dataset id silently strands the ADR-020 decision
    and the failure surfaces as a KeyError far from the cause (or an
    LLM-side improvisation fills the gap).
    """
    pair = {
        "id": "syn-dangling-splice",
        "paleo": "syn-paleo-co2-renamed",  # not a dataset id in the manifest
        "instrumental": "syn-instrumental-co2",
        "splice_year_ce": 1959,
        "rationale": "Invented.",
        "rebaseline": None,
    }
    manifest_path = _dataset_manifest_with_pairs(tmp_path, [pair])
    with pytest.raises(ManifestError) as excinfo:
        load_dataset_manifest(manifest_path)
    message = str(excinfo.value)
    assert "syn-dangling-splice" in message
    assert "syn-paleo-co2-renamed" in message


def test_rebaseline_apply_to_must_be_pair_member(tmp_path):
    """Review #84: rebaseline.apply_to must name one of the pair's two
    datasets — shifting a series that is not in the pair is not a legal
    reading of the manifest-fixed decision.
    """
    pair = {
        "id": "syn-misapplied-rebaseline",
        "paleo": "syn-provisional-reconstruction",
        "instrumental": "syn-instrumental-temp",
        "splice_year_ce": 1880,
        "rationale": "Invented.",
        "rebaseline": {
            "apply_to": "syn-instrumental-co2",  # a real dataset, but not a pair member
            "alignment_period_ce": [1880, 1900],
        },
    }
    manifest_path = _dataset_manifest_with_pairs(tmp_path, [pair])
    with pytest.raises(ManifestError) as excinfo:
        load_dataset_manifest(manifest_path)
    message = str(excinfo.value)
    assert "syn-misapplied-rebaseline" in message
    assert "apply_to" in message


def test_null_rebaseline_requires_rationale():
    """Review #84: the explicit `rebaseline: null` (absolute-scale) form
    documents a decision, and the fixture manifest and schema tests have
    always described it as 'explicit null with rationale' — the code now
    enforces the documented contract.
    """
    pair = {
        "id": "syn-null-no-rationale",
        "paleo": "syn-paleo-co2",
        "instrumental": "syn-instrumental-co2",
        "splice_year_ce": 1959,
        "rebaseline": None,
        # no rationale on purpose
    }
    with pytest.raises(ManifestError) as excinfo:
        validate_splice_pair(pair)
    message = str(excinfo.value)
    assert "syn-null-no-rationale" in message
    assert "rationale" in message


def test_provenance_segment_requires_licence_evidence():
    """Review #46 (Mauna Loa/Scripps): every provenance segment of a

    multi-origin dataset carries its own licence evidence; the refusal
    names the dataset, the field and the offending segment's origin.
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(_violation_dataset("provenance_segment_without_licence_evidence"))
    message = str(excinfo.value)
    assert "bad-segment-no-evidence" in message
    assert "licence_evidence" in message
    assert "Quill Institute Gas Program (fictional)" in message


def test_attribution_must_credit_every_provenance_segment():
    """Review #46: captions are generated from attribution_text, so an

    attribution_text omitting a segment's credit ships an uncredited
    origin — refused, naming the dropped credit.
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(_violation_dataset("attribution_missing_segment_credit"))
    message = str(excinfo.value)
    assert "bad-attribution-missing-credit" in message
    assert "attribution_text" in message
    assert "Quill Institute Gas Program (fictional)" in message


_DATASET_REFUSAL_FIELDS = [
    ("nonopen_dataset_in_chart_pack", "bad-nc-pack-dataset", "in_chart_pack"),
    (
        "dataset_permission_on_file_without_evidence",
        "bad-dataset-permission-no-evidence",
        "permission_evidence",
    ),
    ("provisional_dataset_in_chart_pack", "bad-provisional-in-pack", "in_chart_pack"),
    ("open_dataset_without_licence_evidence", "bad-open-no-evidence", "licence_evidence"),
    ("provenance_segment_without_licence_evidence", "bad-segment-no-evidence", "licence_evidence"),
    ("attribution_missing_segment_credit", "bad-attribution-missing-credit", "attribution_text"),
]


@pytest.mark.parametrize(
    ("invariant", "dataset_id", "field"),
    _DATASET_REFUSAL_FIELDS,
    ids=[row[0] for row in _DATASET_REFUSAL_FIELDS],
)
def test_dataset_refusal_messages_name_dataset_and_field(invariant, dataset_id, field):
    """Acceptance criterion, dataset side: every refusal names the

    offending dataset id and the violated field.
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(_violation_dataset(invariant))
    message = str(excinfo.value)
    assert dataset_id in message
    assert field in message


# ---------------------------------------------------------------------------
# ADR-023: no real dataset files in git + sha256 verification
# ---------------------------------------------------------------------------


def test_data_file_check_flags_unmarked_data_file(tmp_path):
    """ADR-023: a tracked data-like file without the SYNTHETIC FIXTURE

    first-line marker is a real-data candidate and is flagged by name.
    (The tree is built in tmp_path at runtime; nothing is committed.)
    """
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "co2_annual.csv").write_text(
        "year,co2_ppm\n2024,421.1\n2025,424.6\n", encoding="utf-8"
    )
    offenders = find_committed_data_files(tmp_path, ["datasets/co2_annual.csv"])
    assert offenders == ["datasets/co2_annual.csv"]


def test_data_file_check_allows_marked_synthetic_fixture(tmp_path):
    """The only data-like files git may track are marked synthetic

    fixtures; the marker (first line, tolerant of a BOM and either dash
    style, matching the fixtures #24 committed) exempts them.
    """
    fixtures = tmp_path / "tests" / "fixtures" / "charts"
    fixtures.mkdir(parents=True)
    (fixtures / "marked.csv").write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\nyear,v\n2024,1.0\n",
        encoding="utf-8",
    )
    (fixtures / "marked_bom.txt").write_text(
        "\ufeff# SYNTHETIC FIXTURE - authored for this project's tests\n1 2 3\n",
        encoding="utf-8",
    )
    tracked = ["tests/fixtures/charts/marked.csv", "tests/fixtures/charts/marked_bom.txt"]
    assert find_committed_data_files(tmp_path, tracked) == []


def test_data_file_check_ignores_non_data_files(tmp_path):
    """Prose and config are not data-like: .md/.yaml files are governed by

    the prepared-text ship check, not this one.
    """
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text("documents: []\n", encoding="utf-8")
    assert find_committed_data_files(tmp_path, ["README.md", "manifest.yaml"]) == []


_EXTENDED_SUFFIX_CASES = [
    ("real_data.json", b'{"year": 2024, "co2_ppm": 421.1}\n'),
    ("real_data.jsonl", b'{"year": 2024}\n{"year": 2025}\n'),
    ("real_data.xlsx", b"PK\x03\x04\x14\x00\x00\x00\x08\x00binary-xlsx-payload"),
    ("real_data.csv.gz", gzip.compress(b"year,co2_ppm\n2024,421.1\n")),
    ("real_data.zip", b"PK\x03\x04\x14\x00\x00\x00\x08\x00binary-zip-payload"),
    ("real_data.dat", b"2024 421.1\n2025 424.6\n"),
]


@pytest.mark.parametrize(
    ("name", "content"),
    _EXTENDED_SUFFIX_CASES,
    ids=[case[0] for case in _EXTENDED_SUFFIX_CASES],
)
def test_data_file_check_flags_extended_data_suffixes(tmp_path, name, content):
    """Review #83: the MVP pack's own providers ship .json/.xlsx variants
    and archives serve gzipped CSVs — the ADR-023 check must flag every
    unmarked tracked file across those formats, matching the full
    multi-suffix chain so real_data.csv.gz is caught.
    """
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / name).write_bytes(content)
    offenders = find_committed_data_files(tmp_path, [f"datasets/{name}"])
    assert offenders == [f"datasets/{name}"], f"{name}: unmarked data-like file must be flagged"


def test_compressed_data_files_flagged_even_with_marker_inside(tmp_path):
    """Review #83: no marker mechanism exists for compressed/binary
    formats — a gzipped CSV whose *decompressed* first line carries the
    synthetic marker is still flagged (fail closed on undecodable bytes).
    """
    payload = gzip.compress(
        f"# {SYNTHETIC_FIXTURE_MARKER} — authored for this project's tests\nyear,v\n".encode()
    )
    (tmp_path / "marked_inside.csv.gz").write_bytes(payload)
    offenders = find_committed_data_files(tmp_path, ["marked_inside.csv.gz"])
    assert offenders == ["marked_inside.csv.gz"]


def test_json_with_recorded_envelope_provenance_is_exempt(tmp_path):
    """The #83 boundary, both sides: a .json carrying the recorded-envelope
    provenance declaration (top-level _meta with provenance +
    content_signoff, the finding-#67 convention the replay fixtures use)
    is a legitimate non-data fixture and passes; a plain data .json with
    no such declaration is flagged.
    """
    envelope = {
        "_meta": {
            "format_version": 1,
            "scrubbed": True,
            "provenance": "hand-authored in recorded shape for this unit test; all invented",
            "content_signoff": {
                "who": "fixture",
                "date": "2026-08-16",
                "note": "synthetic content only",
            },
        },
        "method": "structured",
        "response": {"content": "invented"},
    }
    (tmp_path / "replay_envelope.json").write_text(json.dumps(envelope), encoding="utf-8")
    assert find_committed_data_files(tmp_path, ["replay_envelope.json"]) == []

    (tmp_path / "owid_style.json").write_text(
        json.dumps({"World": {"data": [{"year": 2024, "co2": 37000.1}]}}), encoding="utf-8"
    )
    assert find_committed_data_files(tmp_path, ["owid_style.json"]) == ["owid_style.json"]

    # A malformed .json fails closed: undecodable/unparseable => offender.
    (tmp_path / "broken.json").write_bytes(b"\xff\xfe not json")
    assert find_committed_data_files(tmp_path, ["broken.json"]) == ["broken.json"]


def test_operational_data_marker_exempts_first_party_records(tmp_path):
    """First-party operational records (e.g. the committed spend ledger
    the cost plan requires, PR #93) are not licensed source data —
    ADR-023's target. A second recognized first-line marker exempts them;
    it is a distinct guarantee from SYNTHETIC FIXTURE (one asserts
    invented test data, the other a first-party record with no external
    licence), so neither marker may satisfy the other's check.
    """
    ledger = tmp_path / "evals" / "spend-ledger.csv"
    ledger.parent.mkdir()
    ledger.write_text(
        f"# {PROJECT_OPERATIONAL_DATA_MARKER}\ndate,run,usd\n2026-08-16,spike,0.42\n",
        encoding="utf-8",
    )
    assert find_committed_data_files(tmp_path, ["evals/spend-ledger.csv"]) == []

    # The markers are distinct guarantees — neither contains the other.
    assert SYNTHETIC_FIXTURE_MARKER not in PROJECT_OPERATIONAL_DATA_MARKER
    assert PROJECT_OPERATIONAL_DATA_MARKER not in SYNTHETIC_FIXTURE_MARKER

    # Control: an unmarked CSV in the same tree still fails.
    (tmp_path / "evals" / "unmarked.csv").write_text("date,usd\n2026-08-16,1.0\n", encoding="utf-8")
    assert find_committed_data_files(tmp_path, ["evals/unmarked.csv"]) == ["evals/unmarked.csv"]


def test_operational_marker_does_not_satisfy_synthetic_fixture_meta_check():
    """The synthetic-fixture meta-test (tests/unit/test_fixture_corpus.py)
    requires the SYNTHETIC FIXTURE marker specifically; a first line
    carrying only the operational marker must not pass it — fixture
    directories hold invented test data, not operational records.
    """
    operational_first_line = f"# {PROJECT_OPERATIONAL_DATA_MARKER}"
    assert SYNTHETIC_FIXTURE_MARKER not in operational_first_line


def test_repo_tracks_no_unmarked_data_files():
    """The CI-facing ADR-023 check over the real tree: every data-like

    file git currently tracks is a marked synthetic fixture. Runs in the
    unit stage so a committed real dataset fails every PR.
    """
    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    offenders = find_committed_data_files(REPO_ROOT, tracked)
    assert offenders == [], (
        f"real data-like files tracked in git (ADR-023 forbids this): {offenders}"
    )


def test_sha256_verification_accepts_matching_file(tmp_path):
    """ADR-023: fetched files verify against their manifest-pinned hash —

    the repo pins which bytes without hosting them.
    """
    fetched = tmp_path / "fetched.csv"
    fetched.write_text("# SYNTHETIC FIXTURE — runtime-only bytes\nyear,v\n2024,1.0\n")
    digest = hashlib.sha256(fetched.read_bytes()).hexdigest()
    assert verify_fetched_sha256("syn-fetch-ok", fetched, digest) is None


def test_sha256_mismatch_refuses_naming_entry_and_field(tmp_path):
    """A hash mismatch is a refusal like any other: it names the entry and

    the sha256 field, so the failing dataset is identifiable from CI logs.
    """
    fetched = tmp_path / "fetched.csv"
    fetched.write_text("# SYNTHETIC FIXTURE — runtime-only bytes\nyear,v\n2024,2.0\n")
    with pytest.raises(ManifestError) as excinfo:
        verify_fetched_sha256("syn-fetch-drift", fetched, "0" * 64)
    message = str(excinfo.value)
    assert "syn-fetch-drift" in message
    assert "sha256" in message


def test_sha256_verification_refuses_missing_file(tmp_path):
    """A dataset the manifest promises but the fetch never produced is a

    verification failure, not a silent skip.
    """
    with pytest.raises(ManifestError) as excinfo:
        verify_fetched_sha256("syn-fetch-absent", tmp_path / "never_fetched.csv", "0" * 64)
    assert "syn-fetch-absent" in str(excinfo.value)


def test_ship_check_fails_on_undeclared_text_in_corpus_dir(tmp_path):
    """Review #77: the ship check is a complete-tree guarantee, not a
    declared-paths spot check. A text file sitting in corpus_dir that is
    not the declared `path` of any `open` document is exactly how NC text
    would ship (committed under a scratch name, or with the manifest entry
    left `path: null`) — it must fail the check, named. Control: declared
    open paths plus the manifest/README housekeeping files pass.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "declared_open.md").write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\nBody.\n", encoding="utf-8"
    )
    (corpus_dir / "manifest.yaml").write_text("documents: []\n", encoding="utf-8")
    (corpus_dir / "README.md").write_text("# corpus\n", encoding="utf-8")
    documents = [
        {"id": "syn-declared-open", "path": "declared_open.md", "permitted_context": "open"}
    ]
    # Control: every file accounted for.
    assert check_prepared_text_shipping(documents, corpus_dir) is None

    (corpus_dir / "unep_gap_report.md").write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\nNC-shaped body.\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError) as excinfo:
        check_prepared_text_shipping(documents, corpus_dir)
    assert "unep_gap_report.md" in str(excinfo.value)


def test_ship_check_fails_on_nonopen_doc_with_committed_text_and_null_path(tmp_path):
    """Review #77: the realistic NC-leak shape — the manifest entry keeps
    `path: null` (as the fixture manifest's own NC entries model) while the
    prepared text is committed under some filename. The orphan rule catches
    it: the file is not the declared path of any open document.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "gap_report_prepared.md").write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\nNC-shaped body.\n",
        encoding="utf-8",
    )
    nc_doc = {
        "id": "syn-nc-null-path",
        "path": None,
        "permitted_context": "non-commercial-educational",
    }
    with pytest.raises(ManifestError) as excinfo:
        check_prepared_text_shipping([nc_doc], corpus_dir)
    assert "gap_report_prepared.md" in str(excinfo.value)


def test_load_refuses_manifest_without_documents_key(tmp_path):
    """Review #77 aggravator: a manifest whose top level lacks `documents`
    (e.g. the key is typo'd) must refuse rather than load as an empty
    corpus — a vacuously green gate over zero documents is the fail-open
    direction. An explicit `documents: []` remains legal.
    """
    typoed = tmp_path / "manifest.yaml"
    typoed.write_text("docs:\n  - id: syn-typoed-section\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="documents"):
        load_corpus_manifest(typoed)

    bare_null = tmp_path / "null.yaml"
    bare_null.write_text("documents:\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="documents"):
        load_corpus_manifest(bare_null)

    explicit_empty = tmp_path / "empty.yaml"
    explicit_empty.write_text("documents: []\n", encoding="utf-8")
    assert load_corpus_manifest(explicit_empty).documents == []


def test_ship_check_scans_only_the_given_corpus_dir(tmp_path):
    """The checks are pure over paths passed in (IMPLEMENTATION.md §1):

    a non-open document whose prepared text exists elsewhere on disk but
    not under the given corpus_dir passes — filesystem reach-around would
    make the check untestable and environment-dependent.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "prepared.md").write_text(
        textwrap.dedent(
            """\
            # SYNTHETIC FIXTURE — authored for this project's tests
            Body.
            """
        ),
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    nc_doc = {
        "id": "syn-inline-nc",
        "path": "prepared.md",  # exists in `elsewhere`, not in corpus_dir
        "permitted_context": "non-commercial-educational",
    }
    assert check_prepared_text_shipping([nc_doc], corpus_dir) is None
