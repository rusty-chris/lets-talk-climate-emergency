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

import hashlib
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from ingestion.manifest import (
    ManifestError,
    check_prepared_text_shipping,
    find_committed_data_files,
    load_corpus_manifest,
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
    with pytest.raises(ManifestError) as excinfo:
        validate_document(_violation_document("missing_required_field"))
    message = str(excinfo.value)
    assert "bad-missing-fields" in message
    for field in ("sha256", "attribution_text", "retrieved_at"):
        assert field in message, f"refusal must name the absent field {field!r}"


def test_build_refuses_unknown_consensus_position_value():
    """Review #79: `consensus_position` is a closed enum (assessed |
    beyond-assessed-range, DESIGN §2.1). A typo'd value must refuse,
    naming the document, the field and the valid choices — fail-open
    coercion here silently disarms the §2.3 severity-skew guardrail for
    exactly the Hansen-style documents it exists for.
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_document(_violation_document("unknown_consensus_position"))
    message = str(excinfo.value)
    assert "bad-unknown-consensus" in message
    assert "consensus_position" in message
    assert "assessed" in message and "beyond-assessed-range" in message


def test_build_refuses_empty_string_consensus_position():
    """Review #79: an explicit empty string is not 'unset' — it refuses
    rather than silently defaulting to assessed. (Inline entry: invented
    metadata only, modelled on the fixture's valid open entry.)
    """
    entry = dict(_VIOLATIONS["unknown_consensus_position"]["document"])
    entry["id"] = "syn-inline-empty-consensus"
    entry["consensus_position"] = ""
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-empty-consensus" in message
    assert "consensus_position" in message


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
    with pytest.raises(ManifestError) as excinfo:
        validate_document(_violation_document(invariant))
    message = str(excinfo.value)
    assert doc_id in message, f"refusal must name the offending document {doc_id!r}"
    assert field in message, f"refusal must name the violated field {field!r}"


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
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-ws-doc" in message
    assert field in message


def test_document_whitespace_only_permission_evidence_refuses():
    """Review #82 x Tier-C: a single space is not a letter on file."""
    entry = dict(
        next(
            d for d in _FIXTURE_RAW["documents"] if d["permitted_context"] == "permission-on-file"
        ),
        id="syn-inline-ws-permission",
    )
    entry["permission_evidence"] = " "
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-ws-permission" in message
    assert "permission_evidence" in message


@pytest.mark.parametrize("subfield", ["who", "note"])
def test_document_whitespace_only_signoff_subfield_refuses(subfield):
    """Review #82: a sign-off whose who/note is whitespace names nobody."""
    entry = _valid_document_entry("syn-inline-ws-signoff")
    entry["human_signoff"] = dict(entry["human_signoff"], **{subfield: " \t"})
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-ws-signoff" in message
    assert "human_signoff" in message


@pytest.mark.parametrize("field", ["licence", "licence_evidence", "attribution_text", "url"])
def test_dataset_whitespace_only_value_refuses(field):
    """Review #82, dataset side: the same strip-before-check discipline."""
    entry = _valid_dataset_entry("syn-inline-ws-dataset")
    entry[field] = " "
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-ws-dataset" in message
    assert field in message


@pytest.mark.parametrize("bad_sha", ["TBD", "0" * 63, "G" * 64, "0" * 64 + "0", "0X" * 32])
def test_sha256_must_be_64_hex(bad_sha):
    """Review #82: the pin says *which bytes* (ADR-023) — a placeholder
    like 'TBD' pins nothing and must refuse at validation time, not at
    (possibly never-reached) fetch time.
    """
    entry = _valid_document_entry("syn-inline-bad-sha")
    entry["sha256"] = bad_sha
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-bad-sha" in message
    assert "sha256" in message


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
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-stringly-bool" in message
    assert "redistributable" in message


def test_dataset_in_chart_pack_must_be_boolean():
    """The dataset-side boolean, same discipline: 'true' as a string is a
    type error, not pack membership.
    """
    entry = _valid_dataset_entry("syn-inline-stringly-pack")
    entry["in_chart_pack"] = "false"
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-stringly-pack" in message
    assert "in_chart_pack" in message


def test_malformed_signoff_type_refuses_with_manifest_error():
    """Review #82: human_signoff: 'signed' must produce a ManifestError
    naming the document and field — never an AttributeError traceback.
    """
    entry = _valid_document_entry("syn-inline-signoff-string")
    entry["human_signoff"] = "signed"
    with pytest.raises(ManifestError) as excinfo:
        validate_document(entry)
    message = str(excinfo.value)
    assert "syn-inline-signoff-string" in message
    assert "human_signoff" in message


def test_malformed_provenance_type_refuses_with_manifest_error():
    """A provenance given as a string (or with string segments) is a
    naming refusal, not an AttributeError.
    """
    entry = _valid_dataset_entry("syn-inline-provenance-string")
    entry["provenance"] = "all fine, trust me"
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-provenance-string" in message
    assert "provenance" in message

    entry = _valid_dataset_entry("syn-inline-segment-string")
    entry["provenance"] = ["all fine, trust me"]
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-segment-string" in message
    assert "provenance" in message


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
    with pytest.raises(ManifestError) as excinfo:
        validate_splice_pair(pair)
    message = str(excinfo.value)
    assert "syn-inline-rebaseline-string" in message
    assert "rebaseline" in message


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

    trail is just an unbacked claim with a softer name — refused. (Inline
    entry: invented metadata only.)
    """
    entry = {
        "id": "syn-inline-provisional-bare",
        "title": "Invented series, provisional with no note",
        "url": "https://archive.example.invalid/bare.csv",
        "licence": "No explicit grant at archive (invented)",
        "permitted_context": "open-provisional",
        "in_chart_pack": False,
        "sha256": "cd" * 32,
        "attribution_text": "Invented Team (fictional)",
        "human_signoff": {"who": "fixture", "date": "2026-08-16", "note": "inline"},
    }
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-provisional-bare" in message
    assert "licence_note" in message


def test_dataset_permission_on_file_requires_permission_evidence():
    """Review #78: the Tier-C rule holds for datasets exactly as for
    documents — `permission-on-file` is a claim about a letter on file,
    and a dataset entry making it with empty `permission_evidence`
    refuses, naming the dataset and the field.
    """
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(_violation_dataset("dataset_permission_on_file_without_evidence"))
    message = str(excinfo.value)
    assert "bad-dataset-permission-no-evidence" in message
    assert "permission_evidence" in message


def test_dataset_requires_retrieved_at():
    """Review #78 / DESIGN §3: datasets carry the §2.1 manifest fields;
    with ADR-023's fetch-at-build model, `retrieved_at` records when the
    pinned bytes were captured and is required. (Inline entry: invented
    metadata only.)
    """
    entry = {
        "id": "syn-inline-no-retrieved-at",
        "title": "Invented series, no retrieval date",
        "url": "https://archive.example.invalid/undated.csv",
        "licence": "Public domain (invented)",
        "licence_evidence": "Invented statement captured 2026-08-16",
        "permitted_context": "open",
        "in_chart_pack": False,
        "sha256": "ab" * 32,
        "attribution_text": "Invented Team (fictional)",
        "human_signoff": {"who": "fixture", "date": "2026-08-16", "note": "inline"},
    }
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-no-retrieved-at" in message
    assert "retrieved_at" in message


def test_dataset_rejects_unknown_permitted_context():
    """The dataset enum is open | open-provisional |

    non-commercial-educational | permission-on-file; anything else
    refuses. (Inline entry: invented metadata only.)
    """
    entry = {
        "id": "syn-inline-bad-context",
        "title": "Invented series, nonsense context",
        "url": "https://archive.example.invalid/nonsense.csv",
        "licence": "Public domain (invented)",
        "licence_evidence": "Invented statement captured 2026-08-16",
        "permitted_context": "basically-fine",
        "in_chart_pack": False,
        "sha256": "ef" * 32,
        "attribution_text": "Invented Team (fictional)",
        "human_signoff": {"who": "fixture", "date": "2026-08-16", "note": "inline"},
    }
    with pytest.raises(ManifestError) as excinfo:
        validate_dataset(entry)
    message = str(excinfo.value)
    assert "syn-inline-bad-context" in message
    assert "permitted_context" in message


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
