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
_VIOLATIONS = {
    entry["violates"]: entry
    for entry in yaml.safe_load(FIXTURE_MANIFEST.read_text(encoding="utf-8"))["violations"]
}


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
