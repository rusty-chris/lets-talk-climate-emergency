"""The real datasets/manifest.yaml as the pack's contract (issue #14) — RED.

Issue TDD plan items 4-7 plus the ADR-023 amendments from the issue's
comments and review finding #52:

- six MVP entries (DESIGN §3.7 as amended), each schema-valid under the
  merged issue #5 validators, each carrying an origin fetch URL and a
  pinned sha256 — the repo pins *which bytes* without hosting them;
- production parser references (charts/pack.py), not spike code;
- open-provisional datasets (Kaufman #23, Bereiter #45) fetch-only:
  never in the chart pack, never mirrored, licence evidence on file;
- coverage recorded as parsed usable extent, not raw-file extent (#52);
- the pack directory ships nothing but the manifest + docs, and the
  fetch landing directory is gitignored (ADR-023).

Unit tier: pure over the committed manifest file + git metadata.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from charts import pack
from ingestion.manifest import load_dataset_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"

#: The two datasets whose open verdict is provisional (spike-04 findings,
#: review #45): fetchable from origin, excluded from every committed or
#: mirrored artefact until written confirmation lands (#23).
OPEN_PROVISIONAL_IDS = {"kaufman2020_temp12k", "bereiter2015_co2"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _raw_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parsed_manifest():
    """The real datasets/manifest.yaml loaded + validated once per module

    (refactor #111, audit finding 2c) — the file is committed and never
    mutated by these tests, so the ~9 per-test `load_dataset_manifest`
    calls collapse to a single parse.
    """
    return load_dataset_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def raw_manifest() -> dict:
    """The real datasets/manifest.yaml as a raw ``yaml.safe_load`` mapping,

    parsed once per module (refactor #111, audit finding 2c) alongside
    :func:`parsed_manifest` — tests that need the unvalidated raw shape
    (e.g. asserting on fields the typed record deliberately drops) use
    this instead of re-reading the file.
    """
    return _raw_manifest()


def test_real_manifest_is_schema_valid(parsed_manifest):
    """Guard: the manifest must keep passing the merged #5 validators

    unchanged as #14 extends it — schema validity is a precondition of
    every other test here.
    """
    assert parsed_manifest.datasets, "dataset manifest is empty"
    assert parsed_manifest.splice_pairs, "splice pairs missing"


def test_every_pack_dataset_has_manifest_entry(parsed_manifest):
    """TDD plan item 4: exactly the six MVP entries (GISTEMP v4, HadCRUT5,

    Mauna Loa CO2, Bereiter 2015, Kaufman 2020, OWID co2-data).
    """
    assert set(parsed_manifest.datasets) == pack.MVP_DATASET_IDS


def test_every_entry_carries_origin_fetch_url_and_sha256_pin(raw_manifest):
    """ADR-023: reproducibility is pinned URL + sha256 — every entry, the

    provisional ones included, fetches over https from its origin archive
    and pins the exact bytes.
    """
    raw = raw_manifest["datasets"]
    for ds_id in sorted(pack.MVP_DATASET_IDS):
        assert ds_id in raw, f"{ds_id}: missing manifest entry"
        entry = raw[ds_id]
        url = entry.get("url") or ""
        assert url.startswith("https://"), f"{ds_id}: fetch URL must be https origin, got {url!r}"
        sha256 = entry.get("sha256") or ""
        assert _SHA256_RE.match(sha256), f"{ds_id}: sha256 pin missing or malformed"


def test_manifest_parser_refs_resolve_to_production_parsers(raw_manifest):
    """Every entry's `parser` field resolves (charts.pack.resolve_parser)

    to the committed production parser the registry names for that
    dataset — spike code is not production (IMPLEMENTATION.md §2 item 6).
    """
    raw = raw_manifest["datasets"]
    for ds_id in sorted(pack.MVP_DATASET_IDS):
        ref = raw.get(ds_id, {}).get("parser")
        assert ref, f"{ds_id}: missing parser reference"
        assert pack.resolve_parser(ref) is pack.PARSERS[ds_id], (
            f"{ds_id}: parser ref {ref!r} does not resolve to the production parser"
        )


def test_pack_contains_only_open_datasets(parsed_manifest):
    """TDD plan item 5 (post-ADR-023 form): in_chart_pack entries are

    exactly the confirmed-open ones; open-provisional is never packable.
    """
    for ds_id, record in parsed_manifest.datasets.items():
        if record.in_chart_pack:
            assert record.permitted_context == "open", (
                f"{ds_id}: in the chart pack without a confirmed open licence"
            )


def test_open_provisional_datasets_are_fetch_only_and_never_mirrored(raw_manifest, parsed_manifest):
    """ADR-023 + the issue comments: Kaufman and Bereiter stay

    open-provisional until #23's written confirmation, out of the chart
    pack, fetched only from their origin archive (NCEI), and no entry of
    any status carries a mirror — a mirror is a separate client decision.
    """
    raw = raw_manifest["datasets"]
    for ds_id in OPEN_PROVISIONAL_IDS:
        record = parsed_manifest.datasets[ds_id]
        assert record.permitted_context == "open-provisional", (
            f"{ds_id}: must stay open-provisional until written confirmation (#23) is on file"
        )
        assert record.in_chart_pack is False
        host = urlparse(record.url).netloc
        assert host.endswith("ncei.noaa.gov"), (
            f"{ds_id}: provisional datasets fetch from the origin archive only, got {host!r}"
        )
    for ds_id, entry in raw.items():
        mirror_keys = {k for k in entry if "mirror" in k.lower()}
        assert not mirror_keys, (
            f"{ds_id}: mirror fields {mirror_keys} — mirroring is a separate client "
            "decision (ADR-023), and provisional datasets are never mirrored anywhere"
        )


def test_new_modern_datasets_are_open_and_in_pack(parsed_manifest):
    """The two entries #14 adds: HadCRUT5 (OGL v3) and OWID co2-data

    (CC BY 4.0) are confirmed open, in the chart pack, with licence
    evidence on file (issue #5 discipline: a claim requires evidence).
    """
    for ds_id in ("hadcrut5", "owid_co2"):
        record = parsed_manifest.datasets[ds_id]
        assert record.permitted_context == "open"
        assert record.in_chart_pack is True
        assert record.licence_evidence and record.licence_evidence.strip()


def test_owid_upstream_licence_evidence_recorded(parsed_manifest):
    """Review finding #110: the pack's one owid_co2 data column (`co2`)

    is third-party data from the Global Carbon Budget, which OWID does
    NOT relicense — OWID's README says third-party data keeps the
    original authors' licence terms. The operative licence is therefore
    GCB's own, and the entry must carry upstream GCB evidence distinct
    from OWID's README claim, per the review-#46 per-segment provenance
    discipline already used for noaa_gml_co2_mlo.
    """
    record = parsed_manifest.datasets["owid_co2"]
    gcb = next(
        (seg for seg in record.provenance if "global carbon" in seg.origin.lower()),
        None,
    )
    assert gcb is not None, (
        "owid_co2: no Global Carbon Budget provenance segment — the co2 column's "
        "operative licence is GCB's own and needs its own evidence (#110)"
    )
    assert "CC BY" in gcb.licence or "Creative Commons Attribution" in gcb.licence, (
        f"owid_co2: GCB segment licence must record GCB's CC BY 4.0 grant, got {gcb.licence!r}"
    )
    evidence = gcb.licence_evidence or ""
    assert evidence.strip(), "owid_co2: GCB segment has no licence evidence"
    assert "zenodo" in evidence.lower(), (
        "owid_co2: GCB evidence must cite where the CC BY declaration was verified "
        "(the GCB Zenodo record), not restate OWID's README"
    )


def test_owid_licence_claim_agrees_with_its_evidence(raw_manifest):
    """Review finding #110: the licence field must state the true chain

    (OWID's own work CC BY 4.0; the third-party co2 column under GCB's
    own CC BY 4.0), never the refuted claim that OWID redistributes
    GCP data 'under the same CC BY terms it publishes the repository
    under' — the evidence (OWID README) says the opposite.
    """
    entry = raw_manifest["datasets"]["owid_co2"]
    licence = entry["licence"]
    assert "redistributes under the same CC BY terms" not in licence, (
        "owid_co2: the licence field still carries the claim its own evidence refutes (#110)"
    )
    assert "Global Carbon Budget" in licence and "CC BY 4.0" in licence, (
        "owid_co2: the licence field must name the co2 column's operative licence — "
        "the Global Carbon Budget's own CC BY 4.0 (#110)"
    )
    note = entry["human_signoff"]["note"]
    assert "not a licence concern" not in note, (
        "owid_co2: the signoff note still waves the GCP provenance off as 'not a licence "
        "concern' — it is one, now a verified one (#110)"
    )


def test_alignment_periods_present_for_splice_pairs(parsed_manifest):
    """TDD plan item 6: the two ADR-020 curation decisions stay fixed in

    the manifest — co2_10k with an explicit no-rebaseline record,
    temp_10k with the [1880, 1900] alignment period and its display
    reference.
    """
    pairs = {p.id: p for p in parsed_manifest.splice_pairs}
    assert set(pairs) >= {"co2_10k", "temp_10k"}

    co2 = pairs["co2_10k"]
    assert (co2.paleo, co2.instrumental) == ("bereiter2015_co2", "noaa_gml_co2_mlo")
    assert co2.splice_year_ce == 1959
    assert co2.rebaseline is None, "CO2 is absolute — no rebaseline is legal for this pair"

    temp = pairs["temp_10k"]
    assert (temp.paleo, temp.instrumental) == ("kaufman2020_temp12k", "gistemp_v4")
    assert temp.splice_year_ce == 1880
    assert temp.rebaseline is not None
    assert temp.rebaseline.apply_to == "gistemp_v4"
    assert tuple(temp.rebaseline.alignment_period_ce) == (1880, 1900)
    assert temp.rebaseline.display_reference


def test_kaufman_licence_evidence_recorded(parsed_manifest):
    """TDD plan item 7: the load-bearing check's paper trail — the

    Kaufman entry carries a non-empty evidence note recording the
    archive-metadata / DataCite / paper-licence trail.
    """
    record = parsed_manifest.datasets["kaufman2020_temp12k"]
    note = record.licence_note or ""
    assert note.strip(), "kaufman2020_temp12k: licence evidence trail missing"
    assert "29712" in note or "DataCite" in note or "NCEI" in note, (
        "the note must record where the (absence of a) licence grant was checked"
    )


def test_gistemp_coverage_not_overstated(raw_manifest):
    """Review finding #52, manifest side: GISTEMP's current-year J-D is

    always '***' while the year is in progress, so the usable annual
    series can never extend into the year the manifest was accessed —
    coverage states parsed usable extent, not raw-file extent.
    """
    raw = raw_manifest
    access_year = int(str(raw["access_date"])[:4])
    last = raw["datasets"]["gistemp_v4"]["coverage"]["last_year_ce"]
    assert last < access_year, (
        f"gistemp_v4 coverage.last_year_ce={last} overstates the usable extent for an "
        f"access date in {access_year} (the in-progress year's J-D is '***' — #52)"
    )


def test_no_year_ce_coverage_reaches_the_access_year(raw_manifest):
    """Review finding #108: a `year_ce` dataset's last_year_ce may never

    reach the year the manifest was accessed — an endpoint in the access
    year is by definition a partial-year (in-progress) mean, which must
    either be dropped (`partial_current_year: drop`) or masked in-band by
    the provider (GISTEMP '***'). Either way the recorded usable extent
    ends strictly before the access year.
    """
    raw = raw_manifest
    access_year = int(str(raw["access_date"])[:4])
    for ds_id, entry in raw["datasets"].items():
        if entry.get("time_axis", {}).get("unit") != "year_ce":
            continue
        last = entry["coverage"]["last_year_ce"]
        assert last < access_year, (
            f"{ds_id}: coverage.last_year_ce={last} reaches the access year {access_year} — "
            "a partial in-progress year is being presented as settled coverage (#108)"
        )


def test_hadcrut5_partial_year_policy_recorded_and_temperature_endpoints_align(raw_manifest):
    """Review finding #108: the HadCRUT5 partial-year decision is a

    machine-readable manifest field (`partial_current_year: drop`), not a
    sign-off-note rationalisation; and with it applied the pack's two
    temperature series end in the same year, so "latest year" comparisons
    across the pack agree.
    """
    raw = raw_manifest["datasets"]
    assert raw["hadcrut5"].get("partial_current_year") == "drop", (
        "hadcrut5 publishes a partial in-progress-year mean with no in-band marker; "
        "the manifest must record the drop policy explicitly (#108)"
    )
    assert (
        raw["hadcrut5"]["coverage"]["last_year_ce"] == raw["gistemp_v4"]["coverage"]["last_year_ce"]
    ), "the two temperature series must agree on their settled last year (#108)"


def test_coverage_blocks_match_time_axis_convention(raw_manifest):
    """Every entry's coverage block uses the endpoint keys its time-axis

    unit dictates, with integer endpoints in the right order — the shape
    charts.pack.dataset_coverage produces and the #15 range validator
    will consume.
    """
    raw = raw_manifest["datasets"]
    for ds_id in sorted(pack.MVP_DATASET_IDS):
        entry = raw.get(ds_id, {})
        unit = entry.get("time_axis", {}).get("unit")
        coverage = entry.get("coverage")
        assert coverage, f"{ds_id}: missing coverage block"
        if unit == "year_ce":
            assert set(coverage) == {"first_year_ce", "last_year_ce"}, f"{ds_id}: {coverage}"
            assert coverage["first_year_ce"] <= coverage["last_year_ce"]
        elif unit == "years_bp":
            assert set(coverage) == {"oldest_bp", "youngest_bp"}, f"{ds_id}: {coverage}"
            assert coverage["oldest_bp"] >= coverage["youngest_bp"]
        else:
            raise AssertionError(f"{ds_id}: unknown time_axis unit {unit!r}")
        assert all(isinstance(v, int) for v in coverage.values()), f"{ds_id}: {coverage}"


# ---------------------------------------------------------------------------
# Review finding #117: pack membership is structural, not metadata-only
# ---------------------------------------------------------------------------


def test_chart_pack_ids_exclude_open_provisional(raw_manifest, parsed_manifest):
    """Review finding #117: the single authoritative chartable-ids surface

    derives from the manifest's `in_chart_pack` (validator-enforced to be
    false for anything not confirmed open) — never from MVP_DATASET_IDS,
    which is the *fetch* set. Kaufman and Bereiter stay fetchable but are
    absent from the chart pack until #23's written confirmation.
    """
    ids = pack.chart_pack_dataset_ids(MANIFEST_PATH)
    raw = raw_manifest["datasets"]
    assert ids == frozenset(ds_id for ds_id, entry in raw.items() if entry["in_chart_pack"] is True)
    assert not (OPEN_PROVISIONAL_IDS & ids), "open-provisional ids leaked into the chart pack"
    assert OPEN_PROVISIONAL_IDS <= pack.MVP_DATASET_IDS, "fetchable is not chartable"
    # The surface accepts however the caller loaded the manifest: a path,
    # the raw mapping, or the ingestion.manifest loaded object.
    assert pack.chart_pack_dataset_ids(raw_manifest) == ids
    assert pack.chart_pack_dataset_ids(parsed_manifest) == ids


def test_pack_facing_surface_refuses_provisional_ids():
    """A pack-facing consumer (#15 validator, #16 renderer) gating on

    require_in_chart_pack gets an honest refusal naming the dataset's
    provisional status and the pending confirmation issue (#23) — never a
    silent pass because the id happens to be fetchable.
    """
    for ds_id in sorted(OPEN_PROVISIONAL_IDS):
        with pytest.raises(ValueError) as excinfo:
            pack.require_in_chart_pack(MANIFEST_PATH, ds_id)
        message = str(excinfo.value)
        assert ds_id in message
        assert "open-provisional" in message
        assert "#23" in message
    # Confirmed-open pack members pass.
    pack.require_in_chart_pack(MANIFEST_PATH, "gistemp_v4")
    # Unknown ids refuse too — the surface never guesses.
    with pytest.raises(ValueError):
        pack.require_in_chart_pack(MANIFEST_PATH, "no_such_dataset")


def test_splice_pairs_involving_non_pack_datasets_are_marked_unrenderable():
    """Review finding #117: both committed splice pairs (the flagship's

    spine) reference an open-provisional dataset, so neither may render
    or be fixture-pinned until #23 — and that dependency must be exposed
    where #15/#17 will hit it, with the provisional dataset named.
    """
    blocked = pack.blocked_splice_pairs(MANIFEST_PATH)
    assert set(blocked) == {"co2_10k", "temp_10k"}
    assert "bereiter2015_co2" in blocked["co2_10k"]
    assert "kaufman2020_temp12k" in blocked["temp_10k"]
    for pair_id, reason in blocked.items():
        assert "open-provisional" in reason and "#23" in reason, f"{pair_id}: {reason}"
        with pytest.raises(ValueError) as excinfo:
            pack.require_renderable_splice_pair(MANIFEST_PATH, pair_id)
        assert "open-provisional" in str(excinfo.value)
        assert "#23" in str(excinfo.value)
    with pytest.raises(ValueError):
        pack.require_renderable_splice_pair(MANIFEST_PATH, "no_such_pair")


def test_splice_pair_with_all_pack_members_is_renderable():
    """The gate is about membership, not about splicing per se: a pair

    whose datasets are all in_chart_pack is unblocked.
    """
    synthetic = {
        "datasets": {
            "syn-paleo": {"in_chart_pack": True, "permitted_context": "open"},
            "syn-instr": {"in_chart_pack": True, "permitted_context": "open"},
        },
        "splice_pairs": [{"id": "syn_pair", "paleo": "syn-paleo", "instrumental": "syn-instr"}],
    }
    assert pack.blocked_splice_pairs(synthetic) == {}
    pack.require_renderable_splice_pair(synthetic, "syn_pair")


# ---------------------------------------------------------------------------
# ADR-023 shipping surface for the pack directory
# ---------------------------------------------------------------------------


def test_datasets_dir_ships_only_manifest_and_docs():
    """ADR-023 coverage extension for the pack dir: datasets/ may track

    the manifest and human documentation, nothing else — a data file of
    *any* suffix parked there is a policy breach even if the repo-wide
    data-like-suffix check would miss it.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", "datasets/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout.split()
    allowed = {"datasets/manifest.yaml", "datasets/README.md"}
    strays = sorted(set(tracked) - allowed)
    assert not strays, f"unexpected files tracked under datasets/ (ADR-023): {strays}"


def test_default_landing_directory_is_gitignored():
    """`make datasets` lands fetched files in data/datasets/ — which must

    be ignored so a landed real dataset can never be staged (ADR-023).
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/datasets/gistemp_v4.csv"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, "data/datasets/ is not gitignored"
