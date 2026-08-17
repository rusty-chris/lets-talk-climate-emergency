"""The fetch → verify → parse → validate → land flow (issue #14) — RED.

Behavioural tests for `charts.datasets.fetch_all` over file:// sourced
synthetic fixtures (IMPLEMENTATION.md §3: fetch tests use local file://
sources; the raw formats come from tests/fixtures/datasets/raw/, every
value invented). The flow must reuse the merged issue #5 primitives —
`validate_dataset` for the schema gate, `verify_fetched_sha256` for the
hash gate — which have their own tests in test_licensing_invariants.py;
here we pin the *flow's* observable behaviour around them.

The four failure modes are distinguishable by exception class and every
message names the offending dataset (ADR-023 operability: CI logs must
identify the dataset without re-running):

    network error   -> DatasetFetchError
    hash mismatch   -> DatasetHashMismatchError
    parse failure   -> DatasetParseError
    schema refusal  -> DatasetSchemaError  (incl. coverage disagreement, #52)

Unit tier: local filesystem only, no network, no Docker.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from charts import datasets
from charts.datasets import (
    DatasetFetchError,
    DatasetHashMismatchError,
    DatasetPackError,
    DatasetParseError,
    DatasetSchemaError,
)

RAW = Path(__file__).parents[1] / "fixtures" / "datasets" / "raw"

GML_BYTES = (RAW / "gml_annmean.csv").read_bytes()
BEREITER_BYTES = (RAW / "bereiter_composite.txt").read_bytes()

#: Usable-extent coverage of the two raw fixtures (matches the values the
#: parser tests pin against the same files).
GML_COVERAGE = {"first_year_ce": 1959, "last_year_ce": 2024}
BEREITER_COVERAGE = {"oldest_bp": 11987, "youngest_bp": -50}


def _entry(
    ds_id: str,
    url: str,
    sha256: str,
    *,
    parser: str,
    time_axis: dict,
    coverage: dict,
    provisional: bool = False,
) -> dict:
    """A schema-valid dataset entry with invented metadata (issue #5 fields

    plus the #14 pack-level fields).
    """
    entry = {
        "title": f"{ds_id} (invented)",
        "url": url,
        "citation": f"Invented citation for {ds_id}",
        "licence": (
            "No explicit grant at archive (invented provisional stand-in)"
            if provisional
            else "Public domain (invented stand-in)"
        ),
        "attribution_text": f"{ds_id} fixture source (fictional)",
        "permitted_context": "open-provisional" if provisional else "open",
        "in_chart_pack": not provisional,
        "sha256": sha256,
        "retrieved_at": "2026-08-16",
        "parser": parser,
        "time_axis": time_axis,
        "coverage": coverage,
        "human_signoff": {
            "who": "issue #14 test author",
            "date": "2026-08-16",
            "note": "synthetic fetch-flow fixture entry",
        },
    }
    if provisional:
        entry["licence_note"] = "Invented evidence trail for a provisional verdict"
    else:
        entry["licence_evidence"] = "Invented open-data statement captured 2026-08-16"
    return entry


def _write_manifest(path: Path, entries: dict[str, dict]) -> Path:
    path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump(
            {"version": "fixture", "access_date": "2026-08-16", "datasets": entries},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def _source(tmp_path: Path, name: str, data: bytes) -> tuple[str, str]:
    """Land raw bytes as a file:// source; return (url, sha256)."""
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    src = sources / name
    src.write_bytes(data)
    return src.as_uri(), hashlib.sha256(data).hexdigest()


def _gml_entry(tmp_path: Path, ds_id: str = "syn-fetch-co2", **overrides) -> dict:
    url, sha = _source(tmp_path, f"{ds_id}.csv", GML_BYTES)
    entry = _entry(
        ds_id,
        url,
        sha,
        parser="charts/pack.py::parse_gml_co2_annual",
        time_axis={"unit": "year_ce"},
        coverage=dict(GML_COVERAGE),
    )
    entry.update(overrides)
    return entry


def _bereiter_entry(tmp_path: Path, ds_id: str = "syn-fetch-paleo") -> dict:
    url, sha = _source(tmp_path, f"{ds_id}.txt", BEREITER_BYTES)
    return _entry(
        ds_id,
        url,
        sha,
        parser="charts/pack.py::parse_bereiter_co2",
        time_axis={"unit": "years_bp", "present_ce": 1950},
        coverage=dict(BEREITER_COVERAGE),
        provisional=True,
    )


class RecordingTransport:
    """Injected transport seam: records every URL asked for."""

    def __init__(self, responses: dict[str, bytes] | None = None) -> None:
        self.calls: list[str] = []
        self._responses = responses

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self._responses is None or url not in self._responses:
            raise AssertionError(f"unexpected transport call: {url}")
        return self._responses[url]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fetch_all_lands_verified_datasets_in_dest_dir(tmp_path):
    """fetch -> sha256 verify -> parse -> coverage check -> land at

    dest/<dataset_id><suffix>. Open-provisional datasets fetch from
    origin exactly like open ones (fetch-only is their *shipping*
    restriction, not a fetch restriction).
    """
    entries = {
        "syn-fetch-co2": _gml_entry(tmp_path),
        "syn-fetch-paleo": _bereiter_entry(tmp_path),
    }
    manifest = _write_manifest(tmp_path / "manifest.yaml", entries)
    dest = tmp_path / "landed"

    result = datasets.fetch_all(manifest, dest)

    assert set(result) == {"syn-fetch-co2", "syn-fetch-paleo"}
    assert result["syn-fetch-co2"] == dest / "syn-fetch-co2.csv"
    assert result["syn-fetch-paleo"] == dest / "syn-fetch-paleo.txt"
    for ds_id, landed in result.items():
        assert landed.is_file(), f"{ds_id}: nothing landed at {landed}"
        digest = hashlib.sha256(landed.read_bytes()).hexdigest()
        assert digest == entries[ds_id]["sha256"], f"{ds_id}: landed bytes drifted from the pin"


def test_fetch_all_return_distinguishes_chart_pack_from_fetch_only(tmp_path):
    """Review finding #117: fetch_all's return carries the fetch/chart

    distinction so a renderer cannot confuse the sets — the mapping
    itself stays the full fetch set (landed files, provisional included),
    while `.chart_pack` exposes only the in_chart_pack members.
    """
    entries = {
        "syn-fetch-co2": _gml_entry(tmp_path),
        "syn-fetch-paleo": _bereiter_entry(tmp_path),  # open-provisional
    }
    manifest = _write_manifest(tmp_path / "manifest.yaml", entries)

    result = datasets.fetch_all(manifest, tmp_path / "landed")

    assert set(result) == {"syn-fetch-co2", "syn-fetch-paleo"}
    assert result.chart_pack == {"syn-fetch-co2": result["syn-fetch-co2"]}, (
        "open-provisional datasets are fetchable but must never appear in the "
        "chart-pack view of the landed mapping"
    )


def test_fetch_all_is_idempotent_in_process(tmp_path):
    """A landed file that still verifies is left untouched — bytes and

    mtime — on a re-run with unchanged upstreams (TDD plan item 10's
    behaviour, exercised in-process; the `make` surface has its own
    integration test).
    """
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-co2": _gml_entry(tmp_path)})
    dest = tmp_path / "landed"

    first = datasets.fetch_all(manifest, dest)
    landed = first["syn-fetch-co2"]
    before = (landed.read_bytes(), landed.stat().st_mtime_ns)

    second = datasets.fetch_all(manifest, dest)
    assert second == first
    assert (landed.read_bytes(), landed.stat().st_mtime_ns) == before, (
        "an already-verified landed file was rewritten on an unchanged re-run"
    )


# ---------------------------------------------------------------------------
# The four distinguishable failure modes, each naming the dataset
# ---------------------------------------------------------------------------


def test_unreachable_origin_raises_fetch_error_naming_dataset(tmp_path):
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-gone")
    entry["url"] = (tmp_path / "sources" / "never-created.csv").as_uri()
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-gone": entry})

    with pytest.raises(DatasetFetchError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    assert excinfo.value.dataset_id == "syn-fetch-gone"
    assert "syn-fetch-gone" in str(excinfo.value)


def test_hash_mismatch_raises_naming_dataset_and_field(tmp_path):
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-drift", sha256="0" * 64)
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-drift": entry})

    with pytest.raises(DatasetHashMismatchError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    message = str(excinfo.value)
    assert excinfo.value.dataset_id == "syn-fetch-drift"
    assert "syn-fetch-drift" in message
    assert "sha256" in message


@pytest.mark.parametrize(
    "html",
    [
        b"<html><body><h1>503 Service Temporarily Unavailable</h1></body></html>",
        b'<!DOCTYPE html>\n<html lang="en"><head><title>Access denied</title></head></html>',
        b"\n\n  <html><body>Maintenance window</body></html>",
    ],
    ids=["soft-503", "doctype-waf", "leading-whitespace"],
)
def test_hash_mismatch_message_flags_nondataset_content(tmp_path, html):
    """Review finding #116: a soft-200 origin error page (CDN/WAF outage

    HTML) fails the hash gate like genuine data drift, and the live-test
    triage guidance says drift's remedy is a re-pin — which would pin the
    error page. When the fetched bytes are HTML-shaped (every pack
    format is CSV/TSV; a leading '<' is never data), the mismatch
    message must say so and steer the operator away from re-pinning,
    alongside the digests.
    """
    entry = _gml_entry(tmp_path, ds_id="syn-errpage")
    transport = RecordingTransport({entry["url"]: html})
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-errpage": entry})

    with pytest.raises(DatasetHashMismatchError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed", transport=transport)
    message = str(excinfo.value)
    assert "sha256" in message, "the digests must still be reported"
    assert "HTML" in message, "the message must say the content looks like an HTML page"
    assert "do NOT re-pin" in message, (
        "the message must counter the live-test 're-pin' guidance for error-page bytes"
    )


def test_genuine_drift_mismatch_message_carries_no_html_warning(tmp_path):
    """The converse guard: bytes that do look like the dataset format

    (real upstream drift, the re-pin case) must not carry the error-page
    warning — the two triage paths stay distinguishable.
    """
    drifted = GML_BYTES + b"2025,424.61,0.12\n"
    url, _sha = _source(tmp_path, "drifted.csv", drifted)
    entry = _gml_entry(tmp_path, ds_id="syn-real-drift", url=url, sha256="0" * 64)
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-real-drift": entry})

    with pytest.raises(DatasetHashMismatchError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    assert "do NOT re-pin" not in str(excinfo.value)


def test_hash_mismatch_lands_nothing(tmp_path):
    """Bytes that fail verification must not remain in the landing dir —

    a half-landed unverified file is exactly what ADR-023's hash gate
    exists to prevent.
    """
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-drift", sha256="0" * 64)
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-drift": entry})
    dest = tmp_path / "landed"

    with pytest.raises(DatasetHashMismatchError):
        datasets.fetch_all(manifest, dest)
    assert not (dest / "syn-fetch-drift.csv").exists()


def test_unparseable_fetched_file_raises_parse_error_naming_dataset(tmp_path):
    corrupt = GML_BYTES.replace(b"352.02", b"oops")
    url, sha = _source(tmp_path, "corrupt-source.csv", corrupt)
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-corrupt", url=url, sha256=sha)
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-corrupt": entry})

    with pytest.raises(DatasetParseError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    assert excinfo.value.dataset_id == "syn-fetch-corrupt"
    assert "syn-fetch-corrupt" in str(excinfo.value)


def test_schema_refusal_names_dataset_and_field_before_any_fetch(tmp_path):
    """An invalid entry (here: missing attribution_text, an issue #5

    violation surfaced through the reused validate_dataset) refuses the
    whole run before a single byte moves — asserted with a recording
    transport that must see zero calls.
    """
    good = _gml_entry(tmp_path)
    bad = _gml_entry(tmp_path, ds_id="syn-fetch-invalid")
    del bad["attribution_text"]
    manifest = _write_manifest(
        tmp_path / "manifest.yaml", {"syn-fetch-co2": good, "syn-fetch-invalid": bad}
    )
    transport = RecordingTransport()

    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed", transport=transport)
    message = str(excinfo.value)
    assert excinfo.value.dataset_id == "syn-fetch-invalid"
    assert "syn-fetch-invalid" in message
    assert "attribution_text" in message
    assert transport.calls == [], "schema refusal must precede any network activity"


def test_coverage_disagreement_refused_even_when_already_landed(tmp_path):
    """Review finding #115: the warm-directory idempotent path must not

    skip the #52 coverage cross-check — manifest coverage drift (the
    hand-edited/stale endpoint shape) has to be refused whether or not
    the landed file already hash-verifies, or every local run with a
    warm data/datasets/ passes silently and only cold-tmpdir scheduled
    CI catches it days later. The bytes need no refetch to be
    re-checked: the transport must see zero calls.
    """
    manifest = _write_manifest(
        tmp_path / "manifest.yaml", {"syn-warm": _gml_entry(tmp_path, ds_id="syn-warm")}
    )
    dest = tmp_path / "landed"
    first = datasets.fetch_all(manifest, dest)
    assert first["syn-warm"].is_file()

    drifted = _gml_entry(tmp_path, ds_id="syn-warm")
    drifted["coverage"] = {"first_year_ce": 1959, "last_year_ce": 2030}
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-warm": drifted})
    transport = RecordingTransport()

    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.fetch_all(manifest, dest, transport=transport)
    assert excinfo.value.dataset_id == "syn-warm"
    assert "coverage" in str(excinfo.value)
    assert transport.calls == [], (
        "the landed file need not be refetched to have its coverage re-checked"
    )


def test_coverage_disagreeing_with_parser_output_is_refused(tmp_path):
    """Review finding #52's flow-side pin: manifest coverage endpoints

    must equal the extent the committed parser yields from the verified
    bytes; an overstated endpoint (the GISTEMP 2026 failure shape) is a
    schema refusal naming the dataset and the coverage field.
    """
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-overstated")
    entry["coverage"] = {"first_year_ce": 1959, "last_year_ce": 2026}  # parser yields ..2024
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-fetch-overstated": entry})

    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    message = str(excinfo.value)
    assert excinfo.value.dataset_id == "syn-fetch-overstated"
    assert "coverage" in message


# ---------------------------------------------------------------------------
# Review finding #108: the partial-current-year policy in the flow
# ---------------------------------------------------------------------------

#: Synthetic HadCRUT5-format bytes whose last row is the in-progress year
#: (2026 == the entries' retrieved_at year) — the shape review #108 is
#: about: a partial-year mean with no in-band missing-value marker.
HADCRUT_PARTIAL_BYTES = (
    "# SYNTHETIC FIXTURE — authored for this project's tests\n"
    "Time,Anomaly (deg C),Lower confidence limit (2.5%),Upper confidence limit (97.5%)\n"
    "2024,1.21,1.13,1.29\n"
    "2025,1.30,1.22,1.38\n"
    "2026,1.05,0.97,1.13\n"
).encode()


def _hadcrut_partial_entry(tmp_path: Path, coverage: dict, ds_id: str = "syn-partial-year") -> dict:
    url, sha = _source(tmp_path, f"{ds_id}.csv", HADCRUT_PARTIAL_BYTES)
    entry = _entry(
        ds_id,
        url,
        sha,
        parser="charts/pack.py::parse_hadcrut5_annual",
        time_axis={"unit": "year_ce"},
        coverage=coverage,
    )
    entry["partial_current_year"] = "drop"
    return entry


def test_partial_current_year_drop_applies_before_coverage_check(tmp_path):
    """Review finding #108 (flow side): under `partial_current_year: drop`

    the coverage cross-check runs against the policy-applied frame, so a
    manifest recording the settled extent (…2025) verifies even though
    the raw file carries a partial 2026 row.
    """
    entry = _hadcrut_partial_entry(tmp_path, {"first_year_ce": 2024, "last_year_ce": 2025})
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-partial-year": entry})

    result = datasets.fetch_all(manifest, tmp_path / "landed")
    assert set(result) == {"syn-partial-year"}


def test_coverage_claiming_the_partial_year_is_refused(tmp_path):
    """The other direction: coverage claiming the dropped in-progress year

    (the pre-#108 hadcrut5 manifest shape, last_year_ce == access year)
    is a schema refusal naming coverage — the partial year can never be
    presented as settled usable extent.
    """
    entry = _hadcrut_partial_entry(tmp_path, {"first_year_ce": 2024, "last_year_ce": 2026})
    manifest = _write_manifest(tmp_path / "manifest.yaml", {"syn-partial-year": entry})

    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.fetch_all(manifest, tmp_path / "landed")
    assert excinfo.value.dataset_id == "syn-partial-year"
    assert "coverage" in str(excinfo.value)


def test_unknown_partial_current_year_policy_is_schema_refusal(tmp_path):
    """The policy vocabulary is closed at the schema gate: an unrecognised

    value refuses before a single byte is fetched, naming the field.
    """
    entry = _hadcrut_partial_entry(tmp_path, {"first_year_ce": 2024, "last_year_ce": 2025})
    entry["partial_current_year"] = "flag"
    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.validate_pack_entry({**entry, "id": "syn-partial-year"})
    assert "partial_current_year" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pack-level schema gate + failure-mode taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["parser", "time_axis", "coverage"])
def test_validate_pack_entry_requires_pack_fields(tmp_path, missing_field):
    """The fields the flow itself consumes are required at the pack level

    (they are additions #14 makes on top of the reused #5 schema).
    """
    entry = _gml_entry(tmp_path, ds_id="syn-fetch-bare")
    del entry[missing_field]
    with pytest.raises(DatasetSchemaError) as excinfo:
        datasets.validate_pack_entry({**entry, "id": "syn-fetch-bare"})
    message = str(excinfo.value)
    assert "syn-fetch-bare" in message
    assert missing_field in message


def test_failure_modes_are_distinguishable_by_class():
    """One shared base carrying dataset_id; four disjoint leaf classes —

    callers (and CI log greps) can tell the modes apart without parsing
    prose.
    """
    leaves = [
        DatasetFetchError,
        DatasetHashMismatchError,
        DatasetParseError,
        DatasetSchemaError,
    ]
    for leaf in leaves:
        assert issubclass(leaf, DatasetPackError)
        exc = leaf("syn-taxonomy", "invented failure")
        assert exc.dataset_id == "syn-taxonomy"
        assert str(exc).startswith("syn-taxonomy:")
        others = [other for other in leaves if other is not leaf]
        assert not any(issubclass(leaf, other) for other in others), (
            f"{leaf.__name__} is not distinguishable from its sibling failure modes"
        )
