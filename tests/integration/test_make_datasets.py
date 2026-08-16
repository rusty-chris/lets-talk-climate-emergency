"""`make datasets` end-to-end (issue #14) — integration, RED phase.

The Makefile target does not exist yet; this is its contract (same
pattern as the issue #5 `make corpus` tests):

    make datasets DATASETS_MANIFEST=<manifest.yaml> DATASETS_DIR=<dir>

with the variables defaulting to datasets/manifest.yaml and the
gitignored data/datasets/. The target must be `.PHONY` (a datasets/
directory exists at the repo root), runnable from the repo root, and on
any failure exit non-zero with the offending dataset id in its output so
CI logs identify the dataset without re-running (ADR-023 flow:
fetch -> sha256 verify -> parse -> schema/coverage validate -> land).

These tests drive the target against temp manifests whose entries fetch
file:// synthetic fixture bytes (tests/fixtures/datasets/raw/, all values
invented) — no network, Docker-free but subprocess-driven, hence the
integration tier (auto-marked by this directory's conftest). The real
six-dataset fetch is the `live`-marked scheduled-CI test in
test_datasets_live.py.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "tests" / "fixtures" / "datasets" / "raw"


def _entry(ds_id: str, url: str, sha256: str, *, parser: str, time_axis: dict, coverage: dict):
    """A schema-valid open entry with invented metadata (issue #5 fields

    plus the #14 pack-level fields).
    """
    return {
        "title": f"{ds_id} (invented)",
        "url": url,
        "citation": f"Invented citation for {ds_id}",
        "licence": "Public domain (invented stand-in)",
        "licence_evidence": "Invented open-data statement captured 2026-08-16",
        "attribution_text": f"{ds_id} fixture source (fictional)",
        "permitted_context": "open",
        "in_chart_pack": True,
        "sha256": sha256,
        "parser": parser,
        "time_axis": time_axis,
        "coverage": coverage,
        "human_signoff": {
            "who": "issue #14 test author",
            "date": "2026-08-16",
            "note": "synthetic make-datasets fixture entry",
        },
    }


def _co2_entry(tmp_path: Path, ds_id: str = "syn-make-co2", sha256: str | None = None) -> dict:
    data = (RAW / "gml_annmean.csv").read_bytes()
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    src = sources / f"{ds_id}.csv"
    src.write_bytes(data)
    return _entry(
        ds_id,
        src.as_uri(),
        sha256 or hashlib.sha256(data).hexdigest(),
        parser="charts/pack.py::parse_gml_co2_annual",
        time_axis={"unit": "year_ce"},
        coverage={"first_year_ce": 1959, "last_year_ce": 2024},
    )


def _write_manifest(tmp_path: Path, entries: dict[str, dict]) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + yaml.safe_dump(
            {"version": "fixture", "access_date": "2026-08-16", "datasets": entries},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _make_datasets(manifest: Path, datasets_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["make", "datasets", f"DATASETS_MANIFEST={manifest}", f"DATASETS_DIR={datasets_dir}"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def test_make_datasets_fetches_verifies_and_lands_clean_manifest(tmp_path):
    """Happy path: every dataset is fetched from its source URL, verified

    against its pin, parsed, coverage-checked, and landed in
    DATASETS_DIR as <dataset_id><suffix>; the target exits zero.
    """
    entries = {ds_id: _co2_entry(tmp_path, ds_id) for ds_id in ("syn-make-a", "syn-make-b")}
    manifest = _write_manifest(tmp_path, entries)
    datasets_dir = tmp_path / "landed"

    result = _make_datasets(manifest, datasets_dir)
    assert result.returncode == 0, (
        f"make datasets failed on a clean manifest:\n{result.stdout}\n{result.stderr}"
    )
    for ds_id, entry in entries.items():
        landed = datasets_dir / f"{ds_id}.csv"
        assert landed.is_file(), f"{ds_id}: not landed at {landed}"
        assert hashlib.sha256(landed.read_bytes()).hexdigest() == entry["sha256"]


def test_make_datasets_fails_on_hash_mismatch_naming_dataset(tmp_path):
    """A wrong pin fails the run, naming the dataset and the sha256 field

    — fetched bytes that do not match the pin never pass silently
    (ADR-023).
    """
    entries = {"syn-make-drift": _co2_entry(tmp_path, "syn-make-drift", sha256="0" * 64)}
    manifest = _write_manifest(tmp_path, entries)

    result = _make_datasets(manifest, tmp_path / "landed")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "syn-make-drift" in output, "failure output must name the offending dataset"
    assert "sha256" in output, "failure output must name the violated field"


def test_make_datasets_fails_on_parse_failure_naming_dataset(tmp_path):
    """Verified bytes the committed parser refuses fail the run, naming

    the dataset — distinguishable in the output from a hash failure so
    scheduled-CI triage knows whether the provider changed bytes or
    format.
    """
    corrupt = (RAW / "gml_annmean.csv").read_bytes().replace(b"352.02", b"oops")
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    src = sources / "syn-make-corrupt.csv"
    src.write_bytes(corrupt)
    entry = _co2_entry(tmp_path, "syn-make-corrupt")
    entry["url"] = src.as_uri()
    entry["sha256"] = hashlib.sha256(corrupt).hexdigest()
    manifest = _write_manifest(tmp_path, {"syn-make-corrupt": entry})

    result = _make_datasets(manifest, tmp_path / "landed")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "syn-make-corrupt" in output, "failure output must name the offending dataset"
    assert "parse" in output.lower(), "failure output must identify the parse failure mode"


def test_make_datasets_idempotent(tmp_path):
    """TDD plan item 10 (over file:// fixtures per ADR-023): a re-run with

    unchanged upstreams changes nothing — landed bytes and mtimes are
    untouched, and the target still exits zero.
    """
    entries = {"syn-make-stable": _co2_entry(tmp_path, "syn-make-stable")}
    manifest = _write_manifest(tmp_path, entries)
    datasets_dir = tmp_path / "landed"

    first = _make_datasets(manifest, datasets_dir)
    assert first.returncode == 0, f"first run failed:\n{first.stdout}\n{first.stderr}"
    landed = datasets_dir / "syn-make-stable.csv"
    before = (landed.read_bytes(), landed.stat().st_mtime_ns)

    second = _make_datasets(manifest, datasets_dir)
    assert second.returncode == 0, f"re-run failed:\n{second.stdout}\n{second.stderr}"
    assert (landed.read_bytes(), landed.stat().st_mtime_ns) == before, (
        "re-running `make datasets` with unchanged upstreams rewrote a landed file"
    )
