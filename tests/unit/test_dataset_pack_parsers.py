"""Productionised pack parsers + coverage arithmetic (issue #14) — RED phase.

Behavioural tests for `charts/pack.py`: the six MVP dataset parsers
(DESIGN.md §3.7 as amended by ADR-023), the parser registry/manifest
reference resolution, and the review-#52 coverage semantics ("coverage"
means *parsed usable extent*, computable only from parser output).

All inputs are committed synthetic fixtures in
tests/fixtures/datasets/raw/ — authored fresh, every value invented
(fixture posture per review finding #51: structure mimics each provider's
real format, no real rows are copied). The issue #4 spike parsers keep
their own characterisation tests as a canary; the tests here are the
production contract the spike code is productionised against.

Unit tier: pure functions over local files, no network, no Docker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from charts import pack

RAW = Path(__file__).parents[1] / "fixtures" / "datasets" / "raw"

BEREITER = RAW / "bereiter_composite.txt"
KAUFMAN = RAW / "kaufman_percentiles.csv"
GML = RAW / "gml_annmean.csv"
GISTEMP = RAW / "gistemp_glb.csv"
HADCRUT5 = RAW / "hadcrut5_annual.csv"
OWID = RAW / "owid_co2.csv"


def test_all_raw_fixtures_carry_the_synthetic_marker():
    """ADR-023 fixture rule: every raw-format fixture is marked, line 1.

    (The repo-wide check in test_licensing_invariants.py enforces this
    for all tracked data-like files; this local guard keeps the rule
    visible where the fixtures are authored.)
    """
    files = sorted(p for p in RAW.iterdir() if p.is_file())
    assert len(files) == 6, "expected exactly the six MVP raw-format fixtures"
    for path in files:
        first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "SYNTHETIC FIXTURE" in first_line, f"{path.name}: missing marker"


# ---------------------------------------------------------------------------
# Issue TDD plan items 1-2 + the new-format equivalents: expected columns
# ---------------------------------------------------------------------------


def test_bereiter_txt_parses_expected_columns():
    """TDD plan item 1: BOM+CRLF+'#'-header TXT -> typed age/CO2 frame,

    sorted by age ascending (the fixture rows are deliberately unsorted).
    """
    df = pack.parse_bereiter_co2(BEREITER)
    assert list(df.columns) == ["age_bp", "co2_ppm", "co2_1s_ppm"]
    assert [str(df[c].dtype) for c in df.columns] == ["float64"] * 3
    assert len(df) == 5
    assert df["age_bp"].is_monotonic_increasing
    assert df.iloc[0]["age_bp"] == pytest.approx(-50.20)
    assert df.iloc[0]["co2_ppm"] == pytest.approx(367.10)
    assert df.iloc[-1]["age_bp"] == pytest.approx(11987.40)
    assert df.iloc[-1]["co2_ppm"] == pytest.approx(241.60)


def test_kaufman_txt_parses_expected_columns():
    """TDD plan item 2: the Temp12k header/units quirks found in #4 —

    space-padded header names, latitude-band columns dropped, global
    percentile columns kept and renamed, ages as float years BP.
    """
    df = pack.parse_kaufman_temp12k(KAUFMAN)
    assert list(df.columns) == ["age_bp", "temp_c", "temp_c_5", "temp_c_95"]
    assert [str(df[c].dtype) for c in df.columns] == ["float64"] * 4
    assert len(df) == 5
    assert df["age_bp"].is_monotonic_increasing
    row = df[df["age_bp"] == 100.0].iloc[0]
    assert row["temp_c"] == pytest.approx(0.0)
    assert row["temp_c_5"] == pytest.approx(-0.32)
    assert row["temp_c_95"] == pytest.approx(0.31)


def test_gml_csv_parses_expected_columns():
    df = pack.parse_gml_co2_annual(GML)
    assert list(df.columns) == ["year_ce", "co2_ppm", "unc_ppm"]
    assert str(df["year_ce"].dtype) == "int64"
    assert [str(df[c].dtype) for c in ("co2_ppm", "unc_ppm")] == ["float64"] * 2
    assert list(df["year_ce"]) == [1959, 1960, 1990, 2024]
    assert df.iloc[0]["co2_ppm"] == pytest.approx(313.34)


def test_gistemp_csv_parses_expected_columns_and_drops_partial_year():
    """The in-progress year's J-D is '***' and its row is dropped — the

    documented missing-value convention behind review finding #52.
    """
    df = pack.parse_gistemp_annual(GISTEMP)
    assert list(df.columns) == ["year_ce", "temp_anomaly_c"]
    assert str(df["year_ce"].dtype) == "int64"
    assert str(df["temp_anomaly_c"].dtype) == "float64"
    assert list(df["year_ce"]) == [1880, 1890, 1950, 2024], (
        "the raw file runs to 2026 but 2026's J-D is '***' — usable rows end at 2024"
    )
    assert df.iloc[0]["temp_anomaly_c"] == pytest.approx(-0.31)
    assert df.iloc[-1]["temp_anomaly_c"] == pytest.approx(1.11)


def test_hadcrut5_csv_parses_expected_columns():
    """New parser for the MVP six: HadCRUT5 global annual summary series,

    confidence limits kept (the dataset ships its uncertainty, DESIGN
    §3.7: bands render when shipped).
    """
    df = pack.parse_hadcrut5_annual(HADCRUT5)
    assert list(df.columns) == [
        "year_ce",
        "temp_anomaly_c",
        "temp_anomaly_c_lower",
        "temp_anomaly_c_upper",
    ]
    assert str(df["year_ce"].dtype) == "int64"
    assert list(df["year_ce"]) == [1850, 1900, 1950, 2000, 2024]
    row = df.iloc[0]
    assert row["temp_anomaly_c"] == pytest.approx(-0.41)
    assert row["temp_anomaly_c_lower"] == pytest.approx(-0.55)
    assert row["temp_anomaly_c_upper"] == pytest.approx(-0.27)


def test_owid_csv_parses_expected_columns():
    """New parser for the MVP six: the tidy country/year/co2 subset of the

    wide OWID file; blank-co2 rows dropped (documented missing-value
    convention); sorted by country then year (fixture rows unsorted).
    """
    df = pack.parse_owid_co2(OWID)
    assert list(df.columns) == ["country", "iso_code", "year_ce", "co2_mt"]
    assert str(df["year_ce"].dtype) == "int64"
    assert str(df["co2_mt"].dtype) == "float64"
    # Sylvania 1950 has a blank co2 cell and is dropped; nothing else is.
    assert len(df) == 5
    assert list(df["country"]) == ["Freedonia", "Freedonia", "Sylvania", "World", "World"]
    assert list(df["year_ce"]) == [1950, 2023, 2023, 1949, 2023]
    assert df.iloc[0]["co2_mt"] == pytest.approx(0.52)
    world_2023 = df[(df["country"] == "World") & (df["year_ce"] == 2023)].iloc[0]
    assert world_2023["co2_mt"] == pytest.approx(37400.0)


# ---------------------------------------------------------------------------
# Issue TDD plan items 3 + 8: corrupt and truncated files fail loudly
# ---------------------------------------------------------------------------

# (parser, source fixture, unique numeric token to corrupt, data-row prefix)
_MALFORMED_CASES = [
    ("parse_bereiter_co2", BEREITER, "241.60"),
    ("parse_kaufman_temp12k", KAUFMAN, "-0.27"),
    ("parse_gml_co2_annual", GML, "352.02"),
    ("parse_gistemp_annual", GISTEMP, "-.31"),
    ("parse_hadcrut5_annual", HADCRUT5, "-0.41"),
    ("parse_owid_co2", OWID, "37400.00"),
]


def _fixture_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize(
    "parser_name,fixture,token", _MALFORMED_CASES, ids=[c[0] for c in _MALFORMED_CASES]
)
def test_parser_rejects_malformed_rows(parser_name, fixture, token, tmp_path):
    """TDD plan item 3: a corrupt numeric cell fails loudly — never a

    silently dropped row (only documented in-band missing-value
    conventions like GISTEMP '***' / OWID blanks may drop rows).
    """
    text = _fixture_text(fixture)
    assert text.count(token) == 1, "fixture drifted: corruption token is no longer unique"
    corrupt = tmp_path / f"corrupt{fixture.suffix}"
    corrupt.write_text(text.replace(token, "oops"), encoding="utf-8")
    parser = getattr(pack, parser_name)
    with pytest.raises(ValueError):
        parser(corrupt)


@pytest.mark.parametrize(
    "parser_name,fixture,token", _MALFORMED_CASES, ids=[c[0] for c in _MALFORMED_CASES]
)
def test_parser_rejects_truncated_file(parser_name, fixture, token, tmp_path):
    """TDD plan item 8 (failure half): a file cut off before any data row

    raises, rather than yielding an empty frame that downstream code
    would treat as a real (empty) dataset.
    """
    del token
    text = _fixture_text(fixture)
    # Chop every data row: keep only comment/title/header lines.
    keep: list[str] = []
    for line in text.splitlines():
        keep.append(line)
        first_cell = line.split(",")[0].split("\t")[0].strip()
        if first_cell in ("age_gas_calBP", "ages", "year", "Year", "Time", "country"):
            break
    truncated_text = "\n".join(keep) + "\n"
    truncated = tmp_path / f"truncated{fixture.suffix}"
    truncated.write_text(truncated_text, encoding="utf-8")
    parser = getattr(pack, parser_name)
    with pytest.raises(ValueError):
        parser(truncated)


def test_parser_rejects_unexpected_columns(tmp_path):
    """A renamed/reordered header refuses with a message naming what was

    expected — schema drift at the provider must never be silently
    reinterpreted.
    """
    renamed = tmp_path / "renamed.csv"
    renamed.write_text(
        "# SYNTHETIC FIXTURE - runtime-only\nyear,average,unc\n1959,313.34,0.11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        pack.parse_gml_co2_annual(renamed)


# ---------------------------------------------------------------------------
# Registry + manifest parser references
# ---------------------------------------------------------------------------


def test_parsers_registry_covers_exactly_the_six_mvp_datasets():
    expected = {
        "gistemp_v4",
        "hadcrut5",
        "noaa_gml_co2_mlo",
        "bereiter2015_co2",
        "kaufman2020_temp12k",
        "owid_co2",
    }
    assert pack.MVP_DATASET_IDS == expected
    assert set(pack.PARSERS) == expected
    assert all(callable(fn) for fn in pack.PARSERS.values())


def test_resolve_parser_resolves_committed_pack_parsers():
    """A manifest `parser` reference resolves to the identical committed

    callable the registry names — the manifest can only ever point at
    committed pack parsers.
    """
    for ds_id in sorted(pack.MVP_DATASET_IDS):
        fn = pack.PARSERS[ds_id]
        assert pack.resolve_parser(f"charts/pack.py::{fn.__name__}") is fn


@pytest.mark.parametrize(
    "ref",
    [
        "charts/spike/parsers.py::parse_bereiter_co2",  # spike code is not production
        "evals/scripts/compute_chart_fixtures.py::main",  # arbitrary module
        "charts/pack.py::dataset_coverage",  # in-module but not a parser
        "charts/pack.py::no_such_function",
        "parse_gml_co2_annual",  # malformed (no module part)
    ],
)
def test_resolve_parser_rejects_non_pack_references(ref):
    with pytest.raises(ValueError):
        pack.resolve_parser(ref)


# ---------------------------------------------------------------------------
# Review finding #52: coverage means parsed usable extent
# ---------------------------------------------------------------------------


def test_dataset_coverage_year_ce_reflects_usable_rows():
    """#52's named semantics test: coverage endpoints equal the first/last

    rows the committed parser actually yields — for GISTEMP that is 2024
    here (raw file extends to 2026; the in-progress row parses away).
    """
    df = pack.parse_gistemp_annual(GISTEMP)
    coverage = pack.dataset_coverage(df, {"unit": "year_ce"})
    assert coverage == {"first_year_ce": 1880, "last_year_ce": 2024}
    assert all(isinstance(v, int) for v in coverage.values())


def test_dataset_coverage_years_bp_reflects_usable_rows():
    df = pack.parse_bereiter_co2(BEREITER)
    coverage = pack.dataset_coverage(df, {"unit": "years_bp", "present_ce": 1950})
    assert coverage == {"oldest_bp": 11987, "youngest_bp": -50}, (
        "BP endpoints are rounded to the nearest whole year (manifest convention)"
    )
    assert all(isinstance(v, int) for v in coverage.values())


def test_dataset_coverage_refuses_empty_frame_and_unknown_unit():
    empty = pd.DataFrame({"year_ce": pd.Series(dtype="int64")})
    with pytest.raises(ValueError):
        pack.dataset_coverage(empty, {"unit": "year_ce"})
    good = pd.DataFrame({"year_ce": [1990, 1991]})
    with pytest.raises(ValueError):
        pack.dataset_coverage(good, {"unit": "decade_ce"})


# ---------------------------------------------------------------------------
# Review finding #108: the manifest-driven partial-current-year policy
# ---------------------------------------------------------------------------

_HADCRUT_HEADER = (
    "Time,Anomaly (deg C),Lower confidence limit (2.5%),Upper confidence limit (97.5%)"
)


def _hadcrut_file(tmp_path, rows: list[str]):
    path = tmp_path / "synthetic_hadcrut.csv"
    path.write_text(
        "# SYNTHETIC FIXTURE — authored for this project's tests\n"
        + _HADCRUT_HEADER
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def _hadcrut_entry(**overrides) -> dict:
    entry = {
        "parser": "charts/pack.py::parse_hadcrut5_annual",
        "time_axis": {"unit": "year_ce"},
        "retrieved_at": "2026-08-16",
        "partial_current_year": "drop",
    }
    entry.update(overrides)
    return entry


def test_hadcrut5_in_progress_year_excluded_by_drop_policy(tmp_path):
    """Review finding #108: HadCRUT5 publishes a partial-year mean for the

    in-progress year with no in-band marker (unlike GISTEMP's '***'), so
    the exclusion must be manifest-driven: an entry carrying
    `partial_current_year: drop` has every row whose year_ce >= the
    retrieved_at year excluded by the pack's load surface — a ~7-month
    mean must never render as a settled annual datum (the
    "Met-Office-shows-cooling screenshot" failure mode).
    """
    path = _hadcrut_file(
        tmp_path,
        ["2024,1.21,1.13,1.29", "2025,1.30,1.22,1.38", "2026,1.05,0.97,1.13"],
    )
    df = pack.load_dataset_frame(_hadcrut_entry(), path)
    assert list(df["year_ce"]) == [2024, 2025], (
        "the in-progress year (== retrieved_at year) must be dropped under the drop policy"
    )
    assert pack.dataset_coverage(df, {"unit": "year_ce"}) == {
        "first_year_ce": 2024,
        "last_year_ce": 2025,
    }


def test_load_dataset_frame_without_policy_keeps_all_parsed_rows(tmp_path):
    """No policy field -> the load surface is exactly the committed parser

    (the policy is opt-in per dataset, recorded in the manifest, never a
    silent global rule).
    """
    path = _hadcrut_file(tmp_path, ["2024,1.21,1.13,1.29", "2025,1.30,1.22,1.38"])
    entry = _hadcrut_entry()
    del entry["partial_current_year"]
    df = pack.load_dataset_frame(entry, path)
    assert list(df["year_ce"]) == [2024, 2025]


def test_partial_current_year_unknown_policy_refused(tmp_path):
    """The policy vocabulary is closed: an unknown value fails loudly

    rather than silently keeping (or dropping) the partial year.
    """
    path = _hadcrut_file(tmp_path, ["2025,1.30,1.22,1.38"])
    with pytest.raises(ValueError):
        pack.load_dataset_frame(_hadcrut_entry(partial_current_year="annotate"), path)


def test_partial_current_year_drop_refuses_emptied_frame(tmp_path):
    """Fail loudly, never silently: a file whose only rows are in-progress

    would otherwise yield an empty frame downstream code treats as data.
    """
    path = _hadcrut_file(tmp_path, ["2026,1.05,0.97,1.13"])
    with pytest.raises(ValueError):
        pack.load_dataset_frame(_hadcrut_entry(), path)
