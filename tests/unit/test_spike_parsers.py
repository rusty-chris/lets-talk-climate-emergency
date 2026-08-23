"""Characterisation tests pinning the issue #4 spike parsers' output shape.

Spike posture (issues/04.md): not test-first — these pin what the parsers do
on small synthetic fixtures that mimic each provider's raw format, as a
canary for the productionisation in charts/pack.py (#16). Fixtures live in
tests/fixtures/spike/ and are fully invented (first line carries the
SYNTHETIC FIXTURE marker; a meta-test below enforces it) — no real dataset
rows are committed, keeping the Kaufman licensing question (see
reviews/spike-04-chart-findings.md) out of the repo.
"""

from pathlib import Path

import pytest

from charts.spike import parsers

FIXTURES = Path(__file__).parents[1] / "fixtures" / "spike"


def _bereiter_fixture_rows() -> list[tuple[float, float, float]]:
    """The Bereiter fixture's (age, co2, unc) rows, read independently of
    the parser under test. Expected values are DERIVED from the fixture
    text — never hard-coded — so the #51 fixture regeneration (perturbing
    the once-verbatim real rows) does not require editing this suite, and
    no real-looking value gets re-pinned here (review finding #51)."""
    rows = []
    for line in (FIXTURES / "bereiter_synthetic.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#") or line.startswith("age_gas_calBP"):
            continue
        age, co2, unc = line.split("\t")
        rows.append((float(age), float(co2), float(unc)))
    return rows


def _gml_fixture_rows() -> list[tuple[int, float, float]]:
    """The GML fixture's (year, mean, unc) rows, read independently of the
    parser under test (same #51 derive-don't-pin rule as above)."""
    lines = [
        line
        for line in (FIXTURES / "gml_synthetic.csv").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and not line.startswith("year,")
    ]
    return [
        (int(year), float(mean), float(unc))
        for year, mean, unc in (line.split(",") for line in lines)
    ]


def _gistemp_fixture_jd_by_year() -> dict[int, float]:
    """The GISTEMP fixture's usable J-D annual means by year, read
    independently of the parser under test: '***' rows are the
    in-progress-year convention and carry no usable annual mean (same
    #51 derive-don't-pin rule as above)."""
    lines = (FIXTURES / "gistemp_synthetic.csv").read_text(encoding="utf-8-sig").splitlines()
    header = lines[1].split(",")
    year_column, jd_column = header.index("Year"), header.index("J-D")
    return {
        int(cells[year_column]): float(cells[jd_column])
        for cells in (line.split(",") for line in lines[2:])
        if cells[jd_column] != "***"
    }


# Interpreter cache artefacts (written e.g. by importlib loading a .py fixture
# at pytest collection) are not fixtures: skip them wherever they appear.
_BYTECODE_SUFFIXES = {".pyc", ".pyo"}


def _is_cache_artefact(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in _BYTECODE_SUFFIXES


def test_all_spike_fixtures_carry_the_synthetic_marker():
    """No real source data may be committed as a fixture (IMPLEMENTATION 5)."""
    files = sorted(
        path
        for path in FIXTURES.rglob("*")
        if path.is_file() and not _is_cache_artefact(path.relative_to(FIXTURES))
    )
    assert files, "spike fixture directory is empty"
    for path in files:
        first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "SYNTHETIC FIXTURE" in first_line, f"{path.name}: missing marker"


class TestMarkerMetaTestRobustness:
    """Regression for #44: the marker meta-test vs bytecode/cache artefacts.

    test_spike_chunker.py (issue #2) loads tests/fixtures/spike/synthetic_doc.py
    via importlib at module import time, which writes __pycache__/ into the
    fixture directory during pytest collection — before any test runs. The
    meta-test must skip such interpreter artefacts (they are not fixtures and
    are never committed) while still flagging every real fixture file that
    lacks the SYNTHETIC FIXTURE marker, wherever it lives under the directory.
    """

    def test_meta_test_ignores_bytecode_cache_artefacts(self):
        """A pre-existing __pycache__/ (and stray .pyc) must not crash it."""
        pycache = FIXTURES / "__pycache__"
        pycache_preexisted = pycache.exists()
        cached_pyc = pycache / "synthetic_doc.cpython-312.pyc"
        stray_pyc = FIXTURES / "synthetic_doc.cpython-312.pyc"
        created = []
        try:
            if not pycache_preexisted:
                pycache.mkdir()
            for artefact in (cached_pyc, stray_pyc):
                if not artefact.exists():
                    # Bytecode is binary: read_text() on it would also crash.
                    artefact.write_bytes(b"\x00\x0d\x0d\x0a fake bytecode")
                    created.append(artefact)
            test_all_spike_fixtures_carry_the_synthetic_marker()
        finally:
            for artefact in created:
                artefact.unlink()
            if not pycache_preexisted and pycache.exists() and not any(pycache.iterdir()):
                pycache.rmdir()

    def test_meta_test_still_flags_unmarked_fixture_files(self):
        """The artefact filter must not weaken the marker guarantee."""
        rogue_top = FIXTURES / "rogue_unmarked_44.csv"
        rogue_dir = FIXTURES / "subdir_44"
        rogue_nested = rogue_dir / "rogue_unmarked_nested_44.csv"
        try:
            rogue_top.write_text("year,co2\n1959,315.98\n", encoding="utf-8")
            with pytest.raises(AssertionError, match="missing marker"):
                test_all_spike_fixtures_carry_the_synthetic_marker()
            rogue_top.unlink()

            rogue_dir.mkdir()
            rogue_nested.write_text("year,co2\n1959,315.98\n", encoding="utf-8")
            with pytest.raises(AssertionError, match="missing marker"):
                test_all_spike_fixtures_carry_the_synthetic_marker()
        finally:
            for path in (rogue_top, rogue_nested):
                if path.exists():
                    path.unlink()
            if rogue_dir.exists():
                rogue_dir.rmdir()


class TestParseBereiterCo2:
    def test_output_shape_and_ordering(self):
        df = parsers.parse_bereiter_co2(FIXTURES / "bereiter_synthetic.txt")
        assert list(df.columns) == ["age_bp", "co2_ppm", "co2_1s_ppm"]
        assert [str(t) for t in df.dtypes] == ["float64"] * 3
        # Sorted by age ascending even though the fixture is shuffled.
        assert df["age_bp"].is_monotonic_increasing
        assert len(df) == len(_bereiter_fixture_rows())

    def test_bom_crlf_and_negative_ages_survive(self):
        """The BOM, the CRLF endings and a negative (post-1950 firn/Law
        Dome-shaped) age all survive parsing, with values passed through
        exactly. Expectations derive from the fixture text (#51)."""
        youngest_age, youngest_co2, _ = min(_bereiter_fixture_rows())
        assert youngest_age < 0, (
            "the Bereiter fixture must keep at least one negative (post-1950) "
            "age so this invariant stays exercised (#51 regeneration constraint)"
        )
        df = parsers.parse_bereiter_co2(FIXTURES / "bereiter_synthetic.txt")
        youngest = df.iloc[0]
        assert youngest["age_bp"] == youngest_age
        assert youngest["co2_ppm"] == youngest_co2

    def test_wrong_columns_rejected(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("# SYNTHETIC FIXTURE\nage\tco2\n1\t2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            parsers.parse_bereiter_co2(bad)


class TestParseKaufmanTemp12k:
    def test_output_shape(self):
        df = parsers.parse_kaufman_temp12k(FIXTURES / "kaufman_synthetic.csv")
        assert list(df.columns) == ["age_bp", "temp_c", "temp_c_5", "temp_c_95"]
        assert [str(t) for t in df.dtypes] == ["float64"] * 4
        assert df["age_bp"].is_monotonic_increasing
        assert len(df) == 9

    def test_latitude_band_columns_dropped_and_values_kept(self):
        df = parsers.parse_kaufman_temp12k(FIXTURES / "kaufman_synthetic.csv")
        row_100bp = df[df["age_bp"] == 100.0].iloc[0]
        # The native reference bin (1800-1900 CE) is 0 by construction.
        assert row_100bp["temp_c"] == 0.0
        assert row_100bp["temp_c_5"] == -0.3
        assert row_100bp["temp_c_95"] == 0.3

    def test_missing_global_columns_rejected(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("# SYNTHETIC FIXTURE\nages, global_5\n0,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing columns"):
            parsers.parse_kaufman_temp12k(bad)


class TestParseGmlCo2Annual:
    def test_output_shape_and_values(self):
        """Comment lines are skipped and the first (earliest-year) row's
        values pass through exactly. Expectations derive from the fixture
        text (#51)."""
        earliest_year, earliest_co2, _ = min(_gml_fixture_rows())
        df = parsers.parse_gml_co2_annual(FIXTURES / "gml_synthetic.csv")
        assert list(df.columns) == ["year_ce", "co2_ppm", "unc_ppm"]
        assert str(df["year_ce"].dtype) == "int64"
        assert df.iloc[0]["year_ce"] == earliest_year
        assert df.iloc[0]["co2_ppm"] == earliest_co2

    def test_wrong_columns_rejected(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("# SYNTHETIC FIXTURE\nyear,co2\n1959,315.98\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            parsers.parse_gml_co2_annual(bad)


class TestParseGistempAnnual:
    def test_output_shape_and_annual_mean_selection(self):
        """Exactly the J-D (annual mean) column is selected, for every
        usable year in the fixture. Expectations derive from the fixture
        text (#51), so this pins the column choice, not the values."""
        expected = _gistemp_fixture_jd_by_year()
        df = parsers.parse_gistemp_annual(FIXTURES / "gistemp_synthetic.csv")
        assert list(df.columns) == ["year_ce", "temp_anomaly_c"]
        assert str(df["year_ce"].dtype) == "int64"
        parsed = dict(zip(df["year_ce"], df["temp_anomaly_c"], strict=True))
        assert parsed == expected

    def test_missing_value_rows_dropped(self):
        """The in-progress year's J-D is '***' and its row must not appear.
        The fixture must keep at least one such row so the drop path stays
        exercised (#51 regeneration constraint)."""
        lines = (FIXTURES / "gistemp_synthetic.csv").read_text(encoding="utf-8-sig").splitlines()
        header = lines[1].split(",")
        year_column, jd_column = header.index("Year"), header.index("J-D")
        starred_years = {
            int(cells[year_column])
            for cells in (line.split(",") for line in lines[2:])
            if cells[jd_column] == "***"
        }
        assert starred_years, (
            "the GISTEMP fixture must keep an in-progress-year row with J-D "
            "'***' so the drop behaviour stays exercised"
        )
        df = parsers.parse_gistemp_annual(FIXTURES / "gistemp_synthetic.csv")
        assert starred_years.isdisjoint(set(df["year_ce"]))
        assert len(df) == len(_gistemp_fixture_jd_by_year())
        assert not df["temp_anomaly_c"].isna().any()
