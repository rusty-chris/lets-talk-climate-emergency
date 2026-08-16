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
        assert len(df) == 6

    def test_bom_crlf_and_negative_ages_survive(self):
        df = parsers.parse_bereiter_co2(FIXTURES / "bereiter_synthetic.txt")
        youngest = df.iloc[0]
        assert youngest["age_bp"] == -51.03  # post-1950 firn/Law Dome sample
        assert youngest["co2_ppm"] == 368.02

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
        df = parsers.parse_gml_co2_annual(FIXTURES / "gml_synthetic.csv")
        assert list(df.columns) == ["year_ce", "co2_ppm", "unc_ppm"]
        assert str(df["year_ce"].dtype) == "int64"
        assert df.iloc[0]["year_ce"] == 1959
        assert df.iloc[0]["co2_ppm"] == 315.98

    def test_wrong_columns_rejected(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("# SYNTHETIC FIXTURE\nyear,co2\n1959,315.98\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            parsers.parse_gml_co2_annual(bad)


class TestParseGistempAnnual:
    def test_output_shape_and_annual_mean_selection(self):
        df = parsers.parse_gistemp_annual(FIXTURES / "gistemp_synthetic.csv")
        assert list(df.columns) == ["year_ce", "temp_anomaly_c"]
        assert str(df["year_ce"].dtype) == "int64"
        assert df[df["year_ce"] == 1880].iloc[0]["temp_anomaly_c"] == -0.20  # the J-D column
        assert df[df["year_ce"] == 2024].iloc[0]["temp_anomaly_c"] == 1.28

    def test_missing_value_rows_dropped(self):
        df = parsers.parse_gistemp_annual(FIXTURES / "gistemp_synthetic.csv")
        # 2026 has J-D = '***' (year in progress) and must not appear.
        assert 2026 not in set(df["year_ce"])
        assert len(df) == 5
        assert not df["temp_anomaly_c"].isna().any()
