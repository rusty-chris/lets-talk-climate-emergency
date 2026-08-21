"""Production chart transforms vs gold fixtures (issue #17) — RED.

Failing tests pinning charts/transforms.py (contract stubs) against the
hand-verified gold CSVs in tests/fixtures/charts/gold/, which are the
byte-for-byte output of the independent generator
evals/scripts/compute_chart_fixtures.py (IMPLEMENTATION.md §5: gold values
come from a second implementation that imports nothing from charts/).

All inputs are SYNTHETIC FIXTURES — invented ramps (#117-safe: no
committed expected value derives from any real dataset). Tolerances per
IMPLEMENTATION.md §5: 1e-9 relative pass-through, 1e-6 post-transform.
Hand-computed spot values are asserted alongside the gold files so the
fixtures cannot silently drift with a regenerated script.
"""

from __future__ import annotations

import ast
import filecmp
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from charts import transforms
from charts.spec import UNIT_CONVERSIONS
from tests._chart_render_fixtures import (
    GOLD_DIR,
    annual_anomaly_frame,
    instr_co2_frame,
    paleo_co2_frame,
    read_gold_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "compute_chart_fixtures.py"

POST_TRANSFORM_TOLERANCE = 1e-6
PASS_THROUGH_TOLERANCE = 1e-9


def _assert_frame_matches_gold(
    result: pd.DataFrame, gold_name: str, tolerance: float = POST_TRANSFORM_TOLERANCE
) -> None:
    gold = read_gold_csv(gold_name)
    assert list(result.columns) == list(gold.columns), (
        f"{gold_name}: columns {list(result.columns)} != gold {list(gold.columns)}"
    )
    assert len(result) == len(gold), f"{gold_name}: {len(result)} rows, gold has {len(gold)}"
    for column in gold.columns:
        if gold[column].dtype == object:
            assert list(result[column]) == list(gold[column]), f"{gold_name}: column {column!r}"
        else:
            pd.testing.assert_series_equal(
                result[column].astype("float64").reset_index(drop=True),
                gold[column].astype("float64").reset_index(drop=True),
                atol=tolerance,
                rtol=tolerance,
                check_names=False,
            )


# ---------------------------------------------------------------------------
# BP -> CE (issue #17 TDD plan item 2)
# ---------------------------------------------------------------------------


def test_bp_to_ce_conversion_1950_epoch():
    """year_ce = 1950 - age_bp, endpoints hand-computed: 10000 BP ->
    -8050 CE and 100 BP -> 1850 CE; output time-ascending with the age
    column replaced."""
    result = transforms.bp_to_ce(paleo_co2_frame())
    assert list(result.columns) == ["year_ce", "co2_ppm"]
    assert result["year_ce"].iloc[0] == pytest.approx(-8050.0, abs=1e-9)
    assert result["year_ce"].iloc[-1] == pytest.approx(1850.0, abs=1e-9)
    assert result["year_ce"].is_monotonic_increasing
    _assert_frame_matches_gold(result, "gold_bp_to_ce.csv", PASS_THROUGH_TOLERANCE)


# ---------------------------------------------------------------------------
# Generic vocabulary ops vs gold fixtures (TDD plan item 1)
# ---------------------------------------------------------------------------


def test_resample_matches_fixture():
    """Consecutive non-overlapping 4-year windows anchored at the first
    year; output year = mean member year. Hand check: 1990-1993 ->
    (1991.5, -0.225); a trailing incomplete window would be dropped."""
    result = transforms.resample(annual_anomaly_frame(), "anomaly_c", 4)
    assert result["year_ce"].iloc[0] == pytest.approx(1991.5, abs=1e-9)
    assert result["anomaly_c"].iloc[0] == pytest.approx(-0.225, abs=1e-9)
    assert len(result) == 5, "20 annual rows resample to exactly five 4-year windows"
    _assert_frame_matches_gold(result, "gold_resample.csv")


def test_resample_drops_trailing_partial_window():
    """A window with fewer than window_years member rows must not emit a
    row — a partial window rendered as settled is the same honesty
    failure as the #108 partial-year datum."""
    frame = annual_anomaly_frame().iloc[:18]  # 18 rows: 4 full windows + 2 leftover
    result = transforms.resample(frame, "anomaly_c", 4)
    assert len(result) == 4
    assert result["year_ce"].max() < 2005, "leftover 2006-2007 rows must not form a window"


def test_rolling_mean_matches_fixture():
    """Centred 5-year window, complete windows only: the linear ramp's
    rolling mean equals the centre value (hand check: 1992 -> -0.20),
    and the edge years 1990-91/2008-09 emit no row."""
    result = transforms.rolling_mean(annual_anomaly_frame(), "anomaly_c", 5)
    assert result["year_ce"].min() == pytest.approx(1992.0)
    assert result["year_ce"].max() == pytest.approx(2007.0)
    centre = result.loc[result["year_ce"] == 1992, "anomaly_c"].iloc[0]
    assert centre == pytest.approx(-0.20, abs=1e-9)
    _assert_frame_matches_gold(result, "gold_rolling_mean.csv")


def test_anomaly_vs_baseline_matches_fixture():
    """Parameterless (#132): subtract the full-record mean. Hand check:
    ramp mean = 0.175, so 1990 -> -0.475 and 2009 -> 0.475 (symmetric)."""
    result = transforms.anomaly_vs_baseline(annual_anomaly_frame(), "anomaly_c")
    assert result["anomaly_c"].iloc[0] == pytest.approx(-0.475, abs=1e-9)
    assert result["anomaly_c"].iloc[-1] == pytest.approx(0.475, abs=1e-9)
    assert result["anomaly_c"].mean() == pytest.approx(0.0, abs=1e-9)
    _assert_frame_matches_gold(result, "gold_anomaly_vs_baseline.csv")


def test_unit_conversion_matches_fixture():
    """degC_anomaly -> degF_anomaly via the code-owned table factor
    (1.8): hand check 1990: -0.30 * 1.8 = -0.54. The renamed column
    follows the gold header (anomaly_f)."""
    assert UNIT_CONVERSIONS[("degC_anomaly", "degF_anomaly")] == pytest.approx(1.8), (
        "the code-owned table and the independent gold script must agree on the factor"
    )
    frame = annual_anomaly_frame().rename(columns={"anomaly_c": "anomaly_f"})
    result = transforms.unit_conversion(frame, "anomaly_f", "degC_anomaly", "degF_anomaly")
    assert result["anomaly_f"].iloc[0] == pytest.approx(-0.54, abs=1e-9)
    _assert_frame_matches_gold(result, "gold_unit_conversion.csv")


def test_unit_conversion_unknown_pair_refuses():
    """A (from, to) pair outside charts.spec.UNIT_CONVERSIONS raises —
    conversion arithmetic is code-owned, never invented (#132)."""
    with pytest.raises(ValueError, match="ppm"):
        transforms.unit_conversion(annual_anomaly_frame(), "anomaly_c", "ppm", "degC_anomaly")


# ---------------------------------------------------------------------------
# Splice (TDD plan item 3; overlap policy per review finding #47)
# ---------------------------------------------------------------------------


def test_splice_series_joins_at_manifest_boundary():
    """The spliced series switches source exactly at the manifest splice
    year: paleo strictly before 1850, instrumental from 1850 onward, no
    smoothing across the join. The overlapping 1850 paleo bin (100 BP)
    is discarded under prefer_instrumental — the #47 disclosure's
    subject — and the instrumental 1850 row is kept."""
    result = transforms.splice_series(
        transforms.bp_to_ce(paleo_co2_frame()),
        instr_co2_frame(),
        1850,
        "prefer_instrumental",
    )
    paleo_rows = result[result["segment"] == "paleo"]
    instrumental_rows = result[result["segment"] == "instrumental"]
    assert paleo_rows["year_ce"].max() < 1850
    assert instrumental_rows["year_ce"].min() == pytest.approx(1850.0)
    # The overlap row: 1850 appears once, from the instrumental side only.
    at_boundary = result[result["year_ce"] == 1850]
    assert list(at_boundary["segment"]) == ["instrumental"]
    assert at_boundary["co2_ppm"].iloc[0] == pytest.approx(285.0, abs=1e-9)
    _assert_frame_matches_gold(
        result[["segment", "year_ce", "co2_ppm"]].reset_index(drop=True),
        "gold_splice.csv",
        PASS_THROUGH_TOLERANCE,
    )


def test_splice_refuses_one_sided_join():
    """Either side contributing zero rows raises: a one-sided 'splice'
    renders no visible join and the mandatory annotation would lie."""
    paleo_ce = transforms.bp_to_ce(paleo_co2_frame())
    with pytest.raises(ValueError):
        transforms.splice_series(paleo_ce, instr_co2_frame(), -9000, "prefer_instrumental")


def test_splice_refuses_unknown_overlap_policy():
    """The overlap policy is the manifest pair's curation-time decision;
    a policy this implementation does not enact must refuse by name,
    never silently fall back to prefer_instrumental (#47)."""
    paleo_ce = transforms.bp_to_ce(paleo_co2_frame())
    with pytest.raises((ValueError, NotImplementedError), match="show_both"):
        transforms.splice_series(paleo_ce, instr_co2_frame(), 1850, "show_both")


# ---------------------------------------------------------------------------
# Rebaseline (TDD plan item 4)
# ---------------------------------------------------------------------------


def test_rebaseline_uses_manifest_alignment_period():
    """Rebaselined values match the gold fixture for the manifest-fixed
    period [1990, 1999] (the synthetic pair's alignment_period_ce — a
    curation-time decision the caller reads from the manifest, ADR-020).
    Hand check: period mean -0.075, so every value shifts by +0.075 and
    the period mean becomes exactly zero."""
    manifest_fixed_period = (1990, 1999)  # mirrors render_manifest()'s pair semantics
    result = transforms.rebaseline(annual_anomaly_frame(), "anomaly_c", manifest_fixed_period)
    window = result[(result["year_ce"] >= 1990) & (result["year_ce"] <= 1999)]
    assert window["anomaly_c"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["anomaly_c"].iloc[0] == pytest.approx(-0.225, abs=1e-9)
    _assert_frame_matches_gold(result, "gold_rebaseline.csv")


def test_rebaseline_refuses_empty_alignment_period():
    """A period with no data rows raises — a silent empty mean would
    rebaseline by NaN and poison the series."""
    with pytest.raises(ValueError):
        transforms.rebaseline(annual_anomaly_frame(), "anomaly_c", (1800, 1810))


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_transforms_do_not_mutate_input_frames():
    """Pure over DataFrames (IMPLEMENTATION.md §1): the input frame is
    unchanged by every transform."""
    original = annual_anomaly_frame()
    frame = original.copy()
    transforms.rolling_mean(frame, "anomaly_c", 5)
    transforms.anomaly_vs_baseline(frame, "anomaly_c")
    transforms.rebaseline(frame, "anomaly_c", (1990, 1999))
    transforms.resample(frame, "anomaly_c", 4)
    pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# The independent gold-fixture generator (IMPLEMENTATION.md §5)
# ---------------------------------------------------------------------------


def test_fixture_script_imports_nothing_from_charts():
    """Import-graph rule: the generator is a second, independent
    implementation — it must not import charts/ (or pandas, which would
    tempt sharing arithmetic helpers): stdlib/numpy only."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"charts", "pandas", "tests"}
    assert not (imported & forbidden), (
        f"compute_chart_fixtures.py imports {sorted(imported & forbidden)} — the gold "
        "generator must stay import-graph-independent of the code it checks"
    )


def test_committed_gold_fixtures_match_generator_output(tmp_path):
    """The committed gold CSVs are byte-for-byte the script's output — a
    hand edit to either side fails here, so the fixtures and their
    provenance can never drift apart."""
    spec = importlib.util.spec_from_file_location("compute_chart_fixtures", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("compute_chart_fixtures", None)
    spec.loader.exec_module(module)

    written = module.write_gold_fixtures(tmp_path)
    committed = sorted(GOLD_DIR.glob("*.csv"))
    assert sorted(p.name for p in written) == [p.name for p in committed]
    for path in written:
        assert filecmp.cmp(path, GOLD_DIR / path.name, shallow=False), (
            f"{path.name}: committed gold fixture differs from the independent "
            "generator's output — regenerate via evals/scripts/compute_chart_fixtures.py "
            "and explain the diff in the same commit (IMPLEMENTATION.md §5)"
        )
