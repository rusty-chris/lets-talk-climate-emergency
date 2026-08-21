"""Shared synthetic fixtures for the issue #17 renderer red suite.

Everything in this module is a SYNTHETIC FIXTURE — invented datasets,
invented coverages, invented values — authored for these tests (ADR-023 /
review finding #117: nothing here derives from Kaufman/Bereiter or any
real dataset; the flagship renders only from origin fetches). The two
committed source CSVs it loads (``tests/fixtures/charts/``) carry the
first-line SYNTHETIC FIXTURE marker and invented ramps.

Layout mirrors the #15 suite's builders (tests/unit/test_chartspec.py):
fresh copies per call, because tests mutate. The manifest here also
carries the caption-facing fields (``attribution_text``, ``licence``,
``access_date``) the renderer must compose captions from (vocabulary
amendment 9 / review finding #137).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from charts.spec import RenderValidatedSpec, validate_spec_for_render

REPO_ROOT = Path(__file__).resolve().parents[1]
CHART_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "charts"
GOLD_DIR = CHART_FIXTURES / "gold"

#: Deployment config, never spec content: the site URL baked into captions.
SITE_URL = "https://climate-chat.example.test"

ACCESS_DATE = "2026-08-18"

#: The manifest rebaseline disclosure that must reach the artefact
#: verbatim (review finding #50).
TEMP_ALIGNMENT_DISCLOSURE = (
    "syn_instr_temp shifted so its 1880-1900 mean equals the 1400-1500 reference (invented)"
)

#: The manifest overlap disclosures that must reach the artefact
#: (review finding #47).
CO2_OVERLAP_NOTE = (
    "Overlapping paleo sample at the 1850 splice omitted in favour of the "
    "instrumental record (invented)"
)
TEMP_OVERLAP_NOTE = (
    "Overlapping instrumental years 1850-1879 before the 1880 splice omitted in "
    "favour of the proxy reconstruction's native span (invented)"
)

CO2_RESOLUTION_NOTE = "Invented: centennial paleo bins before 1850; annual instrumental after."
TEMP_RESOLUTION_NOTE = "Invented: proxy bins before 1880; annual instrumental after."


def render_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: the renderer-facing pack manifest (raw yaml shape)."""
    return {
        "version": "0.0.1-test",
        "access_date": ACCESS_DATE,
        "datasets": {
            "syn_annual_anomaly": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {
                    "name": "anomaly_c",
                    "unit": "degC_anomaly",
                    "native_reference_period": "full record (invented)",
                },
                "coverage": {"first_year_ce": 1990, "last_year_ce": 2009},
                "attribution_text": "Meridian Climate Service (fictional)",
                "licence": "Public domain (invented stand-in)",
                "retrieved_at": ACCESS_DATE,
            },
            "syn_paleo_co2": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {"name": "co2_ppm", "unit": "ppm"},
                "coverage": {"oldest_bp": 10000, "youngest_bp": 100},
                "attribution_text": "Aurelian Ice-Core Consortium (fictional)",
                "licence": "Open with required credit (invented)",
                "retrieved_at": ACCESS_DATE,
            },
            "syn_instr_co2": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "co2_ppm", "unit": "ppm"},
                "coverage": {"first_year_ce": 1850, "last_year_ce": 2009},
                "attribution_text": "Windvane Observatory (fictional)",
                "licence": "Public domain (invented stand-in)",
                "retrieved_at": ACCESS_DATE,
            },
            "syn_paleo_temp": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {
                    "name": "temp_c",
                    "unit": "degC_anomaly",
                    "native_reference_period": "1400-1500 CE (invented)",
                },
                "coverage": {"oldest_bp": 10000, "youngest_bp": 100},
                "attribution_text": "Basin Reconstruction Team (fictional)",
                "licence": "Open with required credit (invented)",
                "retrieved_at": ACCESS_DATE,
            },
            "syn_instr_temp": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {
                    "name": "temp_anomaly_c",
                    "unit": "degC_anomaly",
                    "native_reference_period": "1950-1980 (invented)",
                },
                "coverage": {"first_year_ce": 1850, "last_year_ce": 2009},
                "attribution_text": "Meridian Climate Service (fictional)",
                "licence": "Public domain (invented stand-in)",
                "retrieved_at": ACCESS_DATE,
            },
        },
        "splice_pairs": [
            {
                "id": "syn_co2_splice",
                "paleo": "syn_paleo_co2",
                "instrumental": "syn_instr_co2",
                "splice_year_ce": 1850,
                "rationale": "synthetic",
                "rebaseline": None,
                "resolution_note": CO2_RESOLUTION_NOTE,
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "syn_paleo_co2 bin at 1850 CE (post-splice)",
                    "note": CO2_OVERLAP_NOTE,
                },
            },
            {
                "id": "syn_temp_splice",
                "paleo": "syn_paleo_temp",
                "instrumental": "syn_instr_temp",
                "splice_year_ce": 1880,
                "rationale": "synthetic",
                "rebaseline": {
                    "apply_to": "syn_instr_temp",
                    "alignment_period_ce": [1880, 1900],
                    "display_reference": "1400-1500 CE (invented native reference)",
                    "disclosure": TEMP_ALIGNMENT_DISCLOSURE,
                    "rationale": "synthetic",
                },
                "resolution_note": TEMP_RESOLUTION_NOTE,
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "syn_instr_temp years 1850-1879 (pre-splice)",
                    "note": TEMP_OVERLAP_NOTE,
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Pre-landed frames (parser-output shape, per dataset id — ADR-023: the
# renderer consumes these mappings only, it never loads or fetches)
# ---------------------------------------------------------------------------


def _read_fixture_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CHART_FIXTURES / name, comment="#")


def read_gold_csv(name: str) -> pd.DataFrame:
    """A committed gold expected-values CSV (tests/fixtures/charts/gold/)."""
    return pd.read_csv(GOLD_DIR / name, comment="#")


def annual_anomaly_frame() -> pd.DataFrame:
    """The committed invented ramp, in parser-output shape (year_ce)."""
    return _read_fixture_csv("synthetic_annual_anomaly.csv").rename(columns={"year": "year_ce"})


def paleo_co2_frame() -> pd.DataFrame:
    return _read_fixture_csv("synthetic_paleo_bp.csv")


def instr_co2_frame() -> pd.DataFrame:
    return _read_fixture_csv("synthetic_instrumental_co2.csv")


def paleo_temp_frame() -> pd.DataFrame:
    """Invented proxy reconstruction with a shipped 5-95% band.

    Ages 10000..100 BP -> years -8050..1850 CE; values a ramp
    -0.8..0.0 degC; band = value -/+ 0.3.
    """
    ages = [10000.0, 8000.0, 6000.0, 4000.0, 2000.0, 1000.0, 500.0, 150.0, 100.0]
    values = [-0.8 + 0.1 * i for i in range(len(ages))]
    return pd.DataFrame(
        {
            "age_bp": ages,
            "temp_c": values,
            "temp_c_5": [v - 0.3 for v in values],
            "temp_c_95": [v + 0.3 for v in values],
        }
    )


def instr_temp_frame() -> pd.DataFrame:
    """Invented instrumental anomaly ramp: 1850..2009, -0.2 + 0.01*(y-1850)."""
    years = list(range(1850, 2010))
    return pd.DataFrame(
        {
            "year_ce": years,
            "temp_anomaly_c": [-0.2 + 0.01 * (y - 1850) for y in years],
        }
    )


def frames() -> dict[str, pd.DataFrame]:
    """Fresh pre-landed frames for every synthetic dataset id."""
    return {
        "syn_annual_anomaly": annual_anomaly_frame(),
        "syn_paleo_co2": paleo_co2_frame(),
        "syn_instr_co2": instr_co2_frame(),
        "syn_paleo_temp": paleo_temp_frame(),
        "syn_instr_temp": instr_temp_frame(),
    }


# ---------------------------------------------------------------------------
# Spec builders (fresh copies per call) with hand-computed extents
# ---------------------------------------------------------------------------


def line_spec() -> dict[str, Any]:
    """A minimal single-series line chart over the committed anomaly ramp."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-line",
        "chart_type": "line",
        "title": "Synthetic anomaly over time",
        "time_range_ce": [1990, 2009],
        "series": [
            {
                "id": "anom",
                "label": "Anomaly (degC, invented)",
                "unit": "degC_anomaly",
                "dataset": "syn_annual_anomaly",
                "baseline": {"value": 0, "label": "0 = full-record average (invented)"},
            }
        ],
    }


#: Ramp -0.30..0.65 over 1990-2009 (hand-computed from the committed CSV).
LINE_EXTENTS = {"anom": (-0.30, 0.65)}


def two_series_line_spec() -> dict[str, Any]:
    """Two same-unit series on one axis — the direct-labelling/palette case."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-two-lines",
        "chart_type": "line",
        "title": "Two synthetic anomaly series",
        "time_range_ce": [1990, 2009],
        "series": [
            {
                "id": "anom",
                "label": "Meridian anomaly (invented)",
                "unit": "degC_anomaly",
                "dataset": "syn_annual_anomaly",
            },
            {
                "id": "instr",
                "label": "Instrumental anomaly (invented)",
                "unit": "degC_anomaly",
                "dataset": "syn_instr_temp",
            },
        ],
    }


#: anom: the ramp; instr: -0.2+0.01*(y-1850) over 1990-2009 = 1.20..1.39.
TWO_SERIES_EXTENTS = {"anom": (-0.30, 0.65), "instr": (1.20, 1.39)}


def area_spec() -> dict[str, Any]:
    spec = line_spec()
    spec["chart_id"] = "syn-area"
    spec["chart_type"] = "area"
    return spec


def bar_spec(scale_domain: list[float] | None = None) -> dict[str, Any]:
    """A bar chart over an all-positive absolute quantity (the zero-rule case)."""
    series: dict[str, Any] = {
        "id": "co2",
        "label": "CO2 (ppm, invented)",
        "unit": "ppm",
        "dataset": "syn_instr_co2",
    }
    if scale_domain is not None:
        series["scale_domain"] = list(scale_domain)
        if scale_domain[0] > 0 or scale_domain[1] < 0:
            series["annotations"] = {
                "non_zero_baseline": {"label": "axis starts above zero (invented)"}
            }
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-bar",
        "chart_type": "bar",
        "title": "Synthetic CO2 bars",
        "time_range_ce": [1850, 2009],
        "series": [series],
    }


#: 285 + 0.25*(y-1850) over 1850-2009 = 285.00..324.75.
BAR_EXTENTS = {"co2": (285.0, 324.75)}


def _co2_series() -> dict[str, Any]:
    return {
        "id": "co2",
        "label": "CO2 (ppm, invented)",
        "axis": "left",
        "unit": "ppm",
        "splice_series": ["syn_paleo_co2", "syn_instr_co2"],
        "splice_pair_id": "syn_co2_splice",
        "overlap_policy": "prefer_instrumental",
        "scale_domain": [250, 340],
        "annotations": {
            "splice_point": {"year_ce": 1850, "label": "splice: paleo → instrumental (1850)"},
            "resolution_note": CO2_RESOLUTION_NOTE,
            "non_zero_baseline": {"label": "CO2 axis starts at 250 ppm — no zero (invented)"},
        },
    }


def _temp_series() -> dict[str, Any]:
    return {
        "id": "temp",
        "label": "Temperature anomaly (degC vs 1400-1500, invented)",
        "axis": "right",
        "unit": "degC_anomaly",
        "splice_series": ["syn_paleo_temp", "syn_instr_temp"],
        "splice_pair_id": "syn_temp_splice",
        "overlap_policy": "prefer_instrumental",
        "rebaseline_to": {
            "apply_to": "syn_instr_temp",
            "alignment_period_ce": [1880, 1900],
            "alignment_disclosure": TEMP_ALIGNMENT_DISCLOSURE,
        },
        "scale_domain": [-2.0, 2.0],
        "uncertainty_band": {"lower": "temp_c_5", "upper": "temp_c_95", "source": "syn_paleo_temp"},
        "baseline": {"value": 0, "label": "0 = 1400-1500 average (invented)"},
        "annotations": {
            "splice_point": {"year_ce": 1880, "label": "splice: proxy → instrumental (1880)"},
            "resolution_note": TEMP_RESOLUTION_NOTE,
        },
    }


def dual_axis_spec() -> dict[str, Any]:
    """The flagship-shaped dual-axis chart: spliced CO2 left, spliced +
    rebaselined temperature (with band and zero baseline) right."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-dual",
        "chart_type": "dual_axis_line",
        "title": "Synthetic CO2 and temperature",
        "time_axis": {"calendar": "CE", "convert_bp": True},
        "time_range_ce": [-8050, 2009],
        "series": [_co2_series(), _temp_series()],
    }


#: Hand-computed post-transform extents over the full range:
#: co2 = paleo<1850 (260..274.25) U instrumental 1850-2009 (285..324.75);
#: temp = paleo (-0.8..0.0) U instrumental>=1880 shifted by its
#: 1880-1900 mean (0.2): (-0.1..1.19).
DUAL_EXTENTS = {"co2": (260.0, 324.75), "temp": (-0.8, 1.19)}


def panel_spec(recent_range: tuple[float, float] = (1900, 2009)) -> dict[str, Any]:
    """The context+recent-inset pair over the same two spliced series."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-panel",
        "chart_type": "context_recent_inset",
        "title": "Synthetic CO2 and temperature, the long view",
        "time_axis": {"calendar": "CE", "convert_bp": True},
        "panels": {
            "context": {"label": "Last 10,000 years (invented)", "time_range_ce": [-8050, 2009]},
            "recent": {
                "label": "Recent (invented)",
                "time_range_ce": list(recent_range),
                "shared_y_scale_with": "context",
            },
        },
        "series": [_co2_series(), _temp_series()],
    }


PANEL_EXTENTS = DUAL_EXTENTS


def validated(
    spec: dict[str, Any],
    extents: dict[str, tuple[float, float]],
    manifest: dict[str, Any] | None = None,
) -> RenderValidatedSpec:
    """Mint the RenderValidatedSpec token the renderer entry points demand."""
    return validate_spec_for_render(spec, manifest or render_manifest(), extents)


# ---------------------------------------------------------------------------
# Vega-Lite walking helpers (structural assertions over the emitted JSON)
# ---------------------------------------------------------------------------


def iter_dicts(node: Any):
    """Every dict reachable in a (JSON-shaped) Vega-Lite structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from iter_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_dicts(value)


def mark_types(vl: Any) -> set[str]:
    """Every mark type used anywhere in the chart."""
    types: set[str] = set()
    for node in iter_dicts(vl):
        mark = node.get("mark")
        if isinstance(mark, str):
            types.add(mark)
        elif isinstance(mark, dict) and isinstance(mark.get("type"), str):
            types.add(mark["type"])
    return types


def layers_with_y(vl: Any) -> list[dict[str, Any]]:
    """Every node carrying an encoding with a y channel."""
    return [
        node
        for node in iter_dicts(vl)
        if isinstance(node.get("encoding"), dict) and "y" in node["encoding"]
    ]


def data_values(vl: Any) -> list[dict[str, Any]]:
    """Every inline datum in the chart (each row of every data.values)."""
    rows: list[dict[str, Any]] = []
    for node in iter_dicts(vl):
        data = node.get("data")
        if isinstance(data, dict) and isinstance(data.get("values"), list):
            rows.extend(r for r in data["values"] if isinstance(r, dict))
    return rows
