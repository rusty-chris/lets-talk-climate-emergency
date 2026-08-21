"""Independent chart-fixture generator (issue #20; IMPLEMENTATION.md §5).

Computes the expected rendered values for every expected-spec item in
``evals/gold/chart_requests.yaml`` and commits them to
``evals/gold/chart_fixtures.json`` — the DESIGN §6.2 chart
data-faithfulness contract. The whole point of this script is
**independence from the pipeline under test**: it imports NOTHING from
``charts/`` (enforced by ``test_fixture_script_imports_nothing_from_charts``)
and re-implements the transform arithmetic from the written contracts
(DESIGN §3.7, datasets-manifest semantics, charts/CHARTSPEC.md) using the
stdlib only, so a bug in the renderer cannot silently agree with a fixture
derived from the same code (the non-tautology guarantee).

Data source: the committed synthetic CSVs under ``evals/gold/synthetic_data/``
(regenerable, deterministically, with ``--write-data``). Synthetic-only is
deliberate (review finding #117 / issue #23): committed fixtures must not
embed values derived from the real pack's provisional datasets, and
ADR-023 keeps real data files out of git. The real-manifest flagship item
carries no fixture — a recorded gap in evals/gold/COVERAGE.md.

Transform semantics pinned by these fixtures (the renderer #17 contract):

- **time filter**: a series keeps rows with ``start <= year_ce <= end``
  (inclusive both ends); for ``context_recent_inset`` the context panel's
  range is the rendered extent (the recent panel is a zoom, not a refilter).
- **BP -> CE**: ``year_ce = present_ce - age_bp`` (present_ce from the
  dataset's manifest ``time_axis``; fractional and negative ages allowed).
- **splice** (``prefer_instrumental``): paleo rows strictly before the
  manifest ``splice_year_ce``; instrumental rows from it onward — the
  overlapping paleo samples are the manifest-disclosed discard.
- **rebaseline**: the manifest ``apply_to`` member is shifted by minus its
  own mean over ``alignment_period_ce`` (inclusive), before splicing.
- **rolling_mean(window_years=w)**: centred window — the value at year y
  is the arithmetic mean of all samples with ``|year - y| <= w / 2``
  (inclusive), computed over the assembled series before time filtering.
- **unit_conversion**: multiply by the code-owned factor (the table below
  is an independent copy of charts.spec.UNIT_CONVERSIONS by design —
  duplication IS the independence).
- transform order: assemble (BP->CE, rebaseline, splice) -> per-series
  ``transforms`` list in order -> time filter.

Tolerances (DESIGN §6.2): 1e-9 relative for pass-through series (no
splice, no transforms), 1e-6 relative post-transform.

Usage:
    python evals/scripts/compute_chart_fixtures.py --write-data   # CSVs
    python evals/scripts/compute_chart_fixtures.py                # fixtures
    python evals/scripts/compute_chart_fixtures.py --check        # verify
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "evals" / "gold"
DATA_DIR = GOLD_DIR / "synthetic_data"
MANIFEST_PATH = GOLD_DIR / "chart_pack_fixture.yaml"
REQUESTS_PATH = GOLD_DIR / "chart_requests.yaml"
FIXTURES_PATH = GOLD_DIR / "chart_fixtures.json"

SYNTHETIC_MARKER = "# SYNTHETIC FIXTURE - generated deterministically for this project's tests"

#: Independent copy of the code-owned conversion table (charts.spec
#: UNIT_CONVERSIONS). Kept in sync by the gold items that exercise it: a
#: divergence makes the fixture and the renderer disagree loudly.
UNIT_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    ("Mt CO2/yr", "Gt CO2/yr"): 1e-3,
    ("degC_anomaly", "degF_anomaly"): 1.8,
}

PASS_THROUGH_TOLERANCE = 1e-9
POST_TRANSFORM_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Synthetic data generation (--write-data): closed-form, deterministic
# ---------------------------------------------------------------------------


def _temp_modern_rows() -> list[tuple[str, str]]:
    rows = []
    for year in range(1880, 2021):
        t = year - 1880
        value = -0.30 + 0.011 * t + 0.08 * math.sin(2 * math.pi * t / 11)
        rows.append((str(year), f"{value:.4f}"))
    return rows


def _co2_modern_rows() -> list[tuple[str, str]]:
    rows = []
    for year in range(1959, 2021):
        t = year - 1959
        value = 316.0 + 0.9 * t + 6.0 * (t / 61) ** 2
        rows.append((str(year), f"{value:.3f}"))
    return rows


def _temp_paleo_rows() -> list[tuple[str, str]]:
    rows = []
    for bp in range(12000, -1, -100):
        value = 0.25 * math.sin(math.pi * bp / 9000) - 0.05
        rows.append((str(bp), f"{value:.4f}"))
    return rows


def _co2_paleo_rows() -> list[tuple[str, str]]:
    ages = list(range(10000, -1, -100)) + [-10, -20, -30]
    rows = []
    for bp in ages:
        value = 260.0 + 20.0 * (1 - bp / 10000)
        rows.append((str(bp), f"{value:.3f}"))
    return rows


def _emissions_rows() -> list[tuple[str, str]]:
    rows = []
    for year in range(1850, 2021):
        value = 10.0 * math.exp(0.03 * (year - 1850))
        rows.append((str(year), f"{value:.2f}"))
    return rows


SYNTHETIC_DATASETS: dict[str, tuple[tuple[str, str], Any]] = {
    "syn_temp_modern.csv": (("year", "temp_anomaly_c"), _temp_modern_rows),
    "syn_co2_modern.csv": (("year", "co2_ppm"), _co2_modern_rows),
    "syn_temp_paleo.csv": (("age_bp", "temp_anomaly_c"), _temp_paleo_rows),
    "syn_co2_paleo.csv": (("age_bp", "co2_ppm"), _co2_paleo_rows),
    "syn_emissions.csv": (("year", "co2_emissions_mt"), _emissions_rows),
}


def write_data() -> None:
    """Regenerate the committed synthetic CSVs (deterministic)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (header, row_fn) in SYNTHETIC_DATASETS.items():
        path = DATA_DIR / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(SYNTHETIC_MARKER + "\n")
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(row_fn())
        print(f"wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Independent spec interpretation
# ---------------------------------------------------------------------------


def _load_csv_series(path: Path) -> list[tuple[float, float]]:
    """(x, value) rows from a fixture CSV, skipping the marker comment."""
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    for record in csv.DictReader(lines):
        keys = list(record)
        rows.append((float(record[keys[0]]), float(record[keys[1]])))
    return rows


def _dataset_points(manifest: dict[str, Any], dataset_id: str) -> list[tuple[float, float]]:
    """A dataset's (year_ce, value) points: CSV load + BP->CE conversion."""
    entry = manifest["datasets"][dataset_id]
    points = _load_csv_series(REPO_ROOT / entry["data_file"])
    time_axis = entry["time_axis"]
    if time_axis["unit"] == "years_bp":
        present = float(time_axis["present_ce"])
        points = [(present - age, value) for age, value in points]
    elif time_axis["unit"] != "year_ce":
        raise ValueError(f"unknown time-axis unit {time_axis['unit']!r} on {dataset_id}")
    return sorted(points)


def _rebaselined(
    points: list[tuple[float, float]], period: tuple[float, float]
) -> list[tuple[float, float]]:
    start, end = period
    window = [value for year, value in points if start <= year <= end]
    if not window:
        raise ValueError(f"rebaseline period {start}-{end} contains no data rows")
    shift = sum(window) / len(window)
    return [(year, value - shift) for year, value in points]


def _splice_pair(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    for pair in manifest["splice_pairs"]:
        if pair["id"] == pair_id:
            return pair
    raise ValueError(f"unknown splice pair {pair_id!r}")


def _assemble_series(
    series: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[tuple[float, float]], float | None]:
    """The assembled (year_ce, value) points for one spec series, plus the
    splice year for spliced series (None otherwise)."""
    if "dataset" in series:
        return _dataset_points(manifest, series["dataset"]), None

    pair = _splice_pair(manifest, series["splice_pair_id"])
    if pair.get("overlap", {}).get("policy") != "prefer_instrumental":
        raise ValueError(
            f"pair {pair['id']!r}: only prefer_instrumental splice semantics are "
            "pinned by these fixtures"
        )
    paleo = _dataset_points(manifest, pair["paleo"])
    instrumental = _dataset_points(manifest, pair["instrumental"])

    rebaseline = pair.get("rebaseline")
    if ("rebaseline_to" in series) != bool(rebaseline):
        raise ValueError(f"pair {pair['id']!r}: spec/manifest rebaseline mismatch")
    if rebaseline:
        period = tuple(float(y) for y in rebaseline["alignment_period_ce"])
        target = rebaseline["apply_to"]
        if target == pair["instrumental"]:
            instrumental = _rebaselined(instrumental, period)
        elif target == pair["paleo"]:
            paleo = _rebaselined(paleo, period)
        else:
            raise ValueError(f"pair {pair['id']!r}: apply_to {target!r} is not a member")

    splice_year = float(pair["splice_year_ce"])
    left = [(year, value) for year, value in paleo if year < splice_year]
    right = [(year, value) for year, value in instrumental if year >= splice_year]
    if not left or not right:
        raise ValueError(f"pair {pair['id']!r}: one-sided splice")
    return sorted(left + right), splice_year


def _rolling_mean(
    points: list[tuple[float, float]], window_years: float
) -> list[tuple[float, float]]:
    half = window_years / 2.0
    out = []
    for year, _ in points:
        window = [v for y, v in points if abs(y - year) <= half]
        out.append((year, sum(window) / len(window)))
    return out


def _apply_transforms(
    points: list[tuple[float, float]],
    series: dict[str, Any],
    manifest: dict[str, Any],
) -> list[tuple[float, float]]:
    source_units = {
        manifest["datasets"][ds_id]["variable"]["unit"]
        for ds_id in (
            [series["dataset"]] if "dataset" in series else list(series.get("splice_series") or [])
        )
    }
    for transform in series.get("transforms") or []:
        op = transform["op"]
        if op == "rolling_mean":
            points = _rolling_mean(points, float(transform["window_years"]))
        elif op == "unit_conversion":
            to_unit = transform["to_unit"]
            factors = {
                UNIT_CONVERSION_FACTORS.get((source_unit, to_unit)) for source_unit in source_units
            }
            if len(factors) != 1 or None in factors:
                raise ValueError(
                    f"no single conversion from {sorted(source_units)!r} to {to_unit!r}"
                )
            (factor,) = factors
            points = [(year, value * factor) for year, value in points]
        else:
            raise ValueError(
                f"transform op {op!r} is not exercised by the gold set; add its "
                "independent semantics here before using it in a gold spec"
            )
    return points


def _series_time_range(spec: dict[str, Any]) -> tuple[float, float]:
    if spec["chart_type"] == "context_recent_inset":
        pair = spec["panels"]["context"]["time_range_ce"]
    else:
        pair = spec["time_range_ce"]
    return float(pair[0]), float(pair[1])


def compute_fixture_for_spec(spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The expected-rendered-values fixture body for one gold ChartSpec."""
    start, end = _series_time_range(spec)
    series_out: dict[str, Any] = {}
    derived_from: set[str] = set()
    post_transform = False
    for series in spec["series"]:
        for ds_id in [series["dataset"]] if "dataset" in series else list(series["splice_series"]):
            derived_from.add(manifest["datasets"][ds_id]["data_file"])
        points, splice_year = _assemble_series(series, manifest)
        if splice_year is not None or series.get("transforms"):
            post_transform = True
        points = _apply_transforms(points, series, manifest)
        points = [(year, value) for year, value in points if start <= year <= end]
        entry: dict[str, Any] = {
            "n_points": len(points),
            "points": [[year, value] for year, value in points],
        }
        if splice_year is not None:
            # Segment counts within the rendered range, so the committed
            # counts describe what is actually drawn either side of the
            # rendered splice marker.
            entry["segments"] = {
                "paleo": sum(1 for year, _ in points if year < splice_year),
                "instrumental": sum(1 for year, _ in points if year >= splice_year),
            }
        series_out[series["id"]] = entry
    return {
        "derived_from": sorted(derived_from),
        "transform_kind": "post-transform" if post_transform else "pass-through",
        "tolerance_relative": (
            POST_TRANSFORM_TOLERANCE if post_transform else PASS_THROUGH_TOLERANCE
        ),
        "series": series_out,
    }


def compute_fixtures() -> dict[str, Any]:
    """All fixtures for the gold chart set, keyed by fixture id.

    Items sharing a fixture id (the cherry-pick and flatten-attack answers
    reuse the honest full-range chart) must compute identical values — a
    disagreement is a gold-set authoring bug and raises.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    requests = yaml.safe_load(REQUESTS_PATH.read_text(encoding="utf-8"))
    fixtures: dict[str, Any] = {}
    for item in requests["items"]:
        if item.get("expected") != "spec":
            continue
        fixture_id = item.get("fixture")
        if not fixture_id:
            raise ValueError(f"expected-spec item {item['id']!r} has no fixture id")
        body = compute_fixture_for_spec(item["spec"], manifest)
        if fixture_id in fixtures and fixtures[fixture_id] != body:
            raise ValueError(
                f"fixture id {fixture_id!r} is shared by items whose specs render "
                "different values — gold-set authoring bug"
            )
        fixtures[fixture_id] = body
    return {
        "_meta": {
            # The recorded-envelope provenance declaration (finding #67 /
            # ADR-023 committed-data check): provenance + content_signoff
            # are what exempts this .json from the no-dataset-files rule.
            "content_signoff": {
                "who": "issue #20 gold-set author (Fable)",
                "date": "2026-08-21",
                "note": (
                    "reviewed: fixture ids, tolerances, series shapes and the "
                    "transform semantics note; every value derives from the "
                    "committed synthetic CSVs, no real dataset values present"
                ),
            },
            "provenance": (
                "SYNTHETIC FIXTURE - expected rendered values computed by the "
                "independent generator evals/scripts/compute_chart_fixtures.py "
                "from the committed synthetic CSVs under evals/gold/synthetic_data/; "
                "imports nothing from charts/ (the non-tautology guarantee, "
                "IMPLEMENTATION.md section 5). Regenerate: python "
                "evals/scripts/compute_chart_fixtures.py"
            ),
            "semantics": (
                "inclusive time filter on the context/spec range; BP->CE via "
                "present_ce - age_bp; splice keeps paleo rows strictly before "
                "splice_year_ce and instrumental rows from it onward "
                "(prefer_instrumental); rebaseline shifts the manifest apply_to "
                "member by minus its own mean over alignment_period_ce before "
                "splicing; rolling_mean is a centred window (|year - y| <= "
                "window_years/2); unit_conversion multiplies by the code-owned "
                "factor. Tolerances: 1e-9 relative pass-through, 1e-6 relative "
                "post-transform (DESIGN section 6.2)."
            ),
        },
        "fixtures": fixtures,
    }


def _serialise(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true", help="regenerate the synthetic CSVs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixtures match a fresh recompute",
    )
    args = parser.parse_args(argv)
    if args.write_data:
        write_data()
        return 0
    payload = compute_fixtures()
    if args.check:
        committed = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
        if committed != payload:
            print("chart_fixtures.json does NOT match a fresh recompute", file=sys.stderr)
            return 1
        print("chart_fixtures.json matches a fresh recompute")
        return 0
    FIXTURES_PATH.write_text(_serialise(payload), encoding="utf-8")
    print(f"wrote {FIXTURES_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
