"""Independent chart-fixture generator (IMPLEMENTATION.md §5; issues #17 + #20).

One script, two fixture families, one guarantee — every expected value is
produced by a second, independent implementation of the transform
arithmetic that imports **nothing from charts/** (and no pandas), enforced
by import-graph tests in tests/unit/test_chart_transforms.py and
tests/unit/test_gold_sets.py, so a bug in the pipeline under test cannot
silently agree with its own fixtures (the non-tautology guarantee):

1. **Per-transform gold CSVs** (issue #17): ``tests/fixtures/charts/gold/``
   computed from the synthetic source CSVs in ``tests/fixtures/charts/``.
   ``tests/unit/test_chart_transforms.py`` pins ``charts/transforms.py``
   against them byte-for-byte. Regenerate: ``--transform-golds``.

2. **Eval-level rendered-value fixtures** (issue #20):
   ``evals/gold/chart_fixtures.json`` — the expected rendered values for
   every expected-spec item in ``evals/gold/chart_requests.yaml``, computed
   over the committed synthetic CSVs in ``evals/gold/synthetic_data/``
   (regenerable with ``--write-data``). The DESIGN §6.2 chart
   data-faithfulness contract for the #21 harness. Regenerate: default run.

Every input and output is a SYNTHETIC FIXTURE — invented values authored
for this repo's tests (ADR-023 / review finding #117: no expected values
derived from real datasets are committed anywhere; the flagship's real
data renders only from origin fetches, never from committed rows).

Frozen transform conventions (mirrored, independently, by
``charts/transforms.py`` — the fixtures are the contract):

- bp_to_ce:        year_ce = present_ce - age_bp (1950 for the committed
                   fixtures); rows re-sorted time-ascending.
- resample:        consecutive non-overlapping windows of window_years,
                   anchored at the first year; output year = mean of the
                   member years, value = mean of the member values;
                   trailing incomplete windows dropped.
- rolling_mean:    centred window window_years wide in years (rows within
                   [year - (w-1)/2, year + (w-1)/2]); only complete
                   windows (exactly int(w) members) emitted — no shrinking
                   edge windows.
- anomaly:         value minus the full-record mean (parameterless, #132).
- unit conversion: multiply by the code-owned table factor (restated here
                   independently; a test cross-checks the tables agree).
- splice:          paleo rows strictly before the splice year,
                   instrumental rows from it onward
                   (prefer_instrumental policy, #47).
- rebaseline:      value minus the manifest ``apply_to`` member's own mean
                   over the inclusive alignment period, applied before
                   splicing.
- eval-fixture assembly order: assemble (BP->CE, rebaseline, splice) ->
  per-series ``transforms`` list in order -> inclusive time filter on the
  context-panel range (panel charts) or ``time_range_ce``.

Tolerances (DESIGN §6.2): 1e-9 relative pass-through, 1e-6 post-transform.

Usage:
    python evals/scripts/compute_chart_fixtures.py                    # eval fixtures JSON
    python evals/scripts/compute_chart_fixtures.py --check            # verify committed JSON
    python evals/scripts/compute_chart_fixtures.py --write-data       # regen synthetic CSVs
    python evals/scripts/compute_chart_fixtures.py --transform-golds  # regen per-transform golds
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

CHARTS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "charts"
DEFAULT_OUT_DIR = CHARTS_FIXTURES / "gold"

SYNTHETIC_MARKER = "# SYNTHETIC FIXTURE - generated deterministically for this project's tests"

MARKER = (
    "# SYNTHETIC FIXTURE — authored for this project's tests "
    "(gold expected values computed by evals/scripts/compute_chart_fixtures.py; "
    "invented inputs, not real data)"
)

#: Independent copy of the code-owned conversion table (charts.spec
#: UNIT_CONVERSIONS). A test cross-checks the two tables agree; this
#: script must not import charts — duplication IS the independence.
UNIT_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    ("Mt CO2/yr", "Gt CO2/yr"): 1e-3,
    ("degC_anomaly", "degF_anomaly"): 1.8,
}

#: Restated independently of charts.spec.UNIT_CONVERSIONS (see above).
DEGC_TO_DEGF_ANOMALY_FACTOR = 1.8

#: The manifest-fixed rebaseline alignment period the gold rebaseline
#: fixture uses (mirrors the synthetic splice-pair manifest the renderer
#: tests build — a curation-time decision, never LLM-chosen; ADR-020).
REBASELINE_ALIGNMENT_PERIOD_CE = (1990, 1999)

#: The manifest-fixed splice year for the synthetic CO2 splice pair.
SPLICE_YEAR_CE = 1850

BP_REFERENCE_YEAR_CE = 1950.0

PASS_THROUGH_TOLERANCE = 1e-9
POST_TRANSFORM_TOLERANCE = 1e-6


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ===========================================================================
# Family 1 — per-transform gold CSVs (issue #17)
# ===========================================================================


def _read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    """Read a synthetic fixture CSV: skip '#' comment lines, parse floats."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = list(csv.reader(lines))
    header, data = rows[0], rows[1:]
    return header, [[float(cell) for cell in row] for row in data]


def _fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _write_csv(path: Path, header: list[str], rows: list[list[float | str]]) -> None:
    lines = [MARKER, ",".join(header)]
    lines.extend(",".join(_fmt(cell) for cell in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gold_bp_to_ce(paleo: list[list[float]]) -> list[list[float]]:
    out = [[BP_REFERENCE_YEAR_CE - age, value] for age, value in paleo]
    return sorted(out, key=lambda row: row[0])


def gold_resample(annual: list[list[float]], window_years: int) -> list[list[float]]:
    rows = sorted(annual, key=lambda row: row[0])
    out: list[list[float]] = []
    for start in range(0, len(rows) - window_years + 1, window_years):
        window = rows[start : start + window_years]
        out.append([_mean([r[0] for r in window]), _mean([r[1] for r in window])])
    return out


def gold_rolling_mean(annual: list[list[float]], window_years: int) -> list[list[float]]:
    rows = sorted(annual, key=lambda row: row[0])
    half = (window_years - 1) / 2.0
    out: list[list[float]] = []
    for year, _ in rows:
        window = [v for y, v in rows if year - half <= y <= year + half]
        if len(window) == window_years:
            out.append([year, _mean(window)])
    return out


def gold_anomaly_vs_baseline(annual: list[list[float]]) -> list[list[float]]:
    baseline = _mean([value for _, value in annual])
    return [[year, value - baseline] for year, value in annual]


def gold_unit_conversion(annual: list[list[float]]) -> list[list[float]]:
    return [[year, value * DEGC_TO_DEGF_ANOMALY_FACTOR] for year, value in annual]


def gold_rebaseline(annual: list[list[float]]) -> list[list[float]]:
    start, end = REBASELINE_ALIGNMENT_PERIOD_CE
    window = [value for year, value in annual if start <= year <= end]
    shift = _mean(window)
    return [[year, value - shift] for year, value in annual]


def gold_splice(
    paleo_ce: list[list[float]], instrumental: list[list[float]]
) -> list[list[float | str]]:
    # 'segment' leads so every later column stays numeric (the chart-CSV
    # meta-test in tests/unit/test_fixture_corpus.py floats columns 2+).
    out: list[list[float | str]] = []
    out.extend(["paleo", y, v] for y, v in paleo_ce if y < SPLICE_YEAR_CE)
    out.extend(["instrumental", y, v] for y, v in instrumental if y >= SPLICE_YEAR_CE)
    return sorted(out, key=lambda row: float(row[1]))


def write_gold_fixtures(out_dir: Path) -> list[Path]:
    """Write the per-transform gold CSVs (the #17 renderer contract)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _, annual = _read_csv(CHARTS_FIXTURES / "synthetic_annual_anomaly.csv")
    _, paleo = _read_csv(CHARTS_FIXTURES / "synthetic_paleo_bp.csv")
    _, instrumental = _read_csv(CHARTS_FIXTURES / "synthetic_instrumental_co2.csv")

    paleo_ce = gold_bp_to_ce(paleo)
    outputs: dict[str, tuple[list[str], list[list[float | str]]]] = {
        "gold_bp_to_ce.csv": (["year_ce", "co2_ppm"], paleo_ce),
        "gold_resample.csv": (["year_ce", "anomaly_c"], gold_resample(annual, 4)),
        "gold_rolling_mean.csv": (["year_ce", "anomaly_c"], gold_rolling_mean(annual, 5)),
        "gold_anomaly_vs_baseline.csv": (
            ["year_ce", "anomaly_c"],
            gold_anomaly_vs_baseline(annual),
        ),
        "gold_unit_conversion.csv": (["year_ce", "anomaly_f"], gold_unit_conversion(annual)),
        "gold_rebaseline.csv": (["year_ce", "anomaly_c"], gold_rebaseline(annual)),
        "gold_splice.csv": (["segment", "year_ce", "co2_ppm"], gold_splice(paleo_ce, instrumental)),
    }
    written = []
    for name, (header, rows) in outputs.items():
        path = out_dir / name
        _write_csv(path, header, rows)
        written.append(path)
    return written


# ===========================================================================
# Family 2 — eval-level rendered-value fixtures (issue #20)
# ===========================================================================

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
    shift = _mean(window)
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
    """Frozen rolling-mean convention (charts/transforms.py mirror):
    centred window [year - (w-1)/2, year + (w-1)/2], only complete windows
    (exactly int(w) members) emitted — no shrinking edge windows."""
    width = int(window_years)
    half = (window_years - 1) / 2.0
    out = []
    for year, _ in points:
        window = [v for y, v in points if year - half <= y <= year + half]
        if len(window) == width:
            out.append((year, _mean(window)))
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
                "splicing; rolling_mean is a centred window [year - (w-1)/2, "
                "year + (w-1)/2] emitting only complete windows (exactly int(w) "
                "members, matching charts/transforms.py); unit_conversion "
                "multiplies by the code-owned factor. Tolerances: 1e-9 relative "
                "pass-through, 1e-6 relative post-transform (DESIGN section 6.2)."
            ),
        },
        "fixtures": fixtures,
    }


def _serialise(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


# ===========================================================================
# Entry point
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true", help="regenerate the synthetic CSVs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed eval fixtures match a fresh recompute",
    )
    parser.add_argument(
        "--transform-golds",
        action="store_true",
        help="regenerate the per-transform gold CSVs (issue #17)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="output directory for --transform-golds",
    )
    args = parser.parse_args(argv)
    if args.write_data:
        write_data()
        return 0
    if args.transform_golds:
        for path in write_gold_fixtures(Path(args.out_dir)):
            print(path)
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
