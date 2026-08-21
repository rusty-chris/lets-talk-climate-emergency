"""Independent generator for the chart-transform gold fixtures (IMPLEMENTATION.md §5).

Computes the expected-value gold CSVs in ``tests/fixtures/charts/gold/``
from the synthetic source CSVs in ``tests/fixtures/charts/`` using the
standard library only. Deliberately imports **nothing from charts/** (an
import-graph test enforces this): the whole point of the gold fixtures is
that they are produced by a second, independent implementation of the
transform arithmetic, so a bug in ``charts/transforms.py`` cannot
silently agree with its own fixtures.

Every input and output is a SYNTHETIC FIXTURE — invented values authored
for this repo's tests (ADR-023 / review finding #117: no expected values
derived from real datasets are committed anywhere; the flagship's real
data renders only from origin fetches, never from committed rows).

Frozen conventions (mirrored, independently, by charts/transforms.py):

- bp_to_ce:        year_ce = 1950 - age_bp; rows re-sorted time-ascending.
- resample:        consecutive non-overlapping windows of window_years,
                   anchored at the first year; output year = mean of the
                   member years, value = mean of the member values;
                   trailing incomplete windows dropped.
- rolling_mean:    centred window window_years wide in years (rows within
                   [year - (w-1)/2, year + (w-1)/2]); only complete
                   windows emitted.
- anomaly:         value minus the full-record mean (parameterless, #132).
- unit conversion: degC_anomaly -> degF_anomaly is *1.8 (the code-owned
                   table factor, restated here independently).
- splice:          paleo rows strictly before the splice year,
                   instrumental rows from it onward, 'segment' column
                   labels each side (prefer_instrumental policy, #47).
- rebaseline:      value minus the mean over the inclusive manifest
                   alignment period.

Run:  python evals/scripts/compute_chart_fixtures.py [--out-dir tests/fixtures/charts/gold]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARTS_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "charts"
DEFAULT_OUT_DIR = CHARTS_FIXTURES / "gold"

MARKER = (
    "# SYNTHETIC FIXTURE — authored for this project's tests "
    "(gold expected values computed by evals/scripts/compute_chart_fixtures.py; "
    "invented inputs, not real data)"
)

#: Restated independently of charts.spec.UNIT_CONVERSIONS (a test
#: cross-checks the two tables agree; this script must not import charts).
DEGC_TO_DEGF_ANOMALY_FACTOR = 1.8

#: The manifest-fixed rebaseline alignment period the gold rebaseline
#: fixture uses (mirrors the synthetic splice-pair manifest the renderer
#: tests build — a curation-time decision, never LLM-chosen; ADR-020).
REBASELINE_ALIGNMENT_PERIOD_CE = (1990, 1999)

#: The manifest-fixed splice year for the synthetic CO2 splice pair.
SPLICE_YEAR_CE = 1850

BP_REFERENCE_YEAR_CE = 1950.0


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Per-transform gold computations (stdlib arithmetic only)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_gold_fixtures(out_dir: Path) -> list[Path]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    for path in write_gold_fixtures(Path(args.out_dir)):
        print(path)


if __name__ == "__main__":
    main()
