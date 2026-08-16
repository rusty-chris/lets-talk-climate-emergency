"""One-time parsers: raw source files -> tidy DataFrames (issue #4 spike).

Each parser takes the path of a raw file exactly as downloaded from the
provider (URLs + sha256 in datasets/manifest.yaml) and returns a tidy
DataFrame in the source's *native* time convention — years BP for the paleo
sources, calendar-year CE for the instrumental ones. Time-axis conversion,
splicing and rebaselining are transforms (charts/spike/transforms.py), not
parsing concerns.

Output column contracts (pinned by tests/unit/test_spike_parsers.py):

- parse_bereiter_co2      -> [age_bp, co2_ppm, co2_1s_ppm]         (float64)
- parse_kaufman_temp12k   -> [age_bp, temp_c, temp_c_5, temp_c_95] (float64)
- parse_gml_co2_annual    -> [year_ce (int64), co2_ppm, unc_ppm]
- parse_gistemp_annual    -> [year_ce (int64), temp_anomaly_c]
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


def _read_text(path: Path | str) -> str:
    """Read a raw source file tolerating BOM and CRLF (Bereiter has both)."""
    return Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def parse_bereiter_co2(path: Path | str) -> pd.DataFrame:
    """Bereiter 2015 Antarctic CO2 composite (NOAA NCEI Paleo study 17975).

    Format: '#'-prefixed template header, then a tab-separated block headed
    `age_gas_calBP\tco2_ppm\tco2_1s_ppm`. Ages are years BP (present = 1950
    CE); the youngest ages are negative (post-1950 firn/Law Dome data).
    Rows are returned sorted by age ascending (youngest first, as shipped).
    """
    lines = [ln for ln in _read_text(path).split("\n") if ln and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
    expected = ["age_gas_calBP", "co2_ppm", "co2_1s_ppm"]
    if list(df.columns) != expected:
        raise ValueError(f"Bereiter composite: expected columns {expected}, got {list(df.columns)}")
    df = df.rename(columns={"age_gas_calBP": "age_bp"}).astype("float64")
    return df.sort_values("age_bp").reset_index(drop=True)


def parse_kaufman_temp12k(path: Path | str) -> pd.DataFrame:
    """Kaufman 2020 Temp12k multi-method GMST percentiles (NOAA study 29712).

    Format: plain CSV `ages, global_5, global_median, global_95, <latitude
    bands...>`, ages in years BP as 100-year bin centres (0..12000; bin 100
    covers 1800-1900 CE, the reconstruction's native reference period, where
    the median is 0 by construction). Only the global columns are kept.
    Values are anomalies in degrees C relative to 1800-1900 CE.
    """
    lines = [ln for ln in _read_text(path).split("\n") if ln and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)
    needed = {"ages", "global_5", "global_median", "global_95"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Temp12k percentiles: missing columns {sorted(missing)}")
    df = df.rename(
        columns={
            "ages": "age_bp",
            "global_median": "temp_c",
            "global_5": "temp_c_5",
            "global_95": "temp_c_95",
        }
    )[["age_bp", "temp_c", "temp_c_5", "temp_c_95"]].astype("float64")
    return df.sort_values("age_bp").reset_index(drop=True)


def parse_gml_co2_annual(path: Path | str) -> pd.DataFrame:
    """NOAA GML Mauna Loa annual-mean CO2 (co2_annmean_mlo.csv).

    Format: '#'-prefixed provenance comments, then CSV `year,mean,unc`.
    Concentrations are ppm (micromol/mol) on the WMO X2019 scale.
    """
    lines = [ln for ln in _read_text(path).split("\n") if ln and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    expected = ["year", "mean", "unc"]
    if list(df.columns) != expected:
        raise ValueError(f"GML annual CO2: expected columns {expected}, got {list(df.columns)}")
    df = df.rename(columns={"year": "year_ce", "mean": "co2_ppm", "unc": "unc_ppm"})
    df["year_ce"] = df["year_ce"].astype("int64")
    df[["co2_ppm", "unc_ppm"]] = df[["co2_ppm", "unc_ppm"]].astype("float64")
    return df.sort_values("year_ce").reset_index(drop=True)


def parse_gistemp_annual(path: Path | str) -> pd.DataFrame:
    """NASA GISTEMP v4 global Land-Ocean annual means (GLB.Ts+dSST.csv).

    Format: one title line ("Land-Ocean: Global Means"), then CSV with Year,
    monthly columns, and seasonal aggregates; the J-D column is the
    January-December annual mean anomaly in degrees C relative to 1951-1980.
    Missing values are '***' (e.g. the in-progress current year) and those
    rows are dropped.
    """
    lines = _read_text(path).split("\n")
    df = pd.read_csv(io.StringIO("\n".join(lines[1:])), na_values=["***"])
    if "Year" not in df.columns or "J-D" not in df.columns:
        raise ValueError(f"GISTEMP: expected 'Year' and 'J-D' columns, got {list(df.columns)}")
    df = df[["Year", "J-D"]].rename(columns={"Year": "year_ce", "J-D": "temp_anomaly_c"})
    df = df.dropna(subset=["temp_anomaly_c"])
    df["year_ce"] = df["year_ce"].astype("int64")
    df["temp_anomaly_c"] = df["temp_anomaly_c"].astype("float64")
    return df.sort_values("year_ce").reset_index(drop=True)
