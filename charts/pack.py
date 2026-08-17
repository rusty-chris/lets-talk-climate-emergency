"""Chart data pack: productionised dataset parsers + coverage arithmetic (issue #14).

Pure over paths and DataFrames passed in (IMPLEMENTATION.md §1: "Loaders +
paleo parsers pure over committed sample files") — no network, no
filesystem reach-around beyond the path a caller hands over. The fetch /
sha256-verify / land flow that *uses* these parsers is the imperative
shell in :mod:`charts.datasets`.

RED phase: every function below is a contract stub raising
:class:`NotImplementedError`. The failing tests in
``tests/unit/test_dataset_pack_parsers.py`` pin the contracts; the
implementer makes them pass without weakening them
(ORCHESTRATION.md). The issue #4 spike parsers in
``charts/spike/parsers.py`` are the starting material (their
characterisation tests remain a canary, not the production contract).

Shared parser conventions (pinned by tests):

- Input is the raw file exactly as downloaded from the provider (URLs +
  sha256 pinned in ``datasets/manifest.yaml``). Output is a tidy
  DataFrame in the source's *native* time convention — transforms
  (BP→CE, splicing, rebaselining) are ``charts/transforms.py``'s job.
- Tolerate a UTF-8 BOM and CRLF line endings (Bereiter ships both), and
  skip leading ``#``-prefixed comment lines even for providers whose
  real files carry none — that is what lets every committed synthetic
  fixture carry the first-line ``SYNTHETIC FIXTURE`` marker (ADR-023)
  while mimicking the real format.
- Fail loudly, never silently: unexpected/missing columns, a value that
  does not parse as its column's type, or a file with zero usable data
  rows all raise :class:`ValueError` describing what was wrong. A row is
  *dropped* only for a documented in-band missing-value convention
  (GISTEMP ``***``; OWID blank cells), never because it failed to parse.
- Rows come back sorted (time ascending; OWID by country then year),
  index reset.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

#: The six MVP pack datasets (DESIGN.md §3.7 as amended; issue #14).
#: These ids are the manifest keys in ``datasets/manifest.yaml`` and the
#: keys of :data:`PARSERS`. kaufman2020_temp12k and bereiter2015_co2 are
#: ``open-provisional`` (review #45, spike-04 findings): part of the MVP
#: six for *fetching*, never part of any committed or mirrored artefact.
MVP_DATASET_IDS = frozenset(
    {
        "gistemp_v4",
        "hadcrut5",
        "noaa_gml_co2_mlo",
        "bereiter2015_co2",
        "kaufman2020_temp12k",
        "owid_co2",
    }
)

#: The module reference every manifest ``parser`` field must resolve
#: against (:func:`resolve_parser`) — a manifest can only ever point at
#: committed pack parsers, never spike code or arbitrary modules.
_PACK_MODULE_REF = "charts/pack.py"


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------


def _read_lines(path: Path | str) -> list[str]:
    """Read a raw source file tolerating a UTF-8 BOM and CRLF line endings.

    Blank lines are dropped; leading ``#``-prefixed comment lines are
    dropped regardless of whether the real provider format has any (that
    is what lets a committed synthetic fixture carry the ADR-023
    ``SYNTHETIC FIXTURE`` marker on line 1 while mimicking the real
    format).
    """
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip() != ""]
    idx = 0
    while idx < len(lines) and lines[idx].lstrip().startswith("#"):
        idx += 1
    return lines[idx:]


def _read_csv_after_comments(
    path: Path | str, *, extra_skip: int = 0, sep: str = ",", **read_csv_kwargs: Any
) -> pd.DataFrame:
    """Strip leading comment lines (and ``extra_skip`` further lines, e.g.

    GISTEMP's bare title line), then parse the remainder as CSV/TSV.
    """
    lines = _read_lines(path)
    if extra_skip:
        lines = lines[extra_skip:]
    if not lines:
        raise ValueError(f"{path}: no data found after stripping comment/title lines")
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, **read_csv_kwargs)


def _require_nonempty(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label}: file contains no usable data rows")


def _coerce_float64(df: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    try:
        df[columns] = df[columns].astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: a value did not parse as a number: {exc}") from exc
    return df


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_gistemp_annual(path: Path | str) -> pd.DataFrame:
    """NASA GISTEMP v4 global Land-Ocean annual means (GLB.Ts+dSST.csv).

    Format: one title line, then CSV with ``Year``, monthly columns and
    seasonal aggregates; ``J-D`` is the January–December annual-mean
    anomaly (°C vs 1951–1980); ``***`` marks missing values (always the
    in-progress current year's J-D). Returns
    ``[year_ce (int64), temp_anomaly_c (float64)]``; ``***`` J-D rows are
    dropped (documented missing-value convention — the reason manifest
    ``coverage`` must come from parser output, review finding #52).
    """
    df = _read_csv_after_comments(path, extra_skip=1, na_values=["***"])
    if "Year" not in df.columns or "J-D" not in df.columns:
        raise ValueError(
            f"GISTEMP annual: expected 'Year' and 'J-D' columns, got {list(df.columns)}"
        )
    df = df[["Year", "J-D"]].rename(columns={"Year": "year_ce", "J-D": "temp_anomaly_c"})
    df = df.dropna(subset=["temp_anomaly_c"])
    _require_nonempty(df, "GISTEMP annual")
    try:
        df["year_ce"] = df["year_ce"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"GISTEMP annual: 'Year' did not parse as an integer: {exc}") from exc
    df = _coerce_float64(df, ["temp_anomaly_c"], "GISTEMP annual")
    return df.sort_values("year_ce").reset_index(drop=True)


def parse_hadcrut5_annual(path: Path | str) -> pd.DataFrame:
    """HadCRUT5 analysis summary series, global annual means (Met Office, OGL v3).

    Format: CSV headed ``Time,Anomaly (deg C),Lower confidence limit
    (2.5%),Upper confidence limit (97.5%)``; ``Time`` is the calendar
    year. Returns ``[year_ce (int64), temp_anomaly_c, temp_anomaly_c_lower,
    temp_anomaly_c_upper (float64)]`` — the dataset ships its uncertainty,
    so the pack carries it (DESIGN §3.7: bands render when shipped).
    """
    df = _read_csv_after_comments(path)
    expected = [
        "Time",
        "Anomaly (deg C)",
        "Lower confidence limit (2.5%)",
        "Upper confidence limit (97.5%)",
    ]
    if list(df.columns) != expected:
        raise ValueError(f"HadCRUT5 annual: expected columns {expected}, got {list(df.columns)}")
    _require_nonempty(df, "HadCRUT5 annual")
    df = df.rename(
        columns={
            "Time": "year_ce",
            "Anomaly (deg C)": "temp_anomaly_c",
            "Lower confidence limit (2.5%)": "temp_anomaly_c_lower",
            "Upper confidence limit (97.5%)": "temp_anomaly_c_upper",
        }
    )
    try:
        df["year_ce"] = df["year_ce"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"HadCRUT5 annual: 'Time' did not parse as an integer: {exc}") from exc
    df = _coerce_float64(
        df,
        ["temp_anomaly_c", "temp_anomaly_c_lower", "temp_anomaly_c_upper"],
        "HadCRUT5 annual",
    )
    return df.sort_values("year_ce").reset_index(drop=True)


def parse_gml_co2_annual(path: Path | str) -> pd.DataFrame:
    """NOAA GML Mauna Loa annual-mean CO2 (co2_annmean_mlo.csv).

    Format: ``#``-prefixed provenance comments, then CSV ``year,mean,unc``
    (ppm, WMO scale). Returns ``[year_ce (int64), co2_ppm, unc_ppm
    (float64)]``.
    """
    df = _read_csv_after_comments(path)
    expected = ["year", "mean", "unc"]
    if list(df.columns) != expected:
        raise ValueError(f"GML annual CO2: expected columns {expected}, got {list(df.columns)}")
    _require_nonempty(df, "GML annual CO2")
    df = df.rename(columns={"year": "year_ce", "mean": "co2_ppm", "unc": "unc_ppm"})
    try:
        df["year_ce"] = df["year_ce"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"GML annual CO2: 'year' did not parse as an integer: {exc}") from exc
    df = _coerce_float64(df, ["co2_ppm", "unc_ppm"], "GML annual CO2")
    return df.sort_values("year_ce").reset_index(drop=True)


def parse_bereiter_co2(path: Path | str) -> pd.DataFrame:
    """Bereiter 2015 Antarctic CO2 composite (NOAA NCEI Paleo study 17975).

    Format: UTF-8 BOM + CRLF, ``#``-prefixed template header, then a
    tab-separated block headed ``age_gas_calBP\\tco2_ppm\\tco2_1s_ppm``.
    Ages are years BP (present = 1950 CE), negative for post-1950 samples.
    Returns ``[age_bp, co2_ppm, co2_1s_ppm]`` (all float64), sorted by
    ``age_bp`` ascending.
    """
    df = _read_csv_after_comments(path, sep="\t")
    expected = ["age_gas_calBP", "co2_ppm", "co2_1s_ppm"]
    if list(df.columns) != expected:
        raise ValueError(f"Bereiter composite: expected columns {expected}, got {list(df.columns)}")
    _require_nonempty(df, "Bereiter composite")
    df = df.rename(columns={"age_gas_calBP": "age_bp"})
    df = _coerce_float64(df, list(df.columns), "Bereiter composite")
    return df.sort_values("age_bp").reset_index(drop=True)


def parse_kaufman_temp12k(path: Path | str) -> pd.DataFrame:
    """Kaufman 2020 Temp12k multi-method GMST percentiles (NCEI study 29712).

    Format: plain CSV ``ages, global_5, global_median, global_95,
    <latitude bands…>`` with a space after each comma in the header; ages
    are 100-year bin centres in years BP; anomalies are °C vs 1800–1900 CE
    (the 100 BP bin's median is 0 by construction). Only the global
    columns are kept. Returns ``[age_bp, temp_c, temp_c_5, temp_c_95]``
    (all float64), sorted by ``age_bp`` ascending.
    """
    df = _read_csv_after_comments(path, skipinitialspace=True)
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
    )[["age_bp", "temp_c", "temp_c_5", "temp_c_95"]]
    _require_nonempty(df, "Temp12k percentiles")
    df = _coerce_float64(df, list(df.columns), "Temp12k percentiles")
    return df.sort_values("age_bp").reset_index(drop=True)


def parse_owid_co2(path: Path | str) -> pd.DataFrame:
    """OWID CO2 & GHG dataset (github.com/owid/co2-data, CC BY 4.0).

    Format: wide CSV headed ``country,year,iso_code,…`` with many metric
    columns; blank cells are missing values; country aggregates (e.g.
    ``World``) appear alongside countries. Returns the pack's tidy subset
    ``[country (object), iso_code (object), year_ce (int64), co2_mt
    (float64)]`` from the ``co2`` column (annual fossil+industry
    emissions, Mt), rows with a blank ``co2`` dropped (documented
    missing-value convention), sorted by country then year.
    """
    df = _read_csv_after_comments(path)
    required = {"country", "year", "iso_code", "co2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OWID co2-data: missing columns {sorted(missing)}")
    df = df[["country", "iso_code", "year", "co2"]].rename(
        columns={"year": "year_ce", "co2": "co2_mt"}
    )
    df = df.dropna(subset=["co2_mt"])
    _require_nonempty(df, "OWID co2-data")
    try:
        df["year_ce"] = df["year_ce"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"OWID co2-data: 'year' did not parse as an integer: {exc}") from exc
    df = _coerce_float64(df, ["co2_mt"], "OWID co2-data")
    return df.sort_values(["country", "year_ce"]).reset_index(drop=True)


#: dataset id -> committed parser. Exactly the MVP six; the manifest's
#: per-dataset ``parser`` reference must resolve (via
#: :func:`resolve_parser`) to the same callable this registry names.
PARSERS: dict[str, Callable[[Path | str], pd.DataFrame]] = {
    "gistemp_v4": parse_gistemp_annual,
    "hadcrut5": parse_hadcrut5_annual,
    "noaa_gml_co2_mlo": parse_gml_co2_annual,
    "bereiter2015_co2": parse_bereiter_co2,
    "kaufman2020_temp12k": parse_kaufman_temp12k,
    "owid_co2": parse_owid_co2,
}

#: function name -> committed parser callable — the lookup table behind
#: :func:`resolve_parser`. Keying on ``__name__`` (rather than dataset id)
#: is what lets a manifest reference name the function directly
#: (``charts/pack.py::parse_gistemp_annual``).
_PARSER_FUNCS_BY_NAME: dict[str, Callable[[Path | str], pd.DataFrame]] = {
    fn.__name__: fn for fn in PARSERS.values()
}


def resolve_parser(ref: str) -> Callable[[Path | str], pd.DataFrame]:
    """Resolve a manifest ``parser`` reference to its committed callable.

    ``ref`` has the form ``charts/pack.py::<function>`` (path relative to
    the repo root). Returns the named function object from this module —
    identical (``is``) to the :data:`PARSERS` entry — so a manifest can
    only ever name committed pack parsers. Raises :class:`ValueError` on
    any other module path, an unknown function name, or a malformed
    reference: the manifest must not be able to point dataset parsing at
    arbitrary code.
    """
    if not isinstance(ref, str) or "::" not in ref:
        raise ValueError(f"malformed parser reference (expected 'module::function'): {ref!r}")
    module_ref, _, func_name = ref.partition("::")
    if module_ref != _PACK_MODULE_REF:
        raise ValueError(
            f"parser reference must point at {_PACK_MODULE_REF!r} (committed pack parsers "
            f"only), got {module_ref!r}"
        )
    fn = _PARSER_FUNCS_BY_NAME.get(func_name)
    if fn is None:
        raise ValueError(f"{func_name!r} is not a registered pack parser in {_PACK_MODULE_REF}")
    return fn


#: Closed vocabulary for the manifest's machine-readable in-progress-year
#: handling (review finding #108). ``drop``: rows whose ``year_ce``
#: reaches the entry's ``retrieved_at`` year are excluded by
#: :func:`load_dataset_frame` — a partial-year mean is not the series'
#: quantity (an annual mean) and must never render as a settled datum.
PARTIAL_CURRENT_YEAR_POLICIES = frozenset({"drop"})


def load_dataset_frame(entry: Mapping[str, Any], path: Path | str) -> pd.DataFrame:
    """The pack-facing load surface: parse ``path`` with the entry's

    committed parser, then apply the entry's ``partial_current_year``
    policy (review finding #108). Every consumer of pack data — the
    fetch flow's coverage cross-check, the #15 validator, the #16
    renderer — loads through here, never through :data:`PARSERS`
    directly, so a partial in-progress-year datum can never reach a
    chart as a settled annual mean (the "Met-Office-shows-cooling
    screenshot" failure mode).

    Policy semantics (vocabulary closed, :data:`PARTIAL_CURRENT_YEAR_POLICIES`):

    - absent — parser output returned unchanged (the policy is opt-in
      per dataset; GISTEMP needs none because its ``***`` convention
      masks the in-progress year in-band);
    - ``drop`` — exclude every row whose ``year_ce`` is >= the year of
      the entry's ``retrieved_at`` date (HadCRUT5: the provider
      publishes a partial-year mean with no in-band marker). Requires a
      ``year_ce`` column; raises :class:`ValueError` if the drop would
      empty the frame (fail loudly, never yield a silently-empty
      dataset).

    Raises :class:`ValueError` on an unknown policy value.
    """
    frame = resolve_parser(entry["parser"])(path)
    policy = entry.get("partial_current_year")
    if policy is None:
        return frame
    if policy not in PARTIAL_CURRENT_YEAR_POLICIES:
        raise ValueError(
            f"unknown partial_current_year policy {policy!r} "
            f"(known: {sorted(PARTIAL_CURRENT_YEAR_POLICIES)})"
        )
    if "year_ce" not in frame.columns:
        raise ValueError("partial_current_year 'drop' requires a year_ce time axis")
    retrieved_year = int(str(entry["retrieved_at"])[:4])
    frame = frame[frame["year_ce"] < retrieved_year].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            "partial_current_year 'drop' left no usable rows (every row is in the "
            f"in-progress retrieval year {retrieved_year})"
        )
    return frame


def dataset_coverage(df: pd.DataFrame, time_axis: Mapping[str, Any]) -> dict[str, int]:
    """The usable-extent coverage block for a parsed frame (review finding #52).

    ``coverage`` in ``datasets/manifest.yaml`` means *parsed usable
    extent* — the first/last rows the committed parser actually yields —
    never raw-file extent (GISTEMP's in-progress year made the difference
    concrete: raw file to 2026, usable rows to 2025). This function is
    the single definition of that semantics:

    - ``time_axis.unit == "year_ce"`` → ``{"first_year_ce": int(min),
      "last_year_ce": int(max)}`` over the frame's ``year_ce`` column;
    - ``time_axis.unit == "years_bp"`` → ``{"oldest_bp": round(max),
      "youngest_bp": round(min)}`` over ``age_bp`` (endpoints rounded to
      the nearest whole year, matching the hand-recorded convention).

    Raises :class:`ValueError` on an empty frame or an unknown unit.
    """
    if df.empty:
        raise ValueError("dataset_coverage: empty frame has no usable extent")
    unit = time_axis.get("unit")
    if unit == "year_ce":
        if "year_ce" not in df.columns:
            raise ValueError("dataset_coverage: frame has no 'year_ce' column for unit 'year_ce'")
        return {
            "first_year_ce": int(df["year_ce"].min()),
            "last_year_ce": int(df["year_ce"].max()),
        }
    if unit == "years_bp":
        if "age_bp" not in df.columns:
            raise ValueError("dataset_coverage: frame has no 'age_bp' column for unit 'years_bp'")
        return {
            "oldest_bp": int(round(float(df["age_bp"].max()))),
            "youngest_bp": int(round(float(df["age_bp"].min()))),
        }
    raise ValueError(f"dataset_coverage: unknown time_axis unit {unit!r}")
