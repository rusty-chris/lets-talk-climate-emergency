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
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


def parse_hadcrut5_annual(path: Path | str) -> pd.DataFrame:
    """HadCRUT5 analysis summary series, global annual means (Met Office, OGL v3).

    Format: CSV headed ``Time,Anomaly (deg C),Lower confidence limit
    (2.5%),Upper confidence limit (97.5%)``; ``Time`` is the calendar
    year. Returns ``[year_ce (int64), temp_anomaly_c, temp_anomaly_c_lower,
    temp_anomaly_c_upper (float64)]`` — the dataset ships its uncertainty,
    so the pack carries it (DESIGN §3.7: bands render when shipped).
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


def parse_gml_co2_annual(path: Path | str) -> pd.DataFrame:
    """NOAA GML Mauna Loa annual-mean CO2 (co2_annmean_mlo.csv).

    Format: ``#``-prefixed provenance comments, then CSV ``year,mean,unc``
    (ppm, WMO scale). Returns ``[year_ce (int64), co2_ppm, unc_ppm
    (float64)]``.
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


def parse_bereiter_co2(path: Path | str) -> pd.DataFrame:
    """Bereiter 2015 Antarctic CO2 composite (NOAA NCEI Paleo study 17975).

    Format: UTF-8 BOM + CRLF, ``#``-prefixed template header, then a
    tab-separated block headed ``age_gas_calBP\\tco2_ppm\\tco2_1s_ppm``.
    Ages are years BP (present = 1950 CE), negative for post-1950 samples.
    Returns ``[age_bp, co2_ppm, co2_1s_ppm]`` (all float64), sorted by
    ``age_bp`` ascending.
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


def parse_kaufman_temp12k(path: Path | str) -> pd.DataFrame:
    """Kaufman 2020 Temp12k multi-method GMST percentiles (NCEI study 29712).

    Format: plain CSV ``ages, global_5, global_median, global_95,
    <latitude bands…>`` with a space after each comma in the header; ages
    are 100-year bin centres in years BP; anomalies are °C vs 1800–1900 CE
    (the 100 BP bin's median is 0 by construction). Only the global
    columns are kept. Returns ``[age_bp, temp_c, temp_c_5, temp_c_95]``
    (all float64), sorted by ``age_bp`` ascending.
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


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
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


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
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")


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
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_parsers.py")
