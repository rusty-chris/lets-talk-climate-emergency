"""Production chart transforms: pure Pandas, gold-fixture tested (issue #17).

Productionises the #4 spike transforms (``charts/spike/transforms.py``)
plus the four generic vocabulary ops (:data:`charts.spec.TRANSFORM_OPS`).
Everything here is a pure function over DataFrames passed in — no network,
no filesystem, no manifest loading (IMPLEMENTATION.md §1). The renderer
(:mod:`charts.render`) is the only production caller; parameters that are
curation-time decisions (splice year, overlap policy, alignment period)
are handed in *from the manifest* by the caller, never invented here.

RED phase: the public functions below are contract stubs raising
:class:`NotImplementedError`. The failing tests in
``tests/unit/test_chart_transforms.py`` pin the contracts against
hand-computed gold fixtures (``tests/fixtures/charts/gold/``, regenerated
by the import-graph-independent script
``evals/scripts/compute_chart_fixtures.py``); the implementer makes them
pass without weakening them (ORCHESTRATION.md) and then deletes this
paragraph.

Frozen semantics (pinned by the gold fixtures)
----------------------------------------------

- :func:`bp_to_ce` — ``year_ce = 1950 - age_bp`` (the paleo "before
  present" convention; fractional ages stay fractional, negative ages land
  after 1950 CE). The age column is replaced by ``year_ce`` and rows are
  re-sorted time-ascending.
- :func:`resample` — consecutive non-overlapping windows of
  ``window_years``, anchored at the series' first year. Each output row
  carries the *mean year* of the window's member rows and the mean value;
  a trailing incomplete window is dropped (a partial window would render
  a fake settled datum — the same honesty rule as review finding #108).
- :func:`rolling_mean` — centred window ``window_years`` wide *in years*
  (rows within ``[year - (w-1)/2, year + (w-1)/2]``); only rows whose
  window is complete are emitted (no shrinking edge windows — an edge
  value computed from half a window is silent smoothing distortion).
- :func:`anomaly_vs_baseline` — subtracts the mean of the *entire input
  series*. The op is parameterless by decision (review finding #132): the
  baseline is never LLM-choosable, and the full-record mean is the only
  parameter-free honest choice.
- :func:`unit_conversion` — multiplies by the code-owned factor in
  :data:`charts.spec.UNIT_CONVERSIONS`; an unlisted ``(from_unit,
  to_unit)`` pair raises :class:`ValueError` (review finding #132 — the
  arithmetic is code-owned, never spec-supplied).
- :func:`splice_series` — paleo rows strictly before ``splice_year_ce``,
  instrumental rows from it onward, each row labelled with a ``segment``
  column (``"paleo"`` / ``"instrumental"``); only columns common to both
  inputs survive (plus ``segment``); either side empty raises. The
  ``overlap_policy`` parameter is the manifest pair's ``overlap.policy``;
  only ``"prefer_instrumental"`` (both committed pairs) is implemented in
  MVP and any other value raises :class:`ValueError` naming it — the
  policies must never be silently conflated (review finding #47).
- :func:`rebaseline` — shifts a series so its mean over the inclusive
  ``alignment_period_ce`` is zero; an empty period raises (a silent empty
  mean would rebaseline by NaN). The legal period is fixed per pair in
  ``datasets/manifest.yaml`` (ADR-020): callers pass the manifest value.
"""

from __future__ import annotations

import pandas as pd

#: Paleoclimate convention: "before present" is counted back from 1950 CE.
BP_REFERENCE_YEAR_CE = 1950.0


def bp_to_ce(df: pd.DataFrame, age_col: str = "age_bp") -> pd.DataFrame:
    """``time_axis: {calendar: CE, convert_bp: true}`` — years BP → years CE."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def resample(
    df: pd.DataFrame,
    value_col: str,
    window_years: float,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: resample, window_years: N}`` — mean over consecutive N-year windows."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def rolling_mean(
    df: pd.DataFrame,
    value_col: str,
    window_years: float,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: rolling_mean, window_years: N}`` — centred N-year rolling mean."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def anomaly_vs_baseline(
    df: pd.DataFrame,
    value_col: str,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: anomaly_vs_baseline}`` — subtract the full-record mean (parameterless)."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def unit_conversion(
    df: pd.DataFrame,
    value_col: str,
    from_unit: str,
    to_unit: str,
) -> pd.DataFrame:
    """``{op: unit_conversion, to_unit: U}`` — scale by the code-owned table factor."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def splice_series(
    paleo: pd.DataFrame,
    instrumental: pd.DataFrame,
    splice_year_ce: float,
    overlap_policy: str,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``splice_series`` — join paleo + instrumental at the manifest splice year."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def rebaseline(
    df: pd.DataFrame,
    value_col: str,
    alignment_period_ce: tuple[float, float],
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``rebaseline_to`` — zero the series' mean over the manifest-fixed period."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")
