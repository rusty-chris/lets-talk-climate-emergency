"""Production chart transforms: pure Pandas, gold-fixture tested (issue #17).

Productionises the #4 spike transforms (``charts/spike/transforms.py``)
plus the four generic vocabulary ops (:data:`charts.spec.TRANSFORM_OPS`).
Everything here is a pure function over DataFrames passed in — no network,
no filesystem, no manifest loading (IMPLEMENTATION.md §1). The renderer
(:mod:`charts.render`) is the only production caller; parameters that are
curation-time decisions (splice year, overlap policy, alignment period)
are handed in *from the manifest* by the caller, never invented here.

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

from charts.spec import UNIT_CONVERSIONS

#: Paleoclimate convention: "before present" is counted back from 1950 CE.
BP_REFERENCE_YEAR_CE = 1950.0

#: The only overlap policy this MVP renderer enacts (review finding #47).
#: Both committed splice pairs use it; any other manifest value refuses by
#: name rather than silently falling back — the policies must never be
#: conflated.
IMPLEMENTED_OVERLAP_POLICIES = frozenset({"prefer_instrumental"})


def bp_to_ce(df: pd.DataFrame, age_col: str = "age_bp") -> pd.DataFrame:
    """``time_axis: {calendar: CE, convert_bp: true}`` — years BP → years CE.

    ``year_ce = 1950 - age_bp``; the age column is replaced by ``year_ce``
    and rows are re-sorted time-ascending. Pure: the input frame is copied.
    """
    out = df.copy()
    out["year_ce"] = BP_REFERENCE_YEAR_CE - out[age_col]
    out = out.drop(columns=[age_col])
    ordered = ["year_ce"] + [c for c in out.columns if c != "year_ce"]
    return out[ordered].sort_values("year_ce").reset_index(drop=True)


def resample(
    df: pd.DataFrame,
    value_col: str,
    window_years: float,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: resample, window_years: N}`` — mean over consecutive N-year windows.

    Non-overlapping windows of ``window_years`` consecutive rows, anchored
    at the series' first year; each output row carries the mean year and
    mean value of its members. A trailing window with fewer than
    ``window_years`` rows is dropped (a partial window rendered as a settled
    datum is the #108 honesty failure). Pure.
    """
    width = int(window_years)
    ordered = df.sort_values(year_col).reset_index(drop=True)
    years: list[float] = []
    values: list[float] = []
    for start in range(0, len(ordered) - width + 1, width):
        window = ordered.iloc[start : start + width]
        years.append(window[year_col].mean())
        values.append(window[value_col].mean())
    return pd.DataFrame({year_col: years, value_col: values})


def rolling_mean(
    df: pd.DataFrame,
    value_col: str,
    window_years: float,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: rolling_mean, window_years: N}`` — centred N-year rolling mean.

    A window ``window_years`` wide *in years* centred on each row (members
    with year in ``[year - (N-1)/2, year + (N-1)/2]``); only rows whose
    window holds exactly ``window_years`` members are emitted — no shrinking
    edge windows (an edge value from half a window is silent smoothing
    distortion). Pure.
    """
    width = int(window_years)
    half = (window_years - 1) / 2.0
    ordered = df.sort_values(year_col).reset_index(drop=True)
    all_years = ordered[year_col]
    years: list[float] = []
    values: list[float] = []
    for _, row in ordered.iterrows():
        centre = row[year_col]
        members = ordered[(all_years >= centre - half) & (all_years <= centre + half)]
        if len(members) == width:
            years.append(centre)
            values.append(members[value_col].mean())
    return pd.DataFrame({year_col: years, value_col: values})


def anomaly_vs_baseline(
    df: pd.DataFrame,
    value_col: str,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``{op: anomaly_vs_baseline}`` — subtract the full-record mean.

    Parameterless by decision (review finding #132): the baseline is never
    LLM-choosable and the whole-record mean is the only parameter-free
    honest choice. Pure.
    """
    out = df.copy()
    out[value_col] = out[value_col] - out[value_col].mean()
    return out


def unit_conversion(
    df: pd.DataFrame,
    value_col: str,
    from_unit: str,
    to_unit: str,
) -> pd.DataFrame:
    """``{op: unit_conversion, to_unit: U}`` — scale by the code-owned table factor.

    The ``(from_unit, to_unit)`` factor comes only from
    :data:`charts.spec.UNIT_CONVERSIONS`; an unlisted pair raises
    :class:`ValueError` naming both units — the arithmetic is code-owned,
    never spec-supplied (review finding #132). Pure.
    """
    try:
        factor = UNIT_CONVERSIONS[(from_unit, to_unit)]
    except KeyError:
        raise ValueError(
            f"no code-owned conversion for units {from_unit!r} -> {to_unit!r}: "
            "only pairs in charts.spec.UNIT_CONVERSIONS are legal (review finding #132)"
        ) from None
    out = df.copy()
    out[value_col] = out[value_col] * factor
    return out


def splice_series(
    paleo: pd.DataFrame,
    instrumental: pd.DataFrame,
    splice_year_ce: float,
    overlap_policy: str,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``splice_series`` — join paleo + instrumental at the manifest splice year.

    Paleo rows strictly before ``splice_year_ce``, instrumental rows from it
    onward, each labelled with a ``segment`` column; only columns common to
    both inputs survive (plus ``segment``). ``overlap_policy`` is the
    manifest pair's ``overlap.policy``; only ``prefer_instrumental`` is
    implemented in MVP and any other value raises :class:`ValueError` naming
    it (review finding #47) — the policies must never be silently conflated.
    Either side empty raises: a one-sided splice renders no visible join and
    the mandatory annotation would lie. Pure.
    """
    if overlap_policy not in IMPLEMENTED_OVERLAP_POLICIES:
        raise ValueError(
            f"overlap policy {overlap_policy!r} is not implemented by this renderer "
            f"(only {sorted(IMPLEMENTED_OVERLAP_POLICIES)} in MVP); refusing by name "
            "rather than silently conflating policies (review finding #47)"
        )
    common = [c for c in paleo.columns if c in set(instrumental.columns)]
    if year_col not in common:
        raise ValueError(f"splice_series: both inputs must share the {year_col!r} column")
    left = paleo[paleo[year_col] < splice_year_ce][common].copy()
    right = instrumental[instrumental[year_col] >= splice_year_ce][common].copy()
    if left.empty or right.empty:
        raise ValueError(
            f"splice_series at {splice_year_ce}: paleo contributes {len(left)} rows, "
            f"instrumental {len(right)} — both sides must be non-empty"
        )
    left["segment"] = "paleo"
    right["segment"] = "instrumental"
    out = pd.concat([left, right], ignore_index=True)
    ordered = ["segment"] + common
    return out[ordered].sort_values(year_col, kind="stable").reset_index(drop=True)


def rebaseline(
    df: pd.DataFrame,
    value_col: str,
    alignment_period_ce: tuple[float, float],
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """``rebaseline_to`` — zero the series' mean over the manifest-fixed period.

    Shifts the series so its mean over the inclusive
    ``alignment_period_ce`` is zero; an empty period raises (a silent empty
    mean would rebaseline by NaN and poison the series). The legal period is
    fixed per pair in ``datasets/manifest.yaml`` (ADR-020): callers pass the
    manifest value. Pure.
    """
    start, end = alignment_period_ce
    window = df[(df[year_col] >= start) & (df[year_col] <= end)][value_col]
    if window.empty:
        raise ValueError(
            f"rebaseline alignment period {start}-{end} contains no data rows — "
            "a silent empty mean would rebaseline by NaN"
        )
    out = df.copy()
    out[value_col] = out[value_col] - window.mean()
    return out
