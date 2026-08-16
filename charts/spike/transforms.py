"""Spike implementations of the three flagship transforms (DESIGN section 3.7).

These are the transforms the adversarial review found missing from the
original ChartSpec vocabulary — `time_axis` BP->CE conversion,
`splice_series`, and `rebaseline_to` — implemented minimally to prove the
flagship chart end-to-end. Production versions land in charts/transforms.py
(issue #17) from red tests; the arithmetic pinned here (and hand-checked in
reviews/spike-04-chart-findings.md) seeds those tests' gold fixtures.

All functions are pure over DataFrames (IMPLEMENTATION.md section 1).
"""

from __future__ import annotations

import pandas as pd

#: Paleoclimate convention: "before present" is counted back from 1950 CE.
BP_REFERENCE_YEAR_CE = 1950.0


def bp_to_ce(df: pd.DataFrame, age_col: str = "age_bp") -> pd.DataFrame:
    """`time_axis: {calendar: CE, convert_bp: true}` — years BP -> years CE.

    year_ce = 1950 - age_bp (fractional ages stay fractional; negative BP
    ages, i.e. post-1950 samples, land after 1950 CE). The age column is
    replaced by `year_ce` and rows are re-sorted to time-ascending order.
    """
    out = df.copy()
    out["year_ce"] = BP_REFERENCE_YEAR_CE - out[age_col]
    out = out.drop(columns=[age_col])
    cols = ["year_ce"] + [c for c in out.columns if c != "year_ce"]
    return out[cols].sort_values("year_ce").reset_index(drop=True)


def rebaseline(
    df: pd.DataFrame,
    value_col: str,
    period: tuple[int, int],
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """`rebaseline_to` — shift a series so its mean over `period` is zero.

    `period` is an inclusive (start_year, end_year) range in CE. The legal
    period for each dataset pair is fixed in datasets/manifest.yaml — a
    curation-time scientific decision, never chosen by the LLM (ADR-020).
    Raises if the period has no data (a silent empty mean would rebaseline
    by NaN and poison the whole series).
    """
    start, end = period
    window = df[(df[year_col] >= start) & (df[year_col] <= end)][value_col]
    if window.empty:
        raise ValueError(f"rebaseline period {start}-{end} contains no data rows")
    out = df.copy()
    out[value_col] = out[value_col] - window.mean()
    return out


def splice(
    paleo: pd.DataFrame,
    instrumental: pd.DataFrame,
    splice_year_ce: float,
    year_col: str = "year_ce",
) -> pd.DataFrame:
    """`splice_series` — join a paleo and an instrumental series at a year.

    Paleo rows strictly before `splice_year_ce`, instrumental rows from it
    onward; each row is labelled with a `segment` column ("paleo" /
    "instrumental") so the renderer can annotate the splice point and the
    resolution change — unannotated splices are the "hidden smoothing"
    attack surface (DESIGN section 3.7). Only columns common to both inputs
    survive (plus `segment`).

    Raises if either side contributes no rows: a one-sided "splice" would
    render without a visible join and the mandatory annotation would lie.
    """
    common = [c for c in paleo.columns if c in set(instrumental.columns)]
    if year_col not in common:
        raise ValueError(f"splice: both inputs must share the {year_col!r} column")
    left = paleo[paleo[year_col] < splice_year_ce][common].copy()
    right = instrumental[instrumental[year_col] >= splice_year_ce][common].copy()
    if left.empty or right.empty:
        raise ValueError(
            f"splice at {splice_year_ce}: paleo contributes {len(left)} rows, "
            f"instrumental {len(right)} — both sides must be non-empty"
        )
    left["segment"] = "paleo"
    right["segment"] = "instrumental"
    out = pd.concat([left, right], ignore_index=True)
    return out.sort_values([year_col, "segment"]).reset_index(drop=True)
