"""Deterministic alt text from the ChartSpec (issue #17; DESIGN §3.7).

Every chart ships alt text derived deterministically from the spec and the
plotted frames the renderer already has — title, series, time range and
direction of trend. Pure (IMPLEMENTATION.md §1: ``charts/alt_text.py``
pure from spec); no LLM call is ever involved, so the same validated spec
and frames always produce byte-identical text.

Contract (pinned by tests):

- input is a :class:`charts.spec.RenderValidatedSpec` plus the pre-landed
  frames mapping — never a bare spec dict (review finding #133);
- the text contains the chart title, every series label, and the plotted
  time range's start and end years;
- each series' trend direction over the plotted range is stated with one
  of the closed words ``rising`` / ``falling`` / ``flat`` (compared over
  the first and last plotted values);
- deterministic: identical inputs → identical string, across calls and
  processes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from charts.spec import RenderValidatedSpec, _year_text

#: The closed trend-direction vocabulary the alt text may use.
TREND_WORDS = ("rising", "falling", "flat")

#: Below this absolute first→last change a series reads as ``flat`` rather
#: than ``rising``/``falling`` — a hair of numerical drift is not a trend.
_FLAT_EPSILON = 1e-9


def _trend_word(first: float, last: float) -> str:
    change = last - first
    if abs(change) <= _FLAT_EPSILON:
        return "flat"
    return "rising" if change > 0 else "falling"


def _series_endpoints(
    series: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    x0: float,
    x1: float,
    *,
    series_cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> tuple[float, float]:
    """The plotted series' first and last value within the range, ordered
    by year, through the ONE shared render pipeline
    (:func:`charts.render._series_frame` — spliced/BP series included).
    ``series_cache`` (finding #297) shares that pipeline with the rest of
    one ``render_chart``."""
    from charts.render import _series_frame

    frame, value_col = _series_frame(series, frames, manifest, cache=series_cache)
    ordered = frame[(frame["year_ce"] >= x0) & (frame["year_ce"] <= x1)].sort_values("year_ce")
    return float(ordered[value_col].iloc[0]), float(ordered[value_col].iloc[-1])


def alt_text(
    validated: RenderValidatedSpec,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    *,
    series_cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> str:
    """Alt text for one validated chart: title, series, range, trend.

    Deterministic and LLM-free: the same validated spec and frames always
    produce byte-identical text. Names the title, every series label, the
    plotted range's endpoint years and each series' trend direction (a
    closed :data:`TREND_WORDS` word) over that range. ``manifest`` is
    required — every plotted series routes through the one shared render
    pipeline. ``series_cache`` (finding #297) shares the per-series pipeline
    across one render.
    """
    from charts.render import _plot_range

    spec = validated.spec
    x0, x1 = _plot_range(spec)
    sentences = [
        f"{spec['title']} — a chart of climate data from {_year_text(x0)} to {_year_text(x1)}."
    ]
    for series in spec["series"]:
        first, last = _series_endpoints(series, frames, manifest, x0, x1, series_cache=series_cache)
        sentences.append(f"{series['label']} is {_trend_word(first, last)} over this range.")
    return " ".join(sentences)
