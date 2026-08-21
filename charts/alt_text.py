"""Deterministic alt text from the ChartSpec (issue #17; DESIGN §3.7).

Every chart ships alt text derived deterministically from the spec and the
plotted frames the renderer already has — title, series, time range and
direction of trend. Pure (IMPLEMENTATION.md §1: ``charts/alt_text.py``
pure from spec); no LLM call is ever involved, so the same validated spec
and frames always produce byte-identical text.

RED phase: contract stub raising :class:`NotImplementedError`. The failing
tests in ``tests/unit/test_chart_caption_alt_csv.py`` pin the contract;
the implementer makes them pass without weakening them (ORCHESTRATION.md)
and then deletes this paragraph.

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

import pandas as pd

from charts.spec import RenderValidatedSpec

#: The closed trend-direction vocabulary the alt text may use.
TREND_WORDS = ("rising", "falling", "flat")


def alt_text(validated: RenderValidatedSpec, frames: Mapping[str, pd.DataFrame]) -> str:
    """Alt text for one validated chart: title, series, range, trend."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")
