"""Adversarial-review fixes for the ChartSpec validator (findings #126-#137).

Failing tests written first (ORCHESTRATION.md: Fable test author, red
phase) for the review-15 findings on PR #123:

- #126 mistyped/malformed specs refuse with ChartSpecError — never a raw
  TypeError/AttributeError from the semantic phase;
- #127 manifest-anchored fields (rebaseline_to, overlap_policy, splice
  annotations) refuse on unspliced series; non_zero_baseline requires a
  zero-excluding scale_domain;
- #128 NaN/Infinity refuse at parse, in validation and in hashing;
- #129 scale_domain flattening (zoom-out) refused, not just clipping;
- #130 ordering/degeneracy rules on every year pair; strict recent inset;
- #131 resolution_note is manifest-owned verbatim; splice label names the
  splice year;
- #132 transform parameters validated per op; unit conversions from a
  code-owned closed table;
- #133 render-mode validation is structural (RenderValidatedSpec);
- #134 baseline.value pinned to the reference-period zero line;
- #135 top-level time_range_ce refused on panel charts;
- #136 CHARTSPEC.md names every schema property and required field;
- #137 size bounds on strings/series/_meta and the serialised spec.

All inline specs and manifests are SYNTHETIC FIXTURES (invented "widgets"
datasets) shared with tests/unit/test_chartspec.py; the only real inputs
are the committed flagship spec and datasets/manifest.yaml.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from charts import spec as chartspec
from charts.spec import ChartSpecError
from tests.unit.test_chartspec import (
    line_spec,
    panel_spec,
    syn_manifest,
)

# ---------------------------------------------------------------------------
# #126 — mistyped specs refuse, never crash
# ---------------------------------------------------------------------------


def _mistyped_cases() -> list[tuple[str, Any]]:
    """The reproduced crash shapes from finding #126, plus top-level
    non-object specs. Each must refuse with a path-bearing ChartSpecError."""
    cases: list[tuple[str, Any]] = []

    s = line_spec()
    s["series"][0]["scale_domain"] = ["a", "b"]
    cases.append(("scale_domain_strings", s))

    s = line_spec()
    s["time_range_ce"] = ["1900", "2000"]
    cases.append(("time_range_string_members", s))

    s = line_spec()
    s["time_range_ce"] = [None, 2000]
    cases.append(("time_range_null_member", s))

    s = panel_spec()
    s["panels"]["recent"]["time_range_ce"] = ["1900", 2000]
    cases.append(("panel_range_string_member", s))

    s = panel_spec()
    s["series"][1]["rebaseline_to"] = "not-an-object"
    cases.append(("rebaseline_string", s))

    s = panel_spec()
    s["series"][0]["annotations"] = "has splice_point and resolution_note words"
    cases.append(("annotations_string", s))

    s = panel_spec()
    s["series"][1]["uncertainty_band"] = "band"
    cases.append(("uncertainty_band_string", s))

    s = panel_spec()
    s["time_axis"] = "CE"
    cases.append(("time_axis_string", s))

    s = line_spec()
    s["series"] = "many"
    cases.append(("series_string", s))

    s = line_spec()
    s["series"][0]["transforms"] = "rolling_mean"
    cases.append(("transforms_string", s))

    s = line_spec()
    s["series"][0]["baseline"] = 0
    cases.append(("baseline_number", s))

    cases.append(("top_level_array", ["not", "a", "spec"]))
    cases.append(("top_level_string", "chart please"))
    cases.append(("top_level_null", None))
    return cases


@pytest.mark.parametrize(
    ("name", "bad_spec"), _mistyped_cases(), ids=[c[0] for c in _mistyped_cases()]
)
def test_mistyped_specs_refuse_never_crash(name: str, bad_spec: Any) -> None:
    """#126: every mistyped spec raises ChartSpecError (a refusal carrying
    the structural violation's path), never TypeError/AttributeError."""
    try:
        chartspec.validate_spec(bad_spec, syn_manifest(), data_extents=None)
    except ChartSpecError as err:
        assert err.violations, f"{name}: refused with no violations recorded"
        for violation in err.violations:
            assert violation.path, f"{name}: violation with empty path"
            assert violation.reason.strip(), f"{name}: violation with empty reason"
        return
    raise AssertionError(f"{name}: mistyped spec was accepted")


_WRONG_TYPED = (None, {"unexpected": 1}, ["unexpected"])


def _leaf_paths(node: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every path in a spec (containers and scalars alike) below the root."""
    paths: list[tuple[Any, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            paths.append((*prefix, key))
            paths.extend(_leaf_paths(value, (*prefix, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            paths.append((*prefix, index))
            paths.extend(_leaf_paths(value, (*prefix, index)))
    return paths


def _set_path(spec: Any, path: tuple[Any, ...], value: Any) -> None:
    target = spec
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


@pytest.mark.parametrize("builder", [line_spec, panel_spec], ids=["line", "panel"])
def test_wrong_typed_leaf_sweep_refuses_never_crashes(builder) -> None:
    """#126 property-ish sweep: replacing any node of a valid spec with a
    wrong-typed value must refuse via ChartSpecError, never crash."""
    reference = builder()
    for path in _leaf_paths(reference):
        for wrong in _WRONG_TYPED:
            mutated = copy.deepcopy(reference)
            _set_path(mutated, path, wrong)
            with pytest.raises(ChartSpecError):
                chartspec.validate_spec(mutated, syn_manifest(), data_extents=None)
