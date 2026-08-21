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
import json
from typing import Any

import pytest

from charts import spec as chartspec
from charts.spec import ChartSpecError
from tests.unit.test_chartspec import (
    LINE_EXTENTS,
    PANEL_EXTENTS,
    _assert_refused_at,
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


# ---------------------------------------------------------------------------
# #127 — manifest-anchored fields refuse on unspliced series
# ---------------------------------------------------------------------------


def test_rebaseline_on_plain_series_rejected() -> None:
    """#127: rebaselining exists only as a manifest splice-pair decision
    (ADR-020, amendments 1-2). An LLM-chosen alignment on a plain-dataset
    series is the exact cherry-pick the curation-time rule refuses."""
    spec = line_spec()
    spec["series"][0]["rebaseline_to"] = {
        "apply_to": "syn_abs_new",
        "alignment_period_ce": [1990, 1995],
        "alignment_disclosure": "LLM-invented",
    }
    _assert_refused_at(
        spec,
        "series[0].rebaseline_to",
        contains=("splice",),
        extents=LINE_EXTENTS,
    )


def test_rebaseline_on_plain_series_rejected_against_real_manifest() -> None:
    """#127 reproduced case: rebaseline_to naming a provisional dataset on
    a plain gistemp_v4 series must refuse against the committed manifest."""
    import yaml

    from tests.unit.test_chartspec import MANIFEST_PATH

    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    spec = line_spec()
    spec["chart_id"] = "syn-gistemp-rebaseline"
    spec["time_range_ce"] = [1998, 2012]
    spec["series"][0] = {
        "id": "t",
        "label": "Temperature anomaly",
        "unit": "degC_anomaly",
        "dataset": "gistemp_v4",
        "rebaseline_to": {
            "apply_to": "kaufman2020_temp12k",
            "alignment_period_ce": [1998, 2012],
            "alignment_disclosure": "aligned on a cherry-picked window",
        },
    }
    _assert_refused_at(spec, "series[0].rebaseline_to", manifest=manifest)


def test_overlap_policy_on_plain_series_rejected() -> None:
    """#127: overlap_policy discloses what happened to overlapping splice
    samples; on an unspliced series it is a false disclosure."""
    spec = line_spec()
    spec["series"][0]["overlap_policy"] = "prefer_instrumental"
    _assert_refused_at(
        spec,
        "series[0].overlap_policy",
        contains=("splice",),
        extents=LINE_EXTENTS,
    )


def test_splice_annotations_on_plain_series_rejected() -> None:
    """#127: a fake splice marker or resolution note on continuous data is
    the mirror image of the hidden-splice attack DESIGN §3.7 annotates
    against — both refuse on a series without a splice pair."""
    spec = line_spec()
    spec["series"][0]["annotations"] = {
        "splice_point": {"year_ce": 1950, "label": "spliced with instrumental record (1950)"}
    }
    _assert_refused_at(
        spec,
        "series[0].annotations.splice_point",
        contains=("splice",),
        extents=LINE_EXTENTS,
    )

    spec = line_spec()
    spec["series"][0]["annotations"] = {"resolution_note": "centennial resolution before 1850"}
    _assert_refused_at(
        spec,
        "series[0].annotations.resolution_note",
        contains=("splice",),
        extents=LINE_EXTENTS,
    )


def test_non_zero_baseline_annotation_requires_zero_excluding_domain() -> None:
    """#127: the non_zero_baseline annotation is the disclosure for a
    zero-excluding axis; with no scale_domain, or a zero-including one,
    it asserts a truncation that did not happen."""
    spec = line_spec()
    del spec["series"][0]["scale_domain"]
    spec["series"][0]["annotations"] = {"non_zero_baseline": {"label": "axis excludes zero"}}
    _assert_refused_at(
        spec,
        "series[0].annotations.non_zero_baseline",
        contains=("zero",),
        extents=LINE_EXTENTS,
    )

    spec = line_spec()
    spec["series"][0]["scale_domain"] = [0, 120]  # includes zero
    spec["series"][0]["annotations"] = {"non_zero_baseline": {"label": "axis excludes zero"}}
    _assert_refused_at(
        spec,
        "series[0].annotations.non_zero_baseline",
        contains=("zero",),
        extents=LINE_EXTENTS,
    )


def test_rebaseline_apply_to_must_be_a_series_member() -> None:
    """#127 defence in depth: even when spec and manifest agree, apply_to
    must resolve to a member of the series' splice pair — a manifest
    entry pointing outside the pair must not silently authorise shifting
    a series that is not plotted."""
    manifest = syn_manifest()
    manifest["splice_pairs"][1]["rebaseline"]["apply_to"] = "syn_provisional"
    spec = panel_spec()
    spec["series"][1]["rebaseline_to"]["apply_to"] = "syn_provisional"
    _assert_refused_at(
        spec,
        "series[1].rebaseline_to.apply_to",
        contains=("syn_provisional",),
        extents=PANEL_EXTENTS,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# #128 — NaN/Infinity refused at parse, in validation and in hashing
# ---------------------------------------------------------------------------

_NON_FINITE_JSON = ("NaN", "Infinity", "-Infinity")


def _nonfinite_specs() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    for token in _NON_FINITE_JSON:
        value = json.loads(token)  # Python's permissive parse

        s = line_spec()
        s["series"][0]["scale_domain"] = [value, value]
        cases.append((f"scale_domain_{token}", s))

        s = line_spec()
        s["time_range_ce"] = [1900, value]
        cases.append((f"time_range_{token}", s))

        s = panel_spec()
        s["series"][0]["annotations"]["splice_point"]["year_ce"] = value
        cases.append((f"splice_year_{token}", s))

        s = line_spec()
        s["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": value}]
        cases.append((f"window_years_{token}", s))

        s = line_spec()
        s["series"][0]["baseline"] = {"value": value, "label": "reference"}
        cases.append((f"baseline_value_{token}", s))
    return cases


@pytest.mark.parametrize(
    ("name", "bad_spec"), _nonfinite_specs(), ids=[c[0] for c in _nonfinite_specs()]
)
def test_non_finite_numbers_rejected(name: str, bad_spec: dict[str, Any]) -> None:
    """#128: NaN/Infinity pass jsonschema's number checks and silently
    disable every comparison-based domain-integrity rule — the validator
    must refuse them at the offending path, with extents supplied."""
    extents = dict(LINE_EXTENTS)
    extents.update(PANEL_EXTENTS)
    err = _refuse_any(bad_spec, extents)
    blob = " ".join(v.reason for v in err.violations).lower()
    assert "finite" in blob or "nan" in blob or "infinity" in blob, (
        f"{name}: refusal should name the non-finite value: {blob!r}"
    )


def _refuse_any(bad_spec: dict[str, Any], extents: dict[str, tuple[float, float]]):
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec(bad_spec, syn_manifest(), data_extents=extents)
    return excinfo.value


def test_parse_spec_json_refuses_non_finite() -> None:
    """#128: the parse seam refuses NaN/Infinity tokens outright
    (json.loads parse_constant), so a non-RFC-8259 spec never reaches
    the validator as a Python-dialect float."""
    for token in _NON_FINITE_JSON:
        text = json.dumps(line_spec()).replace("[1900, 2000]", f"[1900, {token}]")
        with pytest.raises(ChartSpecError):
            chartspec.parse_spec_json(text)

    # RFC-8259-clean text parses to the same mapping json.loads gives.
    clean = json.dumps(line_spec())
    assert chartspec.parse_spec_json(clean) == json.loads(clean)


def test_spec_hash_refuses_non_finite() -> None:
    """#128: spec_hash must emit RFC 8259 only — a non-finite spec can
    never mint a permalink (json.dumps allow_nan=False semantics)."""
    for value in (float("nan"), float("inf"), float("-inf")):
        spec = line_spec()
        spec["series"][0]["scale_domain"] = [0, value]
        with pytest.raises(ChartSpecError):
            chartspec.spec_hash(spec)


# ---------------------------------------------------------------------------
# #129 — scale_domain flattening (zoom-out) refused, not just clipping
# ---------------------------------------------------------------------------


def test_scale_domain_flattening_data_rejected() -> None:
    """#129: the inverse cherry-pick — an arbitrarily zoomed-out domain
    renders the signal as a flat line while staying technically complete
    and honestly annotated. With extents supplied, the domain span is
    bounded relative to the data span (plus a free zero-extension)."""
    spec = line_spec()
    spec["series"][0]["scale_domain"] = [-1_000_000, 1_000_000]
    _assert_refused_at(
        spec,
        "series[0].scale_domain",
        contains=("flatten",),
        extents=LINE_EXTENTS,
    )

    # The classic denialist axis game: an anomaly with extent (-0.7, 1.4)
    # plotted on [-30, 30].
    manifest = syn_manifest()
    spec = line_spec()
    spec["series"][0].update({"unit": "wd_anomaly", "dataset": "syn_anom_new"})
    spec["series"][0]["scale_domain"] = [-30, 30]
    err = _refuse_with(spec, manifest, {"w": (-0.7, 1.4)})
    assert any("scale_domain" in v.path for v in err.violations)


def _refuse_with(spec_dict, manifest, extents):
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec(spec_dict, manifest, data_extents=extents)
    return excinfo.value


def test_scale_domain_legal_headroom_stays_valid() -> None:
    """#129 companion acceptance: the committed legal cases keep passing —
    generous zero-inclusion on absolute series and modest headroom on
    anomaly series are not flattening."""
    manifest = syn_manifest()

    # Zero-inclusion with headroom over an absolute series: [0, 120] over
    # (10, 100) — the zero-extension is free, the rest well within bound.
    assert chartspec.validate_spec(line_spec(), manifest, data_extents=LINE_EXTENTS) is None

    # The flagship's own domains against its recorded extents (the
    # acceptance cases the bound was calibrated on): CO2 [180, 440] over
    # (259.6, 427.35) and temperature [-1.5, 2.0] over (-0.71, 1.43).
    from tests.unit.test_chartspec import (
        FLAGSHIP_EXTENTS,
        _flagship,
        _pack_confirmed,
        _real_manifest,
    )

    confirmed = _pack_confirmed(_real_manifest())
    assert chartspec.validate_spec(_flagship(), confirmed, data_extents=FLAGSHIP_EXTENTS) is None


# ---------------------------------------------------------------------------
# #130 — ordering and degeneracy rules on year pairs and panels
# ---------------------------------------------------------------------------


def test_inverted_ranges_rejected() -> None:
    """#130: every [start, end] pair must be strictly increasing — an
    inverted range means whatever the renderer's filtering does (likely a
    blank frame or a mirror-imaged time axis)."""
    spec = line_spec()
    spec["time_range_ce"] = [2000, 1900]
    _assert_refused_at(spec, "time_range_ce", contains=("increas",))

    spec = panel_spec()
    spec["panels"]["recent"]["time_range_ce"] = [1990, 1950]
    _assert_refused_at(spec, "panels.recent.time_range_ce", extents=PANEL_EXTENTS)

    spec = line_spec()
    spec["series"][0]["scale_domain"] = [120, 0]
    _assert_refused_at(spec, "series[0].scale_domain", contains=("increas",))


def test_degenerate_recent_panel_rejected() -> None:
    """#130: the recent inset must be a strict zoom of the context —
    zero-width, full-width and below-minimum-span insets all degrade the
    panel pair §3.7 relies on for the 10-kyr axis problem."""
    spec = panel_spec()
    spec["panels"]["recent"]["time_range_ce"] = [1955, 1955]
    _assert_refused_at(spec, "panels.recent.time_range_ce", extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["panels"]["recent"]["time_range_ce"] = [-9000, 2000]  # equals context
    _assert_refused_at(
        spec, "panels.recent.time_range_ce", contains=("strict",), extents=PANEL_EXTENTS
    )

    spec = panel_spec()
    spec["panels"]["recent"]["time_range_ce"] = [1995, 2000]  # 5y < 10y minimum
    _assert_refused_at(spec, "panels.recent.time_range_ce", extents=PANEL_EXTENTS)


def test_flagship_shaped_panels_stay_valid() -> None:
    """#130 guard: the committed panel shapes (recent [1900, 2000] within
    context [-9000, 2000]; flagship [1850, 2025] within [-8050, 2025])
    stay valid under the ordering/degeneracy rules."""
    assert chartspec.validate_spec(panel_spec(), syn_manifest(), data_extents=PANEL_EXTENTS) is None


# ---------------------------------------------------------------------------
# #131 — resolution_note is manifest-owned verbatim; splice label names year
# ---------------------------------------------------------------------------


def test_resolution_note_must_match_manifest() -> None:
    """#131: vocabulary amendment 1 makes the manifest pair own the
    resolution note; presence-only enforcement lets a planner satisfy the
    validator with a note that denies the smoothing — strictly worse than
    omitting it. Verbatim equality, like the #50 disclosure."""
    spec = panel_spec()
    spec["series"][0]["annotations"]["resolution_note"] = "fully continuous annual data throughout"
    _assert_refused_at(
        spec,
        "series[0].annotations.resolution_note",
        contains=("verbatim",),
        extents=PANEL_EXTENTS,
    )


def test_splice_point_label_names_the_splice_year() -> None:
    """#131 decision pinned: the splice marker label is minimally
    constrained to name the manifest splice year (full manifest ownership
    of the label is deferred until a splice_label field exists) — a label
    that hides the year defeats the marker's purpose."""
    spec = panel_spec()
    spec["series"][0]["annotations"]["splice_point"]["label"] = "continuous series"
    _assert_refused_at(
        spec,
        "series[0].annotations.splice_point.label",
        contains=("1950",),
        extents=PANEL_EXTENTS,
    )


def test_flagship_resolution_notes_equal_manifest_strings() -> None:
    """#131: the committed flagship's drift from the manifest resolution
    notes is the concrete defect — both series must carry the manifest
    pair strings verbatim."""
    from tests.unit.test_chartspec import _flagship, _real_manifest

    flagship = _flagship()
    pairs = {p["id"]: p for p in _real_manifest()["splice_pairs"]}
    co2, temp = flagship["series"]
    assert co2["annotations"]["resolution_note"] == pairs["co2_10k"]["resolution_note"]
    assert temp["annotations"]["resolution_note"] == pairs["temp_10k"]["resolution_note"]
