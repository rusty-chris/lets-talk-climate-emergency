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


# ---------------------------------------------------------------------------
# #132 — transform parameters validated per op; closed conversion table
# ---------------------------------------------------------------------------


def test_unit_conversion_must_be_from_known_table() -> None:
    """#132: to_unit outside the code-owned conversion table is an
    arbitrary axis-label escape hatch — the axis unit, otherwise pinned
    to the manifest's variable.unit, must not become free LLM text."""
    spec = line_spec()
    spec["series"][0]["unit"] = "wd (adjusted for urban heat island bias)"
    spec["series"][0]["transforms"] = [
        {"op": "unit_conversion", "to_unit": "wd (adjusted for urban heat island bias)"}
    ]
    _assert_refused_at(
        spec,
        "series[0].transforms[0].to_unit",
        contains=("conversion",),
        extents=LINE_EXTENTS,
    )


def test_unit_conversion_from_table_accepted() -> None:
    """#132 guard: a conversion recorded in the code-owned table is legal
    (owid_co2's Mt->Gt against the real manifest, pack-confirmed)."""
    from tests.unit.test_chartspec import _pack_confirmed, _real_manifest

    assert ("Mt CO2/yr", "Gt CO2/yr") in chartspec.UNIT_CONVERSIONS
    manifest = _pack_confirmed(_real_manifest())
    spec = {
        "spec_version": "1.0.0",
        "chart_id": "syn-owid-gt",
        "chart_type": "line",
        "title": "Emissions in gigatonnes",
        "time_range_ce": [1750, 2024],
        "series": [
            {
                "id": "e",
                "label": "Emissions (Gt CO2/yr)",
                "unit": "Gt CO2/yr",
                "dataset": "owid_co2",
                "transforms": [{"op": "unit_conversion", "to_unit": "Gt CO2/yr"}],
            }
        ],
    }
    assert chartspec.validate_spec(spec, manifest) is None


def test_transform_params_per_op() -> None:
    """#132: each op's parameters are closed and required — missing
    required params, cross-op smuggling and out-of-bounds windows all
    refuse at the parameter's path."""
    # unit_conversion without to_unit: contradictory instruction.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "unit_conversion"}]
    _assert_refused_at(spec, "series[0].transforms[0].to_unit", extents=LINE_EXTENTS)

    # rolling_mean without window_years.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "rolling_mean"}]
    _assert_refused_at(spec, "series[0].transforms[0].window_years", extents=LINE_EXTENTS)

    # Negative and over-long windows (a 5000-year rolling mean is a
    # legal-looking smoothing attack).
    for bad_window in (-10, 0, 5000):
        spec = line_spec()
        spec["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": bad_window}]
        _assert_refused_at(spec, "series[0].transforms[0].window_years", extents=LINE_EXTENTS)

    # Cross-op smuggling: to_unit on rolling_mean/resample, window_years
    # on unit_conversion.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": 10, "to_unit": "wd"}]
    _assert_refused_at(spec, "series[0].transforms[0].to_unit", extents=LINE_EXTENTS)

    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "resample", "window_years": 10, "to_unit": "wd"}]
    _assert_refused_at(spec, "series[0].transforms[0].to_unit", extents=LINE_EXTENTS)

    spec = line_spec()
    spec["series"][0]["unit"] = "wd"
    spec["series"][0]["transforms"] = [{"op": "unit_conversion", "window_years": 10}]
    _assert_refused_at(spec, "series[0].transforms[0].window_years", extents=LINE_EXTENTS)

    # anomaly_vs_baseline takes no parameters at all.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "anomaly_vs_baseline", "window_years": 10}]
    _assert_refused_at(spec, "series[0].transforms[0].window_years", extents=LINE_EXTENTS)


def test_resample_parameter_contract_decided() -> None:
    """#132 decision pinned: resample takes a required positive
    window_years (the target temporal resolution in years) — it is no
    longer renderer-implicit."""
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "resample"}]
    _assert_refused_at(spec, "series[0].transforms[0].window_years", extents=LINE_EXTENTS)

    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "resample", "window_years": 10}]
    assert chartspec.validate_spec(spec, syn_manifest(), data_extents=LINE_EXTENTS) is None


def test_rolling_mean_guard_stays_valid() -> None:
    """#132 guard: the committed legal transform stays valid."""
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": 10}]
    assert chartspec.validate_spec(spec, syn_manifest(), data_extents=LINE_EXTENTS) is None


# ---------------------------------------------------------------------------
# #133 — render-mode validation is structural, not docstring prose
# ---------------------------------------------------------------------------


def test_render_validation_requires_extents() -> None:
    """#133: the render path's duty to re-validate with extents becomes a
    dedicated entry point that refuses None and partial extents — the #48
    checks can no longer silently vanish if #17 forgets."""
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec_for_render(line_spec(), syn_manifest(), None)
    assert any("extent" in v.reason for v in excinfo.value.violations)

    # Partial extents: a series id missing from the mapping refuses at
    # that series' scale_domain (today validate_spec silently skips it).
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec_for_render(line_spec(), syn_manifest(), {"other": (0.0, 1.0)})
    assert any(v.path == "series[0].scale_domain" for v in excinfo.value.violations)


def test_render_validation_returns_token_type() -> None:
    """#133: complete extents yield a RenderValidatedSpec — the type #17's
    artefact entry point must require, constructible only through
    validate_spec_for_render."""
    validated = chartspec.validate_spec_for_render(line_spec(), syn_manifest(), LINE_EXTENTS)
    assert isinstance(validated, chartspec.RenderValidatedSpec)
    assert validated.spec == line_spec()
    assert dict(validated.data_extents) == dict(LINE_EXTENTS)


def test_render_validated_spec_not_directly_constructible() -> None:
    """#133: the token type cannot be minted around the validator."""
    with pytest.raises(TypeError):
        chartspec.RenderValidatedSpec(line_spec(), LINE_EXTENTS)


def test_render_validation_still_runs_full_validation() -> None:
    """#133: the render entry point is a superset of validate_spec — a
    spec that only fails the extent check refuses through it."""
    spec = line_spec()
    spec["series"][0]["scale_domain"] = [0, 90]  # clips the extent max 100
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec_for_render(spec, syn_manifest(), LINE_EXTENTS)
    assert any(v.path == "series[0].scale_domain" for v in excinfo.value.violations)


# ---------------------------------------------------------------------------
# #134 — baseline.value pinned to the reference-period zero line
# ---------------------------------------------------------------------------


def test_baseline_value_constrained() -> None:
    """#134: baseline draws a labelled horizontal rule on the artefact — a
    model-supplied number. Decision pinned: the only legal value is 0
    (the reference-period zero line, the sole committed use); any other
    reference line waits for a manifest-anchored vocabulary entry."""
    spec = line_spec()
    spec["series"][0]["baseline"] = {"value": 55.5, "label": "safe limit"}
    _assert_refused_at(
        spec,
        "series[0].baseline.value",
        contains=("0",),
        extents=LINE_EXTENTS,
    )

    spec = panel_spec()
    spec["series"][1]["baseline"] = {"value": 1.5, "label": "0 = 1900-1950 average"}
    _assert_refused_at(spec, "series[1].baseline.value", extents=PANEL_EXTENTS)


def test_baseline_label_names_manifest_reference_on_rebaselined_series() -> None:
    """#134: on a rebaselined series the label's reference period must
    match the manifest display_reference (mirror of the #50 disclosure
    treatment) — a numerically false zero-line claim like
    '0 = 1951-1980 average' refuses."""
    spec = panel_spec()
    spec["series"][1]["baseline"] = {"value": 0, "label": "0 = 1951-1980 average"}
    _assert_refused_at(
        spec,
        "series[1].baseline.label",
        contains=("1900", "1950"),
        extents=PANEL_EXTENTS,
    )


def test_baseline_zero_with_reference_label_stays_valid() -> None:
    """#134 guard: the committed uses stay legal — the synthetic panel's
    baseline (0, '0 = 1900-1950 average') and the flagship's
    (0, '0 °C = 1800–1900 average')."""
    assert chartspec.validate_spec(panel_spec(), syn_manifest(), data_extents=PANEL_EXTENTS) is None

    from tests.unit.test_chartspec import (
        FLAGSHIP_EXTENTS,
        _flagship,
        _pack_confirmed,
        _real_manifest,
    )

    confirmed = _pack_confirmed(_real_manifest())
    assert chartspec.validate_spec(_flagship(), confirmed, data_extents=FLAGSHIP_EXTENTS) is None


# ---------------------------------------------------------------------------
# #135 — top-level time_range_ce refused on panel charts
# ---------------------------------------------------------------------------


def test_top_level_time_range_refused_on_panel_charts() -> None:
    """#135: on a context_recent_inset spec the panels own the ranges
    (amendment 4) — a top-level time_range_ce is dead weight with a hash
    consequence and an undefined renderer meaning, so it refuses."""
    spec = panel_spec()
    spec["time_range_ce"] = [999999, -999999]
    _assert_refused_at(
        spec,
        "time_range_ce",
        contains=("panels",),
        extents=PANEL_EXTENTS,
    )

    # Even a sensible-looking range refuses: the field has no meaning on
    # a panel chart, whatever its value.
    spec = panel_spec()
    spec["time_range_ce"] = [1900, 2000]
    _assert_refused_at(spec, "time_range_ce", extents=PANEL_EXTENTS)


# ---------------------------------------------------------------------------
# #137 — size bounds; caption strip carries no spec-derived free text
# ---------------------------------------------------------------------------


def test_spec_size_bounds() -> None:
    """#137: the validator's whole purpose is bounding inputs — strings,
    series counts, _meta and the serialised spec all get ceilings before
    the permalink store inherits unbounded content forever."""
    spec = line_spec()
    spec["title"] = "x" * 100_000
    _assert_refused_at(spec, "title", extents=LINE_EXTENTS)

    spec = line_spec()
    template = spec["series"][0]
    spec["series"] = [dict(template, id=f"w{i}") for i in range(50)]
    _assert_refused_at(spec, "series", extents=None)

    spec = line_spec()
    spec["_meta"] = {"blob": "y" * 100_000}
    _assert_refused_at(spec, "_meta", extents=LINE_EXTENTS)


def test_chart_id_and_spec_version_are_shape_restricted() -> None:
    """#137: chart_id is slug-shaped and spec_version is semver-shaped —
    they are identifiers, not prose ('official-noaa-verified-chart'-style
    impersonation stays possible but at least unbounded text and spoofed
    formatting do not reach logs, filenames or the caption)."""
    spec = line_spec()
    spec["chart_id"] = "Official NOAA Chart! (verified)"
    _assert_refused_at(spec, "chart_id", extents=LINE_EXTENTS)

    spec = line_spec()
    spec["spec_version"] = "v1 (final, do not question)"
    _assert_refused_at(spec, "spec_version", extents=LINE_EXTENTS)


def test_caption_strip_contains_no_spec_strings() -> None:
    """#137: amendment 9 removed captions from the spec so no LLM-authored
    text reaches the attribution strip — but the spike caption bakes in
    chart_id and spec_version. The strip must be composed exclusively of
    manifest attribution strings, deployment config and the spec hash."""
    from charts.spike.render import SITE_URL, caption_lines_from_manifest, load_manifest, load_spec

    spec = load_spec()
    manifest = load_manifest()
    lines = caption_lines_from_manifest(spec, manifest)
    joined = "\n".join(lines)

    assert SITE_URL in joined
    assert str(manifest["access_date"]) in joined
    # The chart is identified by its spec hash (the permalink identity),
    # never by the LLM-authored chart_id / spec_version strings.
    assert chartspec.spec_hash(spec)[:12] in joined
    assert spec["chart_id"] not in joined
    assert f"v{spec['spec_version']}" not in joined
