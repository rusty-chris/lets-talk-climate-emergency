"""ChartSpec schema + validator (issue #15) — RED.

Failing tests pinning the frozen chart vocabulary (DESIGN §3.7 as amended,
ADR-020, the ten binding vocabulary amendments on issue #15) and the
pure-code validator's refusal suite (review findings #47 overlap policy,
#48 scale-domain integrity, #50 on-artefact alignment disclosure, #52
coverage semantics, #117 structural pack gating). Everything here is unit
tier: pure over dicts, the committed flagship spec and the committed
dataset manifest — no data files, no network (ADR-023).

The contracts under test live in charts/spec.py (contract stubs). The
manifest is consumed as the raw datasets/manifest.yaml mapping; the
validator cross-checks curation-time decisions (splice membership + year,
overlap policy, alignment period, disclosure) and never re-decides them.

All inline specs and the synthetic manifest below are SYNTHETIC FIXTURES —
invented datasets ("widgets"), invented coverages — authored for these
tests; the only real inputs are the committed flagship ChartSpec and
datasets/manifest.yaml, neither of which embeds data values.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from charts import pack
from charts import spec as chartspec
from charts.spec import (
    CHART_TYPES,
    OVERLAP_POLICIES,
    TRANSFORM_OPS,
    ChartSpecError,
)
from ingestion.manifest import load_dataset_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAGSHIP_PATH = REPO_ROOT / "charts" / "spike" / "flagship_spec.json"
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"
DOC_PATH = REPO_ROOT / "charts" / "CHARTSPEC.md"

#: Plotted-value extents for the flagship series, recorded first-party from
#: the #4 spike render + findings note (reviews/spike-04-chart-findings.md:
#: MLO 2025 = 427.35 ppm, GISTEMP 2025 aligned ≈ +1.43 °C). Metadata about
#: our own render, not dataset rows (ADR-023-clean).
FLAGSHIP_EXTENTS = {"co2": (259.6, 427.35), "temp": (-0.71, 1.43)}

#: The manifest disclosure string the flagship's rebaseline_to must carry
#: verbatim (review finding #50: alignment method disclosed on the artefact,
#: generated from the manifest so it cannot drift).
FLAGSHIP_DISCLOSURE = (
    "GISTEMP shifted so its 1880-1900 mean equals the 1800-1900 reference "
    "(Kaufman's native baseline)"
)

SYN_DISCLOSURE = "syn_anom_new shifted so its 1950-1960 mean equals the 1900-1950 reference"


# ---------------------------------------------------------------------------
# Synthetic manifest + spec builders (fresh copies per call — tests mutate)
# ---------------------------------------------------------------------------


def syn_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: a minimal pack manifest in the raw yaml shape.

    Deliberately uses dataset ids that appear nowhere in charts.pack's
    registries, so any validator that consults MVP_DATASET_IDS/PARSERS
    instead of the manifest fails these tests (review finding #117).
    """
    return {
        "version": "0.0.1-test",
        "access_date": "2026-08-16",
        "datasets": {
            "syn_abs_new": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"first_year_ce": 1900, "last_year_ce": 2000},
            },
            "syn_abs_old": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"oldest_bp": 12000, "youngest_bp": -40},
            },
            "syn_anom_new": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "widget_anomaly", "unit": "wd_anomaly"},
                "coverage": {"first_year_ce": 1900, "last_year_ce": 2000},
            },
            "syn_anom_old": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {"name": "widget_anomaly", "unit": "wd_anomaly"},
                "coverage": {"oldest_bp": 12000, "youngest_bp": 0},
            },
            "syn_provisional": {
                "permitted_context": "open-provisional",
                "in_chart_pack": False,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"first_year_ce": 1900, "last_year_ce": 2000},
            },
        },
        "splice_pairs": [
            {
                "id": "syn_abs_pair",
                "paleo": "syn_abs_old",
                "instrumental": "syn_abs_new",
                "splice_year_ce": 1950,
                "rationale": "synthetic",
                "rebaseline": None,
                "resolution_note": "coarse before 1950; annual after",
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "syn_abs_old samples 1950-1990 CE",
                    "note": "overlapping paleo samples omitted for the instrumental record",
                },
            },
            {
                "id": "syn_anom_pair",
                "paleo": "syn_anom_old",
                "instrumental": "syn_anom_new",
                "splice_year_ce": 1950,
                "rationale": "synthetic",
                "rebaseline": {
                    "apply_to": "syn_anom_new",
                    "alignment_period_ce": [1950, 1960],
                    "display_reference": "1900-1950 CE (syn_anom_old native reference)",
                    "disclosure": SYN_DISCLOSURE,
                    "rationale": "synthetic",
                },
                "resolution_note": "centennial before 1950; annual after",
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "syn_anom_old bin 1900-2000 CE",
                    "note": "overlapping proxy bin omitted in favour of the instrumental record",
                },
            },
            {
                "id": "syn_provisional_pair",
                "paleo": "syn_provisional",
                "instrumental": "syn_abs_new",
                "splice_year_ce": 1950,
                "rationale": "synthetic — a pair depending on a non-pack dataset (#117)",
                "rebaseline": None,
                "resolution_note": "synthetic",
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "n/a",
                    "note": "synthetic",
                },
            },
        ],
    }


def line_spec() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: a minimal valid single-dataset line chart."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-line",
        "chart_type": "line",
        "title": "Synthetic widgets over time",
        "time_range_ce": [1900, 2000],
        "series": [
            {
                "id": "w",
                "label": "Widgets (wd)",
                "unit": "wd",
                "dataset": "syn_abs_new",
                "scale_domain": [0, 120],
            }
        ],
    }


LINE_EXTENTS = {"w": (10.0, 100.0)}


def panel_spec() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: a flagship-shaped context+recent-inset pair —

    two spliced dual-axis series, one rebaselined, explicit per-series
    scale domains shared across panels (vocabulary amendments 1-8).
    """
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-panel",
        "chart_type": "context_recent_inset",
        "title": "Synthetic widgets, the long view",
        "time_axis": {"calendar": "CE", "convert_bp": True},
        "panels": {
            "context": {"label": "Last 11,000 years", "time_range_ce": [-9000, 2000]},
            "recent": {
                "label": "Since 1900",
                "time_range_ce": [1900, 2000],
                "shared_y_scale_with": "context",
            },
        },
        "series": [
            {
                "id": "abs",
                "label": "Widgets (wd)",
                "axis": "left",
                "unit": "wd",
                "splice_series": ["syn_abs_old", "syn_abs_new"],
                "splice_pair_id": "syn_abs_pair",
                "overlap_policy": "prefer_instrumental",
                "scale_domain": [40, 120],
                "annotations": {
                    "splice_point": {"year_ce": 1950, "label": "splice: old → new (1950)"},
                    "resolution_note": "coarse before 1950; annual after",
                    "non_zero_baseline": {"label": "axis starts at 40 wd — no zero"},
                },
            },
            {
                "id": "anom",
                "label": "Widget anomaly (wd vs 1900-1950)",
                "axis": "right",
                "unit": "wd_anomaly",
                "splice_series": ["syn_anom_old", "syn_anom_new"],
                "splice_pair_id": "syn_anom_pair",
                "overlap_policy": "prefer_instrumental",
                "rebaseline_to": {
                    "apply_to": "syn_anom_new",
                    "alignment_period_ce": [1950, 1960],
                    "alignment_disclosure": SYN_DISCLOSURE,
                },
                "scale_domain": [-2.0, 2.0],
                "baseline": {"value": 0, "label": "0 = 1900-1950 average"},
                "annotations": {
                    "splice_point": {"year_ce": 1950, "label": "splice: proxy → new (1950)"},
                    "resolution_note": "centennial before 1950; annual after",
                },
            },
        ],
    }


PANEL_EXTENTS = {"abs": (45.0, 110.0), "anom": (-1.2, 1.5)}


def _flagship() -> dict[str, Any]:
    return json.loads(FLAGSHIP_PATH.read_text(encoding="utf-8"))


def _real_manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _pack_confirmed(manifest: dict[str, Any]) -> dict[str, Any]:
    """The real manifest as it will look once #23's written confirmations

    land (in-memory only — the committed manifest must NOT change until
    then): every dataset admitted to the chart pack.
    """
    confirmed = copy.deepcopy(manifest)
    for entry in confirmed["datasets"].values():
        entry["in_chart_pack"] = True
    return confirmed


# ---------------------------------------------------------------------------
# Refusal helpers
# ---------------------------------------------------------------------------


def _refuse(
    spec_dict: dict[str, Any],
    extents: dict[str, tuple[float, float]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> ChartSpecError:
    with pytest.raises(ChartSpecError) as excinfo:
        chartspec.validate_spec(spec_dict, manifest or syn_manifest(), data_extents=extents)
    return excinfo.value


def _assert_refused_at(
    spec_dict: dict[str, Any],
    path: str,
    *,
    contains: tuple[str, ...] = (),
    extents: dict[str, tuple[float, float]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> ChartSpecError:
    err = _refuse(spec_dict, extents=extents, manifest=manifest)
    matches = [v for v in err.violations if v.path == path or v.path.startswith(path + ".")]
    assert matches, (
        f"expected a violation at {path!r}; got paths {[v.path for v in err.violations]}"
    )
    blob = " ".join(v.reason for v in matches)
    for fragment in contains:
        assert fragment in blob, f"violation at {path!r} should mention {fragment!r}: {blob!r}"
    return err


# ---------------------------------------------------------------------------
# The frozen schema shape
# ---------------------------------------------------------------------------


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _property_schemas(schema: dict[str, Any], name: str) -> list[dict[str, Any]]:
    found = []
    for node in _walk(schema):
        props = node.get("properties")
        if isinstance(props, dict) and name in props:
            found.append(props[name])
    return found


def _enum_values(prop_schema: dict[str, Any]) -> set[Any]:
    if "const" in prop_schema:
        return {prop_schema["const"]}
    if "enum" in prop_schema:
        return set(prop_schema["enum"])
    for node in _walk(prop_schema):
        if "enum" in node:
            return set(node["enum"])
        if "const" in node:
            return {node["const"]}
    raise AssertionError(f"no enum/const found in {prop_schema!r}")


def test_schema_chart_types_closed_enum():
    """The five MVP chart types are a closed enum — warming stripes et al.

    are Phase 2 and must not be expressible (DESIGN §3.7, ADR-020).
    """
    schemas = _property_schemas(chartspec.chartspec_schema(), "chart_type")
    assert schemas, "schema has no chart_type property"
    for prop in schemas:
        assert _enum_values(prop) == set(CHART_TYPES)


def test_schema_transform_ops_closed_enum():
    """`transforms[*].op` is a closed enum of exactly the four generic

    transforms — the flagship three are dedicated, manifest-anchored
    fields, never free list entries.
    """
    schemas = _property_schemas(chartspec.chartspec_schema(), "op")
    assert schemas, "schema has no transforms op property"
    for prop in schemas:
        assert _enum_values(prop) == set(TRANSFORM_OPS)


def test_schema_overlap_policy_closed_enum():
    """`overlap_policy` (review finding #47) is a closed enum; the

    per-pair legality narrows further to the manifest's recorded policy.
    """
    schemas = _property_schemas(chartspec.chartspec_schema(), "overlap_policy")
    assert schemas, "schema has no overlap_policy property"
    for prop in schemas:
        assert _enum_values(prop) == set(OVERLAP_POLICIES)


def test_schema_time_axis_is_ce_with_convert_bp():
    """`time_axis` is {calendar: CE, convert_bp: bool}. BCE-aware axis

    labelling is time_axis *semantics* owned by the renderer (vocabulary
    amendment 8) — there is no per-spec label-format or styling field.
    """
    schema = chartspec.chartspec_schema()
    calendar = _property_schemas(schema, "calendar")
    assert calendar and all(_enum_values(p) == {"CE"} for p in calendar)
    convert = _property_schemas(schema, "convert_bp")
    assert convert, "schema has no convert_bp property"
    for name in ("label_format", "axis_style", "tick_format"):
        assert not _property_schemas(schema, name), (
            f"axis labelling is renderer-owned time_axis semantics; no {name!r} field"
        )


def test_schema_has_no_caption_property():
    """Vocabulary amendment 9: captions (sources, licences, access date,

    site URL) are generated from the manifest and deployment config —
    free caption text in an LLM-authored spec is a drift and injection
    vector, so the schema has no caption block at all.
    """
    assert not _property_schemas(chartspec.chartspec_schema(), "caption")


def test_schema_rebaseline_requires_disclosure():
    """`rebaseline_to` requires apply_to + alignment_period_ce +

    alignment_disclosure (review finding #50: the alignment method must
    reach the artefact itself, not only repo files).
    """
    schema = chartspec.chartspec_schema()
    needed = {"apply_to", "alignment_period_ce", "alignment_disclosure"}
    for node in _walk(schema):
        required = node.get("required")
        if isinstance(required, list) and needed <= set(required):
            return
    raise AssertionError(
        "no schema object requires apply_to + alignment_period_ce + alignment_disclosure"
    )


def test_schema_names_the_load_bearing_fields():
    """The vocabulary's load-bearing fields all exist as schema properties."""
    schema = chartspec.chartspec_schema()
    for name in (
        "splice_pair_id",
        "splice_series",
        "overlap_policy",
        "scale_domain",
        "non_zero_baseline",
        "uncertainty_band",
        "time_range_ce",
        "panels",
        "shared_y_scale_with",
    ):
        assert _property_schemas(schema, name), f"schema is missing the {name!r} property"


# ---------------------------------------------------------------------------
# Closed vocabulary, behaviourally
# ---------------------------------------------------------------------------


def test_unknown_chart_type_rejected():
    spec = line_spec()
    spec["chart_type"] = "warming_stripes"
    _assert_refused_at(spec, "chart_type", contains=("warming_stripes",))


def test_unknown_transform_rejected():
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "loess_smooth", "window_years": 5}]
    _assert_refused_at(spec, "series[0].transforms[0].op", contains=("loess_smooth",))


def test_unknown_keys_rejected_everywhere():
    """additionalProperties: false throughout — an LLM cannot smuggle

    fields past the frozen vocabulary, and annotation *placement* is
    fixed semantics (amendment 7), not a spec choice.
    """
    spec = line_spec()
    spec["custom_css"] = "body { display: none }"
    _assert_refused_at(spec, "custom_css")

    spec = line_spec()
    spec["series"][0]["renderer_hint"] = "log-scale please"
    _assert_refused_at(spec, "series[0].renderer_hint")

    spec = panel_spec()
    spec["series"][0]["annotations"]["placement"] = "inset_only"
    _assert_refused_at(spec, "series[0].annotations.placement", extents=PANEL_EXTENTS)

    # unit_conversion carries only to_unit: the conversion arithmetic is
    # code-owned — an LLM-authored factor would be a silent scaling attack.
    spec = line_spec()
    spec["series"][0]["unit"] = "kwd"
    spec["series"][0]["transforms"] = [{"op": "unit_conversion", "to_unit": "kwd", "factor": 2.0}]
    _assert_refused_at(spec, "series[0].transforms[0].factor")


# ---------------------------------------------------------------------------
# Dataset existence + structural pack gating (#117)
# ---------------------------------------------------------------------------


def test_unknown_dataset_id_rejected():
    spec = line_spec()
    spec["series"][0]["dataset"] = "atlantis_sst"
    _assert_refused_at(spec, "series[0].dataset", contains=("atlantis_sst",))


def test_dataset_gating_derives_from_manifest_not_pack_registry():
    """Review finding #117: dataset legality comes from the manifest the

    validator is handed — in_chart_pack — never from charts.pack's
    registries. The synthetic ids are deliberately absent from those
    registries, and a fully valid spec over them must pass.
    """
    assert "syn_abs_new" not in pack.MVP_DATASET_IDS
    assert "syn_abs_new" not in pack.PARSERS
    assert chartspec.validate_spec(line_spec(), syn_manifest(), data_extents=LINE_EXTENTS) is None


def test_non_pack_dataset_refused_naming_provisional_status():
    """Review finding #117: in_chart_pack: false is a structural refusal

    at the spec layer — the honest-refusal message names the dataset and
    its provisional status, for both direct references and splice pairs.
    """
    spec = line_spec()
    spec["series"][0]["dataset"] = "syn_provisional"
    _assert_refused_at(
        spec,
        "series[0].dataset",
        contains=("syn_provisional", "open-provisional"),
        extents=None,
    )

    spec = line_spec()
    spec["series"][0] = {
        "id": "w",
        "label": "Widgets (wd)",
        "unit": "wd",
        "splice_series": ["syn_provisional", "syn_abs_new"],
        "splice_pair_id": "syn_provisional_pair",
        "overlap_policy": "prefer_instrumental",
        "annotations": {
            "splice_point": {"year_ce": 1950, "label": "splice (1950)"},
            "resolution_note": "synthetic",
        },
    }
    err = _refuse(spec)
    assert "syn_provisional" in str(err) and "open-provisional" in str(err)


def test_series_unit_must_match_dataset_unit():
    """A series' declared unit must equal the dataset's variable unit (or

    the to_unit of its unit_conversion transform) — anything else is a
    mislabelled axis.
    """
    spec = line_spec()
    spec["series"][0]["unit"] = "wd_anomaly"
    _assert_refused_at(spec, "series[0].unit", contains=("wd",))

    spec = line_spec()
    spec["series"][0]["unit"] = "wd"
    spec["series"][0]["transforms"] = [{"op": "unit_conversion", "to_unit": "kwd"}]
    _assert_refused_at(spec, "series[0].unit", contains=("kwd",))


# ---------------------------------------------------------------------------
# Splice / rebaseline / overlap: manifest cross-checks (ADR-020)
# ---------------------------------------------------------------------------


def test_splice_without_annotation_rejected():
    """DESIGN §3.7: unannotated splices are the hidden-smoothing attack

    surface — splice_point and resolution_note are both mandatory.
    """
    spec = panel_spec()
    del spec["series"][0]["annotations"]["splice_point"]
    _assert_refused_at(spec, "series[0].annotations.splice_point", extents=PANEL_EXTENTS)

    spec = panel_spec()
    del spec["series"][0]["annotations"]["resolution_note"]
    _assert_refused_at(spec, "series[0].annotations.resolution_note", extents=PANEL_EXTENTS)


def test_splice_pair_not_in_manifest_rejected():
    spec = panel_spec()
    spec["series"][0]["splice_pair_id"] = "syn_bogus_pair"
    _assert_refused_at(
        spec, "series[0].splice_pair_id", contains=("syn_bogus_pair",), extents=PANEL_EXTENTS
    )


def test_splice_series_must_match_manifest_pair():
    spec = panel_spec()
    spec["series"][0]["splice_series"] = ["syn_abs_new", "syn_abs_old"]  # reversed
    _assert_refused_at(
        spec,
        "series[0].splice_series",
        contains=("syn_abs_pair",),
        extents=PANEL_EXTENTS,
    )


def test_splice_year_must_match_manifest():
    spec = panel_spec()
    spec["series"][0]["annotations"]["splice_point"]["year_ce"] = 1949
    _assert_refused_at(
        spec,
        "series[0].annotations.splice_point.year_ce",
        contains=("1950",),
        extents=PANEL_EXTENTS,
    )


def test_llm_chosen_alignment_period_rejected():
    """ADR-020: alignment periods are fixed per pair in the manifest — a

    scientific decision made at curation time, never by the LLM.
    """
    spec = panel_spec()
    spec["series"][1]["rebaseline_to"]["alignment_period_ce"] = [1930, 1940]
    _assert_refused_at(
        spec,
        "series[1].rebaseline_to.alignment_period_ce",
        contains=("1950",),
        extents=PANEL_EXTENTS,
    )


def test_rebaseline_apply_to_must_match_manifest():
    """Vocabulary amendment 2: rebaseline_to names which member shifts;

    shifting the wrong member silently moves a published reconstruction
    off its native reference.
    """
    spec = panel_spec()
    spec["series"][1]["rebaseline_to"]["apply_to"] = "syn_anom_old"
    _assert_refused_at(
        spec,
        "series[1].rebaseline_to.apply_to",
        contains=("syn_anom_new",),
        extents=PANEL_EXTENTS,
    )


def test_rebaseline_presence_must_match_manifest():
    """A pair with a manifest rebaseline must carry rebaseline_to; a pair

    with an explicit rebaseline: null must not (no rebaseline is legal).
    """
    spec = panel_spec()
    del spec["series"][1]["rebaseline_to"]
    _assert_refused_at(spec, "series[1].rebaseline_to", extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["series"][0]["rebaseline_to"] = {
        "apply_to": "syn_abs_new",
        "alignment_period_ce": [1950, 1960],
        "alignment_disclosure": "illegitimate",
    }
    _assert_refused_at(spec, "series[0].rebaseline_to", extents=PANEL_EXTENTS)


def test_missing_overlap_policy_rejected():
    """Review finding #47: where spliced sources overlap in time, the

    spec must say what happens to the overlapping data — silence was the
    residual 'hidden the decline'-class attack.
    """
    spec = panel_spec()
    del spec["series"][0]["overlap_policy"]
    _assert_refused_at(spec, "series[0].overlap_policy", extents=PANEL_EXTENTS)


def test_illegal_overlap_policy_rejected():
    """The policy must be the manifest-legal one for the pair (and inside

    the closed enum at all) — the LLM cannot re-decide a curation call.
    """
    spec = panel_spec()
    spec["series"][0]["overlap_policy"] = "show_both"
    _assert_refused_at(
        spec,
        "series[0].overlap_policy",
        contains=("prefer_instrumental",),
        extents=PANEL_EXTENTS,
    )

    spec = panel_spec()
    spec["series"][0]["overlap_policy"] = "discard_quietly"
    _assert_refused_at(spec, "series[0].overlap_policy", extents=PANEL_EXTENTS)


def test_missing_or_drifted_alignment_disclosure_rejected():
    """Review finding #50: the alignment disclosure must be present and

    must equal the manifest's rebaseline.disclosure verbatim — it reaches
    the artefact and cannot drift from the curation record.
    """
    spec = panel_spec()
    del spec["series"][1]["rebaseline_to"]["alignment_disclosure"]
    _assert_refused_at(spec, "series[1].rebaseline_to.alignment_disclosure", extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["series"][1]["rebaseline_to"]["alignment_disclosure"] = "aligned on some period"
    _assert_refused_at(spec, "series[1].rebaseline_to.alignment_disclosure", extents=PANEL_EXTENTS)


def test_uncertainty_band_source_must_be_series_member():
    """Vocabulary amendment 6: bands attach to a splice member; a source

    outside the series' datasets is meaningless.
    """
    spec = panel_spec()
    spec["series"][1]["uncertainty_band"] = {
        "lower": "lo",
        "upper": "hi",
        "source": "syn_abs_new",
    }
    _assert_refused_at(
        spec,
        "series[1].uncertainty_band.source",
        contains=("syn_abs_new",),
        extents=PANEL_EXTENTS,
    )


# ---------------------------------------------------------------------------
# Ranges, coverage semantics (#52) and scale-domain integrity (#48)
# ---------------------------------------------------------------------------


def test_range_outside_dataset_coverage_rejected():
    """Review finding #52 semantics: coverage is *parsed usable extent*,

    and a requested range beyond it is refused — the validator consumes
    the manifest's coverage block, it never re-measures data.
    """
    spec = line_spec()
    spec["time_range_ce"] = [1890, 2000]
    _assert_refused_at(spec, "time_range_ce", contains=("syn_abs_new", "1900"))

    spec = line_spec()
    spec["time_range_ce"] = [1900, 2001]
    _assert_refused_at(spec, "time_range_ce", contains=("syn_abs_new", "2000"))


def test_panel_range_outside_coverage_rejected():
    """Panel ranges obey the same coverage rule (for a splice: the union

    of the members' coverage, BP endpoints converted via present_ce).
    """
    spec = panel_spec()
    spec["panels"]["context"]["time_range_ce"] = [-11000, 2000]  # before syn_abs_old's oldest
    _assert_refused_at(spec, "panels.context.time_range_ce", extents=PANEL_EXTENTS)


def test_coverage_unit_mismatch_rejected():
    """A series over a years_bp dataset in a CE-ranged spec needs

    time_axis.convert_bp: true — without it, the spec's CE ranges cannot
    be compared against BP coverage at all.
    """
    spec = panel_spec()
    del spec["time_axis"]
    _assert_refused_at(spec, "time_axis", contains=("convert_bp",), extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["time_axis"]["convert_bp"] = False
    _assert_refused_at(spec, "time_axis", contains=("convert_bp",), extents=PANEL_EXTENTS)


def test_scale_domain_clipping_data_rejected():
    """Review finding #48: an LLM-authored axis window is a cherry-pick

    vector exactly like a time window — the domain must contain the
    plotted data's extent.
    """
    spec = line_spec()
    spec["series"][0]["scale_domain"] = [0, 90]  # clips the top (extent max 100)
    _assert_refused_at(spec, "series[0].scale_domain", extents=LINE_EXTENTS)

    spec = line_spec()
    spec["series"][0]["scale_domain"] = [20, 120]  # clips the bottom (extent min 10)
    spec["series"][0]["annotations"] = {"non_zero_baseline": {"label": "axis starts at 20 wd"}}
    _assert_refused_at(spec, "series[0].scale_domain", extents=LINE_EXTENTS)


def test_scale_domain_zero_exclusion_requires_annotation():
    """Review finding #48 + DESIGN §3.7 as amended: any axis whose domain

    excludes zero — on every axis of a multi-axis chart, absolute or
    anomaly alike — must carry the non_zero_baseline annotation. The
    flagship's CO₂ axis (starts at 180 ppm) is the precedent.
    """
    spec = line_spec()
    spec["series"][0]["scale_domain"] = [5, 120]
    _assert_refused_at(spec, "series[0].annotations.non_zero_baseline", extents=LINE_EXTENTS)

    # With the annotation the same spec is legal.
    spec["series"][0]["annotations"] = {
        "non_zero_baseline": {"label": "axis starts at 5 wd — it does not include zero"}
    }
    assert chartspec.validate_spec(spec, syn_manifest(), data_extents=LINE_EXTENTS) is None


def test_panel_pair_misuse_rejected():
    """The panel pair is a layout wrapper with hard rules (amendments 4-5):

    panels iff context_recent_inset, both panels present, recent within
    context, shared_y_scale_with resolvable, per-series scale_domain
    explicit (shared per-axis domains cannot be implicit).
    """
    spec = line_spec()
    spec["panels"] = {"context": {"label": "x", "time_range_ce": [1900, 2000]}}
    _assert_refused_at(spec, "panels", extents=LINE_EXTENTS)

    spec = panel_spec()
    del spec["panels"]["recent"]
    _assert_refused_at(spec, "panels.recent", extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["panels"]["recent"]["time_range_ce"] = [1900, 2010]  # beyond context (and coverage)
    _assert_refused_at(spec, "panels.recent.time_range_ce", extents=PANEL_EXTENTS)

    spec = panel_spec()
    spec["panels"]["recent"]["shared_y_scale_with"] = "sidebar"
    _assert_refused_at(spec, "panels.recent.shared_y_scale_with", extents=PANEL_EXTENTS)

    spec = panel_spec()
    del spec["series"][0]["scale_domain"]
    _assert_refused_at(spec, "series[0].scale_domain", extents=PANEL_EXTENTS)


def test_valid_specs_accepted():
    """Guard against over-refusal: the synthetic line and panel specs are

    valid, with and without the optional pieces.
    """
    manifest = syn_manifest()
    assert chartspec.validate_spec(line_spec(), manifest, data_extents=LINE_EXTENTS) is None

    # Structural-only validation (no extents yet) also passes.
    assert chartspec.validate_spec(line_spec(), manifest) is None

    # scale_domain is optional outside the panel pair (the renderer
    # defaults to the full extent — no cherry-picked windows).
    spec = line_spec()
    del spec["series"][0]["scale_domain"]
    assert chartspec.validate_spec(spec, manifest, data_extents=LINE_EXTENTS) is None

    # A semantically inert _meta provenance block is admitted.
    spec = line_spec()
    spec["_meta"] = {"provenance": "synthetic fixture"}
    assert chartspec.validate_spec(spec, manifest, data_extents=LINE_EXTENTS) is None

    # Generic transforms from the closed vocabulary are legal.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": 10}]
    assert chartspec.validate_spec(spec, manifest, data_extents=LINE_EXTENTS) is None

    # The full panel pair, uncertainty band on a splice member included.
    spec = panel_spec()
    spec["series"][1]["uncertainty_band"] = {
        "lower": "lo",
        "upper": "hi",
        "source": "syn_anom_old",
    }
    assert chartspec.validate_spec(spec, manifest, data_extents=PANEL_EXTENTS) is None


# ---------------------------------------------------------------------------
# The committed flagship spec + the real manifest
# ---------------------------------------------------------------------------


def test_flagship_spec_validates_with_pack_confirmed_manifest():
    """The vocabulary-covers-the-flagship regression (issue #15 acceptance

    criterion 1): the committed #4 flagship spec validates end-to-end
    against the real manifest *as it will stand once #23's written
    confirmations land* (in_chart_pack flipped in memory — the flagship's
    splice pairs depend on open-provisional datasets, #117, so against
    today's manifest it must refuse instead; see the next test).
    """
    manifest = _pack_confirmed(_real_manifest())
    assert chartspec.validate_spec(_flagship(), manifest, data_extents=FLAGSHIP_EXTENTS) is None


def test_flagship_refused_today_naming_provisional_datasets():
    """Review finding #117: against the committed manifest, the flagship

    is refused — both splice pairs reference open-provisional datasets —
    and the refusal names each dataset and its provisional status, so the
    honest-refusal path (#16) can explain the block.
    """
    err = _refuse(_flagship(), extents=FLAGSHIP_EXTENTS, manifest=_real_manifest())
    message = str(err)
    for ds_id in ("kaufman2020_temp12k", "bereiter2015_co2"):
        assert ds_id in message
    assert "open-provisional" in message


def test_flagship_spec_carries_frozen_mandatory_fields():
    """Fixture pin: the committed flagship carries the #15 mandatory

    fields and the #49/#52 honesty fixes, so a later edit cannot silently
    regress the reference spec.
    """
    flagship = _flagship()
    assert "caption" not in flagship, "captions are manifest-generated (amendment 9)"
    assert flagship["panels"]["recent"]["label"] == "Since 1850", (
        "#49: the inset shows 1850-1959 ice-core data — it must not claim 'instrumental era'"
    )
    for panel in flagship["panels"].values():
        assert panel["time_range_ce"][1] <= 2025, (
            "#52: panel ranges must not exceed the parsed usable extent (2025)"
        )
    co2, temp = flagship["series"]
    assert co2["overlap_policy"] == "prefer_instrumental"
    assert temp["overlap_policy"] == "prefer_instrumental"
    assert co2["annotations"]["non_zero_baseline"]["label"], (
        "#48: the CO₂ axis starts at 180 ppm and must say so"
    )
    assert temp["rebaseline_to"]["alignment_disclosure"] == FLAGSHIP_DISCLOSURE


def _coverage_ce(entry: dict[str, Any]) -> tuple[float, float]:
    coverage, time_axis = entry["coverage"], entry["time_axis"]
    if time_axis["unit"] == "year_ce":
        return coverage["first_year_ce"], coverage["last_year_ce"]
    present = time_axis["present_ce"]
    return present - coverage["oldest_bp"], present - coverage["youngest_bp"]


def test_manifest_splice_pairs_record_overlap_policy():
    """Review finding #47, manifest side: every splice pair whose member

    coverages overlap in time must record the curation-time overlap
    decision — policy (from the closed set), what was discarded, and the
    disclosure note the renderer will surface. Both committed pairs
    overlap (Bereiter reaches 2001 CE past the 1959 splice; Kaufman's
    0 BP bin reaches 1950 past the 1880 splice).
    """
    raw = _real_manifest()
    datasets = raw["datasets"]
    pairs = {p["id"]: p for p in raw["splice_pairs"]}
    assert {"co2_10k", "temp_10k"} <= set(pairs)

    for pair_id, pair in pairs.items():
        paleo_cov = _coverage_ce(datasets[pair["paleo"]])
        instr_cov = _coverage_ce(datasets[pair["instrumental"]])
        overlaps = max(paleo_cov[0], instr_cov[0]) <= min(paleo_cov[1], instr_cov[1])
        if not overlaps:
            continue
        overlap = pair.get("overlap")
        assert overlap, f"{pair_id}: member coverages overlap but no overlap block (#47)"
        assert overlap.get("policy") in OVERLAP_POLICIES, f"{pair_id}: {overlap.get('policy')!r}"
        assert (overlap.get("note") or "").strip(), f"{pair_id}: overlap.note missing"
        if str(overlap.get("policy", "")).startswith("prefer_"):
            assert (overlap.get("discarded") or "").strip(), (
                f"{pair_id}: a prefer_* policy discards data and must say what (#47)"
            )

    # The curation decision matching the spike's actual behaviour: the
    # instrumental record wins the overlap, and the Bereiter case names
    # the discarded 1959-2001 samples.
    co2_overlap = pairs["co2_10k"]["overlap"]
    assert co2_overlap["policy"] == "prefer_instrumental"
    assert "1959" in (co2_overlap.get("discarded", "") + co2_overlap.get("note", ""))
    assert pairs["temp_10k"]["overlap"]["policy"] == "prefer_instrumental"

    # The issue #5 validators must keep accepting the extended manifest.
    load_dataset_manifest(MANIFEST_PATH)


def test_manifest_rebaseline_disclosure_matches_flagship():
    """Review finding #50, manifest side: the temp_10k rebaseline block

    carries the artefact disclosure string, it names both the alignment
    period and the displayed reference, and the committed flagship spec
    carries it verbatim (the validator's equality contract).
    """
    pairs = {p["id"]: p for p in _real_manifest()["splice_pairs"]}
    disclosure = (pairs["temp_10k"]["rebaseline"] or {}).get("disclosure", "")
    assert disclosure.strip(), "temp_10k rebaseline has no disclosure string (#50)"
    assert "1880" in disclosure and "1800" in disclosure, (
        "the disclosure must name the alignment period (1880-1900) and the "
        "displayed reference (1800-1900)"
    )
    flagship_temp = _flagship()["series"][1]
    assert flagship_temp["rebaseline_to"]["alignment_disclosure"] == disclosure


# ---------------------------------------------------------------------------
# Actionable refusal messages (issue #15 acceptance criterion)
# ---------------------------------------------------------------------------

_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[\d+\])?(\.[A-Za-z_][A-Za-z0-9_]*(\[\d+\])?)*$")


def _invalid_suite() -> list[tuple[str, dict[str, Any], dict | None, dict | None]]:
    """(case id, spec, extents, manifest) for every refusal case above."""
    cases: list[tuple[str, dict[str, Any], dict | None, dict | None]] = []

    def case(name: str, spec: dict[str, Any], extents=None, manifest=None):
        cases.append((name, spec, extents, manifest))

    s = line_spec()
    s["chart_type"] = "warming_stripes"
    case("unknown_chart_type", s)

    s = line_spec()
    s["series"][0]["transforms"] = [{"op": "loess_smooth"}]
    case("unknown_transform", s)

    s = line_spec()
    s["series"][0]["dataset"] = "atlantis_sst"
    case("unknown_dataset", s)

    s = line_spec()
    s["series"][0]["dataset"] = "syn_provisional"
    case("non_pack_dataset", s)

    s = panel_spec()
    del s["series"][0]["annotations"]["splice_point"]
    case("splice_without_annotation", s, PANEL_EXTENTS)

    s = panel_spec()
    s["series"][0]["splice_pair_id"] = "syn_bogus_pair"
    case("splice_pair_not_in_manifest", s, PANEL_EXTENTS)

    s = panel_spec()
    s["series"][1]["rebaseline_to"]["alignment_period_ce"] = [1930, 1940]
    case("llm_chosen_alignment_period", s, PANEL_EXTENTS)

    s = panel_spec()
    del s["series"][0]["overlap_policy"]
    case("missing_overlap_policy", s, PANEL_EXTENTS)

    s = panel_spec()
    del s["series"][1]["rebaseline_to"]["alignment_disclosure"]
    case("missing_alignment_disclosure", s, PANEL_EXTENTS)

    s = line_spec()
    s["time_range_ce"] = [1890, 2000]
    case("range_outside_coverage", s)

    s = panel_spec()
    del s["time_axis"]
    case("coverage_unit_mismatch", s, PANEL_EXTENTS)

    s = line_spec()
    s["series"][0]["scale_domain"] = [0, 90]
    case("scale_domain_clips_data", s, LINE_EXTENTS)

    s = line_spec()
    s["series"][0]["scale_domain"] = [5, 120]
    case("zero_exclusion_unannotated", s, LINE_EXTENTS)

    s = panel_spec()
    del s["panels"]["recent"]
    case("panel_pair_misuse", s, PANEL_EXTENTS)

    case("flagship_provisional_today", _flagship(), FLAGSHIP_EXTENTS, _real_manifest())

    return cases


def test_rejection_messages_name_field_and_reason():
    """Issue #15 acceptance: every rejection across the whole invalid-spec

    suite is actionable — a well-formed spec path, a substantive reason,
    and the exception message itself carries the path so a bare log line
    is enough to fix the spec.
    """
    for name, spec, extents, manifest in _invalid_suite():
        err = _refuse(spec, extents=extents, manifest=manifest)
        assert err.violations, f"{name}: refused with no recorded violations"
        for violation in err.violations:
            assert _PATH_RE.match(violation.path), (
                f"{name}: malformed violation path {violation.path!r}"
            )
            assert len(violation.reason.strip()) >= 20, (
                f"{name}: reason too thin to act on: {violation.reason!r}"
            )
            assert violation.path in str(err), (
                f"{name}: exception message must carry the path {violation.path!r}"
            )


# ---------------------------------------------------------------------------
# Schema doc for the planner prompt (#16)
# ---------------------------------------------------------------------------


def test_schema_doc_matches_schema():
    """Issue #15 acceptance criterion 2: charts/CHARTSPEC.md is derived

    from the schema (byte-identical to render_schema_doc()), so the doc
    the planner prompt embeds can never drift — and it names the whole
    frozen vocabulary.
    """
    assert DOC_PATH.is_file(), "charts/CHARTSPEC.md is missing"
    text = DOC_PATH.read_text(encoding="utf-8")
    assert text == chartspec.render_schema_doc()
    for token in sorted(CHART_TYPES | TRANSFORM_OPS | OVERLAP_POLICIES):
        assert token in text, f"schema doc does not name {token!r}"
    for field in (
        "splice_pair_id",
        "overlap_policy",
        "alignment_disclosure",
        "scale_domain",
        "non_zero_baseline",
        "time_range_ce",
        "convert_bp",
    ):
        assert field in text, f"schema doc does not name the {field!r} field"
