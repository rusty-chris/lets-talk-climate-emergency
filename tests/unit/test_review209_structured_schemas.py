"""Review finding #209 — RED: every structured-request schema stays inside
the structured-outputs supported JSON-Schema subset.

The chart planner's structured-output schema (``charts/planner.py`` →
``charts.spec.chartspec_schema``) carries the same off-subset
``minItems``/``maxItems`` pattern blocker #203 fixed on the citation
validator: the live structured call would 400 (or have the constraint
silently stripped) on EVERY chart plan while the unit suite stays green.

Three layers, mirroring the ratified #203 remedy:

1. **The shared subset lint** (``tests._schema_subset``, promoted from the
   #203 walker) applied to EVERY structured-request builder in the repo,
   with a call-site enumeration sweep so a future builder cannot dodge
   the lint by simply not being listed here.
2. **Prompt-side re-homing**: the counting/shape steering the schema can
   no longer express moves into the planner's system instructions
   ("exactly two numbers" per [start, end] pair, the series and
   transform count bounds) — the analogue of #203's "exactly N verdicts"
   line.
3. **Parser-side re-homing**: ``charts.spec.validate_spec`` must keep
   refusing miscounted arrays *without* relying on schema
   ``minItems``/``maxItems`` — pinned here (green today via jsonschema)
   so stripping the off-subset keys without re-homing the invariant
   fails these tests, not production.

Unit tier: pure builders and validators, no network, no API key.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

from charts import planner
from charts import spec as chartspec
from rag.citation_validator import EntailmentPair, ValidatorConfig, build_validation_request
from rag.query import build_query_processing_request
from tests._schema_subset import assert_schema_within_structured_outputs_subset

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"

#: Production packages swept for ``ProviderAdapter.structured`` call sites.
#: Spike directories are excluded: frozen spike evidence (rag/spike_03
#: deliberately RECORDS the 400 behaviour; charts/spike never calls the
#: adapter) — never live production calls.
PRODUCTION_PACKAGES = (
    "charts",
    "corpus",
    "evals",
    "ingestion",
    "letters",
    "rag",
    "scripts",
    "service",
    "ui",
    "voices",
)


def _entailment_pairs(count: int) -> list[EntailmentPair]:
    return [
        EntailmentPair(
            pair_index=i,
            sentence_index=i + 1,
            sentence_text=f"Synthetic factual sentence {i} about invented basin warming.",
            document_index=i,
            block_text=f"Synthetic passage {i}: the invented basin very likely warmed.",
        )
        for i in range(count)
    ]


def _planner_requests() -> list[tuple[str, dict[str, Any]]]:
    real_catalogue = planner.build_dataset_catalogue(str(MANIFEST_PATH))
    return [
        (
            "build_planner_request[real-manifest-catalogue]",
            planner.build_planner_request("plot co2 over the last 10,000 years", real_catalogue),
        ),
        (
            "build_planner_request[empty-catalogue]",
            planner.build_planner_request("plot widgets", {"datasets": {}, "splice_pairs": []}),
        ),
    ]


def _validation_requests() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            f"build_validation_request[pairs={count}]",
            build_validation_request(_entailment_pairs(count), config=ValidatorConfig()),
        )
        for count in (1, 2, 45)
    ]


def _query_processing_requests() -> list[tuple[str, dict[str, Any]]]:
    history = [
        {"role": "user", "content": "How much has the planet warmed?"},
        {"role": "assistant", "content": "About 1.3 degC above the 1850-1900 baseline."},
    ]
    return [
        ("build_query_processing_request[bare]", build_query_processing_request("plot that")),
        (
            "build_query_processing_request[history]",
            build_query_processing_request("plot that", history=history),
        ),
    ]


#: Module (repo-relative) -> builder sweep, for every production module
#: that calls ``ProviderAdapter.structured``. A NEW structured call site
#: must register its request builder here (and pass the subset lint) —
#: the enumeration test below fails otherwise, so no future schema can
#: dodge the lint the way charts/spec.py dodged #203's.
STRUCTURED_REQUEST_BUILDERS: dict[str, Any] = {
    "charts/planner.py": _planner_requests,
    "rag/citation_validator.py": _validation_requests,
    "rag/query.py": _query_processing_requests,
}


def _structured_call_sites() -> set[str]:
    sites: set[str] = set()
    for package in PRODUCTION_PACKAGES:
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if "spike" in relative:
                continue
            if re.search(r"\.structured\(", path.read_text(encoding="utf-8")):
                sites.add(relative)
    return sites


def _all_requests() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    for factory in STRUCTURED_REQUEST_BUILDERS.values():
        cases.extend(factory())
    return cases


class TestEveryStructuredBuilderIsLinted:
    def test_every_structured_call_site_has_a_registered_builder(self):
        """The lint's coverage is enumerated, not hoped for: every
        production module that calls ``.structured(`` must have its
        request builder registered in STRUCTURED_REQUEST_BUILDERS. A new
        structured call site fails here until its schema joins the
        subset sweep — the guard #203 lacked when charts/spec.py shipped
        the same off-subset pattern (#209)."""
        sites = _structured_call_sites()
        assert sites, "expected at least one production .structured( call site"
        assert sites == set(STRUCTURED_REQUEST_BUILDERS), (
            "production .structured( call sites and the registered builder sweep "
            f"disagree: call sites {sorted(sites)} vs registered "
            f"{sorted(STRUCTURED_REQUEST_BUILDERS)} — register the new builder in "
            "STRUCTURED_REQUEST_BUILDERS so its schema is subset-linted (#209)"
        )

    @pytest.mark.parametrize(
        ("label", "request_payload"),
        _all_requests(),
        ids=[label for label, _ in _all_requests()],
    )
    def test_structured_request_schema_stays_inside_the_subset(self, label, request_payload):
        """Finding #209 (same class as blocker #203): each built request's
        ``schema`` walks clean against the structured-outputs supported
        subset — no ``maxItems``, no ``minItems`` above 1, no
        ``minimum``/``maximum``/``multipleOf``/``minLength``/``maxLength``,
        every object node closed. RED today for the chart planner, whose
        schema embeds charts.spec.chartspec_schema's minItems/maxItems
        bounds and open ``_meta`` object."""
        assert "schema" in request_payload, f"{label}: request has no schema"
        assert_schema_within_structured_outputs_subset(request_payload["schema"], name=label)

    @pytest.mark.parametrize(
        "off_subset_node",
        [
            {"type": "array", "items": {"type": "number"}, "maxItems": 2},
            {"type": "array", "items": {"type": "number"}, "minItems": 2},
            {"type": "number", "minimum": 0},
            {"type": "string", "maxLength": 40},
            {"type": "object", "properties": {}},  # unclosed object
        ],
    )
    def test_shared_walker_actually_flags_off_subset_nodes(self, off_subset_node):
        """Self-check on the promoted helper: a watered-down walker that
        stops flagging the documented-unsupported keys (or unclosed
        objects) would silently disarm both the #203 and #209 guards."""
        nested = {
            "type": "object",
            "additionalProperties": False,
            "required": ["field"],
            "properties": {"field": off_subset_node},
        }
        with pytest.raises(AssertionError):
            assert_schema_within_structured_outputs_subset(nested, name="self-check")


class TestPlannerPromptCarriesReHomedCounts:
    def test_planner_system_instructions_state_the_shape_bounds(self):
        """The counting/shape steering the request schema can no longer
        express (subset: no minItems above 1, no maxItems) must ride the
        prompt instead — the #203 "exactly N verdicts" remedy applied to
        the planner. The system text must state, in some phrasing the
        regexes below admit: every [start, end] range/pair carries
        exactly two numbers; a spec carries at least one and at most 8
        series; a series carries at most 4 transforms. (splice_series
        needs no prompt line: the catalogue names each pair's two members
        and validate_spec refuses any other membership.)"""
        request = planner.build_planner_request(
            "plot widgets", {"datasets": {}, "splice_pairs": []}
        )
        system = request["system"]
        assert re.search(r"exactly two numbers", system, re.IGNORECASE), (
            "the system instructions must state that every [start, end] "
            "range/pair carries exactly two numbers (re-homed from the schema's "
            "minItems/maxItems: 2, #209)"
        )
        assert re.search(r"at least one series", system, re.IGNORECASE), (
            "the system instructions must state the at-least-one-series floor "
            "(re-homed from the schema's series minItems: 1, #209)"
        )
        assert re.search(r"at most (8|eight) series", system, re.IGNORECASE), (
            "the system instructions must state the 8-series ceiling (re-homed "
            "from the schema's series maxItems: 8, #209)"
        )
        assert re.search(r"at most (4|four) transforms", system, re.IGNORECASE), (
            "the system instructions must state the 4-transforms-per-series "
            "ceiling (re-homed from the schema's transforms maxItems: 4, #209)"
        )


# ---------------------------------------------------------------------------
# Parser-side re-homing: validate_spec keeps the counting invariants alive
# without schema minItems/maxItems (green today via jsonschema; these pins
# force the implementer to re-home, not just delete, the bounds)
# ---------------------------------------------------------------------------


def _syn_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: minimal raw-mapping pack manifest (invented
    "widgets" datasets — no real values anywhere)."""
    return {
        "datasets": {
            "syn_widgets": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"first_year_ce": 1900, "last_year_ce": 2000},
            },
            "syn_widgets_old": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"oldest_bp": 12000, "youngest_bp": -40},
            },
        },
        "splice_pairs": [
            {
                "id": "syn_pair",
                "paleo": "syn_widgets_old",
                "instrumental": "syn_widgets",
                "splice_year_ce": 1950,
                "rationale": "synthetic",
                "rebaseline": None,
                "resolution_note": "coarse before 1950; annual after",
                "overlap": {
                    "policy": "prefer_instrumental",
                    "discarded": "syn_widgets_old samples 1950-1990 CE",
                    "note": "overlapping paleo samples omitted",
                },
            }
        ],
    }


def _line_spec() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: minimal valid single-dataset line chart."""
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
                "dataset": "syn_widgets",
                "scale_domain": [0, 120],
            }
        ],
    }


def _spliced_spec() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: minimal valid spliced line chart."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-splice",
        "chart_type": "line",
        "title": "Synthetic widgets, spliced",
        "time_range_ce": [1900, 1990],
        "time_axis": {"calendar": "CE", "convert_bp": True},
        "series": [
            {
                "id": "sw",
                "label": "Widgets (wd)",
                "unit": "wd",
                "splice_series": ["syn_widgets_old", "syn_widgets"],
                "splice_pair_id": "syn_pair",
                "overlap_policy": "prefer_instrumental",
                "annotations": {
                    "splice_point": {"year_ce": 1950, "label": "splice: old → new (1950)"},
                    "resolution_note": "coarse before 1950; annual after",
                },
            }
        ],
    }


def _violation_paths(spec: dict[str, Any]) -> list[str]:
    with pytest.raises(chartspec.ChartSpecError) as excinfo:
        chartspec.validate_spec(spec, _syn_manifest())
    return [violation.path for violation in excinfo.value.violations]


class TestValidateSpecKeepsCountingInvariantsWithoutSchemaBounds:
    """The invariants the off-subset keys carried, pinned at the parse
    path (``validate_spec`` runs on every planner output before anything
    downstream sees it). Green today — jsonschema's minItems/maxItems
    provide them — and they must SURVIVE the #209 fix stripping those
    keys from the request schema: refusal in code, steering in prompt,
    exactly the #203 split."""

    def test_control_specs_are_valid(self):
        """The mutations below must be the only reason for a refusal."""
        assert chartspec.validate_spec(_line_spec(), _syn_manifest()) is None
        assert chartspec.validate_spec(_spliced_spec(), _syn_manifest()) is None

    @pytest.mark.parametrize("bad_range", [[1900], [1900, 1950, 2000]])
    def test_time_range_pair_must_be_exactly_two_numbers(self, bad_range):
        spec = _line_spec()
        spec["time_range_ce"] = bad_range
        paths = _violation_paths(spec)
        assert any(path.startswith("time_range_ce") for path in paths), paths

    @pytest.mark.parametrize("bad_domain", [[0], [0, 60, 120]])
    def test_scale_domain_must_be_exactly_two_numbers(self, bad_domain):
        spec = _line_spec()
        spec["series"][0]["scale_domain"] = bad_domain
        paths = _violation_paths(spec)
        assert any("scale_domain" in path for path in paths), paths

    def test_empty_series_array_refused(self):
        spec = _line_spec()
        spec["series"] = []
        paths = _violation_paths(spec)
        assert any(path.startswith("series") for path in paths), paths

    def test_more_than_eight_series_refused(self):
        spec = _line_spec()
        template = spec["series"][0]
        spec["series"] = [{**copy.deepcopy(template), "id": f"w{index}"} for index in range(9)]
        paths = _violation_paths(spec)
        assert any(path.startswith("series") for path in paths), paths

    def test_splice_series_must_carry_exactly_two_members(self):
        spec = _spliced_spec()
        spec["series"][0]["splice_series"] = [
            "syn_widgets_old",
            "syn_widgets",
            "syn_widgets",
        ]
        paths = _violation_paths(spec)
        assert any("splice_series" in path for path in paths), paths

    def test_more_than_four_transforms_refused(self):
        spec = _line_spec()
        spec["series"][0]["transforms"] = [
            {"op": "rolling_mean", "window_years": 10} for _ in range(5)
        ]
        paths = _violation_paths(spec)
        assert any("transforms" in path for path in paths), paths
