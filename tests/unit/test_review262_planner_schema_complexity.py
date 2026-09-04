"""Review finding #262 — RED: the planner request schema fits the live
structured-outputs complexity budget, with dropped enforcement re-homed.

The #162 recording session drew ``400 invalid_request_error: "Schema is
too complex"`` (unbilled, no fixture) on ``plan_chart_request``'s
structured request — ``charts/planner.py::planner_output_schema()``. This
is DISTINCT from #209's subset-vocabulary fix (already merged and green):
the planner schema is inside the allowed vocabulary but exceeds the
API's undocumented complexity limit. The citation validator's schema was
ACCEPTED live the same session, so it is the known-good complexity
reference point.

Offline comparison (compact-serialized, 2026-09-03):

===================  ==========  ================  ===========
axis                 planner     validator (45p)   classifier
                     (rejected)  (accepted)        (accepted)
===================  ==========  ================  ===========
bytes                3597        284               571
mapping nodes        88          7                 12
object nodes         15          2                 1
property keys        57          3                 6
max depth            13          7                 7
enum members         16          0                 13
if/then blocks       2           0                 0
===================  ==========  ================  ===========

Three layers, mirroring the ratified #203/#209 re-homing remedy:

1. **The shared complexity budget** (``tests._schema_subset``, new #262
   helper) applied to EVERY registered structured-request builder via
   the #209 enumeration sweep — RED for the planner, green for the
   validator and classifier. FLAGGED: the budget constants are offline
   estimates bounded by the one rejected vs two accepted schemas; the
   green-phase live probe confirms they sit under the real limit.
2. **Prompt-side re-homing**: the closed vocabularies the slimmed
   request schema can no longer carry as enums/consts (chart types,
   transform ops, overlap policies, the CE calendar) move into the
   planner's system instructions — the #203 "exactly N verdicts"
   remedy applied to the vocabulary axis. RED today.
3. **Parser-side re-homing**: ``charts.spec.validate_spec`` (and
   ``_parse_planner_outcome`` for the envelope conditionals) must keep
   refusing off-vocabulary output WITHOUT the request schema's help —
   green today via ``chartspec_schema``'s enums/patterns/consts, pinned
   here so stripping the request-side enforcement without keeping the
   validator's fails these tests, not production.

Unit tier: pure builders and validators, no network, no API key.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from charts import planner
from charts import spec as chartspec
from charts.planner import PlannerSpecError
from charts.spec import CHART_TYPES, OVERLAP_POLICIES, TRANSFORM_OPS
from rag.provider import FakeAdapter
from tests._schema_subset import (
    MAX_ENUM_MEMBERS,
    MAX_NESTING_DEPTH,
    MAX_OBJECT_NODES,
    MAX_PROPERTY_KEYS,
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_NODES,
    assert_schema_within_complexity_budget,
    measure_schema_complexity,
)
from tests.unit.test_review209_structured_schemas import (
    _all_requests,
    _line_spec,
    _spliced_spec,
    _syn_manifest,
)

# ---------------------------------------------------------------------------
# Layer 1: the complexity budget over EVERY registered structured builder
# ---------------------------------------------------------------------------


def _over_deep_schema() -> dict[str, Any]:
    """An array-of-array chain nested past MAX_NESTING_DEPTH."""
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(MAX_NESTING_DEPTH):
        schema = {"type": "array", "items": schema}
    return schema


class TestEveryStructuredSchemaFitsTheComplexityBudget:
    @pytest.mark.parametrize(
        ("label", "request_payload"),
        _all_requests(),
        ids=[label for label, _ in _all_requests()],
    )
    def test_structured_request_schema_fits_the_complexity_budget(self, label, request_payload):
        """Finding #262: each built request's ``schema`` fits the shared
        complexity budget the live API evidence bounds. RED today for the
        chart planner (both registered cases), whose schema embeds the
        full ``chartspec_schema`` vocabulary; green for the citation
        validator (live-accepted) and the query classifier."""
        assert "schema" in request_payload, f"{label}: request has no schema"
        assert_schema_within_complexity_budget(request_payload["schema"], name=label)

    def test_known_good_schemas_sit_comfortably_under_the_budget(self):
        """The budget must not merely admit the live-accepted schemas —
        it must leave them real headroom, so ordinary growth of a
        known-good builder does not trip the guard before the guard's
        own constants are revisited. 2x headroom on the byte axis."""
        accepted = [
            (label, payload)
            for label, payload in _all_requests()
            if not label.startswith("build_planner_request")
        ]
        assert accepted, "expected at least one non-planner registered builder"
        for label, payload in accepted:
            metrics = measure_schema_complexity(payload["schema"])
            assert metrics["bytes"] * 2 <= MAX_SCHEMA_BYTES, (
                f"{label}: {metrics['bytes']} bytes leaves less than 2x headroom "
                f"under MAX_SCHEMA_BYTES={MAX_SCHEMA_BYTES}"
            )

    @pytest.mark.parametrize(
        ("axis", "over_budget_schema"),
        [
            (
                "bytes",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {
                        "note": {"type": "string", "description": "x" * MAX_SCHEMA_BYTES}
                    },
                },
            ),
            (
                "nodes",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {
                        f"p{i}": {"type": "string"} for i in range(MAX_SCHEMA_NODES + 1)
                    },
                },
            ),
            (
                "object_nodes",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {
                        f"o{i}": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [],
                            "properties": {},
                        }
                        for i in range(MAX_OBJECT_NODES + 1)
                    },
                },
            ),
            (
                "property_keys",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {
                        f"k{i}": {"type": "boolean"} for i in range(MAX_PROPERTY_KEYS + 1)
                    },
                },
            ),
            ("max_depth", _over_deep_schema()),
            (
                "enum_members",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [f"v{i}" for i in range(MAX_ENUM_MEMBERS + 1)],
                        }
                    },
                },
            ),
        ],
    )
    def test_budget_helper_actually_flags_each_axis(self, axis, over_budget_schema):
        """Self-check on the new helper: a watered-down budget that stops
        measuring any axis would silently disarm the #262 guard."""
        with pytest.raises(AssertionError, match=axis):
            assert_schema_within_complexity_budget(over_budget_schema, name="self-check")

    @pytest.mark.parametrize("conditional_key", ["if", "then", "else", "not", "oneOf"])
    def test_budget_helper_bans_undocumented_conditional_keys(self, conditional_key):
        """``if``/``then``/``else``/``not``/``oneOf`` are absent from the
        documented structured-outputs supported subset and were never
        verified live (the planner — their only user — was rejected).
        The envelope conditional-requireds they carried are re-homed to
        ``_parse_planner_outcome`` (layer 3 below)."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [],
            "properties": {"field": {"type": "string"}},
            conditional_key: [] if conditional_key in ("oneOf",) else {},
        }
        with pytest.raises(AssertionError, match=re.escape(repr(conditional_key))):
            assert_schema_within_complexity_budget(schema, name="self-check")


# ---------------------------------------------------------------------------
# Layer 2: prompt-side re-homing — the closed vocabularies ride the prompt
# ---------------------------------------------------------------------------


def _system_text() -> str:
    """The planner system channel with an empty catalogue, so every
    anchor asserted below is authored instruction text, never an accident
    of catalogue content (the #209 convention)."""
    request = planner.build_planner_request("plot widgets", {"datasets": {}, "splice_pairs": []})
    return request["system"]


class TestPlannerPromptCarriesReHomedVocabulary:
    """The enum/const steering the slimmed request schema can no longer
    express (the #262 budget forbids carrying the full chartspec
    vocabulary on the wire) must ride the system prompt instead — the
    #203 'exactly N verdicts' remedy applied to the vocabulary axis.
    ``validate_spec`` keeps ENFORCING every vocabulary; the prompt's job
    is steering, so the model can still author legal specs once the
    decoder no longer sees the enums. RED today."""

    @pytest.mark.parametrize("chart_type", sorted(CHART_TYPES))
    def test_system_instructions_name_every_chart_type(self, chart_type):
        assert re.search(rf"\b{re.escape(chart_type)}\b", _system_text()), (
            f"the system instructions must name chart type {chart_type!r} — the "
            "chart_type enum is re-homed from the request schema to the prompt "
            "(finding #262; validate_spec still refuses anything off-vocabulary)"
        )

    @pytest.mark.parametrize("op", sorted(TRANSFORM_OPS))
    def test_system_instructions_name_every_transform_op(self, op):
        assert re.search(rf"\b{re.escape(op)}\b", _system_text()), (
            f"the system instructions must name transform op {op!r} — the "
            "transforms[*].op enum is re-homed from the request schema to the "
            "prompt (finding #262)"
        )

    @pytest.mark.parametrize("policy", sorted(OVERLAP_POLICIES))
    def test_system_instructions_name_every_overlap_policy(self, policy):
        assert re.search(rf"\b{re.escape(policy)}\b", _system_text()), (
            f"the system instructions must name overlap policy {policy!r} — the "
            "overlap_policy enum is re-homed from the request schema to the "
            "prompt (finding #262; validate_spec still pins the value to the "
            "manifest pair's policy)"
        )

    def test_system_instructions_state_the_ce_calendar(self):
        assert re.search(r"calendar[^\n]*\bCE\b", _system_text()), (
            "the system instructions must state that time_axis.calendar is "
            'always "CE" — the const is re-homed from the request schema to '
            "the prompt (finding #262; validate_spec refuses any other value)"
        )

    def test_shape_bound_lines_survive(self):
        """The #209 re-homed counting lines must not be lost while the
        vocabulary lines are added (regression guard on the same text)."""
        system = _system_text()
        assert re.search(r"exactly two numbers", system, re.IGNORECASE)
        assert re.search(r"at most (8|eight) series", system, re.IGNORECASE)
        assert re.search(r"at most (4|four) transforms", system, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Layer 3: parser-side re-homing — refusal stays in code once the wire
# schema stops enforcing the vocabulary (green today; must survive the fix)
# ---------------------------------------------------------------------------


def _violation_paths(spec: dict[str, Any]) -> list[str]:
    with pytest.raises(chartspec.ChartSpecError) as excinfo:
        chartspec.validate_spec(spec, _syn_manifest())
    return [violation.path for violation in excinfo.value.violations]


class TestValidateSpecKeepsVocabularyInvariantsWithoutSchemaEnforcement:
    """Every enum/pattern/const the #262 fix strips from the REQUEST
    schema stays enforced by ``validate_spec`` (via the rich
    ``chartspec_schema``, which keeps every bound). Green today — these
    pins force the implementer to keep the rich validator schema intact
    while slimming the wire copy, exactly the #209
    ``TestValidateSpecKeepsCountingInvariants`` pattern."""

    def test_control_specs_are_valid(self):
        assert chartspec.validate_spec(_line_spec(), _syn_manifest()) is None
        assert chartspec.validate_spec(_spliced_spec(), _syn_manifest()) is None

    def test_off_vocabulary_chart_type_refused(self):
        spec = _line_spec()
        spec["chart_type"] = "pie"
        assert any(path.startswith("chart_type") for path in _violation_paths(spec))

    def test_off_vocabulary_transform_op_refused(self):
        spec = _line_spec()
        spec["series"][0]["transforms"] = [{"op": "savitzky_golay"}]
        paths = _violation_paths(spec)
        assert any("transforms[0]" in path for path in paths), paths

    def test_off_vocabulary_overlap_policy_refused(self):
        spec = _spliced_spec()
        spec["series"][0]["overlap_policy"] = "hide_the_decline"
        paths = _violation_paths(spec)
        assert any("overlap_policy" in path for path in paths), paths

    def test_non_ce_calendar_refused(self):
        spec = _spliced_spec()
        spec["time_axis"]["calendar"] = "BP"
        paths = _violation_paths(spec)
        assert any(path.startswith("time_axis.calendar") for path in paths), paths

    def test_malformed_chart_id_refused(self):
        spec = _line_spec()
        spec["chart_id"] = "Syn Chart!"  # violates ^[a-z0-9][a-z0-9-]{0,63}$
        assert any(path.startswith("chart_id") for path in _violation_paths(spec))

    def test_malformed_spec_version_refused(self):
        spec = _line_spec()
        spec["spec_version"] = "one-point-oh"
        assert any(path.startswith("spec_version") for path in _violation_paths(spec))

    def test_unknown_property_refused(self):
        spec = _line_spec()
        spec["caption"] = "authored caption text"  # amendment 9: no caption, ever
        assert any(path.startswith("caption") for path in _violation_paths(spec))

    def test_over_long_title_refused(self):
        spec = _line_spec()
        spec["title"] = "x" * 201  # short_text maxLength 200 — off the wire schema
        assert any(path.startswith("title") for path in _violation_paths(spec))


class TestParserKeepsEnvelopeConditionalsWithoutSchemaIfThen:
    """The request schema's ``allOf``/``if``/``then`` envelope
    conditionals (outcome "spec" requires ``spec``; outcome
    "unavailable" requires ``requested_data``) are banned from the wire
    by the #262 budget. ``_parse_planner_outcome`` already enforces both
    — pinned here so they cannot be lost when the schema sheds its
    conditionals. Green today."""

    def test_spec_outcome_without_spec_is_malformed(self):
        with pytest.raises(planner._MalformedPlannerOutput):
            planner._parse_planner_outcome({"outcome": "spec"})

    def test_unavailable_outcome_without_requested_data_is_malformed(self):
        with pytest.raises(planner._MalformedPlannerOutput):
            planner._parse_planner_outcome({"outcome": "unavailable"})

    def test_unknown_outcome_is_malformed(self):
        with pytest.raises(planner._MalformedPlannerOutput):
            planner._parse_planner_outcome({"outcome": "chart"})

    def test_non_mapping_spec_is_malformed(self):
        with pytest.raises(planner._MalformedPlannerOutput):
            planner._parse_planner_outcome({"outcome": "spec", "spec": "[not-a-mapping]"})


# ---------------------------------------------------------------------------
# Layer 3b: end-to-end honesty — off-schema model output through the
# planner entry point refuses with a violation-naming error, never a
# crash, never silent acceptance (green today; must survive the fix)
# ---------------------------------------------------------------------------


def _licensed(ds_id: str) -> dict[str, Any]:
    """SYNTHETIC FIXTURE: the §2.1 licensing fields validate_dataset
    requires on every raw-mapping manifest entry (finding #161)."""
    return {
        "licence": "CC BY 4.0",
        "licence_evidence": "SYNTHETIC FIXTURE - licence asserted for these tests only",
        "url": f"https://example.test/{ds_id}.csv",
        "attribution_text": "Synthetic Data Consortium",
        "sha256": "0" * 64,
        "retrieved_at": "2026-08-16",
        "human_signoff": {
            "who": "test-fixture",
            "date": "2026-08-16",
            "note": "synthetic fixture entry",
        },
    }


def _planner_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: the #209 widgets manifest plus the licensing
    fields the planner's entry-point validation demands (finding #161)."""
    manifest = _syn_manifest()
    for ds_id, entry in manifest["datasets"].items():
        entry.update(_licensed(ds_id))
    return manifest


class TestSlimmerSchemaCannotSmuggleOffVocabularyOutputPastThePlanner:
    """With the wire schema slimmed, the decoder no longer blocks
    off-vocabulary specs — the model CAN now emit them. The planner must
    refuse honestly: retry once with violations fed back, then raise
    ``PlannerSpecError`` whose violations NAME the offending path.
    Never a bare TypeError/KeyError, never a returned PlannedChart."""

    @pytest.mark.parametrize(
        ("mutate", "expected_fragment"),
        [
            (lambda spec: spec.__setitem__("chart_type", "pie"), "chart_type"),
            (
                lambda spec: spec["series"][0].__setitem__("transforms", [{"op": "smooth_hard"}]),
                "transforms",
            ),
            # An UNRESCUABLE chart_id (off-alphabet only -> "" under
            # charts.planner.normalise_chart_id): a cosmetically-fixable id
            # such as "Bad Id!" is now normalised to a legal slug before
            # validate_spec (review finding #276, ratified), so the
            # off-schema refusal pin needs an id that genuinely cannot be
            # rescued — normalise_chart_id("!!!") == "" and validate_spec
            # still refuses "" on the chart_id pattern.
            (lambda spec: spec.__setitem__("chart_id", "!!!"), "chart_id"),
            (lambda spec: spec.__setitem__("time_range_ce", [1900, 1950, 2000]), "time_range_ce"),
        ],
        ids=["off-vocab-chart-type", "off-vocab-transform-op", "bad-chart-id", "three-item-range"],
    )
    def test_off_schema_spec_refused_with_violation_naming_the_path(
        self, mutate, expected_fragment
    ):
        bad = _line_spec()
        mutate(bad)
        output = {"outcome": "spec", "spec": bad}
        adapter = FakeAdapter(structured_results=[output, output])
        with pytest.raises(PlannerSpecError) as excinfo:
            planner.plan_chart_request(adapter, "plot widgets", _planner_manifest())
        assert len(adapter.calls_to("structured")) == 2  # one retry, never a third
        assert any(expected_fragment in violation for violation in excinfo.value.violations), (
            expected_fragment,
            excinfo.value.violations,
        )

    def test_envelope_missing_spec_refused_as_malformed_not_crash(self):
        output = {"outcome": "spec"}  # the if/then the wire schema used to carry
        adapter = FakeAdapter(structured_results=[output, output])
        with pytest.raises(PlannerSpecError):
            planner.plan_chart_request(adapter, "plot widgets", _planner_manifest())
        assert len(adapter.calls_to("structured")) == 2
