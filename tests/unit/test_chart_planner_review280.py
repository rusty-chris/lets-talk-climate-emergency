"""Chart planner review finding #280 — RED.

Failing behavioural tests for the root cause the #276 Sonnet capability
probe isolated (PR #279, $0.091545): the #262 slim wire schema carries
``"series": {"type": "array"}`` — an ITEMLESS array — and the live
structured-outputs decoder, constrained by that grammar, never emits
OBJECT items inside it. Haiku always returned ``series: []`` (#276.2);
Sonnet always returned ``series: [integers]`` (``[0, 1, 2, 3]``, then
``[35]`` on the violations retry). Both models otherwise authored
everything the four Haiku sessions fought for — full-range spec,
``time_range_ce=[1880, 2025]``, hyphenated ids, honest subtitles — so
the defect is the request grammar, not (only) model tier.

1. **A budget-fitting ``items`` schema for ``series``.** The wire
   schema's ``series`` must carry a CLOSED object ``items`` schema whose
   typed fields are exactly what ``plan_chart_request``'s parse→validate
   path needs to survive ``charts.spec.validate_spec``: ``id``, ``label``
   and ``unit`` (the rich schema's required trio), ``dataset`` (the data
   source — without it every series is refused on the XOR rule), and
   ``transforms`` as an ITEMLESS typed array. ``series`` itself carries
   ``minItems: 1`` (inside the documented structured-outputs subset —
   ``tests._schema_subset`` admits 0/1) so the decoder cannot emit the
   recorded Haiku ``[]`` at all.

   FLAG (field set): the items object is dataset-series-only —
   ``required: [id, label, unit, dataset]``, properties adding only
   ``transforms``. The splice-capable alternative (``splice_series``,
   ``splice_pair_id``, ``overlap_policy``, the mandatory
   ``annotations.splice_point``/``resolution_note`` tree,
   ``rebaseline_to``) was measured at red-authoring time at 1798 B /
   44 nodes / 8 object nodes / 33 property keys / depth 13 — over the
   #262 budget on property_keys (33 > 32) AND depth (13 > 10, the depth
   of the live-REJECTED schema). The chosen minimal items measure
   1000 B / 25 nodes / 4 object nodes / 18 property keys / depth 9 —
   inside every axis with headroom. Consequence, flagged for the owner:
   with ``additionalProperties: false`` on the items object the decoder
   CANNOT author spliced series on this channel; splice charts need a
   follow-up design (the itemless status quo could not author them
   either — the decoder never emitted series objects at all).

2. **Typed ``time_range_ce`` items.** Same constraint class, so the
   two-year pair gets ``items: {"type": "number"}``.

   FLAG (decision): live evidence shows both tiers emitted correct year
   pairs under the itemless schema, so this is cheap insurance (one
   node, ~30 bytes) against the same decoder-prior failure mode — not
   the incident fix. ``number`` (not ``integer``) matches the rich
   schema's ``year_pair``.

3. **Model-conditional output budget.** PR #279 attempt 1: Sonnet 5's
   ADAPTIVE THINKING spends from the same ``max_tokens`` as the JSON
   payload, so the Haiku-sized 4160 truncated into invalid JSON at
   exactly 4160 output tokens. The structured channel offers no
   thinking exemption: ``rag.provider.build_anthropic_structured_request``
   maps only ``model``/``max_tokens``/``output_config.format``/
   ``messages``/``system`` (no ``thinking`` config), and the API's
   ``max_tokens`` is ALWAYS the hard cap on thinking + text — there is
   no parameter that exempts thinking from it (claude-api reference,
   2026-09: Sonnet 5 runs adaptive thinking BY DEFAULT when the field
   is omitted). So the budget must be model-conditional.

   FLAG (numbers): ``planner_max_tokens_for_model(model_id)`` — a pure
   module-level function (the #276 ``normalise_chart_id`` convention),
   consulted by ``build_planner_request`` — returns
   ``PLANNER_MAX_TOKENS`` (4160) for the ``claude-haiku-*`` family
   (thinking off when omitted) and ``PLANNER_MAX_TOKENS_CEILING``
   (8192) for every other family (claude-sonnet-5, claude-opus-*, and
   unknown futures fail SAFE to the larger, still ceiling-bounded
   value): the full cost-guard ceiling, i.e. a documented
   8192 - 4160 = 4032-token thinking allowance above the worst-case
   spec + envelope. The #271 invariants survive UNCHANGED: every
   budget stays within [PLANNER_MAX_TOKENS, PLANNER_MAX_TOKENS_CEILING],
   strictly above both recorded truncation ceilings (2048 in #270, 4160
   in #279), the default Haiku request is byte-identical, and the
   one-runaway-call cost guard (8192) is not raised.

   FLAG (rag/provider.py alternative, out of charts/ scope): Sonnet 5
   and claude-opus-* accept ``thinking: {"type": "disabled"}``, so the
   adapter COULD disable thinking on structured calls instead — but
   that is a seam-contract change in ``rag/provider.py`` (the payload
   ``config`` would need a thinking field through ``validate_request``)
   and is reported to the orchestrator rather than pinned here.

4. **The recorded failure shapes become rescuable.** A synthetic output
   in the recorded Sonnet shape — full-range spec, hyphenated id,
   honest subtitle, series POPULATED with objects (what the probe shows
   the model wants to emit once the grammar allows it) — lands a
   PlannedChart in one call; the recorded ``[integers]`` and ``[]``
   shapes still refuse CLEANLY (typed error naming ``series``, never a
   crash) with the #276-sharpened feedback naming the plottable
   catalogue dataset ids on the retry.

Unit tier: pure builders + FakeAdapter, no network, no API key.
Fixtures reuse the SYNTHETIC gold set of tests/unit/test_chart_planner.py;
the recorded shapes are verbatim from PR #279's body (no fixture file
was landed by the probe protocol).
"""

from __future__ import annotations

from typing import Any

import pytest

from charts import planner
from charts.planner import PlannedChart, PlannerSpecError
from rag.provider import FakeAdapter
from tests._schema_subset import (
    assert_schema_within_complexity_budget,
    assert_schema_within_structured_outputs_subset,
    measure_schema_complexity,
)
from tests.unit.test_chart_planner import (
    gold_catalogue,
    gold_manifest,
    spec_output,
    spec_temp_line,
)

# ---------------------------------------------------------------------------
# Recorded shapes (PR #279, verbatim)
# ---------------------------------------------------------------------------

#: Sonnet attempt 2 emitted ``series: [0, 1, 2, 3]``; the violations retry
#: emitted ``series: [35]``. Integer items are exactly what an itemless
#: array grammar lets the decoder produce.
RECORDED_SONNET_INTEGER_SERIES = [0, 1, 2, 3]
RECORDED_SONNET_RETRY_INTEGER_SERIES = [35]


def _recorded_sonnet_shape_spec(series: Any) -> dict[str, Any]:
    """SYNTHETIC FIXTURE shaped like the PR #279 Sonnet outputs: a genuine
    full-range spec with a hyphenated id and an honest contextualising
    subtitle — everything right except the ``series`` payload, which is
    whatever the grammar let the decoder emit. Ranges use the gold
    manifest's coverage (1880-2020) where the live record read 1880-2025."""
    spec = spec_temp_line()
    spec["chart_id"] = "global-temp-anomaly-full-record"
    spec["subtitle"] = (
        "Full instrumental record shown for context; recent years sit inside a long warming trend"
    )
    spec["series"] = series
    return spec


def _populated_sonnet_spec() -> dict[str, Any]:
    """The counterfactual the probe argues for: the same recorded Sonnet
    spec once the grammar permits object items — series populated with
    the gold single-series entry."""
    return _recorded_sonnet_shape_spec(spec_temp_line()["series"])


def _wire_spec_schema() -> dict[str, Any]:
    schema = planner.planner_output_schema()
    return schema["properties"]["spec"]


def _retry_system_head(adapter: FakeAdapter) -> str:
    calls = adapter.calls_to("structured")
    assert len(calls) == 2
    system = calls[1].payload["system"]
    marker = "Dataset catalogue (JSON):"
    assert marker in system
    return system.split(marker)[0]


# ---------------------------------------------------------------------------
# 1. The wire schema's series carries a closed, typed, budget-fitting items
#    object (RED)
# ---------------------------------------------------------------------------


class TestSeriesWireSchemaCarriesObjectItems:
    def test_series_carries_a_closed_object_items_schema(self):
        """The decoder must be steered TOWARD object items: ``series``
        carries an ``items`` schema that is a closed object (the
        structured-outputs channel requires ``additionalProperties:
        false`` + a ``required`` list on every object node)."""
        series = _wire_spec_schema()["properties"]["series"]
        assert series["type"] == "array"
        items = series.get("items")
        assert isinstance(items, dict), (
            "series must carry an items schema — the itemless array is the #280 "
            "defect: it constrained the live decoder against object items "
            "(Haiku emitted [], Sonnet emitted [integers]; PR #279)"
        )
        assert items.get("type") == "object"
        assert items.get("additionalProperties") is False
        assert isinstance(items.get("required"), list)

    def test_series_items_require_the_validate_spec_essentials(self):
        """The required set is exactly the fields a series needs to survive
        validate_spec: the rich schema's required trio (id, label, unit)
        plus the data source (dataset) — so the decoder EMITS them rather
        than being merely allowed to. FLAG: dataset-series-only; the
        splice-capable set measured over the #262 budget (module
        docstring)."""
        items = _wire_spec_schema()["properties"]["series"]["items"]
        assert set(items["required"]) == {"id", "label", "unit", "dataset"}

    @pytest.mark.parametrize(
        ("field", "expected_type"),
        [
            ("id", "string"),
            ("label", "string"),
            ("unit", "string"),
            ("dataset", "string"),
            ("transforms", "array"),
        ],
    )
    def test_series_items_field_types(self, field, expected_type):
        """Each essential field is TYPED on the wire (types only — no
        enums/patterns/bounds, the #262 slimming rule; validate_spec keeps
        enforcing the vocabulary). ``transforms`` rides as a typed array —
        itemless is acceptable (FLAGGED residual risk: transform OBJECTS
        face the same decoder prior one level down, but transforms are
        optional and the validator feedback loop covers them)."""
        items = _wire_spec_schema()["properties"]["series"]["items"]
        assert items["properties"][field]["type"] == expected_type

    def test_series_forbids_the_recorded_empty_list_at_the_decoder(self):
        """``minItems: 1`` on ``series`` — inside the documented
        structured-outputs subset (0/1 only) — makes the recorded Haiku
        ``series: []`` unemittable at the decoder instead of a burned
        retry."""
        series = _wire_spec_schema()["properties"]["series"]
        assert series.get("minItems") == 1

    def test_time_range_ce_items_are_typed_numbers(self):
        """The [start, end] pair gets number-typed items — the same
        itemless-array constraint class, closed cheaply (FLAG: insurance,
        not the incident fix — both tiers emitted correct year pairs
        live)."""
        time_range = _wire_spec_schema()["properties"]["time_range_ce"]
        assert time_range["type"] == "array"
        assert time_range.get("items", {}).get("type") == "number"

    def test_whole_schema_stays_within_the_262_budget_and_subset(self):
        """THE HARD CONSTRAINT: the enriched schema must still fit the
        empirical #262 complexity budget (2048 B / 48 nodes / 10 objects /
        32 props / depth 10 / 24 enum members) and the supported-subset
        vocabulary — the reason the items object is minimal. Green today
        and it must SURVIVE the fix; measured so a failure names the axis."""
        schema = planner.planner_output_schema()
        assert_schema_within_structured_outputs_subset(schema, name="planner_output_schema#280")
        assert_schema_within_complexity_budget(schema, name="planner_output_schema#280")

    def test_built_requests_fresh_and_retry_carry_the_enriched_schema(self):
        """The schema that actually rides the wire (fresh AND violations
        retry) is the enriched one, still budget-fitting."""
        catalogue = gold_catalogue()
        fresh = planner.build_planner_request("Show me the cooling since 2016", catalogue)
        retry = planner.build_planner_request(
            "Show me the cooling since 2016",
            catalogue,
            violations=("series: schema violation (minItems): should be non-empty",),
        )
        for request in (fresh, retry):
            series = request["schema"]["properties"]["spec"]["properties"]["series"]
            assert isinstance(series.get("items"), dict)
            assert_schema_within_complexity_budget(request["schema"], name="built request#280")


# ---------------------------------------------------------------------------
# 2. Model-conditional max_tokens for adaptive-thinking models (RED)
# ---------------------------------------------------------------------------


class TestModelConditionalPlannerBudget:
    """PR #279 attempt 1: 4160 output tokens billed, invalid JSON —
    Sonnet 5's adaptive thinking (on by default; no thinking-exemption
    exists on the structured channel, and the adapter sends no thinking
    config at all) consumed the Haiku-sized budget. The budget becomes a
    pure function of the model family; every #271 invariant is preserved
    (module docstring FLAG for the numbers)."""

    def test_haiku_family_budget_is_unchanged(self):
        """The default tier keeps the #271 arithmetic to the token —
        thinking is off when the field is omitted on claude-haiku-4-5, so
        nothing shares its budget."""
        assert planner.planner_max_tokens_for_model("claude-haiku-4-5") == (
            planner.PLANNER_MAX_TOKENS
        )
        assert planner.planner_max_tokens_for_model("claude-haiku-4-5") == 4160

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-6"],
        ids=["sonnet-5", "opus-4-8", "opus-4-6"],
    )
    def test_adaptive_thinking_families_get_the_ceiling_budget(self, model):
        """Families that run adaptive thinking by default get the full
        cost-guard ceiling: worst-case spec + envelope (4160) plus a 4032
        thinking allowance. Strictly above the recorded 4160 truncation."""
        budget = planner.planner_max_tokens_for_model(model)
        assert budget == planner.PLANNER_MAX_TOKENS_CEILING
        assert budget == 8192
        assert budget > 4160  # the PR #279 attempt-1 truncation ceiling

    def test_unknown_model_families_fail_safe_to_the_adaptive_budget(self):
        """Every current non-Haiku family runs adaptive thinking by
        default, so an unrecognised future model id takes the LARGER,
        still ceiling-bounded budget — truncation is the failure being
        bought out; the ceiling caps a runaway either way."""
        assert planner.planner_max_tokens_for_model("claude-newtier-9") == (
            planner.PLANNER_MAX_TOKENS_CEILING
        )

    @pytest.mark.parametrize(
        "model",
        ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8", "claude-newtier-9"],
    )
    def test_271_floor_and_ceiling_invariants_hold_for_every_family(self, model):
        """The #271 pins stay coherent without amendment: every budget lies
        in [PLANNER_MAX_TOKENS, PLANNER_MAX_TOKENS_CEILING] and strictly
        above the #270 2048-token truncation. The cost guard itself is not
        raised."""
        budget = planner.planner_max_tokens_for_model(model)
        assert planner.PLANNER_MAX_TOKENS <= budget <= planner.PLANNER_MAX_TOKENS_CEILING
        assert budget > 2048
        assert planner.PLANNER_MAX_TOKENS_CEILING == 8192

    def test_build_planner_request_budget_follows_the_planner_model(self, monkeypatch):
        """The budget rides the request: overriding the planner model at
        runtime (exactly what the #279 probe did) must resize max_tokens —
        the probe's attempt-1 failure was this request going out with the
        Sonnet model and the Haiku budget. Fresh AND violations retry."""
        catalogue = gold_catalogue()
        monkeypatch.setattr(planner, "PLANNER_MODEL", "claude-sonnet-5")
        fresh = planner.build_planner_request("Show me the cooling since 2016", catalogue)
        retry = planner.build_planner_request(
            "Show me the cooling since 2016",
            catalogue,
            violations=("series[0]: is not of type 'object'",),
        )
        for request in (fresh, retry):
            assert request["config"]["model"] == "claude-sonnet-5"
            assert request["config"]["max_tokens"] == planner.PLANNER_MAX_TOKENS_CEILING

    def test_default_request_stays_byte_identical_on_the_budget_axis(self):
        """No behaviour change for the default tier: the un-overridden
        request still carries claude-haiku-4-5 at 4160 — the #271 budget
        tests keep passing on the same numbers."""
        request = planner.build_planner_request("plot temperature", gold_catalogue())
        assert request["config"]["model"] == "claude-haiku-4-5"
        assert request["config"]["max_tokens"] == 4160


# ---------------------------------------------------------------------------
# 3. The recorded failure shapes through the parse→validate path
# ---------------------------------------------------------------------------


class TestRecordedShapesBecomeRescuable:
    """Green guards on the surrounding contract: once the grammar admits
    object items, the shape the probe shows the models WANT to emit lands
    a PlannedChart, and the shapes the old grammar forced still refuse
    cleanly with the #276-sharpened feedback."""

    def test_populated_sonnet_shape_lands_a_planned_chart_in_one_call(self):
        """The counterfactual pin: the recorded Sonnet spec with series
        populated by objects (full range, hyphenated id, honest subtitle)
        passes parse→validate to a PlannedChart on the fresh call — no
        retry burned."""
        adapter = FakeAdapter(structured_results=[spec_output(_populated_sonnet_spec())])
        result = planner.plan_chart_request(
            adapter, "Show me the cooling since 2016", gold_manifest()
        )
        assert isinstance(result, PlannedChart)
        assert len(adapter.calls_to("structured")) == 1
        assert result.spec == _populated_sonnet_spec()

    def test_recorded_integer_series_refuses_cleanly_and_feedback_names_datasets(self):
        """The recorded Sonnet shape (integer series items) is refused by
        validate_spec naming ``series`` — never a bare TypeError — and the
        single retry's INSTRUCTION section names the plottable catalogue
        dataset ids (#276 remedy 3 applies to invalid, not just empty,
        series); a populated retry then succeeds."""
        good = spec_temp_line()
        adapter = FakeAdapter(
            structured_results=[
                spec_output(_recorded_sonnet_shape_spec(RECORDED_SONNET_INTEGER_SERIES)),
                spec_output(good),
            ]
        )
        result = planner.plan_chart_request(
            adapter, "Show me the cooling since 2016", gold_manifest()
        )
        assert isinstance(result, PlannedChart)
        assert result.spec == good
        retry_head = _retry_system_head(adapter)
        for ds_id in gold_catalogue()["datasets"]:
            assert ds_id in retry_head

    def test_recorded_integer_series_twice_raises_typed_error_naming_series(self):
        """Both recorded Sonnet payloads in sequence ([0,1,2,3] then [35]):
        exactly two calls, then PlannerSpecError whose violations name the
        series path — the honest-refusal contract survives the schema fix."""
        adapter = FakeAdapter(
            structured_results=[
                spec_output(_recorded_sonnet_shape_spec(RECORDED_SONNET_INTEGER_SERIES)),
                spec_output(_recorded_sonnet_shape_spec(RECORDED_SONNET_RETRY_INTEGER_SERIES)),
            ]
        )
        with pytest.raises(PlannerSpecError) as excinfo:
            planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
        assert len(adapter.calls_to("structured")) == 2
        assert any("series" in violation for violation in excinfo.value.violations)

    def test_recorded_empty_series_twice_still_refuses_cleanly(self):
        """The recorded Haiku shape (series: []) twice: two calls, typed
        error naming series — the #276 contract is not loosened by the
        wire-schema change (defence in depth: the schema is steering, the
        validator is enforcement)."""
        empty = _recorded_sonnet_shape_spec([])
        adapter = FakeAdapter(structured_results=[spec_output(empty), spec_output(dict(empty))])
        with pytest.raises(PlannerSpecError) as excinfo:
            planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
        assert len(adapter.calls_to("structured")) == 2
        assert any("series" in violation for violation in excinfo.value.violations)


# ---------------------------------------------------------------------------
# Complexity documentation (assertion-backed, so the report numbers can
# never drift from what the suite pinned)
# ---------------------------------------------------------------------------


def test_enriched_schema_complexity_is_documented():
    """The enriched wire schema's measured complexity stays comfortably
    inside every #262 axis — pinned loosely (upper bounds only, not exact
    counts) so ordinary implementation choices don't thrash this test
    while gross regrowth toward the rejected 3597 B / 88-node shape
    fails loudly."""
    metrics = measure_schema_complexity(planner.planner_output_schema())
    assert metrics["bytes"] <= 1400
    assert metrics["nodes"] <= 32
    assert metrics["object_nodes"] <= 6
    assert metrics["property_keys"] <= 24
    assert metrics["max_depth"] <= 10
    assert metrics["enum_members"] <= 2
