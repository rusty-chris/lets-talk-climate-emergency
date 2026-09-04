"""Chart planner review finding #276 — RED.

Failing behavioural tests for the three mechanical defects of the third
#162 recording session (PR #275, 2026-09-04): every live attempt produced
a genuinely full-range spec (the #274 cherry-pick steering worked), but
all three were validator-refused on the SAME two defects — an underscored
``chart_id`` (``cooling_since_2016`` vs ``^[a-z0-9][a-z0-9-]{0,63}$``)
and an empty ``series: []`` — unshaken by the violations-feedback retry.

1. **Deterministic ``chart_id`` normalisation before validate_spec.**
   ``chart_id`` is cosmetic identity, not chart semantics — refusing a
   full-range spec over underscores burns the whole retry budget on a
   defect code can fix in microseconds. Pinned algorithm (the issue's
   ratifiable remedy, in this order): lowercase; underscores and spaces
   become hyphens; characters outside ``[a-z0-9-]`` are stripped; repeat
   hyphens collapse; leading/trailing hyphens are stripped; clamp to 64
   chars. Applied to BOTH the fresh and the retry outcome, before
   ``validate_spec`` ever sees the spec, so the recorded
   ``cooling_since_2016`` spec is ACCEPTED (no retry burned). Genuinely
   unrescuable ids (empty / off-alphabet-only) still refuse. The
   normalised id — never the raw one — lands in ``PlannedChart.spec``
   and therefore in the permalink hash input.

   FLAG (pure API): pinned as ``charts.planner.normalise_chart_id``, a
   pure function returning ``""`` for unrescuable input (the caller then
   lets ``validate_spec`` refuse) — mirroring how #271 pinned
   ``is_degenerate_output_text`` as a module-level pure predicate.

   FLAG (hash input): ``charts.spec.spec_hash`` hashes the spec mapping
   verbatim (``chart_id`` included), and the permalink store
   (``service/chart_store.py``) hashes ``PlannedChart.spec`` as returned
   — so raw-vs-normalised is NOT ambiguous once normalisation happens
   inside the planner: downstream only ever sees the normalised spec.
   Pinned at the planner boundary: the planned spec's hash equals the
   hash of the same spec authored with the already-normalised id (two
   cosmetic variants converge on ONE permalink identity).

2. **Worked spec skeleton in the prompt.** Root cause hypothesis (#276):
   the #262 slim wire schema correctly shed the interior vocabulary, and
   the re-homed prompt anchors describe VOCABULARY but show no STRUCTURE
   — the model has never seen a filled ``series`` entry. Pinned: the
   system prompt's instruction section (fresh AND retry) carries one
   compact complete example — a hyphenated ``chart_id`` literal, ONE
   filled series entry (``dataset`` naming a catalogue id, ``label``,
   the ``transforms``/``op`` syntax) and ``time_range_ce`` — within the
   #165 per-line cap and a ≲200-token addition.

   FLAG (token measure): the repo's token helpers point the wrong way
   for a cap — ``rag.generation.estimate_tokens_lower_bound``
   deliberately UNDER-estimates (a lenient cap) and
   ``ingestion.chunk.estimate_tokens`` is a whitespace word count that
   under-counts JSON. Pinned instead as chars/4 over the addition:
   instruction-section growth over the measured pre-#276 baseline
   (2315 chars on the gold catalogue) is capped at 200 tokens * 4
   chars/token = 800 chars.

   FLAG (skeleton dataset id): pinned that the example's ``dataset``
   value is a real catalogue id (the builder already holds the
   catalogue, and a made-up id in the one worked example would teach the
   model to invent ids — the exact ADR-021 failure the prompt forbids).

3. **Sharpened empty-series feedback.** When ``validate_spec`` refuses
   on the series ``minItems`` violation, the retry's violations-feedback
   section must NAME the catalogue dataset ids the request could plot,
   so the retry holds the material to comply (live, the #165-redacted
   ``series: <redacted> should be non-empty`` line gave the model
   nothing to fill the list with).

   FLAG (coverage filtering): the planner's catalogue filtering is
   licence/pack-based only (#117/#164) — there is NO per-request
   coverage-relevance distinction in the planner today (the only
   per-request ranking, ``nearest_available_datasets``, is lexical and
   scores 0 on the live request, degrading to alphabetical order). So
   these tests pin that ALL catalogue dataset ids are named (the
   catalogue is small by design, <= 8 pack datasets) and do NOT pin the
   exclusion of coverage-unserviceable datasets — no such filter exists
   to pin against.

Everything reuses the SYNTHETIC gold fixtures of
tests/unit/test_chart_planner.py; the recorded raw ids
(``cooling_since_2016``, ``temp_cooling_since_2016``) are verbatim from
PR #275's body.
"""

from __future__ import annotations

import re

import pytest

from charts import planner
from charts import spec as chartspec
from charts.planner import PlannedChart, PlannerSpecError
from rag.provider import FakeAdapter
from tests.unit.test_chart_planner import (
    cherry_pick_domain_spec,
    gold_catalogue,
    gold_manifest,
    spec_output,
    spec_temp_line,
)

#: The chart_id slug rule of charts/spec.py's chartspec_schema — every
#: non-empty normalisation result must satisfy it (that is the point of
#: normalising).
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: The system prompt's instruction/skeleton/feedback section ends where
#: the catalogue payload begins (build_planner_request appends the
#: catalogue JSON last). Anchors asserted on the HEAD cannot be satisfied
#: by the catalogue dump itself.
_CATALOGUE_MARKER = "Dataset catalogue (JSON):"


def _system_head(system: str) -> str:
    """The instruction section of a planner system prompt — everything
    before the appended catalogue JSON."""
    assert _CATALOGUE_MARKER in system
    return system.split(_CATALOGUE_MARKER)[0]


def _retry_system_head(adapter: FakeAdapter) -> str:
    calls = adapter.calls_to("structured")
    assert len(calls) == 2
    return _system_head(calls[1].payload["system"])


# ---------------------------------------------------------------------------
# 1a. The pure normaliser: the pinned table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The recorded live ids, verbatim from PR #275.
        pytest.param("cooling_since_2016", "cooling-since-2016", id="recorded-underscores"),
        pytest.param(
            "temp_cooling_since_2016", "temp-cooling-since-2016", id="recorded-underscores-2"
        ),
        # Case is cosmetic.
        pytest.param("Cooling_Since_2016", "cooling-since-2016", id="mixed-case"),
        pytest.param("COOLING-SINCE-2016", "cooling-since-2016", id="upper-case-hyphenated"),
        # Spaces become hyphens like underscores do.
        pytest.param("cooling since 2016", "cooling-since-2016", id="spaces"),
        pytest.param("cooling  since __ 2016", "cooling-since-2016", id="mixed-separator-runs"),
        # Unicode/punctuation junk outside [a-z0-9-] is stripped.
        pytest.param("cooling™ since 2016!", "cooling-since-2016", id="unicode-junk"),
        pytest.param('"cooling_since_2016"', "cooling-since-2016", id="quoted-id"),
        # Repeat hyphens collapse; leading/trailing hyphens are stripped.
        pytest.param("--cooling--since--2016--", "cooling-since-2016", id="hyphen-runs"),
        pytest.param("_cooling_since_2016_", "cooling-since-2016", id="edge-underscores"),
        # Length clamps to the slug rule's 64 chars (clamp is the LAST
        # step, so a clamp-created trailing hyphen survives — the slug
        # pattern permits it).
        pytest.param("a" * 80, "a" * 64, id="overlong-clamped"),
        pytest.param("a-" * 40, "a-" * 32, id="overlong-hyphenated-clamped"),
        # Already-normal ids pass through untouched (identity).
        pytest.param("gold-temp-line", "gold-temp-line", id="identity"),
        pytest.param("2016", "2016", id="digits-only-identity"),
        # Genuinely unrescuable ids normalise to "" — the caller must
        # then still refuse (validate_spec's pattern rejects "").
        pytest.param("", "", id="empty"),
        pytest.param("___", "", id="underscores-only"),
        pytest.param("!!!***", "", id="punctuation-only"),
        pytest.param("--_ _--", "", id="separators-only"),
    ],
)
def test_normalise_chart_id_table(raw, expected):
    """The deterministic normalisation table (#276 remedy 1): lowercase;
    underscores/spaces -> hyphens; strip outside [a-z0-9-]; collapse
    hyphen runs; strip edge hyphens; clamp to 64."""
    assert planner.normalise_chart_id(raw) == expected


def test_normalise_chart_id_is_idempotent_and_slug_shaped():
    """Normalising twice never changes the answer, and every non-empty
    result satisfies the validator's slug pattern — the whole point of
    normalising is that validate_spec then has nothing to refuse."""
    inputs = [
        "cooling_since_2016",
        "Cooling Since 2016",
        "cooling™ since 2016!",
        "a-" * 40,
        "gold-temp-line",
        "___",
        "",
    ]
    for raw in inputs:
        once = planner.normalise_chart_id(raw)
        assert planner.normalise_chart_id(once) == once
        assert once == "" or SLUG_RE.fullmatch(once)


# ---------------------------------------------------------------------------
# 1b. Normalisation through the parse path: before validate_spec, on both
#     attempts, and into the permalink hash input
# ---------------------------------------------------------------------------


def _recorded_underscore_spec() -> dict:
    """SYNTHETIC FIXTURE shaped like the live refusals: an otherwise
    fully valid gold spec whose chart_id carries the recorded
    underscores."""
    spec = spec_temp_line()
    spec["chart_id"] = "cooling_since_2016"
    return spec


def test_underscored_chart_id_is_normalised_not_refused():
    """The recorded defect: an otherwise-valid spec with an underscored
    chart_id is ACCEPTED on the fresh call — normalisation runs before
    validate_spec, so no retry is burned on a cosmetic id. Only the
    chart_id changes; every semantic field survives verbatim."""
    adapter = FakeAdapter(structured_results=[spec_output(_recorded_underscore_spec())])
    result = planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert len(adapter.calls_to("structured")) == 1
    expected = spec_temp_line()
    expected["chart_id"] = "cooling-since-2016"
    assert result.spec == expected


def test_normalised_chart_id_is_the_permalink_hash_input():
    """The permalink identity (charts.spec.spec_hash over
    PlannedChart.spec — service/chart_store.py hashes the planned spec
    verbatim) is minted from the NORMALISED id: the planned spec hashes
    identically to the same spec authored clean, and differently from
    the raw underscored form — cosmetic id variants converge on one
    permalink."""
    adapter = FakeAdapter(structured_results=[spec_output(_recorded_underscore_spec())])
    result = planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
    assert isinstance(result, PlannedChart)
    clean = spec_temp_line()
    clean["chart_id"] = "cooling-since-2016"
    assert chartspec.spec_hash(result.spec) == chartspec.spec_hash(clean)
    assert chartspec.spec_hash(result.spec) != chartspec.spec_hash(_recorded_underscore_spec())


def test_chart_id_normalisation_applies_to_the_retry_outcome_too():
    """Normalisation is not a fresh-call special: a genuine violation
    burns the single retry, and the retry's spec — arriving with an
    underscored id — is normalised and accepted the same way."""
    retry_spec = spec_temp_line()
    retry_spec["chart_id"] = "temp_cooling_since_2016"
    adapter = FakeAdapter(
        structured_results=[spec_output(cherry_pick_domain_spec()), spec_output(retry_spec)]
    )
    result = planner.plan_chart_request(adapter, "plot temperature", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert len(adapter.calls_to("structured")) == 2
    assert result.spec["chart_id"] == "temp-cooling-since-2016"


def test_unrescuable_chart_id_still_refuses_and_feeds_back():
    """Genuine emptiness is NOT normalised into acceptance: an
    off-alphabet-only id (normalises to "") is still validator-refused,
    the feedback retry fires naming chart_id, and the retry's clean spec
    is returned."""
    bad = spec_temp_line()
    bad["chart_id"] = "___"
    good = spec_temp_line()
    adapter = FakeAdapter(structured_results=[spec_output(bad), spec_output(good)])
    result = planner.plan_chart_request(adapter, "plot temperature", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert result.spec == good
    assert "chart_id" in _retry_system_head(adapter)


def test_unrescuable_chart_id_twice_raises_typed_error_naming_chart_id():
    """Two unrescuable ids: exactly two calls, then the typed
    PlannerSpecError whose violations name chart_id — never a silently
    invented id, never a bare crash."""
    first = spec_temp_line()
    first["chart_id"] = "!!!"
    second = spec_temp_line()
    second["chart_id"] = ""
    adapter = FakeAdapter(structured_results=[spec_output(first), spec_output(second)])
    with pytest.raises(PlannerSpecError) as excinfo:
        planner.plan_chart_request(adapter, "plot temperature", gold_manifest())
    assert len(adapter.calls_to("structured")) == 2
    assert any("chart_id" in violation for violation in excinfo.value.violations)


# ---------------------------------------------------------------------------
# 2. The worked spec skeleton in the prompt (fresh AND retry), within the
#    #165 line cap and a ~200-token budget
# ---------------------------------------------------------------------------

#: Instruction-section size (chars) of the pre-#276 prompt, measured on
#: the gold catalogue at red-authoring time (2026-09-04, main @ b1fc696):
#: len of the system head before the catalogue payload.
PRE_276_HEAD_CHARS = 2315

#: The issue's budget for the skeleton addition, at the flagged chars/4
#: heuristic (see module docstring: the repo's token helpers both
#: under-estimate, the wrong direction for a cap).
SKELETON_TOKEN_BUDGET = 200
CHARS_PER_TOKEN = 4

#: A hyphenated example chart_id literal — the skeleton must SHOW the
#: slug shape the live model kept getting wrong, not merely state it.
_EXAMPLE_CHART_ID_RE = re.compile(r'"chart_id"\s*:\s*"([a-z0-9]+(?:-[a-z0-9]+)+)"')

#: A filled series entry's dataset reference and transform syntax.
_EXAMPLE_DATASET_RE = re.compile(r'"dataset"\s*:\s*"([^"]+)"')
_EXAMPLE_TRANSFORM_OP_RE = re.compile(r'"op"\s*:\s*"([^"]+)"')


def test_prompt_carries_worked_spec_skeleton_on_fresh_and_retry():
    """The instruction section carries one compact COMPLETE worked
    skeleton, on the fresh request AND the violations-feedback retry: a
    hyphenated chart_id literal, one FILLED series entry (a real
    catalogue dataset id, a label, the transforms/op syntax) and
    time_range_ce. The live sessions proved vocabulary bullets alone do
    not teach structure — three attempts, zero filled series entries."""
    catalogue = gold_catalogue()
    fresh = planner.build_planner_request("Show me the cooling since 2016", catalogue)
    retry = planner.build_planner_request(
        "Show me the cooling since 2016",
        catalogue,
        violations=("series: schema violation (minItems): should be non-empty",),
    )
    for request in (fresh, retry):
        head = _system_head(request["system"])
        # A hyphenated example chart_id literal (the slug SHAPE shown,
        # not just stated).
        chart_ids = _EXAMPLE_CHART_ID_RE.findall(head)
        assert chart_ids, "worked skeleton must show a hyphenated chart_id literal"
        assert all(SLUG_RE.fullmatch(chart_id) for chart_id in chart_ids)
        # One filled series entry: the example plots a REAL catalogue
        # dataset (never an invented id) and carries a label.
        datasets = _EXAMPLE_DATASET_RE.findall(head)
        assert datasets, "worked skeleton must fill a series dataset reference"
        assert all(ds_id in catalogue["datasets"] for ds_id in datasets)
        assert '"label"' in head
        # The transforms syntax, with an op from the frozen vocabulary.
        assert '"transforms"' in head
        ops = _EXAMPLE_TRANSFORM_OP_RE.findall(head)
        assert ops, "worked skeleton must show the transforms op syntax"
        assert all(op in chartspec.TRANSFORM_OPS for op in ops)
        # The range field the full-context rule keeps talking about.
        assert '"time_range_ce"' in head


def test_worked_skeleton_stays_within_line_cap_and_token_budget():
    """The skeleton addition is bounded: every instruction-section line
    stays within the #165 trusted-channel line cap, and the section
    grows at most ~200 tokens (chars/4, flagged) over the measured
    pre-#276 baseline. The skeleton-presence precondition keeps this
    test red until the skeleton exists (a bound over an absent addition
    would pin nothing)."""
    request = planner.build_planner_request("plot temperature", gold_catalogue())
    head = _system_head(request["system"])
    assert _EXAMPLE_CHART_ID_RE.search(head), "budget is measured over the added skeleton"
    for line in head.splitlines():
        # The '- ' allowance mirrors test_retry_violation_lines_are_length_capped.
        assert len(line) <= planner.VIOLATION_FEEDBACK_MAX_LENGTH + 2
    assert len(head) <= PRE_276_HEAD_CHARS + SKELETON_TOKEN_BUDGET * CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# 3. Sharpened empty-series feedback: the retry names the plottable
#    catalogue dataset ids
# ---------------------------------------------------------------------------


def _empty_series_spec() -> dict:
    """SYNTHETIC FIXTURE: the live defect isolated — a valid spec but
    for ``series: []`` (the minItems refusal)."""
    spec = spec_temp_line()
    spec["series"] = []
    return spec


def test_empty_series_feedback_names_catalogue_dataset_ids():
    """When validate_spec refuses on the empty-series minItems
    violation, the retry's INSTRUCTION section (not merely the appended
    catalogue payload, which was always there and demonstrably did not
    help live) names the catalogue dataset ids the request could plot —
    the material the #165-redacted violation line withheld. Coverage
    filtering deliberately unpinned (flagged: no per-request coverage
    distinction exists in the planner's catalogue filtering)."""
    good = spec_temp_line()
    adapter = FakeAdapter(structured_results=[spec_output(_empty_series_spec()), spec_output(good)])
    result = planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert result.spec == good
    retry_head = _retry_system_head(adapter)
    for ds_id in gold_catalogue()["datasets"]:
        assert ds_id in retry_head


def test_recorded_failure_shape_now_lands_a_planned_chart():
    """End-to-end pin of the live incident's remedy: attempt 1 returns
    the recorded shape (underscored chart_id AND empty series), the
    retry feedback names the plottable dataset ids, and the retry's
    populated spec — still underscored, as the live model insisted —
    is normalised and accepted. What cost three billed sessions with no
    fixture must now be a routine two-call success."""
    recorded = _recorded_underscore_spec()
    recorded["series"] = []
    retry_spec = spec_temp_line()
    retry_spec["chart_id"] = "temp_cooling_since_2016"
    adapter = FakeAdapter(structured_results=[spec_output(recorded), spec_output(retry_spec)])
    result = planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert result.spec["chart_id"] == "temp-cooling-since-2016"
    assert result.spec["series"] == spec_temp_line()["series"]
    retry_head = _retry_system_head(adapter)
    for ds_id in gold_catalogue()["datasets"]:
        assert ds_id in retry_head
