"""Chart planner review finding #271 — RED.

Failing behavioural tests for the two live failures of the second #162
recording session (PR #270, 2026-09-03) plus the output-budget gap that
truncated its second attempt:

1. **Cherry-pick prompt rule (DESIGN §3.7 full-context default).** Driven
   live on "Show me the cooling since 2016", the planner refused
   (``outcome="unavailable"``: "datasets show warming, not cooling")
   instead of serving the FULL-RANGE chart that shows the cherry-picked
   window in context. The existing scaffold pins only the
   full-available-range *default*; these tests pin an EXPLICIT rule — a
   requested window that exists inside a dataset's coverage but reflects
   a selective framing gets the full-range spec, never a refusal, with
   ``unavailable`` reserved for genuinely unplottable requests — plus a
   worked cherry-pick example (decision flagged in the PR-less red
   report: an abstract rule alone demonstrably failed live on the Haiku
   tier, an in-prompt example is the strongest cheap steering).

2. **Degenerate-output hardening.** Attempt 1's payload was garbled:
   embedded BOM, fullwidth glyphs (GISTEMP/HadCRUT5 in fullwidth forms),
   CJK punctuation, the year 2016 dropped. Today the parse path accepts
   that text into the curation-gap record. These tests pin: degenerate
   text is detected (``is_degenerate_output_text``; rule flagged below),
   consumes the existing shared retry-once budget (same-request retry,
   the #10 malformed-output convention), and a second degenerate output
   degrades to the typed :class:`~charts.planner.PlannerSpecError`
   naming the reason — never a crash, never silent acceptance of garbled
   content into a spec, a refusal record, or an error echo.

   Flagged detection rule pinned by the predicate tests: text is
   degenerate when it carries a BOM (U+FEFF) anywhere, or any character
   from the Halfwidth/Fullwidth Forms block (U+FF00-U+FFEF), or any
   character whose NFKC normalisation introduces ASCII alphanumerics the
   raw text did not carry (fullwidth/confusable letters and digits).
   Benign non-ASCII (degree signs, accented letters, typographic dashes)
   is explicitly NOT degenerate — pinned by the negative cases.

3. **Scaled output budget (#205 pattern).** Attempt 2 died as invalid
   JSON, consistent with max_tokens truncation at the 2048 cap. The
   validator's ``SPEC_MAX_BYTES`` (8 KiB) bounds every spec that can
   validate, so the honest worst-case planner output is a spec at that
   ceiling. Pinned formula floor (flagged):
   ``max_tokens >= ceil(SPEC_MAX_BYTES / 2) + ENVELOPE_ALLOWANCE`` —
   1 token per 2 bytes is a deliberate overestimate (ASCII-heavy JSON
   tokenises at ~2.5-4 bytes/token), so truncation of ANY validatable
   spec is impossible; the allowance covers the
   ``{"outcome": "spec", "spec": ...}`` wrapper. Like #205, this raises
   the CEILING, never the typical spend, and stays far under
   claude-haiku-4-5's 64K output cap.

The degenerate payload below is a SYNTHETIC FIXTURE derived from live
output: reconstructed verbatim from the attempt-1 ``requested_data``
preserved in PR #270's body (the recording protocol landed no fixture
file). Everything else reuses the SYNTHETIC gold fixtures of
tests/unit/test_chart_planner.py.
"""

from __future__ import annotations

import logging
import math

import pytest

from charts import planner
from charts import spec as chartspec
from charts.planner import ChartRefusal, PlannedChart, PlannerSpecError
from rag.provider import FakeAdapter
from tests.unit.test_chart_planner import (
    gold_catalogue,
    gold_manifest,
    spec_output,
    spec_temp_line,
    unavailable_output,
)

# ---------------------------------------------------------------------------
# SYNTHETIC FIXTURE derived from live output (PR #270 attempt 1, verbatim)
# ---------------------------------------------------------------------------

#: The recorded degenerate ``requested_data`` of the 2026-09-03 recording
#: session's first attempt, reconstructed character-for-character from PR
#: #270's body: CJK ideographic commas/full stop where '2016' should be
#: (U+3001/U+3002), fullwidth GISTEMP (U+FF27..) and HadCRUT5 (U+FF28..,
#: U+FF15) with an embedded BOM (U+FEFF) before the latter.
DEGENERATE_ATTEMPT_1_REQUESTED_DATA = (
    "a dataset showing temperature decline or cooling trend since 、 "
    "the user's request assumes a cooling trend since 、 but available "
    "temperature datasets (ＧＩＳＴＥＭＰ and "
    "\ufeffＨａｄＣＲＵＴ５) show warming, "
    "not cooling, since 。 the catalogue contains no dataset that would "
    "support visualizing a cooling trend in this period"
)


def _carries_garbled_glyphs(text: str) -> bool:
    """True when text carries the live incident's garbling markers: a BOM
    or any Halfwidth/Fullwidth Forms character."""
    return "\ufeff" in text or any(0xFF00 <= ord(ch) <= 0xFFEF for ch in text)


def degenerate_spec() -> dict:
    """SYNTHETIC FIXTURE: an otherwise-valid gold spec whose title carries
    the incident's garbling (fullwidth '2016' + BOM) — validate_spec's
    schema admits it (free text under maxLength), so only the #271
    degeneracy check can stop it reaching a rendered chart title."""
    bad = spec_temp_line()
    bad["title"] = "Global mean surface temperature anomaly since ２０１６\ufeff"
    return bad


def benign_unicode_spec() -> dict:
    """SYNTHETIC FIXTURE: a gold spec with legitimate non-ASCII text — a
    degree sign and a typographic en-dash — that must NOT be treated as
    degenerate."""
    good = spec_temp_line()
    good["subtitle"] = "Anomalies in °C relative to the 1951–1980 baseline"
    return good


# ---------------------------------------------------------------------------
# 1. The explicit cherry-pick rule in the planner system prompt (§3.7)
# ---------------------------------------------------------------------------


def test_prompt_carries_explicit_cherry_pick_rule():
    """The system prompt carries the #271 cherry-pick rule by name, on the
    fresh request AND the violations-feedback retry: a requested window
    that lies inside a dataset's coverage but reflects a selective
    framing gets the full-range spec ("never refuse a plottable" range),
    and the unavailable outcome is reserved for "genuinely unplottable"
    requests. The pre-#271 scaffold carried only the default-range
    instruction — live attempt 1 (PR #270) shows that was not enough."""
    catalogue = gold_catalogue()
    fresh = planner.build_planner_request("show me the cooling since 2016", catalogue)
    retry = planner.build_planner_request(
        "show me the cooling since 2016",
        catalogue,
        violations=("series[0].scale_domain: excludes zero",),
    )
    for request in (fresh, retry):
        system = request["system"].lower()
        # The rule names the framing it defends against...
        assert "selective framing" in system
        # ...forbids the live failure mode by name...
        assert "never refuse a plottable" in system
        # ...and reserves the honest exit for the genuinely unservable.
        assert "genuinely unplottable" in system
        # The existing full-range default stays alongside the new rule.
        assert "full available range" in system


def test_prompt_carries_cherry_pick_worked_example():
    """The prompt carries a concrete worked example of the rule (the live
    cherry-pick request family: a "cooling since <year>" framing answered
    with the full-range chart). Decision flagged for the report: Haiku
    ignored the abstract default live, an in-prompt example is the
    strongest cheap steering, and its ~50 input tokens are negligible
    against the ~1.5K-token catalogue payload."""
    request = planner.build_planner_request("plot temperature", gold_catalogue())
    system = request["system"].lower()
    assert "cooling since" in system
    # The example must teach the full-range answer, not just name the trap.
    assert "full available range" in system


# ---------------------------------------------------------------------------
# 2. Degenerate-output detection (the flagged Unicode rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(DEGENERATE_ATTEMPT_1_REQUESTED_DATA, True, id="live-attempt-1-payload"),
        pytest.param("\ufeffglobal mean sea level", True, id="embedded-bom"),
        pytest.param("ＧＩＳＴＥＭＰ anomaly", True, id="fullwidth-glyphs"),
        pytest.param("global mean sea level since 1900", False, id="plain-ascii"),
        pytest.param("temperature anomaly in °C", False, id="benign-degree-sign"),
        pytest.param("café résumé — 1951–1980", False, id="benign-accents-and-dashes"),
    ],
)
def test_degenerate_output_text_predicate(text, expected):
    """The pure detection predicate, pinned on the live payload and on
    benign-Unicode negative controls so the rule cannot be over-broad
    (legitimate degree signs, accents and typographic dashes are not
    "degenerate")."""
    assert planner.is_degenerate_output_text(text) is expected


# ---------------------------------------------------------------------------
# 2. Degenerate output through the parse path: retry-once, then honest error
# ---------------------------------------------------------------------------


def test_degenerate_unavailable_output_retries_once_with_same_request(caplog):
    """A degenerate unavailable outcome is treated as malformed output —
    retried ONCE with the SAME request (the #10 convention; there are no
    validator violations to feed back) — and the garbled text never
    reaches the curation-gap record or log: only the clean retry's phrase
    does."""
    adapter = FakeAdapter(
        structured_results=[
            unavailable_output(DEGENERATE_ATTEMPT_1_REQUESTED_DATA),
            unavailable_output("global mean sea level"),
        ]
    )
    with caplog.at_level(logging.INFO, logger=planner.CURATION_GAP_LOGGER_NAME):
        result = planner.plan_chart_request(
            adapter, "Show me the cooling since 2016", gold_manifest()
        )
    assert isinstance(result, ChartRefusal)
    calls = adapter.calls_to("structured")
    assert len(calls) == 2
    assert calls[0].payload == calls[1].payload
    # Only the clean phrase survives, nowhere the garbled glyphs.
    assert result.gap.requested_data == "global mean sea level"
    assert not _carries_garbled_glyphs(result.gap.requested_data)
    assert not _carries_garbled_glyphs(result.message)
    records = [r for r in caplog.records if r.name == planner.CURATION_GAP_LOGGER_NAME]
    assert len(records) == 1
    assert not _carries_garbled_glyphs(records[0].requested_data)


def test_degenerate_unavailable_twice_degrades_to_typed_error_naming_reason():
    """Two degenerate outputs: exactly one retry, then the typed
    PlannerSpecError whose detail NAMES the degeneracy (so a log line
    alone is actionable) without echoing the garbled glyphs into the
    error/log channel — never a crash, never a ChartRefusal built from
    garbled content."""
    adapter = FakeAdapter(
        structured_results=[
            unavailable_output(DEGENERATE_ATTEMPT_1_REQUESTED_DATA),
            unavailable_output(DEGENERATE_ATTEMPT_1_REQUESTED_DATA),
        ]
    )
    with pytest.raises(PlannerSpecError) as excinfo:
        planner.plan_chart_request(adapter, "Show me the cooling since 2016", gold_manifest())
    assert len(adapter.calls_to("structured")) == 2
    detail = str(excinfo.value) + "".join(excinfo.value.violations)
    # The reason is named (anchor set flagged for the report)...
    assert any(anchor in detail.lower() for anchor in ("degenerate", "normalis", "unicode"))
    # ...and the garbled model text never rides the error channel.
    assert not _carries_garbled_glyphs(detail)


def test_degenerate_spec_text_never_reaches_planned_chart():
    """Garbled text inside a SPEC outcome (a fullwidth/BOM chart title
    that would flow verbatim into a rendered artefact) is likewise not
    silently accepted: the retry's clean spec is returned instead."""
    good = spec_temp_line()
    adapter = FakeAdapter(structured_results=[spec_output(degenerate_spec()), spec_output(good)])
    result = planner.plan_chart_request(adapter, "plot temperature", gold_manifest())
    assert isinstance(result, PlannedChart)
    assert result.spec == good
    assert len(adapter.calls_to("structured")) == 2


def test_degenerate_spec_twice_raises_typed_error_without_echoing_glyphs():
    """Two degenerate specs: exactly two calls, the typed error, and no
    BOM/fullwidth glyphs echoed into the error detail."""
    adapter = FakeAdapter(
        structured_results=[
            spec_output(degenerate_spec()),
            spec_output(degenerate_spec()),
        ]
    )
    with pytest.raises(PlannerSpecError) as excinfo:
        planner.plan_chart_request(adapter, "plot temperature", gold_manifest())
    assert len(adapter.calls_to("structured")) == 2
    detail = str(excinfo.value) + "".join(excinfo.value.violations)
    assert not _carries_garbled_glyphs(detail)


def test_benign_unicode_outputs_are_not_flagged_as_degenerate():
    """Negative control (green guard): legitimate non-ASCII in either
    outcome — degree signs, accents, typographic dashes — passes through
    with NO retry, so the hardening cannot regress honest content."""
    refusal_adapter = FakeAdapter(
        structured_results=[unavailable_output("sea-surface temperature in °C")]
    )
    refusal = planner.plan_chart_request(
        refusal_adapter, "plot sea-surface temperature", gold_manifest()
    )
    assert isinstance(refusal, ChartRefusal)
    assert len(refusal_adapter.calls_to("structured")) == 1

    spec = benign_unicode_spec()
    spec_adapter = FakeAdapter(structured_results=[spec_output(spec)])
    planned = planner.plan_chart_request(spec_adapter, "plot temperature", gold_manifest())
    assert isinstance(planned, PlannedChart)
    assert planned.spec == spec
    assert len(spec_adapter.calls_to("structured")) == 1


# ---------------------------------------------------------------------------
# 3. Scaled output budget (#205 pattern): truncation of a valid spec is
#    impossible
# ---------------------------------------------------------------------------

#: Tokens allowed for the ``{"outcome": "spec", "spec": ...}`` wrapper
#: around the spec payload — the analogue of #205's VERDICT_TOKENS_BASE.
ENVELOPE_TOKEN_ALLOWANCE = 64

#: The honest worst-case output: a spec at the validator's SPEC_MAX_BYTES
#: ceiling (every larger spec is refused by validate_spec, so the budget
#: need never cover it) at a deliberately pessimistic 2 bytes/token —
#: ASCII-heavy JSON tokenises at ~2.5-4 bytes/token, so this OVERestimates
#: the token count and makes mid-spec truncation impossible (the #205
#: never-truncate direction), while staying far under claude-haiku-4-5's
#: 64K output cap and raising only the ceiling, never the typical spend.
WORST_CASE_SPEC_TOKENS = math.ceil(chartspec.SPEC_MAX_BYTES / 2)


def test_planner_max_tokens_covers_a_spec_at_the_validator_byte_ceiling():
    """The #205-pattern budget floor: on the fresh request AND the
    violations-feedback retry, max_tokens covers a maximal valid spec
    (SPEC_MAX_BYTES at 2 bytes/token) plus the outcome envelope, and
    never drops below the configured PLANNER_MAX_TOKENS floor (the
    ``max(floor, scaled)`` shape)."""
    catalogue = gold_catalogue()
    fresh = planner.build_planner_request("plot temperature", catalogue)
    retry = planner.build_planner_request(
        "plot temperature",
        catalogue,
        violations=("series[0].scale_domain: excludes zero",),
    )
    floor = WORST_CASE_SPEC_TOKENS + ENVELOPE_TOKEN_ALLOWANCE
    for request in (fresh, retry):
        max_tokens = request["config"]["max_tokens"]
        assert max_tokens >= floor
        assert max_tokens >= planner.PLANNER_MAX_TOKENS
        # Cost-model guard: a ceiling, not spend — but still bounded
        # (claude-haiku-4-5 caps output at 64K; one runaway call must
        # never cost more than ~8K output tokens).
        assert max_tokens <= 8192


def test_planner_max_tokens_exceeds_the_recorded_truncation_ceiling():
    """The named #270 regression: attempt 2 of the recording session was
    (in all likelihood) truncated at the 2048-token cap into invalid
    JSON. 2048 must now be strictly impossible as the effective budget."""
    request = planner.build_planner_request("plot temperature", gold_catalogue())
    assert request["config"]["max_tokens"] > 2048
