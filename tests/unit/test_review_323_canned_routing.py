"""Canned out_of_scope routing regression (review finding #323) — RED.

The verification smoke (PR #321, `data/verification-smoke/REPORT.md`)
measured canned_out_of_scope 9/9 -> 5/9: qa-na-c-03/-04/-09 and
qa-na-g-03 now classify as retrieval instead of canned. Each still
declines honestly via the #313 marker — no unsafe answer — but four
out-of-scope questions now take the paid retrieval+generation path
before declining: a cost/latency/routing regression. Prime suspect: the
#315 voices-boundary instruction rewrite crowded out the (previously
implicit) out_of_scope guidance — the current instructions name usage
rules for chart_request, voices, adversarial_in_scope and unsafe, but
say NOTHING about when to use out_of_scope.

The four regressed shapes share a signature: climate-ADJACENT surface
(electric cars, airline emissions, flood-defence spending, a carbon
footprint) wrapped around an ask the evidence corpus cannot answer
(market statistics, consumer comparisons, local-government finance, the
chatbot's own operations).

Pinned at the #315 file's three seams (this fix must NOT reopen that
boundary — tests/unit/test_review_315_voices_boundary.py stays green
untouched):

1. The classifier's committed processing instructions carry an explicit
   out_of_scope rule naming the climate-adjacent trap shapes — anchor
   language, characterisation-guard style (never exact phrasing). RED.
2. The labelled classifier set carries all four gold question texts
   VERBATIM as out_of_scope regression cases (fixture entries land with
   this suite; the labelled set is a fixture, not a gold), cross-checked
   verbatim against evals/gold/climate_qa.yaml so the two can never
   drift.
3. The routing seam: the gold label drives process_query to the CANNED
   route — the polite redirect, no retrieval query, zero generation
   calls (the cost model the gate measures).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from rag.provider import FakeAdapter, StructuredResult
from rag.query import (
    Route,
    ScopeClass,
    build_query_processing_request,
    process_query,
)

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
LABELLED_SET_PATH = FIXTURES_ROOT / "classifier" / "labelled_queries.yaml"
CLIMATE_QA_GOLD_PATH = Path(__file__).resolve().parents[2] / "evals" / "gold" / "climate_qa.yaml"

#: The four regressed canned golds (gold id -> question text VERBATIM from
#: evals/gold/climate_qa.yaml — authored gold-set text, never harvested
#: user input; the q50/qa-tg-01 convention).
REGRESSED_CANNED_GOLDS: dict[str, str] = {
    "qa-na-c-03": "How many electric cars were sold in Brazil in 2025?",
    "qa-na-c-04": "Which airline has the lowest emissions per passenger mile?",
    "qa-na-c-09": "How much has Somerset Council spent on flood defences since 2020?",
    "qa-na-g-03": "What's the carbon footprint of running this chatbot?",
}


def _instructions() -> str:
    return str(build_query_processing_request("placeholder question")["system"])


def _has(pattern: str) -> bool:
    return re.search(pattern, _instructions(), flags=re.IGNORECASE | re.DOTALL) is not None


# ---------------------------------------------------------------------------
# 1. The classifier prompt carries explicit out_of_scope guidance (RED)
# ---------------------------------------------------------------------------


class TestClassifierPromptCannedAnchors:
    def test_instructions_carry_an_explicit_out_of_scope_rule(self):
        """Every other routed class gets a 'Use <class> ...' rule;
        out_of_scope appears only in the bare enum listing — the
        crowding the #315 rewrite left behind. The refined instructions
        must state when to USE out_of_scope, in the prompt's own
        house style."""
        assert _has(r"\buse out_of_scope\b"), (
            "the classifier instructions must carry an explicit "
            "out_of_scope usage rule (every other routed class has one; "
            "the smoke run's 4 canned misroutes trace to its absence)"
        )

    def test_instructions_name_the_climate_adjacent_trap(self):
        """The regressed shapes' shared signature, stated explicitly:
        a climate-adjacent or climate-flavoured surface does not by
        itself make a question answerable from the evidence corpus."""
        assert _has(r"climate.{0,3}(adjacent|related|flavoured|flavored|themed)"), (
            "the classifier instructions must name the climate-adjacent "
            "trap: mentioning cars/emissions/flooding/carbon does not by "
            "itself put a question in scope"
        )

    def test_instructions_name_consumer_and_market_statistic_shapes(self):
        """qa-na-c-03/-04's shapes: consumer comparisons or purchasing
        advice, and market/sales statistics, are out_of_scope even when
        the product or metric is climate-flavoured."""
        assert _has(
            r"\b(consumer|purchas\w*|product)\b.{0,80}\b(advice|recommendation|comparison|ranking)"
        ), (
            "the classifier instructions must name the consumer "
            "advice/comparison shape (qa-na-c-04, q27/q52)"
        )
        assert _has(r"\b(market|sales)\b.{0,60}\b(statistic|figure|data|number)"), (
            "the classifier instructions must name the market/sales "
            "statistic shape (qa-na-c-03, q51)"
        )

    def test_instructions_name_finance_and_spending_shapes(self):
        """qa-na-c-09's shape: company or government finance/spending
        questions are not assessed climate evidence."""
        assert _has(r"\b(spend\w*|budget\w*|financ\w*)\b"), (
            "the classifier instructions must name the finance/spending shape (qa-na-c-09, q53)"
        )

    def test_instructions_name_the_service_itself_shape(self):
        """qa-na-g-03's shape: questions about the chatbot/service
        itself (its own footprint, operations, construction) are
        out_of_scope — the corpus describes the climate, not this
        assistant."""
        assert _has(
            r"(chatbot|assistant|bot|service|site)\b.{0,40}\bitself"
            r"|about (me|this (chatbot|assistant|bot|service|site))"
        ), (
            "the classifier instructions must name the "
            "service-itself/self-referential shape (qa-na-g-03, q54)"
        )

    def test_instructions_keep_the_despair_boundary_untouched(self):
        """Non-contradiction guard (finding #89's other side): the new
        out_of_scope guidance must not sweep doom/despair-phrased
        evidence questions out of scope — nothing may instruct that
        emotional or despairing phrasing routes out_of_scope."""
        text = _instructions().lower()
        for poison in (
            "despairing questions are out_of_scope",
            "emotional questions are out_of_scope",
            "use out_of_scope for doom",
            "hopeless questions are out_of_scope",
        ):
            assert poison not in text, poison


# ---------------------------------------------------------------------------
# 2. The labelled set carries the four regressed golds verbatim
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labelled_queries() -> list[dict]:
    data = yaml.safe_load(LABELLED_SET_PATH.read_text(encoding="utf-8"))
    return data["queries"]


@pytest.fixture(scope="module")
def gold_items_by_id() -> dict[str, dict]:
    data = yaml.safe_load(CLIMATE_QA_GOLD_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["items"]}


def _entry_for(queries: list[dict], text: str) -> dict:
    matches = [q for q in queries if q["text"] == text]
    assert matches, (
        f"the labelled classifier set must carry this regressed canned "
        f"gold's question verbatim (issue #323): {text!r}"
    )
    return matches[0]


class TestLabelledSetCarriesTheRegressedGolds:
    @pytest.mark.parametrize("gold_id", sorted(REGRESSED_CANNED_GOLDS))
    def test_regressed_gold_question_is_labelled_out_of_scope(self, labelled_queries, gold_id):
        """Each smoke-run misroute becomes a permanent labelled
        regression case: the gold question text, verbatim, labelled
        out_of_scope."""
        entry = _entry_for(labelled_queries, REGRESSED_CANNED_GOLDS[gold_id])
        assert entry["label"] == ScopeClass.OUT_OF_SCOPE.value, (
            f"{entry['id']} ({gold_id}): expected out_of_scope, got {entry['label']!r}"
        )

    @pytest.mark.parametrize("gold_id", sorted(REGRESSED_CANNED_GOLDS))
    def test_regressed_gold_entries_are_flagged_edge_cases(self, labelled_queries, gold_id):
        """All four are deliberate climate-adjacent near-misses — flagged
        edge_case with explanatory notes so the accuracy report slices
        them distinctly."""
        entry = _entry_for(labelled_queries, REGRESSED_CANNED_GOLDS[gold_id])
        assert entry.get("edge_case") is True, entry.get("id")
        assert entry.get("notes"), entry.get("id")
        assert gold_id in entry["notes"], (
            f"{entry['id']}: notes must name the gold item it regresses ({gold_id})"
        )

    @pytest.mark.parametrize("gold_id", sorted(REGRESSED_CANNED_GOLDS))
    def test_labelled_texts_match_the_gold_set_verbatim(self, gold_items_by_id, gold_id):
        """Anti-drift cross-check: this suite's verbatim constants (and
        so the labelled fixture entries they locate) match the gold
        set's question text character-for-character, and the gold item
        still expects the canned route."""
        gold_item = gold_items_by_id[gold_id]
        assert gold_item["question"] == REGRESSED_CANNED_GOLDS[gold_id]
        assert gold_item["expected_route"] == "canned_out_of_scope"


# ---------------------------------------------------------------------------
# 3. The routing seam: the gold label drives the CANNED route at zero
#    generation cost
# ---------------------------------------------------------------------------


def _decision_for(question: str):
    adapter = FakeAdapter()
    adapter.queue(
        "structured",
        StructuredResult(
            value={
                "scope": ScopeClass.OUT_OF_SCOPE.value,
                "rewritten_query": question,
                "language": "en",
            },
            usage={"input_tokens": 200, "output_tokens": 30},
        ),
    )
    return adapter, process_query(adapter, question)


class TestGoldLabelRoutesCannedAtZeroGenerationCost:
    @pytest.mark.parametrize("gold_id", sorted(REGRESSED_CANNED_GOLDS))
    def test_out_of_scope_label_routes_canned_with_no_generation_call(self, gold_id):
        """The cost model the canned_out_of_scope gate measures: an
        out_of_scope classification produces the polite canned redirect
        with NO retrieval query and NO generation call — one structured
        classify call is the exchange's entire LLM spend."""
        adapter, decision = _decision_for(REGRESSED_CANNED_GOLDS[gold_id])
        assert decision.route is Route.CANNED
        assert decision.canned_response
        assert "climate" in decision.canned_response.lower()
        assert decision.retrieval_query is None
        assert decision.chart_request is None
        assert [call.method for call in adapter.calls] == ["structured"]
