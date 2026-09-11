"""Issue #350 red phase (Fable): the qa-va-03 action-question ranking gap —
the rewrite seam must preserve action/personal-agency intent.

Run 4: qa-va-03 ("What can one person in the UK actually do about the
climate emergency?") marker-declined on starving retrieval — its
gold-named OWID travel/food and NOAA slow-reverse ACTION chunks are in the
corpus but never ranked (top answerable score 0.054881, the single lowest
in the run — it set the pre-filter floor), while the battery retrieved
Met Office UK-projection chunks; generation then coin-flipped between a
hedged answer and a decline (answered 3/3 on probe).

DIAGNOSTIC FINDING (recorded here because the run journals do NOT carry
the classifier's rewritten-query text): a zero-API local replay of the
run-4 retrieval stack (the run's qdrant snapshot + the pinned local
BGE-M3 embedder and bge-reranker-v2-m3) shows the battery's rewrite was
retrieval-IDENTICAL to the raw question — the same 8 Met Office/Hansen
chunk ids in the same order with the same 0.054881 top score. So the
rewrite did NOT swap the question for topic nouns (a topic-only rewrite
like "UK climate change impacts" reranks Met Office chunks at 0.95-0.99,
nothing like the battery's scores); it preserved the user's wording, as
the current instruction demands — and the bare interrogative phrasing
itself ranks the gold action chunks at 5.8e-05..8.2e-04. The same replay
shows a declarative ACTION-vocabulary rewrite ("individual personal
actions to reduce carbon footprint...: travel choices, food choices, what
one person can do") fills the whole top-8 with OWID travel/food action
content at 0.15-0.71, gold included. The honest in-repo seam is therefore
exactly the issue's prescription: the REWRITE INSTRUCTIONS must direct
the standalone query to carry the vocabulary of the content the question
asks for — for an action/personal-agency question, the action vocabulary,
never just topic nouns or the unmodified interrogative — with the
labelled set scoring the live model against that expectation. NOT
reranker gaming.

FLAGGED CONTRACT (test-author decisions):

- The labelled set gains an optional ``rewrite_must_carry`` key (schema
  extended in tests/unit/test_labelled_query_set.py, bound already 62):
  a list of term GROUPS; the expectation is met iff every group has at
  least one term present in the rewritten query — case-insensitive,
  WHOLE-WORD ("act" never matches inside "impacts").
- The scoring helper is pure:
  ``evals.scripts.classifier_accuracy.rewrite_meets_expectation``.
- ``classify_query`` records the classifier's actual rewrite on the
  Prediction (``predicted_rewrite``) and scores any labelled expectation
  (``rewrite_expectation_met``: True/False, None when the entry labels
  no expectation or the call errored); ``summarise`` surfaces a
  ``rewrite_expectation`` block ({total, met, misses: [{id,
  predicted_rewrite}]}) so the release battery reports rewrite misses
  instead of losing them (run 4's rewrite text was unjournalled — this
  is also the observability fix). Scope accuracy is UNCHANGED by the
  expectation: a rewrite miss is its own reported slice, never a
  scope-accuracy miss.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

import re

from evals.scripts.classifier_accuracy import (
    Prediction,
    classify_query,
    load_labelled_queries,
    rewrite_meets_expectation,
    summarise,
)
from rag.provider import FakeAdapter
from rag.query import build_query_processing_request, process_query
from tests.unit.test_labelled_query_set import LABELLED_SET_PATH

QA_VA_03_QUESTION = "What can one person in the UK actually do about the climate emergency?"

#: The action-vocabulary group the q62 fixture entry labels.
ACTION_GROUP = ["do", "action", "actions", "act", "steps", "choices", "footprint"]

#: The failure-shape rewrite: topic nouns only, action intent dropped.
TOPIC_ONLY_REWRITE = "United Kingdom climate change impacts"

#: An action-preserving rewrite of the same question (the shape the local
#: run-4 replay showed ranks the OWID/NOAA action content).
ACTION_REWRITE = (
    "individual actions one person in the UK can take about the climate "
    "emergency — travel and food choices, personal carbon footprint"
)


def _q62_entry() -> dict:
    entries = [q for q in load_labelled_queries(LABELLED_SET_PATH) if q["id"] == "q62"]
    assert entries, "the q62 action-intent entry must exist (added with this suite)"
    return entries[0]


# ---------------------------------------------------------------------------
# The rewrite instructions: intent-preservation anchors.
# Characterisation anchors, never phrasing (house rule).
# ---------------------------------------------------------------------------


def _rewrite_instruction_region() -> str:
    """The rewrite half of the combined call's system instructions: from
    the rewrite directive up to the scope-classification directive."""
    system = build_query_processing_request("placeholder question")["system"]
    rewrite_at = system.lower().index("rewrite")
    classify_at = system.lower().index("classif", rewrite_at)
    return system[rewrite_at:classify_at]


class TestRewriteInstructionAnchors:
    def test_rewrite_instruction_preserves_what_the_question_asks_for(self) -> None:
        """ANCHOR: the rewrite directive must say the standalone query
        preserves what the question ASKS FOR — its intent — not merely
        resolve pronouns and expand acronyms. Run 4's floor item shows a
        rewrite can be faithful to the words and still starve retrieval
        of the asked-for content's vocabulary."""
        region = _rewrite_instruction_region()
        assert re.search(r"\bintent\b|asks?\s+for|asking\s+for|seeks|sought", region, re.I), (
            "the rewrite instruction must direct the query to preserve the "
            "question's intent (what it asks for), not only its referents "
            f"— got: {region!r}"
        )

    def test_rewrite_instruction_names_the_action_agency_shape(self) -> None:
        """ANCHOR: the action/personal-agency shape is named — a
        what-can-I-do question's rewrite carries the action vocabulary
        (actions, choices, personal/individual agency), never just the
        topic nouns. This is run 4's qa-va-03 (the last in-repo-fixable
        false_refusal item)."""
        region = _rewrite_instruction_region()
        assert re.search(
            r"\baction\w*\b|what\s+.{0,40}\bcan\s+do\b|\bagency\b|\bpersonal\b|\bindividual\b",
            region,
            re.I,
        ), (
            "the rewrite instruction must name the action/personal-agency "
            f"question shape and its vocabulary — got: {region!r}"
        )


# ---------------------------------------------------------------------------
# rewrite_meets_expectation: the pure scoring helper.
# ---------------------------------------------------------------------------


class TestRewriteMeetsExpectation:
    def test_topic_only_rewrite_fails_the_action_expectation(self) -> None:
        """The run-4 failure shape: topic nouns carry no action term —
        and 'act' inside 'impacts' must NOT count (whole-word rule)."""
        assert rewrite_meets_expectation(TOPIC_ONLY_REWRITE, [ACTION_GROUP]) is False

    def test_action_rewrite_meets_the_expectation(self) -> None:
        assert rewrite_meets_expectation(ACTION_REWRITE, [ACTION_GROUP]) is True

    def test_the_users_own_wording_meets_the_expectation(self) -> None:
        """The raw question carries 'do' — the user's own action wording
        satisfies the group, so a faithful standalone rewrite of it can
        never be scored a miss for keeping the user's words."""
        assert rewrite_meets_expectation(QA_VA_03_QUESTION, [ACTION_GROUP]) is True

    def test_matching_is_case_insensitive(self) -> None:
        assert rewrite_meets_expectation("Actions a UK household can take", [ACTION_GROUP]) is True

    def test_every_group_must_be_satisfied(self) -> None:
        """Groups are conjunctive: one satisfied group cannot carry an
        unsatisfied one."""
        groups = [ACTION_GROUP, ["aviation", "flight"]]
        assert rewrite_meets_expectation(ACTION_REWRITE, groups) is False
        assert rewrite_meets_expectation(ACTION_REWRITE + " and flight habits", groups) is True


# ---------------------------------------------------------------------------
# classify_query + summarise: the FakeAdapter labelled expectation for the
# shape — the battery scores and SURFACES rewrite misses.
# ---------------------------------------------------------------------------


def _classify_with_rewrite(rewritten: str) -> Prediction:
    adapter = FakeAdapter(structured_results=[{"scope": "in_scope", "rewritten_query": rewritten}])
    return classify_query(adapter, _q62_entry())


class TestLabelledExpectationScoring:
    def test_prediction_records_the_actual_rewrite(self) -> None:
        """Run 4's rewrite text was unjournalled and the diagnosis had to
        be reconstructed by local replay: the battery must RECORD the
        classifier's rewritten query on the prediction."""
        prediction = _classify_with_rewrite(TOPIC_ONLY_REWRITE)
        assert prediction.predicted_rewrite == TOPIC_ONLY_REWRITE

    def test_topic_only_rewrite_is_scored_a_rewrite_miss(self) -> None:
        prediction = _classify_with_rewrite(TOPIC_ONLY_REWRITE)
        assert prediction.rewrite_expectation_met is False

    def test_action_preserving_rewrite_is_scored_met(self) -> None:
        prediction = _classify_with_rewrite(ACTION_REWRITE)
        assert prediction.rewrite_expectation_met is True

    def test_rewrite_miss_never_dents_scope_accuracy(self) -> None:
        """The expectation is its own reported slice: q62 classified
        in_scope with a topic-only rewrite is still scope-CORRECT."""
        prediction = _classify_with_rewrite(TOPIC_ONLY_REWRITE)
        assert prediction.correct is True

    def test_entries_without_expectation_are_unscored_not_missed(self) -> None:
        adapter = FakeAdapter(
            structured_results=[{"scope": "in_scope", "rewritten_query": "how warm is it"}]
        )
        entry = {"id": "q01", "text": "How warm is it?", "label": "in_scope"}
        prediction = classify_query(adapter, entry)
        assert prediction.rewrite_expectation_met is None

    def test_summary_surfaces_rewrite_expectation_misses(self) -> None:
        """The release battery's results payload must carry the rewrite
        slice — total scored, met count, and each miss with the ACTUAL
        rewrite text (the observability run 4 lacked)."""
        miss = _classify_with_rewrite(TOPIC_ONLY_REWRITE)
        adapter = FakeAdapter(
            structured_results=[{"scope": "in_scope", "rewritten_query": ACTION_REWRITE}]
        )
        met = classify_query(adapter, _q62_entry())
        summary = summarise([miss, met])
        slice_ = summary["rewrite_expectation"]
        assert slice_["total"] == 2
        assert slice_["met"] == 1
        (recorded_miss,) = slice_["misses"]
        assert recorded_miss["id"] == "q62"
        assert recorded_miss["predicted_rewrite"] == TOPIC_ONLY_REWRITE


# ---------------------------------------------------------------------------
# Fixture + routing pins (green guards).
# ---------------------------------------------------------------------------


class TestFixtureAndRouting:
    def test_labelled_set_carries_the_action_intent_expectation(self) -> None:
        """The what-can-I-do shape is labelled with a rewrite expectation
        whose vocabulary is ACTION terms (not topic nouns)."""
        entry = _q62_entry()
        assert entry["label"] == "in_scope"
        assert entry["text"] == QA_VA_03_QUESTION
        groups = entry["rewrite_must_carry"]
        assert groups and all(group for group in groups), "term groups must be non-empty"
        flattened = {term.lower() for group in groups for term in group}
        assert flattened & {"do", "action", "actions"}, (
            "the expectation must demand action vocabulary — that is the intent "
            "the run-4 retrieval starved without"
        )

    def test_routing_carries_the_rewrite_verbatim_into_retrieval(self) -> None:
        """GREEN GUARD (the seam the fix rides): whatever the classifier
        rewrites, routing hands it to retrieval unmodified — so fixing
        the rewrite instruction IS fixing the retrieval query."""
        adapter = FakeAdapter(
            structured_results=[{"scope": "in_scope", "rewritten_query": ACTION_REWRITE}]
        )
        decision = process_query(adapter, QA_VA_03_QUESTION)
        assert decision.retrieval_query == ACTION_REWRITE
