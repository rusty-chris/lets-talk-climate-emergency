"""The voices_bias classification boundary (finding #315, blocker) — RED.

Release-run voices_separation FAILED: qa-tg-01 ("Chris Packham keeps
saying we're on the brink of irreversible tipping points. Is the
science actually behind him?") classified as ``voices``, so the
voices-biased include list legitimately admitted three ``source_type:
voices`` chunks into a TARGETED-science generation document set. The
include-list filter itself is correctly wired in eval AND production —
the leak is upstream, in the classifier's voices-class definition,
which does not exclude is-the-science-behind-X questions.

The boundary these tests pin (issue #315):

- "Is the science actually behind <named campaigner>?" is a SCIENTIFIC
  CLAIM ASSESSMENT — a science route (``in_scope`` or
  ``adversarial_in_scope``), ``voices_bias=False``, evidence-only
  include list.
- "What does <campaigner> say drives him?" seeks the voice's own
  words/motivations — ``voices`` stays ``voices``,
  ``voices_bias=True``, evidence AND voices (bias, never exclusion —
  §3.2).

Pinned at three seams:

1. The classifier's committed processing instructions
   (``build_query_processing_request(...)['system']``) carry the
   sharpened voices definition — anchor language, characterisation-
   guard style (never exact phrasing).
2. The labelled classifier query set gains BOTH shapes — the current
   48-item set scores voices 5/5 while permitting the leak, so it
   under-specifies the boundary. The qa-tg-01 question text rides
   VERBATIM as the regression case (it is authored gold-set text, not
   harvested user input).
3. The §6.2 invariant end-to-end at the unit-visible seam: driving
   ``process_query`` with the gold labels through a FakeAdapter, the
   science shape's decision yields an evidence-only
   ``permitted_source_types`` include list (a voices_bias=False
   retrieval can NEVER include voices chunks — include-list-first,
   finding #158), while the voices shape keeps evidence AND voices.

The live-index integration pin (qa-tg-01 through the release retrieve
-> zero voices chunks in ``passages``) is the eval harness's seam and
rides with the eval-side work; the voices_separation gate itself stays
exactly as is — it caught this.
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
from rag.retrieval import (
    EVIDENCE_SOURCE_TYPES,
    VOICES_SOURCE_TYPE,
    permitted_source_types,
)

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
LABELLED_SET_PATH = FIXTURES_ROOT / "classifier" / "labelled_queries.yaml"

#: qa-tg-01's question text, VERBATIM (evals/gold/climate_qa.yaml — the
#: §6.1 'science question phrased around Packham' item): the regression
#: case the release run leaked on.
QA_TG_01_QUESTION = (
    "Chris Packham keeps saying we're on the brink of irreversible tipping "
    "points. Is the science actually behind him?"
)

#: The keep-side companion shape: seeks the campaigner's own
#: words/motivations — legitimately voices.
CAMPAIGNER_MOTIVATION_QUESTION = "What does Chris Packham say drives his campaigning?"

#: The classes a scientific-claim-assessment question may carry: never
#: voices.
SCIENCE_CLASSES = {ScopeClass.IN_SCOPE.value, ScopeClass.ADVERSARIAL_IN_SCOPE.value}


def _instructions() -> str:
    return str(build_query_processing_request("placeholder question")["system"])


def _has(pattern: str) -> bool:
    return re.search(pattern, _instructions(), flags=re.IGNORECASE | re.DOTALL) is not None


# ---------------------------------------------------------------------------
# 1. The classifier prompt carries the sharpened boundary
# ---------------------------------------------------------------------------


class TestClassifierPromptBoundaryAnchors:
    def test_instructions_name_the_science_behind_a_person_shape(self):
        """The instructions must describe the trap shape explicitly:
        a question asking whether the science/evidence is behind (backs,
        supports) a person's claim."""
        assert _has(
            r"\b(whether|if|is)\b.{0,120}\b(science|evidence)\b"
            r".{0,120}\b(behind|backs?|supports?)\b"
        ), (
            "the classifier instructions must name the "
            "is-the-science-behind-X shape (science/evidence ... "
            "behind/backs/supports a named person's claim)"
        )

    def test_instructions_route_claim_assessment_away_from_voices(self):
        """And must say that shape is NOT voices — sharpened exclusion,
        not just the positive testimony definition."""
        assert _has(r"\b(never|not) voices\b"), (
            "the classifier instructions must state explicitly that "
            "assessing whether the science supports a named person's "
            "claim is never the voices class"
        )

    def test_instructions_define_voices_as_words_or_motivations(self):
        """The keep side, stated positively: voices is for questions
        seeking the voice's own words / motivations / story — so
        'what drives him?' stays voices while 'is the science behind
        him?' leaves."""
        assert _has(r"(own words|motivations?|what (they|he|she) (say|says))"), (
            "the classifier instructions must define voices as seeking "
            "the voice's own words/motivations"
        )

    def test_instructions_do_not_route_by_name_alone(self):
        """Non-contradiction: nothing may instruct that a named
        campaigner/person makes a question voices."""
        text = _instructions().lower()
        for poison in (
            "any question naming a campaigner",
            "questions that mention a campaigner are voices",
            "mentions an activist is voices",
            "use voices whenever a person is named",
        ):
            assert poison not in text, poison


# ---------------------------------------------------------------------------
# 2. The labelled classifier set gains both boundary shapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labelled_queries() -> list[dict]:
    data = yaml.safe_load(LABELLED_SET_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def _entry_for(queries: list[dict], text: str) -> dict:
    matches = [q for q in queries if q["text"] == text]
    assert matches, (
        f"the labelled classifier set must carry this boundary query "
        f"verbatim (issue #315): {text!r}"
    )
    return matches[0]


class TestLabelledSetCarriesTheBoundary:
    def test_qa_tg_01_shape_is_labelled_a_science_class(self, labelled_queries):
        """The released leak, as a permanent labelled regression case:
        qa-tg-01's text, verbatim, labelled in_scope or
        adversarial_in_scope — never voices."""
        entry = _entry_for(labelled_queries, QA_TG_01_QUESTION)
        assert entry["label"] in SCIENCE_CLASSES, (
            f"{entry['id']}: the is-the-science-behind-him shape is a "
            f"scientific claim assessment, got label {entry['label']!r}"
        )

    def test_campaigner_motivation_shape_stays_voices(self, labelled_queries):
        """The boundary's other side, so sharpening the definition can
        never over-rotate: seeking the campaigner's own account stays
        voices."""
        entry = _entry_for(labelled_queries, CAMPAIGNER_MOTIVATION_QUESTION)
        assert entry["label"] == ScopeClass.VOICES.value

    def test_boundary_pair_is_marked_edge_case(self, labelled_queries):
        """Both entries are the deliberate near-miss pair — flagged
        edge_case so the accuracy report surfaces them distinctly."""
        for text in (QA_TG_01_QUESTION, CAMPAIGNER_MOTIVATION_QUESTION):
            entry = _entry_for(labelled_queries, text)
            assert entry.get("edge_case") is True, entry.get("id")


# ---------------------------------------------------------------------------
# 3. §6.2 end-to-end at the unit seam: gold label -> routing -> include list
# ---------------------------------------------------------------------------


def _decision_for(question: str, scope: str):
    adapter = FakeAdapter()
    adapter.queue(
        "structured",
        StructuredResult(
            value={"scope": scope, "rewritten_query": question, "language": "en"},
            usage={"input_tokens": 200, "output_tokens": 30},
        ),
    )
    return process_query(adapter, question)


class TestGoldLabelsProduceVoicesSafeRetrieval:
    def test_science_labelled_qa_tg_01_never_admits_voices(self, labelled_queries):
        """The full §6.2 chain for the leaked item, healed: the gold
        label drives routing, and the resulting decision's include list
        is evidence-only — a voices chunk is structurally outside it
        (finding #158: include-list-first; the run's #174 per-candidate
        assertion enforces the same list on every returned chunk)."""
        entry = _entry_for(labelled_queries, QA_TG_01_QUESTION)
        decision = _decision_for(QA_TG_01_QUESTION, entry["label"])
        assert decision.route is Route.RETRIEVAL
        assert decision.voices_bias is False
        include_list = permitted_source_types(decision)
        assert include_list == EVIDENCE_SOURCE_TYPES
        assert VOICES_SOURCE_TYPE not in include_list

    def test_voices_labelled_motivation_question_keeps_bias_not_exclusion(self, labelled_queries):
        """The companion shape keeps §3.2's stance intact: voices_bias
        True, include list = evidence AND voices."""
        entry = _entry_for(labelled_queries, CAMPAIGNER_MOTIVATION_QUESTION)
        decision = _decision_for(CAMPAIGNER_MOTIVATION_QUESTION, entry["label"])
        assert decision.route is Route.RETRIEVAL
        assert decision.voices_bias is True
        include_list = permitted_source_types(decision)
        assert VOICES_SOURCE_TYPE in include_list
        for evidence_type in EVIDENCE_SOURCE_TYPES:
            assert evidence_type in include_list
