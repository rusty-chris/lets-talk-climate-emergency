"""voices_action classification boundary (review finding #339) — RED.

Release run 2 (`data/release-run-2/REPORT.md`): false_refusal 4/55 = 7.27%
vs the <5% gate — qa-va-02/-04/-05 plus qa-sev-09 declined. The run-2
journals settle the issue's hypotheses PER ITEM (`calib_decisions.json`,
answers journal, `top_scores.json`):

- **qa-va-05** ("What did the world scientists' 'warning to humanity'
  letters actually call for?") classified **in_scope**,
  ``voices_bias=False`` — so the §3.2 evidence-only include list
  structurally excluded the voices layer, whose Alliance of World
  Scientists entry (link-only, per the licensing rule) is the ONE chunk
  that can answer it; top rerank over evidence chunks was 0.0317 and the
  item declined. **Fixable now — this suite's subject.** The classifier's
  voices definition (#315) covers an individual voice's own
  words/motivations but says nothing about the movement's COLLECTIVE
  statements (letters, campaigns, networks) or §2.5's get-involved shape
  ("What can I watch or join?").
- **qa-va-02** ("What does Packham say drives him?") classified voices,
  ``voices_bias=True``, retrieved the right chunk at rerank 0.94 — and
  declined honestly because the first-party Packham prose deliberately
  describes his role, not his own motivation statements. **Letter/
  editorial-blocked** (authoring motivation prose needs the issue #8
  editorial sign-off): no red.
- **qa-va-04** ("What would a genuine emergency response look like?")
  classified in_scope; neither the evidence corpus nor the voices layer
  holds emergency-response content (top rerank 0.1216; the UNEP owner
  letter is unsent). **Letter-blocked**: no red.
- **qa-sev-09** classified in_scope (correct); retrieval's top-8 missed
  the gold chunk ``esd_tipping_review:4d226c29b08f7570`` under the
  expanded 795-chunk corpus (top score 0.304 came from terminology
  chunks) and the model declined honestly. A ranking/coverage property,
  not a mechanical defect: no red.

Pinned at the #315 suite's three seams — the fix must extend the voices
definition COHERENTLY with #315's science-behind-X exclusion (that suite
stays green untouched):

1. The classifier's committed processing instructions cover the
   movement/collective-statement and get-involved shapes — anchor
   language, characterisation-guard style. RED.
2. The labelled classifier set carries qa-va-05 verbatim as voices plus
   the §2.5 get-involved shape (q58/q59, fixture entries land with this
   suite), cross-checked verbatim against evals/gold/climate_qa.yaml.
3. The routing seam: the voices label drives ``voices_bias=True`` and an
   include list holding evidence AND voices (bias, never exclusion —
   §3.2), so the AWS link-only entry can reach the generation call.
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
CLIMATE_QA_GOLD_PATH = Path(__file__).resolve().parents[2] / "evals" / "gold" / "climate_qa.yaml"

#: qa-va-05's question text VERBATIM (evals/gold/climate_qa.yaml —
#: authored gold-set text, never harvested user input; the q50/qa-tg-01
#: convention).
QA_VA_05_QUESTION = (
    "What did the world scientists' 'warning to humanity' letters actually call for?"
)

#: The §2.5 get-involved shape (fixture entry lands with this suite).
GET_INVOLVED_QUESTION = "How do I get involved with the campaign for an emergency briefing?"


def _instructions() -> str:
    return str(build_query_processing_request("placeholder question")["system"])


def _has(pattern: str) -> bool:
    return re.search(pattern, _instructions(), flags=re.IGNORECASE | re.DOTALL) is not None


# ---------------------------------------------------------------------------
# 1. The classifier prompt covers the movement's collective side of voices
# ---------------------------------------------------------------------------


class TestClassifierPromptMovementAnchors:
    def test_instructions_name_the_collective_statement_shape(self):
        """The voices definition must cover the movement's COLLECTIVE
        voice — what a campaign, network, open letter or warning letter
        said, called for, or asked of governments — not only an
        individual's words/motivations. Run 2's qa-va-05 in_scope
        misroute traces to this gap."""
        assert _has(
            r"(open letters?|warning\w* (letters?|papers?)|petitions?"
            r"|(collective|joint) statements?"
            r"|(campaign\w*|movement|network)s?[^.]{0,120}"
            r"(calls? (for|on)|asks? (for|of)|demands?))"
        ), (
            "the classifier instructions must cover the movement's "
            "collective statements (what a campaign/network/letter calls "
            "for) as voices — qa-va-05's in_scope misroute excluded the "
            "voices layer that holds the answer (issue #339)"
        )

    def test_instructions_name_the_get_involved_shape(self):
        """§2.5's own examples ('Who is calling for an emergency
        briefing?' / 'What can I watch or join?'): questions about
        joining, watching or getting involved with the movement are
        voices questions, answered from the voices layer's campaign
        entries."""
        assert _has(r"(get involved|watch or join|\bjoin\w*\b|take part|participat\w*)"), (
            "the classifier instructions must name the get-involved / "
            "watch-or-join movement shape as voices (DESIGN.md §2.5; "
            "issue #339)"
        )

    def test_science_behind_x_exclusion_survives(self):
        """Coherence guard (the #315 boundary, untouched): assessing
        whether the science/evidence backs a named person's claim stays
        never-voices — extending the class toward collective statements
        must not reopen the qa-tg-01 leak."""
        assert _has(
            r"\b(whether|if|is)\b.{0,120}\b(science|evidence)\b"
            r".{0,120}\b(behind|backs?|supports?)\b"
        )
        assert _has(r"\b(never|not) voices\b")

    def test_instructions_do_not_sweep_evidence_questions_into_voices(self):
        """Non-contradiction guard: nothing may instruct that climate
        action/evidence questions are voices per se — the class covers
        the movement's testimony and participation, never the scientific
        evidence itself."""
        text = _instructions().lower()
        for poison in (
            "every action question is voices",
            "all climate action questions are voices",
            "use voices for scientific evidence",
            "evidence questions are voices",
        ):
            assert poison not in text, poison


# ---------------------------------------------------------------------------
# 2. The labelled set carries qa-va-05 verbatim plus the §2.5 shape
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labelled_queries() -> list[dict]:
    data = yaml.safe_load(LABELLED_SET_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def _entry_for(queries: list[dict], text: str) -> dict:
    matches = [q for q in queries if q["text"] == text]
    assert matches, (
        f"the labelled classifier set must carry this voices_action shape "
        f"verbatim (issue #339): {text!r}"
    )
    return matches[0]


class TestLabelledSetCarriesTheMovementShapes:
    def test_qa_va_05_question_is_labelled_voices(self, labelled_queries):
        entry = _entry_for(labelled_queries, QA_VA_05_QUESTION)
        assert entry["label"] == ScopeClass.VOICES.value, (
            f"{entry['id']}: expected voices, got {entry['label']!r}"
        )
        assert entry.get("edge_case") is True
        assert entry.get("notes") and "qa-va-05" in entry["notes"]

    def test_get_involved_question_is_labelled_voices(self, labelled_queries):
        entry = _entry_for(labelled_queries, GET_INVOLVED_QUESTION)
        assert entry["label"] == ScopeClass.VOICES.value
        assert entry.get("edge_case") is True
        assert entry.get("notes")

    def test_labelled_text_matches_the_gold_set_verbatim(self):
        """Anti-drift cross-check (the #323 convention): this suite's
        verbatim constant matches qa-va-05's gold question text
        character-for-character."""
        data = yaml.safe_load(CLIMATE_QA_GOLD_PATH.read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in data["items"]}
        assert by_id["qa-va-05"]["question"] == QA_VA_05_QUESTION

    def test_the_science_side_regression_case_stays_adversarial(self, labelled_queries):
        """Anti-over-rotation guard: q50 (qa-tg-01 verbatim, the #315
        leak) keeps its never-voices label while the class grows."""
        matches = [q for q in labelled_queries if q["id"] == "q50"]
        assert matches and matches[0]["label"] == ScopeClass.ADVERSARIAL_IN_SCOPE.value


# ---------------------------------------------------------------------------
# 3. The routing seam: voices label -> bias, never exclusion (§3.2)
# ---------------------------------------------------------------------------


def _decision_for(question: str):
    adapter = FakeAdapter()
    adapter.queue(
        "structured",
        StructuredResult(
            value={
                "scope": ScopeClass.VOICES.value,
                "rewritten_query": question,
                "language": "en",
            },
            usage={"input_tokens": 200, "output_tokens": 30},
        ),
    )
    return process_query(adapter, question)


class TestVoicesLabelReachesTheVoicesLayer:
    @pytest.mark.parametrize("question", (QA_VA_05_QUESTION, GET_INVOLVED_QUESTION))
    def test_voices_label_yields_bias_and_inclusive_source_list(self, question):
        """The chain the run-2 misroute broke, healed at the unit seam:
        the voices label drives voices_bias=True and an include list
        holding evidence AND voices — so the AWS link-only entry (the
        licensing-honest answer for qa-va-05) can reach generation."""
        decision = _decision_for(question)
        assert decision.route is Route.RETRIEVAL
        assert decision.voices_bias is True
        include_list = permitted_source_types(decision)
        assert VOICES_SOURCE_TYPE in include_list
        for evidence_type in EVIDENCE_SOURCE_TYPES:
            assert evidence_type in include_list
