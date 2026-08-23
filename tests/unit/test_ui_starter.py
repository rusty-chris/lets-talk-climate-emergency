"""Issue #18 RED — the landing page model (DESIGN §7.1, ADR-022).

Pins ``ui.starter``: the four starter-topic groups partition the ONE
source of truth (``service.starter_cache.STARTER_QUESTIONS`` — imported,
never duplicated), a starter click is an immediate verbatim submission,
and the landing page carries the ADR-022 names, the §7.1 tagline, and
the ADR-018 footer.
"""

from __future__ import annotations

import inspect

import ui.starter
from service.starter_cache import STARTER_QUESTIONS
from ui.footer import PageFooter
from ui.starter import (
    STARTER_GROUP_HEADINGS,
    landing_page_model,
    starter_groups,
    starter_submission,
)


class TestStarterGroups:
    def test_groups_partition_the_design_list_verbatim_and_in_order(self) -> None:
        groups = starter_groups()
        assert [group.heading for group in groups] == [
            "How bad is it?",
            "Is it really us? / I've heard that…",
            "What happens next?",
            "What can we do?",
        ]
        assert [len(group.questions) for group in groups] == [4, 3, 3, 3]
        flattened = tuple(q for group in groups for q in group.questions)
        assert flattened == STARTER_QUESTIONS

    def test_headings_constant_matches_the_groups(self) -> None:
        assert tuple(g.heading for g in starter_groups()) == STARTER_GROUP_HEADINGS

    def test_question_strings_are_imported_not_duplicated(self) -> None:
        """The landing page must consume service.starter_cache's tuple;
        a literal copy in ui/starter.py could drift from the cache."""
        source = inspect.getsource(ui.starter)
        for question in STARTER_QUESTIONS:
            assert question not in source, (
                f"starter question duplicated in ui.starter: {question!r}"
            )


class TestStarterClick:
    def test_click_submits_the_exact_question_immediately_with_no_history(self) -> None:
        question = STARTER_QUESTIONS[0]
        submission = starter_submission(question)
        assert submission.question == question
        assert submission.history == ()

    def test_chart_demo_starter_is_a_plain_submission_too(self) -> None:
        """The wow-moment chart question is an ordinary chat submission;
        routing (chart vs retrieval, live vs paused cache) is the
        service's decision, never the UI's."""
        chart_question = "Show me CO₂ and temperature over the last 10,000 years"
        assert chart_question in STARTER_QUESTIONS
        submission = starter_submission(chart_question)
        assert submission.question == chart_question


class TestLandingPage:
    def test_landing_page_carries_adr022_names_and_the_design_tagline(self) -> None:
        page = landing_page_model()
        assert page.name == "Let's Talk About the Climate Emergency"
        assert page.name_short == "Let's Talk Climate Emergency"
        assert page.tagline == (
            "The emergency briefing you haven't had. Ask anything; every answer cites the science."
        )

    def test_landing_page_groups_are_the_starter_groups(self) -> None:
        page = landing_page_model()
        assert page.groups == starter_groups()

    def test_landing_page_includes_the_adr018_footer(self) -> None:
        page = landing_page_model()
        assert isinstance(page.footer, PageFooter)
        assert page.footer.credit.credit_text == "Built by Rusty Data"
        assert page.footer.credit.noncommercial_note
