"""The landing page model (issue #18, DESIGN §7.1, ADR-022 naming).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_ui_starter.py`` pins the contract.

The 13 starter questions have exactly ONE source of truth —
``service.starter_cache.STARTER_QUESTIONS`` (imported, never duplicated:
the same tuple drives the paused cache's completeness check, so the
landing page and the cached answers cannot drift apart). This module
adds only what §7.1 layers on top:

- :func:`starter_groups` partitions that tuple, in order, into the four
  §7.1 groups (4/3/3/3) under their verbatim headings
  (:data:`STARTER_GROUP_HEADINGS`).
- :func:`starter_submission` — a starter-topic click IS an immediate
  chat submission of the exact question text with empty history
  (§7.1: "tap → the question is asked immediately"). The UI never
  branches on service mode: live answers stream, and while the budget
  is paused the SAME submission is served from the dated starter cache
  — that decision belongs to the service (#22), not here.
- :func:`landing_page_model` — the full landing view: the ADR-022 name
  (long form; short form for narrow chrome), the §7.1 above-the-fold
  tagline, the groups, and the ADR-018 footer
  (``ui.footer.build_page_footer``).
"""

from __future__ import annotations

from dataclasses import dataclass

from service.starter_cache import STARTER_QUESTIONS
from ui.footer import PageFooter

__all__ = [
    "SITE_NAME",
    "SITE_NAME_SHORT",
    "TAGLINE",
    "STARTER_GROUP_HEADINGS",
    "STARTER_QUESTIONS",
    "StarterGroup",
    "ChatSubmission",
    "LandingPage",
    "starter_groups",
    "starter_submission",
    "landing_page_model",
]

#: ADR-022: the accepted name and its short form for narrow UI chrome.
SITE_NAME = "Let's Talk About the Climate Emergency"
SITE_NAME_SHORT = "Let's Talk Climate Emergency"

#: DESIGN §7.1, verbatim — the above-the-fold line.
TAGLINE = "The emergency briefing you haven't had. Ask anything; every answer cites the science."

#: DESIGN §7.1, verbatim — the four starter-topic group headings, in order.
STARTER_GROUP_HEADINGS: tuple[str, ...] = (
    "How bad is it?",
    "Is it really us? / I've heard that…",
    "What happens next?",
    "What can we do?",
)


@dataclass(frozen=True)
class StarterGroup:
    """One §7.1 starter-topic group: a verbatim heading and its questions."""

    heading: str
    questions: tuple[str, ...]


@dataclass(frozen=True)
class ChatSubmission:
    """An immediate chat submission (what a starter click produces)."""

    question: str
    history: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class LandingPage:
    """The landing view model: name, tagline, starter groups, footer."""

    name: str
    name_short: str
    tagline: str
    groups: tuple[StarterGroup, ...]
    footer: PageFooter


def starter_groups() -> tuple[StarterGroup, ...]:
    """The §7.1 groups: STARTER_QUESTIONS partitioned 4/3/3/3, in order."""
    raise NotImplementedError("issue #18 red phase: starter_groups is not implemented yet")


def starter_submission(question: str) -> ChatSubmission:
    """A starter click: the exact question, submitted immediately, no history."""
    raise NotImplementedError("issue #18 red phase: starter_submission is not implemented yet")


def landing_page_model() -> LandingPage:
    """The assembled landing page (ADR-022 names, §7.1 tagline/groups, footer)."""
    raise NotImplementedError("issue #18 red phase: landing_page_model is not implemented yet")
