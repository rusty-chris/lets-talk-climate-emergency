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
from ui.footer import PageFooter, build_page_footer

__all__ = [
    "SITE_NAME",
    "SITE_NAME_SHORT",
    "TAGLINE",
    "STARTER_GROUP_HEADINGS",
    "STARTER_QUESTIONS",
    "StarterGroup",
    "ChatSubmission",
    "ChatInputModel",
    "LandingPage",
    "starter_groups",
    "starter_submission",
    "free_text_submission",
    "chat_input_model",
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


#: The §7.1 partition of the ONE source of truth into its four groups,
#: in order (4/3/3/3). Sizes only — the questions themselves are never
#: duplicated here (they live in ``service.starter_cache``).
_GROUP_SIZES: tuple[int, ...] = (4, 3, 3, 3)


def starter_groups() -> tuple[StarterGroup, ...]:
    """The §7.1 groups: STARTER_QUESTIONS partitioned 4/3/3/3, in order."""
    groups: list[StarterGroup] = []
    start = 0
    for heading, size in zip(STARTER_GROUP_HEADINGS, _GROUP_SIZES, strict=True):
        groups.append(
            StarterGroup(heading=heading, questions=STARTER_QUESTIONS[start : start + size])
        )
        start += size
    return tuple(groups)


def starter_submission(question: str) -> ChatSubmission:
    """A starter click: the exact question, submitted immediately, no history."""
    # §7.1: tap -> the question is asked immediately, verbatim, with a clean
    # history. Whether it streams live or is served from the paused cache is
    # the service's (#22) decision, never the UI's — no mode branch here.
    return ChatSubmission(question=question, history=())


@dataclass(frozen=True)
class ChatInputModel:
    """The free-text question input's pure model (review finding #225).

    RED-phase contract stub (review-18 fix wave): the §7.1 tagline is
    "Ask anything" — the chat surface needs a typed-question path, and
    the privacy contract requires users to see the one-line logging
    disclosure BEFORE they type personal things, not only under a
    finished answer. ``disclosure`` is the pinned
    ``service.exchange_log.LOGGING_DISCLOSURE`` line; ``placeholder``
    is the input's invitation text.
    """

    placeholder: str
    disclosure: str


def free_text_submission(text: str) -> ChatSubmission | None:
    """A typed question -> the same immediate submission a starter click makes.

    RED-phase contract stub (review-18 fix wave); the failing tests in
    ``tests/unit/test_ui_starter.py::TestFreeTextSubmission`` pin the
    contract: outer whitespace is stripped and NOTHING else changes (the
    service owns query processing — inner whitespace and non-ASCII pass
    byte-for-byte); a blank or whitespace-only input yields ``None`` (the
    shell must never POST an empty question); history is always empty
    (multi-turn memory is Phase-2 deferred, DESIGN §10).
    """
    raise NotImplementedError


def chat_input_model() -> ChatInputModel:
    """The pure model for the free-text input widget (finding #225).

    RED-phase contract stub; carries the logging disclosure so it is
    visible AT the input, before anything is typed.
    """
    raise NotImplementedError


def landing_page_model() -> LandingPage:
    """The assembled landing page (ADR-022 names, §7.1 tagline/groups, footer)."""
    return LandingPage(
        name=SITE_NAME,
        name_short=SITE_NAME_SHORT,
        tagline=TAGLINE,
        groups=starter_groups(),
        footer=build_page_footer(),
    )
