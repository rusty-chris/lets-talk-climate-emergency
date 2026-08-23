"""The pre-generated starter-answer cache (ADR-015 as amended, DESIGN §7.1/§9).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_service_read_only.py`` pins the
contract.

The read-only paused state is only honest if there is something to
serve: the §7.1 starter-topic answers (and the flagship charts, which
live as stored specs behind ``/chart/<hash>`` permalinks) are generated
AT RELEASE TIME through the real pipeline, cached as a committed/deployed
artifact, and served for $0 while the budget is paused — clearly dated,
so a visitor knows they are reading a cached briefing.

This module pins the cache STRUCTURE and its loading/serving contract.
Generating the cache contents is a release step (the deployment runbook
runs the pipeline once per release with the release corpus) — content
generation is deliberately NOT part of this module's unit contract.

On-disk format (one deployable artifact):
``<starter_cache_dir>/starter_answers.json`` — a JSON object
``{"generated_on": "<ISO date>", "entries": [<entry>, ...]}`` where each
entry carries ``question``, ``answer_text``, ``citations`` (list of
``{chunk_id, attribution_text, cited_text}`` mappings), ``footer`` (the
§3.5 verification note + corpus vintage the answer was generated
against), and optional ``chart_spec_hash`` (the permalink of the chart
the answer demonstrates, e.g. the flagship 10k-year chart).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "STARTER_QUESTIONS",
    "STARTER_CACHE_FILENAME",
    "StarterCacheError",
    "StarterAnswerEntry",
    "StarterCache",
    "load_starter_cache",
]

#: The §7.1 starter questions, verbatim — the four clickable groups on
#: the landing page. This tuple is the single source of truth the cache
#: completeness check and #18's landing page both consume; drift from
#: DESIGN §7.1 fails tests.
STARTER_QUESTIONS: tuple[str, ...] = (
    # How bad is it?
    "Why are scientists calling this an emergency?",
    "How much has the planet warmed — and how fast is it accelerating?",
    "What happens at 1.5°C, 2°C, 3°C?",
    "Show me CO₂ and temperature over the last 10,000 years",
    # Is it really us? / I've heard that…
    "Hasn't the climate always changed?",
    "How sure are scientists it's human-caused?",
    "Didn't warming pause?",
    # What happens next?
    "What are tipping points and how close are we?",
    "How will this affect food, water and health?",
    "Are we on track to meet the Paris targets?",
    # What can we do?
    "Is it too late?",
    "What would an emergency response actually look like?",
    "Who is speaking up, and how do I get involved?",
)

STARTER_CACHE_FILENAME = "starter_answers.json"


class StarterCacheError(Exception):
    """The starter cache is missing, incomplete, undated, or malformed.

    The read-only state cannot fail closed onto a cache that is not
    there — an invalid cache is a LOUD deploy-time failure, naming every
    missing/invalid question, never a silent empty paused state.
    """


@dataclass(frozen=True)
class StarterAnswerEntry:
    """One cached starter-topic answer (release-time pipeline output)."""

    question: str
    answer_text: str
    citations: tuple[Mapping[str, Any], ...]
    footer: str
    generated_on: str
    chart_spec_hash: str | None = None


class StarterCache:
    """The loaded, validated cache; lookup is by exact starter question."""

    def __init__(self, generated_on: str, entries: tuple[StarterAnswerEntry, ...]) -> None:
        self.generated_on = generated_on
        self.entries = entries

    def lookup(self, question: str) -> StarterAnswerEntry | None:
        """The cached entry for ``question`` (whitespace-normalised exact
        match against :data:`STARTER_QUESTIONS`), or None for any other
        question — the cache never fuzzy-matches arbitrary user text."""
        key = _normalise_question(question)
        for entry in self.entries:
            if _normalise_question(entry.question) == key:
                return entry
        return None


def _normalise_question(question: str) -> str:
    """Collapse surrounding/interior whitespace for exact-match lookup."""
    return " ".join(question.split())


def load_starter_cache(cache_dir: Path) -> StarterCache:
    """Load + validate ``<cache_dir>/starter_answers.json``.

    Contract (pinned): raises :class:`StarterCacheError` — naming every
    offender — when the file is absent; when ``generated_on`` is missing
    or not an ISO date; when any :data:`STARTER_QUESTIONS` question has
    no entry; or when any entry lacks a non-empty ``answer_text``,
    ``citations``, or ``footer``. A valid file yields a
    :class:`StarterCache` whose entries preserve the stored fields.
    """
    path = Path(cache_dir) / STARTER_CACHE_FILENAME
    if not path.is_file():
        raise StarterCacheError(
            f"starter cache file is absent: {path} — the read-only paused "
            "state cannot fail closed onto a cache that is not there"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StarterCacheError(f"starter cache at {path} is unreadable/malformed: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise StarterCacheError(f"starter cache at {path} is not a JSON object")

    problems: list[str] = []

    generated_on = raw.get("generated_on")
    if not isinstance(generated_on, str) or not generated_on.strip():
        problems.append("generated_on is missing")
        generated_on = ""
    else:
        try:
            date.fromisoformat(generated_on)
        except ValueError:
            problems.append(f"generated_on {generated_on!r} is not an ISO date")

    entries_by_question: dict[str, Mapping[str, Any]] = {}
    for item in raw.get("entries", []) or []:
        if isinstance(item, Mapping) and isinstance(item.get("question"), str):
            entries_by_question[_normalise_question(item["question"])] = item

    entries: list[StarterAnswerEntry] = []
    for question in STARTER_QUESTIONS:
        item = entries_by_question.get(_normalise_question(question))
        if item is None:
            problems.append(f"no entry for starter question: {question!r}")
            continue
        answer_text = item.get("answer_text")
        citations = item.get("citations")
        footer = item.get("footer")
        if not isinstance(answer_text, str) or not answer_text.strip():
            problems.append(f"entry for {question!r} lacks a non-empty answer_text")
        if not citations:
            problems.append(f"entry for {question!r} lacks citations")
        if not isinstance(footer, str) or not footer.strip():
            problems.append(f"entry for {question!r} lacks a non-empty footer")
        entries.append(
            StarterAnswerEntry(
                question=question,
                answer_text=answer_text if isinstance(answer_text, str) else "",
                citations=tuple(citations) if citations else (),
                footer=footer if isinstance(footer, str) else "",
                generated_on=str(generated_on),
                chart_spec_hash=item.get("chart_spec_hash"),
            )
        )

    if problems:
        raise StarterCacheError(f"starter cache at {path} is invalid — " + "; ".join(problems))
    return StarterCache(generated_on=str(generated_on), entries=tuple(entries))
