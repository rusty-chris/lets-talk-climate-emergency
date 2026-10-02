"""Release-results CURRENCY: the config fingerprint and the staleness
waiver (issue #427, detectability half).

The defect this module exists to make impossible: ``evals/results.json``
was published on 2026-09-12 from a single ``claude-haiku-4-5`` arm, and
production then switched live generation to Opus, rewrote the generation
system prompt five times, raised generation top-k 8->12 and re-pinned the
corpus — while ``/about`` kept publishing those numbers as the project's
transparency credential. IMPLEMENTATION.md §6 makes the full eval suite a
per-release / per-corpus-version gate, and nothing since has passed it.
The #249/#353 boot check verifies ``RESULTS.md``'s PRESENCE; nothing
verified its CURRENCY, so nothing could notice.

Contract:

- :func:`current_config_fingerprint` DERIVES the live side from the repo
  — nothing hand-copied, because a hand-copied constant is the defect:
  the prompt digest from the committed artifact
  (:func:`rag.generation.system_prompt_sha256`), ``top_k`` from
  :data:`rag.retrieval.GENERATION_TOP_K`, the live-servable model set
  from ``rag.generation``'s constants, and ``corpus_version`` from
  :func:`ingestion.manifest.corpus_version` (a content address of the
  manifest's pinned set — see that function for why it is derived and
  not a declared release label).
- :func:`fingerprint_mismatches` and :func:`check_results_currency` are
  PURE over mappings (IMPLEMENTATION.md §4.4): the caller supplies both
  sides and the waiver, so the comparison is synthetically testable and
  reaches no network, clock or filesystem.
- :func:`check_results_currency` REFUSES on any drift, naming every
  differing field — unless a recorded waiver acknowledges exactly the
  differing set, restates both sides of each field truthfully, and
  carries an ISO date plus an attribution and a reason.

Why a waiver at all: the committed results genuinely ARE stale, the
re-run that fixes that is real spend needing owner approval (#427 half
(ii)), and a test that stayed red until then would block every unrelated
PR. The waiver converts "silently stale" into "explicitly, datedly
acknowledged as stale" — which is the whole point. It is deliberately
NOT a boolean override: there is no ``waived: true`` to set. Every
waiver restates the two values it accepted, so it expires MECHANICALLY —
further drift, a re-run that makes the fingerprint match, or an
acknowledgment that no longer differs all refuse. The only way to make a
leftover waiver stop refusing is to delete it.

Scope note: this module detects and refuses. The live Opus-arm battery
re-run and any ``/about`` wording about the staleness are #427's other,
paid half and are not implemented here; the standing invariant in
``tests/unit/test_release_gates.py`` is phrased "current OR waived"
precisely so it keeps passing once the re-run republishes the artefacts
and the waiver file is deleted.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.manifest import corpus_version
from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    system_prompt_sha256,
)
from rag.retrieval import GENERATION_TOP_K

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The committed staleness waiver, when one exists. Absence is the
#: NORMAL state (a current artefact needs no waiver), which is why
#: :func:`load_staleness_waiver` returns None rather than raising.
STALENESS_WAIVER_PATH = REPO_ROOT / "evals" / "results-staleness-waiver.json"

#: The key ``results.json`` stamps. Part of the published artefact's
#: shape, so it is a named constant on both sides of the contract.
RESULTS_FINGERPRINT_KEY = "config_fingerprint"

#: The four fields of the fingerprint (#427 acceptance criteria): the
#: corpus the run measured, the prompt vintage it ran, the model set it
#: covered, and the retrieval depth it saw. Each one is a change that
#: production actually shipped unmeasured.
FINGERPRINT_FIELDS = ("corpus_version", "prompt_sha256", "generation_models", "top_k")

#: Waiver fields that must be present and non-empty. ``accepted_on``
#: additionally has to parse as an ISO date (see :func:`_waiver_problems`):
#: "WHEN was this accepted" is the one thing a dated acknowledgment
#: cannot be vague about.
_WAIVER_REQUIRED_TEXT_FIELDS = ("accepted_by", "reason")

_WAIVER_FIELDS_KEY = "acknowledged_mismatches"


class StaleResultsError(AssertionError):
    """The published release-gate results do not describe the live
    configuration, and no valid waiver acknowledges the difference.

    An ``AssertionError`` subclass because the only consumer today is
    the standing unit-tier invariant, and a release-gate refusal should
    read as a failed assertion about the repo, not as a bug.
    """


@dataclass(frozen=True)
class CurrencyVerdict:
    """The outcome of a currency check — LOUD even when it passes.

    A waived pass still reports ``fingerprint_matches=False``, the
    fields that differ and the acceptance date, so no caller can mistake
    "acknowledged as stale" for "current".
    """

    fingerprint_matches: bool
    waived: bool
    mismatched_fields: tuple[str, ...] = ()
    waiver_accepted_on: str | None = None


def current_config_fingerprint() -> dict[str, Any]:
    """The fingerprint of the configuration the repo would run TODAY.

    Every field is derived, never declared. Reads the committed prompt
    artifact and corpus manifest; touches no network, no environment and
    no clock, so the comparison is reproducible on any checkout and in
    CI (where ``CLIMATE_CHAT_CORPUS_VERSION`` is a smoke placeholder and
    must not leak into a release-gate verdict).
    """
    return {
        "corpus_version": corpus_version(),
        "prompt_sha256": system_prompt_sha256(),
        # The pair production can actually SERVE: the Haiku default plus
        # the best-mode Opus model service/app.py dispatches to. The
        # #427 defect is exactly that the published run covered only the
        # first of these, so the fingerprint records the set, not the
        # default alone.
        "generation_models": sorted({GENERATION_MODEL_DEFAULT, OPUS_BEST_MODEL}),
        "top_k": GENERATION_TOP_K,
    }


def _comparable(field: str, value: Any) -> Any:
    """One fingerprint value in a form two sides can be compared in.

    ``generation_models`` is a SET of models that happens to travel as a
    JSON list: re-ordering the arms is not a configuration change, so it
    must not read as drift. Everything else compares as published.
    """
    if field == "generation_models" and isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(sorted(str(model) for model in value))
    return value


def fingerprint_mismatches(
    published: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> tuple[str, ...]:
    """The fingerprint fields on which ``published`` differs from
    ``current``, in :data:`FINGERPRINT_FIELDS` order.

    Fail-closed: a missing fingerprint (``None``) counts as drift on
    EVERY field — today's unstamped artefact is maximal staleness, never
    a free pass — and so does a field absent from either side.
    """
    if published is None:
        return FINGERPRINT_FIELDS
    differing = []
    for field in FINGERPRINT_FIELDS:
        if field not in published or field not in current:
            differing.append(field)
        elif _comparable(field, published[field]) != _comparable(field, current[field]):
            differing.append(field)
    return tuple(differing)


def load_staleness_waiver(path: Path = STALENESS_WAIVER_PATH) -> dict[str, Any] | None:
    """The committed staleness waiver, or None when there is no file.

    Absence is not an error: a current artefact carries no waiver, and a
    re-run's final step is DELETING this file. Returning None keeps the
    standing invariant free of try/except plumbing. A file that exists
    but is unparseable still raises — a corrupt waiver is a real problem
    and must not read as "no waiver recorded".
    """
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _describe(field: str, published: Any, current: Any) -> str:
    return f"{field}: published {published!r} != current {current!r}"


def _waiver_problems(
    waiver: Mapping[str, Any],
    *,
    published: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    mismatched: tuple[str, ...],
) -> list[str]:
    """Every reason this waiver does not authorise this drift.

    All reasons are collected, not just the first (the manifest-loader
    convention): whoever has to fix the waiver should see the whole gap
    in one refusal.
    """
    problems: list[str] = []

    accepted_on = waiver.get("accepted_on")
    if not _is_iso_date(accepted_on):
        problems.append(
            f"accepted_on must be an ISO date (YYYY-MM-DD), got {accepted_on!r} — an "
            "undated acknowledgment records no WHEN and is not a waiver"
        )
    for field in _WAIVER_REQUIRED_TEXT_FIELDS:
        value = waiver.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"waiver {field!r} must be a non-empty string, got {value!r}")

    acknowledged = waiver.get(_WAIVER_FIELDS_KEY)
    if not isinstance(acknowledged, Mapping):
        problems.append(
            f"waiver {_WAIVER_FIELDS_KEY!r} must be a mapping of field -> "
            "{published, current}; there is no blanket override"
        )
        return problems

    if published is None:
        # Stamping the artefact costs nothing and needs no re-run, so a
        # missing fingerprint is never waivable: it is a gap in the
        # RECORD, not an accepted difference between two configurations.
        problems.append(
            f"results carry no {RESULTS_FINGERPRINT_KEY!r}, so no waiver can restate the "
            "published side — stamp the fingerprint instead of waiving it"
        )
        return problems

    unstamped = [field for field in FINGERPRINT_FIELDS if field not in published]
    if unstamped:
        # Same rule as a wholly missing stamp, per field: a gap in the
        # record is not a waivable difference between configurations.
        problems.append(
            f"results' {RESULTS_FINGERPRINT_KEY!r} omits "
            + ", ".join(unstamped)
            + " — stamp the missing field(s) instead of waiving them"
        )

    unacknowledged = [field for field in mismatched if field not in acknowledged]
    if unacknowledged:
        problems.append(
            "waiver does not acknowledge "
            + ", ".join(unacknowledged)
            + " — a partial acknowledgment waives only what it names"
        )
    residue = [field for field in acknowledged if field not in mismatched]
    if residue:
        # Self-cleaning (the leftover-after-a-re-run case): acknowledging
        # a field that no longer differs means the waiver outlived the
        # staleness it described. Delete it.
        problems.append(
            "waiver acknowledges "
            + ", ".join(sorted(residue))
            + " which no longer differ — the waiver has expired and must be deleted"
        )
    if not acknowledged:
        problems.append(
            f"waiver acknowledges no fields — {_WAIVER_FIELDS_KEY!r} is empty, and there is "
            "no 'accept whatever differs' escape hatch"
        )

    for field in mismatched:
        if field in unstamped:
            continue  # already reported as an unstamped gap
        entry = acknowledged.get(field)
        if entry is None:
            continue  # already reported as unacknowledged
        if not isinstance(entry, Mapping):
            problems.append(
                f"waiver entry for {field!r} must be a mapping with 'published' and "
                f"'current', got {entry!r}"
            )
            continue
        # BOTH sides are restated and BOTH are checked. Restating the
        # current value is what makes the waiver self-expiring on further
        # drift; restating the published value is what stops a waiver
        # being written against results that do not say what it claims.
        for side, actual in (("published", published[field]), ("current", current[field])):
            if _comparable(field, entry.get(side)) != _comparable(field, actual):
                problems.append(
                    f"waiver misstates the {side} {field!r}: waiver says "
                    f"{entry.get(side)!r}, artefact/repo says {actual!r}"
                )
    return problems


def check_results_currency(
    results: Mapping[str, Any],
    *,
    current: Mapping[str, Any],
    waiver: Mapping[str, Any] | None = None,
) -> CurrencyVerdict:
    """Refuse unless the published results are current, or their
    staleness is validly waived.

    ``current`` is passed in rather than computed so this stays pure
    over mappings (§4.4) and every drift can be staged synthetically;
    production callers pass :func:`current_config_fingerprint`.

    Raises :class:`StaleResultsError` naming every differing field (and,
    for an invalid waiver, every reason it does not authorise the drift).
    """
    published = results.get(RESULTS_FINGERPRINT_KEY)
    if published is not None and not isinstance(published, Mapping):
        raise StaleResultsError(
            f"{RESULTS_FINGERPRINT_KEY!r} must be a mapping of "
            f"{', '.join(FINGERPRINT_FIELDS)}, got {published!r}"
        )
    mismatched = fingerprint_mismatches(published, current)

    if waiver is None:
        if not mismatched:
            return CurrencyVerdict(fingerprint_matches=True, waived=False)
        missing_stamp = (
            f"evals/results.json carries no {RESULTS_FINGERPRINT_KEY!r} — the published "
            "gate numbers are tied to no configuration at all. "
            if published is None
            else ""
        )
        details = [
            _describe(
                field,
                None if published is None else published.get(field),
                current.get(field),
            )
            for field in mismatched
        ]
        raise StaleResultsError(
            f"{missing_stamp}The published release-gate results do not describe the live "
            f"configuration — {len(mismatched)} field(s) differ:\n  "
            + "\n  ".join(details)
            + "\nRe-run the release battery and republish, or record a dated waiver at "
            f"{STALENESS_WAIVER_PATH.name} acknowledging exactly these fields (#427)."
        )

    problems = _waiver_problems(waiver, published=published, current=current, mismatched=mismatched)
    if problems:
        raise StaleResultsError(
            "the recorded staleness waiver does not authorise the published results:\n  "
            + "\n  ".join(problems)
        )

    return CurrencyVerdict(
        fingerprint_matches=False,
        waived=True,
        mismatched_fields=mismatched,
        waiver_accepted_on=waiver["accepted_on"],
    )
