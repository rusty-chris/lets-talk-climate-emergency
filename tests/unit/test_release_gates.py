"""Issue #427 red phase (detectability half ONLY): the published
release-gate results must carry a config fingerprint, and a fingerprint
that no longer describes the live repo configuration must fail LOUDLY —
unless a dated, explicit staleness waiver is recorded.

The defect: evals/results.json says generated_on 2026-09-12 with a
single claude-haiku-4-5 arm, and evals/RESULTS.md states "Production
model: claude-haiku-4-5" — while production has since switched live
generation to Opus (an arm never in the bake-off), rewritten the
generation system prompt five times, raised generation top-k 8→12 and
signed off the paleo corpus. /about publishes those numbers as the
project's transparency credential; the #249/#353 boot check verifies
RESULTS.md's PRESENCE, never its CURRENCY, and IMPLEMENTATION.md §6
makes the full suite a per-release gate that nothing since 2026-09-12
has actually passed.

Contract pinned here (new module ``evals.fingerprint`` — pure over
mappings per IMPLEMENTATION.md §4.4, no network, no judge):

- ``results.json`` stamps ``config_fingerprint`` with
  {corpus_version, prompt_sha256, generation_models, top_k}.
- ``current_config_fingerprint()`` derives the LIVE side mechanically
  from the repo — the prompt hash from the committed prompt artifact
  (``rag.generation.system_prompt_sha256``, the shared primitive #469's
  journal headers and #481's stamping also want), top_k from
  ``rag.retrieval.GENERATION_TOP_K``, the live-servable model set from
  rag.generation's constants. Nothing is hand-copied.
- ``check_results_currency`` refuses on any mismatch, naming every
  differing field — UNLESS a recorded waiver acknowledges exactly the
  differing fields, restates both sides of each, and carries an ISO
  date + non-empty reason. A waiver is mechanically self-expiring: any
  further config drift, any acknowledged field that no longer differs,
  or a leftover waiver after a re-run all refuse. There is no boolean
  override — the waiver cannot be satisfied by editing a number.

DELIBERATELY NOT HERE (the paid half of #427): the live Opus-arm
battery re-run and any new /about wording — real spend + owner
sign-off, never a unit tier concern. The standing invariant below is
phrased "current OR waived" precisely so it keeps passing once the
re-run republishes the artefacts.

Red phase: ``evals.fingerprint`` and ``system_prompt_sha256`` do not
exist yet; the schema assertions fail against the REAL committed
results.json, which carries no fingerprint at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    OPUS_BEST_MODEL,
    SYSTEM_PROMPT_PATH,
)
from rag.retrieval import GENERATION_TOP_K

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_RESULTS_PATH = REPO_ROOT / "evals" / "results.json"

#: The four fields the fingerprint must carry (issue #427 acceptance
#: criteria, verbatim). Pinned as data here so the parametrized
#: mismatch tests and the schema test cannot drift apart.
FINGERPRINT_FIELDS = ("corpus_version", "prompt_sha256", "generation_models", "top_k")

#: The key the published results stamp (test below reads the RAW
#: committed JSON through it, so the key name is part of the contract).
RESULTS_FINGERPRINT_KEY = "config_fingerprint"


def _fingerprint_module():
    """Import the contract-under-test lazily so every test reds
    individually with ModuleNotFoundError instead of one collection
    error hiding the per-test intent."""
    from evals import fingerprint

    return fingerprint


def _load_published_results() -> dict:
    return json.loads(PUBLISHED_RESULTS_PATH.read_text(encoding="utf-8"))


# --- Synthetic fingerprints (pure-mapping tier, §4.4) -----------------

SYNTHETIC_PROMPT_SHA = hashlib.sha256(b"synthetic prompt vintage A\n").hexdigest()
EDITED_PROMPT_SHA = hashlib.sha256(b"synthetic prompt vintage B\n").hexdigest()


def _fingerprint(**overrides) -> dict:
    """A synthetic fingerprint in the published shape; overrides mutate
    one side to stage a mismatch."""
    base = {
        "corpus_version": "v1.1.0-launch-2026-09-13",
        "prompt_sha256": SYNTHETIC_PROMPT_SHA,
        "generation_models": ["claude-haiku-4-5", "claude-opus-4-8"],
        "top_k": 12,
    }
    base.update(overrides)
    return base


def _results_payload(fingerprint: dict | None) -> dict:
    """A minimal results.json-shaped payload; None omits the stamp
    (today's real artefact shape)."""
    payload = {
        "schema_version": 1,
        "generated_on": "2026-09-12",
        "release_verdict": "passed",
        "selected_model": "claude-haiku-4-5",
        "arms": [{"model": "claude-haiku-4-5", "gates": []}],
    }
    if fingerprint is not None:
        payload[RESULTS_FINGERPRINT_KEY] = fingerprint
    return payload


def _waiver(
    acknowledged: dict[str, tuple[object, object]],
    accepted_on: str | None = "2026-10-02",
    **overrides,
) -> dict:
    """A staleness waiver in the pinned shape: dated, attributed,
    reasoned, and restating BOTH sides of every differing field."""
    waiver: dict = {
        "accepted_by": "orchestrator (issue #427 triage)",
        "reason": (
            "Opus went live before a full battery re-run; the re-run is "
            "real spend awaiting owner approval (#427 half (ii))."
        ),
        "acknowledged_mismatches": {
            field: {"published": published, "current": current}
            for field, (published, current) in acknowledged.items()
        },
    }
    if accepted_on is not None:
        waiver["accepted_on"] = accepted_on
    waiver.update(overrides)
    return waiver


# --- A. The published artefact carries the fingerprint ----------------


def test_published_results_carry_config_fingerprint():
    """#427 acceptance criterion 1, against the REAL committed
    evals/results.json: it stamps config_fingerprint with all four
    fields, well-typed. Red today — the artefact carries no fingerprint
    at all, which is exactly why nothing could notice the staleness."""
    payload = _load_published_results()
    assert RESULTS_FINGERPRINT_KEY in payload, (
        f"evals/results.json carries no {RESULTS_FINGERPRINT_KEY!r} — the "
        "published gate numbers cannot be tied to any configuration"
    )
    stamp = payload[RESULTS_FINGERPRINT_KEY]
    for required in FINGERPRINT_FIELDS:
        assert required in stamp, f"fingerprint is missing {required!r}"
    assert isinstance(stamp["corpus_version"], str) and stamp["corpus_version"]
    assert isinstance(stamp["prompt_sha256"], str)
    assert len(stamp["prompt_sha256"]) == 64
    int(stamp["prompt_sha256"], 16)  # hex digest, not prose
    assert isinstance(stamp["generation_models"], list) and stamp["generation_models"]
    assert all(isinstance(model, str) for model in stamp["generation_models"])
    assert isinstance(stamp["top_k"], int) and stamp["top_k"] > 0


# --- D. The prompt hash is computed, never hand-copied ----------------


def test_prompt_hash_is_computed_from_the_live_prompt_file(tmp_path):
    """The shared primitive (#427 here, #469 journal headers, #481
    stamping): ``rag.generation.system_prompt_sha256`` hashes the
    committed prompt artifact's BYTES, defaulting to SYSTEM_PROMPT_PATH,
    so a prompt edit changes the digest mechanically — no stored
    constant to forget to update."""
    from rag.generation import system_prompt_sha256

    expected = hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()
    assert system_prompt_sha256() == expected

    # An edit changes the digest — including a re-read of the SAME path,
    # so no process-lifetime cache can serve a pre-edit hash.
    prompt = tmp_path / "generation_system_prompt.md"
    prompt.write_text("Rule one: cite the evidence.\n", encoding="utf-8")
    before = system_prompt_sha256(prompt)
    prompt.write_text("Rule one: state the conclusion.\n", encoding="utf-8")
    after = system_prompt_sha256(prompt)
    assert before != after
    assert after == hashlib.sha256(prompt.read_bytes()).hexdigest()


def test_current_config_fingerprint_is_derived_from_the_repo():
    """The LIVE side of the comparison reads the repo's sources of
    truth: prompt hash from the committed artifact, top_k from
    rag.retrieval, and the live-servable model set (the Haiku default
    plus the best-mode Opus model — the pair production can actually
    serve, per service/app.py's best-mode dispatch). corpus_version's
    exact source is the implementer's choice; it must be a non-empty
    string, never absent."""
    fingerprint = _fingerprint_module()

    current = fingerprint.current_config_fingerprint()
    for required in FINGERPRINT_FIELDS:
        assert required in current, f"live fingerprint is missing {required!r}"
    expected_prompt = hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()
    assert current["prompt_sha256"] == expected_prompt
    assert current["top_k"] == GENERATION_TOP_K
    assert set(current["generation_models"]) == {
        GENERATION_MODEL_DEFAULT,
        OPUS_BEST_MODEL,
    }
    assert isinstance(current["corpus_version"], str) and current["corpus_version"]


# --- B. Any single-field drift trips the check ------------------------

#: One staged drift per fingerprint field — each is the REAL change
#: production shipped unmeasured (model switch, 5 prompt rewrites,
#: top-k 8→12, paleo corpus sign-off).
FIELD_DRIFTS = {
    "corpus_version": "v1.2.0-paleo-2026-09-28",
    "prompt_sha256": EDITED_PROMPT_SHA,
    "generation_models": ["claude-haiku-4-5"],  # Opus live, never an arm
    "top_k": 8,  # the published run's vintage
}


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_published_results_fingerprint_matches_current_config(field):
    """#427 TDD plan, verbatim: mutate any one of model set / prompt
    hash / top-k / corpus version and the currency check goes red,
    naming exactly the field that drifted — with no waiver recorded,
    staleness is a refusal, not a note."""
    fingerprint = _fingerprint_module()

    current = _fingerprint()
    published = _fingerprint(**{field: FIELD_DRIFTS[field]})
    assert published != current  # the staging itself must stage a drift

    assert fingerprint.fingerprint_mismatches(published, current) == (field,)

    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(published), current=current, waiver=None
        )
    assert field in str(excinfo.value)


def test_matching_fingerprint_passes_without_waiver():
    """The post-re-run steady state: fingerprint matches, no waiver on
    disk — the check passes and reports itself unwaived."""
    fingerprint = _fingerprint_module()

    stamp = _fingerprint()
    verdict = fingerprint.check_results_currency(
        _results_payload(stamp), current=_fingerprint(), waiver=None
    )
    assert verdict.fingerprint_matches is True
    assert verdict.waived is False
    assert verdict.mismatched_fields == ()


def test_results_without_fingerprint_are_stale_on_every_field():
    """Today's real artefact shape — no stamp at all — is maximal
    staleness, never a free pass: every field counts mismatched and the
    refusal names the missing key."""
    fingerprint = _fingerprint_module()

    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(None), current=_fingerprint(), waiver=None
        )
    message = str(excinfo.value)
    assert RESULTS_FINGERPRINT_KEY in message
    for field in FINGERPRINT_FIELDS:
        assert field in message


# --- C. The waiver: loud, dated, explicit, self-expiring ---------------


def _staged_mismatch() -> tuple[dict, dict, dict[str, tuple[object, object]]]:
    """A two-field drift (the real one: Opus live + prompt rewritten)
    with the acknowledgment entries a VALID waiver must restate."""
    current = _fingerprint(
        generation_models=["claude-haiku-4-5", "claude-opus-4-8"],
        prompt_sha256=EDITED_PROMPT_SHA,
    )
    published = _fingerprint(
        generation_models=["claude-haiku-4-5"],
        prompt_sha256=SYNTHETIC_PROMPT_SHA,
    )
    acknowledged = {
        "generation_models": (["claude-haiku-4-5"], ["claude-haiku-4-5", "claude-opus-4-8"]),
        "prompt_sha256": (SYNTHETIC_PROMPT_SHA, EDITED_PROMPT_SHA),
    }
    return published, current, acknowledged


def test_dated_waiver_naming_the_mismatches_passes_loudly():
    """#427's design constraint: the committed results genuinely ARE
    stale, and a test that stays red until a paid re-run would block
    every unrelated PR. A recorded waiver makes the check pass — but
    LOUDLY: the verdict still reports the mismatch, the waived fields
    and the acceptance date, so nothing downstream can mistake waived
    for current."""
    fingerprint = _fingerprint_module()

    published, current, acknowledged = _staged_mismatch()
    verdict = fingerprint.check_results_currency(
        _results_payload(published),
        current=current,
        waiver=_waiver(acknowledged),
    )
    assert verdict.fingerprint_matches is False
    assert verdict.waived is True
    assert set(verdict.mismatched_fields) == set(acknowledged)
    assert verdict.waiver_accepted_on == "2026-10-02"


@pytest.mark.parametrize("accepted_on", [None, "", "soonish"], ids=["absent", "empty", "non-iso"])
def test_undated_waiver_does_not_satisfy(accepted_on):
    """An undated (or vaguely-dated) waiver is not a waiver: accepted_on
    must be a real ISO date, so the record always says WHEN the
    staleness was accepted."""
    fingerprint = _fingerprint_module()

    published, current, acknowledged = _staged_mismatch()
    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(published),
            current=current,
            waiver=_waiver(acknowledged, accepted_on=accepted_on),
        )
    assert "accepted_on" in str(excinfo.value)


def test_fieldless_waiver_does_not_satisfy():
    """A waiver that acknowledges nothing waives nothing — there is no
    blanket 'accept whatever differs' escape hatch."""
    fingerprint = _fingerprint_module()

    published, current, _ = _staged_mismatch()
    with pytest.raises(fingerprint.StaleResultsError):
        fingerprint.check_results_currency(
            _results_payload(published),
            current=current,
            waiver=_waiver({}),
        )


def test_waiver_must_acknowledge_every_differing_field():
    """Partial acknowledgment refuses, naming the unacknowledged drift:
    waiving the model switch does not quietly also waive the prompt
    rewrites."""
    fingerprint = _fingerprint_module()

    published, current, acknowledged = _staged_mismatch()
    only_models = {"generation_models": acknowledged["generation_models"]}
    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(published),
            current=current,
            waiver=_waiver(only_models),
        )
    assert "prompt_sha256" in str(excinfo.value)


def test_waiver_expires_mechanically_on_further_drift():
    """Each acknowledgment restates the CURRENT value it accepted; if
    the config drifts again after the waiver was recorded (here: a
    sixth prompt rewrite), the restated value no longer matches and the
    waiver expires — it cannot ride along indefinitely."""
    fingerprint = _fingerprint_module()

    published, current, acknowledged = _staged_mismatch()
    current = dict(current, prompt_sha256=hashlib.sha256(b"vintage C\n").hexdigest())
    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(published),
            current=current,
            waiver=_waiver(acknowledged),  # restates vintage B, now wrong
        )
    assert "prompt_sha256" in str(excinfo.value)


def test_waiver_must_restate_the_published_side_truthfully():
    """The published side of each acknowledgment must equal what the
    artefact actually says — a waiver cannot be written against
    imaginary results."""
    fingerprint = _fingerprint_module()

    published, current, acknowledged = _staged_mismatch()
    acknowledged = dict(
        acknowledged,
        generation_models=(["claude-opus-4-8"], ["claude-haiku-4-5", "claude-opus-4-8"]),
    )
    with pytest.raises(fingerprint.StaleResultsError) as excinfo:
        fingerprint.check_results_currency(
            _results_payload(published),
            current=current,
            waiver=_waiver(acknowledged),
        )
    assert "generation_models" in str(excinfo.value)


def test_leftover_waiver_after_a_rerun_refuses():
    """Self-cleaning: once a re-run republishes a matching fingerprint,
    a waiver still on disk is stale residue and REFUSES — acknowledging
    fields that no longer differ is never silently ignored, so the
    waiver cannot be left behind by accident."""
    fingerprint = _fingerprint_module()

    _, current, acknowledged = _staged_mismatch()
    with pytest.raises(fingerprint.StaleResultsError):
        fingerprint.check_results_currency(
            _results_payload(dict(current)),  # re-run: published == current
            current=current,
            waiver=_waiver(acknowledged),
        )


# --- E. The standing invariant, against the REAL committed artefacts ---


def test_committed_results_are_current_or_carry_a_recorded_waiver():
    """The invariant every PR rides on from now on: the committed
    evals/results.json either fingerprints the live configuration, or a
    committed, dated waiver acknowledges exactly what differs. Red
    today — the 2026-09-12 Haiku-only artefact describes a configuration
    that no longer exists and no waiver is recorded. It goes green when
    the implementer records the #427 waiver, STAYS green when the paid
    Opus-arm re-run republishes the artefacts (and the waiver is
    deleted), and goes red again on the next unacknowledged drift."""
    fingerprint = _fingerprint_module()

    verdict = fingerprint.check_results_currency(
        _load_published_results(),
        current=fingerprint.current_config_fingerprint(),
        waiver=fingerprint.load_staleness_waiver(),
    )
    # Loud even when green: a waived pass must say so.
    if verdict.waived:
        assert verdict.mismatched_fields
        assert verdict.waiver_accepted_on
    else:
        assert verdict.fingerprint_matches is True


def test_absent_waiver_file_loads_as_none():
    """load_staleness_waiver on a path with no file is None — absence
    of a waiver is the normal, non-error state, so the standing
    invariant above needs no try/except plumbing."""
    fingerprint = _fingerprint_module()

    missing = REPO_ROOT / "evals" / "no-such-staleness-waiver.json"
    assert not missing.exists()
    assert fingerprint.load_staleness_waiver(missing) is None
