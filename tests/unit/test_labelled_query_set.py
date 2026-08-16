"""Meta-tests for the labelled classifier query set (issue #10 acceptance).

Issue #10: "Labelled ~40-query set committed; a committed script reports
live-model per-class accuracy into the release eval results (#21). Release
gate: 100% recall on the labelled unsafe items." Live accuracy is a release
eval, never a pytest gate (IMPLEMENTATION.md §4.4) — what pytest CAN pin is
that the labelled set exists, is wholly synthetic (invented queries, never
real user text — DESIGN.md §8 harvesting rules), and actually covers what the
classifier must survive: every class, unsafe items in multiple subtypes,
non-English items, denialist-phrased in-scope items, and ambiguous edge cases.

Fixture: tests/fixtures/classifier/labelled_queries.yaml. Entry schema:
  id (unique), text (unique), label (six-class enum value),
  unsafe_subtype (unsafe entries only), language (when not "en"),
  edge_case (optional bool), notes (optional str).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag.query import ScopeClass, UnsafeSubtype

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
LABELLED_SET_PATH = FIXTURES_ROOT / "classifier" / "labelled_queries.yaml"
ACCURACY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "evals" / "scripts" / "classifier_accuracy.py"
)

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE — authored for this project's tests"

ALLOWED_KEYS = {"id", "text", "label", "unsafe_subtype", "language", "edge_case", "notes"}


@pytest.fixture(scope="module")
def labelled_set() -> dict:
    assert LABELLED_SET_PATH.is_file(), (
        f"missing {LABELLED_SET_PATH} - the committed labelled ~40-query set "
        "(issue #10 acceptance criterion)"
    )
    return yaml.safe_load(LABELLED_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def queries(labelled_set: dict) -> list[dict]:
    return labelled_set["queries"]


def test_labelled_set_carries_synthetic_marker(labelled_set: dict):
    """First line + document field carry the SYNTHETIC marker — no real user text ever.

    Same convention the corpus fixtures use (IMPLEMENTATION.md §5): unmarked
    query text could be mistaken for (or become) harvested real user input,
    which DESIGN.md §8 forbids committing.
    """
    first_line = LABELLED_SET_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#")
    assert SYNTHETIC_MARKER in first_line
    assert labelled_set["marker"] == SYNTHETIC_MARKER


def test_labelled_set_entries_are_well_formed(queries: list[dict]):
    """Every entry: known keys only, valid label, unique id and text."""
    valid_labels = {c.value for c in ScopeClass}
    ids = [q["id"] for q in queries]
    texts = [q["text"] for q in queries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(texts) == len(set(texts)), "duplicate query texts"
    for q in queries:
        assert set(q) <= ALLOWED_KEYS, f"{q['id']}: unknown keys {set(q) - ALLOWED_KEYS}"
        assert q["label"] in valid_labels, f"{q['id']}: bad label {q['label']!r}"
        assert isinstance(q["text"], str) and q["text"].strip()


def test_labelled_set_is_about_forty_queries(queries: list[dict]):
    assert 38 <= len(queries) <= 50, f"~40 queries required, got {len(queries)}"


def test_labelled_set_covers_every_class(queries: list[dict]):
    """Each of the six classes has enough items for per-class accuracy to mean something."""
    for scope in ScopeClass:
        count = sum(1 for q in queries if q["label"] == scope.value)
        assert count >= 4, f"{scope.value}: only {count} items, need >= 4"


def test_labelled_set_unsafe_items_span_subtypes(queries: list[dict]):
    """Unsafe items carry a valid subtype; both canned-response subtypes appear.

    The release gate demands 100% recall on these items, and the canned
    responses differ per subtype (DESIGN.md §3.1) — so both self_harm and
    harassment need multiple exemplars, and only unsafe entries may carry a
    subtype.
    """
    valid_subtypes = {s.value for s in UnsafeSubtype}
    unsafe = [q for q in queries if q["label"] == ScopeClass.UNSAFE.value]
    for q in unsafe:
        assert q.get("unsafe_subtype") in valid_subtypes, f"{q['id']}: missing/bad subtype"
    for q in queries:
        if q["label"] != ScopeClass.UNSAFE.value:
            assert "unsafe_subtype" not in q, f"{q['id']}: subtype on non-unsafe entry"
    per_subtype = {s: sum(1 for q in unsafe if q["unsafe_subtype"] == s) for s in valid_subtypes}
    for subtype, count in per_subtype.items():
        assert count >= 2, f"unsafe subtype {subtype}: only {count} items, need >= 2"


def test_labelled_set_includes_non_english_items(queries: list[dict]):
    """At least three non-English queries exercise the answer-in-English rule."""
    non_english = [q for q in queries if q.get("language", "en") != "en"]
    assert len(non_english) >= 3
    for q in non_english:
        assert q["language"].isalpha() and q["language"].islower()


def test_labelled_set_includes_denialist_phrased_in_scope_items(queries: list[dict]):
    """Denialist-phrased but answerable queries are labelled adversarial_in_scope.

    These route to normal retrieval with a tone flag (DESIGN.md §3.1) — the
    set must prove the classifier neither refuses them (false out_of_scope /
    unsafe) nor misses the framing.
    """
    adversarial = [q for q in queries if q["label"] == ScopeClass.ADVERSARIAL_IN_SCOPE.value]
    assert len(adversarial) >= 5


def test_labelled_set_flags_ambiguous_edge_cases(queries: list[dict]):
    """Deliberately ambiguous items are flagged so accuracy reports can slice them."""
    edge_cases = [q for q in queries if q.get("edge_case")]
    assert len(edge_cases) >= 4
    for q in edge_cases:
        assert q.get("notes"), f"{q['id']}: edge cases must explain the chosen label in notes"


DESPAIR_PHRASINGS = ("too late", "doomed", "hopeless", "point")


def test_labelled_set_pins_the_despair_boundary_from_both_sides(queries: list[dict]):
    """Despair-phrased in_scope items pin the self-harm boundary's other side.

    Finding #89: the unsafe self_harm items (rhetorical despair, 100% recall
    gate) pin one side of the boundary only. DESIGN.md §6 plans
    inflation-bait phrasings ('is it too late, are we doomed?') that the bot
    must ANSWER at assessed severity — so a classifier tuned until unsafe
    recall hits 100% by routing every doom-phrased question to the
    Samaritans canned response must be visible to the eval. At least four
    in_scope items must be despair/doom-phrased, each flagged edge_case with
    notes explaining why they are in_scope rather than unsafe.
    """
    despair_in_scope = [
        q
        for q in queries
        if q["label"] == ScopeClass.IN_SCOPE.value
        and any(phrase in q["text"].lower() for phrase in DESPAIR_PHRASINGS)
    ]
    assert len(despair_in_scope) >= 4, (
        f"only {len(despair_in_scope)} despair-phrased in_scope item(s); the "
        "self-harm boundary must be pinned from BOTH sides (finding #89)"
    )
    for q in despair_in_scope:
        assert q.get("edge_case") is True, f"{q['id']}: despair-phrased in_scope must be edge_case"
        assert q.get("notes"), f"{q['id']}: must explain why in_scope rather than unsafe"


def test_ambiguous_unsafe_items_flagged(queries: list[dict]):
    """Rhetorical-despair unsafe items carry edge_case: true (finding #89).

    q38/q40 are deliberately ambiguous (rhetorical despair, not explicit
    intent) and carry explanatory notes — per the fixture's own schema
    convention they must also set edge_case: true, so accuracy reports can
    slice the ambiguous boundary items.
    """
    by_id = {q["id"]: q for q in queries}
    for ambiguous_id in ("q38", "q40"):
        entry = by_id[ambiguous_id]
        assert entry["label"] == ScopeClass.UNSAFE.value
        assert entry.get("edge_case") is True, (
            f"{ambiguous_id}: ambiguous unsafe item must be flagged edge_case"
        )
        assert entry.get("notes")


def test_classifier_accuracy_script_committed():
    """A committed script reports live per-class accuracy into #21's release evals.

    Not a pytest gate (IMPLEMENTATION.md §4.4) — but its existence IS an issue
    #10 acceptance criterion, so pytest pins that the script is committed and
    reads the labelled fixture.
    """
    assert ACCURACY_SCRIPT_PATH.is_file(), (
        f"missing {ACCURACY_SCRIPT_PATH} - the committed live-accuracy script "
        "(issue #10 acceptance criterion; reports into #21's release evals)"
    )
    text = ACCURACY_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "labelled_queries.yaml" in text
