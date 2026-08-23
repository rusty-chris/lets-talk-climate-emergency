"""Badge-application latency evidence (issue #13, TDD plan step 8) —
integration tier, `perf` marked.

Issue #13 acceptance: badge latency is MEASURED and RECORDED into the
perf log every run, and the ~2 s target is ASSERTED only on the demo
hardware profile (the issue #11 / finding #176 convention — CI hardware
records honest evidence, it never gates on its own speed).

Honesty note on what this measures: the unit/integration tiers never
make a live LLM call (IMPLEMENTATION.md §3), so the wall clock here
covers OUR post-stream work — segmentation, pairing, request build, the
adapter round-trip (FakeAdapter), verdict parsing and badge emission.
The live call's latency share is release-gated through the #21 eval
harness and the live tier, not here; on the demo profile the ~2 s
budget must hold with enormous margin for the local share, or the live
share has no room at all.
"""

from __future__ import annotations

import csv
import os
import time

import pytest

from rag.citation_validator import (
    BADGE_EVENT,
    BADGE_LATENCY_BUDGET_SECONDS,
    ValidatorConfig,
    append_validation_events,
    record_badge_latency,
    validate_exchange,
)
from rag.provider import FakeAdapter
from tests._citation_validator_fixtures import (
    corrupted_answer_events,
    make_grounded_answer,
    verdicts_output,
)

pytestmark = pytest.mark.perf

#: The env var the release runbook sets on the demo hardware profile;
#: only then is the ~2 s budget asserted (never on CI hardware).
PERF_PROFILE_ENV = "CLIMATE_CHAT_PERF_PROFILE"
DEMO_PROFILE = "demo"


def test_badge_latency_recorded(tmp_path):
    """Measure stream-end -> badges-delivered; record one CSV row; gate
    on the budget only under the demo profile."""
    fake = FakeAdapter(structured_results=[verdicts_output(True, False)])
    answer = make_grounded_answer()
    source_events = corrupted_answer_events()

    def validate(events):
        return validate_exchange(fake, answer, events, config=ValidatorConfig())

    composed = append_validation_events(iter(source_events), validate)
    # Drain the delivered stream (the answer the user already has)...
    for _ in range(len(source_events)):
        next(composed)
    # ...then time the post-stream validation + badge emission.
    started = time.perf_counter()
    badges = list(composed)
    elapsed = time.perf_counter() - started

    assert badges, "the corrupted fixture must produce badge events"
    assert all(e["event"] == BADGE_EVENT for e in badges)

    hardware_profile = os.environ.get(PERF_PROFILE_ENV, "unspecified")
    log_path = tmp_path / "badge-latency.csv"
    record = record_badge_latency(
        log_path,
        pair_count=2,
        wall_clock_seconds=elapsed,
        hardware_profile=hardware_profile,
    )

    # The evidence survives the run: header + one row, budget documented.
    assert log_path.is_file()
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["wall_clock_seconds"]) == pytest.approx(elapsed)
    assert int(row["pair_count"]) == 2
    assert float(row["budget_seconds"]) == BADGE_LATENCY_BUDGET_SECONDS
    assert row["hardware_profile"] == hardware_profile
    assert record["within_budget"] == (elapsed <= BADGE_LATENCY_BUDGET_SECONDS)

    if hardware_profile == DEMO_PROFILE:
        assert elapsed <= BADGE_LATENCY_BUDGET_SECONDS, (
            f"demo hardware profile blew the badge budget: {elapsed:.3f}s of purely "
            f"local work against the ~{BADGE_LATENCY_BUDGET_SECONDS}s end-to-end target"
        )


def test_appending_a_second_row_preserves_the_first(tmp_path):
    """The log accumulates evidence: append semantics, single header."""
    log_path = tmp_path / "badge-latency.csv"
    record_badge_latency(log_path, pair_count=2, wall_clock_seconds=0.01, hardware_profile="ci")
    record_badge_latency(log_path, pair_count=8, wall_clock_seconds=0.02, hardware_profile="ci")
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["pair_count"]) for r in rows] == [2, 8]
