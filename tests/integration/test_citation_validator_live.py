"""Live validation check (issue #13) — `live` tier, per release / on
demand only, never per PR (IMPLEMENTATION.md §3).

One real batched validation call, exercising the AnthropicAdapter
``structured`` transport this issue delivers (rag/provider.py flags it
as issue-#13 work): the canonical corrupted-answer batch must come back
as ONE parseable per-pair verdict set with usage reported — the plumbing
proof. What the model *decides* per pair is pinned by the replay
acceptance tests once recorded, and re-verified per release by the #21
eval harness; this check only proves the batched round-trip works live.

Skip-marked awaiting a key (the #162 visible-gap convention): without
ANTHROPIC_API_KEY this SKIPS loudly rather than vanish.
"""

from __future__ import annotations

import os

import pytest

from rag.citation_validator import ValidatorConfig, validate_exchange
from rag.provider import AnthropicAdapter
from tests._citation_validator_fixtures import (
    corrupted_answer_events,
    make_grounded_answer,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="live validation check needs ANTHROPIC_API_KEY — "
        "skip-marked awaiting a key (the #162 visible-gap convention)",
    ),
]


def test_live_batched_validation_round_trip():
    """One live batched call: the outcome validates, every pair gets a
    verdict, usage is reported for the ledger, and the answer is
    never touched (no generate call exists on this path by
    construction)."""
    outcome = validate_exchange(
        AnthropicAdapter(),
        make_grounded_answer(),
        corrupted_answer_events(),
        config=ValidatorConfig(),
    )
    assert outcome.validated is True
    # The corrupted fixture carries exactly two sentence<->block pairs.
    assert len(outcome.verdicts) == 2
    assert {v.pair_index for v in outcome.verdicts} == {0, 1}
    assert outcome.usage is not None
    assert outcome.usage.get("input_tokens", 0) > 0
