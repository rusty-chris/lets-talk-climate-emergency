"""Replay-pinned acceptance tests for the corrupted-answer fixture
(issue #13, TDD plan steps 3-4) — awaiting a recording session.

These are the issue's named acceptance tests: the REAL model's recorded
verdicts over the committed corrupted-answer scenario (a real citation
carrying a non-entailing sentence) flag exactly the corrupted sentence,
with the supported sentence as the control. Per the #162 convention they
are skipif-guarded on the recording's existence — a hand-authored replay
fixture is never an acceptable substitute for a recording (the #67
rule), so they skip VISIBLY in every run until the implementer's
recording session lands (CLIMATE_CHAT_RECORD=1 + a live key, ledger
discipline, within the ratified shared fixture cap). Coordinator note:
the #187-#189 system-prompt edits land BEFORE any recording session — a
changed prompt invalidates recordings by design, so record only once the
prompt is settled.

The FakeAdapter twins in test_citation_validator_flow.py pin the same
pairing/flag logic deterministically today; these tests add the one
thing fakes cannot: what the actual model does with the batch.
"""

from __future__ import annotations

import pytest

from rag.citation_validator import (
    UNVERIFIED_REASON_ENTAILMENT,
    ValidatorConfig,
    validate_exchange,
)
from rag.provider import ReplayAdapter, canonical_request_hash
from tests._citation_validator_fixtures import (
    corrupted_answer_events,
    make_grounded_answer,
)
from tests.conftest import FIXTURES_ROOT

REPLAY_FIXTURES_DIR = FIXTURES_ROOT / "replay"

CONFIG = ValidatorConfig()


def _validation_replay_recorded() -> bool:
    """True when a genuine recorded validation exchange exists for the
    canonical corrupted-answer request. Guarded: at red the builders
    are unimplemented (the #162 pattern, as in
    test_generation_streaming.py)."""
    try:
        from rag.citation_validator import (
            build_entailment_pairs,
            build_validation_request,
            segment_answer_sentences,
        )

        pairs = build_entailment_pairs(
            segment_answer_sentences(corrupted_answer_events()),
            make_grounded_answer().cited_passages,
        )
        request = build_validation_request(pairs, config=CONFIG)
    except NotImplementedError:
        return False
    request_hash = canonical_request_hash("structured", request)
    return (REPLAY_FIXTURES_DIR / f"{request_hash}.json").is_file()


pytestmark = pytest.mark.skipif(
    not _validation_replay_recorded(),
    reason="awaiting a validation recording session — the #162 pattern; record "
    "AFTER the #187-#189 prompt edits settle, via CLIMATE_CHAT_RECORD=1 with a "
    "live key",
)


def _replayed_outcome():
    return validate_exchange(
        ReplayAdapter(REPLAY_FIXTURES_DIR),
        make_grounded_answer(),
        corrupted_answer_events(),
        config=CONFIG,
    )


def test_failing_sentence_gets_unverified_badge():
    """Issue #13 acceptance: the recorded model flags exactly the
    corrupted sentence (real citation, non-entailing claim) — sentence
    index 2 against its cited block 1."""
    outcome = _replayed_outcome()
    assert outcome.validated is True
    entailment_flags = {
        (u.sentence_index, u.document_index)
        for u in outcome.unverified
        if u.reason == UNVERIFIED_REASON_ENTAILMENT
    }
    assert entailment_flags == {(2, 1)}


def test_supported_sentences_not_flagged():
    """Issue #13 acceptance, the control on the same fixture: the
    supported sentence (index 1, faithfully restating its cited block)
    is never flagged."""
    outcome = _replayed_outcome()
    assert outcome.validated is True
    assert 1 not in {u.sentence_index for u in outcome.unverified}
