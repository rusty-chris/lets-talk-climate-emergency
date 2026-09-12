"""Footprint feature RED — the privacy-safe persistent aggregate.

``service.footprint.FootprintLedger`` journals the application-lifetime
token totals + exchange count (+ measured CPU-seconds) beside the #217
spend journal, under the same atomic-write conventions (fsync'd temp
file + rename — finding #302; journal on EVERY record under a lock so a
crash-loop never loses the aggregate). Privacy is structural: the
journal carries counts and a since-date ONLY — no content, no
identifiers, nothing joinable to a conversation — pinned against the
actual bytes on disk, whatever poison rides in on the inputs.

Honesty rule: a corrupt/unreadable journal means the totals are UNKNOWN
— both reads and writes refuse loudly (naming the path), the /footprint
page shows the unavailable notice, and history is never clobbered by
silently restarting the count from zero.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from service.exchange_log import FORBIDDEN_IDENTIFIER_FIELDS
from service.footprint import (
    FOOTPRINT_JOURNAL_ALLOWED_KEYS,
    FOOTPRINT_STATE_FILENAME,
    FootprintLedger,
    FootprintLedgerError,
    FootprintTotals,
)


def fixed_clock(day: str = "2026-09-12"):
    moment = datetime.fromisoformat(f"{day}T12:00:00+00:00").astimezone(UTC)
    return lambda: moment


def make_ledger(tmp_path: Path, *, day: str = "2026-09-12") -> FootprintLedger:
    return FootprintLedger(state_dir=tmp_path / "spend-state", clock=fixed_clock(day))


GENERATION_USAGE = {
    "input_tokens": 900,
    "output_tokens": 42,
    "cache_read_input_tokens": 4200,
    "cache_creation_input_tokens": 0,
}
CLASSIFIER_USAGE = {"input_tokens": 300, "output_tokens": 7}

EXCHANGE_RECORDS = [
    {"model": "claude-haiku-4-5", "usage": CLASSIFIER_USAGE},
    {"model": "claude-haiku-4-5", "usage": GENERATION_USAGE},
]


class TestAccumulation:
    def test_fresh_ledger_reads_zero_totals(self, tmp_path) -> None:
        totals = make_ledger(tmp_path).totals()
        assert totals == FootprintTotals(
            since=None,
            exchanges=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cpu_seconds=0.0,
        )

    def test_one_exchange_sums_its_usage_records(self, tmp_path) -> None:
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(EXCHANGE_RECORDS)
        totals = ledger.totals()
        assert totals.exchanges == 1
        assert totals.input_tokens == 1200
        assert totals.output_tokens == 49
        assert totals.cache_read_input_tokens == 4200
        assert totals.cache_creation_input_tokens == 0
        assert totals.since == "2026-09-12"

    def test_exchange_count_counts_exchanges_not_calls(self, tmp_path) -> None:
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(EXCHANGE_RECORDS)
        ledger.record_exchange([])  # a cached/refused exchange: zero usage
        totals = ledger.totals()
        assert totals.exchanges == 2
        assert totals.input_tokens == 1200

    def test_none_and_absent_usage_count_zero(self, tmp_path) -> None:
        # The service.budget convention: None usage / absent keys are zero.
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(
            [
                {"model": "claude-haiku-4-5", "usage": None},
                {"model": "claude-haiku-4-5", "usage": {"output_tokens": 5}},
            ]
        )
        totals = ledger.totals()
        assert totals.output_tokens == 5
        assert totals.input_tokens == 0

    def test_cpu_seconds_accumulate(self, tmp_path) -> None:
        ledger = make_ledger(tmp_path)
        ledger.record_exchange([], cpu_seconds=1.5)
        ledger.record_exchange([], cpu_seconds=1.5)
        assert ledger.totals().cpu_seconds == pytest.approx(3.0)

    def test_concurrent_records_lose_nothing(self, tmp_path) -> None:
        # The #217 discipline: the aggregate under the public count is
        # only as good as its lock.
        ledger = make_ledger(tmp_path)

        def hammer() -> None:
            for _ in range(50):
                ledger.record_exchange([{"model": "m", "usage": {"input_tokens": 10}}])

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        totals = ledger.totals()
        assert totals.exchanges == 200
        assert totals.input_tokens == 2000


class TestPersistence:
    def test_journal_lives_beside_the_spend_state(self, tmp_path) -> None:
        # Same state_dir as service.budget's journal; the pinned filename.
        ledger = make_ledger(tmp_path)
        assert FOOTPRINT_STATE_FILENAME == "footprint-state.json"
        assert ledger.state_path == tmp_path / "spend-state" / FOOTPRINT_STATE_FILENAME

    def test_journals_on_every_record(self, tmp_path) -> None:
        # A crash (not just a clean shutdown) must leave the journal
        # current — the same rule as the spend journal (#217).
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(EXCHANGE_RECORDS)
        data = json.loads(ledger.state_path.read_text(encoding="utf-8"))
        assert data["exchanges"] == 1
        assert data["input_tokens"] == 1200
        ledger.record_exchange(EXCHANGE_RECORDS)
        data = json.loads(ledger.state_path.read_text(encoding="utf-8"))
        assert data["exchanges"] == 2

    def test_no_temp_file_residue(self, tmp_path) -> None:
        # atomic_write choreography: temp + fsync + rename, nothing left.
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(EXCHANGE_RECORDS)
        residue = [p.name for p in ledger.state_path.parent.iterdir() if p.name.endswith(".tmp")]
        assert residue == []

    def test_restart_reads_the_aggregate_back(self, tmp_path) -> None:
        make_ledger(tmp_path).record_exchange(EXCHANGE_RECORDS)
        reborn = make_ledger(tmp_path, day="2026-10-01")
        totals = reborn.totals()
        assert totals.exchanges == 1
        assert totals.input_tokens == 1200
        # Lifetime aggregate: no day reset, and the since-date is the
        # FIRST record's date, preserved across restarts.
        assert totals.since == "2026-09-12"
        reborn.record_exchange(EXCHANGE_RECORDS)
        assert reborn.totals().exchanges == 2
        assert reborn.totals().since == "2026-09-12"


class TestCorruptJournalHonesty:
    def write_garbage(self, tmp_path) -> Path:
        state_dir = tmp_path / "spend-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / FOOTPRINT_STATE_FILENAME
        path.write_text("{not json", encoding="utf-8")
        return path

    def test_totals_raise_naming_the_path(self, tmp_path) -> None:
        path = self.write_garbage(tmp_path)
        with pytest.raises(FootprintLedgerError) as excinfo:
            make_ledger(tmp_path).totals()
        assert str(path) in str(excinfo.value)

    def test_recording_refuses_to_clobber_history(self, tmp_path) -> None:
        # Restarting the public count from zero would erase history and
        # present an undercount as truth — refuse loudly instead.
        path = self.write_garbage(tmp_path)
        before = path.read_text(encoding="utf-8")
        with pytest.raises(FootprintLedgerError):
            make_ledger(tmp_path).record_exchange(EXCHANGE_RECORDS)
        assert path.read_text(encoding="utf-8") == before

    # ------------------------------------------------------------------
    # Review finding #365 — the corrupt-refuses convention has a
    # type-shaped hole: only UNPARSEABLE corruption is wrapped. A journal
    # that is valid JSON but carries wrong-typed values (a realistic
    # crash-loop artifact for a file rewritten on every exchange) leaks a
    # raw ValueError/TypeError out of totals()/record_exchange — which is
    # NOT FootprintLedgerError, so service.main's except clause misses it
    # and GET /footprint 500s. The class docstring's promise ("corrupt
    # journal … raises FootprintLedgerError naming the path") must cover
    # wrong-typed values too, reads AND writes, never clobbering.
    # ------------------------------------------------------------------

    WRONG_TYPED_OVERRIDES = [
        pytest.param({"exchanges": "many"}, id="string-count"),
        pytest.param({"input_tokens": [1, 2]}, id="list-value"),
        pytest.param({"since": 20260901}, id="non-string-since"),
        pytest.param({"cpu_seconds": "lots"}, id="string-cpu-seconds"),
    ]

    def write_wrong_typed(self, tmp_path, overrides: dict) -> Path:
        state = {
            "since": "2026-09-01",
            "exchanges": 3,
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cpu_seconds": 0.0,
        }
        state.update(overrides)
        state_dir = tmp_path / "spend-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / FOOTPRINT_STATE_FILENAME
        path.write_text(json.dumps(state), encoding="utf-8")
        return path

    @pytest.mark.parametrize("overrides", WRONG_TYPED_OVERRIDES)
    def test_wrong_typed_journal_totals_raise_naming_the_path(self, tmp_path, overrides) -> None:
        path = self.write_wrong_typed(tmp_path, overrides)
        with pytest.raises(FootprintLedgerError) as excinfo:
            make_ledger(tmp_path).totals()
        assert str(path) in str(excinfo.value)

    @pytest.mark.parametrize("overrides", WRONG_TYPED_OVERRIDES)
    def test_wrong_typed_journal_refuses_writes_and_never_clobbers(
        self, tmp_path, overrides
    ) -> None:
        path = self.write_wrong_typed(tmp_path, overrides)
        before = path.read_text(encoding="utf-8")
        with pytest.raises(FootprintLedgerError) as excinfo:
            make_ledger(tmp_path).record_exchange(EXCHANGE_RECORDS)
        assert str(path) in str(excinfo.value)
        assert path.read_text(encoding="utf-8") == before, (
            "a wrong-typed journal must refuse writes — never clobber history"
        )


class TestPrivacyBySchema:
    POISONED_RECORDS = [
        {
            "model": "claude-haiku-4-5",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 3,
                # Poison keys that must never reach the journal, however
                # they arrive:
                "question": "SECRET-QUESTION-TEXT",
            },
            "question": "SECRET-QUESTION-TEXT",
            "answer_text": "SECRET-ANSWER-TEXT",
            "exchange_id": "deadbeefcafe",
            "ip": "203.0.113.7",
        }
    ]

    def test_journal_carries_counts_and_the_since_date_only(self, tmp_path) -> None:
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(self.POISONED_RECORDS, cpu_seconds=0.5)
        data = json.loads(ledger.state_path.read_text(encoding="utf-8"))
        assert set(data) <= FOOTPRINT_JOURNAL_ALLOWED_KEYS, (
            f"journal carries keys outside the privacy schema: "
            f"{sorted(set(data) - FOOTPRINT_JOURNAL_ALLOWED_KEYS)}"
        )
        # The counters themselves are present.
        assert {"exchanges", "input_tokens", "output_tokens", "since"} <= set(data)

    def test_no_forbidden_identifier_or_content_bytes(self, tmp_path) -> None:
        ledger = make_ledger(tmp_path)
        ledger.record_exchange(self.POISONED_RECORDS)
        raw = ledger.state_path.read_text(encoding="utf-8")
        assert "SECRET-QUESTION-TEXT" not in raw
        assert "SECRET-ANSWER-TEXT" not in raw
        assert "deadbeefcafe" not in raw
        assert "203.0.113.7" not in raw
        for field in FORBIDDEN_IDENTIFIER_FIELDS:
            assert f'"{field}"' not in raw, f"journal carries forbidden field {field!r}"

    def test_allowed_keys_are_the_pinned_schema(self) -> None:
        # The schema is itself a contract: counts, cpu-seconds, since.
        assert FOOTPRINT_JOURNAL_ALLOWED_KEYS == frozenset(
            {
                "since",
                "exchanges",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "cpu_seconds",
            }
        )
