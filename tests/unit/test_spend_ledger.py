"""Unit tests for the spend ledger and pricing arithmetic (finding #92).

reviews/dev-cost-plan-2026-08.md rule M8: a committed `evals/spend-ledger.csv`
appended by every API-touching tool, with `cost_usd` computed from a single
pricing table in `evals/pricing.py` (per-MTok rates, 0.5x for batch, 0.1x /
1.25x for cache read/write) and `cumulative_usd` recomputed on append and
asserted monotonic. All arithmetic is pure and pinned here with hand-computed
values, like the rest of the eval-harness arithmetic (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.ledger import LEDGER_COLUMNS, append_row, read_rows
from evals.pricing import estimate_cost_usd
from ingestion.manifest import PROJECT_OPERATIONAL_DATA_MARKER, find_committed_data_files

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_LEDGER = REPO_ROOT / "evals" / "spend-ledger.csv"


class TestPricing:
    def test_haiku_live_cost_hand_computed(self):
        """Haiku 4.5 $1/$5 per MTok (dev-cost-plan pricing basis).

        19k in + 2.6k out = 0.019 + 0.013 = $0.032 — the plan's own
        classifier-accuracy estimate.
        """
        cost = estimate_cost_usd(
            "claude-haiku-4-5", input_tokens=19_000, output_tokens=2_600, mode="live"
        )
        assert cost == pytest.approx(0.032)

    def test_batch_mode_halves_cost(self):
        """Batches API: 50% off all token usage."""
        live = estimate_cost_usd("claude-haiku-4-5", input_tokens=10_000, output_tokens=1_000)
        batch = estimate_cost_usd(
            "claude-haiku-4-5", input_tokens=10_000, output_tokens=1_000, mode="batch"
        )
        assert batch == pytest.approx(live / 2)

    def test_cache_token_multipliers(self):
        """Cache reads ~0.1x and writes 1.25x the input rate.

        Haiku live: 1,000,000 cache-read tokens = $0.10; 1,000,000
        cache-creation tokens = $1.25.
        """
        read = estimate_cost_usd(
            "claude-haiku-4-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
        )
        assert read == pytest.approx(0.10)
        write = estimate_cost_usd(
            "claude-haiku-4-5", input_tokens=0, output_tokens=0, cache_creation_tokens=1_000_000
        )
        assert write == pytest.approx(1.25)

    def test_sonnet_and_opus_rates(self):
        """Sonnet $3/$15, Opus $5/$25 per MTok, matched by model-family prefix."""
        assert estimate_cost_usd(
            "claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0
        ) == pytest.approx(3.0)
        assert estimate_cost_usd(
            "claude-opus-4-8", input_tokens=0, output_tokens=1_000_000
        ) == pytest.approx(25.0)

    def test_unknown_model_is_rejected(self):
        """No silent $0 rows for unknown models — the ledger must not lie."""
        with pytest.raises(ValueError, match="model"):
            estimate_cost_usd("gpt-unknown", input_tokens=1, output_tokens=1)


class TestLedger:
    ROW = {
        "date": "2026-08-16",
        "session_id": "SYNTHETIC-run-001",
        "activity": "classifier-accuracy",
        "issue": "10",
        "model": "claude-haiku-4-5",
        "mode": "batch",
        "calls": 48,
        "input_tokens": 19_000,
        "output_tokens": 2_600,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "cost_usd": 0.016,
        "notes": "SYNTHETIC test row",
    }

    def test_append_creates_ledger_with_m8_columns(self, tmp_path):
        path = tmp_path / "spend-ledger.csv"
        written = append_row(path, self.ROW)
        assert written["cumulative_usd"] == pytest.approx(0.016)
        [row] = read_rows(path)
        assert row["session_id"] == "SYNTHETIC-run-001"
        assert float(row["cumulative_usd"]) == pytest.approx(0.016)
        header_line = next(
            line for line in path.read_text(encoding="utf-8").splitlines() if line[:1] != "#"
        )
        assert header_line.split(",") == LEDGER_COLUMNS

    def test_cumulative_recomputed_on_append(self, tmp_path):
        """cumulative_usd = previous cumulative + this row's cost, monotonic."""
        path = tmp_path / "spend-ledger.csv"
        append_row(path, self.ROW)
        second = append_row(path, {**self.ROW, "session_id": "SYNTHETIC-run-002"})
        assert second["cumulative_usd"] == pytest.approx(0.032)
        rows = read_rows(path)
        assert [float(r["cumulative_usd"]) for r in rows] == pytest.approx([0.016, 0.032])

    def test_negative_cost_is_rejected(self, tmp_path):
        """Monotonicity is enforced, not hoped for."""
        path = tmp_path / "spend-ledger.csv"
        with pytest.raises(ValueError, match="cost"):
            append_row(path, {**self.ROW, "cost_usd": -0.01})

    def test_committed_ledger_exists_with_m8_header(self):
        """The M8 ledger is committed, ready for the first live run's row."""
        assert COMMITTED_LEDGER.is_file(), "evals/spend-ledger.csv must be committed (M8)"
        header_line = next(
            line
            for line in COMMITTED_LEDGER.read_text(encoding="utf-8").splitlines()
            if line[:1] != "#"
        )
        assert header_line.split(",") == LEDGER_COLUMNS

    def test_ledger_carries_operational_data_marker(self, tmp_path):
        """Regression: the ledger passes the ADR-023 no-committed-data guard.

        A tracked .csv with no recognized first-line exemption fails
        `find_committed_data_files` (review #83 / PR #93) in CI's unit
        stage — exactly what broke PR #106's first CI run. The committed
        ledger AND every ledger file `append_row` creates must carry the
        operational-data marker (a first-party record about this project's
        own operations, the marker's documented motivating case).
        """
        first_line = COMMITTED_LEDGER.read_text(encoding="utf-8").splitlines()[0]
        assert PROJECT_OPERATIONAL_DATA_MARKER in first_line
        assert find_committed_data_files(REPO_ROOT, ["evals/spend-ledger.csv"]) == []

        fresh = tmp_path / "spend-ledger.csv"
        append_row(fresh, self.ROW)
        fresh_first_line = fresh.read_text(encoding="utf-8").splitlines()[0]
        assert PROJECT_OPERATIONAL_DATA_MARKER in fresh_first_line
        assert find_committed_data_files(tmp_path, ["spend-ledger.csv"]) == []
