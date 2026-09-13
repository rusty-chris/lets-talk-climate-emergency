"""Red phase: the resumable release starter-cache generator.

Regression cover for the 2026-09-13 deploy incident (data/DEPLOY-CHECKPOINT.md):
the server-only generator restarted from question 0 every run, was not
resumable, wrote entries non-atomically, and carried its cross-run spend in a
fragile glob-plus-hardcoded-constant meter — so six non-idempotent restarts
burned $0.38 of a $0.50 cap while writing only 3 of 13 entries.

These tests pin the promoted ``scripts/generate_starter_cache.py`` contract
with ZERO live API calls: a fake per-question answer function and a fake cost
function stand in for the pipeline, so the resumability / cross-run-cap /
atomic-write / validate-on-resume logic is exercised on its own.

The generator's testable core must:
  * carry prior spend across runs through a single ledger file beside the
    cache (a fresh process reads the previous total and keeps counting);
  * enforce its cap at a QUESTION boundary — refusing before it starts a new
    question rather than part-way through writing the cache file;
  * validate each already-written entry and SKIP only the complete ones,
    regenerating anything malformed instead of trusting it blindly;
  * write each entry atomically on completion, so a killed run loses at most
    the single in-flight question and never a half-written aggregate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from service.starter_cache import STARTER_QUESTIONS, load_starter_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator_module():
    path = REPO_ROOT / "scripts" / "generate_starter_cache.py"
    spec = importlib.util.spec_from_file_location("generate_starter_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator_module()


# --- fakes (no API, no model weights) --------------------------------------

def fixed_cost(_model, **_usage):
    """A deterministic $0.10-per-call stand-in for evals.pricing."""
    return 0.10


def make_entry(index: int, question: str) -> dict:
    """A well-formed cache entry + report the way the live pipeline returns."""
    return {
        "entry": {
            "question": question,
            "answer_text": f"Answer to {question}",
            "citations": [{"chunk_id": f"doc:{index}", "cited_text": "x"}],
            "footer": "Generated 2026-09-13 against v1.1.0-launch.",
            "chart_spec_hash": None,
        },
        "report": {"question": question, "declined": False, "attempts": 1},
    }


def recording_answer_fn(calls: list[int]):
    """answer_fn that records which indices it was invoked for and bills the
    meter once per call (to drive cross-run cap tests)."""

    def answer_fn(index, question, meter=None):
        calls.append(index)
        if meter is not None:
            meter.record("generation", "claude-haiku-4-5", {"input_tokens": 1})
        return make_entry(index, question)

    return answer_fn


# --- SpendMeter: cross-run carried spend + cap -----------------------------

def test_meter_starts_at_zero_without_a_ledger(tmp_path):
    meter = gen.SpendMeter(
        tmp_path / gen.CARRIED_SPEND_FILENAME,
        hard_cap_usd=0.50,
        pre_call_line_usd=0.45,
        cost_fn=fixed_cost,
    )
    assert meter.prior == 0.0
    assert meter.spent == 0.0


def test_meter_carries_prior_spend_from_the_ledger_file(tmp_path):
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    ledger.write_text(json.dumps({"total_usd": 0.31, "rows": []}), encoding="utf-8")
    meter = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost
    )
    assert meter.prior == pytest.approx(0.31)
    assert meter.spent == pytest.approx(0.31)


def test_record_accumulates_and_persists_a_ledger(tmp_path):
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    meter = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost
    )
    meter.record("classify", "claude-haiku-4-5", {"input_tokens": 10})
    assert meter.spent == pytest.approx(0.10)
    written = json.loads(ledger.read_text(encoding="utf-8"))
    assert written["total_usd"] == pytest.approx(0.10)
    assert written["prior_usd"] == pytest.approx(0.0)
    assert written["run_usd"] == pytest.approx(0.10)
    assert written["rows"] and written["rows"][0]["segment"] == "classify"


def test_check_refuses_at_the_precall_line(tmp_path):
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    ledger.write_text(json.dumps({"total_usd": 0.46, "rows": []}), encoding="utf-8")
    meter = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost
    )
    with pytest.raises(gen.SpendCapReached):
        meter.check("next question")


def test_record_raises_on_hard_cap_breach(tmp_path):
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    ledger.write_text(json.dumps({"total_usd": 0.45, "rows": []}), encoding="utf-8")
    meter = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.60, cost_fn=lambda *_a, **_k: 0.10
    )
    with pytest.raises(gen.SpendCapReached):
        meter.record("generation", "claude-haiku-4-5", {"input_tokens": 1})
    # the breaching spend is still persisted for the next run to inherit
    assert json.loads(ledger.read_text(encoding="utf-8"))["total_usd"] >= 0.50


def test_two_runs_share_the_ledger_so_the_cap_spans_runs(tmp_path):
    """The heart of the incident: a fresh process must inherit prior spend."""
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    run1 = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost
    )
    for _ in range(4):  # $0.40 spent in run 1
        run1.record("generation", "claude-haiku-4-5", {"input_tokens": 1})
    assert run1.spent == pytest.approx(0.40)

    run2 = gen.SpendMeter(
        ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost
    )
    assert run2.prior == pytest.approx(0.40)  # inherited, not restarted at 0
    with pytest.raises(gen.SpendCapReached):
        run2.check("would exceed the cross-run cap")


# --- entry_is_valid: validate-on-resume ------------------------------------

def test_entry_is_valid_accepts_a_complete_entry():
    ok, _ = gen.entry_is_valid(make_entry(0, STARTER_QUESTIONS[0])["entry"], STARTER_QUESTIONS[0])
    assert ok is True


def test_entry_is_valid_rejects_a_question_mismatch():
    ok, reason = gen.entry_is_valid(
        make_entry(0, STARTER_QUESTIONS[0])["entry"], STARTER_QUESTIONS[1]
    )
    assert ok is False
    assert reason


@pytest.mark.parametrize("field", ["answer_text", "citations", "footer"])
def test_entry_is_valid_rejects_missing_required_fields(field):
    entry = make_entry(0, STARTER_QUESTIONS[0])["entry"]
    entry[field] = "" if field != "citations" else []
    ok, _ = gen.entry_is_valid(entry, STARTER_QUESTIONS[0])
    assert ok is False


# --- generate_starter_cache: the driver ------------------------------------

def _fresh_meter(tmp_path, **kw):
    kw.setdefault("hard_cap_usd", 100.0)
    kw.setdefault("pre_call_line_usd", 100.0)
    kw.setdefault("cost_fn", fixed_cost)
    return gen.SpendMeter(tmp_path / gen.CARRIED_SPEND_FILENAME, **kw)


def test_generates_every_entry_in_deterministic_order(tmp_path):
    entries_dir = tmp_path / "entries"
    out_dir = tmp_path / "cache"
    calls: list[int] = []
    meter = _fresh_meter(tmp_path)
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=out_dir,
        meter=meter,
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-09-13",
    )
    # deterministic order = enumerate(STARTER_QUESTIONS)
    assert calls == list(range(len(STARTER_QUESTIONS)))
    # each entry persisted atomically to its own indexed file
    for i in range(len(STARTER_QUESTIONS)):
        assert (entries_dir / f"{i:02d}.json").is_file()
    # the aggregate loads + validates exactly as the service will at boot
    cache = load_starter_cache(out_dir)
    assert len(cache.entries) == len(STARTER_QUESTIONS)
    assert cache.generated_on == "2026-09-13"


def test_resumes_valid_entries_without_respending(tmp_path):
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    # Pre-write the first three entries as already-complete (the incident's
    # 3/13). They must be skipped for $0, exactly like the real resume.
    for i in range(3):
        (entries_dir / f"{i:02d}.json").write_text(
            json.dumps(make_entry(i, STARTER_QUESTIONS[i])), encoding="utf-8"
        )
    calls: list[int] = []
    meter = _fresh_meter(tmp_path)
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=meter,
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-09-13",
    )
    # the three pre-written questions were NOT regenerated
    assert 0 not in calls and 1 not in calls and 2 not in calls
    assert calls == list(range(3, len(STARTER_QUESTIONS)))


def test_regenerates_an_invalid_existing_entry(tmp_path):
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    # A malformed leftover (empty answer_text) must NOT be trusted.
    broken = make_entry(0, STARTER_QUESTIONS[0])
    broken["entry"]["answer_text"] = ""
    (entries_dir / "00.json").write_text(json.dumps(broken), encoding="utf-8")
    calls: list[int] = []
    meter = _fresh_meter(tmp_path)
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=meter,
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-09-13",
    )
    assert 0 in calls  # regenerated, not skipped


def test_cap_halt_at_question_boundary_keeps_completed_entries(tmp_path):
    """Refuse mid-question, never mid-file: a cap hit leaves the completed
    entries intact and writes NO partial aggregate, and the spend carries."""
    entries_dir = tmp_path / "entries"
    out_dir = tmp_path / "cache"
    calls: list[int] = []
    # cost $0.10/question, refuse once spent reaches the $0.25 pre-call line:
    # Q0->0.10, Q1->0.20, Q2 boundary check sees 0.20<0.25 ok ->0.30, Q3
    # boundary check sees 0.30>=0.25 -> refuse. Entries 00,01,02 written.
    meter = _fresh_meter(tmp_path, hard_cap_usd=1.0, pre_call_line_usd=0.25)
    with pytest.raises(gen.SpendCapReached):
        gen.generate_starter_cache(
            STARTER_QUESTIONS,
            entries_dir=entries_dir,
            out_cache_dir=out_dir,
            meter=meter,
            answer_fn=recording_answer_fn(calls),
            generated_on="2026-09-13",
        )
    written = sorted(p.name for p in entries_dir.glob("*.json"))
    assert written == ["00.json", "01.json", "02.json"]
    # no half-written aggregate cache was produced
    assert not (out_dir / "starter_answers.json").exists()
    # spend was persisted so the NEXT run inherits it (cross-run cap)
    ledger = json.loads((tmp_path / gen.CARRIED_SPEND_FILENAME).read_text(encoding="utf-8"))
    assert ledger["total_usd"] == pytest.approx(0.30)
