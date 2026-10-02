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
    the single in-flight question and never a half-written aggregate;
  * verify each resumed entry's MODEL stamp against the run's resolved
    generation model (#426) — a different or missing stamp regenerates, so a
    resume can never silently ship another model's content under the #410
    provenance line;
  * meter each generation call against the RESOLVED generation model, priced
    by evals.pricing at THAT model's rates (#475) — never the committed
    default, which bills an Opus run at Haiku rates (x5 under) so the $0.50
    hard cap guards fiction instead of money.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.pricing import estimate_cost_usd
from rag.generation import (
    CITATION_EVENT,
    FOOTER_EVENT,
    GENERATION_MODEL_DEFAULT,
    TEXT_EVENT,
    USAGE_EVENT,
)
from rag.query import Classification, QueryDecision, Route
from rag.retrieval import RetrievedPassages
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
    meter = gen.SpendMeter(ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost)
    assert meter.prior == pytest.approx(0.31)
    assert meter.spent == pytest.approx(0.31)


def test_record_accumulates_and_persists_a_ledger(tmp_path):
    ledger = tmp_path / gen.CARRIED_SPEND_FILENAME
    meter = gen.SpendMeter(ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost)
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
    meter = gen.SpendMeter(ledger, hard_cap_usd=0.50, pre_call_line_usd=0.45, cost_fn=fixed_cost)
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
    run1 = gen.SpendMeter(ledger, hard_cap_usd=1.0, pre_call_line_usd=0.45, cost_fn=fixed_cost)
    for _ in range(4):  # $0.40 spent in run 1 — still under the $0.45 line
        run1.record("generation", "claude-haiku-4-5", {"input_tokens": 1})
    assert run1.spent == pytest.approx(0.40)

    # A fresh process inherits the $0.40; one more of its own calls crosses the
    # $0.45 pre-call line, so the very next question is refused BEFORE spending.
    run2 = gen.SpendMeter(ledger, hard_cap_usd=1.0, pre_call_line_usd=0.45, cost_fn=fixed_cost)
    assert run2.prior == pytest.approx(0.40)  # inherited, not restarted at 0
    run2.record("generation", "claude-haiku-4-5", {"input_tokens": 1})  # -> $0.50
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


def test_entry_is_valid_accepts_a_citation_free_chart_entry_on_resume():
    # A curated chart entry carries its attribution on the rendered chart, not
    # as sentence citations — the generator's resume check must trust it (in
    # lockstep with service.starter_cache), not re-render it forever.
    entry = make_entry(0, STARTER_QUESTIONS[0])["entry"]
    entry["citations"] = []
    entry["chart_spec_hash"] = "a" * 64
    ok, reason = gen.entry_is_valid(entry, STARTER_QUESTIONS[0])
    assert ok is True, reason


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


# --- resolve_caps: owner-approved deploy-step cap raise (env-driven) --------


def test_resolve_caps_defaults_to_the_incident_lines():
    """With no env override the caps are the module's $0.50 / $0.45 defaults."""
    hard, pre = gen.resolve_caps({})
    assert hard == pytest.approx(gen.HARD_CAP_USD)
    assert pre == pytest.approx(gen.PRE_CALL_LINE_USD)


def test_resolve_caps_reads_an_owner_approved_raise_from_the_env():
    """The deploy finisher raises the whole-deploy-step cap via env, no code
    patch: a resumed cache that already carries $0.38 needs a cap above the
    $0.50 default to finish."""
    hard, pre = gen.resolve_caps({gen.HARD_CAP_ENV: "0.98", gen.PRE_CALL_LINE_ENV: "0.93"})
    assert hard == pytest.approx(0.98)
    assert pre == pytest.approx(0.93)


def test_resolve_caps_rejects_a_precall_line_at_or_above_the_hard_cap():
    """An inverted pair would never guard — it must be a loud error."""
    with pytest.raises(ValueError):
        gen.resolve_caps({gen.HARD_CAP_ENV: "0.50", gen.PRE_CALL_LINE_ENV: "0.50"})
    with pytest.raises(ValueError):
        gen.resolve_caps({gen.HARD_CAP_ENV: "0.50", gen.PRE_CALL_LINE_ENV: "0.60"})


# --- resolve_generation_config_kwargs: explicit, auditable regen model ------
# Regression guard for the silent-downgrade incident: a later deploy ran the
# script's Haiku default and clobbered a prior Opus cache. The model is now an
# explicit env choice; the default path must stay byte-identical to before.


class _RecordingMeter:
    """Minimal meter double: records budget_guard delegations, can fail closed."""

    def __init__(self, raise_on_check: bool = False):
        self.checks: list[str] = []
        self.raise_on_check = raise_on_check

    def check(self, label: str) -> None:
        self.checks.append(label)
        if self.raise_on_check:
            raise gen.SpendCapReached(f"cap reached before {label}")


def test_generation_config_defaults_to_the_committed_model_no_best_mode():
    """No override -> {} -> GenerationConfig() defaults: no best mode, no guard.
    This is the exact prior behaviour, so the default deploy path is unchanged."""
    kwargs = gen.resolve_generation_config_kwargs("claude-haiku-4-5", _RecordingMeter(), {})
    assert kwargs == {}


def test_generation_config_opts_into_best_mode_for_a_gated_model():
    """Selecting a gated 'best' model turns best mode on, sets the best-mode
    max_tokens, and installs a budget_guard (so best mode does not fail closed)."""
    meter = _RecordingMeter()
    kwargs = gen.resolve_generation_config_kwargs(
        "claude-haiku-4-5", meter, {gen.GENERATION_MODEL_ENV: "claude-opus-4-8"}
    )
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["best_mode_enabled"] is True
    assert kwargs["max_tokens"] == gen.BEST_MODE_MAX_TOKENS
    assert callable(kwargs["budget_guard"])


def test_best_mode_budget_guard_delegates_to_the_deploy_step_meter():
    """The installed guard is REAL, not a no-op: it calls meter.check with the
    model id (so a crossed pre-call line refuses the gated request)."""
    meter = _RecordingMeter(raise_on_check=True)
    kwargs = gen.resolve_generation_config_kwargs(
        "claude-haiku-4-5", meter, {gen.GENERATION_MODEL_ENV: "claude-opus-4-8"}
    )
    with pytest.raises(gen.SpendCapReached):
        kwargs["budget_guard"]("claude-opus-4-8")
    assert meter.checks == ["budget_guard/claude-opus-4-8"]


def test_best_mode_max_tokens_is_env_overridable():
    """max_tokens for the best path can be raised via env without a code patch."""
    kwargs = gen.resolve_generation_config_kwargs(
        "claude-haiku-4-5",
        _RecordingMeter(),
        {gen.GENERATION_MODEL_ENV: "claude-opus-4-8", gen.GENERATION_MAX_TOKENS_ENV: "3000"},
    )
    assert kwargs["max_tokens"] == 3000


def test_blank_model_env_falls_back_to_the_default():
    """An empty/whitespace override is treated as 'unset' -> default, not a
    crash or an invalid empty model id."""
    kwargs = gen.resolve_generation_config_kwargs(
        "claude-haiku-4-5", _RecordingMeter(), {gen.GENERATION_MODEL_ENV: "   "}
    )
    assert kwargs == {}


# --- #426: a resume must verify the GENERATING MODEL ------------------------
# The #410 silent-Haiku-downgrade incident, re-armed through the resumability
# seam: entry_is_valid checked question/answer_text/citations/footer but never
# the generating model, and entries carried no model field at all — so a run
# invoked with STARTER_CACHE_GENERATION_MODEL=claude-opus-4-8 resumed Haiku
# leftovers from a prior run at $0 (the Opus answer_fn was called zero times)
# while the #410 provenance line printed the Opus id. The provenance line
# lied; the run exited 0. RUN defaults to /root/release-build, which persists
# on the host across releases, so stale other-model entries are the NORMAL
# case there, not a freak one.
#
# Contract pinned here: the driver takes the resolved model as a keyword-only
# ``generation_model``, stamps it on every entry it writes (entry field:
# ``model``), and trusts an existing entry only when its stamp MATCHES — a
# different or MISSING stamp regenerates with a printed reason naming the
# model (fail-closed: an unstamped entry's provenance is unverifiable).


def stamped_entry(index: int, question: str, model: str) -> dict:
    """A well-formed, resumable entry file stamped with its generating model —
    valid on every pre-#426 axis, so ONLY the model check can reject it."""
    saved = make_entry(index, question)
    saved["entry"]["model"] = model
    return saved


def test_resume_regenerates_an_entry_stamped_with_a_different_model(tmp_path, capsys):
    """The #410 downgrade, resume edition: a Haiku-stamped leftover under an
    Opus-resolved run must be REGENERATED, never resumed — otherwise the
    aggregate ships Haiku content under a provenance line that says Opus. The
    regeneration must also say WHY (a reason naming the model), so the operator
    watching the deploy log sees the mismatch rather than a bare regen."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    stale = stamped_entry(0, STARTER_QUESTIONS[0], "claude-haiku-4-5")
    stale["entry"]["answer_text"] = "STALE Haiku-generated answer that must not ship."
    (entries_dir / "00.json").write_text(json.dumps(stale), encoding="utf-8")
    calls: list[int] = []
    summary = gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=_fresh_meter(tmp_path),
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-10-02",
        generation_model="claude-opus-4-8",
    )
    assert 0 in calls, "a model-mismatched entry must be regenerated, not resumed"
    # The shipped aggregate carries the regenerated answer, not the stale one.
    assert summary["entries"][0]["answer_text"] != stale["entry"]["answer_text"]
    # And a reason is printed naming the model — behaviour + a model-shaped
    # reason, not exact prose (the issue proposes 'REGENERATING: model mismatch').
    out = capsys.readouterr().out
    regen_lines = [line for line in out.splitlines() if "REGENERATING" in line]
    assert regen_lines, "the regeneration must be announced, not silent"
    assert any("model" in line.lower() for line in regen_lines), (
        f"the regeneration reason must name the model mismatch — got {regen_lines!r}"
    )


def test_resume_trusts_matching_model_stamps_at_zero_dollars(tmp_path):
    """Resumability is the feature, not the bug: entries stamped with the SAME
    resolved model still resume for $0 — the model check must not turn every
    restart into a full-price regeneration (that would re-open the original
    2026-09-13 cap-burn incident this script exists to prevent)."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    for i, question in enumerate(STARTER_QUESTIONS):
        (entries_dir / f"{i:02d}.json").write_text(
            json.dumps(stamped_entry(i, question, "claude-opus-4-8")), encoding="utf-8"
        )
    calls: list[int] = []
    meter = _fresh_meter(tmp_path)
    summary = gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=meter,
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-10-02",
        generation_model="claude-opus-4-8",
    )
    assert calls == [], "every matching-stamp entry resumes — zero answer_fn calls"
    assert meter.spent == pytest.approx(0.0), "a full resume spends nothing"
    assert summary["resumed"] == len(STARTER_QUESTIONS)


def test_resume_regenerates_a_legacy_entry_with_no_model_stamp(tmp_path, capsys):
    """Fail-closed on unverifiable provenance: a pre-#426 entry carries no
    model stamp, so a resume CANNOT know what generated it — it regenerates
    (with a printed reason), it is never silently trusted. Even when the
    resolved model equals the committed default, unverifiable is unverifiable;
    this is also what re-stamps the production box's legacy entries on the
    next funded run."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    # make_entry predates #426: no model field — exactly the deployed shape.
    (entries_dir / "00.json").write_text(
        json.dumps(make_entry(0, STARTER_QUESTIONS[0])), encoding="utf-8"
    )
    calls: list[int] = []
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=_fresh_meter(tmp_path),
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-10-02",
        generation_model="claude-haiku-4-5",
    )
    assert 0 in calls, "an unstamped (legacy) entry must be regenerated, not trusted"
    out = capsys.readouterr().out
    regen_lines = [line for line in out.splitlines() if "REGENERATING" in line]
    assert regen_lines and any("model" in line.lower() for line in regen_lines), (
        f"the reason must say the model stamp is missing/unverifiable — got {regen_lines!r}"
    )


def test_fresh_entries_are_stamped_so_the_next_run_can_verify_them(tmp_path):
    """The driver stamps every entry it writes with the resolved model — the
    live answer_fn knows nothing about stamping (ours here returns the
    pre-#426 shape, like the real one) — in BOTH the per-question file and the
    aggregate. And the stamp round-trips: an immediate second run under the
    same model resumes everything, which is the whole point of stamping."""
    entries_dir = tmp_path / "entries"
    out_dir = tmp_path / "cache"
    calls: list[int] = []
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=out_dir,
        meter=_fresh_meter(tmp_path),
        answer_fn=recording_answer_fn(calls),
        generated_on="2026-10-02",
        generation_model="claude-opus-4-8",
    )
    for i in range(len(STARTER_QUESTIONS)):
        saved = json.loads((entries_dir / f"{i:02d}.json").read_text(encoding="utf-8"))
        assert saved["entry"].get("model") == "claude-opus-4-8", (
            f"entry {i:02d} was written without a model stamp — the next run cannot verify it"
        )
    aggregate = json.loads((out_dir / "starter_answers.json").read_text(encoding="utf-8"))
    assert all(e.get("model") == "claude-opus-4-8" for e in aggregate["entries"])
    # Round trip: a second run under the same model resumes every entry.
    second_calls: list[int] = []
    gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=out_dir,
        meter=_fresh_meter(tmp_path),
        answer_fn=recording_answer_fn(second_calls),
        generated_on="2026-10-02",
        generation_model="claude-opus-4-8",
    )
    assert second_calls == [], "freshly stamped entries must resume on the next same-model run"


def test_run_summary_reports_the_resumed_count_under_the_checked_model(tmp_path, capsys):
    """#410's provenance line must not overstate what it checked: the returned
    summary carries the model the resume check ran under (``generation_model``),
    and the printed summary line names it NEXT TO the resumed count — so
    'generation model: claude-opus-4-8' can never again sit above thirteen
    silently-resumed entries of unknown pedigree."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    for i, question in enumerate(STARTER_QUESTIONS):
        (entries_dir / f"{i:02d}.json").write_text(
            json.dumps(stamped_entry(i, question, "claude-opus-4-8")), encoding="utf-8"
        )
    summary = gen.generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=entries_dir,
        out_cache_dir=tmp_path / "cache",
        meter=_fresh_meter(tmp_path),
        answer_fn=recording_answer_fn([]),
        generated_on="2026-10-02",
        generation_model="claude-opus-4-8",
    )
    assert summary["generation_model"] == "claude-opus-4-8"
    assert summary["resumed"] == len(STARTER_QUESTIONS)
    out = capsys.readouterr().out
    # The final summary line already prints '<n> resumed' (lowercase; the
    # per-entry lines say 'RESUMED'); it must now carry the checked model too.
    summary_lines = [line for line in out.splitlines() if "resumed" in line]
    assert summary_lines, "the run summary line must report the resumed count"
    assert any("claude-opus-4-8" in line for line in summary_lines), (
        f"the summary must name the model the resume check ran under — got {summary_lines!r}"
    )


# --- #475: generation spend is metered against the RESOLVED model -----------
# The money bug on the funded release step: main()'s streaming loop records
# every generation USAGE_EVENT as
#     meter.record("generation", GENERATION_MODEL_DEFAULT, event["data"])
# while the request itself is built from generation_config.model — Opus on any
# best-mode release (STARTER_CACHE_GENERATION_MODEL=claude-opus-4-8, the #410
# discipline). USAGE_EVENT carries token counts only, never a model id
# (rag/generation.py answer_stream_to_sse), so the meter has nothing to
# reconcile against: an Opus run is priced at Haiku rates — exactly x5 under —
# and a $0.50 hard cap can really spend ~$2.50 before check/record trips,
# while DEPLOYMENT.md §3 tells the operator to size that cap for a full 13.
#
# The defective metering lives inside main()'s own wiring (the closures that
# consume the stream), so these tests drive the REAL main() with every live
# seam faked — zero network, zero spend — and read the carried-spend ledger
# back. Scope guard: ONLY the generation segment is wrong. The classify row
# genuinely runs on the committed Haiku default, and the validator row already
# records outcome.model; both are pinned below so a careless fix cannot
# "correct" them too.

RESOLVED_GENERATION_MODEL = "claude-opus-4-8"
#: The issue's confirmed arithmetic: 7,000 in / 700 out per exchange prices at
#: $0.01050 on Haiku vs $0.05250 on Opus — the x5.00 undercount. The dollar
#: figures are never asserted directly; evals.pricing is the oracle throughout,
#: so a price-table change moves both sides of every assertion.
GENERATION_USAGE = {"input_tokens": 7000, "output_tokens": 700}
CLASSIFY_USAGE = {"input_tokens": 400, "output_tokens": 60}
VALIDATOR_USAGE = {"input_tokens": 1200, "output_tokens": 80}
#: Deliberately a THIRD model family — neither the committed default nor the
#: resolved generation model — so a fix that pins validator rows to either
#: constant (instead of keeping ``outcome.model``) fails the scope-guard pin.
VALIDATOR_REPORTED_MODEL = "claude-sonnet-4-5"


def _wire_main_to_a_faked_pipeline(
    monkeypatch, tmp_path, *, classify_usage, validator_usage, roomy_caps
):
    """Point ``gen.main()`` at tmp dirs and fake every live seam it wires up.

    The metering under test is inside ``main()``'s own closures, so the
    contract can only be exercised by driving ``main()`` itself. Everything
    that would touch the network or disk-heavy weights — qdrant, the Anthropic
    adapter, the chart pack, retrieval and its lazily-imported multi-GB
    embedder, the generation stream, the validator — is replaced with $0
    fakes; the env/config resolution, the SpendMeter, the driver and the
    ledger file are the REAL code under test.

    ``roomy_caps=True`` raises the deploy-step caps far above any honest
    13-question Opus total, so ledger-inspection tests complete under BOTH the
    buggy (Haiku-priced) and the honest (Opus-priced) meter; ``False`` keeps
    the incident's committed $0.50/$0.45 defaults for the cap-interaction
    test. Returns the paths plus every GenerationConfig the fake stream was
    called with (the premise check that the requests really went out as Opus).
    """
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "cache"
    run_dir.mkdir()
    monkeypatch.setattr(gen, "RUN", run_dir)
    monkeypatch.setattr(gen, "OUT_CACHE_DIR", out_dir)
    # No curated chart starters: every question takes the metered generation
    # path, so the ledger carries one generation row per starter question.
    monkeypatch.setattr(gen, "CURATED_CHART_STARTERS", {})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-never-sent")
    monkeypatch.setenv(gen.GENERATION_MODEL_ENV, RESOLVED_GENERATION_MODEL)
    if roomy_caps:
        monkeypatch.setenv(gen.HARD_CAP_ENV, "10.0")
        monkeypatch.setenv(gen.PRE_CALL_LINE_ENV, "9.0")
    else:
        monkeypatch.delenv(gen.HARD_CAP_ENV, raising=False)
        monkeypatch.delenv(gen.PRE_CALL_LINE_ENV, raising=False)

    # ---- the live seams, faked ----
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda url: SimpleNamespace(url=url))

    class _FakeAdapter:
        """Holds the key; any attempt at a real call would AttributeError."""

        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr("rag.provider.AnthropicAdapter", _FakeAdapter)
    monkeypatch.setattr(
        "rag.retrieval.load_prefilter_artifact",
        lambda path: SimpleNamespace(enabled=True, threshold=0.25, reason="faked for #475"),
    )
    monkeypatch.setattr("rag.retrieval.CrossEncoderReranker", lambda: SimpleNamespace())
    # run_retrieval lazily constructs the multi-GB embedder on first use; hand
    # it a feather so the fake retrieve below is reached without any weights.
    monkeypatch.setattr("rag.indexing.Bgem3EmbeddingModel", lambda: SimpleNamespace())

    passage = SimpleNamespace(
        chunk_id="chunk-0001",
        rerank_score=0.91,
        clears_threshold=True,
        payload={"body": "Observed warming is unequivocal."},
    )
    monkeypatch.setattr(
        "rag.retrieval.retrieve",
        lambda *args, **kwargs: RetrievedPassages(passages=(passage,)),
    )
    monkeypatch.setattr(
        "charts.pack.load_chart_pack_frames",
        lambda manifest, pack_dir: ({"datasets": {}}, {"faked_frame": SimpleNamespace()}),
    )

    def fake_process_query(adapter, question, history):
        return QueryDecision(
            route=Route.RETRIEVAL,
            classification=Classification(
                scope="in_scope", rewritten_query=question, usage=classify_usage
            ),
            retrieval_query=question,
        )

    monkeypatch.setattr("rag.query.process_query", fake_process_query)

    generation_configs: list = []

    def fake_stream_grounded_answer(adapter, retrieved, question, *, config, corpus_vintage):
        # Capture the config the request is BUILT from — the resolved model —
        # then emit exactly the stream shape rag.generation yields: the
        # USAGE_EVENT carries token counts ONLY, no model id, so the meter
        # cannot learn the model from the event; it must use the config's.
        generation_configs.append(config)
        return [
            {
                "event": TEXT_EVENT,
                "data": {"text": "Scientists call it an emergency because warming is rapid."},
            },
            {
                "event": CITATION_EVENT,
                "data": {
                    "chunk_id": "chunk-0001",
                    "cited_text": "Observed warming is unequivocal.",
                    "document_title": "Faked Assessment Report",
                },
            },
            {"event": USAGE_EVENT, "data": dict(GENERATION_USAGE)},
            {"event": FOOTER_EVENT, "data": {"text": "Faked footer; data as of 2026-09-13."}},
        ]

    monkeypatch.setattr("rag.generation.stream_grounded_answer", fake_stream_grounded_answer)
    monkeypatch.setattr(
        "service.app._grounded_answer_from_sse",
        lambda transcript, retrieved: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "rag.citation_validator.validate_exchange",
        lambda adapter, answer, transcript, config: SimpleNamespace(
            validated=True, usage=validator_usage, model=VALIDATOR_REPORTED_MODEL
        ),
    )
    return SimpleNamespace(
        ledger_path=run_dir / gen.CARRIED_SPEND_FILENAME,
        entries_dir=run_dir / "entries",
        out_dir=out_dir,
        generation_configs=generation_configs,
    )


def _ledger_rows(wiring, segment: str) -> list[dict]:
    ledger = json.loads(wiring.ledger_path.read_text(encoding="utf-8"))
    return [row for row in ledger["rows"] if row["segment"] == segment]


def test_generation_ledger_row_records_the_resolved_model(monkeypatch, tmp_path):
    """The generation row must carry the model the request was BUILT with
    (``generation_config.model`` — Opus here), not the committed Haiku
    default: the ledger is the record the cap enforces against and the one
    artifact that can corroborate the #426 provenance stamp, so a row
    asserting the wrong model makes both guards fiction."""
    wiring = _wire_main_to_a_faked_pipeline(
        monkeypatch,
        tmp_path,
        classify_usage=CLASSIFY_USAGE,
        validator_usage=VALIDATOR_USAGE,
        roomy_caps=True,
    )
    assert gen.main() == 0
    # Premise, not the contract under test: every request really went out
    # under the resolved Opus config — so a Haiku row below is a false record,
    # not a Haiku run.
    assert [config.model for config in wiring.generation_configs] == (
        [RESOLVED_GENERATION_MODEL] * len(STARTER_QUESTIONS)
    )
    generation_rows = _ledger_rows(wiring, "generation")
    assert len(generation_rows) == len(STARTER_QUESTIONS)
    assert [row["model"] for row in generation_rows] == (
        [RESOLVED_GENERATION_MODEL] * len(STARTER_QUESTIONS)
    ), (
        "generation spend must be recorded against the RESOLVED model the "
        "request was made with, never the committed default"
    )


def test_generation_cost_is_priced_at_the_resolved_models_rates(monkeypatch, tmp_path):
    """Each generation row's ``cost_usd`` must be evals.pricing's estimate for
    the RESOLVED model over the emitted token counts — the oracle, never a
    hardcoded dollar figure. Today the row prices the same tokens at Haiku
    rates: exactly x5 under on an Opus run, which is the money bug."""
    wiring = _wire_main_to_a_faked_pipeline(
        monkeypatch,
        tmp_path,
        classify_usage=CLASSIFY_USAGE,
        validator_usage=VALIDATOR_USAGE,
        roomy_caps=True,
    )
    assert gen.main() == 0
    honest_cost = estimate_cost_usd(
        RESOLVED_GENERATION_MODEL,
        input_tokens=GENERATION_USAGE["input_tokens"],
        output_tokens=GENERATION_USAGE["output_tokens"],
    )
    for row in _ledger_rows(wiring, "generation"):
        assert row["cost_usd"] == pytest.approx(honest_cost), (
            "generation cost must be priced at the resolved model's rates "
            f"(expected the pricing oracle's ${honest_cost:.5f} for "
            f"{RESOLVED_GENERATION_MODEL}, got ${row['cost_usd']:.5f})"
        )


def test_classify_and_validator_rows_are_untouched_by_the_fix(monkeypatch, tmp_path):
    """Scope guard (#475) — this pin PASSES today and must KEEP passing: only
    the generation segment misrecords its model. The classify row genuinely
    runs on the committed Haiku default even when the generation override is
    Opus, and the validator row records ``outcome.model`` — the model the
    validator itself reported (a third family here, precisely so a fix that
    rewires validator rows to the generation model or the default is caught)."""
    wiring = _wire_main_to_a_faked_pipeline(
        monkeypatch,
        tmp_path,
        classify_usage=CLASSIFY_USAGE,
        validator_usage=VALIDATOR_USAGE,
        roomy_caps=True,
    )
    assert gen.main() == 0
    classify_rows = _ledger_rows(wiring, "classify")
    assert len(classify_rows) == len(STARTER_QUESTIONS)
    classify_cost = estimate_cost_usd(GENERATION_MODEL_DEFAULT, **CLASSIFY_USAGE)
    for row in classify_rows:
        assert row["model"] == GENERATION_MODEL_DEFAULT
        assert row["cost_usd"] == pytest.approx(classify_cost)
    validator_rows = _ledger_rows(wiring, "validator")
    assert len(validator_rows) == len(STARTER_QUESTIONS)
    validator_cost = estimate_cost_usd(VALIDATOR_REPORTED_MODEL, **VALIDATOR_USAGE)
    for row in validator_rows:
        assert row["model"] == VALIDATOR_REPORTED_MODEL
        assert row["cost_usd"] == pytest.approx(validator_cost)


def test_an_undersized_cap_refuses_an_opus_run_before_spending_through_it(monkeypatch, tmp_path):
    """The property that protects the owner's money: under the committed
    $0.50/$0.45 defaults, a 13-question Opus run does not fit (~$0.68 honest),
    so the meter's pre-call discipline must refuse fail-closed BEFORE spending
    past the line — partial entries kept, no aggregate, ledger carried. Today
    the x5 undercount lets the run 'complete' with the meter reading ~$0.14
    while ~$0.68 of real Opus spend walks out the door, so no cap ever trips."""
    wiring = _wire_main_to_a_faked_pipeline(
        monkeypatch,
        tmp_path,
        # Only generation bills, so the halt arithmetic is the oracle's alone.
        classify_usage=None,
        validator_usage=None,
        roomy_caps=False,
    )
    with pytest.raises(gen.SpendCapReached) as excinfo:
        gen.main()
    # Refused by the PRE-CALL guard — before more money leaves — never by the
    # after-the-fact hard-cap breach.
    assert "HARD-CAP GUARD" in str(excinfo.value)
    ledger = json.loads(wiring.ledger_path.read_text(encoding="utf-8"))
    assert ledger["total_usd"] < gen.HARD_CAP_USD, (
        "the halt must land before the hard cap is crossed, not after"
    )
    # The metered total is the honest Opus total for exactly the generations
    # that ran — priced by the oracle, so this cannot rot with the price table.
    per_generation = estimate_cost_usd(RESOLVED_GENERATION_MODEL, **GENERATION_USAGE)
    generation_rows = _ledger_rows(wiring, "generation")
    assert 0 < len(generation_rows) < len(STARTER_QUESTIONS)
    assert ledger["total_usd"] == pytest.approx(len(generation_rows) * per_generation)
    # A cap halt stops mid-step, never mid-file: completed entries stay on
    # disk for the next (owner-approved, raised-cap) run; no partial aggregate.
    assert len(list(wiring.entries_dir.glob("*.json"))) < len(STARTER_QUESTIONS)
    assert not (wiring.out_dir / "starter_answers.json").exists()
