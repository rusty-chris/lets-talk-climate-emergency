# Development-phase API cost plan — 2026-08

**Scope:** every development activity that touches the paid Anthropic API between now and MVP launch, against the client's **~$10 USD** credit balance. Runtime/production serving cost is out of scope (DESIGN §9 covers that). Estimation is offline; `count_tokens` is free but no live calls were made to produce this plan.

**Pricing basis** (per MTok, input/output): Haiku 4.5 **$1 / $5**; Sonnet **$3 / $15**; Opus **$5 / $25**. **Batches API: 50% off** all token usage. Prompt-cache reads ~0.1×, writes 1.25×; **Haiku's minimum cacheable prefix is 4096 tokens and caching fails silently below it** (issue #12 comment).

**Reference per-call costs (Haiku, live, non-batch)** — consistent with the DESIGN §9 runtime model:

| Call | Tokens (in / out) | Arithmetic | Cost |
|---|---|---|---|
| Grounded generation (8 doc blocks) | ~5,500 / ~500 | 5.5K×$1/M + 0.5K×$5/M | **$0.0080** |
| Rewrite + scope classify | ~700 / ~150 | 0.7K×$1/M + 0.15K×$5/M | **$0.0015** (≈ §9's $0.001) |
| Citation-support validation (one batched call) | ~2,500 / ~300 | 2.5K×$1/M + 0.3K×$5/M | **$0.0040** (≈ §9's $0.003) |
| Chart planner | ~1,500 / ~300 | 1.5K×$1/M + 0.3K×$5/M | **$0.0030** (≈ §9's $0.002) |
| LLM-judge call (per-answer batch of sentence-pairs) | ~2,500 / ~250 | Haiku: $0.0038 · Sonnet: 2.5K×$3/M + 0.25K×$15/M | **$0.004 (H) / $0.011 (S)** |

**Full 75-question gold-suite pipeline run** (Haiku): 55 answerable × $0.012 (rewrite+gen+validate) + 20 no-answer × $0.001 (refusal gate fires on local reranker scores; no generation call) ≈ **$0.68 live, $0.34 batched**. The 15 chart golds add 15 × $0.002 ≈ $0.03 live / $0.015 batched. A 10-question smoke subset ≈ $0.12 live / **$0.06 batched**.

---

## Deliverable 1 — Per-activity estimates

All development traffic on `claude-haiku-4-5` except where the bake-off requires otherwise (Rule M2). "Cap" is the hard per-activity budget posted on the corresponding issue.

### 1. Spike #3 — 20-question probe (+ retries)

Spike calls are bare generation (no rewrite/validator in the spike loop): 20 × $0.008 = **$0.16 per full probe run**. Realistic development includes iterating on block construction, observing the §3.4 constraints live (deliberate 400s cost ~$0 — rejected requests aren't billed), and ad-hoc debug calls.

- Best: 2 clean runs + ~10 debug calls ≈ 50 calls × $0.008 = **$0.40**
- Expected: 3 runs + ~40 debug calls ≈ 100 calls = **$0.80**
- **Cap: $1.00** (≈ 125 generation calls)

### 2. Fixture-recording sessions — #10, #12, #13, #16

One clean recording pass, counting distinct recorded requests per issue's TDD plan:

| Issue | Distinct recorded requests | Cost |
|---|---|---|
| #10 classifier/rewriter | ~6 structured calls (classifier schema fixture, one per interesting class, rewriter reference-resolution; the *malformed* fixture is hand-corrupted from a recording, $0) | 6 × $0.0015 ≈ $0.009 |
| #12 generation | ~4 generate calls (response-shape fixture, streamed SSE fixture, **2 identical-prefix calls for the cache-usage fixture** — prefix deliberately ≥4096 tokens, run sequentially so call 2 reads call 1's cache) | ≈ $0.03 |
| #13 support validator | ~3 calls (1 generation to produce a real cited answer, 1–2 entailment calls; the non-entailing case is produced by corrupting the recorded answer, not by extra API calls) | ≈ $0.015 |
| #16 chart planner | ~4 plan_chart calls (flagship, cherry-pick, one refusal, one spare; the invalid→invalid retry sequence uses FakeAdapter, $0) | ≈ $0.01 |

≈ **17 recorded requests ≈ $0.06 per clean session.** Prompt changes invalidate recordings by design (`canonical_request_hash` in `rag/provider.py`), so expect several re-record sessions as system prompts stabilise.

- Best: 2 sessions = **$0.12** · Expected: 5 sessions = **$0.30** · **Cap: $0.75** (~12 sessions)

### 3. Refusal-threshold calibration

**$0.** The refusal gate thresholds on **local** reranker scores (`bge-reranker-v2-m3`, DESIGN §3.2/§3.5); calibration is a committed script over reranker outputs, and refused queries make no generation call. Allow **$0.10** worst case for a handful of live sanity checks of the refusal-message path.

### 4. Gold-set development iterations (#20 → #21, pre-release)

Being honest: the 75-question suite will be run repeatedly while tuning prompts, chunking, and retrieval — not once. The minimization plan (Rule M5) confines iteration to the **10-question smoke subset** with full-suite runs reserved for gate checks, all via Batches:

- Expected: 15 smoke runs × $0.06 + 3 full runs (deterministic metrics only, judges withheld until gates) × $0.37 = $0.90 + $1.11 ≈ **$2.00**
- Best: 10 smoke + 2 full ≈ **$1.35**
- **Cap: $3.50** (≤ 4 full pre-release runs)
- *Counterfactual without the rules:* 15 full live runs with judges ≈ 15 × ($0.68 + ~$0.45) ≈ **$17** — this line alone would blow the budget, which is why Rules M3/M5 are enforced in the harness, not advisory.

### 5. Release bake-off (#21) — 75 q × full pipeline × {Haiku, Sonnet} + judges

Per the #21 comments: Anthropic-only, run through Batches, production default = cheapest model passing every gate; Opus only if neither passes (escalate with numbers first — **no Opus spend in this plan**).

| Component | Arithmetic | Batched cost |
|---|---|---|
| Haiku arm, pipeline + charts | ($0.68 + $0.03) × 0.5 | **$0.36** |
| Sonnet arm (generation on Sonnet at 5.5K in × $3 + 0.5K out × $15 = $0.024/q; rewrite + validator stay Haiku) | 55 × $0.028 + $0.02, × 0.5 | **$0.79** |
| Judges on Haiku arm — **Sonnet judge** | (55 faithfulness + 55 confidence + 15 severity) = 125 × $0.011 × 0.5 | **$0.69** |
| Judges on Sonnet arm — **Haiku judge** | 125 × $0.004 × 0.5 | **$0.25** |
| **Total** | | **≈ $2.10** |

**Judge model choice, justified:** #21's `test_judge_model_differs_from_generator` makes judge ≠ generator a hard config assertion, so no single judge model can score both arms. Cheapest compliant scheme is **cross-judging**: Sonnet judges the Haiku arm (the stronger judge scrutinises the likely production model, where judge quality matters most), Haiku judges the Sonnet arm; the sampled human audit (DESIGN §6.2) backstops both. "Sonnet judges everything" fails the config test on the Sonnet arm; "Haiku judges everything" fails it on the Haiku arm.

- **Cap: $3.00** (allows one re-run of a *failing gate's subset*, never a full-suite re-run).
- Contingency: if the ledger shows > $7.00 spent when the release eval starts, the **Sonnet arm + its Haiku judging (~$1.05) is deferred** pending client approval; the minimum releasable eval is the Haiku arm + Sonnet judges ≈ **$1.05**.

### 6. Severity/faithfulness judge sampling (judge calibration)

Before trusting the gates, run each judge over ~20 hand-labelled answers and compare to human labels: ~20 × 2 judge configurations × ~$0.0075 avg, batched ≈ **$0.15** expected. **Cap: $0.30.**

### 7. #58 contextual-retrieval experiment — **DEFERRED, does not fit in $10**

Corpus ~1–2M tokens (DESIGN §2.3) → ~2,000–4,000 chunks at ≤500 tokens. Per chunk, a Haiku call generates 2–3 context sentences. Cost is dominated by how much document context each call carries:

- **Naive whole-document context** (docs avg ~20K tokens, no cache benefit — parallel batch requests with identical prefixes don't share cache): ~3,000 × 20.5K ≈ 61M input tokens × $0.50/M batched ≈ **$31** + ~$0.75 output. Never do this.
- **Lean design** (containing section + short doc summary, ~1.5–2K context/chunk): ~3,000 × 2.2K ≈ 6.6M in × $0.50/M = $3.30 + 3,000 × 80 out ≈ 0.24M × $2.50/M = $0.60 ≈ **$4**. With 2M-token corpus / richer context: **$6–8**.

Even the lean design consumes 40–80% of the entire development budget, and the A/B needs the #21 harness runs on top. **Deferred until the client tops up or explicitly approves the spend.** When approved: lean context design, Batches mandatory, sequential cache-warmed recording of the per-document prefix where the batch is chunked per document.

### 8. Headroom — debugging, unplanned re-records, live-marked release smoke tests

The `live`-marked per-release tests (#12's `test_live_cited_answer_end_to_end`, #10's classifier accuracy script feeding #21) plus general slack: expected **$1.00**, **cap $1.00–2.00**.

### Totals vs the $10

| # | Activity | Best | Expected | Cap (worst, enforced) |
|---|---|---:|---:|---:|
| 1 | Spike #3 probe | $0.40 | $0.80 | $1.00 |
| 2 | Fixture recordings #10/#12/#13/#16 | $0.12 | $0.30 | $0.75 |
| 3 | Refusal-threshold calibration | $0.00 | $0.00 | $0.10 |
| 4 | Gold-set dev iterations | $1.35 | $2.00 | $3.50 |
| 5 | Release bake-off (both arms + judges) | $1.80 | $2.10 | $3.00 |
| 6 | Judge sampling/calibration | $0.10 | $0.15 | $0.30 |
| 7 | #58 contextual retrieval | — | **deferred** | (would be $4–8 lean; $30+ naive) |
| 8 | Headroom / live release smoke | $0.50 | $1.00 | $1.00 |
| | **Total (excl. #58)** | **$4.27** | **$6.35** | **$9.65** |

- **Expected case ≈ $6.35 — fits in $10 with ~$3.65 margin.**
- The **caps sum to $9.65**, i.e. even if every activity hits its cap the phase stays under $10 — but only because the caps are enforced (below). The *uncapped* worst case (full-suite iteration live, whole-doc #58) is $30–45.
- **Out of the $10:** #58 (defer), the Opus bake-off arm (escalation-only per #21), any open-model evaluation (Phase 2 per #21 comment).

---

## Deliverable 2 — Minimization plan (enforceable rules)

**M1 — Replay-first: no pytest tier ever hits the live API.** Already structural: `RecordingAdapter` refuses to construct without `CLIMATE_CHAT_RECORD=1` **and** a live key; `ReplayAdapter` raises `ReplayFixtureMissingError` on any unrecorded request. Tests hit the API once per recording session, then replay forever. *Quantified:* the road to MVP is realistically ~40–60 PRs × ~4–6 CI pushes each ≈ 250 CI runs; each lint→smoke run exercises ~50 would-be LLM calls (contract tests, replay-backed unit tests, replay-driven e2e smoke). Replay eliminates **~12,500 live calls ≈ $60–90 at Haiku prices** — 6–9× the whole budget. The marginal test run is $0.
**Enforcement:** already merged in `rag/provider.py`; CI has no `ANTHROPIC_API_KEY` secret configured for PR stages.

**M2 — All dev/eval traffic on `claude-haiku-4-5` until the final bake-off.** Sonnet appears exactly once (the bake-off arm) and as the Haiku arm's judge; Opus never appears in the dev phase (escalation path only).
**Enforcement:** model id is config (ADR-012); the recording tooling and eval harness assert `model.startswith("claude-haiku")` unless invoked with an explicit `--bake-off` flag, and the ledger (M8) records the model per row so any violation is visible in review.

**M3 — Batches API for every eval, judge, and bulk run.** 50% off all tokens; release gates and fixture-batch jobs are not latency-sensitive (per the #21 comment). Applies to: gold-suite runs, judge runs, judge calibration, the bake-off, and #58 if ever approved.
**Enforcement:** the #21 harness's live runner submits via `client.messages.batches.create` by default; per-request live mode requires `--no-batch` plus a reason string that lands in the ledger.

**M4 — Prompt caching sized past the Haiku 4096-token floor.** The generation call's stable prefix (system prompt + stable framing) must clear **4096 tokens** or Haiku caching silently does nothing (issue #12 comment). During recording sessions and any sequential live runs, order identical-prefix calls **sequentially** (a cache entry is only readable after the first response starts streaming; parallel identical requests all pay full price).
**Enforcement:** #12's smoke check asserts `usage.cache_read_input_tokens > 0` on the second of two identical-prefix calls; the recording script performs the cache-fixture pair sequentially.

**M5 — Smoke subset for iteration; full 75 only at gates.** A committed ~10-question subset (1–2 per category, incl. one no-answer, one severity, one chart) is the default for every development eval run (~$0.06 batched). The full 75-question suite requires `--full`, is limited to **4 pre-release runs**, and judge metrics run only at gate checks, not during iteration.
**Enforcement:** harness default `--subset smoke`; `--full` writes a ledger row and the harness warns when the count of `--full` rows exceeds 4.

**M6 — Hard per-session recording budget.** A recording session (one `CLIMATE_CHAT_RECORD=1` pytest invocation) may write at most **25 fixtures / ~$0.10**; the recorder counts writes and refuses beyond the limit, so a misconfigured loop can't burn credits.
**Enforcement:** add a write-counter guard to `RecordingAdapter._record` (raise after 25 writes per process unless `CLIMATE_CHAT_RECORD_LIMIT` overrides).

**M7 — Deferrals are explicit, not drift.** #58 is deferred (this plan, and a comment on the issue); the Sonnet bake-off arm auto-defers if cumulative spend > $7.00 at release-eval time; Opus runs only after client escalation with numbers. Any deferred item runs only after a top-up or an explicit written client approval on the issue.

**M8 — Committed spend ledger: `evals/spend-ledger.csv`.** Appended by the recording tooling and the eval harness from response `usage` fields; committed in the same PR as the recordings/eval results it accounts for. Spec:

```
# evals/spend-ledger.csv — one row per API-touching session or batch
date,session_id,activity,issue,model,mode,calls,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,cost_usd,cumulative_usd,notes
2026-08-17,rec-001,fixture-recording,12,claude-haiku-4-5,live,4,22000,2000,5500,5500,0.03,0.03,"cache pair sequential"
```

- `mode` ∈ `live | batch`; `activity` ∈ `spike-probe | fixture-recording | eval-smoke | eval-full | bake-off | judge-calibration | experiment | debug`.
- `cost_usd` computed from a single pricing table in `evals/pricing.py` (per-MTok rates above, 0.5× for `batch`, 0.1×/1.25× for cache read/write) — one source of truth, unit-tested with hand-computed values like the rest of the harness arithmetic.
- `cumulative_usd` is recomputed on append and asserted monotonic by a unit test; a pre-flight check in the recording script and harness **refuses to start any live/batch run when `cumulative_usd ≥ $9.00`** unless `CLIMATE_CHAT_BUDGET_OVERRIDE=1` is set (mirrors the production fail-closed budget philosophy of DESIGN §9).
- Implementers: usage fields come from `response.usage` (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`); for batches, sum per-result usage when streaming `batches.results(id)`.

---

## Per-issue budget caps (posted as comments)

| Issue | Cap | Note |
|---|---:|---|
| #3 | $1.00 | ~125 Haiku generation calls incl. retries/debug; expected $0.80 |
| #20 | $3.50 | Gold-set dev iterations; smoke subset default, ≤4 full pre-release runs, Batches only |
| #21 | $3.00 | Bake-off both arms + cross judges, Batches only; Sonnet arm defers if ledger > $7; Opus escalation-only |
| #58 | $0.00 | Deferred — $4–8 even in the lean design; runs only on top-up/explicit approval |

(Fixture recording across #10/#12/#13/#16 shares a single $0.75 cap tracked in the ledger under `activity=fixture-recording`.)
