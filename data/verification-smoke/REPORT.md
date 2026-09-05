# Verification smoke: post-fix gate measurements (diagnostic)

**Run id:** `verification-smoke-2026-09-05` · **Arm:** `claude-haiku-4-5` (single) ·
**Spend:** **$0.5898** of the $0.60 hard cap, every billed call ledgered
(`evals/spend-ledger.csv`, 4 aggregate rows) · **Base:** `origin/main` @ `e5de829`
(all of #318 span/density/voices, #319 pool/journal/affordability, #320 refusal
redesign merged) · **NOT a release run** — nothing publishes; per fail-closed
policy the machinery's `results.json`/`RESULTS.md` stayed in the gitignored
run directory and are quoted here, not published.

Purpose: measure whether the release-run fixes moved the two failed measurement
gates BEFORE the corpus-expansion investment. Fresh rebuild in this worktree:
`make corpus` (all invariants passed) → `make datasets` (6/6) → `make ingest`
(Docling; 141 evidence + 11 voices chunks, 2 documents) → embedded local qdrant
index (152 chunks, corpus_version `rel-2026-09-05-1dfe3571`) → the journalled
`run_release_eval` answer-path machinery end-to-end (true-SSE streamed
generation, #13 validator per exchange), judges via ONE Batches submission.

## Gate table vs the 2026-09-04/05 release run

| Gate | Release run | This run | Delta / reading |
|---|---|---|---|
| refusal | **FAILED 10/20** | **PASSED 20/20** | **FIXED.** The #320 authoritative signal works: 17 marker declines (`[[NO-ANSWER-DECLINE]]` first-line) + 3 canned routes; every no-answer gold covered by the gate refused. |
| false_refusal | **FAILED 3/55 (5.45%)** | **PASSED 2/43 (4.65%)** | **Improved but marginal + incomplete.** qa-adv-05 now answers (fixed). qa-va-03 STILL declines (persists). qa-sev-09 is a NEW decliner. qa-va-05 + 11 more answerables unmeasured (budget); at full coverage the rate is 2–3/55 = 3.6–5.5% — straddling the <5% line. |
| canned_out_of_scope | passed 9/9 | **FAILED 5/9** | **NEW REGRESSION.** qa-na-c-03/-04/-09 and qa-na-g-03 now route `retrieval` instead of `canned` (each still declines honestly via the marker, so no unsafe answer — a cost/latency/routing regression). Prime suspect: the #315 classifier-instruction changes. |
| route_accuracy | passed | BLOCKED | Unmeasured: the classifier-accuracy script was not re-run this session (budget). The 4 canned misroutes above suggest it would NOT be clean. |
| citation_support | **FAILED 144/524 (27.5%)** | **FAILED 48/334 (14.4%)** | **REGRESSED — the #310 fix is defective on the live stream** (full diagnosis below). Not a corpus problem: every sentence that DID attach a citation was entailed (48/48). |
| severity | passed 15/15 | machinery 0/11; **offline recompute 11/11 PASSED** | **No regression** on the judged subset (10 exact + 1 adjacent, zero two-level). The 0/11 is a measurement artifact of this run's haiku self-judge deviation: haiku wraps its verdict JSON in ```json fences, which `evals/judges.py`'s bare `json.loads` rejects → all unscored. Verdicts re-parsed offline from the same batch (fence-stripped) score 11/11. 4 of 15 severity items unjudged (budget clamp). |
| chart_spec | passed 11/11 | passed 11/11 | Unchanged ($0 deterministic compare; flagship skipped-visibly per #23/#281). |
| chart_faithfulness | passed 1320/1320 | passed 1320/1320 | Unchanged. |
| chart_refusal | passed 3/3 | passed 3/3 | Unchanged. |
| voices_separation | **FAILED (3 voices chunks)** | **PASSED (0 violations)** | **FIXED.** qa-tg-01 routes `retrieval` with an evidence-only generation set (8/8 `source_type: evidence`; the release run had 3 voices chunks). #315's boundary work landed. |

Release verdict (machinery): FAILED — expected for a diagnostic; the verdict
drivers are analyzed below.

## The citation pool-cleaning split (headline)

Over the 67 answered exchanges (validator detail journalled per item):

| Pool stage | Sentences |
|---|---|
| Total delivered sentences | 445 |
| − meta/non-factual excluded (#312 classifier: furniture, passage-meta, referral, bare evaluative) | 85 |
| − generation-decline items excluded from the pool (#312/#313; 26 decline items) | 26 |
| **= pooled factual sentences (gate denominator)** | **334** |
| Factual sentences with a span-attached citation | 48 |
| … of which entailment-supported (gate numerator) | **48 (100%)** |
| Residual factual-but-uncited | **286** |

The #312 pool cleaning works as designed (declines and meta-sentences are out,
visibly). The diagnosis's predicted 70–90% band did **not** materialise —
because of a live-transport defect in the #310 span attachment, not because the
corpus cannot support the answers:

**#310 span attachment is broken against real arrival order.** All **202 of
202** citation SSE events in this run's journal carry a **zero-width span**
(`answer_block_start == answer_block_end`). `rag/generation.py`'s block-extent
tracking (~line 724) assumes "the citations_delta arrives after the block's
text deltas"; on the live stream the citations_delta arrives at block open,
BEFORE its block's text, so the stamped span is `[block_start, block_start)` —
empty. The validator's strict-overlap rule (`rag/citation_validator.py` ~line
653: `sentence_start < end and sentence_end > start`) attaches an empty span to
NO sentence; the 48 that did attach are the accidental cases where a block
boundary fell mid-sentence. This is WORSE than the legacy last-text-char rule
the fix replaced (27.5%): the release-eval machinery did exactly its job and
caught the live-API drift.

**Corrected-attachment ceiling ≈ 58%.** Re-attaching each citation offline to
the sentence at/after its block-start point (the text a block-open citation
actually cites) covers **195/334 = 58.4%** of pooled factual sentences even
assuming 100% entailment. So a second, independent gap remains behind the
attachment bug: **citation density** — the model emits ~202 citation blocks
for 334 pooled factual sentences (~0.6/sentence). The #311 prompt anchors did
not move Haiku to per-sentence citing. 95% is unreachable from here by corpus
expansion alone.

## Refusal / false-refusal detail (authoritative gates, #320)

- All 20 refusal-gate no-answer golds fire the authoritative signal: 3 canned
  routes + 17 first-line marker declines. The marker's injection guard held
  (no mid-answer marker flips observed).
- 29/29 covered no_answer items (gate + calibration categories) declined or
  refused — the generation-level decline is behaving as the reliable signal
  the redesign bet on.
- False refusals on covered answerables: **qa-va-03** (persists from the
  release run) and **qa-sev-09** (new). qa-adv-05 — a release-run offender —
  now answers. These two decliners are retrieval-adequacy candidates the
  corpus expansion may genuinely help.
- Pre-filter (demoted, #320): run with the floor disabled (inert by design);
  the honest artifact was calibrated post-hoc from the run's own captured top
  rerank scores: floor **0.002465** = min(answerable)/2, `separable: false`
  recorded as the diagnostic (no-answer max 0.186 ≫ answerable min 0.0049 —
  same inseparable geometry the redesign was built for).

## Coverage and deviations (all deliberate, all budget-driven)

- **72/94 answer-path items** ($0.60 cannot fit the full 94-item streamed arm:
  the release run's haiku arm alone was ~$0.90). Coverage-first ordering made
  every release gate measurable before the guard stopped: all 20 refusal-gate,
  all 9 canned, all 15 severity, all 3 targeted (incl. qa-tg-01), all 15
  single + 8/10 multi, plus the 3 prior false-refusal offenders topped up
  (2 of 3; the guard stopped before qa-va-05).
- **Not covered (22):** qa-mp-09/-10, 10 uncalibrated no_answer declines
  (qa-na-c-02/-05/-06/-07/-10..-15 — feed no release gate), 6 adversarial
  (qa-adv-01..04/-06/-07), 4 voices_action (qa-va-01/-02/-04/-05).
- **Judges:** severity-only (the sole judge kind feeding a release gate in
  `build_gate_battery`), **11 of 15** items (largest subset whose exact
  count_tokens worst-case fit the cap; unjudged: qa-sev-03/-04/-05/-10),
  **haiku self-judge** (operator haiku-only rule; deviation from the
  Sonnet-judges-Haiku cross-judge scheme — and the fence-parse artifact above
  means the machinery's severity row needs the Sonnet judge OR a
  fence-tolerant collector to score in-band next time), max_tokens clamped
  to 320 (verdicts averaged 17 output tokens; nothing truncated). ONE Batches
  submission (`msgbatch_01XzDRAKrzZ6HcRkDtve7PGg`), journalled via the #316
  JudgesJournal.
- **route_accuracy** not re-run (BLOCKED in the table).
- **Chart gold set** ran at $0 (deterministic compares) — reported for
  regression visibility as unchanged.

## Spend (ledgered, run id `verification-smoke-2026-09-05`)

| Segment | Mode | Calls | Cost |
|---|---|---|---|
| release-eval-generation (streamed, cached prompt) | live | 67 | $0.4622 |
| release-eval-classify | live | 72 | $0.0645 |
| release-eval-validator | live | 30 | $0.0466 |
| release-eval-judges (severity, Batches) | batch | 11 | $0.0165 |
| **Total** | | | **$0.5898** |

$0 segments: corpus/datasets/ingest rebuild, index build + embedding, all
retrieval/rerank (local models), chart battery, pre-filter calibration,
severity offline recompute (free batch re-read + count_tokens sizing).

## Honest assessment: is corpus expansion the only remaining gap?

**No.** The fixes genuinely closed two gates — refusal (10/20 → 20/20) and
voices_separation (3 violations → 0) — and severity held (11/11 judged). But
three product gaps and one measurement gap stand between here and a passing
release run, and corpus expansion fixes none of them:

1. **#310 span attachment is defective on the live transport** (zero-width
   spans, 202/202) and has REGRESSED citation_support below the legacy rule.
   This is the first blocker and it is a code fix (stamp the block extent when
   the block CLOSES, or attach block-open citations to the following block
   text), not a corpus fix.
2. **Citation density is a second, independent ceiling (~58%)**: even with
   perfect attachment and 100% entailment, ~0.6 citations per factual sentence
   caps the gate far below 0.95. Either the generation policy must change
   (harder per-sentence citing than #311's anchors achieved on Haiku) or the
   gate's definition/threshold needs an owner decision.
3. **canned_out_of_scope regressed 9/9 → 5/9** (4 out-of-scope items now take
   the paid retrieval path before declining) — likely #315 classifier-prompt
   fallout; needs a classifier-instruction fix and a route_accuracy re-run.
4. **false_refusal is marginal**: 4.65% measured on 43/55; qa-va-03 persists
   and qa-sev-09 is new. These two ARE plausibly retrieval-adequacy cases —
   the one place corpus expansion may help this battery — but with 12
   answerables unmeasured the gate could land either side of 5%.

Where corpus expansion DOES look justified: the entailment precision signal
(48/48 attached sentences supported) says the current 2-document corpus
entails what the model actually cites, and the two residual decliners suggest
coverage gaps at the retrieval edge. But sequencing matters: fix #310's
attachment and re-measure density FIRST — those two determine citation_support
almost entirely, and both are invariant to corpus size.

*Diagnostic artifacts (gitignored, in this worktree): journals under
`data/release-run/journals/` (72 answers + charts + judges), machinery output
`data/release-run/out/{results.json,RESULTS.md}`, per-item validator detail
`data/release-run/vs_validation_detail.jsonl`, captured rerank top-scores
`data/release-run/vs_top_scores.json`, pre-filter artifact
`data/release-run/prefilter.json`, phase logs under `data/verification-smoke/`.*
