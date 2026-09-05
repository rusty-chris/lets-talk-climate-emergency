# Re-smoke: post-fix citation measurements (diagnostic)

**Run id:** `resmoke-2026-09-05` · **Arms:** `claude-haiku-4-5` (35 items) +
`claude-sonnet-5` mini-arm (9 items) · **Spend:** **$0.9452** of the $1.00
hard cap, every billed segment ledgered (`evals/spend-ledger.csv`, 5 rows incl.
one conservative estimate row) · **Base:** `origin/main` @ `031cf92` (#326
block-close citation spans, #323 canned-routing anchors, #324 fenced-verdict
parsing all merged, on top of #318/#319/#320) · **NOT a release run** — nothing
publishes; this run exists to arm the owner's #325 decision with the TRUE
post-fix citation numbers.

Purpose: measure (1) the real Haiku citation_support number now that #326 fixed
the zero-width-span defect the verification smoke caught, (2) whether a Sonnet
generation arm cites densely enough to change the #325 calculus, and (3)
whether the four #323 canned-routing regressions route `canned` again live.

Setup: fresh rebuild in this checkout (prompts changed again via #323/#326 —
nothing reused): corpus ingest via the production pipeline (Docling; 141
evidence + 11 voices chunks, 2 documents), bge-m3 hybrid index into a local
embedded qdrant (152 chunks, corpus_version `resmoke-2026-09-05`), then the
journalled `run_answer_path` machinery end-to-end per item: production
classifier -> retrieval (pre-filter disabled per #313's degrade, as in the
smoke) -> true-SSE streamed cited generation -> #13 validator per answered
exchange (haiku validator on BOTH arms, deliberately, so the entailment meter
is constant and only the generation arm varies). No judges were needed — this
is a citation/attachment measurement, not a full battery — so no Batches
submission.

Subset: 35 answerable items covering every answer-producing family
(single_passage 8, multi_passage 6, severity 8, adversarial 5, voices_action 5,
targeted 3), first-n by id within each family. no_answer items skipped
(refusal proven 20/20 in the smoke), charts skipped (unchanged, $0 gates).
Sonnet ran the same ordering (sp 3, mp 2, sev 3, adv 2, va 1, tg 1 planned);
the $1.00-cap margin guard stopped it after 9 items (sp 3, mp 2, sev 3,
adv 1 — no va/tg coverage on the Sonnet arm).

## Headline comparison (the #325 table)

| Measure | Smoke (broken spans) | **Haiku (this run)** | **Sonnet mini-arm** |
|---|---|---|---|
| Answered items pooled | 67 | 31 (of 35; 4 declines excluded) | 9 |
| Pooled factual sentences | 334 | 262 | 107 |
| **citation_support (cleaned pool)** | **14.4%** (48/334) | **64.5%** (169/262) | **65.4%** (70/107) |
| Sentences span-attached | 48 | 175 (66.8%) | 75 (70.1%) |
| … of which entailed | 48 (100%) | **169 (96.6%)** | **70 (93.3%)** |
| Factual-but-uncited | 286 | 87 (33.2%) | 32 (29.9%) |
| Citations per factual sentence | ~0.60 | **0.60** | **0.70** |
| **Zero-width citation spans** | 202/202 | **0/162** | **0/75** |
| Generation cost per item (gen-only) | — | $0.0071 | $0.0462 (6.5x) |
| All-in marginal cost per item | — | $0.0115 | $0.0531 (4.6x) |

**Matched 9-item comparison** (the same nine questions through both arms —
the honest density read, since Sonnet answers at greater length):

| Measure | Haiku (same 9) | Sonnet (same 9) |
|---|---|---|
| Factual sentences delivered | 76 | 107 (+41%) |
| citation_support | **69.7%** (53/76) | **65.4%** (70/107) |
| Citation events / factual sentence | 0.57 | 0.70 |
| Attached-sentence entailment precision | 98.1% (53/54) | 93.3% (70/75) |
| Uncited rate | 28.9% | 29.9% |

## What the numbers say

1. **#326 is verified on the live transport.** 0 of 237 citation events across
   both arms carried a zero-width span (the smoke measured 202/202). The
   MUST-be-0 condition holds; the block-close stamping works against real
   arrival order.
2. **The true post-fix Haiku citation_support is ~64.5%** — above the smoke's
   58.4% offline-corrected ceiling estimate (block-close spans cover
   multi-sentence block extents, which the offline point-reattachment
   under-counted), and far above the broken 14.4%. Still far below the 0.95
   DESIGN target.
3. **Entailment precision stays excellent where a citation attaches:** 96.6%
   (Haiku) / 93.3% (Sonnet) of span-attached factual sentences are
   entailment-supported. What the model cites, the corpus supports. The gap
   is coverage: ~30–33% of factual sentences carry no citation at all, on
   both arms.
4. **Sonnet does not close the density gap.** Its raw citations-per-sentence
   is slightly higher (0.70 vs 0.60) but it writes ~41% more factual
   sentences on the same questions, so per-sentence support lands in the
   same band (65.4% vs 69.7% matched — if anything Haiku is ahead) at 4.6x
   the all-in per-item cost. The ~0.6–0.7 citations-per-sentence habit is a
   model-family generation-policy property, not a Haiku-tier deficiency.
5. **#323 verified live:** all four regression questions (qa-na-c-03/-04/-09,
   qa-na-g-03) route `canned` through the live classifier again ($0.0042).
6. **Side-finding (false-refusal signal, not this run's gate):** 4 of 5
   voices_action items (qa-va-02/-03/-04/-05) decline via the #313 marker on
   the Haiku arm; qa-va-01 answers. qa-va-03 was a known decliner; -02/-04/-05
   were unmeasured in the smoke. The voices retrieval edge is a genuine
   corpus/coverage candidate. (The Sonnet arm never reached its va item.)

## Spend (ledgered, run id `resmoke-2026-09-05`)

| Segment | Mode | Calls | Cost |
|---|---|---|---|
| resmoke-generation (haiku, streamed, cached prompt) | live | 35 | $0.2487 |
| resmoke-generation (sonnet, streamed, 8192 budget) | live | 9 | $0.4159 |
| resmoke-classify (both arms + 4-question spot-check) | live | 48 | $0.0509 |
| resmoke-validator (haiku validator, both arms) | live | 42 | $0.1697 |
| aborted first haiku attempt (ESTIMATE, 4 items re-run) | live | ~12 | $0.0600 |
| **Total** | | | **$0.9452** |

$0 segments: corpus/voices ingest, index build + embedding, retrieval/rerank
(local models), analysis (pure recompute over journalled transcripts). The
estimate row exists because the first Haiku attempt was killed by a tool
timeout after 4 journalled items and its in-memory meter died with it;
estimated conservatively so cumulative >= real billing.

## Honest read for the #325 decision

The data supports **option (b) — re-spec the gate target semantics** — and
argues against spending on (a):

- **(a) Sonnet generation arm: measured, and it does not help.** Same
  per-sentence support band, ~30% uncited either way, 4.6x the cost. A full
  Sonnet bake-off arm cannot be expected to clear 0.95 per-sentence support
  when its density habit is 0.70 citations/sentence on longer answers.
- **(b) is what the measurements have been pointing at all along:** the two
  quantities that are actually excellent — attached-sentence entailment
  precision (93–98%) and zero unsafe attachment (0 zero-width spans, 100%
  block-scoped) — are exactly what a "per-sentence support over sentences the
  answer claims as cited, plus an uncited-rate ceiling" gate would measure.
  A gate of the form *attached-precision >= 0.95 AND uncited-rate <= some
  ceiling* passes Haiku today on the first clause (96.6%) and makes the
  second an honest, tunable product lever backed by the #19 unverified
  badges. The current 0.95-of-all-factual-sentences target is unreachable on
  either arm without a generation-policy change no prompt anchor has
  achieved (#311 tried).
- **(c) (accept a lower target) is strictly worse than (b):** it keeps a
  metric whose denominator mixes two different failure modes (bad citation
  vs no citation) and hides the precision signal that is genuinely strong.

Sequencing note: whatever (b)'s exact ceiling, the voices_action decline
cluster (4/5) is now the largest measured answer-quality gap and is
corpus/retrieval-shaped — consistent with the smoke's conclusion that corpus
expansion helps the retrieval edge, not citation density.

*Diagnostic artifacts (session scratchpad, not committed): per-item journals
with full SSE transcripts + validation records (`haiku_journal.jsonl`,
`sonnet_journal.jsonl`), the metered usage ledger (`resmoke_ledger.json`),
the attachment analysis (`analysis.json`), phase logs. The committed
artifacts are this report and the spend-ledger rows.*
