# Gold-set coverage ledger (issue #20)

> GENERATED — do not edit by hand. Regenerate with
> `python evals/scripts/gold_coverage.py`; the meta-test
> `test_coverage_ledger_matches_generator` fails on any drift.
> This file is the no-silent-caps record: every gold item that
> cannot be fully evaluated against today's corpus/pack is listed
> below with its reason. Shrinking coverage silently is impossible
> without a diff here.

## Climate-QA composition (DESIGN §6.1 as amended)

| category | items | blocked | with gold chunk ids |
|---|---|---|---|
| single_passage | 15 | 0 | 15 |
| multi_passage | 10 | 0 | 10 |
| no_answer | 39 | 0 | 0 |
| adversarial | 7 | 0 | 7 |
| severity | 15 | 0 | 15 |
| voices_action | 5 | 2 | 1 |
| targeted | 3 | 2 | 2 |
| **total** | **94** | **4** | **50** |

Smoke subset (10 items, the dev-iteration budget set): qa-sp-01, qa-sp-10, qa-mp-01, qa-mp-05, qa-na-c-01, qa-na-g-01, qa-adv-01, qa-sev-01, qa-sev-10, qa-sev-11

## No-answer gate arithmetic (review findings #192/#193)

Every no-answer item annotates `expected_route`; the reranker threshold calibration and the DESIGN §6.2 refusal release gate consume ONLY `retrieval_refusal` items (selection seam: `evals/gold_selection.py`). `canned_out_of_scope` items exercise the classifier's canned decline and are gated by the classifier's labelled query set, never by the reranker gate.

- no_answer items: 39 (30 retrieval_refusal + 9 canned_out_of_scope)
- release-gate subset (`gate` ∩ `retrieval_refusal`): 20 items — the '20-item no-answer gate subset' issue #21 expects. One flake is 19/20 = 95%, so the strict '>90%' gate survives a single flake (critic finding 15's intent); two flakes fail it.
- calibration subset (`calibration` ∩ `retrieval_refusal`): 10 items, disjoint from the gate subset — threshold-calibration items never grade the threshold they tuned.
- cost note: the growth from 75 to 94 climate-QA items adds only refusal-path items (~$0.001 each, no generation call), leaving the dev-cost-plan full-run estimate materially unchanged.

## Blocked climate-QA items

Corpus today: 32 ingested documents (corpus/manifest.yaml; chunk-id snapshot: evals/gold/ingest_chunk_ids.txt).
Each item below is authored and schema-complete but waits on corpus
expansion; unblocking = assigning gold chunk ids (and, for severity,
the pending source passage) after the named ingest, then rerunning
this generator.

- **qa-va-01** (voices_action; blocked on corpus-expansion): No voices-layer documents are ingested (corpus/manifest.yaml pending: ripple_bioscience_warnings letters, voices custom content). Gold chunks follow the voices ingest.
- **qa-va-04** (voices_action; blocked on corpus-expansion): Needs the pending UNEP Emissions Gap Report (Tier B, unep_egr) and voices/action layer for the response-shape content.
- **qa-tg-01** (targeted; blocked on corpus-expansion): The literature half is evaluable today (gold chunks above). The separation trap is inert until the Packham voices document is ingested — with no voices content in the index, voices-leakage cannot yet fire.
- **qa-tg-03** (targeted; blocked on corpus-expansion): carbon_brief_verbatim_set is Tier B pending (verbatim-chunk ingest and the NC-confirmation letter are Phase-1.5 actions); the paraphrase check is meaningless until ND-licensed text is in the index. The corpus still lacks the attribution-methodology text itself: the 2026-09-07 Tier-A expansion's NOAA/Met Office effects pages carry partial event-attribution background only (EXPANSION-SIGNOFF gap map: NOT covered), so the item stays blocked.

## Chart gold set

- items: 15 (expected-spec 11, expected-refusal 4)
- expected-spec items with committed rendered-value fixtures: 11/11 (independent generator: evals/scripts/compute_chart_fixtures.py; synthetic data only)
- refusal sub-schema (#194): synthetic-manifest refusal golds pin `requested_data`, an always-populated `nearest_dataset_first` (the planner's deterministic nearest-first — its list is never empty while the catalogue has datasets; zero-score requests resolve by alphabetical tiebreak) and `curation_gap_logged`; full field semantics documented in the chart_requests.yaml schema comment and enforced against the live planner by `test_refusal_golds_match_planner_nearest_semantics`.

### Chart-side gaps

- **chart-15-flagship-spec-validation-refusal-of-commitment** (blocked on issue-23-licence-confirmations): Binding #117 constraint (issue #20 comment): committed fixtures must exclude flagship expected-values derived from Kaufman/Bereiter (open-provisional) until #23's written confirmations arrive. Today the real manifest blocks both flagship splice pairs (require_renderable_splice_pair names the provisional member and issue #23), so the gold behaviour is refusal-of-commitment; the expected-values fixture for the real flagship is a recorded gap in evals/gold/COVERAGE.md, not a silently absent item. chart-02 and chart-06 keep the flagship's transform arithmetic (splice, BP->CE, rebaseline, overlap) under fixture coverage with synthetic data meanwhile.

## Standing caps (recorded, not silent)

- Gold chunk ids reference the CURRENT ingest (snapshot: evals/gold/ingest_chunk_ids.txt). Chunk ids are content-hash based: any corpus or chunker change invalidates them loudly via the snapshot tests, never silently.
- Real-pack chart fixtures (including the flagship) are excluded until issue #23's licence confirmations (review finding #117); the transform arithmetic is fixture-covered with synthetic data meanwhile.
- Owner severity audit: **complete** (review finding #197; packet: evals/gold/severity-audit-packet.md — the release severity gate refuses to run via `evals.severity_audit.assert_owner_severity_audit_complete()` while the packet header says pending; the owner flips it after review). Second-pass peer review of item quality is recorded on the issue-20 PR; the adversarial review (findings #192-#197) served as the further independent pass.
