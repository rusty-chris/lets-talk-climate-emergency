# Release eval results

Generated: 2026-09-12 (results schema v1)

## Release verdict: PASSED

Production model: claude-haiku-4-5

## Arm: claude-haiku-4-5 — run cost $1.70

| gate | status | score | threshold | evidence |
|---|---|---|---|---|
| refusal | PASSED | 20/20 | 0.9 | [results.json](results.json#arms/claude-haiku-4-5/refusal) |
| false_refusal | PASSED | 2/55 | 0.05 | [results.json](results.json#arms/claude-haiku-4-5/false_refusal) |
| canned_out_of_scope | PASSED | 9/9 | — | [results.json](results.json#arms/claude-haiku-4-5/canned_out_of_scope) |
| route_accuracy | PASSED | — | — | [results.json](results.json#arms/claude-haiku-4-5/route_accuracy) |
| citation_entailment_precision | PASSED | 335/348 | 0.95 | [results.json](results.json#arms/claude-haiku-4-5/citation_entailment_precision) |
| uncited_factual_rate | PASSED | 128/476 | 0.35 | [results.json](results.json#arms/claude-haiku-4-5/uncited_factual_rate) |
| verified_claim_group_coverage | PASSED | 175/206 | 0.75 | [results.json](results.json#arms/claude-haiku-4-5/verified_claim_group_coverage) |
| citation_invariants | PASSED | — | — | [results.json](results.json#arms/claude-haiku-4-5/citation_invariants) |
| severity | PASSED | 15/15 | 0.9 | [results.json](results.json#arms/claude-haiku-4-5/severity) |
| chart_spec | PASSED | 11/11 | — | [results.json](results.json#arms/claude-haiku-4-5/chart_spec) |
| chart_faithfulness | PASSED | 1320/1320 | — | [results.json](results.json#arms/claude-haiku-4-5/chart_faithfulness) |
| chart_refusal | PASSED | 3/3 | — | [results.json](results.json#arms/claude-haiku-4-5/chart_refusal) |
| voices_separation | PASSED | 0/0 | — | [results.json](results.json#arms/claude-haiku-4-5/voices_separation) |

Skipped-visibly:

- chart_spec / chart-15-flagship-spec-validation-refusal-of-commitment — Binding #117 constraint (issue #20 comment): committed fixtures must exclude flagship expected-values derived from Kaufman/Bereiter (open-provisional) until #23's written confirmations arrive. Today the real manifest blocks both flagship splice pairs (require_renderable_splice_pair names the provisional member and issue #23), so the gold behaviour is refusal-of-commitment; the expected-values fixture for the real flagship is a recorded gap in evals/gold/COVERAGE.md, not a silently absent item. chart-02 and chart-06 keep the flagship's transform arithmetic (splice, BP->CE, rebaseline, overlap) under fixture coverage with synthetic data meanwhile.
