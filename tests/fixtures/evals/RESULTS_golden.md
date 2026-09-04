# Release eval results

Generated: 2026-08-23 (results schema v1)

## Release verdict: BLOCKED

Production model: NONE SELECTED

## Arm: claude-haiku-4-5 — run cost $0.36

| gate | status | score | threshold | evidence |
|---|---|---|---|---|
| refusal | PASSED | 19/20 | 0.9 | [results.json](results.json#arms/claude-haiku-4-5/refusal) |
| route_accuracy | PASSED | — | — | [results.json](results.json#arms/claude-haiku-4-5/route_accuracy) |
| citation_support | BLOCKED | — | — | citation-support validation never executed for this run (#303) |
| severity | BLOCKED | — | — | owner severity audit pending (finding #197) |
| chart_spec | PASSED | 2/2 | — | [results.json](results.json#arms/claude-haiku-4-5/chart_spec) |

Skipped-visibly:

- chart_spec / chart-15-flagship-spec-validation-refusal-of-commitment — fixtures embargoed until issue #23 (#117)
