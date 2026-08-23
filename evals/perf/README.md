# PROJECT-OPERATIONAL DATA — first-party record, no external licence

# evals/perf/ — rerank latency evidence (review finding #176)

The perf-evidence home. `rag.retrieval.record_rerank_latency` appends one
CSV row per measured run to `rerank-latency.csv` here (or to the path in
the `CLIMATE_CHAT_PERF_LOG` env var when set), so "rerank latency
measured and recorded" survives the run that produced it: locally the
file accumulates on disk, and CI uploads it as a build artifact from the
integration job.

Conventions:

- Generated logs (`*.csv`) are **gitignored** — recorded on disk and in
  CI artifacts, never committed by accident (ADR-023 spirit).
- Any *deliberately* committed sample must carry the ADR-023
  scope-clarification first-line marker —
  `PROJECT-OPERATIONAL DATA — first-party record, no external licence` —
  the same convention as `evals/spend-ledger.csv`.
- The measured workload is 40 **distinct** chunks at the chunker's max
  word budget (`ingestion.chunk.ChunkConfig.max_tokens`), each
  tokenising past the reranker's pair cap so the windowed scoring path
  (finding #175) is what gets measured — never repeated short toy pairs.
- The ~100 ms budget (`RERANK_LATENCY_BUDGET_SECONDS`, ADR-006) is
  documented in every row and **asserted only** when
  `CLIMATE_CHAT_PERF_PROFILE=demo` is declared (issue #11 acceptance
  criteria); other hardware records honest evidence without gating.
