# Let's Talk About the Climate Emergency

A free, open-source, public-benefit chatbot that gives people the emergency briefing on climate they have never had: straight answers grounded strictly in authoritative publications, with inline citations to the exact source passages — plus the ability to generate shareable, source-stamped charts from canonical climate datasets.

## What this is

This is an **educational piece of software for public benefit**. It is free to use, carries no advertising, sells nothing, and its code and evaluation results are public. Chris McWilliams (Rusty Data) is the author and owner of this project, and may point to it as his own work, but the product itself is **non-commercial** — several key sources are ingested only on that basis (see `DESIGN.md` §2.1 for the licensing consequence: if the project ever becomes commercial, every non-commercial-licensed document must be removed from the corpus).

**Not affiliated with or endorsed by** the National Emergency Briefing campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or the IPCC. All sources cited and linked.

## Status

Phase 0 — repo scaffolding. See `issues/` for the full build plan and `DESIGN.md` §10 for the roadmap and release gates.

## Documentation

- [`DESIGN.md`](DESIGN.md) — design document: mission, corpus & ingestion, RAG pipeline, guardrails, evaluation, tech stack, deployment (source of truth for scope and architecture).
- [`DECISIONS.md`](DECISIONS.md) — architecture decision record (ADR) log.
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — TDD implementation design: module boundaries, test-first workflow, test pyramid, CI stages.
- [`ORCHESTRATION.md`](ORCHESTRATION.md) — autonomous build methodology: roles, the per-issue loop, review process.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — version-control protocol.

## Techniques

The design commits this project to a set of industry-recognised techniques. Each line below links to the `DESIGN.md` section or ADR that specifies it — the linked text, not this list, is the source of truth. The project is early in its build (see [Status](#status)): except where a line says "in the merged code today", these are design commitments being implemented test-first, not descriptions of finished features.

- **Hybrid retrieval** (dense + learned sparse + Reciprocal Rank Fusion) — dense and learned-sparse term-weight signals from a single embedding model (bge-m3), fused with RRF server-side in the vector store; deliberately not bolted-on BM25. Specified in [DESIGN §3.2](DESIGN.md#32-retrieval), [ADR-006](DECISIONS.md#adr-006--hybrid-retrieval-dense--sparse-rrf-with-the-cross-encoder-reranker-in-the-mvp), [ADR-007](DECISIONS.md#adr-007--vector-store-qdrant).
- **Cross-encoder reranking** (two-stage retrieval) — bge-reranker-v2-m3 over the fused candidate set; the reranker's query-comparable scores are load-bearing because they drive the refusal gate, not just precision. Specified in [DESIGN §3.2](DESIGN.md#32-retrieval), [ADR-006](DECISIONS.md#adr-006--hybrid-retrieval-dense--sparse-rrf-with-the-cross-encoder-reranker-in-the-mvp).
- **Selective answering / calibrated refusal** — the system abstains when retrieval confidence falls below a threshold calibrated on a held-out no-answer set, with CI-gated refusal and false-refusal rates. Specified in [DESIGN §3.5](DESIGN.md#35-refusal--uncertainty-behaviour-unchanged-from-v2), [ADR-010](DECISIONS.md#adr-010--refusal-when-unsupported-and-calibrated-language-fidelity-as-measured-behaviours).
- **Contextual chunk headers** — every chunk carries a prepended header situating it in its source document (the header variant of Anthropic's contextual retrieval), on top of structure-aware chunking that never crosses headings. Specified in [DESIGN §2.4](DESIGN.md#24-ingestion-pipeline-unchanged-from-v2-in-mechanism), [ADR-011](DECISIONS.md#adr-011--parsing--chunking-docling--structure-aware--statement-id-chunking-pymupdf-fallback); the chunker was validated in the merged parsing spike.
- **Grounded generation with native citations** — answers cite retrieved text via API-native citations over custom-content document blocks, so every citation mechanically resolves to a passage; the citable unit is defined by us so a calibrated qualifier can never be split from its claim. Specified in [DESIGN §3.3](DESIGN.md#33-grounded-generation-with-inline-citations-unchanged-mechanism-updated-prompt)–[§3.4](DESIGN.md#34-native-citations-constraints-unchanged), [ADR-008](DECISIONS.md#adr-008--citations-claude-native-citations-over-custom-content-document-blocks).
- **Runtime groundedness verification** — a post-generation check of every sentence against its cited blocks; sentences whose citations do not support them are badged "unverified" in the UI, and the failure rate is published rather than hidden. Specified in [DESIGN §3.3](DESIGN.md#33-grounded-generation-with-inline-citations-unchanged-mechanism-updated-prompt).
- **LLM-as-judge evaluation** — faithfulness, confidence-level-fidelity, and severity-fidelity judges with a disjoint judge model, human-annotated ordinal gold labels, sampled human audit, and hard release gates. Specified in [DESIGN §6.2](DESIGN.md#62-metrics), [ADR-014](DECISIONS.md#adr-014--evaluation-as-a-first-class-published-deliverable).
- **Layered guardrails, enforced in code where possible** — corpus curation as the first guardrail; misleading-graphics rules living in the chart validator/renderer; the evidence/advocacy separation as a structural filter verified by a deterministic eval, never a prompt hope. Specified in [DESIGN §4](DESIGN.md#4-faithfulness--guardrails).
- **Eval-driven development / CI evals** — gold sets with gold chunk-ids, deterministic retrieval metrics blocking every PR, judge gates blocking every release, and results published in-repo. Specified in [DESIGN §6](DESIGN.md#6-evaluation-first-class-published-in-ci), [ADR-014](DECISIONS.md#adr-014--evaluation-as-a-first-class-published-deliverable), [IMPLEMENTATION §6](IMPLEMENTATION.md#6-ci-stages--what-blocks-what).
- **Red-teaming / adversarial evaluation** — a standing adversarial gold subset: denialist framings from the Skeptical Science myth taxonomy, cherry-pick chart requests, prompt-injection probes, and severity-bait phrasings, regression-tested on every release. Specified in [DESIGN §6.1](DESIGN.md#61-eval-datasets-mvp).
- **Structured outputs** (schema-constrained generation) — the query rewriter, scope classifier, and chart planner are schema-constrained calls, and ChartSpec is additionally validated in code. Specified in [DESIGN §3.1](DESIGN.md#31-query-processing), [§3.7](DESIGN.md#37-chart-generation-new), [ADR-020](DECISIONS.md#adr-020--chart-generation-curated-data-pack--closed-declarative-chartspec--server-side-rendering-never-llm-code-generation).
- **Prompt caching** — the cost model specifies caching of the static system prefix to hold per-query cost down; cache effectiveness is to be verified by a smoke check, not assumed. Specified in [DESIGN §9](DESIGN.md#9-deployment--cost), [ADR-012](DECISIONS.md#adr-012--model-anthropic-cloud-haiku-default-opus-gated-behind-a-thin-provider-adapter).
- **Fail-closed cost governance** (denial-of-wallet defence with designed degradation) — a server-side hard daily spend cap that fails closed to a read-only cached state, an Opus sub-cap, and per-IP limits; when the budget dies, the front door serves pre-generated briefings instead of going dark. Specified in [DESIGN §9](DESIGN.md#9-deployment--cost), [ADR-015](DECISIONS.md#adr-015--hard-daily-budget-cut-off-that-fails-closed).
- **Pre-generation / response caching of known queries** — the starter-topic answers and flagship charts are generated at release time and served from cache, clearly dated. Specified in [DESIGN §9](DESIGN.md#9-deployment--cost).
- **Human-in-the-loop data governance** (licensing gates, provenance manifests) — every document passes a layered licensing gate (multi-source automated candidate filter, publisher-page evidence capture, named human sign-off) and the build refuses to index anything without it. Specified in [DESIGN §2.1](DESIGN.md#21-permission-tiers-researched-2026-08-re-verify-at-ingest-per-document)–[§2.2](DESIGN.md#22-the-cc-by-licensing-gate-unchanged-from-v2-hardened), [ADR-003](DECISIONS.md#adr-003--hardened-cc-by-licensing-gate-2-sources--publisher-page-confirmation--human-sign-off).
- **Model-agnostic provider adapter** — a thin, honestly-scoped abstraction isolating the LLM vendor; the adapter protocol, its contract validators, and its test doubles are in the merged code today (`rag/provider.py`). Specified in [DESIGN §5](DESIGN.md#5-model-agnostic-architecture-unchanged), [ADR-012](DECISIONS.md#adr-012--model-anthropic-cloud-haiku-default-opus-gated-behind-a-thin-provider-adapter).
- **Offline A/B evaluation harness** — every pipeline change is compared against the eval suite before it ships, with golden chunk fixtures making chunker changes reviewable. Specified in [DESIGN §6.3](DESIGN.md#63-process--publication-unchanged).
- **Record/replay LLM testing** (deterministic test doubles for a nondeterministic dependency) — programmable fakes, checked-in replay fixtures keyed by a canonical request hash that invalidate when prompts change, and pure contract tests enforcing the API's real constraints before any network call; in the merged code today (`rag/provider.py`, `tests/`). Specified in [IMPLEMENTATION §4](IMPLEMENTATION.md#4-tdd-around-llm-calls--the-seams).

### What we deliberately don't do

Reasoned rejections, each recorded where linked:

- **No fine-tuned "climate model"** — parametric knowledge cannot be cited, cannot be updated without retraining, cannot be licence-governed by any manifest, and flattens the IPCC's calibrated language into the model's own confident register; retrieval keeps knowledge inspectable, versioned, and refusable. [ADR-004](DECISIONS.md#adr-004--rag-grounding-not-a-fine-tuned-climate-model).
- **No open web search** — charting or citing whatever a search engine returns imports unvetted claims, unknown licences, and a prompt-injection vector; the system refuses honestly and the corpus grows demand-driven with a human in the loop. [ADR-021](DECISIONS.md#adr-021--no-open-web-search-allowlisted-provider-fetch-as-the-phase-2-fallback).
- **No LLM code generation for charts** — the model plans a small declarative ChartSpec over a closed, code-validated vocabulary and a server-side renderer makes the pixels; no generated code ever executes, and chart-integrity rules live where the model cannot negotiate with them. [ADR-020](DECISIONS.md#adr-020--chart-generation-curated-data-pack--closed-declarative-chartspec--server-side-rendering-never-llm-code-generation).
- **No GraphRAG** — the corpus is small, hand-audited assessment literature whose authors already did the cross-document synthesis a graph layer tries to recover mechanically; graph extraction would add an LLM-heavy ingest stage, a second store, and a new hallucination surface for no measured benefit. [Rejection recorded in the 2026-08 portfolio review](reviews/sota-portfolio-review-2026-08.md#b7-graphrag--reject-write-the-rejection-up).
- **No RAG framework** (LangChain/LlamaIndex) — the interesting parts of this pipeline (statement-ID chunking, custom-content citation blocks, reranker-thresholded refusal) are exactly the layers frameworks abstract away; hand-rolling them is less code than configuring around a framework's opinions, and easier to read. [ADR-016](DECISIONS.md#adr-016--core-stack-python-312--fastapi-single-public-monorepo).
- **No managed vector database or API embeddings** — published eval numbers must stay reproducible, and a vendor retiring an embedding model silently invalidates them; embeddings and reranking run locally on open weights pinned by revision, over OSS Qdrant. [ADR-005](DECISIONS.md#adr-005--embeddings-bge-m3-run-locally), [ADR-007](DECISIONS.md#adr-007--vector-store-qdrant).
- **No "can't hallucinate" claims** — native citations guarantee that citations resolve to real retrieved text, not that the text entails the claim; resolution is guaranteed by construction, entailment is measured and published, with runtime "unverified" badges as the backstop. [ADR-009](DECISIONS.md#adr-009--guaranteed-vs-measured-no-cant-hallucinate-claim).
- **No broad web corpus** — RAG faithfully repeats its corpus, so curation is the first guardrail: only assessed, consensus-grade, clearly-licensed publications are admitted, small enough to hand-audit every document; contested material informs the adversarial eval set, never the corpus. [ADR-002](DECISIONS.md#adr-002--source-authority-over-coverage).
- **No local open-weight generation for the MVP** — the headline behaviour (mechanically verifiable citations) is an Anthropic API feature, and small local models are measurably worse at exactly this product's failure mode; the coupling is contained behind the provider adapter with a designed fallback. [ADR-012](DECISIONS.md#adr-012--model-anthropic-cloud-haiku-default-opus-gated-behind-a-thin-provider-adapter).
- **No query expansion / HyDE at MVP** — with hybrid retrieval and a cross-encoder reranker in place there is limited headroom for expansion stacks, HyDE risks off-corpus drift on specialist material, and every variant adds an LLM call of latency and cost to every query. [Rejection recorded in the 2026-08 portfolio review](reviews/sota-portfolio-review-2026-08.md#b3-query-expansion--hyde--multi-query-retrieval--reject-for-mvp-evidence-backed-keep-one-narrow-piece).
- **No "agentic" branding without agentic substance** — the pipeline is a fixed, auditable DAG with bounded cost and latency per query (a hard requirement under a fail-closed spend cap), not an agent loop; selective query decomposition is held back until the multi-hop evals demand it. [Deferral recorded in the 2026-08 portfolio review](reviews/sota-portfolio-review-2026-08.md#b6-agentic-rag--query-decomposition-for-multi-hop-questions--defer-post-mvp-experiment-narrowly-scoped).

## Repo layout

| Path | Purpose |
|---|---|
| `ingestion/` | Manifest validation, licensing gate, fetch, parse, chunk (DESIGN §2) |
| `rag/` | Embedding, indexing, retrieval, rerank, refusal gate, provider adapter (DESIGN §3) |
| `charts/` | Chart data pack, transforms, ChartSpec validation, planner, renderer (DESIGN §3.7) |
| `ui/` | Streamlit UI (DESIGN §7) |
| `service/` | FastAPI service: routes, budget tracker, rate limiter (DESIGN §9) |
| `evals/` | Evaluation harness (DESIGN §6) |
| `corpus/` | Corpus manifest + fetch scripts (manifest only — see `corpus/README.md`) |
| `datasets/` | Chart dataset manifest + fetch scripts (see `datasets/README.md`) |
| `voices/` | The voices layer — campaigns and people communicating the emergency (DESIGN §2.5) |
| `reviews/` | Design-review reports |
| `scripts/` | Repo maintenance scripts (e.g. `publish_issues.py`) |
| `tests/` | pytest suite: unit, `integration`, `smoke`, `live` (IMPLEMENTATION.md §3) |

## Development

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync --all-extras       # install dependencies into .venv, incl. ingestion parse backends
uv run pre-commit install  # install git hooks (ruff lint/format)
uv run pytest              # unit tests (default); add -m integration / -m smoke for the other tiers
docker compose up          # start the api, qdrant and ui stub services
```

## Licence

Code is licensed under [Apache-2.0](LICENSE). Corpus and dataset text are governed by the per-document licensing terms recorded in the manifests (`corpus/`, `datasets/`) — see `DESIGN.md` §2.1; nothing outside the `open` permission tier is redistributed in this repository.
