# IMPLEMENTATION.md — TDD implementation design

**Companion to DESIGN.md v3.1 and DECISIONS.md. This document governs how all build work is done: module boundaries, the test-first workflow, what gets tested at which level, and how LLM-dependent code stays deterministically testable. Every issue in `issues/` carries a `## TDD plan` section derived from this document; an implementation agent picking up an issue writes those tests first, in order, before any implementation.**

Status: adopted 2026-08-16. Amend via PR like any design change.

---

## 1. Module map: components → testable units

The rule that makes everything below work: **pure core, imperative shell**. Every module that embodies a design invariant (chunk boundaries, licensing refusals, ChartSpec legality, refusal thresholds, voices filtering, budget caps) is a pure function or a class with injected dependencies. Network, filesystem, clock, model inference and LLM calls live at the edges behind narrow interfaces. If a unit test needs Docker, a model download, or an API key, the seam is in the wrong place.

| Design component (DESIGN §) | Module | Testable unit & seam |
|---|---|---|
| Manifest + licensing invariants (§2.1) | `ingestion/manifest.py` | Pure: load/validate corpus and dataset manifests; every refusal path a unit test. Filesystem paths passed in, never assumed. |
| CC-BY gate (§2.2) | `ingestion/gate.py` | Lookup clients (OpenAlex/Crossref/Unpaywall) and page fetcher injected; decision logic pure over recorded response fixtures. |
| Fetchers (§2.4) | `ingestion/fetch.py` | Transport injected; tests fetch from local `file://` fixture sources; sha256 verification pure. |
| Parsing (§2.4) | `ingestion/parse.py` | `parse_document(path) -> StructuredDoc` interface; Docling and PyMuPDF are two implementations behind it. Tests assert on `StructuredDoc`, not on Docling internals. |
| Chunker (§2.4) | `ingestion/chunk.py` | Pure: `chunk(StructuredDoc, config) -> list[Chunk]`. The showcase bespoke code; the most heavily unit-tested module in the repo. Golden-output fixtures. |
| Citation blocks (§2.4, §3.3) | `ingestion/blocks.py` | Pure: chunks → custom-content block payloads, one per citable unit. |
| Embedding + indexing (§3.2) | `rag/embed.py`, `rag/index.py` | `Embedder` protocol (real bge-m3 vs deterministic stub); Qdrant client injected. Incremental-by-content-hash logic pure. |
| Retrieval + rerank (§3.2) | `rag/retrieve.py`, `rag/rerank.py` | `Reranker` protocol; hybrid-query assembly pure; real models only in integration tests. |
| Voices filter (§3.2) | `rag/filters.py` | Pure function over (classification, chunk list). The structural invariant lives here, nowhere else. |
| Refusal gate (§3.5) | `rag/refusal.py` | Pure function over (reranker scores, threshold). Threshold calibration is a committed script, reproducible from inputs. |
| Provider adapter (§5) | `rag/provider.py` | `ProviderAdapter` protocol: `generate(messages, documents, config) -> AnswerWithCitations`, `structured(messages, schema, config) -> dict`, `plan_chart(request, catalog) -> ChartSpec`. Three implementations: `AnthropicAdapter` (live), `FakeAdapter`, `ReplayAdapter` (§5 below). **Request construction is a pure builder layer**, separately testable — the §3.4 contract tests live against the builders, not the network. |
| Query rewrite + scope classify (§3.1) | `rag/query.py` | Calls through the adapter; canned unsafe responses and routing logic pure. |
| Grounded generation (§3.3) | `rag/generate.py` | Orchestration over adapter + prompt assembly (prompt text is data, its required elements asserted). |
| Citation-support validator (§3.3) | `rag/support.py` | Sentence→cited-block pairing pure; the single batched entailment call through the adapter; badge-event emission pure. |
| Chart data pack (§3.7) | `charts/pack.py` | Loaders + paleo parsers pure over committed sample files. |
| Transforms (§3.7) | `charts/transforms.py` | Pure Pandas: resample, anomaly-vs-baseline, rolling-mean, unit conversion, `time_axis` BP→CE, `splice_series`, `rebaseline_to`. Tested against independently computed fixtures. |
| ChartSpec validator (§3.7) | `charts/spec.py` | Pure: JSON Schema + semantic checks + canonical spec hashing. |
| Chart planner (§3.7) | `charts/planner.py` | Through the adapter; retry-once and nearest-dataset-refusal logic pure. |
| Renderer (§3.7) | `charts/render.py` | Split seam: `spec -> Vega-Lite JSON` is pure (integrity rules, annotations, caption strip, all assertable on the JSON/SVG text); `vl-convert` invocation is the thin side-effect edge. Alt text (`charts/alt_text.py`) pure from spec. The artefact entry point consumes only `charts.spec.validate_spec_for_render`'s `RenderValidatedSpec` (extents mandatory and complete — review finding #133), never a bare spec mapping. |
| UI (§7) | `ui/` | Streamlit files stay thin; all decision logic (chip mapping, badge application, excerpt bounding, voices styling flags) in `ui/presenters.py`, pure. |
| Eval harness (§6) | `evals/` | Metric computations pure over run records (unit-testable with synthetic runs); judges through the adapter, live-only (§5.4). |
| FastAPI service (§9) | `service/` | Routes thin; budget tracker and retention logic clock-injected; rate limiter salt-rotation pure; log redaction pure. |

## 2. Red–green–refactor, wired into the VC protocol

CONTRIBUTING.md governs; this section instantiates it for TDD.

1. **Branch:** one topic branch per issue, `issue-<n>-<slug>`, off `main`. Never stack build work on another issue's branch.
2. **Red:** the first commit(s) on the branch add the issue's TDD-plan tests, failing. Subject: `Add failing tests for <behaviour> (#<n>)`. Body: which acceptance criterion each named test encodes, and confirmation they fail for the right reason (assertion, not import error).
3. **Green:** implement the minimum to pass. Subject: `Implement <unit> to pass <named tests> (#<n>)`. Body: what changed and why — per CONTRIBUTING, verbose and why-focused. One atomic change per commit; a multi-part issue alternates red/green per behaviour rather than batching all reds.
4. **Refactor:** separate commits, subject `Refactor <unit>: <what>`, body stating "no behaviour change; tests unchanged and green". Refactors never share a commit with behaviour changes.
5. **CI and merging:** the topic branch may be red mid-cycle — that is what red means. The **PR head must be green** through lint → unit → integration → smoke, and the PR merges to `main` only when the issue's acceptance criteria pass (CONTRIBUTING merge rule). Never commit a skipped/xfail-ed test to `main` to force green; if a test can't pass yet, its issue isn't done.
6. **Spikes are exempt** (#2–#4): exploratory de-risking uses characterisation tests and committed findings, not test-first — see those issues' TDD-plan sections. Spike code is not production code; productionisation (#7, #14, #15) starts from red tests as usual.

## 3. Test pyramid

Markers registered in `pyproject.toml`: `integration`, `smoke`, `live`. Unmarked = unit.

**Unit (seconds, no network, no Docker, no model weights, no API key).** The bulk of the suite. Belongs here: chunker boundary behaviour and golden outputs; manifest invariant refusals; ChartSpec validation; chart transforms vs fixtures; refusal-gate and voices-filter logic; §3.4 contract tests on request builders; structured-output schema validation over replay fixtures; caption/alt-text generation; budget/rate-limit/redaction logic; eval-metric arithmetic on synthetic run records. Runs on every commit locally and first in CI.

**Integration (minutes, Docker + local model weights, still no LLM key).** Belongs here: Qdrant in Docker — index build, hybrid RRF query, metadata round-trip; real bge-m3/bge-reranker over the tiny fixture corpus (weights cached in CI); end-to-end ingestion of the fixture corpus from `file://` sources through manifest gate → parse → chunk → blocks; renderer output via vl-convert — structural SVG snapshot tests (assert on SVG text/elements, never PNG byte-equality) and rendered-value checks against gold fixtures; FastAPI test-client flows including the budget-breach fail-closed path with a `FakeAdapter`.

**E2E smoke (thin).** `docker compose up`; health checks; one chat query and one chart request driven end-to-end with the `ReplayAdapter` so the run is deterministic and free; paused-state check. A handful of `live`-marked tests (one real cited generation call, one real classifier call) run per release and on demand, never per PR.

The pyramid is enforced by cost: if a test needs a live LLM to assert a *code* behaviour, the seam is missing — fix the seam, don't promote the test.

## 4. TDD around LLM calls — the seams

LLM calls are non-deterministic, slow, and cost money. Nothing in the unit or integration tiers may make one. Four mechanisms, built in issue #24:

**4.1 FakeAdapter (programmable).** Implements `ProviderAdapter`; returns whatever the test programs, records every call (method, request payload). Used to assert *our* behaviour around the model: the unsafe path makes zero generation calls; the citation-support validator makes exactly one batched call; the paused service makes none; retry-once-then-refuse sequences.

**4.2 ReplayAdapter (recorded fixtures, checked in).** Responses recorded once from the live API into `tests/fixtures/replay/` (JSON, keyed by a canonical request hash; scrubbed of keys/headers), replayed deterministically forever. Recording mode requires an explicit env flag and a live key; replay mode raises loudly on any unrecorded request — a changed prompt invalidates its recordings by design, forcing a conscious re-record commit. Used for: parsing real response *shapes* (citation deltas, streaming event order, cache-usage metadata), structured-output happy paths, and regression-pinning specific behaviours (the cherry-pick chart request, the flagship planner request).

**4.3 Contract tests for the native-citations constraints (§3.4).** Pure tests against the request builders, run in the unit tier, failing before any network call is possible:
- the generation request **never** carries structured-output/tool-choice configuration;
- the generation request **never** contains more than 8 documents;
- **never** mixed cited/uncited blocks — every document block has `citations: {enabled: true}`;
- structured-output calls (rewriter, classifier, planner, judges) **never** enable citations;
- violation raises before the adapter's transport is touched (asserted with a FakeAdapter that counts calls).

Schema-validation tests cover every structured-output call site: valid replay fixtures parse; malformed fixtures exercise the retry/failure path.

**4.4 Evals are not tests — don't blur the two.** The LLM-judge evals of §6 (faithfulness, confidence-level fidelity, severity fidelity, adversarial rubric, citation-support rate) and the live-model accuracy of the classifier and planner are **release gates run against the live model** by the eval harness (#21), per release/corpus version. They measure model behaviour; they are non-deterministic; they cost money; they do not run under pytest's unit/integration tiers and they never block a PR merge (only a release). Conversely, unit tests never assert answer *quality* via a faked judge — a FakeAdapter proves plumbing (the judge was called with the right pairs, the gate arithmetic applied its thresholds), never that answers are good. The harness's *arithmetic* (gate thresholds, ordinal-agreement computation, tolerance comparisons) is pure and fully unit-tested against synthetic run records.

## 5. Fixtures strategy

All fixtures live under `tests/fixtures/` and are small enough to read in review.

- **Synthetic corpus** (`tests/fixtures/corpus/`): 4–6 short documents **authored fresh for this repo** — fictional but structurally realistic: nested headings, a table, footnotes, calibrated-language sentences ("very likely", "high confidence", a "not likely" negation trap), one document flagged `consensus_position: beyond-assessed-range`, one `source_type: voices` document, one headline-statements document. Every fixture doc carries a first-line marker `SYNTHETIC FIXTURE — authored for this project's tests`, and a meta-test asserts the marker's presence — **no real Tier B/C text is ever committed as a fixture** (that would violate the shipping invariant, DESIGN §2.1). A fixture manifest covers every `permitted_context` value plus one deliberate violation per invariant.
- **Synthetic data fixtures** (`tests/fixtures/spike/`, `tests/fixtures/datasets/raw/`): tabular fixtures that mimic a named real source (Bereiter, Mauna Loa, GISTEMP, Kaufman, OWID, HadCRUT) reproduce the *byte shape* — BOM/CRLF, `#`-prefixed headers, `***` missing-value convention, column layout — but **never a real data row**. The `SYNTHETIC FIXTURE — all values invented` marker is not self-enforcing (a pasted real row passes it), so review finding #51 requires every such fixture to also declare its **row-copy posture** in the header, in one of two forms (a meta-test, `test_review51_fixture_honesty.py`, enforces the choice): (a) **the no-copy posture (preferred)** — perturb every value so no row matches the archive, and state `No rows are copied from the real file (review finding #51)`; or (b) **the explicit-copy posture** — only when a row genuinely must be verbatim, state exactly which rows are copied and under what public-domain/open licence. The parsers do not need real values (transform tests use inline synthetic frames), so posture (a) is the default; posture (b) is a last resort backed by a licence check.
- **Golden chunk outputs:** committed expected chunk lists (ids, section paths, headers, boundaries) for the fixture docs. A chunker change that alters goldens must update them in the same commit with the diff explained — the "chunking-experiment" scope guard made mechanical.
- **Chart gold fixtures:** expected rendered values computed by an **independent script** (`evals/scripts/compute_chart_fixtures.py`) that uses numpy/stdlib directly and — enforced by an import-graph test — imports nothing from `charts/`. Tolerances: 1e-9 relative pass-through, 1e-6 post-transform. Tiny synthetic CSVs for transform unit tests; real pack data only at integration/eval level.
- **Licensing-gate regression fixtures:** recorded lookup responses (OpenAlex/Crossref/Unpaywall) and publisher-page HTML for: a clean CC-BY case, a three-way-disagreement case, a hybrid-journal trap, and **the free-to-read-but-not-CC trap** (Ripple 2019, 10.1093/biosci/biz088) which must be rejected.
- **LLM replay fixtures:** §4.2. Include at least one recorded response with a *non-entailing* citation for the citation-support validator, and one malformed structured output.

## 6. CI stages — what blocks what

| Stage | Runs | When | Blocks |
|---|---|---|---|
| lint | ruff + pre-commit hooks | every PR push | merge |
| unit | pytest, unmarked | every PR push | merge |
| integration | pytest `-m integration` (Docker Qdrant, cached weights, fixture corpus) | every PR push | merge |
| smoke | compose up + replay-driven e2e | every PR push | merge |
| full eval suite | #21 harness: deterministic metrics on gold sets + live LLM-judge gates | per release / corpus version; on demand | **release** |
| scheduled | live dataset fetch + sha256 verification; `live`-marked API tests | weekly / pre-release | release |

A PR merges when lint→smoke are green and the issue's acceptance criteria pass. A **release** additionally requires every DESIGN §10 gate: faithfulness/citation-support targets, refusal >90% / false-refusal <5%, severity gate (≥90% exact-or-adjacent, zero two-level errors), severity-retrieval recall, chart data-faithfulness 100% vs fixtures, voices-separation 100%, fail-closed cut-off verified. Eval-gate failures block the release build, never retroactively a merged PR.

## 7. Definition of done, per issue

An issue is done when:
1. Every acceptance criterion is either (a) expressed as one or more named automated tests, written **before** the implementation they verify (visible in branch history as red→green), or (b) explicitly a process/manual item with committed evidence (findings note, checklist, recorded audit) — the issue's TDD plan says which.
2. The TDD-plan tests exist at the stated tier and pass; no test was weakened to pass.
3. New seams follow §1 (no LLM/network/clock reach into pure logic).
4. Branch merged to `main` per CONTRIBUTING; branch history shows the red/green/refactor cadence.

**Coverage stance:** no numeric coverage threshold — vanity percentages reward asserting the easy lines. The pragmatic bar: every design invariant named in DESIGN §2.1, §3.2, §3.4, §3.7 and §9 has at least one test that fails when the invariant is broken, and every bug fixed gets a regression test in the same PR. Review enforces this, not a coverage gate.
