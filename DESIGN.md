# Ask About the Climate — Design Document v2 (build-ready)

**An open-source, retrieval-augmented chatbot that answers questions about climate change, grounded strictly in openly-licensed authoritative publications, with inline citations to the exact source passages.**

- **Client / owner:** Chris McWilliams (Rusty Data)
- **Status:** Design v2 — 2026-08 — hand-off spec for a build agent
- **Code licence:** Apache-2.0 (explicit patent grant)
- **Repo shape:** single public monorepo — `ingestion/`, `rag/`, `ui/`, `evals/`, `corpus/` (manifest only)
- **Changes vs v1:** legal footing corrected (IPCC deferred behind written permission); CC-BY licensing gate hardened (multi-source + publisher-page confirm + human sign-off); citation guarantees stated precisely (no "can't hallucinate" overclaim); reranker moved into the MVP; Claude custom-content citation blocks; native-citations constraints documented; calibration check reframed as a measured proxy plus an LLM-judge level check; statement-ID chunking flagged as schedule long-pole; hard daily budget cut-off.

> **Framing (read first, kept consistent throughout).** This is a **portfolio piece for Rusty Data**, i.e. a **commercial** context (its purpose includes winning client work). We therefore do **not** rely on any non-commercial research exception, and we do **not** self-certify fair use / UK CDPA s.29A text-and-data-mining (that exception is non-commercial-research only). The product **launches on a "safe tier" of clearly-permitted sources**. IPCC content is powerful and desirable but is **copyright IPCC and not openly licensed**; it is added **only after** a written permission request has been **answered affirmatively** (and, given the commercial framing, ideally a qualified UK legal review). Every place that used to lean on "educational/non-commercial" has been corrected to this position.

---

## 1. Vision, mission & audience

### Mission

People make better decisions about the climate when they are well-informed, and the best synthesis of the evidence already exists in authoritative public reports and open-access reviews. *Ask About the Climate* makes that body of knowledge conversational **without diluting it**: every answer is assembled from retrieved passages of openly-licensed authoritative publications, cited inline, with the retrieved text one click away. If the corpus doesn't support an answer, the bot says so.

Anti-misinformation principles that drive every design decision:

1. **Source authority over coverage.** A small corpus of assessed, consensus-grade, clearly-licensed literature beats a large corpus of blogs and news.
2. **Epistemic honesty over fluency.** The bot preserves the sources' own calibrated uncertainty (e.g. IPCC-style likelihood/confidence language, once IPCC is in scope; and the calibrated language present in NCA5 and review papers) rather than flattening it into false certainty in either direction.
3. **Verifiability over convenience.** Citations point to document, section and page/paragraph; the UI shows the retrieved passages.
4. **The bot is a librarian, not an oracle.** It routes people to the evidence and is candid about what it can and cannot guarantee (see section 4).

### Audience
| Audience | Need | Design implication |
|---|---|---|
| General public | Plain answers to "is X true?", "how bad is Y?" | Readable answers, jargon explained, myths handled calmly |
| Students & educators | Traceable claims | Copyable citations; source library |
| Journalists | Fast, quotable, checkable facts | Exact page/section references; retrieved-passages panel |
| Policymakers & analysts | Impacts, adaptation, mitigation | Coverage grows as corpus tiers unlock |

### Portfolio goals
Flagship demonstration of end-to-end RAG, chatbot UX, AI integration, responsible-AI guardrails, and **evaluation as a first-class, published deliverable**. Calibrated honesty about what is guaranteed vs measured is itself part of the pitch.

---

## 2. Corpus & ingestion

### 2.1 Source tiers and licensing (researched; re-verify before launch)

Licensing governs what we may **ingest / index**, **quote**, and **redistribute**. Note that **building a private full-text index is itself reproduction of the whole work**, not merely a display act — so indexing rights, not just display rights, decide what is in scope.

**SAFE TIER — ships in the MVP (clearly permitted for commercial reuse):**

| Source | Licence / terms (verified 2026-08) | Use |
|---|---|---|
| **US National Climate Assessment (NCA5)** | US federal government work — **public domain** in the US | Ingest, index, quote, redistribute freely; attribute anyway. Strong impacts/adaptation coverage. |
| **NASA** climate content (climate.nasa.gov, Earthdata explainers) | Generally **not copyrighted**; NASA-mission data default **CC0** unless individually marked ([Earthdata data-use guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance)); citation urged | Ingest text; attribute; **exclude any item marked with a restriction or third-party credit**. |
| **NOAA** (climate.gov, NCEI explainers) | US government work — **public domain** ([climate.gov FAQs](https://www.climate.gov/faqs), [NOAA NAO 205-17A](https://www.noaa.gov/organization/administration/nao-205-17a-information-access-dissemination)); third-party-credited items excluded | Ingest explainers; attribute; exclude third-party-credited items. |
| **Copernicus / C3S** (Climate Data Store docs, European State of the Climate) | [Licence to Use Copernicus Products](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products): free reuse and redistribution, **commercial use permitted**, conditional on visible attribution — *"Generated using Copernicus Climate Change Service information [Year]"* ([citation guidance](https://confluence.ecmwf.int/pages/viewpage.action?pageId=222484359)) | Ingest report text; carry the exact attribution string into citation metadata. |
| **Hand-verified CC-BY / CC0 review & synthesis papers** | Per-item, via the hardened gate in section 2.2. CC BY and CC0 permit commercial reuse + redistribution with attribution. | Ingest only items that pass the gate; store DOI + confirmed licence + evidence. |

**DEFERRED TIER — added only after conditions are met:**

| Source | Status | Gate to add |
|---|---|---|
| **IPCC AR6** (WG1/2/3, SYR), **SR1.5**, **SROCC**, **SRCCL** — SPMs, TS, chapters, glossary | **Copyright IPCC, NOT openly licensed.** Per [ipcc.ch/copyright](https://www.ipcc.ch/copyright/), reproduction of *limited figures or short excerpts* is allowed with acknowledgement, but **building a private full-text index of whole reports is reproduction of the whole work** and is not covered by that allowance; broader reproduction needs written permission from the IPCC Secretariat. Because this project is commercial, we cannot lean on any non-commercial TDM/research exception. | (a) Email the IPCC Secretariat describing the project **as a commercial portfolio product** and requesting written permission to index full text for retrieval with short-excerpt display + deep links; (b) **written affirmative reply received**; (c) ideally a qualified UK legal sign-off. **Only then** does IPCC ingestion code run and IPCC enter the live corpus. Until then: **link out** to ipcc.ch, do not index. |

**Out of scope for the corpus (any tier):** news articles, advocacy sites, blogs (e.g. Carbon Brief is CC BY-**NC-ND** — link-only), preprints, paywalled papers, social media, AI-generated summaries. Skeptical Science's myth taxonomy informs the **adversarial eval set only**, never the corpus.

**Licensing invariants (enforced in code):**
- Every document record carries `licence`, `licence_evidence`, `attribution_text`, `canonical_url`, `redistributable: bool`, `commercial_use_ok: bool`, `sha256`, `retrieved_at`, `human_signoff`.
- The build **refuses to index** any document whose `commercial_use_ok` is not `true` or whose `human_signoff` is empty.
- Public-domain and confirmed CC-BY/CC0 text may ship as a prepared dataset in the repo; anything deferred ships as **manifest + fetch scripts only**, never as re-hosted text.

### 2.2 The CC-BY licensing gate (hardened — no single-lookup authorization)

Automated licence lookups disagree often (OpenAlex / Crossref / Unpaywall agree only ~63% of the time), so **a single automated lookup is a candidate filter, never authorization.** To admit a paper:

1. **Candidate filter (automated):** query OpenAlex + Crossref + Unpaywall. Require **at least 2 sources agreeing** the licence is CC BY / CC BY-SA / CC0 to become a *candidate*. Disagreement -> reject candidate.
2. **Publisher-page confirmation (automated fetch, human-checked):** fetch the **actual article page / PDF** and confirm the licence statement there (the canonical source of truth), capturing the exact licence text and URL into `licence_evidence`.
3. **Human sign-off (required):** a person reviews the candidate + evidence and records `human_signoff: {who, date, note}` per document. No document enters the index without this.

The gate's decision, evidence, and sign-off are committed to the manifest so the corpus is auditable.

### 2.3 MVP corpus vs later

| | MVP corpus (safe tier) | Later (post-gates) |
|---|---|---|
| Gov/agency | NCA5 key chapters + NASA/NOAA explainers + C3S ESOTC (~150-300 pages) | Full NCA5, more C3S |
| Reviews | ~10-15 hand-verified CC-BY/CC0 reviews covering core topics (attribution, tipping points, impacts, mitigation) | Curated CC-BY set via the DOI pipeline (**deferred**) |
| IPCC | **none at launch** (link-out only) | AR6 SPMs/TS/chapters **after written permission** |
| Size | ~500-1,000 pages, ~0.7-1.5M tokens | grows per tier |

The MVP corpus is small enough to **hand-audit every document**. That is a feature (trust), not a limitation.

### 2.4 Ingestion pipeline
```
fetch (verify sha256) -> parse -> structure-aware chunk -> attach citation metadata
       -> custom-content citation blocks -> embed -> index
                     \-> manifest (licence + evidence + human_signoff + provenance)
```

1. **Acquisition & manifest.** `corpus/manifest.yaml` lists every document: `canonical_url`, `sha256`, `licence`, `licence_evidence`, `commercial_use_ok`, `human_signoff`, `retrieved_at`, `source_tier`. `make corpus` re-fetches and verifies hashes; refuses to proceed on any item failing the licensing invariants (section 2.1).
2. **Parsing.** HTML sources (NASA/NOAA/C3S/NCA5 web) parse cleanly. PDFs use **Docling** (layout-aware -> structured Markdown) with **PyMuPDF** fallback. Figures/tables -> placeholders; **captions retained** as citable text; figure images are not re-served.
3. **Chunking (structure-aware).**
   - **Web explainers / NCA5:** chunk by section heading, <= ~500 tokens, one-sentence overlap, never across headings.
   - **Papers:** chunk by section (abstract/intro/results/...), <= ~500 tokens.
   - **IPCC SPM statements (deferred code path):** chunk on **assessment-statement boundaries** by their inline IDs (`A.1.2`, `B.3`, ...). **This is the schedule long-pole** — see section 8/section 10. One chunk = one statement + sub-bullets.
   - Every chunk gets a **prepended context header** (document -> section title), embedded with the chunk.
4. **Citation metadata per chunk (schema):**
   ```json
   {
     "chunk_id": "nca5-ch2-3",
     "source_id": "nca5",
     "document": "Fifth National Climate Assessment, Ch.2",
     "section_path": ["Ch2", "2.1", "2.1.3"],
     "page_start": 41, "page_end": 42,
     "statement_id": null,
     "licence": "US Government public domain",
     "commercial_use_ok": true,
     "attribution": "USGCRP, 2023: Fifth National Climate Assessment, Ch.2, pp.41-42",
     "canonical_url": "https://nca2023.globalchange.gov/chapter/2/#section-2.1.3",
     "confidence_markers": ["very likely"],
     "corpus_version": "2026.09"
   }
   ```
   `confidence_markers` extracted at ingest by regex over calibrated vocabulary (*virtually certain, extremely likely, very likely, likely, more likely than not; very high/high/medium/low confidence*). Used by guardrails and eval.
5. **Custom-content citation blocks (see section 3.3).** For each chunk we also emit the **custom-content document block** the generator will receive, so one block == one citable unit (one section chunk; for IPCC later, one SPM statement).
6. **Embedding & indexing.** Idempotent, incremental (re-embed only changed chunks by content hash).

---

## 3. RAG pipeline

```
question
  -> [separate call] query rewrite + scope classify (structured output)
  -> hybrid retrieval (top-40)  -> cross-encoder rerank (top-8, calibrated scores)
  -> refusal gate (rerank score threshold)
  -> [separate call] grounded generation w/ Claude native citations (custom-content blocks)
  -> citation-support validation + calibrated-language check
  -> response + retrieved-passages panel (verbatim)
```

### 3.1 Query processing (kept OFF the generation call — see section 3.4)
- **Query rewriting** (`claude-haiku-4-5`, structured output): resolve conversational references, expand acronyms.
- **Scope classifier** (same or adjacent cheap call, structured output): `in_scope | out_of_scope | adversarial_in_scope | unsafe`. Adversarial-in-scope is routed into normal retrieval with a tone flag (section 4).
- These use **structured outputs** and therefore **must be separate API calls** from the generation call, which uses native citations (the two features are mutually exclusive — section 3.4).

### 3.2 Retrieval
- **Hybrid search:** dense + BM25/sparse, fused with reciprocal-rank fusion (RRF), top-40.
- **Embeddings:** `BAAI/bge-m3` (open weights, hybrid-friendly, CPU-viable at this corpus size, free, reproducible).
- **Reranking — IN THE MVP (moved up from v1).** Cross-encoder `BAAI/bge-reranker-v2-m3` over the top-40 -> top-8. **Rationale for MVP inclusion:** the refuse-when-unsupported gate needs a **calibrated relevance score to threshold on**; **RRF scores are rank-fusion artifacts and are not calibrated for absolute thresholding**, whereas the cross-encoder yields comparable per-passage scores. So the reranker is not just a quality boost — it is what makes the refusal gate trustworthy. ~100 ms CPU for 40 pairs.
- The **refusal gate** (section 3.5) thresholds on the reranker's top score(s).

### 3.3 Grounded generation with inline citations (Claude, custom-content blocks)

**Mechanism.** Pass each retrieved chunk as a **custom-content document block** (`type: "document"`, `source.type: "content"`, an array of text blocks) with `citations: {enabled: true}`. Custom-content blocks make Claude cite **at the granularity of the blocks we define** rather than auto-chunking plain text to sentences.

**Why custom-content and not plain text (fixes a v1 hazard):** with a plain-text document, Claude auto-chunks to sentences, which **can split a calibrated qualifier from its claim** (e.g. "very likely" landing in a different auto-chunk than the statement it modifies), corrupting both the citation and the calibration check. **One custom-content block per citable unit** (one section chunk now; one SPM statement once IPCC is in scope) keeps each calibrated claim intact and citable as a whole.

**What is guaranteed vs measured (corrected overclaim from v1).**
- **Guaranteed by native citations:** every citation the model emits **points to actual text that was in the provided documents** (real `cited_text`, real block/location). The model cannot cite a source that was not retrieved.
- **NOT guaranteed:** that the cited passage **entails** the sentence. Native citations do **not** prevent the model from writing a claim the passage does not actually support and attaching a (real) citation to it. Entailment/faithfulness is **measured**, not guaranteed — see the eval numbers in section 6, which we publish. We market this honestly.
- Backstop: a **citation-support validator** (below) plus the published faithfulness/citation-support scores.

**Citation-support validation (post-generation).** For each factual sentence: it must carry at least 1 citation; each citation must reference a provided block (native citations make this true by construction); an **LLM/NLI entailment check** asks "does the cited block support this sentence?" Sentences failing entailment are flagged "unverified" in the UI and/or trigger one regeneration pass. Aggregate rate is the **citation-support** eval metric (section 6).

**Models.** Generation default `claude-haiku-4-5` ($1/$5 per MTok); `claude-opus-4-8` ($5/$25 per MTok) as an optional "best" mode **behind the budget cut-off** (section 9). Model id is config.

**System prompt core rules:**
1. Answer **only** from the provided passages; no outside knowledge even when confident.
2. Cite every factual claim.
3. **Preserve calibrated language verbatim** — if a source says *likely*, the answer says *likely*; never upgrade, downgrade, or drop a confidence/likelihood qualifier.
4. If passages don't answer the question, say so plainly and point to the right source (canned refusal template).
5. Note source vintage where relevant.
6. Plain language; define jargon on first use.

### 3.4 Native-citations constraints (must be respected by the build)
- **All-or-none:** citations are enabled **for all or none** of the documents in a request. Do not mix cited and uncited document blocks in the same generation call.
- **Incompatible with structured outputs:** a request using native citations **cannot** also use structured/JSON-schema output. Therefore the **scope classifier and query rewriter (structured) run in separate calls** from the generation call (native citations). This is already reflected in section 3.1.
- Keep the generation call's documents to the reranked top-8 to bound cost/latency.

### 3.5 Refusal & uncertainty behaviour
- **No relevant passages** (top reranker score below calibrated threshold): honest refusal — "The sources I have access to don't cover this" + what the corpus does cover. Threshold set empirically on the gold set's no-answer subset (section 6).
- **Partial support:** answer the supported part; name what is unsupported.
- **Contested/uncertain science:** present the assessed range with its confidence level; don't adjudicate beyond sources.
- **Out of scope:** brief redirect, no retrieval.
- Response footer: *"Generated from the cited sources — verify important claims against the originals (linked)."*

### 3.6 "Show the retrieved passages" panel
Every answer ships the top-8 passages verbatim (length-bounded per licence), with attribution line and deep link. Cited passages highlighted; with native citations, the exact `cited_text` span is highlighted within the block. Visible by default on desktop, one tap away on mobile.

---

## 4. Faithfulness & guardrails (critical)

The product-killing failure mode is a single confidently-wrong or miscalibrated answer screenshotted out of context. Layers:

1. **Corpus curation is the first guardrail.** Only assessed/peer-reviewed, clearly-licensed synthesis literature is ingested (section 2). It can't be retrieved if it was never ingested.
2. **Grounding contract** (system prompt + citation-support validation, section 3.3): every claim cited; no outside knowledge; refuse when unsupported.
3. **Calibrated-language fidelity — measured, not guaranteed.** The calibrated scales (likelihood: *virtually certain*/*extremely likely*/*very likely*/*likely*/...; confidence: very high...very low) encode the science's own uncertainty. We enforce and **measure** two things:
   - **Calibrated-term preservation rate (proxy metric):** regex check that calibrated terms present in cited blocks also appear in the corresponding answer sentence. This is a **proxy** (word presence), explicitly not a guarantee of correct meaning.
   - **Confidence-level fidelity (LLM-judge):** a judge checks whether the **confidence *level*** is preserved (not merely the word) — e.g. flags an answer that says "likely" where the source said "very likely", or that drops the qualifier while keeping the claim. This catches level-shift that the regex proxy misses.
   - Custom-content blocks (section 3.3) keep the qualifier attached to its claim so both checks are meaningful.
   - Symmetric honesty: neither soft-pedal strong findings nor overstate uncertain ones.
4. **Bad-faith / denialist prompts.** Policy: **answer from the evidence, calmly, with citations — don't get baited, don't lecture, don't refuse.** "Hasn't warming paused?" gets the assessed answer on variability and observed trends, cited. Loaded framings ("since the hoax...") are not echoed or endorsed; only the factual substance is addressed. Persistent abuse -> polite disengage; no debate-me loops. Regression-tested by the adversarial eval subset (section 6).
5. **Prompt-injection resistance.** Retrieved passages are data, not instructions — delimited as document blocks; system prompt states passage content can never override the rules.
6. **Citation integrity (stated precisely).** Native citations guarantee citations resolve to real retrieved text (no fabricated sources). They do **not** guarantee entailment; entailment is measured and published (section 3.3, section 6). We do **not** claim the system "structurally cannot hallucinate."
7. **Transparency page (`/about`).** What the corpus is (tiers + versions), what's **excluded and why** (IPCC pending permission), how citations work, **what is guaranteed vs measured** with the current eval numbers, known limitations (source vintage), and the full licence/attribution table. This page is linked from the demo and carries the published eval results.

---

## 5. Model-agnostic architecture (interface now; alt backends deferred)

Thin **provider adapter** — single interface `generate(messages, documents, config) -> AnswerWithCitations` — implemented for **Anthropic (native citations via custom-content blocks)** in the MVP. **Deferred:** other-cloud (LiteLLM passthrough) and **local/open-weight** (Ollama/vLLM; the fallback path uses validated `[n]` markers since local models lack native citations). Embeddings + reranker are already local, so the eventual local story is credible; it is simply not in the MVP build target.

---

## 6. Evaluation (first-class, published, in CI)

### 6.1 Eval dataset (MVP)
- **Climate-QA gold set (~40 questions)** over the **safe-tier corpus**, each with a reference answer + **gold `chunk_id`s**. Mix: ~15 single-passage answerable, ~10 multi-passage, ~8 **no-answer-in-corpus** (refusal-expected), ~7 adversarial/denialist framings.
- **Deferred:** synthetic Q/A augmentation at scale.

### 6.2 Metrics (all runnable in CI on the safe-tier corpus)
| Layer | Metric | Notes |
|---|---|---|
| Retrieval | Recall@8, MRR, nDCG@8 vs gold chunk_ids | Deterministic |
| Groundedness / faithfulness | LLM-judge: is every claim entailed by its cited blocks? | Judge != generator; sample human-audited |
| **Citation-support** | % factual sentences whose cited block entails them (NLI/LLM) | This is the number that backs the honest "measured, not guaranteed" claim (section 3.3) |
| **Calibrated-term preservation** | % answers preserving calibrated terms present in cited blocks (regex) | Explicitly labelled a **proxy** |
| **Confidence-level fidelity** | LLM-judge: is the confidence *level* preserved (not just the word)? | Catches level-shift the proxy misses (section 4.3) |
| Refusal quality | Refusal rate on no-answer subset (target >90%); false-refusal on answerable (<5%) | Sets the reranker threshold (section 3.5) |
| Adversarial | Human rubric on denialist items: cites evidence? stays calm? doesn't echo framing? | Manual per release |

### 6.3 Process & publication
- CI on every PR: retrieval metrics + a smoke eval; **full suite per release / per corpus version**.
- Results committed to `evals/RESULTS.md` and **linked from the `/about` page** in the live demo.
- A/B harness for pipeline changes (reranker on/off, chunking, model swap) — claims get numbers.

---

## 7. UX

**Desktop:** chat (centre) + retrieved-passages panel (right, open by default) + source library (left drawer). **Mobile:** chat with per-answer expandable "Sources (n)" sheet.
- **Chat:** streaming answers; inline citation chips that highlight the matching block on hover/tap; calibrated terms styled with a tooltip to the likelihood legend.
- **Passages panel:** verbatim bounded excerpts, attribution lines, deep links, cited-span highlighting.
- **Source library:** every document with licence, version, link — doubles as the attribution surface licences require.
- **Starter questions:**
  - "How much has the planet warmed, and how sure are scientists it's human-caused?"
  - "What's the difference between 1.5C and 2C of warming?"
  - "Is it too late to do anything?"
  - "What does 'net zero' actually require?"
  - "How will climate change affect food and water?"
  - "Didn't the climate change naturally before? Why is this different?"
- **Framing furniture:** footer verification note; `/about` transparency page (with live eval numbers + guaranteed-vs-measured); likelihood-scale legend.
- **Deferred:** source-scope filters, "copy with citations" export.
- Accessibility: keyboard-navigable, semantic HTML, no citation info by colour alone.

---

## 8. Tech stack (with justification)

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.12** | RAG/eval ecosystem; client's data-science context |
| Ingestion | **Docling** + PyMuPDF fallback | Best OSS layout-aware PDF->Markdown; PyMuPDF+regex is the serious fallback for statement-ID parsing (section 10) |
| Chunking/pipeline | Hand-rolled (~300 lines) | Structure-aware / statement-ID chunking is bespoke and is the part worth showcasing; no framework lock-in |
| Embeddings | **bge-m3** (local) | Strong, hybrid-friendly, CPU-viable, free, reproducible |
| Vector store | **Qdrant** (Docker or embedded) | OSS, native hybrid search, free tier fits corpus; LanceDB fallback |
| Reranker | **bge-reranker-v2-m3** | Calibrated scores for the refusal gate (section 3.2); precision win; CPU-OK |
| LLM (cloud) | **Anthropic** `claude-haiku-4-5` default, `claude-opus-4-8` optional | Native citations via custom-content blocks (section 3.3); prompt caching on static prefix; Opus gated by cut-off |
| API service | **FastAPI** (SSE streaming) | Standard, typed, async |
| UI | **Streamlit** (MVP) | Credible chat + passages panel + /about fast; **Next.js deferred** |
| Eval | Custom scripts + `pytest` in CI (Ragas optional for faithfulness) | Citation-support + calibration checks are the novel bits |
| Infra | Docker Compose (`api`, `qdrant`, `ui`) | One-command reproduction |
| Observability | Structured JSON logs (question / retrieved ids / answer / citations; no user identifiers) | Each exchange is a future eval case |

**Deferred stack items:** Next.js UI, Ollama/vLLM local backend, automated CC-BY DOI pipeline, Langfuse.

---

## 9. Deployment & cost

### Embedding the corpus (one-off per version)
Safe-tier corpus (~1M tokens) with local bge-m3 on CPU: **$0**, minutes to a couple of hours. **Verify token count from the manifest before any paid run.** (No paid embedding is needed at MVP scale.)

### Live demo hosting (target < ~$20/month)
| Component | Where | Cost |
|---|---|---|
| Vector DB | Qdrant free tier or co-located on app VM | $0 |
| API + UI (Streamlit) | Fly.io / Railway small VM, or Streamlit Community Cloud / HF Spaces | $0-10/mo |
| LLM calls | Anthropic `claude-haiku-4-5` default | see below |

**Per-query cost (haiku):** ~8 passages x ~400 tok + prompt ~= 5-6K input, ~500 output -> ~$0.008/query, less with prompt caching. 1,000 queries/mo ~= **$5-8**. Opus ~$0.05-0.08/query.

### Hard daily budget cut-off (required, MVP)
- A **hard daily spend cap** tracked server-side. On breach the app **fails closed** to a **"demo paused for today"** state (no LLM calls, chat disabled, /about + source library still viewable).
- **Opus "best" mode is gated behind the same cut-off** (and behind a lower sub-cap), so it can never blow the budget.
- Additional controls: per-IP rate limit; the cap is the backstop against a runaway public endpoint (the #1 cost risk).

---

## 10. Roadmap, acceptance gates, scope guards, naming

### Phase 0 — Spike & de-risk the long-pole (~1 week)
- Parse safe-tier PDFs (NCA5 chapter, a CC-BY review) with Docling; validate section chunking.
- **De-risk statement-ID chunking (schedule long-pole):** SPM statement IDs are **inline bold labels embedded in two-column text**, so boundary detection needs **regex over Docling's structured output, with PyMuPDF+regex as a serious fallback**. Prototype on one public IPCC SPM PDF *for parsing R&D only* (no indexing/redistribution) to prove the approach and **budget several days** for it. (This informs the deferred IPCC path; it does not put IPCC in the corpus.)
- Minimal retrieve -> rerank -> Claude-native-citations loop in a notebook over the safe-tier corpus.
- **GATE:** on a 20-question probe, citations resolve to the correct source blocks on **at least 18/20**.

### Phase 1 — MVP build target (~3-4 weeks)
Safe-tier corpus only (hand-audited; `manifest.yaml` with SHA-256 + licence + `licence_evidence` + `human_signoff`) -> hybrid retrieval **+ reranker** -> **Claude native citations via custom-content blocks** -> **refusal gate** (reranker-thresholded) + **calibrated-language preservation** (proxy + confidence-level judge) -> **Streamlit** chat + passages panel + **/about** transparency page -> **~40-question gold set** with gold `chunk_id`s and **retrieval + faithfulness + citation-support + calibrated-term-preservation** evals in CI, **results published and linked from the demo** -> **one hosted demo with the hard daily budget cut-off**.
- **GATES:** faithfulness + citation-support at target on the gold set; refusal >90% on no-answer subset, false-refusal <5%; cut-off verified to fail closed; every corpus doc has `commercial_use_ok: true` + `human_signoff`; eval results visible on `/about`.

### Phase 2+ — Deferred (explicitly out of the MVP)
IPCC ingestion **after written permission is answered** (+ legal nod); local-model backend; Next.js UI; full AR6 pipeline (using the Phase-0 statement-ID work); automated CC-BY DOI pipeline; source-scope filters; multi-turn memory; synthetic eval augmentation.

### Scope guards
| Temptation | Guard |
|---|---|
| "Just index IPCC now" | Blocked until written permission is **answered** (section 2.1); build refuses to index `commercial_use_ok=false`. |
| Trust one licence lookup | Requires 2+ sources + publisher-page confirm + human sign-off (section 2.2). |
| "Native citations mean no hallucination" | Doc states guaranteed vs measured; publish the numbers (section 3.3, section 6). |
| Threshold refusal on RRF scores | Use the calibrated reranker score (section 3.2). |
| Plain-text citation docs | Custom-content blocks, one per citable unit (section 3.3). |
| Mix cited+uncited docs / add JSON schema to the gen call | Forbidden by native-citations constraints (section 3.4). |
| News/current events, multilingual, fine-tuning | Out of scope; currency comes from corpus-version releases. |
| Runaway spend | Hard daily cut-off, Opus gated behind it (section 9). |
| Endless chunking experiments | A change must beat the retrieval eval; else ship. |

### Naming
**Keep "Ask About the Climate."** Tagline: **"Answers from the climate science literature — with receipts."** Repo `ask-about-the-climate`; candidate domains `askabouttheclimate.org` / `askaboutthe.climate` (verify availability). Rejected: "ClimateGPT" (implies fine-tuned model; collisions), "IPCC Chat" (implies endorsement — and IPCC isn't even in the launch corpus), "Climate Librarian" (obscure). Required disclaimer wherever sources appear: *"Not affiliated with or endorsed by NASA, NOAA, Copernicus, USGCRP, or the IPCC. All sources are cited and linked."*

---

## Appendix A — Key licensing sources
- [IPCC — Copyright](https://www.ipcc.ch/copyright/) (deferred tier)
- [NASA Earthdata — Data Use and Citation Guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance)
- [NOAA Climate.gov FAQs](https://www.climate.gov/faqs) - [NOAA NAO 205-17A](https://www.noaa.gov/organization/administration/nao-205-17a-information-access-dissemination)
- [Licence to Use Copernicus Products](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) - [C3S citation & attribution guidance](https://confluence.ecmwf.int/pages/viewpage.action?pageId=222484359)
- US National Climate Assessment (NCA5), USGCRP — US Government public domain
- CC-BY journals (candidates for the hardened gate): [Environmental Research Letters](https://iopscience.iop.org/journal/1748-9326) - [Environmental Research: Climate](https://publishingsupport.iopscience.iop.org/journals/environmental-research-climate/about-environmental-research-climate/) - [PLOS Climate](https://www.scienceopen.com/collection/e2f80610-6d9f-4ad0-a654-7e77ee34cc1b) - [Communications Earth & Environment](https://en.wikipedia.org/wiki/Communications_Earth_%26_Environment)

## Appendix B — Guaranteed vs measured (one-liner for the /about page)
> **Guaranteed:** every citation points to real text we retrieved from a named, openly-licensed source. **Measured (and published):** how often the cited text actually supports the sentence, and how faithfully the science's calibrated uncertainty is preserved. We show these numbers rather than claiming perfection.
