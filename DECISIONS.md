# DECISIONS.md — Let’s Talk About the Climate Emergency

Architecture Decision Record log for the *Let's Talk About the Climate Emergency* RAG chatbot (formerly *Ask About the Climate*; renamed in ADR-022). Companion to the design document (v3, 2026-08). Each ADR records the decision, the forces that shaped it, the alternatives considered fairly, and the contexts in which the alternative would have been the right call.

**Standing forces** (referenced throughout rather than repeated; **amended by ADR-018**): the mission of communicating the climate emergency to an under-informed public, without misinformation in either direction; strict grounding (answers only from retrieved, cited passages); a **non-commercial educational public-benefit** context (superseding v2's commercial-portfolio framing — see ADR-018; ADRs 001–017 were written under the old framing and their licensing reasoning is amended accordingly); legal/licensing exposure as a first-order constraint; a deliberately tiny, hand-auditable corpus; running cost under ~£20/month; verifiability (citations to exact passages; charts rendered from named datasets); and honesty about what is guaranteed versus merely measured.

## Decision index

| # | Decision | Choice |
|---|---|---|
| [001](#adr-001) | Corpus legality tier at launch | Safe tier only; IPCC deferred behind answered written permission |
| [002](#adr-002) | Corpus source selection | Authority over coverage: NCA5, NASA, NOAA, Copernicus, gated CC-BY reviews |
| [003](#adr-003) | Licence verification for CC-BY papers | Hardened gate: ≥2 agreeing lookups + publisher-page confirm + human sign-off |
| [004](#adr-004) | Knowledge mechanism | RAG grounding, not fine-tuning |
| [005](#adr-005) | Embeddings | bge-m3, run locally |
| [006](#adr-006) | Retrieval | Hybrid (dense + sparse, RRF) with cross-encoder reranker **in the MVP** |
| [007](#adr-007) | Vector store | Qdrant |
| [008](#adr-008) | Citation mechanism | Claude native citations via custom-content document blocks |
| [009](#adr-009) | Truthfulness claims | Guaranteed vs measured framing; no "can't hallucinate" claim |
| [010](#adr-010) | Unsupported questions & uncertainty language | Reranker-thresholded refusal; calibration checked by proxy metric + LLM judge |
| [011](#adr-011) | Parsing & chunking | Docling + structure-aware / statement-ID chunking; PyMuPDF fallback |
| [012](#adr-012) | Generation model | Anthropic cloud: Haiku default, Opus gated; thin provider adapter |
| [013](#adr-013) | UI | Streamlit for MVP; Next.js deferred |
| [014](#adr-014) | Evaluation | First-class, published deliverable in CI |
| [015](#adr-015) | Cost control | Hard daily budget cut-off that fails closed |
| [016](#adr-016) | Core stack & repo shape | Python 3.12 + FastAPI, single public monorepo |
| [017](#adr-017) | Naming & affiliation | ~~"Ask About the Climate"~~ **superseded by ADR-022** |
| [018](#adr-018) | Project framing | **Non-commercial educational public-benefit** (supersedes the commercial-portfolio framing) |
| [019](#adr-019) | Corpus restructure | Permission tiers incl. NC-licensed Tier B + non-corpus "voices" layer (amends 001/002) |
| [020](#adr-020) | Chart generation | Curated data pack + closed declarative ChartSpec + server-side render; never LLM codegen |
| [021](#adr-021) | Data beyond the pack | No open web search; allowlisted provider fetch, deferred to Phase 2 |
| [022](#adr-022) | Naming | "Let's Talk About the Climate Emergency" (supersedes 017) |

---

<a name="adr-001"></a>
## ADR-001 — Launch on a clearly-permitted corpus tier; defer IPCC behind answered written permission

**Decision:** The MVP indexes only sources whose terms clearly permit commercial reproduction; IPCC material is excluded until a written permission request to the IPCC Secretariat has been answered affirmatively.
**Status:** Accepted-MVP (IPCC ingestion: Deferred). **Amended by ADR-018/019 (v3):** the permission bar stays, but the criterion is now "clearly permits *non-commercial educational* reproduction", which admits NC-licensed sources (Tier B) and strengthens the IPCC permission request.

**Context & forces.** IPCC assessment reports are the single most authoritative synthesis of climate science, which makes them the most tempting corpus. But building a RAG index means fetching, storing, parsing, chunking and embedding the **full text** — that is reproduction of the whole work under copyright law (UK CDPA s.16–17), regardless of how little is ever displayed to a user. The display layer is not where the legal exposure sits; the ingestion layer is. The project is commercial (its purpose includes winning Rusty Data client work), and it is a *credibility product*: an anti-misinformation tool caught cutting licensing corners would be self-refuting.

**Options considered.**
- **Index IPCC now under a claimed exception.** Pros: best possible corpus on day one; the "short excerpts" language on [ipcc.ch/copyright](https://www.ipcc.ch/copyright/) superficially resembles what the UI displays. Cons: the IPCC allowance covers reproduction of *limited figures or short excerpts* with acknowledgement — it does not cover a private full-text index of entire reports, for which the policy requires written permission. [CDPA s.29A](https://www.legislation.gov.uk/ukpga/1988/48/section/29A) (the UK text-and-data-mining exception) permits copies "for the sole purpose of research for a **non-commercial purpose**", and using or transferring the copy for anything else makes it an infringing copy; no commercial TDM exception is in force in the UK as of 2026. US fair use is both jurisdictionally inapposite for a UK business and an unresolved litigation posture, not a permission.
- **Index IPCC now, quietly, and seek permission later.** Pros: fastest. Cons: converts a permission request into a confession; indistinguishable from bad faith if refused.
- **Safe tier now, IPCC after written permission (chosen).** Pros: zero legal ambiguity at launch; the permission email describes the project honestly as commercial; the deferral itself becomes a portfolio talking point (responsible-AI diligence). Cons: weaker launch corpus; permission may be slow or refused.

**Decision & rationale.** The safe tier wins because the downside asymmetry is extreme: the upside of early IPCC ingestion is coverage; the downside is an infringement claim against a product whose entire pitch is trustworthiness. Enforced in code: the build refuses to index any document without `commercial_use_ok: true` and a recorded `human_signoff`.

**Trade-offs / consequences.** Launch answers lean on NCA5/agency explainers rather than IPCC assessed language; some questions get link-outs to ipcc.ch instead of grounded answers. The statement-ID chunking work (ADR-011) is prototyped on one public SPM PDF for parsing R&D only, so the deferred path is ready if permission lands.

**When you'd choose differently.** (a) Genuinely non-commercial academic research with lawful access — s.29A then applies and full-text TDM of IPCC reports is lawful, provided outputs stay within the research purpose. (b) A jurisdiction with a commercial TDM exception (e.g. Japan's Art. 30-4; the EU DSM Art. 4 exception where no opt-out is reserved) — though a UK-based service would still need UK analysis. (c) If the IPCC published under an open licence, the tier distinction dissolves. (d) An internal-only tool never exposed or marketed might justify a different risk posture — but "private" indexing is still reproduction, so this is risk appetite, not legality.

---

<a name="adr-002"></a>
## ADR-002 — Source authority over coverage

**Decision:** The corpus admits only assessed, consensus-grade, clearly-licensed publications — NCA5, NASA and NOAA explainers, Copernicus/C3S reports, and hand-verified CC-BY/CC0 review papers — rather than a broad web crawl.
**Status:** Accepted-MVP. **Amended by ADR-019 (v3):** the authority-over-coverage principle stands; the admitted set widens (Hansen CC-BY papers, UNEP EGR, Met Office OGL, Carbon Brief verbatim under NC-ND) and a non-corpus "voices" layer is added for campaign/communicator content.

**Context & forces.** RAG quality is upper-bounded by corpus quality: retrieval cannot un-say what a bad source says, and the grounding contract (ADR-004) makes the bot *faithfully repeat* its corpus. For an anti-misinformation product, corpus curation is therefore the first guardrail, not a nicety. Licensing narrows the field further (ADR-001), and the £20/month budget and hand-audit requirement cap corpus size anyway.

**Options considered.**
- **Big, loose web corpus** (news, blogs, Wikipedia, advocacy sites). Pros: coverage of current events and niche questions; cheap to assemble; higher recall. Cons: injects contested or wrong claims that the bot would then cite as if authoritative; licence status of most web text is unclear or restrictive (e.g. Carbon Brief is CC BY-NC-ND — link-only for a commercial product); impossible to hand-audit; source vintage and provenance chaos.
- **Primary research papers at scale.** Pros: depth. Cons: single studies contradict each other by design; a chatbot cannot do the weighing that assessment reports exist to do; per-paper licence checking at scale is exactly the failure mode ADR-003 guards against.
- **Curated authoritative tier (chosen).** Pros: every retrieved passage is already the *output* of expert synthesis (NCA5 is a legislatively mandated, peer-reviewed US assessment and public domain as a US Government work, 17 U.S.C. §105; NOAA content likewise; NASA content generally not copyrighted per [Earthdata guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance); the [Copernicus licence](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) explicitly permits commercial reuse with attribution). ~500–1,000 pages is small enough to hand-audit every document. Cons: coverage gaps; US-weighted until C3S/reviews balance it; corpus vintage must be disclosed.

**Decision & rationale.** A librarian who stocks only assessed literature beats one who stocks everything: the product's one unforgivable failure is confidently citing a bad source. Small-and-audited is marketed as a feature ("with receipts"), which it genuinely is.

**Trade-offs / consequences.** More refusals (handled honestly by ADR-010); no current-events answers — currency arrives via corpus-version releases, not crawling. Skeptical Science's myth taxonomy is used only to build the adversarial eval set, never as corpus.

**When you'd choose differently.** A news-literacy or claim-checking product *needs* the contested material in scope (with stance metadata, not as ground truth). A general-purpose assistant optimises recall and tolerates noise. And at enterprise scale with a licensing budget and a legal team, broad ingestion under negotiated licences (or publisher APIs) is the standard play — the constraint here is a one-person shop with zero licensing budget.

---

<a name="adr-003"></a>
## ADR-003 — Hardened CC-BY licensing gate (≥2 sources + publisher-page confirmation + human sign-off)

**Decision:** A review paper enters the corpus only if at least two of OpenAlex/Crossref/Unpaywall agree it is CC BY/CC BY-SA/CC0, the licence statement is confirmed on the publisher's own page with evidence captured, and a named human signs off — all recorded in the committed manifest.
**Status:** Accepted-MVP.

**Context & forces.** Redistribution and commercial reuse of paper text rides entirely on the licence being what the metadata says it is. But bibliographic licence metadata is unreliable: a 2026 corpus-building study that harmonised licence records across the three services found all three agreed on the licence for only ~63% of records ([arXiv:2604.12498](https://arxiv.org/pdf/2604.12498)), with the remainder resolved only by pairwise agreement. Hybrid journals are the classic trap: the *journal* supports CC BY, the *article* is subscription-only, and aggregators disagree about which applies.

**Options considered.**
- **Single automated lookup** (e.g. trust OpenAlex's `license` field). Pros: zero marginal effort; scales to thousands of DOIs. Cons: at ~63% three-way agreement, a single source is wrong or unverifiable too often for a decision that creates infringement liability; errors are silent and compound (an infringing PDF shipped in the public repo dataset).
- **Manual-only checking.** Pros: highest accuracy. Cons: no candidate filter means humans wade through obviously-ineligible papers; unauditable unless evidence capture is forced; doesn't scale even to the deferred DOI pipeline.
- **Layered gate (chosen):** automated multi-source *candidate filter* → automated fetch of the actual article page with the licence text and URL stored as `licence_evidence` → mandatory `human_signoff: {who, date, note}`. Pros: automation does triage, the canonical source of truth (the publisher page) does verification, a human owns the decision, and the manifest makes the whole chain auditable. Cons: per-paper cost of minutes; sign-off is a bottleneck by design.

**Decision & rationale.** The gate treats automated metadata as *evidence, never authorisation* — the same posture a data-protection or clinical pipeline takes with upstream metadata. For ~10–15 MVP papers the human cost is trivial; the audit trail is the point.

**Trade-offs / consequences.** The deferred automated DOI pipeline inherits the gate, so it can only ever auto-*reject*, never auto-admit. Some genuinely open papers will be rejected on metadata disagreement (acceptable: the corpus philosophy is precision over recall).

**When you'd choose differently.** For a non-redistributive internal analytics corpus, a single lookup is proportionate — the harm of a licence error is contained. At the scale of millions of papers (e.g. building an embedding training set), human sign-off is impossible; you'd instead accept a quantified error rate, restrict to publishers with machine-readable licence APIs you've validated, and carry the residual risk explicitly. And where a signed publisher agreement exists, the gate is redundant for that publisher's content.

---

<a name="adr-004"></a>
## ADR-004 — RAG grounding, not a fine-tuned "climate model"

**Decision:** Answers are generated strictly from retrieved corpus passages at inference time; no model is trained or fine-tuned on climate text.
**Status:** Accepted-MVP (fine-tuning explicitly out of scope, all phases).

**Context & forces.** The mission demands verifiability (every claim traceable to a source passage), honesty about provenance, and corpus updateability as new reports release. Legal exposure also differs sharply: training on copyrighted text is a live, unsettled legal battleground, whereas retrieval-with-display of licensed text is comparatively clean and controllable (ADR-001).

**Options considered.**
- **Fine-tune an open-weight model on climate literature ("ClimateGPT" approach).** Pros: knowledge available without retrieval latency; offline capability; a fashionable-sounding artefact. Cons: fine-tuning teaches *style and distribution*, not reliable *facts* — parametric knowledge cannot be cited, cannot be attributed to a passage, hallucinates fluently, and cannot be updated without retraining; a fine-tune on IPCC text would also reproduce ADR-001's legal problem in the weights themselves; training runs blow the budget; and calibrated language ("very likely") would be flattened into the model's own confident register — the precise failure the mission forbids.
- **RAG (chosen).** Pros: citations point at real retrieved text (mechanically checkable); corpus swaps are a re-index, not a training run; refusal is natural (no passages → no answer); licensing is governed at the document level; costs are per-query pennies. Cons: retrieval quality becomes the bottleneck (hence ADR-006); latency includes a retrieval hop; answers limited to corpus coverage.
- **Hybrid (fine-tune for style/format + RAG for facts).** Pros: could improve tone adherence and citation formatting. Cons: adds a training pipeline for gains that prompting already achieves with a strong instruction-following model; not worth it at MVP scale.

**Decision & rationale.** For a truth-critical, attribution-critical product, the knowledge must live where it can be inspected, versioned and licensed — in documents, not weights. "The bot is a librarian, not an oracle" is an architecture statement, not just a tagline.

**Trade-offs / consequences.** Everything downstream (refusal gate, citation validation, eval design) is built on retrieval quality; the corpus's coverage gaps become the product's coverage gaps, surfaced honestly.

**When you'd choose differently.** Fine-tuning is right when the target is *behaviour*, not *facts*: a domain-specific extraction format, a classification head, consistent house style at high volume, or latency/cost floors where retrieval is prohibitive. It also wins for closed-domain paraphrase tasks where attribution is not a requirement. For embedding models (not generators), domain fine-tuning on retrieval pairs is cheap and often worthwhile once you have eval data proving the gain — a plausible Phase 3 experiment here.

---

<a name="adr-005"></a>
## ADR-005 — Embeddings: bge-m3 run locally

**Decision:** Chunks and queries are embedded with BAAI's bge-m3, running locally on CPU.
**Status:** Accepted-MVP.

**Context & forces.** The corpus is ~0.7–1.5M tokens; embedding cost, reproducibility, and the hybrid-retrieval design (ADR-006) dominate. A public demo also re-embeds queries on every request, so per-query cost and vendor coupling matter.

**Options considered.**
- **OpenAI / Cohere / Voyage embedding APIs.** Pros: strong retrieval quality; zero ops; trivially scalable. Cons: adds a second cloud vendor and API key for a component that is *not* differentiating; embeddings become irreproducible if the provider retires the model (a re-embed of the entire corpus under a different model invalidates the index and any published retrieval evals); per-query network hop; ongoing cost — small at this scale, but non-zero against a £20/month ceiling; and no native sparse output, so lexical search must be bolted on separately.
- **Small local models (e.g. all-MiniLM, gte-small).** Pros: even lighter. Cons: measurably weaker retrieval; no sparse head; 512-token context is tight against ~500-token chunks plus context headers.
- **bge-m3 (chosen).** Pros: open weights (reproducible forever — pin the revision); one model natively emits **dense and sparse (learned lexical) representations** — a direct fit for hybrid retrieval ([BGE-M3, Chen et al. 2024](https://huggingface.co/BAAI/bge-m3)); 8192-token context; competitive quality; CPU-viable at this corpus size; $0 to embed and re-embed; pairs with bge-reranker-v2-m3 from the same family. Cons: heavier than MiniLM-class models; multi-vector (ColBERT) head unused; self-hosting means owning inference.

**Decision & rationale.** At ~1M tokens, embedding locally costs nothing and takes minutes, so the API's only advantage (managed scale) buys nothing. Reproducibility is load-bearing here because the eval results are *published* — a vendor-side model change silently invalidating published numbers is unacceptable. The native sparse output means hybrid search comes from one model instead of two systems.

**Trade-offs / consequences.** The demo VM carries the embedding model in memory; query embedding adds ~tens of ms on CPU. Corpus re-embeds are incremental by content hash.

**When you'd choose differently.** At 100M+ tokens or high QPS, managed embedding APIs (or GPU-served local inference) win on throughput and ops. If the corpus were heavily multilingual-query-against-English-docs, it would be worth benchmarking API models against bge-m3's multilingual strength — m3 often still wins. If the team had no capacity to own model serving at all (pure-serverless deployments), an API is the pragmatic call despite the reproducibility cost — pin the model version and record it in the manifest.

---

<a name="adr-006"></a>
## ADR-006 — Hybrid retrieval (dense + sparse, RRF) with the cross-encoder reranker in the MVP

**Decision:** Retrieval is hybrid — dense and sparse searches fused by reciprocal-rank fusion into a top-40 — followed by a bge-reranker-v2-m3 cross-encoder cutting to a top-8; the refusal gate thresholds on the reranker's scores.
**Status:** Accepted-MVP (reranker promoted from "later" in v1).

**Context & forces.** Climate questions mix semantic paraphrase ("is it too late?") with exact-term lookups ("SSP2-4.5", "AMOC", "1.5°C vs 2°C") where dense embeddings alone are weak. Separately, the refusal gate (ADR-010) needs a per-query *absolute* signal of "is anything retrieved actually relevant?" — and that requirement, not answer quality, is what forces the reranker into the MVP.

**Options considered.**
- **Dense-only.** Pros: simplest; one index. Cons: misses exact terminology and IDs; embeddings retrieve *something* for any query, so there is no usable "nothing relevant" signal; cosine similarities are corpus- and model-relative, making refusal thresholds fragile.
- **Hybrid without reranker (v1 position).** Pros: better recall than either channel alone; RRF is robust, parameter-light rank fusion ([Cormack, Clarke & Büttcher, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)). Cons: **RRF scores are rank-fusion artefacts** — Σ 1/(k + rank) depends only on rank positions, never on how relevant anything actually is. The top RRF score for a query with excellent matches and a query with garbage matches can be identical. You cannot threshold refusal on it; the refusal gate would be theatre.
- **Hybrid + cross-encoder reranker (chosen).** Pros: the cross-encoder reads *(query, passage)* jointly and emits a relevance score on a consistent scale across queries — thresholdable once calibrated empirically on the gold set's no-answer subset; large precision gain on the top-8 actually fed to the LLM; ~100 ms on CPU for 40 pairs. Cons: extra model, extra latency, threshold needs eval data to set.
- **LLM-as-reranker.** Pros: highest quality ceiling. Cons: per-query cost and latency an order of magnitude worse; non-deterministic scores make thresholding worse, not better.

**Decision & rationale.** The reranker is in the MVP because it is *load-bearing for honesty*: refuse-when-unsupported is a headline behaviour, and it needs a score that means something in absolute terms. (Stated precisely: cross-encoder logits are not calibrated probabilities either — they are query-comparable relevance scores, which is the property thresholding needs and RRF lacks; the threshold itself is set empirically against refusal/false-refusal targets.) Quality improvement in the top-8 is the bonus, not the reason.

**Trade-offs / consequences.** Two-stage retrieval to maintain; the threshold is a tuned parameter that must be re-checked per corpus version (automated in CI via the no-answer subset).

**When you'd choose differently.** Dense-only is fine when there is no refusal requirement, queries are conversational paraphrase over homogeneous content, and latency budgets are tight (autocomplete, recommendations). Skip the reranker when the generator sees a large context anyway and you trust the LLM to ignore irrelevant passages — defensible for internal tools, not for a public bot that must *decline*. BM25-only remains right for pure known-item/keyword search (log search, case lookup).

---

<a name="adr-007"></a>
## ADR-007 — Vector store: Qdrant

**Decision:** Qdrant (Docker-composed alongside the API, or its free cloud tier), with LanceDB noted as fallback.
**Status:** Accepted-MVP.

**Context & forces.** The store must serve hybrid retrieval natively (dense + sparse + fusion, per ADR-006), run inside a ~£20/month footprint, be open source (public monorepo, one-command reproduction), and carry chunk-level citation metadata as payload.

**Options considered.**
- **pgvector.** Pros: it's just Postgres — one database for everything, mature ops, SQL joins over metadata; the default answer when an app already has Postgres. Cons: this app has no other relational workload, so "one database" buys nothing; sparse/lexical hybrid means bolting `tsvector` full-text search onto vector search and fusing ranks in application code — exactly the plumbing Qdrant ships natively.
- **Chroma.** Pros: simplest developer experience; embedded mode. Cons: historically weaker/immature sparse-hybrid story and a less production-shaped server mode; fine for notebooks, less convincing as the named store in a portfolio architecture.
- **LanceDB.** Pros: embedded, serverless, zero-ops, columnar on-disk format; genuinely attractive at this scale. Cons: hybrid search and ecosystem younger; embedded-only shape couples store lifecycle to the API process. Retained as the explicit fallback if running a Qdrant container proves annoying.
- **Weaviate.** Pros: native hybrid search, mature. Cons: heavier operational footprint (more RAM, more moving parts) for identical capability at this scale.
- **Managed-only stores (Pinecone).** Pros: zero ops. Cons: closed source breaks one-command local reproduction; free-tier terms drift; vendor lock-in for no capability gain.
- **Qdrant (chosen).** Pros: OSS (Apache-2.0); **native sparse vectors and server-side hybrid fusion with RRF via the Query API** ([Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)) — the bge-m3 dense+sparse output maps onto it directly; payload filtering for corpus-version scoping; small Docker image; free tier fits the corpus with room.

**Decision & rationale.** The deciding feature is first-class hybrid: dense+sparse+RRF is one API call, keeping fusion out of application code. Everything else at this corpus size (~thousands of chunks) is a wash — honestly, *any* of these stores handles 5,000 vectors.

**Trade-offs / consequences.** One more container in the compose file; Qdrant's lexical channel is sparse vectors (bge-m3's learned term weights), not classical BM25 — acceptable and arguably better, but worth stating precisely in interviews.

**When you'd choose differently.** Already running Postgres with relational data the chunks must join against → pgvector, no contest. Pure-embedded desire (single-binary demo, no services) → LanceDB or even SQLite+sqlite-vec. Billions of vectors with a platform team → the conversation changes entirely (Vespa, Milvus, or managed). Notebook-stage prototyping → Chroma is fine; the store choice only starts mattering at the refusal-gate/hybrid stage.

---

<a name="adr-008"></a>
## ADR-008 — Citations: Claude native citations over custom-content document blocks

**Decision:** Retrieved chunks are passed as custom-content document blocks (`source.type: "content"`, `citations: {enabled: true}`), one block per citable unit, and the model's native citation output is used as the citation mechanism.
**Status:** Accepted-MVP.

**Context & forces.** Citations are the product. The mechanism must (a) tie each claim to a specific retrieved passage, (b) be resistant to fabricated references, (c) preserve the unit "one calibrated claim + its qualifier" intact, and (d) obey the API's real constraints, documented on [Anthropic's citations docs](https://platform.claude.com/docs/en/build-with-claude/citations): citations are enabled **all-or-none** across a request's documents, and a request using citations **cannot also use structured outputs** (400 error).

**Options considered.**
- **Prompt-engineered `[n]` markers.** Pros: works on any model, including the deferred local backend (where it remains the designated fallback); simple. Cons: the model can emit `[7]` when there are five passages, cite the wrong passage, or fabricate the mapping; every marker needs post-hoc validation you must build; span-level "which words came from where" is unavailable.
- **Structured-output JSON (answer + quoted evidence per claim).** Pros: guaranteed parseable shape; easy UI binding. Cons: guarantees *shape*, not *provenance* — quoted "evidence" can be hallucinated text that appears in no passage, so a verbatim-match validator is needed anyway; and structured outputs are mutually exclusive with native citations, so this forfeits the stronger mechanism.
- **Native citations over plain-text documents.** Pros: citations are constrained to real provided text. Cons: plain text is auto-chunked to sentences, and the sentence chunker can split a calibrated qualifier from its claim — "very likely" landing in a different auto-chunk than the statement it modifies corrupts both the citation and the calibration check (ADR-010). This was a live hazard in design v1.
- **Native citations over custom-content blocks (chosen).** Pros: the API guarantees every emitted citation resolves to actual provided text with a real `content_block_location`; **we define the citable unit** — one section chunk now, one SPM statement later — so qualifier and claim are never split; `cited_text` spans enable exact highlighting in the passages panel. Cons: Anthropic-specific (mitigated by the provider adapter, ADR-012, whose local path uses validated `[n]` markers); the constraints ripple through the architecture — the scope classifier and query rewriter need structured outputs, so they **must** be separate API calls from generation, and cited/uncited documents can't be mixed in one call.

**Decision & rationale.** Native citations convert "the model probably cited correctly" into "the citation mechanically resolves to retrieved text" — the strongest provenance guarantee available, and the foundation of the guaranteed-vs-measured stance (ADR-009). Custom-content blocks are what make the guarantee *meaningful* for this corpus, because the block boundary is the epistemic boundary.

**Trade-offs / consequences.** Two extra cheap API calls per query (rewrite + scope classify); pipeline is coupled to one vendor's feature for its headline behaviour — accepted consciously, with the fallback path designed but deferred.

**When you'd choose differently.** Multi-provider or local-first products must use validated `[n]` markers (plus an entailment checker) as the primary mechanism — it's workable, just weaker and more code. If the application needed machine-readable structured answers *and* citations in one hop, the incompatibility forces marker-based citations inside the JSON. Plain-text native citations are fine when sources have no intra-sentence qualifiers that matter — general document Q&A, not calibrated scientific language.

---

<a name="adr-009"></a>
## ADR-009 — Guaranteed vs measured: no "can't hallucinate" claim

**Decision:** Marketing and the `/about` page state precisely: citations pointing to real retrieved text is **guaranteed** by construction; whether the cited text *entails* the sentence (faithfulness/citation-support) is **measured** and the numbers published.
**Status:** Accepted-MVP.

**Context & forces.** Design v1 drifted toward "the system structurally cannot hallucinate". That is false: native citations prevent citing text that wasn't provided, but nothing prevents the model writing an unsupported claim and attaching a *real but non-entailing* citation to it. For an epistemic-honesty product, an overclaim about its own epistemics is the most embarrassing possible bug — and the one a hostile screenshot would immortalise.

**Options considered.**
- **Claim "can't hallucinate".** Pros: punchy marketing. Cons: false; one counterexample destroys the product's entire premise; indefensible in front of any technically literate interviewer or client.
- **Claim nothing / boilerplate disclaimer.** Pros: safe. Cons: wastes a genuine technical differentiator (the guarantee that *is* real) and forfeits the trust dividend of publishing eval numbers.
- **Guaranteed-vs-measured split (chosen).** Pros: every claim is defensible; the published citation-support and faithfulness scores (ADR-014) convert honesty into evidence; a per-sentence entailment validator flags "unverified" sentences in the UI as a runtime backstop. Cons: subtler message; requires maintaining published numbers per release.

**Decision & rationale.** The product's thesis is that calibrated honesty beats confident fluency. Applying that thesis to the product's own claims is both intellectually consistent and commercially distinctive — "we show you our failure rates" is rare enough to be a pitch.

**Trade-offs / consequences.** The `/about` page must stay current with eval results; a bad faithfulness number must be published, not hidden (which is precisely the credible-commitment mechanism).

**When you'd choose differently.** You wouldn't — this is a values decision as much as a technical one. What changes elsewhere is the *split point*: a system with a mandatory post-generation NLI gate that blocks non-entailed sentences could move entailment closer to "enforced" (still not "guaranteed" — the NLI model has its own error rate, which you'd then have to publish instead). In low-stakes creative products the whole framing is unnecessary weight.

---

<a name="adr-010"></a>
## ADR-010 — Refusal-when-unsupported and calibrated-language fidelity as measured behaviours

**Decision:** The bot refuses when the top reranker score falls below an empirically calibrated threshold; preservation of calibrated uncertainty language is enforced by prompt and *measured* two ways — a regex proxy (calibrated-term preservation) and an LLM-judge check of confidence-*level* fidelity — rather than assumed.
**Status:** Accepted-MVP.

**Context & forces.** Two mission-critical behaviours cannot be left to vibes. First: when the corpus doesn't support an answer, saying so is the anti-misinformation behaviour (an LLM will otherwise answer from parametric memory, uncited). Second: IPCC-style calibrated vocabulary (*virtually certain … more likely than not*; confidence levels) encodes the science's own uncertainty; silently upgrading "likely" to a bare assertion — or downgrading it — is misinformation by tone.

**Options considered — refusal.**
- **Trust the prompt** ("say so if the passages don't answer"). Pros: free. Cons: models comply inconsistently, especially when near-miss passages are present; unmeasurable without a gate to instrument.
- **Threshold on retrieval similarity/RRF.** Cons: not thresholdable (ADR-006).
- **Reranker-score gate + canned refusal + prompt-level partial-support handling (chosen).** Threshold set on the gold set's no-answer subset; CI targets: >90% refusal on no-answer items, <5% false-refusal on answerable ones. The gate makes refusal a *tested system property* rather than a model mood.

**Options considered — calibration fidelity.**
- **Assume the model preserves qualifiers.** Cons: unverified; models paraphrase, and paraphrase is exactly where level-shift happens.
- **Regex-only check.** Pros: deterministic, free, CI-friendly. Cons: word presence ≠ meaning preserved — "not likely" contains "likely"; a qualifier can survive verbatim while being attached to the wrong claim. Explicitly labelled a **proxy**.
- **LLM-judge-only.** Pros: judges meaning. Cons: costs money per run; noisy; unauditable alone.
- **Both, layered (chosen):** regex proxy for cheap always-on regression signal; LLM judge (different model from the generator, sample human-audited) for confidence-*level* fidelity — catching "very likely"→"likely" shifts and dropped qualifiers the regex misses. Custom-content blocks (ADR-008) keep qualifier and claim in one citable unit so both checks are well-posed.

**Decision & rationale.** Both behaviours follow the same pattern as ADR-009: don't assume, instrument. The prompt *requests* the behaviour; the eval *verifies* it; the published numbers *prove* it.

**Trade-offs / consequences.** Refusal threshold is corpus-version-coupled (re-calibrated per release); judge costs are bounded by running the full suite per release rather than per PR. Symmetric honesty is in the judge rubric: overstating uncertainty is flagged as much as understating it.

**When you'd choose differently.** Products without calibrated-source language don't need the fidelity checks at all. A conversational assistant where refusing is worse than an imperfect answer (internal brainstorming tools) would soften the gate to a caveat banner instead of refusal. If the corpus were large and redundant, an answerability *classifier* trained on labelled data could replace the reranker threshold; with 40 gold questions, a threshold is the right-sized tool.

---

<a name="adr-011"></a>
## ADR-011 — Parsing & chunking: Docling + structure-aware / statement-ID chunking, PyMuPDF fallback

**Decision:** PDFs parse via Docling (layout-aware → structured Markdown) with PyMuPDF+regex as fallback; chunking is structure-aware (by heading/section, ≤~500 tokens, never across headings, prepended context header) and, for the deferred IPCC path, on assessment-statement ID boundaries (`A.1.2`, `B.3` …). Statement-ID chunking is flagged the schedule long-pole and de-risked in Phase 0.
**Status:** Accepted-MVP (statement-ID path: Deferred with IPCC, prototyped Phase 0).

**Context & forces.** Citation quality is set at ingest: if a chunk straddles two sections, its `section_path`/page metadata lies, and the "receipts" are wrong even when retrieval and generation are perfect. IPCC SPM statements are the natural citable unit for the deferred tier — but their IDs are inline bold labels in two-column PDF layout, which naive text extraction scrambles.

**Options considered.**
- **Naive fixed-size chunking** (N tokens, sliding overlap). Pros: ten lines of code; corpus-agnostic. Cons: splits mid-claim and can separate a qualifier from its sentence; chunk boundaries ignore document structure, so citations can't be precise to a section; retrieval quality suffers from headerless fragments. Cheap in exactly the place this product cannot afford cheapness.
- **Framework chunkers (LangChain/LlamaIndex splitters).** Pros: off the shelf. Cons: mostly fixed-size-with-separators under the hood; statement-ID chunking is bespoke regardless; pulling a framework for ~300 lines of pipeline trades transparency (the showcase asset) for lock-in.
- **Structure-aware, hand-rolled (chosen)** on Docling output. Pros: Docling is the strongest OSS layout-aware PDF→structured-text converter (reading order, headings, tables — [Docling](https://github.com/docling-project/docling)); chunk = section unit means citation metadata is true by construction; one-sentence overlap and context headers recover cross-boundary recall; ~300 auditable lines. Cons: bespoke code to maintain; Docling is heavier than plain extractors; statement-ID boundary detection over two-column SPMs is genuinely hard — hence the Phase-0 spike on one public SPM PDF (parsing R&D only, no indexing) with **several budgeted days** and PyMuPDF+regex as the serious fallback if Docling's structure output proves unreliable there.
- **Commercial parsing APIs** (Azure Document Intelligence, etc.). Pros: strong on hard layouts. Cons: per-page cost against a £0 ingest budget; closed component in an open, reproducible pipeline; still doesn't do statement-ID logic.

**Decision & rationale.** In a citations-first system the chunker *is* product code, not plumbing. Scheduling honesty matters as much as the choice: naming statement-ID chunking as the long-pole and spiking it first is the difference between a plan and a hope.

**Trade-offs / consequences.** Ingest is slower and heavier than naive splitting (irrelevant at ~1,000 pages); the "endless chunking experiments" scope guard applies — a chunking change must beat the retrieval eval or it doesn't ship.

**When you'd choose differently.** Fixed-size chunking is entirely reasonable when sources are structureless (chat logs, transcripts), citations are document-level not passage-level, or you're prototyping retrieval feasibility before investing in ingest. Commercial parsers earn their fee on scanned/handwritten/tabular corpora at volume. Framework chunkers make sense when the pipeline isn't a differentiator and team familiarity dominates.

---

<a name="adr-012"></a>
## ADR-012 — Model: Anthropic cloud (Haiku default, Opus gated), behind a thin provider adapter

**Decision:** Generation uses Anthropic `claude-haiku-4-5` by default with `claude-opus-4-8` as an optional "best" mode gated behind the budget cut-off; a single-interface provider adapter (`generate(messages, documents, config) → AnswerWithCitations`) isolates the choice. Other clouds and local models are deferred.
**Status:** Accepted-MVP (alt backends: Deferred).

**Context & forces.** The citation mechanism (ADR-008) is the deciding constraint: native citations with custom-content blocks are an Anthropic API feature, and they are load-bearing for the guaranteed-vs-measured stance. Cost ceiling is ~£20/month on a public endpoint; prompt caching on the static prefix matters at that margin.

**Options considered.**
- **Local/open-weight generation (Ollama/vLLM).** Pros: $0 marginal cost; full-stack-local story (embeddings and reranker already are); no data leaves the box. Cons: no native citations — falls back to validated `[n]` markers, weakening the headline guarantee; instruction-following and calibrated-language fidelity of small local models is measurably worse, and this product's failure mode is precisely subtle unfaithfulness; hosting a GPU (or tolerating slow CPU inference) breaks the budget worse than Haiku does at ~$0.008/query (~8×400-token passages + prompt ≈ 5–6K input, ~500 output at $1/$5 per MTok; 1,000 queries/month ≈ $5–8).
- **Other clouds via LiteLLM from day one.** Pros: optionality, price arbitrage. Cons: lowest-common-denominator abstraction forfeits native citations; multi-provider testing surface for zero MVP benefit.
- **Opus-only.** Pros: best answers. Cons: ~$0.05–0.08/query blows the cap under modest public traffic.
- **Anthropic, Haiku default + gated Opus, behind an adapter (chosen).** Pros: the only backend whose citation feature satisfies ADR-008; Haiku fits the budget with prompt caching; Opus is available for demos without being a runaway-cost vector (sub-cap within the daily cut-off, ADR-015); the adapter keeps model IDs in config and honestly scopes the "model-agnostic" claim to the interface, not the implementations. Cons: single-vendor coupling for the flagship behaviour; cloud dependency means the demo dies if the API does.

**Decision & rationale.** Choose the vendor whose primitives match the product's hardest requirement, and contain the coupling behind an interface instead of pretending it away. The local path stays *credible* (embeddings/reranker already local; marker-based fallback designed) without being built prematurely.

**Trade-offs / consequences.** Eval numbers are per-model; the A/B harness re-runs the suite on any model swap. The scope classifier/rewriter also run on Haiku as separate structured-output calls (per the citations incompatibility).

**When you'd choose differently.** Data-sovereignty or air-gapped deployments force local models — accept marker citations plus a mandatory entailment gate, and say so in the guarantees. High-volume products where inference cost dominates justify multi-provider routing and serving open weights on owned GPUs. If another provider ships equivalent block-anchored citation output, the adapter is the migration path and the decision genuinely reopens.

---

<a name="adr-013"></a>
## ADR-013 — UI: Streamlit for the MVP; Next.js deferred

**Decision:** The MVP UI is Streamlit: chat with streaming, retrieved-passages panel, source library, and the `/about` transparency page. A Next.js front-end is deferred.
**Status:** Accepted-MVP (Next.js: Deferred).

**Context & forces.** The portfolio value of this project is the RAG/eval/guardrail engineering, not front-end craft — and Rusty Data's website (separately, a Next.js build) already evidences front-end capability. The 3–4 week MVP window and £20/month ceiling both punish UI scope creep. But the UI must still credibly carry the product's distinctive furniture: passages panel with cited-span highlighting, citation chips, likelihood-legend tooltips.

**Options considered.**
- **Next.js from day one.** Pros: full control of interaction design (hover-linked citation chips, mobile sheet, polish); one less rewrite later; matches the client's existing stack. Cons: a week-plus of UI work displacing eval/guardrail work that *is* the differentiator; needs a separate API consumption layer and hosting story; the MVP gates (faithfulness, refusal, published evals) don't get easier with a prettier shell.
- **Gradio.** Pros: fastest chat scaffold. Cons: layout control too coarse for a three-surface layout (chat + passages panel + source drawer); harder to make the transparency page feel like a page.
- **Streamlit (chosen).** Pros: chat + side panel + multipage (`/about`) in days, in Python, deployable on the same small VM (or Community Cloud/HF Spaces at $0); streaming-friendly; the passages panel and about-page — the parts that carry the product's honesty story — are fully achievable. Cons: interaction ceiling (per-token citation-chip hover behaviour is clunky); Streamlit "look" reads as prototype; session-state model gets awkward if the app grows.

**Decision & rationale.** Spend the scarce weeks where the differentiation is. Streamlit clears the bar of "credible demo that makes the guarantees visible"; pixel-grade chat UX is Phase-2 polish once the pipeline's numbers are published and stable — and the FastAPI boundary (ADR-016) means the swap is additive, not a rewrite.

**Trade-offs / consequences.** Some UX ambitions (hover-highlight choreography, mobile citation sheets) ship in reduced form; accessibility is achievable but needs deliberate care in Streamlit.

**When you'd choose differently.** If the portfolio story were "product design" rather than "responsible AI engineering", invert the priority. A client-facing paid product would justify Next.js immediately (branding, auth, analytics, SEO for the source library). And if the team were JS-native rather than Python-native, the calculus flips — Streamlit's advantage is that it keeps one person in one language.

---

<a name="adr-014"></a>
## ADR-014 — Evaluation as a first-class, published deliverable

**Decision:** A ~40-question gold set with gold `chunk_id`s drives retrieval (Recall@8/MRR/nDCG@8), faithfulness, citation-support, calibrated-term-preservation, confidence-level-fidelity, refusal-rate and adversarial evals; retrieval + smoke evals run in CI per PR, the full suite per release, and results are committed to `evals/RESULTS.md` and linked from the live `/about` page.
**Status:** Accepted-MVP.

**Context & forces.** Three forces converge. Product: the guaranteed-vs-measured stance (ADR-009) is *empty without published measurements* — "measured" must point at numbers. Engineering: refusal thresholds (ADR-010) and chunking/rerank decisions (ADR-006/011) are tuned against eval subsets, so the evals are part of the control loop, not commentary. Portfolio: "evaluation as deliverable" is itself the pitch to clients who have seen too many vibes-based LLM demos.

**Options considered.**
- **Ship without evals.** Pros: weeks faster. Cons: the honesty claims become marketing copy; regressions from any pipeline change are invisible; every scope-guard ("a change must beat the retrieval eval") loses its enforcement mechanism; the strongest differentiator is deleted to save the time that made it.
- **Ragas/off-the-shelf harness wholesale.** Pros: fast start; standard metrics. Cons: the *novel* metrics here — citation-support against native-citation blocks, calibrated-term preservation, confidence-level fidelity — don't exist off the shelf; a framework adds a dependency without covering the parts that matter. (Ragas stays optional for the generic faithfulness metric.)
- **Custom scripts + pytest in CI, published (chosen).** Pros: metrics match the product's actual promises; deterministic retrieval metrics are free to run per-PR; LLM-judge metrics (judge ≠ generator, sample human-audited) bounded to per-release; publication creates a credible commitment.
- **Big synthetic eval set now.** Pros: statistical power. Cons: synthetic Q/A quality is its own project; 40 hand-built questions with gold chunks beat 400 unvetted ones for a corpus this small. Deferred.

**Decision & rationale.** In a product whose claim is calibrated honesty, the eval suite is not QA — it is the mechanism that makes the claims true, the thresholds settable, and the marketing falsifiable. Publishing is the forcing function that keeps it maintained.

**Trade-offs / consequences.** Gold set curation is real work and couples to corpus versions; per-release judge runs cost a few pounds; a public bad number must be shipped and fixed in the open (feature, not bug).

**When you'd choose differently.** Throwaway prototypes and internal spikes don't warrant CI evals — a notebook probe (like the Phase-0 18/20 citation gate) is proportionate. Products with fast-shifting corpora need synthetic generation early because hand-built gold sets rot. And where the differentiator is UX rather than epistemics, lighter-weight eval (regression prompts + human review) is the right cost point.

---

<a name="adr-015"></a>
## ADR-015 — Hard daily budget cut-off that fails closed

**Decision:** A server-side hard daily spend cap; on breach the app fails closed to a "demo paused for today" state (no LLM calls, chat disabled, `/about` and source library still served). Opus mode sits behind the same cut-off plus a lower sub-cap. Per-IP rate limiting is an additional layer, not the backstop.
**Status:** Accepted-MVP (verified to fail closed as an MVP gate).

**Context & forces.** A public, unauthenticated LLM endpoint is a runaway-cost machine: scripted abuse, a viral link, or a retry loop can turn $0.008/query into hundreds of pounds overnight. The budget is ~£20/month; the operator is one person who is sometimes asleep.

**Options considered.**
- **Soft caps (alerts/emails).** Pros: no availability impact. Cons: alert-to-action latency is the whole exposure; overnight incidents run for hours; relies on a human being the circuit breaker.
- **Rate limiting only.** Pros: throttles abuse. Cons: caps *rate*, not *spend*; distributed traffic or long-context queries slip under any per-IP limit; no invariant is enforced.
- **Provider-side spend limits only.** Pros: exists anyway (and should be set). Cons: granularity is monthly/org-level and failure UX is an opaque API error mid-answer, not a designed degraded state.
- **Hard daily cap, fail closed, designed degradation (chosen).** Pros: converts worst case from "unbounded bill" to "demo pauses until midnight" — for a portfolio demo, pausing is embarrassing-proof, cheap, and even demonstrates operational maturity; keeping `/about` and the source library up means the informational value survives the pause; the Opus sub-cap makes the expensive mode structurally incapable of consuming the day's budget. Cons: adversaries *can* deliberately exhaust the budget (a sub-£1/day denial-of-wallet — acceptable); legitimate users hit the wall on a popular day; spend tracking must itself be reliable (server-side, atomic; verified by a fail-closed test in the MVP gates).

**Decision & rationale.** Fail-closed inverts the default: the invariant "spend ≤ cap" holds without a human in the loop, and every failure of the tracking mechanism degrades toward *not spending*. For an unattended public demo this is the only posture where the worst case is bounded by design.

**Trade-offs / consequences.** Requires honest UX copy for the paused state; daily reset semantics and the sub-cap are config; provider-side monthly limits remain configured as defence in depth.

**When you'd choose differently.** Revenue-generating products invert the asymmetry — availability is worth more than marginal spend, so you'd use soft caps + autoscaling budgets + paging, failing *open* with throttling. Authenticated products shift to per-user quotas, which are fairer than a global cap. An internal tool behind SSO may need only provider-side limits. The fail-closed global cap is specifically the right tool for *public + unauthenticated + fixed tiny budget + unattended*.

---

<a name="adr-016"></a>
## ADR-016 — Core stack: Python 3.12 + FastAPI, single public monorepo

**Decision:** Python 3.12 throughout; FastAPI (SSE streaming) as the API service; hand-rolled ~300-line pipeline instead of a RAG framework; Docker Compose (`api`, `qdrant`, `ui`) for one-command reproduction; a single public monorepo — `ingestion/`, `rag/`, `ui/`, `evals/`, `corpus/` (manifest only) — under Apache-2.0.
**Status:** Accepted-MVP.

**Context & forces.** One developer, 3–4 weeks, a portfolio artefact that must be *readable* by prospective clients, and a pipeline whose interesting parts (chunking, citation blocks, refusal gate, evals) are bespoke.

**Options considered.**
- **Python + FastAPI (chosen).** Pros: every component's ecosystem is Python-native (Docling, FlagEmbedding/sentence-transformers for bge-m3, Qdrant client, eval tooling); FastAPI gives typed contracts, async, SSE streaming to the UI, and OpenAPI docs for free; matches the client's data-science context. Cons: Python serving is less lean than Go/Node — irrelevant at demo QPS.
- **TypeScript/Node end-to-end.** Pros: one language if the UI were Next.js. Cons: the ML tooling (Docling, local embedding/reranking, eval stack) is Python-first; you'd end up with Python sidecars anyway.
- **LangChain/LlamaIndex as the pipeline.** Pros: speed to first demo; abstractions for free. Cons: the abstractions hide exactly the layers this project exists to showcase; framework churn and indirection make a public codebase *harder* to read; the bespoke parts (statement-ID chunking, custom-content citation blocks, reranker-thresholded refusal) fight the framework's opinions. Hand-rolling ~300 lines is less code than configuring around a framework.
- **Polyrepo** (ingestion / api / ui / evals split). Pros: independent versioning. Cons: four repos for one person is pure overhead; cross-cutting changes (chunk schema → retrieval → eval fixtures) become multi-repo dances; the monorepo *is* the portfolio artefact — one link, one `docker compose up`, one README arc. `corpus/` holds manifest + fetch scripts only, so no restricted text is ever committed (public-domain/CC prepared data may ship; deferred-tier content never does).

**Decision & rationale.** Minimise the distance between "clone" and "understand": one language, one repo, no framework indirection, one compose file. Apache-2.0 (explicit patent grant) over MIT is a deliberate professional signal for client-facing reuse.

**Trade-offs / consequences.** Public repo means licensing hygiene is enforced by structure (manifest-only corpus dir) not just policy; structured JSON logs (no user identifiers) keep observability lightweight with Langfuse deferred.

**When you'd choose differently.** A team of several with independent deploy cadences justifies service/repo splits. A product where RAG is a commodity feature (not the showcase) is exactly where LangChain/LlamaIndex earn their keep — speed over transparency. If the org standardises on TypeScript and the ML surface is thin (API embeddings, no local models), Node end-to-end is defensible.

---

<a name="adr-017"></a>
## ADR-017 — Naming: "Ask About the Climate", with an explicit non-affiliation disclaimer

**Decision:** Keep the name **Ask About the Climate** (tagline: "Answers from the climate science literature — with receipts"), repo `ask-about-the-climate`, and display wherever sources appear: *"Not affiliated with or endorsed by NASA, NOAA, Copernicus, USGCRP, or the IPCC. All sources are cited and linked."*
**Status:** ~~Accepted-MVP~~ **Superseded by ADR-022 (v3)** — renamed to *Let's Talk About the Climate Emergency*. The non-affiliation reasoning below stands and is extended to the National Emergency Briefing campaign.

**Context & forces.** The name is a trust surface. A product built on institutional sources can accidentally borrow institutional authority it doesn't have — which for an anti-misinformation product would be a self-inflicted credibility wound, and for some sources a licence/attribution problem (attribution must not imply endorsement; Copernicus requires a specific attribution string, carried in citation metadata per document).

**Options considered.**
- **"ClimateGPT".** Pros: instantly legible category. Cons: implies a fine-tuned model, which ADR-004 explicitly rejects — the name would misdescribe the architecture; "GPT" points at a competitor's brand; existing name collisions.
- **"IPCC Chat" / source-anchored names.** Pros: authority by association. Cons: implies endorsement that does not exist — and IPCC isn't even in the launch corpus (ADR-001), so the name would be false twice over.
- **"Climate Librarian".** Pros: matches the librarian-not-oracle framing. Cons: obscure to the general audience; explains the philosophy, not the use.
- **"Ask About the Climate" (chosen).** Pros: describes the *action*, not a claimed authority; plain-language, matching the general-public audience; no borrowed branding; the tagline carries the differentiator ("with receipts"). Cons: generic-sounding; longer; domain availability (`askabouttheclimate.org` / `askaboutthe.climate`) must be verified before launch.

**Decision & rationale.** Name the interaction, disclaim the affiliation, and let the citations panel earn the authority. The disclaimer is not legal boilerplate — it is the naming decision's enforcement mechanism, placed on every surface where sources-by-implication could mislead (source library, passages panel, `/about`).

**Trade-offs / consequences.** A descriptive name does less marketing work; the disclaimer costs UI space on small screens (kept to one line, always adjacent to source listings).

**When you'd choose differently.** With a formal partnership or written endorsement from a source institution, an anchored name (and co-branding) becomes accurate and valuable — that is a licensing outcome, not a naming choice. If the product later fine-tuned models (contra ADR-004), model-implying names would stop being false. And a B2B white-label version would drop the public-facing name entirely in favour of the client's brand — with the same non-affiliation disclaimer surviving underneath, because it protects the sources, not the brand.

---

<a name="adr-018"></a>
## ADR-018 — Non-commercial educational public-benefit framing (supersedes the commercial-portfolio framing)

**Decision:** The project is an educational piece of software for public benefit: free to use, no advertising, nothing sold, code and evals public. Rusty Data builds and stewards it and may reference it as its work, but the *product* is non-commercial, and licensing decisions may rely on that status. Every corpus document records `permitted_context` (`open | non-commercial-educational | permission-on-file`); commercialising the project later requires first removing everything not `open` or covered by permission — a mechanical operation over the manifest.
**Status:** Accepted (v3). Supersedes the "commercial context" standing force under which ADRs 001–017 were written.

**Context & forces.** The client's restated purpose (2026-08): close the public's awareness gap about the severity of the climate emergency, in line with the National Emergency Briefing campaign's diagnosis that the public has never been properly briefed. That is a public-benefit mission, and the commercial framing was actively costing us corpus: UNEP's Emissions Gap Report permits educational/non-profit reproduction; Carbon Brief and Berkeley Earth are CC BY-NC-*; WMO leans NC; and permission requests (IPCC, OUP) are far stronger from a free educational tool than from a consultancy's portfolio piece.

**Options considered.**
- **Stay commercial.** Pros: no ambiguity about Rusty Data's benefit; wider future monetisation. Cons: locks out the NC tier; weakens permission letters; and misstates the actual purpose.
- **Fully separate legal entity (CIC/charity).** Pros: cleanest non-commercial status. Cons: administrative overhead disproportionate to an MVP; can be revisited if the project grows.
- **Non-commercial project under Rusty Data stewardship (chosen).** Pros: honest, lightweight, unlocks Tier B, strengthens permissions. Cons: a grey zone — Rusty Data derives reputational benefit. Mitigations: the product itself has no revenue, no ads, no lead-capture; NC-licensed content is never redistributed via the repo; the `permitted_context` field makes de-commercialisation auditable and re-commercialisation mechanical; if in doubt on a specific NC source, ask the licensor (recorded in `permission_evidence`).

**Trade-offs / consequences.** Rusty Data must not use the hosted product in paid engagements (demoing the open-source *code* is fine — the code is Apache-2.0; it is the NC-licensed *content* that is restricted). The `/about` page states the non-commercial commitment publicly, which is both a trust surface and a self-binding mechanism.

**When you'd choose differently.** If the project were to become a paid product or embedded in client deliverables, Tier B comes out first and the corpus reverts to roughly v2's safe tier — the manifest is designed so that this is a filter, not a forensic investigation.

---

<a name="adr-019"></a>
## ADR-019 — Corpus restructure: permission tiers, emergency-communications sources, and a non-corpus "voices" layer

**Decision:** Restructure the corpus into Tier A (open/public-domain), Tier B (non-commercial licences, unlocked by ADR-018), and Tier C (permission-pending, link-only until a written affirmative reply is on file). Add emergency-relevant sources on the strength of verified licences: Hansen 2023 & 2025 (both CC BY 4.0 — Tier A), UNEP Emissions Gap Report (UN educational-use terms — Tier A under our framing), Met Office (OGL v3 — Tier A), OWID (CC BY — Tier A), Carbon Brief (CC BY-NC-ND, verbatim chunks — Tier B). Keep Ripple et al./BioScience, WMO, and IPCC full text in Tier C (the Ripple papers are free-to-read but **all-rights-reserved OUP**, verified 2026-08 — a trap the hardened gate exists to catch). The Conversation is excluded outright pending clarification of its explicit no-AI/ML-use republishing term. Campaign and communicator content (Packham, NEB, Climate Majority Project, warming stripes…) lives in a **voices layer**: first-party descriptive text authored by this project, ingested as a distinct labelled source for "who is saying this / what can I do" questions, and a public "Voices & action" page — never cited for scientific claims.
**Status:** Accepted (v3). Amends ADR-001/002.

**Context & forces.** The mission update demands the corpus actually contain the emergency-severity syntheses (acceleration, pipeline warming, emissions gap, tipping points) and that users can find the movement. But an anti-misinformation product must never cite a broadcaster or campaign for a scientific claim — that is the exact attack surface bad-faith critics look for. The severity content is available *in the literature* under usable licences; the movement content is factual meta-information we can author ourselves.

**Options considered.**
- **Ingest campaign/broadcast material into the evidence corpus.** Pros: directly reflects the client's "specific outlets and specific people" intent. Cons: mostly all-rights-reserved; and it would collapse the evidence/advocacy distinction that gives the bot its authority. Rejected.
- **Ignore the movement entirely.** Pros: purest. Cons: fails the mission — users who ask "what can I do?" or "who is calling for this?" deserve grounded answers, and the campaign is the project's own context. Rejected.
- **Voices layer of first-party text + link library (chosen).** Pros: freely licensable (we wrote it); clearly labelled; keeps the evidence corpus pristine; gives "what can I do" answers somewhere real to land. Cons: our descriptions need their own upkeep (`as_of` dates on snapshot facts) and editorial care to stay descriptive, not promotional.

**Trade-offs / consequences.** Two content pipelines with different trust labels; a tested invariant (science answers never cite `source_type: voices`); Carbon Brief chunks must be displayed unadapted (ND) — the position that answers may synthesise *facts* from them is recorded here and flagged for the Phase-1.5 legal sanity-check, alongside the IPCC curated-headline-statements question.

**When you'd choose differently.** With written permissions in hand (Ripple/OUP, WMO, IPCC), Tier C promotes into the evidence corpus and the voices layer shrinks to genuinely non-scientific content. A pure science-Q&A product with no mobilisation goal would skip the voices layer; a pure campaign tool would invert the balance — this project is deliberately both, with the boundary enforced.

---

<a name="adr-020"></a>
## ADR-020 — Chart generation: curated data pack + closed declarative ChartSpec + server-side rendering (never LLM code generation)

**Decision:** Chart requests are planned by a structured-output LLM call that emits a **ChartSpec** — a JSON document over a *closed* vocabulary (dataset ids from the curated pack, a fixed transform set, enumerated chart types, title/range/annotations). A pure-code validator checks it; a server-side renderer (Vega-Lite via `vl-convert`) produces SVG/PNG with a baked-in caption strip (sources, licences, access date, site URL). Outputs: inline chart, PNG/SVG/CSV download, `/chart/<spec-hash>` permalink, iframe embed. The model never writes executable code and never supplies data values.
**Status:** Accepted-MVP (v3).

**Context & forces.** Shareable graphics are the mission's force multiplier — a chart travels further than a paragraph — but a *wrong* chart travels furthest of all, and this product cannot afford one. The same guaranteed-vs-measured discipline as citations (ADR-009) must apply: what is guaranteed is that every pixel derives from named public datasets through auditable code.

**Options considered.**
- **LLM writes matplotlib/Python, sandboxed.** Pros: unlimited flexibility. Cons: code injection surface; irreproducible outputs; data values can be hallucinated inside generated code; sandbox engineering out of proportion to the MVP. Rejected.
- **Client-side charting library driven by the model.** Pros: interactive. Cons: same hallucination risk, plus no server-side artefact for download/embed with baked attribution. Rejected for MVP (a Vega-Lite spec can later power interactivity from the *same* ChartSpec).
- **Closed ChartSpec + server render (chosen).** Pros: validatable, reproducible from ~1 KB of JSON, injection-proof, $0/chart, permalinks trivial; chart-integrity rules (labelled baselines, axis rules, full-range defaults, uncertainty bands, colour-blind-safe palette, direct labelling) enforced in the renderer rather than requested of the model. Cons: bounded expressiveness — requests outside the vocabulary get an honest refusal naming the nearest supported chart; the vocabulary grows by code review, not prompt engineering.

**Trade-offs / consequences.** The data pack becomes a second manifest-disciplined corpus (provenance, licence, sha256, pinned versions, update scripts). The flagship "CO₂ + temperature over 10,000 years" request requires paleo datasets (Bereiter composite, Jouzel EPICA temperature, Kaufman Temp-12k) whose fixed-format TXT needs one-time committed parsers — de-risked in Phase 0. Chart evals (spec accuracy, data faithfulness, refusal correctness, cherry-pick resistance) join CI.

**When you'd choose differently.** An internal analyst tool with trusted users could justify sandboxed codegen for flexibility. A data-journalism shop with a human reviewing every output could too. A public anti-misinformation tool cannot.

---

<a name="adr-021"></a>
## ADR-021 — No open web search; allowlisted provider fetch as the Phase-2 fallback

**Decision:** When a chart request needs data outside the pack, the MVP refuses honestly, names the nearest available datasets, and logs the gap for curation. Phase 2 adds a fetcher restricted to an allowlist of trusted data providers (NOAA, NASA, Met Office, NSIDC, OWID, Copernicus), with schema validation, a visible "new source — not yet curated" label, and automatic nomination of fetched datasets for pack promotion. General web search for data is rejected outright.
**Status:** Accepted (v3): refusal+logging in MVP; allowlisted fetch in Phase 2.

**Context & forces.** The client's intent — "if the data is not already available… it can go and find the data" — is right about the need but the open web is where the misinformation lives; an anti-misinformation tool that charts whatever a search engine returns has imported its adversary's supply chain. Fetched web content is also a prompt-injection vector, and unvetted data breaks the "every pixel from named datasets" guarantee (ADR-020).

**Options considered.** Open web search (rejected: misinformation, injection, licence-unknown data, guarantee-breaking); MVP-scope allowlisted fetch (rejected for schedule: schema validation across providers is real work, and the 10-dataset pack already covers the overwhelming majority of plausible public questions); refuse-and-log now, allowlisted fetch later (chosen: the gap log tells us which datasets people actually want, so the pack grows demand-driven with human review in the loop).

**When you'd choose differently.** If usage shows high refusal rates on reasonable requests, promote the fetcher into the next release — the allowlist and "not yet curated" labelling are designed so this is an addition, not a rework.

---

<a name="adr-022"></a>
## ADR-022 — Naming: "Let's Talk About the Climate Emergency"

**Decision:** Rename to **Let's Talk About the Climate Emergency** (short form *Let's Talk Climate Emergency* in UI chrome; repo `lets-talk-climate-emergency`). Tagline: *"The emergency briefing you haven't had — answers from the science, with receipts."* Non-affiliation disclaimer extended to the National Emergency Briefing campaign; approach NEB about a friendly listing/partnership but build assuming none.
**Status:** Accepted (v3). Supersedes ADR-017 (whose non-affiliation reasoning stands).

**Context & forces.** "Ask About the Climate" named the interaction but not the mission; the reframed project is explicitly about the *emergency* and about starting conversations (the client's own words: "let's talk about the climate emergency"). Naming checks (2026-08): the exact string appears clear; adjacent brands are ecoAmerica's *Let's Talk Climate*, Climate Outreach's *Britain Talks Climate*, and the NEB campaign; the chatbot space (ChatClimate, ClimateQ&A, ClimateGPT) has no collision.

**Options considered.** **"Climate Emergency Briefing"** — rejected: reads as the NEB campaign's own product; implied affiliation is exactly what ADR-017 exists to prevent. **"How Bad Is It?"** — arresting, but flippant out of context and search-hostile; kept as a landing-page section heading where it works hard. **Keep "Ask About the Climate"** — rejected: neutral to the point of undercutting the mission. **"Let's Talk About the Climate Emergency" (chosen)** — invitational rather than hectoring (matches the evidence-calm-cited voice, and the door-knocking use case: something you'd actually send a neighbour), states the emergency plainly, and does not borrow anyone's authority. Cons: long (mitigated by the short form); "emergency" in the name will be called alarmist — which is survivable precisely because every answer carries receipts, and is honest labelling of what the sources say.

**Trade-offs / consequences.** Rename ripples through repo name, domains (`letstalkclimateemergency.org`, `climateemergency.chat` — verify), UI chrome, and the disclaimer line. A trademark search is still prudent before public launch (the check performed was a search-engine collision check only).

---

## Key sources

- [UK CDPA 1988, s.29A — legislation.gov.uk](https://www.legislation.gov.uk/ukpga/1988/48/section/29A) (TDM copies: "sole purpose of research for a non-commercial purpose"; dealing otherwise makes an infringing copy)
- [IPCC — Copyright](https://www.ipcc.ch/copyright/) (limited figures/short excerpts free with acknowledgement; other reproduction requires written permission)
- [Anthropic — Citations](https://platform.claude.com/docs/en/build-with-claude/citations) (all-or-none per request; incompatible with structured outputs; custom-content blocks cite at defined block granularity via `content_block_location`)
- [BGE-M3 — BAAI model card](https://huggingface.co/BAAI/bge-m3), Chen et al. 2024 (single model, dense + sparse + multi-vector, 8192-token context)
- [Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet…*, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114) (RRF definition — rank-only fusion)
- [Qdrant — Hybrid queries / Query API](https://qdrant.tech/documentation/concepts/hybrid-queries/) (native sparse vectors, server-side RRF fusion)
- Licence-metadata agreement (~63% three-way) — legally-screened corpus study, [arXiv:2604.12498](https://arxiv.org/pdf/2604.12498)
- [NASA Earthdata — Data Use Guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) · [NOAA NAO 205-17A](https://www.noaa.gov/organization/administration/nao-205-17a-information-access-dissemination) · [Licence to Use Copernicus Products](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products)
- Anthropic pricing (verified 2026-08): `claude-haiku-4-5` $1/$5 per MTok; `claude-opus-4-8` $5/$25 per MTok
