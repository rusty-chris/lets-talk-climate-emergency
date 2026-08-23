# Let's Talk About the Climate Emergency — Design Document v3.1 (build-ready)

**A free, open-source, public-benefit chatbot that gives people the emergency briefing on climate they have never had: straight answers grounded strictly in authoritative publications, with inline citations to the exact source passages — plus the ability to generate shareable, source-stamped charts from canonical climate datasets.**

- **Client / owner:** Chris McWilliams (Rusty Data) — acting as steward of a **non-commercial public-benefit project**
- **Status:** Design v3.1 — 2026-08 — hand-off spec for implementation agents. v3.1 incorporates all 20 findings of an adversarial design review (see `reviews/critic-2026-08-16.md`), including two blockers: licensing-invariant/tier-table contradictions, and a ChartSpec vocabulary that could not express the flagship chart.
- **Code licence:** Apache-2.0 (explicit patent grant)
- **Repo shape:** single public monorepo — `ingestion/`, `rag/`, `charts/`, `ui/`, `evals/`, `corpus/` (manifest only), `datasets/` (manifest + fetch scripts + pinned hashes; **no data files in git** — ADR-023)
- **Changes vs v2:** project reframed from commercial portfolio piece to **non-commercial educational public-benefit** (ADR-018) — this changes the licensing calculus throughout; mission recentred on **communicating the climate emergency** to a public that largely does not know how serious it is, aligned with the National Emergency Briefing campaign (ADR-019); corpus restructured into permission tiers with an emergency-communications tier and a separate non-corpus "voices" layer (ADR-019); **chart-generation feature** added — curated data pack + declarative chart specs + server-side rendering, with download/embed and baked-in attribution (ADR-020); open web search rejected in favour of an allowlisted data-fetch fallback (ADR-021); renamed (ADR-022); landing page redesigned around clickable starter topics; severity-fidelity ("don't bury the lede") added as a guardrail and eval alongside the existing calibration checks.

> **Framing (read first).** This is an **educational piece of software for public benefit**. It is free to use, carries no advertising, sells nothing, and its code and evaluation results are public. Rusty Data builds and stewards it and may point to it as work it has done, but the product itself is non-commercial: we rely on this in licensing decisions (several key sources permit *non-commercial / educational* reuse and are ingestable only on that basis). Consequence, recorded in the manifest per document: **if the project ever becomes commercial, every Tier-B (non-commercial-licensed) document must be removed from the corpus** — the manifest's `permitted_context` field makes that mechanical. We still do **not** self-certify fair use / UK TDM exceptions for full-text indexing of unlicensed works; those remain behind written permission.

---

## 1. Mission, problem & audience

### The problem this project addresses

Most people do not know how serious the climate emergency is. The gap is not primarily one of data — the science is public — but of **communication**: very few mainstream outlets state plainly what the assessed literature says about where we are and where we are heading. This is the gap the **National Emergency Briefing** campaign (nebriefing.org) exists to close: in November 2025 Chris Packham convened leading scientists to brief 1,200+ UK politicians and leaders at Westminster; the campaign's open letter — backed by a parliamentary petition (~100k signatures), an Early Day Motion, 91 MPs, and the April 2026 film *The People's Emergency Briefing* (2,000+ community screenings) — calls for a televised national emergency briefing on climate and nature. That briefing has not yet happened.

*Let's Talk About the Climate Emergency* is a self-serve version of that briefing: a conversational interface to the authoritative literature, for the neighbour or acquaintance who has never had the situation laid out for them. It communicates the emergency **at the severity the sources themselves state** — no inflation, no soft-pedalling — with every claim cited and the source text one click away.

### Mission principles

1. **The emergency is in the sources.** The assessed literature — IPCC Synthesis Report, UNEP Emissions Gap, national assessments, Hansen et al. — states the situation plainly and severely. We do not need to editorialise; we need to *surface*. Severity claims come from cited scientific sources, never from tone.
2. **Don't bury the lede.** Answers lead with the headline finding at its assessed severity. An answer that is technically accurate but structured to reassure is a failure mode we test for (severity fidelity, section 6), symmetric with overclaiming.
3. **Epistemic honesty over fluency.** Calibrated uncertainty language (*very likely*, *high confidence*) is preserved verbatim — in both directions. Credibility is the product: one inflated claim, screenshotted, undoes the mission.
4. **Verifiability over convenience.** Citations point to document, section and page; the UI shows retrieved passages; generated charts carry their data sources on the image itself.
5. **The bot briefs; the voices mobilise.** Scientific claims are grounded in the literature corpus. The people and campaigns publicly communicating the emergency (Packham, the NEB experts, the Alliance of World Scientists…) are surfaced through a distinct, clearly-labelled **voices layer** (section 2.5) — connecting users to the human movement without ever citing a broadcaster for a scientific claim.

### Audience

| Audience | Need | Design implication |
|---|---|---|
| **The uninformed-but-reachable public** (primary — "my neighbours") | "How bad is it, actually?" laid out plainly, from scratch | Zero-jargon answers; clickable starter topics; shareable charts; no assumed knowledge |
| | | **Reach model (be honest about it):** emergency framing lands with the already-concerned and can repel disengaged/sceptical visitors (the Britain Talks Climate segmentation finding). The primary conversion path is therefore **second-hand**: concerned users sending charts, permalinks and cited answers to their neighbours — the product optimises for shareable artefacts first, organic sceptic arrival second. Landing-page framing variants are an explicit post-launch test; sceptic-facing starter answers are tone-audited for "welcoming to the unconvinced" in the adversarial rubric (section 6). |
| Concerned people who want to talk to others | Material for conversations — facts, charts, links to voices | Downloadable/embeddable graphics; copyable cited claims; voices layer |
| Students & educators | Traceable claims, classroom-usable graphics | Copyable citations; source library; chart permalinks |
| Journalists | Fast, quotable, checkable facts | Exact page/section references; retrieved-passages panel |

---

## 2. Corpus & ingestion

### 2.1 Permission tiers (researched 2026-08; re-verify at ingest, per-document)

Licensing governs what we may **ingest / index**, **quote**, and **redistribute**. Building a private full-text index is itself reproduction of the whole work, so indexing rights — not just display rights — decide scope. The non-commercial reframing (ADR-018) unlocks Tier B; it does not change our refusal to self-certify exceptions for unlicensed full-text indexing (Tier C).

**TIER A — open / public-domain (ships in MVP, safe under any framing):**

| Source | Licence / terms (verified 2026-08) | Use |
|---|---|---|
| **US National Climate Assessment (NCA5)** | US federal government work — public domain | Ingest, quote, redistribute; attribute anyway. Impacts/adaptation coverage. |
| **NASA** climate content (climate.nasa.gov, Earthdata explainers) | Generally not copyrighted; mission data default CC0; exclude items with third-party credits | Ingest text; attribute. |
| **NOAA** (climate.gov, NCEI explainers) | US government work — public domain; exclude third-party-credited items | Ingest explainers; attribute. |
| **Copernicus / C3S** (European State of the Climate, CDS docs) | Licence to Use Copernicus Products — free reuse incl. redistribution, attribution required (*"Generated using Copernicus Climate Change Service information [Year]"*) | Ingest; carry exact attribution string. |
| **Met Office** climate explainers & UKCP material | **Open Government Licence v3.0** by default (metoffice.gov.uk/policies/legal); check per-page exceptions | Ingest OGL-covered pages; attribute *"Contains public sector information licensed under the OGL v3.0"*. UK-relevant framing — valuable for a UK-first launch. |
| **Hansen et al. 2023, "Global warming in the pipeline"** (Oxford Open Climate Change) | **CC BY 4.0** (Crossref-verified) | Ingest full text. Key emergency-relevant synthesis (pipeline warming, sensitivity). **Manifest flag `consensus_position: beyond-assessed-range`** — Hansen's sensitivity/committed-warming claims deliberately exceed IPCC assessed ranges; see the severity-skew guardrail in 2.3. |
| **Hansen et al. 2025, "Global Warming Has Accelerated"** (*Environment*, Taylor & Francis — **not** OOCC) | **CC BY 4.0** (Crossref-verified) | Ingest full text. Acceleration + Earth energy imbalance. Same `consensus_position` flag. |
| **Our World in Data** climate explainers | **CC BY 4.0** for OWID's own content; upstream third-party data keeps original licences | Ingest OWID-authored text; attribute; verify upstream per item. |
| **Hand-verified CC-BY / CC0 review papers** | Per-item via the hardened gate (2.2) | Ingest items passing the gate; store DOI + licence + evidence. |

**TIER B — non-commercial licences (unlocked by ADR-018; ships in MVP with per-doc sign-off):**

| Source | Licence / terms (verified 2026-08) | Use |
|---|---|---|
| **UNEP Emissions Gap Report** | UN standard terms printed in each edition: reproduction in whole or part **"for educational or non-profit services without special permission"** with acknowledgement; no commercial use without written UN permission | Ingest under the educational framing (`permitted_context: non-commercial-educational`); attribute. The single best "how far off track are we" source. NC-conditioned, so **Tier B** — never committed to the repo (see invariants). |
| **Carbon Brief** explainers & analyses | **CC BY-NC-ND**: unadapted reproduction in full for non-commercial use, credited with link (carbonbrief.org/about-us) | Ingest as **verbatim chunks only** (ND: excerpts displayed unadapted; answers synthesise *facts*, which are not copyrightable — but LLM paraphrase that tracks an ND source's expression is the realistic failure mode: a gold-set item verifies Carbon-Brief-derived answer text either quotes verbatim or states bare facts. Position recorded in ADR-019, in scope for the Phase-1.5 legal sanity-check). |
| **Berkeley Earth** (data + text) | **CC BY-NC 4.0** | Ingest **via fetch script only — never committed to the repo, and never in the chart data pack** (exported charts travel into arbitrary contexts, including commercial ones). Flag for removal if the project ever commercialises. |

**NC-confirmation letters (Phase 1.5, week 1):** write to Carbon Brief and Berkeley Earth describing the project and requesting written confirmation that this use qualifies as non-commercial under their licences; record replies in `permission_evidence`. Reliance on our own NC self-assessment is defensible in the interim; the letters bound the project's largest licensing assumption for the cost of two emails.

**TIER C — permission-pending (link-only until a written affirmative reply is on file):**

| Source | Status (verified 2026-08) | Gate to add |
|---|---|---|
| **IPCC AR6** (SYR, SPMs, TS, chapters) | Copyright IPCC. Policy allows **limited figures / short excerpts** free with full acknowledgement; anything more needs written permission from the Secretariat | (a) Write to the IPCC Secretariat describing the project as **non-commercial educational public-benefit** (a materially stronger request than v2's commercial framing) and requesting permission to index full text for retrieval with short-excerpt display + deep links; (b) written affirmative reply; (c) ideally qualified legal sign-off. Meanwhile: the **curated headline-statements set** below, plus link-outs. |
| **IPCC curated headline statements** (hand-picked, **capped at ≤10 statements per SPM** until permission lands) | The short-excerpt allowance covers *limited* short excerpts with full acknowledgement. Caution: UK substantiality is qualitative — a report's *complete* headline-statement set is its distilled expressive core, so we take a strict subset, not the set | Hand-curate; display verbatim with complete acknowledgement (report, publisher, section); **behind a feature flag** documented in the manifest (a refusal from the IPCC is then a config change, not a scramble); the curated set is named explicitly in the permission letter; legal sanity-check + human sign-off before enabling. |
| **Ripple et al. "World Scientists' Warning of a Climate Emergency" + annual State of the Climate reports** (BioScience) | **NOT open access in the licensing sense** — free-to-read under OUP's all-rights-reserved standard model (verified on the 2023 PDF and Crossref). A common mistake to assume otherwise | Write to OUP (journals.permissions@oup.com) and/or the authors (Alliance of World Scientists) for permission to index; the papers' stated purpose — public warning — makes an affirmative reply plausible. Until then: link-only in the voices layer, short quotes with attribution only. |
| **WMO State of the Global Climate** | Site policy: short excerpts free with credit; full publications need permission (publications@wmo.int). Per-edition PDF copyright pages vary — check each | Request permission; or ingest only editions whose own copyright page grants NC reuse. |
| **The Conversation** | CC BY-ND **but** republishing terms explicitly prohibit **use of articles for AI/ML training** and systematic republication | **Excluded from the corpus** unless written clarification confirms retrieval-indexing is acceptable. Link-only. Do not assume the CC licence overrides the sitewide terms. |

**Out of scope for the corpus (any tier):** general news, advocacy sites, blogs, broadcast/video transcripts, preprints, paywalled papers, social media, AI-generated summaries. Skeptical Science's myth taxonomy informs the **adversarial eval set only**. Campaign material (NEB, film) lives in the voices layer (2.5), not the corpus.

**Licensing invariants (enforced in code — all keyed on `permitted_context`, not on tier labels):**
- Every document record carries `licence`, `licence_evidence`, `attribution_text`, `canonical_url`, `redistributable: bool`, `permitted_context: open | non-commercial-educational | permission-on-file`, `permission_evidence` (required when `permission-on-file`), `consensus_position` (default `assessed`; `beyond-assessed-range` where applicable), `sha256`, `retrieved_at`, `human_signoff`.
- The build **refuses to index** any document whose `permitted_context` is unset, or whose `human_signoff` is empty, or whose `permitted_context` is `permission-on-file` with empty `permission_evidence`.
- **Only `permitted_context: open` documents may ship as prepared text in the repo.** Everything else — regardless of tier label — ships as manifest + fetch scripts only (committing NC-conditioned text into an Apache-2.0 repo would exceed the grant).
- **The chart data pack (3.7) must contain only `open` datasets** — exported chart images are redistributed by users into arbitrary contexts, including commercial ones, and must never put *them* in breach. **No dataset files are committed to git in any case** (ADR-023): CI enforces that the only data-like files in the repo are marked synthetic fixtures, and that fetched datasets verify against their pinned hashes.

### 2.2 The CC-BY licensing gate (unchanged from v2, hardened)

Automated licence lookups disagree often (~63% agreement across OpenAlex/Crossref/Unpaywall), so a single lookup is a candidate filter, never authorization:

1. **Candidate filter (automated):** ≥2 of OpenAlex/Crossref/Unpaywall agreeing on CC BY / CC BY-SA / CC0.
2. **Publisher-page confirmation:** fetch the actual article page/PDF; capture the licence statement verbatim into `licence_evidence`. *(The Ripple case above is exactly the failure this step catches: free-to-read pages that automated sources mislabel as open.)*
3. **Human sign-off (required):** `human_signoff: {who, date, note}` per document.

### 2.3 MVP corpus vs later

| | MVP corpus | Later (post-permissions) |
|---|---|---|
| Gov/agency | NCA5 key chapters; NASA/NOAA/Met Office explainers; C3S ESOTC; UNEP EGR latest (~250–400 pages) | Full NCA5; more C3S; WMO |
| Emergency syntheses | Hansen 2023 + 2025 (CC BY); OWID explainers; Carbon Brief verbatim set | Ripple/BioScience papers; IPCC full text |
| IPCC | Curated headline-statements set (post legal sanity-check) + link-outs | AR6 SPMs/TS/chapters after written permission |
| Reviews | ~10–15 hand-verified CC-BY/CC0 reviews (attribution, tipping points, impacts, mitigation, health) | Curated CC-BY set via DOI pipeline |
| Size | ~700–1,200 pages, ~1–2M tokens | grows per permission |

Small enough to hand-audit every document — a feature (trust), not a limitation.

**Severity-skew guardrail (launch dependency).** The Hansen papers deliberately sit *above* IPCC assessed ranges; with IPCC full text deferred, a corpus of "Hansen in, assessed ranges out" would skew answers high — the inflation-screenshot failure mode is as fatal as soft-pedalling. Therefore: **the corpus must contain the assessed-range statements for sensitivity, committed warming and warming levels before launch** — via the IPCC curated headline-statements set if the legal check clears it, otherwise via NCA5's equivalent assessed statements (public domain, always available). A system-prompt rule requires answers drawing on `consensus_position: beyond-assessed-range` documents to say where they exceed assessed findings, and gold-set items on sensitivity/committed warming verify both the retrieval of assessed ranges and the labelling.

### 2.4 Ingestion pipeline (unchanged from v2 in mechanism)

```
fetch (verify sha256) -> parse -> structure-aware chunk -> attach citation metadata
       -> custom-content citation blocks -> embed -> index
                     \-> manifest (licence + evidence + human_signoff + provenance)
```

Parsing: HTML direct; PDFs via **Docling**; PyMuPDF is a **degraded fallback, not an equivalent path** — the Phase 0 spike measured it recovering 3/33 headings on a two-column paper (59/60 chunks under one bogus section), so structure-aware chunking effectively requires Docling. The fallback exists to keep ingestion alive, must be **loud** (per-document warning recorded in the manifest entry, never a silent code path), and fallback-parsed documents are flagged for hand review before indexing. Chunking: by section heading, ≤~500 tokens, never across headings; papers by section; **IPCC headline statements: one statement = one chunk** (natural units — no statement-ID parsing long-pole for the curated set; full statement-ID chunking remains the deferred-tier long-pole). Every chunk gets a prepended context header and the citation-metadata schema from v2 (with `commercial_use_ok` replaced by `permitted_context`). Custom-content citation blocks: one block per citable unit. Embedding idempotent/incremental.

### 2.5 The voices layer (NEW — not part of the RAG evidence corpus)

Purpose: connect users to the people and campaigns publicly communicating the emergency, without ever citing them as scientific evidence.

- **What it is:** a curated, hand-written library — `voices/voices.yaml` + first-party descriptive text **authored by this project** (and therefore freely ingestable) — covering: the **National Emergency Briefing** campaign (open letter, petition, EDM, the Westminster briefing, *The People's Emergency Briefing* film and community screenings); the named NEB experts (Anderson, Lenton, Fowler, Montgomery, Seddon, Behrens, Mann, Haigh, Nugee, Francis, Khan…); Chris Packham's climate work; the Alliance of World Scientists / Ripple warnings (link-only); Ed Hawkins' warming stripes / #ShowYourStripes; The Climate Majority Project & SAFER; Covering Climate Now; Sir David King / CCAG.
- **How the bot uses it:** the first-party descriptions are ingested as a distinct source (`source_type: voices`) so questions like *"Who is calling for an emergency briefing?"* or *"What can I watch or join?"* get grounded, linked answers. Answers from voices chunks are visually labelled **"About the movement"** — never mixed into scientific-claim answers.
- **How the UI uses it:** a **"Voices & action"** page: campaign links, the film, petition, screenings, each expert with one line and a link. This is where "what can I do?" answers point.
- **Update cadence:** snapshot facts (signature counts, MP counts) carry `as_of` dates in the YAML and are rendered with them.

---

## 3. RAG pipeline

```
question
  -> [separate call] query rewrite + scope classify (structured output)
        |-- chart_request --> chart pipeline (section 3.7)
  -> hybrid retrieval (top-40) -> cross-encoder rerank (top-8, calibrated scores)
  -> refusal gate (rerank score threshold)
  -> [separate call] grounded generation w/ Claude native citations (custom-content blocks)
  -> citation-support validation (runtime, one batched call; calibration &
     severity judges are EVAL-TIME ONLY — see 3.3)
  -> response + retrieved-passages panel (verbatim) [+ inline chart if requested]
```

### 3.1 Query processing
- **Query rewriting** (`claude-haiku-4-5`, structured output): resolve references, expand acronyms.
- **Scope classifier** (structured output): `in_scope | chart_request | voices | out_of_scope | adversarial_in_scope | unsafe`. `chart_request` routes to the chart pipeline (3.7); `voices` biases retrieval toward the voices source; adversarial-in-scope routes to normal retrieval with a tone flag. Structured-output calls remain **separate** from the generation call (native-citations incompatibility, 3.4).
- **Unsafe handling (a public unauthenticated chatbot will see all of this in week one):** `unsafe` inputs get canned responses per subtype — self-harm content gets a signposting response (Samaritans 116 123 for a UK-first deployment), harassment/abuse gets a polite disengage — with **no LLM generation call** (cost + safety). Unsafe-classified exchanges are excluded from eval-case harvesting.
- **Non-English queries (MVP rule):** the corpus is English; detect non-English input and answer in English with a one-line note explaining why (bge-m3 is multilingual, so silent cross-lingual retrieval would otherwise produce untested behaviour).

### 3.2 Retrieval
Hybrid dense + **learned sparse (bge-m3's sparse vectors — not classical BM25**; ADR-007) fused with RRF, top-40 → cross-encoder rerank (`bge-reranker-v2-m3`) → top-8. The reranker stays in the MVP because the refusal gate thresholds on its **query-comparable** scores (RRF scores are rank artifacts; ADR-006). Embeddings: `bge-m3`, local.
**Structural voices filter:** for every query not classified `voices`, chunks with `source_type: voices` are removed from retrieval results *in code* before the generation call — the voices/evidence separation is a structural invariant, not a model behaviour, and the eval verifies the filter rather than hoping the model obeys a prompt.

### 3.3 Grounded generation with inline citations (unchanged mechanism; updated prompt)

Custom-content document blocks with `citations: {enabled: true}`, one block per citable unit, so calibrated qualifiers stay attached to their claims. Guaranteed vs measured framing carried over exactly: citations provably resolve to retrieved text; **entailment is measured, not guaranteed**.

**Runtime vs eval-time (explicit, so nobody guesses):**
- **Runtime, per query:** query rewrite + scope classify (one small structured Haiku call); **citation-support validation** as a *single batched* Haiku call after generation completes, checking all sentence→cited-block pairs at once; sentences failing entailment get an "unverified" badge applied to their citation chips *after* the stream finishes (no runtime regeneration — regeneration would fight streaming and double cost; failure rates are instead driven down offline via the eval).
- **Eval-time only:** faithfulness judge, confidence-level fidelity judge, **severity-fidelity judge** (6.2), adversarial rubric. The per-request path never includes these.
- The §9 cost model includes all runtime calls.

**System prompt core rules (updated):**
1. Answer **only** from the provided passages; no outside knowledge.
2. Cite every factual claim.
3. **Preserve calibrated language verbatim** — never upgrade, downgrade, or drop a qualifier.
4. **Lead with the headline finding at the severity the source states it.** Do not restructure an alarming assessed finding into reassurance; do not add alarm the source doesn't carry. The emergency is in the sources — surface it.
5. If passages don't answer, say so plainly; point to the right source.
6. Plain language for a reader with no background; define jargon on first use; note source vintage where relevant.
7. Voices-sourced content is context about the movement, labelled as such — never evidence for a scientific claim.
8. Passages from `consensus_position: beyond-assessed-range` documents (the Hansen papers) must be presented as such: state the assessed range where retrieved alongside, and attribute above-range claims to their authors ("Hansen and colleagues argue…"), never as consensus.

**Models.** Generation default `claude-haiku-4-5`; `claude-opus-4-8` optional "best" mode behind the budget cut-off. Model id is config.

### 3.4 Native-citations constraints (unchanged)
All-or-none citations per request; incompatible with structured outputs (hence separate classifier/rewriter/chart-spec calls); generation call documents bounded to reranked top-8.

### 3.5 Refusal & uncertainty behaviour (unchanged from v2)
Reranker-thresholded honest refusal; partial support named; contested science presented with assessed ranges; footer verification note.

### 3.6 Retrieved-passages panel (unchanged from v2)
Top-8 verbatim (length-bounded per licence — Tier B excerpts always unadapted), attribution + deep link, cited-span highlighting.

### 3.7 Chart generation (NEW)

**User story:** *"Show me CO₂ and global temperature over the last 10,000 years"* → an accurate, attributed, downloadable/embeddable chart in seconds.

**Architecture — declarative spec, never code generation:**

```
chart_request
  -> [structured-output call] chart planner (claude-haiku-4-5):
        request -> ChartSpec JSON  (constrained: dataset ids from the pack,
                   series, transforms from a fixed vocabulary, chart type,
                   title, annotations, time range)
  -> spec validator (pure code: datasets exist, transforms legal, ranges sane)
  -> renderer (server-side, local, $0): spec -> Vega-Lite -> SVG + PNG
        with a baked-in caption strip: data sources, licences, access date,
        site URL
  -> response: inline chart + alt text + Download PNG/SVG + CSV of plotted
        data (attribution in header comments) + permalink (/chart/<hash>)
        [iframe embed snippet: Phase 2]
  -> if the request needs data the pack doesn't have:
        honest refusal naming the nearest available datasets
        + the gap is logged for curation (allowlisted live-fetch is Phase 2, ADR-021)
```

Design rules:
- **The LLM writes a spec, not code.** No generated Python/JS executes, ever. The spec vocabulary is closed and validated in code. This eliminates code-injection risk and makes every chart reproducible from a ~1 KB JSON.
  - **Chart types (MVP):** line, area, bar, dual-axis line, context+recent-inset panel pair (see below). Warming stripes: Phase 2.
  - **Transforms (MVP):** resample, anomaly-vs-baseline, rolling-mean, unit conversion, **and the three the flagship chart requires** (found missing in review — the vocabulary must be validated against the flagship in Phase 0):
    - `time_axis: {calendar: CE, convert_bp: true}` — years-BP (paleo convention, before 1950) → CE calendar so paleo and instrumental series share an axis;
    - `splice_series: [dataset_id, dataset_id]` — join paleo + instrumental into one series (e.g. Bereiter ice-core CO₂ + Mauna Loa; Kaufman Temp-12k + GISTEMP), with a **mandatory rendered splice-point annotation** and a resolution note ("centennial resolution before 1850; annual after") — unannotated splices are the "hidden smoothing" attack surface hostile audiences target;
    - `rebaseline_to: <reference_period>` — aligning series with different reference periods; **legal alignment periods are fixed per dataset pair in the dataset manifest**, a scientific decision made at curation time, never by the LLM.
  - **The 10-kyr axis problem, decided:** on a linear 10,000-year axis the modern spike occupies ~1.5% of chart width. Default treatment for multi-millennial ranges is the **context+recent-inset panel pair** (full range left, instrumental era right, shared y-scale) — the honest way to show both deep-time context and the modern blade.
- **Chart-integrity defaults (misinformation guardrails for graphics):** baselines and reference periods always labelled; y-axis zero-inclusion rules per chart type with any non-zero baseline visibly annotated — **enforced per axis in the renderer, including on dual-axis charts** (review finding #48: the rule must hold for every axis, not just the primary); uncertainty bands rendered when the dataset ships them; no cherry-picked default ranges — full available range unless the user asks, and **any LLM-authored axis domain (`scale_domain`) is validated in code against the plotted data's extent** (an axis window is a cherry-pick vector exactly like a time window); dual-axis charts get explicit per-axis colour-matched labels; **splice points and resolution changes always annotated** (above); **splice overlap policy explicit and disclosed** (review finding #47): where spliced datasets overlap in time, the ChartSpec carries a manifest-legal `overlap_policy` (e.g. `prefer-instrumental`, `show-both`) and the rendered annotation states what was done with the overlapping data — silently discarding overlapping samples is the residual "hidden the decline"-class attack the other annotations don't answer; **alignment periods used for rebaselining are disclosed on the artefact itself** (review finding #50), not only in repo files.
- **Attribution is part of the artefact — every artefact.** The caption strip (sources, licence, retrieval date, site URL) is rendered into exported PNG/SVG; **CSV downloads carry the same attribution as header comment lines**; the caption strip has a responsive rule (minimum render width; below it, the strip drops to source names + URL so it stays legible in a 360 px embed).
- **Alt text, generated from the spec:** every chart ships alt text derived deterministically from the ChartSpec (title, series, ranges, direction of trend) — accessibility for free, from data the renderer already has.
- **Permalinks:** `/chart/<spec-hash>` re-renders from stored spec + pinned dataset version — tiny to store, stable to embed (embed iframes: Phase 2; permalink + download are the MVP sharing surface).
- The prose around a chart (if any) goes through the normal cited-generation path; the chart itself cites its datasets via the caption strip.

**The data pack** (`datasets/manifest.yaml`, same manifest discipline as the text corpus — provenance, licence, attribution string, sha256, update script, version):

**MVP pack — six datasets** (trimmed from ten in review: the flagship's four plus the two most-asked modern series; breadth is a fast-follow, not an MVP gate):

| Dataset | Provider | Licence | Covers |
|---|---|---|---|
| GISTEMP v4 (1880–now, monthly) | NASA GISS | Public domain | Modern warming |
| HadCRUT5 (1850–now) | Met Office | OGL v3 | Modern warming (UK-sourced) |
| Mauna Loa + global mean CO₂ | NOAA GML / Scripps | Public domain | Keeling curve |
| Antarctic ice-core CO₂ composite, 800 kyr (Bereiter 2015) | NOAA NCEI Paleo | Public domain | Deep-time CO₂ |
| Temperature 12k Holocene reconstruction (Kaufman 2020) | NOAA NCEI Paleo | **Not assumed PD** — author-contributed archive data, not US-gov work; verify licence at bundling (load-bearing check) | **The 10,000-year question** |
| OWID CO₂ & GHG dataset (github.com/owid/co2-data) | OWID | CC BY 4.0 (upstream caveats per column) | Emissions by country/sector |

**Phase-2 pack additions:** global mean sea level (NOAA STAR — direct CSV, no login), Arctic sea ice extent (NSIDC G02135 v4, cite Fetterer 2025), ocean heat content (NOAA NCEI), EPICA Dome C temperature 800 kyr (Jouzel 2007). Berkeley Earth is **excluded from the pack in any phase** (CC BY-NC — see the pack invariant in 2.1).

All plain CSV/TXT. **No dataset files are ever committed to git** (client decision 2026-08-16, ADR-023): the repo carries `datasets/manifest.yaml`, fetch scripts, committed parsers, and pinned sha256 hashes; `make datasets` fetches from the origin archives and verifies hashes at build/deploy time. Provisionally-licensed datasets (`open-provisional`) are origin-fetch only and never mirrored. Monthly refresh via `make datasets` (owner: project steward); dataset versions pinned per corpus release; every dataset carries the manifest fields from 2.1 plus its legal splice/rebaseline alignment periods.

---

## 4. Faithfulness & guardrails

Carried over from v2 in full (corpus curation as first guardrail; grounding contract; calibrated-language fidelity measured by regex proxy + confidence-level LLM judge; bad-faith/denialist policy — answer calmly from evidence, cite, never echo framing; prompt-injection resistance; precise guaranteed-vs-measured citation claims; `/about` transparency page). Additions:

8. **Severity fidelity (new, symmetric with calibration).** The failure mode this project exists to fight is *underplaying*: technically-accurate answers structured to reassure. An LLM-judge eval (section 6) checks that the answer's lead reflects the severity of the cited findings — flagging both soft-pedalling (burying "rapidly closing window" under caveats) and inflation (adding alarm the source doesn't state). The two calibration checks and this one together operationalise "the emergency is in the sources".
9. **Chart integrity** (section 3.7 defaults) — misleading-graphics guardrails are enforced in the validator/renderer, not left to the model.
10. **Voices/evidence separation.** Scientific claims cite Tier A/B/C literature only; movement content is labelled. Regression-tested (a gold-set item asks a science question phrased around Packham; the answer must cite literature, not the voices layer).
11. **Non-affiliation.** Everywhere sources or the campaign appear: *"Not affiliated with or endorsed by the National Emergency Briefing campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or the IPCC. All sources cited and linked."* (Do approach the NEB campaign about a friendly listing/partnership — but build assuming none.)

---

## 5. Model-agnostic architecture (unchanged)
Thin provider adapter `generate(messages, documents, config) -> AnswerWithCitations`, Anthropic-native-citations implementation in MVP; LiteLLM/local backends deferred. Embeddings + reranker already local. The chart planner sits behind the same adapter pattern (`plan_chart(request, catalog) -> ChartSpec`).

---

## 6. Evaluation (first-class, published, in CI)

### 6.1 Eval datasets (MVP)
- **Climate-QA gold set (~95 questions)** over the MVP corpus: ~15 single-passage, ~10 multi-passage, **~40 no-answer** (refusal-expected — cheapest items to write, and the gate maths needs the n: at n=8 a single flake fails a ">90%" gate; **threshold-calibration items are disjoint from gate items**; per review findings #192/#193 every no-answer item annotates its `expected_route` — `canned_out_of_scope` items exercise the classifier's canned decline and are never counted in the reranker calibration or gate, while the release gate counts a **20-item `retrieval_refusal` gate subset** (one flake = 19/20 = 95%, so the strict ">90%" gate survives) and calibration consumes a disjoint 10-item `retrieval_refusal` subset), ~7 adversarial/denialist, **~15 severity-sensitive** (both soft-pedal-bait — "aren't we basically fine?" — and inflation-bait — "is it too late, are we doomed?" — phrasings), ~5 voices/action, plus targeted items: science-question-phrased-around-Packham (must cite literature, not voices), sensitivity/committed-warming (must retrieve assessed ranges and label Hansen as beyond-assessed-range), Carbon-Brief-derived answers (must quote verbatim or state bare facts, not close-paraphrase ND text).
- **Severity annotations:** each severity-sensitive question carries a **human-annotated expected severity of the answer's lead on a 3-point ordinal scale** (reassuring / serious / emergency-level), with the source passage the annotation derives from.
- **Chart gold set (~15 requests):** each with expected `ChartSpec` **and hand-computed expected rendered values committed as fixtures** (computed once by an independent script/human, not by the pipeline under test), or expected refusal for unavailable data. Includes the 10k-year CO₂+temperature flagship (splice + rebaseline + BP→CE + inset panel), a cherry-pick attempt ("show cooling since 2016") that must render the full-context default, and unit/baseline edge cases.

### 6.2 Metrics
All v2 metrics carry over (Recall@8/MRR/nDCG; faithfulness LLM-judge; citation-support %; calibrated-term preservation proxy; confidence-level fidelity judge; refusal >90% on the 20-item `retrieval_refusal` no-answer gate subset (#192/#193; canned out-of-scope declines are the classifier gate's, never this one's) / false-refusal <5%; adversarial rubric — **extended with a persuadable-sceptic reception check**: sceptic-facing answers are rubric-scored for "welcoming to the unconvinced", not just resistance to attack). New:

| Layer | Metric | Notes |
|---|---|---|
| **Severity fidelity** | LLM-judge scores the answer's lead on the same 3-point ordinal scale as the gold annotation (6.1); **gate: ≥90% exact-or-adjacent agreement, zero two-level errors** (an "emergency-level" question answered "reassuring" or vice versa is an automatic release blocker) | Judge ≠ generator; sampled human audit; flags soft-pedal AND inflation symmetrically |
| **Severity retrieval** | On the severity subset, Recall@8 must include the gold headline-severity chunk | The check the judge can't do: a lead faithful to its citations still misleads if retrieval buried the headline passage |
| Chart planning | Spec accuracy vs gold specs (dataset selection, transforms incl. splice/rebaseline, range) | Deterministic compare |
| Chart data faithfulness | Rendered series values match **committed gold fixtures** (6.1), tolerance 1e-9 relative for pass-through, 1e-6 post-transform | Deterministic and non-tautological (fixtures are computed independently of the pipeline under test) |
| Chart refusal | Correct refusal + nearest-dataset suggestion on unavailable-data set | Deterministic |
| Voices separation | Non-voices queries: zero `source_type: voices` chunks in the generation call's document set | Deterministic — verifies the structural filter (3.2), not model behaviour |

### 6.3 Process & publication (unchanged)
CI per PR (retrieval + smoke), full suite per release/corpus version; results in `evals/RESULTS.md`, linked from `/about`; A/B harness for pipeline changes.

---

## 7. UX

### 7.1 Landing page (redesigned)
The landing page is the front door for someone who has never had the briefing. Above the fold: the name, one line — *"The emergency briefing you haven't had. Ask anything; every answer cites the science."* — and **clickable starter topics** (tap → the question is asked immediately):

**How bad is it?**
- "Why are scientists calling this an emergency?"
- "How much has the planet warmed — and how fast is it accelerating?"
- "What happens at 1.5°C, 2°C, 3°C?"
- "Show me CO₂ and temperature over the last 10,000 years" *(chart demo — the wow moment)*

**Is it really us? / I've heard that…**
- "Hasn't the climate always changed?"
- "How sure are scientists it's human-caused?"
- "Didn't warming pause?"

**What happens next?**
- "What are tipping points and how close are we?"
- "How will this affect food, water and health?"
- "Are we on track to meet the Paris targets?" *(UNEP EGR shines here)*

**What can we do?**
- "Is it too late?"
- "What would an emergency response actually look like?"
- "Who is speaking up, and how do I get involved?" *(routes to voices layer)*

### 7.2 Chat & panels
Desktop: chat centre + retrieved-passages panel right (open by default) + source library left drawer. Mobile: expandable "Sources (n)" sheet. Streaming; citation chips highlight blocks; calibrated terms tooltip to the likelihood legend; **"About the movement" styling** on voices content; charts render inline with Download PNG/SVG · Copy embed · View data & sources.
### 7.3 Framing furniture
Footer verification note; `/about` transparency page (corpus tiers + exclusions and why — including "Ripple et al.: permission requested", guaranteed-vs-measured with live eval numbers, licence/attribution table, **non-commercial statement**); **Voices & action page** (2.5); likelihood-scale legend; non-affiliation disclaimer.
**Steward credit (client decision, 2026-08-16):** a small "Built by Rusty Data" credit with logo in the site footer (or a slim banner), paired with an explicit note that this is a **free, open-source, non-commercial project** — the credit and the non-commercial note always appear together. Code licence remains Apache-2.0 (the open-source commitment); the non-commercial commitment binds the hosted product and its Tier-B content, not the code.
Accessibility: keyboard-navigable, semantic HTML, no information by colour alone — including in generated charts (colour-blind-safe default palette, direct labelling over legend-only).

---

## 8. Tech stack (deltas from v2 in bold)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | RAG/eval ecosystem |
| Ingestion | Docling + PyMuPDF fallback | Layout-aware PDF→Markdown |
| Chunking | Hand-rolled (~300 lines) | Structure-aware; the showcase-worthy bespoke part |
| Embeddings | bge-m3 (local) | Hybrid-friendly, CPU-viable, free |
| Vector store | Qdrant (Docker or embedded) | OSS hybrid search; LanceDB fallback |
| Reranker | bge-reranker-v2-m3 | Calibrated scores for the refusal gate |
| LLM | Anthropic `claude-haiku-4-5` default, `claude-opus-4-8` gated | Native citations; prompt caching |
| **Charts** | **Vega-Lite specs rendered server-side via `vl-convert` (no browser, no JS runtime); Pandas for transforms** | **Declarative → validatable/reproducible; SVG+PNG export; $0 per chart** |
| API service | FastAPI (SSE streaming) **+ chart permalink/embed routes** | Standard, typed, async |
| UI | Streamlit (MVP); **chart permalink pages served by FastAPI** (Streamlit can't serve them; iframe embeds themselves are Phase 2) | Fast credible MVP; Next.js deferred |
| Eval | Custom scripts + pytest in CI | Citation-support, calibration, severity, chart evals are the novel bits |
| Infra | Docker Compose (`api`, `qdrant`, `ui`) | One-command reproduction |
| Observability | Structured JSON logs, no user identifiers | Each exchange a future eval case |

Deferred: Next.js UI, local-model backend, automated DOI pipeline, Langfuse, allowlisted live data fetch (ADR-021).

---

## 9. Deployment & cost

Hosting target unchanged: < ~£20/month. **Per-query cost model (all runtime calls, Haiku):** rewrite+classify (~$0.001) + generation (~5–6K in / ~500 out ≈ $0.008, less with prompt caching) + batched citation-support validation (~$0.003) ≈ **$0.012/query**; chart queries swap generation for a planner call (~$0.002) + local rendering ($0) ≈ $0.003. 1,000 queries/mo ≈ **$10–14**. Opus "best" mode ~$0.06–0.09/query, behind its sub-cap.

**Hard daily budget cut-off** retained exactly as v2 — server-side spend cap, fails closed, Opus behind a lower sub-cap, per-IP rate limits — with one mission-driven upgrade found in review: v2's "demo paused for today" dark-state predates the reframe, and the moment a chart goes viral is exactly when the budget dies. So the paused state is **read-only, not dark**: the ~14 starter-topic answers and flagship charts are pre-generated and cached at release time ($0 to serve, clearly dated), and `/about`, the source library, voices page and chart permalinks all remain up. The front door still briefs visitors when the LLM budget is gone.

### Privacy & data protection (UK GDPR — a public UK deployment that logs queries cannot skip this)
- **What we log and why:** question text, retrieved chunk ids, answer, citations — for service operation and evaluation improvement. No accounts, no cookies beyond essentials, no analytics identifiers.
- **User queries are personal data by default** (people type personal things into chat boxes). Retention: raw logs **90 days**, then deleted; queries promoted into eval sets are first **hand-reviewed and irreversibly detached** from timestamps/IP-derived data, and excluded entirely if they contain personal details. Unsafe-classified content is never harvested (3.1).
- **IPs** (rate limiting): hashed with a rotating salt, retained ≤7 days, never joined to query logs.
- **Disclosure:** one line in the chat UI ("Conversations are logged anonymously to improve the service — don't share personal details"), full privacy page linked from `/about`, with the lawful basis (legitimate interests) stated and an ICO-registration check before launch.

---

## 10. Roadmap, gates, scope guards, naming

### Phase 0 — Spike & de-risk (~1 week)
- Parse NCA5 chapter + one CC-BY paper with Docling; validate section chunking.
- Minimal retrieve → rerank → native-citations loop in a notebook.
- **Chart spike (validates the extended vocabulary from 3.7):** hand-write the ChartSpec for the flagship 10k-year CO₂+temperature chart — splice (Bereiter+Mauna Loa; Kaufman+GISTEMP), BP→CE conversion, rebaseline, context+inset panels, splice annotations; parse the paleo TXT formats; render via vl-convert with caption strip. Proves the flagship end-to-end **before** the vocabulary is frozen.
- **GATE:** 20-question probe ≥18/20 correct-source citations; flagship chart renders correctly from spec with all integrity annotations.

### Phase 1 — MVP (~6–8 weeks — re-planned upward in review; 4–5 weeks was not credible)
Corpus (Tier A + gated Tier B, hand-audited manifest, **assessed-range statements in before launch** per 2.3) → hybrid retrieval + reranker → native citations + runtime citation-support validation → refusal gate → **chart pipeline with the 6-dataset pack** → Streamlit UI with starter-topic landing page, passages panel, voices & action page, `/about`, privacy page → **~75-question + 15-chart gold sets** in CI, results published → hosted demo with budget cut-off + cached read-only paused state.
- **The real long pole** is the serial chain **corpus ingestion QA → gold-set curation**: gold chunk-ids can't be written until chunking is stable, and every gate (refusal threshold, faithfulness, severity) depends on the gold set. Docling cleanup across ~8 heterogeneous source families always overruns. Start ingestion QA on day 1; write no-answer and chart golds (which don't need chunk-ids) in parallel.
- **Cut from MVP in review:** iframe embeds (permalink + download suffice), warming-stripes chart type, four pack datasets (Phase-2 list in 3.7). **Kept despite temptation:** the reranker, runtime citation-support validation, the voices layer, the privacy page.
- **GATES:** v2 gates (faithfulness/citation-support targets; refusal >90% on the 20-item no-answer set / false-refusal <5%; cut-off fails closed to the cached read-only state; every doc signed off) **plus**: chart data-faithfulness 100% vs fixtures; severity gate ≥90% exact-or-adjacent, zero two-level errors; severity-retrieval recall on headline chunks; voices-separation structural check 100%.

### Phase 1.5 — Permissions round (parallel, calendar time not build time)
Week-1 letters: **IPCC** (naming the curated headline-statements set explicitly), **OUP/Ripple**, **WMO**, plus **NC-confirmation requests to Carbon Brief and Berkeley Earth** (2.1). Approach the NEB campaign re: listing/partnership. Legal sanity-check scope: IPCC headline-statements position (capped set, feature-flagged), Carbon Brief ND close-paraphrase risk, privacy notice.

### Phase 2+ — Deferred
IPCC full text (post-permission, using statement-ID chunking R&D); Ripple/WMO ingestion (post-permission); **allowlisted live data fetch** for chart requests outside the pack (NOAA/NASA/OWID/Met Office domains only, schema-validated, "new source — not yet curated" label, auto-logged for pack promotion); iframe embeds; warming stripes; the four deferred pack datasets; Next.js UI; local backend; DOI pipeline; source filters; multi-turn memory.

### Cadence & staleness (ongoing)
Quarterly corpus review; monthly `make datasets`; the chat footer carries the corpus vintage ("Answers reflect sources as of <corpus version date>"); voices-layer snapshot facts rendered with their `as_of` dates and reviewed on the quarterly cycle.

### Scope guards
All v2 guards carry over (no IPCC full text until permission answered; no single-lookup licensing; no "can't hallucinate" claims; reranker-thresholded refusal; custom-content blocks; no cited+structured mixing; hard cut-off; chunking-experiment guard). New:

| Temptation | Guard |
|---|---|
| "Ripple's warning papers are open access, just ingest" | Free-to-read ≠ licensed. OUP all-rights-reserved (verified). Permission letter first; link-only meanwhile. |
| "Let the model write matplotlib code" | Never. Closed ChartSpec vocabulary + code-side validation + server-side render (ADR-020). |
| "Add web search so charts can use any data" | Open web data = misinformation + injection vector. Allowlisted fetch only, Phase 2 (ADR-021). |
| "Make the tone more urgent" | Urgency comes from cited sources; severity-fidelity eval flags inflation as hard as soft-pedalling. |
| "Blend Packham quotes into answers" | Voices/evidence separation is a tested invariant. |
| "It's basically non-commercial, use it in a paid pitch demo" | `permitted_context` is load-bearing: commercial use requires dropping Tier B first (manifest makes it mechanical). |

### Naming (ADR-022)
**"Let's Talk About the Climate Emergency."** Warm, invitational, says exactly what it is, and echoes the conversations the project exists to start. Verified clear as an exact string (2026-08); adjacent brands to respect with disclaimers: ecoAmerica's *Let's Talk Climate*, Climate Outreach's *Britain Talks Climate*, and the NEB campaign itself. **Rejected:** "Climate Emergency Briefing" (reads as the NEB campaign's own product — implied affiliation), "Ask About the Climate" (v2 name; too neutral for the mission), "How Bad Is It?" (great hook, kept as landing-page section heading). Short form for UI chrome: **"Let's Talk Climate Emergency"**; repo `lets-talk-climate-emergency`; domains to check: `letstalkclimateemergency.org`, `climateemergency.chat`. Tagline: **"The emergency briefing you haven't had — answers from the science, with receipts."** Disclaimer as in section 4 (non-affiliation item).

---

## Appendix A — Key sources (verified 2026-08)
- National Emergency Briefing — https://www.nebriefing.org/ ; petition https://petition.parliament.uk/petitions/767687 ; EDM 65810; Oxford Martin account of the Westminster briefing
- IPCC Copyright — https://www.ipcc.ch/copyright/ (short-excerpt allowance; permission address)
- Ripple et al. BioScience papers — OUP standard model, all rights reserved (permissions: journals.permissions@oup.com)
- Hansen 2023 (OOCC, CC BY 4.0); Hansen 2025 (*Environment*, T&F, CC BY 4.0)
- UNEP Emissions Gap Report — UN educational/non-profit reproduction terms (printed in each edition)
- WMO copyright — https://wmo.int/copyright ; Carbon Brief — https://www.carbonbrief.org/about-us/ (CC BY-NC-ND); The Conversation republishing guidelines (AI-training prohibition)
- Met Office legal — https://www.metoffice.gov.uk/policies/legal (OGL v3); Our World in Data FAQ (CC BY 4.0)
- Datasets: GISTEMP; HadCRUT5 (OGL); NOAA GML CO₂; NCEI Paleo studies 17975 (Bereiter), Jouzel 2007, Kaufman Temp-12k; NOAA STAR sea level; NSIDC G02135 v4 (cite Fetterer 2025, DOI 10.7265/a98x-0f50); NCEI OHC; OWID co2-data
- Prior art (differentiation, no collisions): ChatClimate (ETH), ClimateQ&A (Ekimetrics), ClimateGPT, WaPo Climate Answers, DSF CliMate

## Appendix B — Guaranteed vs measured (one-liner for /about)
> **Guaranteed:** every citation points to real text we retrieved from a named, clearly-licensed source, and every chart is rendered by our own code from named public datasets — the model writes neither the numbers nor the pixels. **Measured (and published):** how often cited text actually supports each sentence, how faithfully the science's calibrated uncertainty is preserved, and whether answers convey the severity the sources state — no more, no less. We show these numbers rather than claiming perfection.
