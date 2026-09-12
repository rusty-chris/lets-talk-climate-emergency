# Carbon & energy footprint indicator — estimation methodology

Status: **proposed, for owner review** (methodology only — no implementation in this PR).
Scope: the per-exchange footprint estimate, the session cumulative, the application
total, and the `/footprint` transparency page.

This product's register is calibrated honesty (the /about guaranteed-vs-measured
contract). The same contract governs this feature: **every figure is labelled
measured, estimated, or unknown; estimates are always ranges; nothing is presented
with more precision than its provenance supports.** The single biggest input —
the energy Anthropic's serving stack spends per token — is not published by
Anthropic, so the headline figure is and must remain an order-of-magnitude
estimate. We say so, everywhere it appears.

---

## 1. What we can already measure (and what we cannot)

| Input | Status | Where it comes from |
|---|---|---|
| Tokens per exchange (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`), for **every** runtime call (classifier, generation, planner, validation) | **Measured** (provider-reported) | The adapter `usage` mappings already on the wire: `rag/generation.py` streams one `usage` SSE event per generation; `service/app.py` aggregates every call's usage into the exchange log's `usage_records` (the same records the §9 spend cap charges). Zero new instrumentation needed. |
| Local embedding + rerank CPU time per exchange (bge-m3 + bge-reranker-v2-m3 on the Hetzner CX32) | **Measurable** — process CPU-seconds per request (`resource.getrusage` / `time.process_time` delta around the retrieval call), to be measured at deploy | `rag/indexing.py` / retrieval path; runs in the api process on the CX32's 4 shared vCPUs |
| Energy per token inside Anthropic's serving stack | **Unknown** (no Anthropic disclosure as of 2026-09) → estimated range from published literature | §3 |
| Anthropic serving region / grid carbon intensity | **Unknown** → disclosed assumption (US average) | §4 |
| Hetzner server electricity | Supplier-claimed renewable (market-based); location-based German/Finnish grid factors published | §4 |

The design consequence: **token counts and local CPU-seconds are facts; everything
that converts them to Wh and gCO2e is a sourced, ranged conversion factor.** The
`/footprint` page separates the two columns exactly as /about separates
guaranteed from measured.

## 2. The estimation model

Per exchange, summed over all `usage_records` in the exchange (classifier +
generation + validation, or classifier + planner for chart queries):

```
E_api (Wh) = Σ [ input_tokens/1000 × E_IN
              + cache_creation_input_tokens/1000 × E_IN
              + cache_read_input_tokens/1000 × E_IN × CACHE_READ_FACTOR
              + output_tokens/1000 × E_OUT ]

E_local (Wh) = cpu_seconds_measured × W_VCPU × PUE_HETZNER / 3600

CO2e (g)  = E_api × CIF_API + E_local × CIF_LOCAL
```

Low and high bounds are computed with the low/high ends of each factor, so the
displayed range is an honest propagation, not a point estimate with decoration.
Session cumulative = sum of exchange ranges; application total = the same sum
over the exchange log (plus the flat, disclosed exclusions in §7).

## 3. Recommended constants (the table for owner review)

All energy factors are **facility energy** (PUE included — the ranges absorb a
PUE of 1.1–1.4, bracketing Google's reported overheads and Jegham et al.'s
AWS PUE 1.14 assumption). "Haiku-class" = `claude-haiku-4-5`, the default
generation/classifier/validation model (DESIGN §3.3/§9). Opus/Sonnet best-mode
exchanges scale per §3.4.

| Constant | Central | Range | Status | Provenance (primary) |
|---|---|---|---|---|
| `E_OUT` — Wh per 1k **output** tokens, Haiku-class | 0.5 | **0.1 – 1.5** | Estimated | Bottom: ML.ENERGY v3.0 measured well-batched open models of plausibly comparable scale (e.g. Qwen3-32B chat ≈ 95 J ≈ 0.026 Wh/response on B200; ~0.1 Wh/1k out with overheads). Middle: Google's measured production median (0.24 Wh/prompt, Gemini Apps, Aug 2025); Epoch AI's GPT-4o estimate (~0.3 Wh/~500-token query ≈ 0.6 Wh/1k out). Top: Jegham et al. 2025 (arXiv:2505.09598) Claude 3.7 Sonnet 0.95–5.67 Wh/query, scaled ×0.5 for a Haiku-class model (the claude-carbon derivation) ≈ 1.4 Wh/1k out. |
| `E_IN` — Wh per 1k **input** tokens (incl. cache writes) | 0.02 | **0.005 – 0.07** | Estimated | Prefill is far cheaper per token than decode; Jegham et al.-derived marginal output:input energy ratio ≈ 21:1 (OLS fit over the three measured Sonnet prompt lengths, per gwittebolle/claude-carbon `factors.json` derivation, cross-checked against EcoLogits). |
| `CACHE_READ_FACTOR` — cache-read token energy as a fraction of an uncached input token | 0.08 | **0.05 – 0.20** | Estimated | Prefill-residual engineering estimate (claude-carbon methodology; cf. Irminsul, arXiv:2605.05696, measuring 14–37% at short prefixes). Explicitly **not** Anthropic's 0.1× billing ratio (a price, not energy). Matters here: the §9 prompt-caching design makes cache reads a large share of our input tokens. |
| `CIF_API` — grid carbon intensity for Anthropic serving, gCO2e/kWh | 384 | **287 – 450** | Assumption (region unknown) | US grid average 384 gCO2/kWh (Ember, 2024 — record low). Low end: AWS-region-weighted 287 (Jegham et al.); high end: pessimism margin for fossil-heavier regions. Location-based; provider market-based renewable claims deliberately not netted off (§4). |
| `W_VCPU` — watts per shared cloud vCPU at typical utilisation | 2.2 | **0.7 – 3.8** | Estimated | Cloud Carbon Footprint methodology: `min + util × (max − min)` with min ≈ 0.71–0.78 W, max ≈ 3.5–4.26 W per vCPU across AWS/GCP/Azure coefficient sets; 2.2 W ≈ 50% utilisation mid-point. |
| `PUE_HETZNER` | 1.13 | 1.1 – 1.2 | Supplier-published | Hetzner: "average PUE value of 1.13" (docs.hetzner.com sustainability pages). |
| `CIF_LOCAL` — location-based, gCO2e/kWh | 363 (DE) / ~95 (FI) | — | Published statistic | German grid 2024 ≈ 363 gCO2/kWh (UBA/Ember); Finnish grid ≈ 79–95 gCO2/kWh (2023–24). Pick the factor for the DC actually chosen (fsn1/nbg1 → DE, hel1 → FI). Market-based treatment in §4. |

**Typical exchange (DESIGN §9: ~5–6k in / ~500 out generation, plus classifier
and batched validation → ≈7k input-class + ≈0.7k output tokens):**

- `E_api` ≈ **0.1 – 1.5 Wh** (central ≈ 0.5 Wh)
- CO2e ≈ **0.04 – 0.6 g** (central ≈ 0.2 g)
- `E_local` (expected order, to be measured at deploy): a few CPU-seconds
  → ≈ 0.002 – 0.01 Wh — roughly 1% of the total, but the **measured** 1%.

### 3.4 Non-default models

Opus best-mode: ×3–4 on both token factors (parameter-count and price ratios
both imply 2–5×; Jegham et al. give no Opus measurement — labelled
"extrapolated" on the page). Sonnet bake-off arm: ×2 (closest to the actually
measured Jegham model). These multipliers are the weakest numbers in the table
and are flagged as such; best-mode is behind a sub-cap and rare by design.

## 4. Grid carbon intensity — the two supply chains, treated honestly

**Anthropic serving (the API calls).** Anthropic has published no per-query
energy, no serving-region breakdown, and (as of 2026-09) no audited scope 1–3
emissions. Standard practice in the estimation literature (Epoch AI, Jegham et
al.) is a US-average or provider-region-weighted grid factor with the
assumption disclosed. We use the **US average, location-based** (384 gCO2/kWh,
Ember 2024) with the 287–450 range, and state on the page: *"We do not know
where Anthropic serves our requests; we assume the US average grid. Cloud
providers' market-based renewable claims would reduce this on paper; we report
the location-based figure."*

**Our Hetzner server.** Hetzner states its German data centres obtain "100
percent of our electricity from renewable energy sources" (hydropower in
Germany; hydro + wind in Finland, per hetzner.com sustainability pages), holds
EMAS certification for the German sites, and publishes the 1.13 average PUE.
This is a **market-based** claim (a certified green-supply tariff — the public
pages name EMAS but do not itemise the guarantee-of-origin scheme backing the
tariff). The honest one-sentence treatment, used verbatim on the page:
*"Market-based accounting (counting our host's certified renewable purchases)
puts our own server near zero gCO2e; location-based accounting (the average
grid where the server physically draws power) puts it at ~363 gCO2e/kWh in
Germany — we show the location-based figure and note the renewable claim."*
Showing the higher figure is deliberate: this product does not get to buy its
way out of a range with someone else's certificate.

## 5. Local compute share — the measured slice

The api process runs bge-m3 embedding + bge-reranker-v2-m3 cross-encoding per
query on the CX32's 4 shared vCPUs. Recommended instrumentation (at deploy, not
in this PR): wrap the retrieval+rerank call with a process-CPU-time delta
(`resource.getrusage(RUSAGE_SELF)` before/after, or `time.process_time()`)
and store `cpu_seconds` on the exchange record alongside `usage_records`.
Conversion: `cpu_seconds × 2.2 W/vCPU-equivalent (0.7–3.8) × PUE 1.13`.

CPU-seconds are a genuine per-request **measurement**; only the
watts-per-vCPU conversion is estimated. The page therefore labels this slice
*"measured CPU time × estimated per-vCPU wattage"* — the only component of the
whole figure with a measured energy basis, and it is disclosed as such (and as
small: ~1% of the central estimate).

### CodeCarbon assessment (owner addendum)

[CodeCarbon](https://github.com/mlco2/codecarbon) (MIT) was evaluated as the
off-the-shelf alternative for this slice. On a shared-vCPU cloud instance with
no RAPL access — exactly our CX32 — CodeCarbon **falls back to constant mode**:
it looks up the host CPU's TDP in its database (or a global 85 W constant) and
assumes 50% of TDP as average draw. That is (a) an *estimate*, not a
measurement — it would not make our local share "measured"; (b) mis-scoped for
a 4-vCPU slice of a larger shared physical CPU (half of a full-socket TDP can
overstate our slice several-fold); and (c) a non-trivial dependency (pandas,
background scheduler thread) for a figure our two-line `getrusage` delta
captures more honestly. **Recommendation: cite-only.** We reuse its published
methodology as a reference for the fallback-wattage problem, and its
grid-intensity data sources conceptually, but do not add the dependency. Our
own approach yields a strictly more honest label (measured CPU-seconds ×
sourced wattage estimate) at zero dependency cost.

## 6. Off-the-shelf components — adopt / adapt / cite-only (owner addendum)

| Component | Licence | What it offers | Verdict |
|---|---|---|---|
| [gwittebolle/claude-carbon](https://github.com/gwittebolle/claude-carbon) | MIT | Claude Code plugin; `data/factors.json` with unusually good provenance notes: Sonnet factors from a 3-point OLS fit to Jegham et al.'s measured Claude 3.7 Sonnet per-query energies (0.95/2.99/5.67 Wh), ~21:1 output:input marginal ratio, Haiku = 0.5× Sonnet (extrapolated, flagged), cache-read factor 0.08 (range 0.05–0.20) with the price-vs-energy distinction spelled out; golden methodology test vectors | **Adapt.** MIT permits adopting the factor *derivation* with attribution. But adopt the derivation and its primary sources, not the JSON verbatim: its factors bake in AWS-region CIF 0.287 and PUE 1.14, which our model parameterises separately (we keep energy and carbon factors apart so the grid assumption stays visible). Its golden-test-vector pattern is worth copying outright. |
| [HelmutZechmann/claude-carbon-py](https://github.com/HelmutZechmann/claude-carbon-py) | MIT | Python port; same Jegham citation but a **~5× different Haiku factor set** (95/570 vs 20/413 gCO2e/Mtok) with thinner derivation notes | **Cite-only** — and cite it *as the cautionary example*: two MIT projects citing the same paper disagree 5× on Haiku, because Haiku was never measured. This is precisely why our page publishes a range with the extrapolation flagged, not a borrowed point value. |
| [metztim/claude-carbon](https://github.com/metztim/claude-carbon) | MIT | macOS menu-bar tracker; J/token values partly *inferred from pricing ratios* (self-disclosed); editable `Methodology.json`; household-equivalents UX | **Cite-only.** Price-ratio-inferred energy factors are below our provenance bar (price ≠ energy — the same trap the cache-read note warns about). Its equivalents-presentation UX informed §8. |
| [Carbonlog](https://github.com/CNaught-Inc/claude-code-plugins/tree/main/plugins/carbonlog) (CNaught) | **No licence found** in the plugin README — treat as all-rights-reserved until checked | Latency-based model: `E = (TTFT + out_tokens/TPS) × (GPU+host power × util) × PUE` with live Artificial Analysis throughput data and statistical hardware inference | **Cite-only** (licence unresolved, and the latency-based approach needs live TPS telemetry we don't collect server-side). The idea of provider-specific rather than country-average CIF is noted for a future revision if Anthropic ever discloses regions. |
| [CodeCarbon](https://github.com/mlco2/codecarbon) | MIT | Local process energy tracking | **Cite-only** — see §5: constant-mode fallback on our VPS is an estimate dressed as a measurement. |
| All of the above target Claude **Code session tracking** (transcript parsing, statusline). None is a drop-in for server-side per-request estimation over our SSE `usage_records` — component reuse only, as briefed. | | | |

## 7. What the totals include and exclude (disclosed on the page)

Included: every runtime adapter call's tokens (the same records the spend cap
charges — including refused/declined exchanges and validation calls) + local
retrieval compute. Excluded, listed verbatim on the page: model **training**
(amortised training energy is unknowable for a closed model — Mistral's LCA
shows it is material; we say "not included, not zero"); embedding/index
**build-time** compute (one-off, small, could be added as a constant later);
network transfer and the **visitor's own device** (the IEA streaming factcheck
shows end-user devices dominate streaming footprints — for a text chat both
are tiny, but they are listed as excluded, not silently dropped); Anthropic's
water use (no data — listed as unknown); serving-hardware embodied carbon.

## 8. Everyday-equivalent anchors (each with a citable source)

For the central estimate ≈ 0.2 gCO2e / 0.5 Wh per exchange:

1. **Seconds of video streaming:** one hour of streaming ≈ 36 gCO2e (IEA,
   Kamiya 2020 fact-check, global average, viewing device included; the
   figure that corrected the popular estimates ~90× too high). One exchange
   ≈ **20–30 seconds of streaming** (range 4–60 s).
2. **Metres driven:** a typical passenger car ≈ 400 gCO2/mile ≈ 0.25 g/metre
   (US EPA Greenhouse Gas Equivalencies Calculator). One exchange ≈ **driving
   about one metre** (range 0.2–2.5 m).
3. **Cups of tea:** heating 250 ml of water from 15 °C to boil is 0.025 kWh of
   physics (4.186 J/g·K), ≈ 0.031 kWh at a realistic ~80% kettle efficiency.
   **About 60 exchanges ≈ one mug of tea** (range 20–300). Sourced from first
   principles (specific heat of water) + the stated efficiency assumption —
   deliberately, because the consumer-site kettle figures we found were not
   citable to our bar.

Rejected: "trees absorbing CO2" (sequestration rates vary an order of
magnitude; no single defensible factor); smartphone charges (EPA's factor
mixes marginal-grid assumptions we could not cleanly reconcile with our
location-based framing); water-drop equivalents (we have no water data for
Anthropic serving — it stays in the "unknown" column, not in a cute anchor).

## 9. The honest presentation contract

### Footer indicator (exact wording, template)

> **Footprint: est. {lo}–{hi} Wh this answer · est. {session_lo}–{session_hi} Wh this session — estimates, not measurements · how we know → /footprint**

Rules (the register contract, enforceable by tests like the ADR-018 pair):
always a **range**, never a point; the word **"est."/"estimates"** always
present and never further from the figures than the range itself; the
`/footprint` link always present; **no gCO2e in the footer** (the carbon figure
adds the grid assumption on top of the energy range — it belongs on the page
where the assumption is disclosed beside it, not in a chip that implies one
more measured digit). Wh, not J or kWh: the numbers land in a human range
(0.1–10) without scientific notation.

### /footprint page structure (mirrors `service/transparency.py` conventions:
constants live in one module, the page interpolates them at render time,
figures can never drift from the code that computes them)

1. **Headline totals** — application lifetime: "est. X–Y kWh · est. A–B kg CO2e
   since {launch date}, over N answers", each labelled *estimated*; beside
   them, the one measured line: "of which our own server's retrieval compute:
   {measured CPU-hours} (measured) ≈ est. C–D kWh".
2. **Measured / estimated / unknown** — the three-column honesty table from §1,
   the page's centrepiece (the guaranteed-vs-measured pattern, extended).
3. **How the estimate is built** — the §2 formula in prose + the §3 constants
   table verbatim, ranges and provenance included.
4. **The biggest uncertainty, stated plainly:** *"Anthropic does not publish
   the energy its models use. Our per-token factor spans 15×; the true figure
   is somewhere in that range, and we will not pretend to know where. If
   Anthropic publishes measurements, we will replace this section with them."*
5. **The two grids** — §4's assumption + the market-vs-location sentence.
6. **Equivalents** — §8's three anchors, each with its source inline.
7. **What is not counted** — §7's exclusion list.
8. **Revision note** — constants are versioned in the repo; the page shows the
   factors-file version/date, so a factor update is a visible, dated event.

## 10. Primary sources

- Google (2025), *Measuring the environmental impact of delivering AI at
  Google scale* — arXiv:2508.15734 (median Gemini Apps text prompt: 0.24 Wh,
  0.03 gCO2e; production-measured, full serving stack).
- Epoch AI (2025), *How much energy does ChatGPT use?* —
  epoch.ai/gradient-updates (≈0.3 Wh per GPT-4o query; estimation
  methodology and the correction of the older 3 Wh figure).
- Jegham et al. (2025), *How Hungry is AI? Benchmarking Energy, Water, and
  Carbon Footprint of LLM Inference* — arXiv:2505.09598 (infrastructure-aware
  API-side estimates; the only per-model Claude figures in the literature;
  Claude 3.7 Sonnet 0.95–5.67 Wh/query by prompt length).
- ML.ENERGY Leaderboard v3.0 (2025) — ml.energy (direct GPU measurements,
  open models, H100/B200; demonstrates batching/configuration swings of 3–5×
  and task-mix swings of ~25×: the reason our factor is a range).
- Luccioni, Viguier & Ligozat (2022), *Estimating the carbon footprint of
  BLOOM* — arXiv:2211.02001; and Luccioni et al. (2024), *Power Hungry
  Processing* — the founding per-inference estimation literature.
- Mistral AI (July 2025), lifecycle analysis of Mistral Large 2 (with
  Carbone 4 / ADEME): 1.14 gCO2e + 45 ml water per 400-token Le Chat
  response — the closest thing to a full provider LCA; also the precedent
  for disclosing marginal per-response impact without energy figures.
- Ember (2025), *US Electricity 2025* — US grid 384 gCO2/kWh (2024);
  German grid ≈ 363 gCO2/kWh (2024, UBA/Ember); Finland ≈ 79–95 gCO2/kWh.
- Hetzner — docs.hetzner.com sustainability pages ("In Germany, we obtain 100
  percent of our electricity from renewable energy sources"; average PUE
  1.13; EMAS certification, German sites; hydropower DE / hydro+wind FI per
  hetzner.com sustainability page).
- Cloud Carbon Footprint — cloudcarbonfootprint.org/docs/methodology
  (per-vCPU min/max wattage coefficients; `avg = min + util×(max−min)`).
- IEA, Kamiya (2020), *The carbon footprint of streaming video: fact-checking
  the headlines*; Carbon Brief fact-check version (≈36 gCO2e/hour, device
  included).
- US EPA, *Greenhouse Gas Equivalencies Calculator — Calculations and
  References* (typical passenger vehicle ≈ 400 g CO2/mile).
- CodeCarbon — docs.codecarbon.io methodology (RAPL vs constant-mode
  fallback: TDP lookup, 50% load assumption, 85 W default).
- gwittebolle/claude-carbon `data/factors.json` (MIT) — the Sonnet OLS
  derivation and cache-read-factor reasoning adapted here with attribution.
- Anthropic: no per-query energy, serving-region, or scope 1–3 disclosure
  found as of 2026-09 (checked; stated as an absence, not assumed).
