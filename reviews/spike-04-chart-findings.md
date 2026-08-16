# Spike #4 findings — flagship 10,000-year CO₂ + temperature chart

Date: 2026-08-16 · Branch: `issue-4-flagship-chart-spike` · Author: implementer session (Fable)

## Gate verdict (DESIGN §10, Phase 0)

**PASS.** The flagship chart renders end-to-end from the hand-written
ChartSpec (`charts/spike/flagship_spec.json`) through committed parsers,
the three new transforms (`time_axis` BP→CE, `splice_series`,
`rebaseline_to`), Vega-Lite and `vl-convert` (no browser), to SVG + PNG
(`reviews/spike-04/flagship.{svg,png}`, intermediate Vega-Lite JSON
alongside) with every integrity annotation rendered:

- splice-point rules for both series (CO₂ at 1959, temperature at 1880),
  each with an on-chart label and resolution note ("multi-decadal ice-core
  resolution before 1959; annual instrumental after" / "100-year proxy bins
  before 1880; annual instrumental after");
- labelled zero baseline naming the reference period ("0 °C = 1800–1900
  average");
- Kaufman 5–95% uncertainty band rendered (dataset ships it, so it renders
  — DESIGN §3.7 default);
- colour-matched per-axis titles on the dual axis (CVD-validated pair
  #2a78d6 / #e34948, worst-pair ΔE 74.6, both ≥3:1 on the light surface);
- context + recent-inset panel pair, shared per-axis y-domains across
  panels;
- caption strip baked into both exports: sources, licence posture, access
  date 2026-08-16, site-URL placeholder, ChartSpec id + version.

Values spot-checked by hand against the raw source files (below). The
extended §3.7 vocabulary is validated *with amendments* — the concrete
schema changes required before #15 freezes it are listed at the end and
filed on issue #15.

## Sources fetched (2026-08-16; raw files in gitignored `data/spike/`)

| Dataset | URL | sha256 |
|---|---|---|
| Bereiter 2015 CO₂ composite (NCEI Paleo study 17975) | https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/antarctica2015co2composite.txt | `de57b3967758725b9650717a71bd665ebc87e23d41d3d3ad05c9095a6e7e393f` |
| Kaufman 2020 GMST percentiles (NCEI Paleo study 29712) | https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/kaufman2020/temp12k_allmethods_percentiles.csv | `b81c3ae9d9356b0629915db2c20c45eebaa1690c18a46b116d4a071e8f050b59` |
| — its NCEI readme (format + reference-period documentation) | https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/kaufman2020/readme-kaufman2020gmst.txt | `d2caf79adab2142f31d7eca60bc1d32a8e8885e6558ba6dea7b8664115b2176e` |
| NOAA GML Mauna Loa CO₂ annual means | https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv | `73d44cb1087cf77856379dc803af9003bfabf6532f9c767dcfdc61cbf082577f` |
| NASA GISTEMP v4 global Land-Ocean annual | https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts%2BdSST.csv | `6cfa44e7bbacd9b12cb10bdd64b3182c2735fa3f3a95688e1f7bc8e5dfcece93` |

Clarification (not a deviation): issue #4 names "Kaufman 2020 Temperature
12k". NCEI holds two Kaufman 2020 studies: **27330** (the Temp12k proxy
*database*) and **29712** (the GMST *reconstructions* built from it, DOI
10.25921/vzys-1280). The chart needs the reconstruction, so the spike uses
study 29712's multi-method percentiles file; its readme confirms 100-year
bins, ages in yr BP (present = 1950 CE), anomalies referenced to 1800–1900
CE (the 100 yr BP bin's median is 0 by construction). All four primary
sources were reachable — no mirrors or substitutions needed.

## Kaufman 2020 licence verification (load-bearing check, issue #4)

Checked 2026-08-16. Verdict: **NOT public domain; no explicit licence grant
exists at the archive.** Evidence:

1. **NCEI landing-page metadata** (https://www.ncei.noaa.gov/access/paleo-search/study/29712,
   a JS app; machine-readable equivalent
   `https://www.ncei.noaa.gov/access/paleo-search/study/search.json?NOAAStudyId=29712`):
   the study record's `dataLicenseDescription` and `dataLicenseUrl` fields
   are both **null**, and the only use constraint is: *"Please cite
   original publication, online resource, dataset and publication DOIs
   (where available), and date accessed when using downloaded data. …"* —
   a citation request, not a licence.
2. **DataCite record** for dataset DOI 10.25921/vzys-1280: `rightsList` is
   empty — no rights declared.
3. The **describing paper** (Kaufman et al. 2020, *Sci Data* 7:201, DOI
   10.1038/s41597-020-0530-7) is **CC BY 4.0** (Crossref licence records,
   vor + tdm). That covers the article; it is at best an argument, not a
   grant, for the archived data files.
4. The data are author-contributed (NSF/SNSF-funded academic work), so the
   US-government public-domain rationale that covers Bereiter/GML/GISTEMP
   does **not** apply.

Consequence: `datasets/manifest.yaml` records the dataset as
`permitted_context: open-provisional`. Charting with attribution is
well-supported (NOAA archives exist to be used, the constraint asks only
for citation, the paper is CC BY), but the DESIGN §2.1 pack invariant
demands `open` with confidence — **before #16 bundles the pack, send a
written confirmation request to the authors (D. Kaufman / N. McKay) or
NCEI**, per the project's permission-letters practice. If confirmation is
refused or unanswered, the fallback is to drop the Kaufman series from
*exported* charts or replace it with an explicitly-licensed reconstruction;
the chart pipeline itself is unaffected.

## Alignment-period decision (fixed in `datasets/manifest.yaml`)

Pair `temp_10k` (kaufman2020_temp12k + gistemp_v4): **rebaseline GISTEMP by
its 1880–1900 mean; splice at 1880; displayed reference period 1800–1900 CE
(Kaufman's native reference).**

Why: Kaufman anomalies are natively zeroed on 1800–1900 CE (readme: the
100 yr BP bin is 0 by construction); GISTEMP is natively vs 1951–1980.
**1880–1900 is the maximal overlap between the GISTEMP record (starts
1880) and Kaufman's native reference century**, so shifting GISTEMP so its
1880–1900 mean is zero puts both series on (an estimate of) the same
1800–1900 zero without touching the published reconstruction. The residual
bias from representing the 1800–1900 century by its last two decades is
order 0.01–0.03 °C in instrumental-era analyses — an order of magnitude
below the reconstruction's ~0.7 °C 5–95% band width. Alternatives
rejected: 1850–1900 (IPCC pre-industrial) is not realisable from GISTEMP;
aligning on Kaufman's 0 BP bin (1900–2000 CE) would zero the display on a
warming century and understate modern warming — exactly the cherry-pick
the guardrails exist to prevent. Pair `co2_10k` (bereiter2015_co2 +
noaa_gml_co2_mlo): splice at 1959 (first complete Mauna Loa year);
**no rebaseline is legal** — CO₂ is an absolute concentration on a common
WMO scale.

The splice years and the alignment period live in the manifest, and the
spike renderer refuses a spec that disagrees with them
(`check_spec_against_manifest`) — the ADR-020 "never chosen by the LLM"
invariant, exercised end-to-end.

## Hand spot-checks (rendered values vs raw source lines)

Computed by hand from the raw files, then verified in the plotted frames
and visually in `flagship.png`:

1. **BP→CE endpoints:** Kaufman 12000 BP → −10050 CE (outside the panel;
   correctly clipped); 10000 BP → −8050 CE = context-panel start; 100 BP →
   1850 CE. Bereiter youngest sample −51.03 BP → 2001.03 CE. ✓
2. **CO₂ splice boundary:** last pre-splice Bereiter sample: raw line
   `-8.56  316.33` → 1958.56 CE, 316.33 ppm; first instrumental point: raw
   line `1959,315.98,0.12`. Both present, correctly segmented, in the
   plotted frame; the rendered line is visually continuous across the 1959
   rule (0.35 ppm step). ✓
3. **Rebaseline arithmetic:** GISTEMP 1880–1900 J-D mean vs 1951–1980 =
   **−0.218571 °C** (mean of the 21 raw J-D values). Rebaselined 1880 =
   −0.17 + 0.218571 = **+0.048571**; 2024 = 1.28 + 0.218571 =
   **+1.498571** — both match the plotted frame exactly, and the
   post-rebaseline 1880–1900 mean is 0 (−2e-17). PNG shows the red peak at
   ≈1.5 °C. ✓
4. **Kaufman native reference:** raw 100 BP row's global_median is exactly
   0.0 → plots at (1850 CE, 0.0), on the labelled baseline. ✓
5. **Holocene shape:** Kaufman median maximum is +0.5428 °C at 6600 BP
   (−4650 CE) — the PNG red curve peaks at ≈0.54 °C around 4650 BCE. ✓
6. **Modern endpoints:** MLO 2025 = 427.35 ppm; GISTEMP 2025 = +1.19 °C
   (partial 2026 row `***` correctly dropped by the parser). PNG blue line
   tops out just below 430 ppm. ✓

These values seed #20's independent gold fixtures and #17's transform
tests (issues/04.md TDD plan); the same arithmetic is pinned by
characterisation tests in `tests/unit/test_spike_transforms.py`.

## Vocabulary gaps → required changes to #15 (also filed on issue #15)

1. **`splice_series` must be manifest-keyed, not free dataset pairs.** The
   spec needs `splice_pair_id` referencing a manifest `splice_pairs` entry
   that owns the datasets, the splice year, the resolution note and the
   rebaseline legality. The validator resolves and cross-checks these
   (reject on any mismatch); #15's `test_splice_pair_not_in_manifest_rejected`
   should key on the pair id.
2. **`rebaseline_to: <reference_period>` (scalar) is under-specified.** It
   must say *which* series member is shifted: `{apply_to: <dataset_id>,
   alignment_period_ce: [a, b]}`. Shifting the wrong member (Kaufman
   instead of GISTEMP) silently moves a published reconstruction off its
   native reference.
3. **Alignment period ≠ displayed reference period.** The pair aligns on
   1880–1900 but the axis/baseline honestly reads "vs 1800–1900" (Kaufman's
   native reference). The manifest needs both fields
   (`alignment_period_ce`, `display_reference`) and the renderer must label
   from `display_reference` — labelling from the alignment period would be
   wrong.
4. **Chart types compose; a flat enum can't express the flagship.** The
   flagship is a context+recent-inset panel *pair of dual-axis line
   charts*. Model panel-pair as a layout wrapper (with per-panel
   `time_range_ce` and label) over series that carry `axis: left|right`,
   rather than five mutually-exclusive chart types.
5. **"Shared y-scale" across panels means shared *per-axis* domains.** With
   a dual axis there are two y-scales to share. The spec needs explicit
   `scale_domain` per series applied identically in both panels; "shared:
   true" alone is unimplementable.
6. **Uncertainty bands attach to a splice *member*, not the series.** Only
   Kaufman ships a 5–95% band; GISTEMP's segment has none. Vocabulary:
   `uncertainty_band: {lower, upper, source: <dataset_id>}`.
7. **Annotations need per-panel presence rules.** Splice rules render in
   every panel they intersect; the text labels/resolution notes render once
   (context panel) or the inset drowns. The schema should encode
   rule-everywhere/label-once rather than leaving it to the renderer's
   discretion.
8. **BP→CE implies BCE-aware axis labelling.** Negative CE years must
   format as "8000 BCE" etc. (Vega-Lite `labelExpr`); also suppress or
   special-case a year-0 tick (no year 0 exists). This belongs to
   `time_axis` semantics, not per-spec styling.
9. **Caption licences must come from the manifest.** The spike hand-wrote
   "public data, cite source" in the caption; production must generate the
   caption strip from manifest attribution/licence strings so the caption
   can never drift from the licensing record (and `open-provisional`
   datasets can render a distinguishable posture).
10. **Renderer note (dual-axis integrity, for #17/#19):** in Vega-Lite,
    layered dual-axis charts silently drop an axis if any layer in the
    shared-scale group sets `axis: null` — the spike hit this (missing
    right axis). The renderer must set the identical axis object on every
    layer of a scale group, and the #17 SVG-structural tests should assert
    both axis titles are present (a one-line regression that catches a
    silent integrity failure).

Also noted for the record: general dataviz practice (and this repo's chart
skill guidance) disfavours dual-axis charts outright; DESIGN §3.7
deliberately sanctions them with colour-matched per-axis labels as the
mitigation, and the flagship exercises that. If #15 review wants a stricter
posture, the honest alternative is a stacked pair of single-axis panels per
variable — the vocabulary of gap 4 can express either.

## Deviations

- None on data sources (all four primary URLs reachable; hashes above).
- Study-id clarification for Kaufman (27330 database vs 29712
  reconstruction) documented above.
- Kaufman licence: treated as open-with-attribution **provisionally**;
  written confirmation is a pre-#16 action (above).
