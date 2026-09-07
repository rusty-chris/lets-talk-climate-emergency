# Tier-A corpus expansion — OWNER SIGN-OFF PACKET

Prepared 2026-09-07 by the corpus-expansion agent session (Claude Fable 5).
Companion file: `corpus/manifest-expansion-PROPOSED.yaml` — 30 proposed,
schema-valid, pinned entries.

## OWNER DECISION — 2026-09-07 (Chris McWilliams, Rusty Data — author & owner)

Two rulings recorded this day and executed by the corpus-activation session:

1. **Sign all 30.** Counter-sign and ACTIVATE every one of the 30 proposed
   Tier-A entries below — including `nasa_cc_evidence` with its recorded
   third-party image credit (Ashwin Kumar, CC BY-SA 2.0 Generic; an
   openly-licensed image, not part of the ingested text). All 30 checkboxes
   are ticked (2026-09-07) and the entries are now in
   `corpus/manifest.yaml`'s `documents:` list, each `human_signoff`
   carrying the owner counter-signature with the agent verification record
   kept inline (the nca5_ch2 convention).
2. **c3s_esotc → Tier C.** Move c3s_esotc (§7 licensing trap) to the Tier C
   skeleton in `corpus/manifest.yaml` and request written permission:
   `letters/07-ecmwf-esotc.md` (to copernicus-press@ecmwf.int, cc
   publications@wmo.int), added to the Phase-1.5 permission-letter batch.

**This packet is now ACTIVATED.** The section below is the historical
sign-off record; the boxes are ticked as executed.

## How to activate

1. Tick an entry's checkbox below (or strike it out with a reason).
2. Move the ticked entries from `manifest-expansion-PROPOSED.yaml` into
   `corpus/manifest.yaml`'s `documents:` list, replacing/extending each
   `human_signoff` with your counter-signature (the nca5_ch2 convention:
   keep the agent verification record inside your note).
3. Run `make corpus` (all pins re-verify from origin through the production
   transport) and `make ingest`, then the `corpus/hand-audit-checklist.md`
   pass over the new chunks **before** any index build ships.

## What was verified, and how

- **Licence evidence is primary and verbatim** — quoted from the operative
  policy page or from the artefact itself (both Hansen PDFs carry their CC BY
  statements in-document), with URL + access date in every entry.
- **Every pin is real bytes**: each artefact fetched twice on 2026-09-07 and
  the sha256 verified identical across both fetches; then the entire merged
  manifest (2 active + 30 proposed) was run through `make corpus` semantics
  against a scratch directory — full re-fetch through the production
  transport, every pin verified, every §2.1 invariant green, exit 0. The live
  `corpus/manifest.yaml` was never modified.
- **Per-page third-party scan**: every HTML page scanned for third-party
  copyright credits (Getty/AP/©/credit lines). Findings per source below.
- **Parse sanity**: 3 representative pages (NOAA Q&A, Met Office questions,
  NASA evidence) run through the production `parse_document` → `chunk` path.
  All parse with Docling (no degraded fallback) into section-pathed chunks
  with the substantive content intact. Honest finding: HTML nav/footer
  boilerplate also becomes chunks (breadcrumbs, menus, "Follow NASA",
  related-content teasers) — roughly a third to a half of chunks per page.
  The hand-audit pass must mark these, and a boilerplate-filtering step for
  HTML sources is worth its own issue before these pages are indexed.

## #314 gap coverage map

| Gap (issue #314) | Covered by |
|---|---|
| qa-adv-05 — 1970s-cooling myth | **metoffice_questions** (a direct "Weren't there warnings of global cooling years ago?" Q&A — the single best carrier), **noaa_qa_gw_vs_cc** (narrates the 1970s cooling-vs-warming literature history), **nasa_no_mini_ice_age** + **nasa_faq_scientists_agree** + **noaa_qa_disagreement** (trust/consensus framing the question actually attacks) |
| qa-va-03 — "what can one person do" action content | **noaa_qa_slow_reverse**, **nasa_cc_mitigation**, **nasa_faq_too_late**, **owid_food_local**, **owid_travel_footprint**, **owid_safest_energy** (individual-action content with numbers) |
| Conversational register (voices_action decline cluster, resmoke REPORT) | The two FAQ/Q&A families are conversational by construction: 2 NASA FAQ pages, 6 NOAA Climate Q&A pages, metoffice_questions |
| qa-tg-02 — Hansen labelling half | **hansen_2023_pipeline** + **hansen_2025_acceleration**, both carrying `consensus_position: beyond-assessed-range` as DESIGN §2.1 requires |
| qa-tg-03 — Carbon Brief paraphrase check | **NOT covered** — carbon_brief_verbatim_set is Tier B (CC BY-NC-ND) and out of scope for this Tier-A expansion; qa-tg-03 stays `blocked_on: corpus-expansion`. noaa/metoffice effects pages give partial event-attribution background only. |

---

## 1. nasa_climate_explainers — 10 entries — VERDICT: open (not subject to copyright), one flag

**Licence evidence** (https://www.nasa.gov/nasa-brand-center/images-and-media/,
accessed 2026-09-07), verbatim:

> "NASA content – images, audio, video, and media files used in the rendition
> of 3-dimensional models, such as texture maps and polygon data in any format
> – generally are not subject to copyright in the United States. You may use
> this material for educational or informational purposes, including photo
> collections, textbooks, public exhibits, computer graphical simulations and
> Internet Web pages."

Third-party rule, same page: "NASA occasionally uses copyright-protected
material of third parties with permission on its website. Those images will be
marked identified as copyright protected with the name of the copyright
holder."

**What it adds**: the core evidence/causes/effects/consensus explainers, two
conversational FAQ pages, action content (Mitigation and Adaptation; "Is it
too late?"), extreme weather, and a modern ice-age-myth debunk.

**Flag — one third-party image credit**: the Evidence page contains "Image
credit: Ashwin Kumar, Creative Commons Attribution-Share Alike 2.0 Generic".
It is an image (not ingested text) and itself openly licensed; recorded in the
entry rather than silently passed. If you prefer the strict reading of
"exclude items with third-party credits", strike nasa_cc_evidence and the
remaining 9 stand alone. All other media credits on all 10 pages are
NASA/JPL/GSFC/NPS/USGS/USDA (federal).

- [x] nasa_cc_evidence — counter-signed 2026-09-07 (note the Ashwin Kumar image credit above)
- [x] nasa_cc_causes — counter-signed 2026-09-07
- [x] nasa_cc_effects — counter-signed 2026-09-07
- [x] nasa_cc_consensus — counter-signed 2026-09-07
- [x] nasa_cc_what_is — counter-signed 2026-09-07
- [x] nasa_cc_mitigation — counter-signed 2026-09-07
- [x] nasa_cc_extreme_weather — counter-signed 2026-09-07
- [x] nasa_faq_scientists_agree — counter-signed 2026-09-07
- [x] nasa_faq_too_late — counter-signed 2026-09-07
- [x] nasa_no_mini_ice_age — counter-signed 2026-09-07

## 2. noaa_climate_explainers — 8 entries — VERDICT: public domain (US Gov work), three caveats

**Licence evidence**: 17 U.S.C. §105 (same basis as the active nca5_ch2
entry). USA.gov (https://www.usa.gov/government-copyright, accessed
2026-09-07): "Government work is something created by a U.S. government
officer or employee as part of their official duties." Sibling-agency
statement (https://www.weather.gov/disclaimer): "The information on National
Weather Service (NWS) Web pages are in the public domain, unless specifically
noted otherwise".

**Caveats for your review (all recorded in every entry)**:
1. **climate.gov is an archived site.** Site banner, verbatim: "This website
   is an ARCHIVED version of NOAA Climate.gov as of June 25, 2025. Content is
   not being updated or maintained, and some links may no longer work." The
   pages still serve and pins verified, but longevity of the URLs is not
   guaranteed, content will not be updated, and the banner text will appear
   in parsed chunks (hand-audit/boilerplate item).
2. climate.gov carries **no site-specific reuse statement** we could locate
   (its /about/copyright-information URL 404s in the archived site); the
   public-domain verdict rests on §105 + the federal-work basis above.
3. Some climate.gov articles are written by **contractor science writers**;
   §105 strictly covers officer/employee works. NOAA published these as
   official agency content, which is the normal basis for treating them as
   government works, but it is an inference, not a page statement.

**Per-page third-party scan**: clean on all 8.

**What it adds**: six conversational Climate Q&A pages (evidence, causes,
consensus/disagreement, CO2 attribution, the global-warming/climate-change
distinction, and the action-oriented "Can we slow or even reverse global
warming?"), plus the two flagship Understanding Climate explainers
(global temperature, atmospheric CO2) that back chart-adjacent numbers.

- [x] noaa_qa_gw_vs_cc — counter-signed 2026-09-07
- [x] noaa_qa_evidence — counter-signed 2026-09-07
- [x] noaa_qa_humans_causing — counter-signed 2026-09-07
- [x] noaa_qa_disagreement — counter-signed 2026-09-07
- [x] noaa_qa_slow_reverse — counter-signed 2026-09-07
- [x] noaa_qa_co2_humans — counter-signed 2026-09-07
- [x] noaa_uc_global_temperature — counter-signed 2026-09-07
- [x] noaa_uc_atmospheric_co2 — counter-signed 2026-09-07

## 3. metoffice_explainers — 5 entries — VERDICT: OGL v3.0, one operational flag

**Licence evidence** (https://www.metoffice.gov.uk/policies/legal, accessed
2026-09-07), verbatim:

> "The Website terms of use contain licence provisions under the terms of the
> UK Open Government Licence for Public Sector Information v3.0 which govern
> the use of material presented on this website, other than material which we
> tell you is governed by different licence terms."

Each of the 5 pages was individually checked for a different-licence notice
or third-party credit: none found (footer carries the standard "© Crown
Copyright", which the OGL covers). Every entry carries the attribution line
"Contains public sector information licensed under the Open Government
Licence v3.0" per the skeleton.

**What it adds**: UK-first framing (DESIGN §2.1 names this valuable),
UK-specific impacts (climate-change-in-the-uk), and **the best single
qa-adv-05 carrier in this expansion** — the climate-change-questions page
answers "Weren't there warnings of global cooling years ago?" directly, in
conversational register, alongside ~15 other sceptic-question answers
(hiatus, sun, Antarctic ice, model trust).

**Operational flag — pin drift**: Met Office pages embed rotating
cache-generation CSS class UUIDs (cosmetic only; observed live: the
what-is-climate-change pin changed within ~1 hour while page text was
byte-identical apart from those UUIDs). Pins were re-taken at the last
validation pass and verified, but **expect `make corpus` exit 3 (upstream
drift) on these five entries at some future run**. Options, your call at
activation: (a) accept re-pin churn under the existing exit-3
review-and-re-pin flow; (b) a small normalisation step for HTML hashing
(product change, own review); (c) commit the OGL-covered text as in-repo
prepared text (OGL permits redistribution; the schema supports committed
open text with a pin and no source_url — would need a deliberate
`.gitignore` exception, so it is NOT done in this PR per ADR-023 discipline).

- [x] metoffice_what_is_cc — counter-signed 2026-09-07
- [x] metoffice_causes — counter-signed 2026-09-07
- [x] metoffice_effects — counter-signed 2026-09-07
- [x] metoffice_cc_in_uk — counter-signed 2026-09-07
- [x] metoffice_questions — counter-signed 2026-09-07

## 4. owid_climate_explainers — 5 entries — VERDICT: CC BY 4.0 (OWID-authored text)

**Licence evidence** — every article page carries the per-page statement
(verbatim, accessed 2026-09-07):

> "All visualizations, data, and articles produced by Our World in Data are
> completely open access under the Creative Commons BY license. You have the
> permission to use, distribute, and reproduce these in any medium, provided
> the source and authors are credited. The data produced by third parties and
> made available by Our World in Data is subject to the license terms from
> the original third-party authors."

We ingest **OWID-authored article text only**. Upstream third-party data
licences noted per entry (we cite, never redistribute, their data):
food_local → Poore & Nemecek 2018 (Science); travel_footprint → UK
BEIS/Defra factors (Crown/OGL) + IEA-derived figures; co2_emissions → Global
Carbon Budget (CC BY 4.0); emissions_by_sector → Climate Watch/WRI (CC BY
4.0); safest_energy → Markandya & Wilkinson 2007, Sovacool et al. 2016.

**Fetch note**: ourworldindata.org returns HTTP 403 to Python's default
User-Agent. This PR gives the production corpus transport an honest project
User-Agent ("lets-talk-climate-emergency corpus fetcher") — mirroring the
gate's existing convention — which OWID serves normally. No browser spoofing.

**What it adds**: the strongest "what can I do, with numbers" content in the
expansion (diet, transport, energy choices) — squarely the qa-va-03 /
voices_action decline-cluster gap — plus global emissions context.

- [x] owid_food_local — counter-signed 2026-09-07
- [x] owid_travel_footprint — counter-signed 2026-09-07
- [x] owid_co2_emissions — counter-signed 2026-09-07
- [x] owid_emissions_by_sector — counter-signed 2026-09-07
- [x] owid_safest_energy — counter-signed 2026-09-07

## 5. hansen_2023_pipeline — 1 entry — VERDICT: CC BY 4.0 (verified on the VoR itself)

**Licence evidence** — the published version-of-record PDF states on p. 1,
verbatim:

> "© The Author(s) 2023. Published by Oxford University Press. This is an
> Open Access article distributed under the terms of the Creative Commons
> Attribution License (https://creativecommons.org/licenses/by/4.0/), which
> permits unrestricted reuse, distribution, and reproduction in any medium,
> provided the original work is properly cited."

Corroborated per the §2.2 gate's candidate filter: Crossref (licence
creativecommons.org/licenses/by/4.0/, content-version **vor**) and Unpaywall
(publishedVersion, cc-by) agree; queried 2026-09-07. DOI 10.1093/oxfclm/kgad008.

**Artefact / mirror note**: academic.oup.com refuses all non-browser clients
(HTTP 403 to urllib *and* curl, any headers). The pinned bytes are the
Internet Archive Wayback Machine's immutable 2023-11-03 capture of the
publisher's own PDF URL, verified to be the OUP version of record (OOCC
layout, correct DOI, the in-PDF CC BY statement above, 33 pages).
`canonical_url` points at the OUP landing page. Precedent: the active
nca5_ch2 entry pins an official mirror where the canonical host doesn't
serve the build environment. CC BY expressly permits redistribution, so the
mirror is licence-clean.

**Severity guardrail**: carries `consensus_position: beyond-assessed-range`
(REQUIRED by DESIGN §2.1/§2.3). Reminder at activation: §2.3 requires the
assessed-range statements (NCA5, already active with
`provides_assessed_ranges: true`) in the same index — satisfied today.

- [x] hansen_2023_pipeline — counter-signed 2026-09-07

## 6. hansen_2025_acceleration — 1 entry — VERDICT: CC BY 4.0 (verified on the VoR itself)

**Licence evidence** — the published version-of-record PDF states, verbatim:

> "© 2025 The Author(s). Published with license by Taylor & Francis Group,
> LLC. This is an Open Access article distributed under the terms of the
> Creative Commons Attribution License
> (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted
> use, distribution, and reproduction in any medium, provided the original
> work is properly cited."

Corroborated: Crossref (cc-by, content-version **vor**) + Unpaywall
(publishedVersion, cc-by), queried 2026-09-07. DOI 10.1080/00139157.2025.2434494.

**Artefact / mirror note**: tandfonline.com refuses non-browser clients
(HTTP 403). The pinned bytes are the publisher's own deposit on the Taylor &
Francis figshare portal (article 28335279, licence recorded there as CC BY
4.0), file `FULL-Hansen_etal2025.pdf` — verified to be the VoR (journal
cover sheet, *Environment* 67:1 pp. 6–44, correct DOI, in-PDF CC BY
statement). 18.6 MB (figure-heavy) — ingest cost note only.

Carries `consensus_position: beyond-assessed-range`.

- [x] hansen_2025_acceleration — counter-signed 2026-09-07

## 7. c3s_esotc — **EXCLUDED — licensing trap caught, decision needed**

The skeleton (and DESIGN §2.1) assumed the European State of the Climate is
under the open "Licence to Use Copernicus Products". **Verification against
the latest edition says otherwise.** ESOTC 2025 (published April 2026,
climate.copernicus.eu/esotc/2025) is now a **joint C3S/ECMWF–WMO
publication**, and its report copyright page states, verbatim:

> "© World Meteorological Organization and European Union, represented by the
> European Centre for Medium-Range Weather Forecasts (ECMWF), 2026. The right
> of publication in print, electronic and any other form and in any language
> is reserved by ECMWF and WMO. Short extracts from this publication may be
> reproduced without authorisation, provided that the complete source is
> clearly indicated. Editorial correspondence and requests to publish,
> reproduce or translate this publication (articles) in part or in whole
> should be addressed to: ECMWF Communication Section – Copernicus Team …
> copernicus-press@ecmwf.int … Chair, Publications Board, World
> Meteorological Organization (WMO) … publications@wmo.int"

That is the WMO-style short-extracts-only position DESIGN already classes as
**Tier C** for the WMO State of the Global Climate — not Tier A, not
self-certifiable. Additional findings: the executive-summary PDF carries no
notice of its own (same publication; no affirmative open grant); the
2022/2023 summary PDFs also carry no notice (absence ≠ licence); ECMWF's web
terms do put no-rights-notice public content under CC BY 4.0, but those
terms scope themselves to the ecmwf.int domain, so stretching them to
climate.copernicus.eu would be self-certification we refuse. Note the CDS
*data* licence did move to CC BY on 2025-07-02 — that covers datasets, not
this report.

**Recommended handling (your decision, none taken)**:
(a) move c3s_esotc to the Tier C skeleton and add it to the Phase-1.5
permission letters (copernicus-press@ecmwf.int, cc publications@wmo.int —
the same week-1 letter batch as WMO); (b) ask ECMWF whether the ESOTC web
summary pages fall under their CC BY web terms; (c) drop ESOTC from the MVP.
Until then nothing from ESOTC is proposed, and the manifest skeleton's
c3s_esotc TODO comment should be updated to record this finding when the
owner signs off (left untouched in this PR because the live manifest is
untouched).

## Cross-cutting honest flags

1. **HTML pin volatility (general)**: NASA/NOAA/OWID pins verified stable
   across double-fetch and again ~40 min later through the production
   transport, but any site redeploy will rotate them; the exit-3
   drift-review flow is the designed handling. Met Office rotates fastest
   (see §3).
2. **Ingest boilerplate**: HTML pages chunk cleanly but nav/footer chunks
   need hand-audit marking or a boilerplate filter before indexing (first
   HTML sources through the pipe; the two active docs are PDFs).
3. **Production-code changes in this PR** (both small, both reviewed by the
   test suite): honest User-Agent on `ingestion.fetch.urllib_transport`
   (OWID 403s the Python default UA), and two exact-name housekeeping
   exemptions in the corpus ship check for this staging file + packet
   (fail-closed behaviour preserved and pinned by a new test).
4. **qa-adv-05 gold marker**: issue #314's ask — once these entries
   activate and ingest, qa-adv-05's `blocked_on: corpus-expansion` can be
   resolved with gold chunks from metoffice_questions and/or
   noaa_qa_gw_vs_cc, qa-va-03's from noaa_qa_slow_reverse/owid pages plus
   the voices layer. Gold edits are deliberately NOT in this PR (separate
   change, per #314's own scoping).
5. **Zero Anthropic API calls** were made in preparing this expansion; all
   verification was web-fetch + local pipeline work.
