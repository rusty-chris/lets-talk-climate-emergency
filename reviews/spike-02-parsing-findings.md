# Spike #2 — Docling parsing + structure-aware section chunking: findings

**Issue:** #2 (Phase 0 spike / de-risk). **Author:** implementer session, 2026-08-16.
**Design refs:** DESIGN §2.4 (ingestion pipeline), §10 Phase 0; IMPLEMENTATION.md §1 (parse/chunk seams), §2 item 6 (spikes exempt from test-first).

**Purpose.** De-risk the schedule long pole (ingestion QA) *before* the production
pipeline (#7) is built: prove that Docling (PyMuPDF fallback wired) can parse a
real NCA5 chapter and a real CC-BY review paper into a structure we can chunk on
headings, hand-review the chunk boundaries on both, and enumerate the parsing
failure modes #7 must handle.

**This is spike code**, under `ingestion/` and clearly marked as prototype
(`ingestion/parse.py`, `ingestion/chunk.py`, `ingestion/spike_run.py`). It is not
the production parser/chunker; #7 starts from a red test suite. The characterisation
tests in `tests/unit/test_spike_chunker.py` pin the prototype's observed behaviour
on a committed **synthetic** fixture (no real Tier B/C text is committed, DESIGN
§2.1 / IMPLEMENTATION.md §5).

---

## GATE VERDICT

**Chunk boundaries correct per hand review: YES**, with one recorded structural
limitation (heading *nesting* is not recovered — see Finding 1).

Evidence (hand review of the recorded chunk dumps in `data/spike/`, Docling backend):

- **No cross-heading bleed.** Every chunk carries exactly one section path and its
  body is drawn only from blocks under that heading. Boundaries land exactly on
  heading transitions — e.g. ESD `c0005` (`1.1/1.2` region) ends and `c0007`
  begins precisely at the `1.3 Motivation…` heading; NCA5 chunks switch section
  path cleanly at each Key Message / sub-heading. The invariant is enforced
  structurally in `group_sections` (a heading always resets the current section),
  and is pinned by `test_spike_chunks_never_cross_headings`.
- **Section paths correct.** Docling recovered the full outline of *both*
  documents, including the two-column ESD paper's numbered section tree
  (`1`,`1.1`,…,`2.2.1`,…,`5`,`References`). Paths are correct as **flat labels**;
  nesting depth is lost (Finding 1).
- **~500-token target respected on these documents — but the cap is NOT a
  mechanism guarantee** (adversarial review, issue #41). The spike chunker has
  three empirically confirmed cap-violation paths that these two documents
  happened not to trigger: (a) overlap seeding appends the triggering sentence
  with no cap re-check (a 598-token chunk is constructible under a 500 cap);
  (b) an atomic unit longer than the cap (e.g. a 600-token footnote) passes
  through whole; (c) the prepended context header is excluded from token
  accounting. There is also an unrecorded rule: overlap is skipped after an
  atomic-unit chunk. **Requirement for #7:** define the cap's semantics
  explicitly — recommended: cap applies to chunk text *including* the context
  header; overlap seeding must re-check the cap; oversized atomic units are
  split at sentence boundaries or hard-truncated with the decision logged —
  and write the red-phase cap tests against that definition, not against the
  spike's observed behaviour. Measured on these documents: NCA5 n=117,
  mean≈240, max=500; ESD n=72, mean≈381, max=499.
- **One-sentence overlap works.** On real data, ESD `c0006` opens with the exact
  last sentence of `c0005` ("We do not restrict our definition to specific spatial
  scales, timescales, or severity of impact of the tipping elements.").
- **Figures/tables retained as citable text.** Placeholders `[FIGURE]` / `[TABLE]`
  survive chunking and the caption text (e.g. "Figure 2.4. Changes shown are the
  difference between…") is kept in the same chunk (DESIGN §2.4). Caveat: caption↔
  object *association* is unreliable (Finding 3).
- **In-text citations preserved.** Author-year cites ("(Kravtsov et al., 2018)")
  and the calibrated-language phrasing NCA5 depends on ("virtually certain, very
  high confidence") come through intact.

The gate is the hand review above, not test coverage (issue #2 acceptance).

---

## Source documents (NOT committed — gitignored under `data/spike/`)

| doc_id | Title | URL | licence | sha256 |
|---|---|---|---|---|
| `nca5_ch2` | NCA5 Chapter 2: Climate Trends | https://toolkit.climate.gov/sites/default/files/2025-07/NCA5_Ch2_Climate-Trends.pdf | **Public domain** — U.S. Government work (Fifth National Climate Assessment, USGCRP, 2023) | `90298e25aee94684334b7964c61e030854bd250107037ec23babf3fac90b243e` |
| `esd_tipping_review` | Wunderling et al., "Climate tipping point interactions and cascades: a review", *Earth Syst. Dynam.* 15, 41–74, 2024 | https://esd.copernicus.org/articles/15/41/2024/esd-15-41-2024.pdf | **CC-BY 4.0** | `39906d865f171139878eada6b5825ec02e7e43da1d09248d94918d7ea8b75013` |

**Licence evidence:**
- **NCA5 Ch.2** — the Fifth National Climate Assessment is a U.S. Global Change
  Research Program (federal) product; U.S. Government works are not subject to
  domestic copyright (public domain). Canonical home is
  `nca2023.globalchange.gov/downloads/`; that host would not resolve from this
  environment (see Deviations), so the byte-identical chapter PDF was fetched from
  the NOAA Climate Program **toolkit.climate.gov** mirror.
- **ESD review** — DOI **10.5194/esd-15-41-2024**. The article landing page
  (https://esd.copernicus.org/articles/15/41/2024/) states verbatim, on the figure
  and article: *"© Author(s). Distributed under the Creative Commons Attribution
  4.0 License."* Copernicus Publications publishes all ESD articles under CC-BY 4.0.
  Title explicitly self-describes as "a review".

Reproduce the parse/chunk artefacts with: `uv run python -m ingestion.spike_run`
(Docling) or `… spike_run pymupdf` (fallback). Outputs land in `data/spike/`:
`*.blocks.txt`, `*.chunks.txt`, `*.chunks.json` (Docling) and `*.pymupdf.*`.

**Tooling used:** docling 2.120.1, pymupdf 1.28.2, Python 3.12.3. These are spike-only
deps installed with `uv pip install docling pymupdf` and deliberately **not** added
to `pyproject.toml`/`uv.lock` (docling pulls torch/transformers/OCR — ~GB). #7 owns
the decision to pin them as production dependencies.

---

## Docling vs PyMuPDF fallback (why Docling is the primary backend)

| | Docling | PyMuPDF fallback |
|---|---|---|
| NCA5 headings recovered | 105 (real section tree) | 122 but **noisy** — bold "Key Message" prose lines and ToC dot-leader lines mis-tagged as headings |
| ESD headings recovered | 33 (full `1`…`5`,`References` tree) | **3** — only the title-page big text; **every numbered section heading missed** |
| ESD section paths | correct per-section | **59 of 60 chunks collapse under one mis-detected title fragment** — structure-aware chunking effectively defeated |
| Tables / figures / captions | typed blocks (NCA5: 2 tbl, 16 fig, 16 cap; ESD: 3 tbl, 7 fig, 5 cap) | none — no document model |
| Two-column reading order | correct | mostly correct for full-width text; fragments every visual line (NCA5: 2370 "blocks", ESD: 3672) |

**Conclusion:** the PyMuPDF fallback's font-size heuristic is unusable for journal
articles whose headings are the *same font size* as body text (the entire 34-page
ESD review became one section). Docling's layout model is required for structure;
the fallback is a last resort for text extraction only, and #7 should treat a
PyMuPDF result as "no reliable structure" rather than trusting its heading guesses.

---

## Parsing failure modes the production pipeline (#7) MUST handle

Each becomes a named test case in #7's TDD plan (issue #2 acceptance criterion 3).

**1. Heading nesting is flat — Docling returns every heading at level 1.**
Both documents' hierarchy (NCA5 Key Message → sub-heading; ESD `2` → `2.2` →
`2.2.1`) is present only in the heading *text* (numeric prefixes / wording), not in
Docling's `level`. Result: section paths are single-element (`('2.2.1 …',)`) instead
of nested (`('2 …','2.2 …','2.2.1 …')`). Boundaries are still correct, but retrieval
context and the "document → section" header lose depth.
→ #7 must reconstruct hierarchy: parse numeric section prefixes for papers, and for
NCA5 use the Key-Message structure / font metrics. Test: nested `section_path` for a
`2.2.1`-style fixture.

**2. Front-matter and author/affiliation blocks become spurious headings/sections.**
NCA5 page 2 alone produced 31 "headings" (author roles: "Chapter Lead Author",
"Cover Art", "Recommended Citation", the affiliation list). ESD emits a wall of
affiliation `text` blocks with an empty section path (`()`).
→ #7 needs a front-matter/boilerplate stripper (drop author lists, running
"Published by Copernicus…", the recommended-citation block) before chunking. Test:
front-matter blocks excluded from chunk output.

**3. Caption↔figure/table association is unreliable; table *data* is dropped.**
Docling emits the figure/table object and its caption as **separate** blocks, and
`caption_text()` returned empty for most objects (many placeholders render
`[FIGURE — no caption extracted]`) even though the caption text sits in the very
next block. Some captions are also **duplicated** (the figure's short title appears
both as an attached caption and as a standalone text block, e.g. NCA5 Fig 2.4).
Table *cell content* is lost entirely — only a `[TABLE]` placeholder is kept, so a
data table like NCA5 "Table 2.1 — greenhouse-gas concentrations" contributes no
citable numbers.
→ #7 must (a) pair captions to their object via Docling's parent/caption refs or
spatial proximity, (b) de-duplicate caption vs body text, and (c) export table
cells (Docling can emit tables as structured/markdown) so numeric tables stay
citable. Tests: one caption attached per figure/table; no duplicate caption text;
table cells present in the chunk.

**4. Superscript reference/footnote markers are flattened into body text as stray
numerals.** NCA5 renders "…risen by about 11 inches, which is considerably more…
**67** In just the last three decades… **68**". The superscript citation numbers
become inline integers glued to sentence ends, corrupting both the prose and the
sentence splitter (which then treats " 67 In…" as a sentence start).
→ #7 must detect and either strip or structurally capture superscript note/citation
markers before sentence splitting. Test: a fixture line with a superscript marker
yields clean sentence text (marker removed or moved to metadata).

**5. Reference lists are ~20–30 % of all chunks and low citable value.**
References were 22/117 chunks (NCA5) and 23/72 (ESD). They chunk "correctly" but a
bibliography is not evidence prose; embedding hundreds of citation strings dilutes
retrieval.
→ #7 should segregate the reference/bibliography section (drop from the evidence
index, or store as structured citation metadata). Test: `References` section not
emitted as evidence chunks.

**6. Two-column reading order — good with Docling, but hyphenation/ligatures leak.**
Docling reassembles the two-column ESD text in correct reading order (no column
jumping — verified by continuous sentences). Residual artefacts: line-break
hyphens sometimes join wrong ("large-scale" → "largescale"), and PDF ligatures
survive as unicode (ﬁ, ﬂ) in the raw PyMuPDF path.
→ #7 should normalise de-hyphenation and ligatures. Test: hyphenated line-break and
a ﬁ-ligature fixture normalise to plain text.

**7. Naive sentence splitter over-segments on abbreviations/figure refs.**
The prototype's regex splits on `.?!` + capital/digit, so "(see Fig. 1), tipping"
and decimals/"e.g." can produce false sentence breaks (harmless to boundaries, but
it makes the one-sentence overlap and packing granularity ragged).
→ #7 should use a real tokenizer's sentence segmentation (or a protected-abbrev
list) — and the real embedding tokenizer for token counts, not the whitespace-word
estimate the spike uses. Test: known abbreviation cases don't split.

**8. Tiny/degenerate chunks.** Min chunk sizes were ~2–10 tokens (a lone
heading-follower fragment, an axis label like "45°N" pulled from a figure region).
→ #7 should merge or suppress sub-threshold chunks. Test: no chunk below a minimum
token floor except deliberately atomic units.

---

## Cost / performance note

Docling ran the RT-DETR layout model **and** RapidOCR on every page (default
pipeline), on GPU: ≈2.5 min for the 40-page NCA5 chapter, ≈similar for ESD, plus a
one-time ~30 MB model download. OCR is unnecessary for these born-digital PDFs and
dominates runtime. → #7 should disable OCR for born-digital sources (detect a text
layer first) and budget parse time / consider caching parsed `StructuredDoc`s.

---

## Deviations

1. **NCA5 source host unreachable.** `nca2023.globalchange.gov` (and other
   `globalchange.gov` subdomains, and `repository.library.noaa.gov`) returned DNS
   "no answer" / HTTP 403 from this environment. Adapted per issue guidance: fetched
   the **byte-identical** chapter PDF from the official NOAA **toolkit.climate.gov**
   mirror. Same public-domain U.S. Government work; sha256 recorded above so #7 can
   re-verify against the canonical host when reachable.
2. **Review paper source.** MDPI (a first CC-BY candidate) Cloudflare-blocked
   automated download; pivoted to a Copernicus ESD review, which is CC-BY 4.0 and
   download-friendly — and whose two-column layout is exactly the failure mode this
   spike needed to exercise. No compromise to the CC-BY requirement.
3. **Spike dependencies not pinned in `pyproject.toml`.** Docling's heavy transitive
   deps (torch/transformers/OCR) were installed ad hoc for the spike and recorded
   here rather than committed to the lockfile; #7 owns productionising them.
