# Chunk-quality hand-audit checklist (issue #7 acceptance criterion 2; review #145)

Run this checklist **per source family** before any document family's
chunks are admitted to the index, and re-run it whenever the chunker or
parser changes materially. Evidence for each completed audit is recorded
in the "Completed audits" table below; the mechanical signals come from
`corpus/ingest_run.json` and the persisted `data/ingest/chunks.jsonl`
(`make ingest`), the judgement calls from reading a sample of chunks.

## The checklist (apply to every family)

Mechanical (scriptable over `chunks.jsonl` — all must be zero/true):

- [ ] **Cap**: no chunk `token_count > max_tokens` except chunks flagged
      `oversized_atomic` (single unsplittable table row / token-dense
      word — #139); count the flagged ones and eyeball each.
- [ ] **Ids**: chunk ids pairwise unique (#138); two identical runs give
      byte-identical `chunks.jsonl` (idempotency).
- [ ] **Zero bare placeholders**: no body is `[FIGURE]`/`[TABLE]` (#138).
- [ ] **Front matter**: no chunk under `Authors` / `Table of Contents` /
      `Contents` or role-line sections; no author/affiliation walls or
      dot-leader ToC lines in bodies (#140).
- [ ] **References**: no bibliography text as evidence chunks; no chunk
      filed under a running-head pseudo-section (#141).
- [ ] **Floor**: no non-atomic chunk with body below `min_tokens` (#147).
- [ ] **Degraded flags**: every chunk of a PyMuPDF-parsed document has
      `needs_hand_review: true` and the run record warns (#143). A
      degraded document must NOT be indexed before this audit.

Judgement (read ≥10 chunks sampled across the document):

- [ ] **Section paths** reflect the document's real hierarchy (numeric
      prefixes for papers; Key Messages for NCA5 — #148/#151); context
      headers read sensibly.
- [ ] **Boundaries**: no chunk mixes content across headings; overlap
      sentences look right.
- [ ] **Prose integrity**: no stray superscript-marker numerals in
      quoted prose (#150); ligatures/hyphenation normalised; tables keep
      row labels with values (#139).
- [ ] **Calibrated language**: `confidence_markers` match what the
      sampled bodies actually assert (negation/quotes handled — #146).
- [ ] **Assessed-range presence** (corpus-level, §2.3): the documents
      declaring `provides_assessed_ranges` actually surface assessed
      warming/sensitivity statements in their chunks.

## Source families

| Family | Representative(s) | Status |
|---|---|---|
| Gov/agency assessment PDF (Docling) | NCA5 chapters (`nca5_ch2`) | audited — see below |
| Journal CC-BY PDF, two-column (Docling) | `esd_tipping_review` (Copernicus ESD) | audited — see below |
| HTML explainer (HTML-direct path) | NASA / NOAA / Met Office / OWID pages | audited 2026-09-07 (activation pass, below) — boilerplate marking required before indexing (#331) |
| Journal CC-BY PDF, Hansen pair (Docling) | `hansen_2023_pipeline`, `hansen_2025_acceleration` | audited 2026-09-07 (activation pass, below) — affiliation-line chunks flagged for the #331 loop |
| Tier B non-commercial | UNEP EGR, Carbon Brief verbatim set | pending — licensing letters / pins outstanding; text never lands in-repo (#144) |
| Curated headline statements (Tier C) | IPCC SPM curated set; ESOTC (letters/07-ecmwf-esotc.md) | blocked on the #23 legal check / permission letters; feature flag default OFF |

## Completed audits

Recorded 2026-08-21 from the review-7 fix trial: both documents
re-fetched from their manifest `source_url`s and verified byte-identical
against the spike-02 sha256 pins, parsed with Docling (575 / 496
blocks), chunked with the production defaults (500-token cap,
punctuation-aware counter). Before-numbers are the merged PR #124
pipeline over the identical parses; see the fix PR body for the full
table.

| Check (mechanical) | nca5_ch2 | esd_tipping_review |
|---|---|---|
| Chunks | 75 (was 112 incl. noise) | 64 (was 81 incl. noise) |
| Cap violations, unflagged (was max 539 / 1734 tokens) | 0 | 0 |
| `oversized_atomic`-flagged chunks | 0 | 0 |
| Duplicate chunk ids | 0 | 0 (was 1 pair) |
| Bare `[FIGURE]`/`[TABLE]` bodies | 0 (was 10) | 0 (was 2) |
| Authors / ToC section chunks + dot-leader bodies | 0 (was 8 + 7) | 0 |
| Affiliation-wall chunks | 0 (was 1) | 0 (was 3 + fragments) |
| Running-head pseudo-section chunks | 0 (was 16) | 0 |
| Stray superscript-marker bodies | 0 (was 40) | 0 |
| Min non-atomic body tokens (floor 20) | 31 (was 14) | 57 (was 5) |
| Section-path depth histogram | {1: 6, 2: 69} (was flat {1: 112}) | {1: 11, 2: 32, 3: 21} |

Judgement items (sampled per the checklist): section paths carry the
Key-Message parents on NCA5 and the numeric tree on ESD; sampled bodies
read as clean prose; calibrated phrases in sampled chunks match their
`confidence_markers`.

Known limitations recorded honestly (kept open, not hidden):

- Superscript-marker stripping is text-heuristic; #150's parse-time
  font-information option was not taken for MVP. A numeral+capital
  sentence opening after a marker-bearing sentence is stripped by
  design (the pinned disambiguation keeps lowercase counts like
  ". 24 stations").
- NCA5 Key-Message depth relies on Docling labelling the "Key Message
  N.M" text as headings; sub-sub structure below the headline level is
  not reconstructed.
- Journal back matter that Docling files as plain text under the last
  section is caught by the inline-label list (Copernicus statement
  shapes); an unlabelled acknowledgement paragraph would still pass.
- The tiny-chunk floor suppresses sub-20-token non-atomic fragments
  outright (greedy packing has already merged anything the cap
  allows); a cap-blocked trailing fragment is dropped, not re-packed.

## Completed audit — 2026-09-07 activation pass (30 new Tier-A documents)

Recorded by the corpus-activation session after the owner's sign-all-30
decision. `make corpus` exit 0 over the activated 32-document manifest
(after the 11 recorded re-pins — see the entries' human_signoff notes);
ingest run over the pin-verified artefacts: **871 chunks / 871 blocks from
32 documents, 0 skips** (nca5_ch2 76, esd_tipping_review 65,
hansen_2023_pipeline 148, hansen_2025_acceleration 234; HTML families:
NASA 161, Met Office 81, OWID 66, NOAA 40).

| Check (mechanical, all 871 chunks) | Result |
|---|---|
| Cap violations, unflagged (500-token cap) | 0 |
| `oversized_atomic`-flagged chunks | 0 |
| Duplicate chunk ids | 0 |
| Bare `[FIGURE]`/`[TABLE]` bodies | 0 |
| Degraded (PyMuPDF) documents | 0 — all four PDFs parsed by Docling; HTML by the HTML-direct path |
| Min non-atomic body tokens (floor 20) | 30 |
| Front-matter section chunks | 0 under `Authors`/`Contents`/role lines (16 sections stripped with warnings); see Hansen flags below for what the stripper does NOT catch |

Judgement items (≥10 chunks read per family, sampled across documents):

- **§2.3 assessed-range presence**: satisfied. `nca5_ch2`
  (`provides_assessed_ranges: true`) surfaces assessed statements in 34 of
  its 76 chunks, including Key Message 2.1 ("It is unequivocal that human
  activities have increased atmospheric levels of carbon dioxide…") — the
  assessed-range carrier is in the same index as the two Hansen documents,
  whose 382 chunks all carry `consensus_position: beyond-assessed-range`
  propagated from the manifest.
- **qa-adv-05 carrier confirmed**: `metoffice_questions` yields a clean
  350-token chunk under the section "Weren't there warnings of global
  cooling years ago?" answering the 1970s-cooling myth directly;
  neighbouring sceptic-question chunks (hiatus, sun, CO2-lag) are equally
  clean one-chunk-per-question.
- **Boundaries/prose**: sampled Met Office, NOAA and OWID chunks are clean
  prose, one Q&A per chunk, sensible starts/ends; confidence markers in
  sampled NASA/NOAA chunks match the bodies ("very likely", "extremely
  likely", "high confidence" all verbatim in-body).

**Boilerplate findings (the known #331 gap — marked, not fixed here):**

- **Pure nav-menu chunks: 71 of 348 HTML chunks (20%), all in the NASA
  family** (44% of NASA's 161). Every NASA page contributes ~7 "Suggested
  Searches" menu chunks plus "Discover More Topics" teasers. NOAA, Met
  Office and OWID contribute ~0-2 boilerplate chunks per page (a looser
  heuristic that also counts short link-lists and footer fragments puts
  the HTML-wide proportion at ~30%, consistent with the sign-off packet's
  "a third to a half per page" estimate for the worst family).
- **Nav-polluted section paths on 15 substantive NASA chunks**: real
  explainer prose filed under nav headings (e.g. "Suggested Searches/La
  NASA refuerza Artemis…" carrying the 97%-consensus answer). NOAA has a
  milder variant ("RSS Feed" as a section path on substantive chunks).
  The #331 filter must fix the paths, not just drop nav chunks, or these
  bodies lose their context headers.
- **Hansen affiliation-line chunks**: 13 chunks in `hansen_2023_pipeline`
  and 2 in `hansen_2025_acceleration` are single numbered author
  affiliation lines (30-38 tokens, e.g. "3 NASA Goddard Institute for
  Space Studies…") filed under the root section — Docling emitted them as
  individual paragraphs, so the #140 affiliation-wall stripper (which
  catches walls, not lines) passed them. Plus 3 CRediT "Authors'
  contributions" chunks and one line-number artefact chunk in
  hansen_2023. Harmless for licensing, noise for retrieval — filed with
  the #331 boilerplate loop.
- **Good news**: the NOAA "ARCHIVED site" banner predicted by the
  sign-off packet's caveat does NOT reach any chunk (0 occurrences) — the
  HTML path drops it.

**Verdict**: mechanical checks all green; substantive content of all 30
new documents chunks cleanly and the #314 gap carriers are present. The
HTML families (and the Hansen affiliation lines) must go through the
#331 boilerplate marking/filtering before an index build ships — that is
a pre-existing scoped follow-up, not a regression in this activation.
