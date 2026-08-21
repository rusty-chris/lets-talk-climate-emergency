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
| HTML explainer (HTML-direct path) | NASA / NOAA / Met Office / OWID pages | pending — no real pages pinned yet (manifest skeleton) |
| Tier B non-commercial | UNEP EGR, Carbon Brief verbatim set | pending — licensing letters / pins outstanding; text never lands in-repo (#144) |
| Curated headline statements (Tier C) | IPCC SPM curated set | blocked on the #23 legal check; feature flag default OFF |

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
