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

Recorded from the review-7 fix trial (re-fetched, sha-verified spike
documents; results appended by the final verification run of PR
"Fix ingestion review findings (#138–#151)" — see that PR's body for the
before/after numbers):

| Check | nca5_ch2 | esd_tipping_review |
|---|---|---|
| Cap violations (unflagged) | see PR body | see PR body |
| Duplicate ids | see PR body | see PR body |
| Bare placeholders | see PR body | see PR body |
| Front-matter / ToC / affiliation leakage | see PR body | see PR body |
| Reference / running-head leakage | see PR body | see PR body |

Known limitations recorded honestly (kept open as review follow-ups):
NCA5 Key-Message hierarchy depth depends on how Docling labels the
headline sub-headings on a given chapter; superscript-marker stripping
is text-heuristic (#150's parse-time font-information option was not
taken for MVP).
