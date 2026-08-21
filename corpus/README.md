# corpus/

Manifest only — **never prepared/full document text**, except for
`permitted_context: open` documents (DESIGN.md §2.1).

`manifest.yaml` is the real MVP corpus manifest (per-document `licence`,
`licence_evidence`, `attribution_text`, `canonical_url`, `redistributable`,
`permitted_context`, `permission_evidence`, `consensus_position`, `sha256`,
`retrieved_at`, `human_signoff`, optional `ingest_profile`). Sources whose
pins/licensing are not ready yet live in its commented skeleton — never as
placeholder entries (review #145). Non-`open` documents (Tier B/C) are
fetched at ingest time into a transient workspace and are never committed
to this repo (review #144) — see the licensing invariants in DESIGN.md
§2.1 and the manifest schema in `ingestion/manifest.py` (issue #5).

Two targets consume this manifest (the split is deliberate, review #145):

- `make corpus` — the fast #5 gate: fetch open-tier text (fail-closed,
  temp-verify-rename), verify every pin, run every licensing invariant.
- `make ingest` — the #7 production pipeline: manifest gate → verified
  fetch → parse (Docling, loud PyMuPDF fallback) → structure-aware chunk
  → citation metadata → citation blocks. Writes the run record to
  `corpus/ingest_run.json` (gitignored) and chunk/block payloads to
  `data/ingest/` (gitignored).

`hand-audit-checklist.md` is the per-source-family chunk-quality audit
(issue #7 acceptance criterion 2). Fetched artefacts (`*.pdf`, `*.html`)
are re-fetchable from their pins and gitignored.

Nothing is ingested or indexed until it has passed the CC-BY licensing gate
(DESIGN.md §2.2) and carries a `human_signoff`.
