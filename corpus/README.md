# corpus/

Manifest only — **never prepared/full document text**, except for
`permitted_context: open` documents (DESIGN.md §2.1).

This directory will hold the corpus manifest (per-document `licence`,
`licence_evidence`, `attribution_text`, `canonical_url`, `redistributable`,
`permitted_context`, `permission_evidence`, `consensus_position`, `sha256`,
`retrieved_at`, `human_signoff`) and fetch scripts for `open`-tier text.
Non-`open` documents (Tier B/C) are fetched at ingest time via the scripts
here and are never committed to this repo — see the licensing invariants in
DESIGN.md §2.1 and the manifest schema work in `ingestion/manifest.py`
(issue #5).

Nothing is ingested or indexed until it has passed the CC-BY licensing gate
(DESIGN.md §2.2) and carries a `human_signoff`.
