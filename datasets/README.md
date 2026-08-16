# datasets/

Manifest for the chart data pack (DESIGN.md §3.7). This directory tracks
only `manifest.yaml` and this file — **no dataset file is ever committed
to git** (ADR-023, client decision 2026-08-16): `make datasets` fetches
every dataset from its origin URL, verifies the bytes against the
manifest-pinned sha256, and lands them in the gitignored
`data/datasets/`. The manifest pins *which bytes* without hosting them;
the public archives are the data store.

**Invariant:** the chart data pack must contain only `permitted_context:
open` datasets (`in_chart_pack: true`) — exported chart images are
redistributed by users into arbitrary contexts, including commercial
ones, and must never risk breach of a non-open licence (DESIGN.md §2.1).
`open-provisional` datasets (Kaufman 2020, Bereiter 2015 — licence
confirmation pending, issue #23) are fetchable like any other entry but
excluded from `in_chart_pack` and never mirrored anywhere.

Fetch/verify/land flow: `charts/datasets.py`. Committed parsers (raw file
-> tidy DataFrame): `charts/pack.py`. Run `make datasets` to refresh.
