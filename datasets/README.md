# datasets/

Manifest + fetch scripts for the chart data pack (DESIGN.md §3.7). Small,
`open`-licensed CSVs may be bundled directly; everything else is fetched at
build time by the scripts here.

**Invariant:** the chart data pack must contain only `permitted_context:
open` datasets — exported chart images are redistributed by users into
arbitrary contexts, including commercial ones, and must never risk breach of
a non-commercial licence (DESIGN.md §2.1).

Loaders and parsers land in `charts/pack.py` (issue #16).
