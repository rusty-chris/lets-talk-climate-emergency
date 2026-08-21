# ChartSpec vocabulary (issue #15)

> Generated from `charts.spec.chartspec_schema()` by
> `charts.spec.render_schema_doc()`. Do not edit by hand — the tests
> `test_schema_doc_matches_schema` (byte-equality),
> `test_schema_doc_names_every_property` and
> `test_schema_doc_states_required_fields` pin this file to the schema,
> so it cannot drift from what the validator enforces (review #136).

The chart planner (#16) emits a **ChartSpec** — a JSON document, never
code. `charts.spec.validate_spec` decides legality in pure code and
cross-checks every curation-time decision against `datasets/manifest.yaml`
(ADR-020): the model supplies neither numbers, pixels, nor artefact text.
The schema is closed everywhere (`additionalProperties: false`); the only
tolerated extra is a top-level `_meta` provenance block, which is ignored
semantically, excluded from the permalink hash, and bounded to 2 KiB
serialised (the whole spec is bounded to 8 KiB; review finding #137).
Every string field has a maxLength; every `[start, end]` pair must be
strictly increasing (start < end) and every number finite — NaN/Infinity
refuse at parse time (review findings #128/#130).

## Closed vocabularies

- **chart_type** — one of: `area`, `bar`, `context_recent_inset`, `dual_axis_line`, `line`. The five MVP types are a closed
  enum; warming stripes and other Phase 2 types are not expressible.
- **transforms[*].op** — one of: `anomaly_vs_baseline`, `resample`, `rolling_mean`, `unit_conversion`. These generic transforms
  live in a per-series `transforms` list (at most 4). Each op carries
  exactly its own parameters (review finding #132):
  - `rolling_mean` and `resample` require **window_years** — the window /
    target resolution in years; strictly positive, at most 100 (the
    coarsest pack resolution is a 100-year bin, so longer windows smooth
    harder than the data legitimises);
  - `unit_conversion` requires **to_unit**, and the (source unit ->
    to_unit) pair must exist in the code-owned `UNIT_CONVERSIONS` table —
    the conversion arithmetic is code-owned (no LLM-authored factor), and
    the axis unit label can never become free text; at most one
    unit_conversion per series;
  - `anomaly_vs_baseline` takes no parameters.
- **overlap_policy** — one of: `prefer_instrumental`, `prefer_paleo`, `show_both`. What a spec may carry for
  a given splice is further narrowed to the manifest-legal value recorded in
  that pair's `overlap.policy` (review finding #47).

## Top-level fields

- **spec_version**, **chart_id**, **chart_type**, **title** — required.
  **subtitle** is optional. `chart_id` is slug-shaped
  (`^[a-z0-9][a-z0-9-]{0,63}$`) and `spec_version` is semver-shaped
  (review finding #137).
- **time_range_ce** `[start, end]` — CE years for a single-panel chart; must
  lie within every series' dataset coverage (the parsed usable extent,
  review finding #52). Refused on `context_recent_inset` specs — the
  panels own the time ranges there (review finding #135).
- **time_axis** `{calendar: "CE", convert_bp: <bool>}` — both members
  required when present; set `convert_bp: true` when any series plots a
  `years_bp` dataset on the CE axis. BCE-aware axis labelling is
  renderer-owned time_axis semantics (amendment 8); there is no per-spec
  label/tick styling field.
- **panels** — present if and only if `chart_type` is `context_recent_inset`.
  A `context` and a `recent` panel (both required), each with a required
  `label` and `time_range_ce`; `shared_y_scale_with` names the other panel
  (amendments 4-5). The recent range must lie strictly inside the context
  range: narrower than the context and spanning at least 10 years, so the
  inset is a real zoom, not a duplicate or a sliver (review finding #130).
- **series** — required; one to eight series (see below).

## Series fields

A series names exactly one data source: a single `dataset`, or a splice
pair (`splice_series` + `splice_pair_id`). Both must resolve to chart-pack
datasets (`in_chart_pack`, review finding #117).

- **id**, **label**, **unit** — required. `unit` must equal the source
  dataset's variable unit (or a `unit_conversion` transform's `to_unit`).
- **axis** `left|right`, **color** — optional per-series presentation.
- **dataset** — a single chart-pack dataset id.
- **splice_series** `[paleo, instrumental]` + **splice_pair_id** — a
  manifest-keyed splice. `splice_series` must equal the pair's
  `[paleo, instrumental]`; the splice year, resolution note and rebaseline
  legality are all owned by the manifest pair (ADR-020, review finding
  #131).
- **overlap_policy** — required on a spliced series (and refused on an
  unspliced one); must equal the pair's manifest `overlap.policy` (review
  findings #47/#127).
- **rebaseline_to** `{apply_to, alignment_period_ce, alignment_disclosure}`
  — all three members required; present if and only if the manifest pair
  records a rebaseline (refused on unspliced series, review finding #127).
  `apply_to` and `alignment_period_ce` must equal the manifest values and
  `apply_to` must name a plotted splice member; the `alignment_disclosure`
  string must equal the manifest `rebaseline.disclosure` verbatim, so the
  alignment method reaches the artefact and cannot drift (review finding
  #50).
- **scale_domain** `[min, max]` — required per series for
  `context_recent_inset` (shared per-axis domains across panels). It must
  contain the plotted data's extent (no clipping, review finding #48) and
  must not span more than 4x the data span plus the extension to zero —
  an over-wide axis flattens the signal (review finding #129). Any domain
  that excludes zero requires the `non_zero_baseline` annotation (and the
  annotation is refused without such a domain, review finding #127).
- **uncertainty_band** `{lower, upper, source}` — all three required;
  `lower`/`upper` are source-frame column names (e.g. the flagship's
  `temp_c_5`/`temp_c_95`) and `source` names a member dataset of this
  series (amendment 6).
- **baseline** `{value, label}` — both required when present. `value` must
  be 0 (the reference-period zero line — any other reference line awaits a
  manifest-anchored vocabulary entry, review finding #134); on a
  rebaselined series the label must name the manifest display-reference
  period.
- **transforms** — a list of generic transform ops (`op` required; see the
  per-op parameters above).
- **annotations** `{splice_point, resolution_note, non_zero_baseline}` — a
  spliced series must carry `splice_point` (whose required `year_ce` must
  equal the manifest splice year, and whose required `label` must name
  that year) and `resolution_note` (which must equal the manifest pair's
  `resolution_note` verbatim, review finding #131); all three are refused
  on an unspliced series (`non_zero_baseline` and its required `label`
  belong with a zero-excluding scale_domain). Annotation placement is
  fixed semantics, not a spec choice (amendment 7).

## Captions

There is no `caption` field anywhere. Captions (sources, licences, access
date, site URL) are generated by the renderer from the manifest
attribution/licence strings and deployment config, so they cannot drift
from the licensing record and cannot be an injection vector (amendment 9).
The rendered artefact is identified by its spec hash — the permalink
identity — never by `chart_id`/`spec_version` (review finding #137).
