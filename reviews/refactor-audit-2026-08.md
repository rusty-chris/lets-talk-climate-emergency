# Logic-efficiency refactor audit — 2026-08

**Scope:** every production module merged to `main` as of 2026-08-17
(`ingestion/manifest.py`, `ingestion/gate.py`, `rag/provider.py`,
`rag/query.py`, `charts/pack.py`, `charts/datasets.py`, `evals/pricing.py`,
`evals/ledger.py`, `evals/scripts/classifier_accuracy.py`, `scripts/*.py`)
plus the test suites for structural issues. Spike prototypes
(`charts/spike/`, `rag/spike_03/`, `ingestion/parse.py`/`chunk.py`/`spike_run.py`)
are exempt per their PROTOTYPE markers.

**Brief:** logic economy only — can the same thing be done more efficiently?
Verbose, human-readable *names* are explicitly preferred and are not
findings. No licensing/safety/§3.4 invariant check is proposed for removal
anywhere in this report: those boundaries are the product.

**Ground rule applied throughout:** every sketch below is behaviour-preserving.
The tests named per finding must stay green, unweakened.

---

## Summary table (ranked by value / effort)

| # | Module | Theme | Est. saving | Risk | Issue |
|---|--------|-------|-------------|------|-------|
| 1 | `rag/provider.py` | Triplicated adapter method wrappers → shared dispatch mixin | ~60–70 lines | Low | #109 |
| 2 | `tests/unit/test_licensing_invariants.py` (+ `test_dataset_pack_manifest.py`) | Refusal-assertion helper; reuse valid-entry builders; module-scoped manifest fixture | ~95 lines | Very low | #111 |
| 3 | `charts/pack.py` | Shared column-check / int-coercion helpers across the six parsers; stale RED-phase docstring | ~35–40 lines | Low | #112 |
| 4 | `evals/scripts/classifier_accuracy.py` | Ratio helper + usage-totals loop in `summarise` | ~25 lines | Very low | #113 |
| 5 | `ingestion/manifest.py` + `ingestion/gate.py` | Small dedupe helpers (`_require_date`, `_strip_or_none` reuse, one 404-tolerant live-fetch helper); stale RED-phase docstring in gate | ~25 lines | Low | #114 |

Total estimated saving: **~240–255 lines** across ~4,600 audited production +
~2,700 audited test lines (≈5% of the audited surface), with no invariant
check touched.

---

## Per-module findings

### 1. `rag/provider.py` — triplicated adapter method wrappers (highest value)

**Location:** `FakeAdapter.generate/structured/plan_chart` (lines 360–388),
`ReplayAdapter.generate/structured/plan_chart` (590–618),
`RecordingAdapter.generate/structured/plan_chart` (851–885).

Nine near-identical wrappers build the same three payload shapes and forward
to a per-class dispatcher (`_next` / `_replay` / delegate-and-`_record`).
The payload construction — the thing that must stay identical across
adapters for canonical request hashing to work — is currently maintained in
triplicate (only `structured` shares `_structured_payload`). The planned
`AnthropicAdapter` (#13) makes it quadruplicate.

**Leaner shape:** one mixin owning payload construction; each adapter
implements a single `_dispatch(method, payload)`:

```python
class _AdapterMethodsMixin:
    def generate(self, messages, documents, config):
        return self._dispatch(
            "generate", {"messages": messages, "documents": documents, "config": config}
        )

    def structured(self, messages, schema, config, system=None):
        return self._dispatch("structured", _structured_payload(messages, schema, config, system))

    def plan_chart(self, request, catalog):
        return self._dispatch("plan_chart", {"request": request, "catalog": catalog})


class FakeAdapter(_AdapterMethodsMixin):
    _dispatch = _next  # (rename)

class RecordingAdapter(_AdapterMethodsMixin):
    def _dispatch(self, method, payload):
        validate_request(method, payload)
        response = getattr(self._inner, method)(**payload)
        self._record(method, payload, response)
        return response
```

**Saving:** ~85 wrapper lines replaced by ~15 mixin lines + 3 one-line
dispatch bindings ⇒ net **~60–70 lines**, and the payload shape becomes
single-source (a hashing-relevant correctness benefit, not just brevity).

**Risk:** low. The §3.4 seam validators (`validate_request` /
`validate_response`) are untouched and still run per adapter (finding #62's
one-validator-one-contract property is *strengthened*: payload shape can no
longer drift between adapters). One new indirection layer (the mixin), paid
for by removing three-way duplication. Tests that must stay green:
`tests/unit/test_provider_adapter.py` (all), `tests/unit/test_query_classifier.py`.

### 2. Test-suite structure — `test_licensing_invariants.py` (and one fixture in `test_dataset_pack_manifest.py`)

**(a) Repeated refusal-assertion tail.** The four-line pattern

```python
with pytest.raises(ManifestError) as excinfo:
    validate_document(entry)
message = str(excinfo.value)
assert "syn-inline-..." in message
assert "field" in message
```

appears ~30 times (e.g. lines 154–159, 167–174, 244–252, 262–270, and
throughout the review-#82 lexical-strictness block). One helper collapses
each body to one or two lines:

```python
def assert_refusal_names(validator, entry, *fragments):
    with pytest.raises(ManifestError) as excinfo:
        validator(entry)
    for fragment in fragments:
        assert fragment in str(excinfo.value), f"refusal must name {fragment!r}"
```

Docstrings (the TDD-plan / review-finding record) stay on each test.
**Saving: ~60 lines.** Risk: very low — assertion strength is identical.

**(b) Full inline valid entries rebuilt by hand.**
`test_open_provisional_requires_licence_note`,
`test_dataset_requires_retrieved_at`,
`test_dataset_rejects_unknown_permitted_context` each write out a complete
~11-line dataset entry when `_valid_dataset_entry(...)` + one or two field
mutations expresses the same single-fault input (and matches the
single-fault fixture discipline merged in #85). **Saving: ~25 lines.**
Risk: very low; makes the mutated-field the visible diff.

**(c) `test_dataset_pack_manifest.py` re-reads the manifest per test.**
`load_dataset_manifest(MANIFEST_PATH)` is called in ~8 tests and
`_raw_manifest()` (a fresh `yaml.safe_load`) in 6 more — 14 parses of the
same committed file. Two module-scoped fixtures (`parsed_manifest`,
`raw_manifest`) make it single-pass and shave a line from most tests.
**Saving: ~10 lines** plus 12 redundant file parses. Risk: very low (the
manifest is never mutated by these tests).

Tests that must stay green: the two files themselves — every test keeps its
name, docstring, and assertion strength.

### 3. `charts/pack.py` — repeated per-parser ceremony

**Location:** the six parsers, lines 123–283.

Each parser repeats two blocks verbatim:

- exact/subset column check + `ValueError` naming the format
  (6 occurrences, ~4–6 lines each);
- the `year_ce → int64` try/except coercion (4 occurrences, ~5 lines each —
  GISTEMP, HadCRUT5, GML, OWID).

**Leaner shape:** two helpers next to the existing `_coerce_float64`:

```python
def _require_columns(df, expected, label, *, exact=True):
    missing_or_wrong = (list(df.columns) != expected) if exact else (set(expected) - set(df.columns))
    if missing_or_wrong:
        raise ValueError(f"{label}: expected columns {expected}, got {list(df.columns)}")

def _coerce_int64(df, column, label, source_name):
    try:
        df[column] = df[column].astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: {source_name!r} did not parse as an integer: {exc}") from exc
```

**Saving: ~35–40 lines.** Risk: low — error-message text is preserved (the
parser-rejection tests in `test_dataset_pack_parsers.py` match on it), and
each parser's docstring keeps carrying the format documentation.

**Considered and NOT recommended:** collapsing all six parsers into a
table-driven spec engine (would save a further ~40–50 lines). The per-format
docstrings and the visible one-function-per-provider structure are what make
a new dataset's parser reviewable against its provider's format notes;
a spec dict would bury exactly the details (na_values, seps, drop rules)
that review #52 showed matter. Dropped per the brief's honesty rule.

**Also in this module:** the module docstring (lines 9–15) still says
"RED phase: every function below is a contract stub raising
NotImplementedError" — stale since the green phase merged. Misleading to
future readers; delete the paragraph.

Tests that must stay green: `tests/unit/test_dataset_pack_parsers.py`,
`tests/unit/test_dataset_pack_fetch.py`, `tests/integration/test_make_datasets.py`.

### 4. `evals/scripts/classifier_accuracy.py` — `summarise` arithmetic ceremony

**Location:** `summarise`, lines 204–331.

The guarded-ratio pattern `sum(...)/len(...) if xs else None` appears five
times (unsafe recall, scope-only recall, self-harm signposting, language,
edge slice), and the four token totals are four separate `sum(...)`
comprehensions over the same predicate.

**Leaner shape:**

```python
def _recall(items):
    return sum(1 for p in items if p.correct) / len(items) if items else None

token_totals = {
    ledger_key: sum(p.usage.get(api_key, 0) for p in predictions if p.usage)
    for ledger_key, api_key in _USAGE_KEYS  # 4 pairs, incl. cache_* renames
}
```

**Saving: ~25 lines.** Risk: very low — pure arithmetic, pinned by
`tests/unit/test_classifier_accuracy.py` (which must stay green, especially
`test_summary_gates_on_self_harm_subtype` and
`test_accuracy_summary_carries_usage_totals`).

### 5. `ingestion/manifest.py` + `ingestion/gate.py` — small dedupe helpers

These are the licensing modules, so only mechanical dedupe that leaves every
check individually visible is proposed. Nothing here removes or merges an
invariant.

**(a) `manifest.py`: duplicated `retrieved_at` block** (lines 443–448 and
562–567 — identical 6-line require-then-parse-date). One helper:

```python
def _require_date_field(entry, field, violations):
    raw = entry.get(field)
    if not raw:
        _missing(violations, field)
        return None
    return _parse_date(raw, field, violations)
```

**(b) `manifest.py`: strip ceremony not using its own helper** — lines
535–540 hand-strip `licence_note` / `licence_evidence`; `_strip_or_none`
(line 243) already expresses it. Two lines each.

**Saving (a+b): ~12 lines.** Risk: very low; refusal messages unchanged.

**(c) `gate.py`: three near-identical live fetchers** — `_fetch_openalex_live`
/ `_fetch_crossref_live` / `_fetch_unpaywall_live` (lines 577–607) share the
try/HTTPError-404→None/raise skeleton. One helper:

```python
def _get_json_or_404_none(url, *, headers=None):
    try:
        return _http_get_json(url, headers=headers or {})
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
```

leaving each source's URL construction and the Unpaywall email requirement
as visible 2–4-line functions. **Saving: ~12 lines.** Risk: low — this is
the live-only edge no pytest tier reaches; behaviour identical.

**(d) `gate.py`: stale RED-phase docstring** — lines 3–7 still claim "every
function below raises NotImplementedError". Delete the stale paragraph
(same class of fix as pack.py's).

Tests that must stay green: `tests/unit/test_licensing_invariants.py`,
`tests/unit/test_manifest_schema.py`, `tests/unit/test_licensing_gate.py`,
`tests/integration/test_gate_cli.py`, `tests/integration/test_make_corpus.py`.

---

## Leave-alone list — looks verbose, is load-bearing

- **`ingestion/manifest.py` — the explicit per-field sequences in
  `validate_document` / `validate_dataset`.** A declarative field-spec table
  could halve them (~80 lines), but these functions *are* the §2.1 legal
  wall: an auditor must be able to read each invariant check as a plain
  statement, with its review-finding comment attached to the exact line.
  A table indirection would save lines and cost exactly the auditability the
  module exists for. Dropped.
- **`ingestion/manifest.py` — `check_prepared_text_shipping`'s two-arm
  structure and `find_committed_data_files`'s marker taxonomy.** Fail-closed
  shipping invariants (reviews #77/#83); every branch is a distinct legal
  case and reads as one.
- **`rag/provider.py` — `validate_request`/`validate_response` running at
  every adapter, including twice on a recorded call.** Deliberate
  defence-in-depth (finding #62: the fakes can never be laxer than live).
  Not over-defensiveness; do not consolidate away.
- **`rag/provider.py` — `FakeAdapter`'s deep-copy of recorded payloads and
  `StructuredResult`'s Mapping ceremony.** Findings #69 and #92
  respectively; both are behaviour consumers rely on.
- **`rag/query.py` — the six explicit branches of `route_classification`.**
  A scope→kwargs table saves ~20 lines but hides which flags
  (`voices_bias`, `tone_flag`, `exclude_from_harvest`, canned text) each
  route carries — the §3.1 routing contract currently reads at a glance,
  branch by branch, including the defence-in-depth unsafe-subtype check.
  The saving is not worth the opacity. Dropped.
- **`rag/query.py` — `parse_classifier_output`'s field-by-field validation.**
  Same reasoning as the manifest validators: each malformation case carries
  its finding reference (#86/#87) and its own error semantics
  (`exclude_from_harvest` differs per case). Not collapsible without loss.
- **`charts/datasets.py` — `fetch_all` walking the raw YAML instead of
  reusing `load_dataset_manifest`.** Looks like a duplicated manifest walk;
  it is not: the pack-level fields (`parser`, `time_axis`, `coverage`) are
  deliberately not carried on `DatasetRecord`, and `validate_pack_entry`
  still delegates the §2.1 schema gate to `validate_dataset` (reuse, not
  re-implementation). The single re-read is the cost of keeping issue #5's
  typed record clean.
- **`evals/pricing.py`, `evals/ledger.py`, `scripts/make_corpus.py`,
  `scripts/make_datasets.py`, `scripts/publish_issues.py`.** Already
  minimal; the make_corpus exit-code taxonomy and temp-file/verify/rename
  dance are review findings #80/#81, not bloat.
- **`tests/unit/test_licensing_invariants.py` — the individually named
  document-refusal tests alongside the parametrized
  `test_refusal_messages_name_document_and_field`.** The seven single tests
  are ~90 collapsible lines, but each is a named TDD-plan item
  (IMPLEMENTATION.md §7: acceptance criteria are *named* tests, visible in
  branch history as red→green). Collapsing them erases that record for a
  saving with no behavioural value. Left alone deliberately.
- **`tests/unit/test_provider_adapter.py` and
  `tests/unit/test_query_classifier.py`.** Both already use module-level
  payload constants, local builders (`_answer`, `_cited_doc`, `_output`) and
  conftest fixtures; the per-test FakeAdapter constructions that remain each
  program a *different* response sequence and are the test's substance, not
  setup bloat. No structural findings.
