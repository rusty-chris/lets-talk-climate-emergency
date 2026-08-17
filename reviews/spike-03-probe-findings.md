# Spike #3 — minimal retrieve→rerank→native-citations loop + 20-question probe: findings

**Issue:** #3 (Phase 0 spike / de-risk). **Author:** implementer session, 2026-08-16.
**Design refs:** DESIGN §3.2 (retrieval), §3.3 (grounded generation with inline
citations), §3.4 (native-citations constraints), §10 Phase 0 gate; the dev cost
plan `reviews/dev-cost-plan-2026-08.md` (binding cost rules, $1.00 cap on #3);
builds on `reviews/spike-02-parsing-findings.md` (the chunk corpus).

**Purpose.** Prove the core mechanism — Claude native citations over
custom-content document blocks — end to end, *before* anything is productionised
(#7/#12/#13/#14). The 20-question probe *is* the gate.

**This is spike code**, under a clearly-marked path (`rag/spike_03/`), imports its
heavy deps lazily, and is not imported by anything shippable. Productionisation
starts from red tests as usual (IMPLEMENTATION.md §2 item 6).

---

## GATE VERDICT

**GATE (DESIGN §10): ≥18/20 questions cite the correct source blocks → PASS at 20/20.**

Every one of the 20 questions produced a native citation whose custom-content
`document_index` resolves back to the committed **gold** source chunk for that
question (the chunk that actually contains the answer). Zero retrieval misses:
bge-m3 + bge-reranker-v2-m3 ranked the gold chunk **#1 of the top-8** for all 20.

The gate is the per-question evidence below, not test coverage (issue #3
acceptance). Grading rule (committed in `rag/spike_03/probe.py` /
`run_probe.py`): a question passes iff the answer carries ≥1 citation **and** at
least one cited block's `document_index` maps to a gold chunk.

### Per-question results (cited block ids vs expected)

Chunk ids abbreviated (drop the `nca5_ch2::` / `esd_tipping_review::` prefix).
Documents are presented to the model in reranked order, so `document_index 0` is
the top-ranked chunk; the gold chunk was rank #1 for every question, so each
correct citation points at `document_index 0` (q19/q20 add extra citations to
adjacent same-topic blocks — still all resolving).

| Q | Question (abbrev) | Expected source block(s) | Cited block(s) | Rerank #1 score | Result |
|---|---|---|---|---|---|
| q01 | global 2012–2021 warming vs preindustrial | c0019 | c0019 | 0.997 | PASS |
| q02 | Alaska warming since 1970 | c0023 | c0023 | 0.996 | PASS |
| q03 | US coastline sea-level rise past century | c0026 | c0026 | 1.000 | PASS |
| q04 | how far back ice cores reconstruct GHGs | c0022 | c0022 | 0.999 | PASS |
| q05 | largest single-country annual CO₂ emitter | c0015 | c0015 | 1.000 | PASS |
| q06 | how long CO₂ lingers | c0014 | c0014 | 0.964 | PASS |
| q07 | confidence: heatwaves worse in West since 1980s | c0034 | c0034 | 0.995 | PASS |
| q08 | billion-dollar disasters in 2022 | c0036 | c0036 | 1.000 | PASS |
| q09 | Bering Sea sea-ice record-low year | c0030 | c0030 | 0.996 | PASS |
| q10 | number of drought definitions | c0041 | c0041 | 0.989 | PASS |
| q11 | Paris Agreement warming limit (verbatim) | c0046 | c0046 | 0.953 | PASS |
| q12 | era of most extreme US heatwaves on record | c0037 | c0037 | 0.989 | PASS |
| q13 | three core tipping elements | c0009 | c0009 | 0.988 | PASS |
| q14 | Eocene–Oligocene transition age | c0031 | c0031 | 0.999 | PASS |
| q15 | what is the Grande Coupure | c0031/c0032 | c0032 | 0.989 | PASS |
| q16 | when Bølling-Allerød began | c0034 | c0034 | 0.908 | PASS |
| q17 | AMOC-collapse effect on NH temperature | c0014 | c0014 | 0.999 | PASS |
| q18 | most important interannual variability mode | c0021 | c0021 | 0.997 | PASS |
| q19 | Dansgaard-Oeschger events | c0035 | c0004; c0035 | 0.979 | PASS |
| q20 | permafrost hydrological change | c0027 | c0027 | 0.996 | PASS |

Machine-readable evidence (gitignored, reproducible): `data/spike/spike03_graded.json`
(per-question verdict), `data/spike/spike03_results.json` (raw answer text +
citation objects incl. `document_index`, `start/end_block_index`, `cited_text`),
`data/spike/spike03_retrieval.json` (top-8 + rerank scores per question).

**Calibrated language preserved verbatim** (DESIGN §3.3 rule 3) — spot checks:
- q07 → *"…states with **high confidence** that heatwaves have become more common
  and severe in the West since the 1980s."*
- q11 → *"…limiting global warming to **'well below 2°C'** relative to preindustrial
  temperatures, **preferably to 1.5°C**."*
- q03 → *"…risen by **about 11 inches**, which is considerably more than the global
  average sea level rise of **7 inches**."*

No qualifier was upgraded, downgraded, or dropped in any of the 20 answers.

---

## Method (what was actually run)

Pipeline (`rag/spike_03/`), over the two real spike documents from #2:

1. **Chunk corpus** (`chunk_corpus.py`): reran #2's committed prototype parser
   (`ingestion.parse`, Docling) + chunker (`ingestion.chunk`) verbatim. Reproduced
   #2's counts **exactly** — NCA5 n=117, ESD n=72, total **189 chunks** — confirming
   deterministic reproduction. Source PDFs re-fetched to gitignored `data/spike/`;
   **sha256 verified byte-identical** to #2's recorded values (below).
2. **Embed + retrieve** (`run_probe.py`): `bge-m3` dense embeddings (local,
   FlagEmbedding), naive cosine **top-40** retrieval. (Spike scope says "naive
   top-k"; production §3.2 is hybrid dense + learned-sparse RRF — deferred to #14.)
3. **Rerank**: `bge-reranker-v2-m3` cross-encoder → **top-8** (sigmoid-scored).
4. **Generate**: one **custom-content document block per top-8 chunk**, all with
   `citations: {enabled: true}` (all-or-none), **no structured output** on the
   generation call (§3.4), `claude-haiku-4-5`. `document_index 0..7` == reranked
   rank; the chunk id is carried in each block's `context` (not cited) for
   traceability.
5. **Resolve + grade**: parse each answer's `content_block_location` citations,
   map `document_index` → chunk id, compare to gold.

### Source documents (NOT committed — gitignored under `data/spike/`)

| doc_id | sha256 | licence |
|---|---|---|
| `nca5_ch2` | `90298e25aee94684334b7964c61e030854bd250107037ec23babf3fac90b243e` | Public domain (US Govt work, NCA5) |
| `esd_tipping_review` | `39906d865f171139878eada6b5825ec02e7e43da1d09248d94918d7ea8b75013` | CC-BY 4.0 (Copernicus ESD) |

Both match `reviews/spike-02-parsing-findings.md` exactly. Reproduce with:
`uv run python -m rag.spike_03.chunk_corpus` then
`uv run python -m rag.spike_03.run_probe --run` (batch) or `--run --live`.

---

## §3.4 native-citations constraints — observed LIVE

These become the contract tests in #12 (against the request builders) and inform
#24's replay fixtures. Both deliberate-400 probes were **not billed** (rejected
requests aren't billed). Verbatim API responses:

1. **Structured output on the citations call → 400.**
   > `invalid_request_error: Citations cannot be enabled when output format is set.
   > Please disable citations on uploaded document blocks.`
   Confirms DESIGN §3.4: the generation call must never carry
   `output_config.format` while citations are enabled — hence the separate
   classifier/rewriter/chart-spec structured calls.

2. **Mixed cited / uncited document blocks → 400 (all-or-none).**
   > `messages.0.content: Citations must be either enabled or disabled on all
   > `document` blocks. A mixture of enabling and disabling is not supported at this
   > time.`
   Confirms §3.4 all-or-none. The §3.2 **structural voices filter** must therefore
   remove voices chunks *before* the generation call (you cannot leave a voices
   block in with citations off).

3. **Document-count handling.** The probe sent exactly **8** documents per request
   and `document_index` resolved cleanly across the full 0–7 range on every
   question. The API imposes no hard 8-doc limit itself; the ≤8 bound is *our*
   design rule (reranked top-8), to be enforced in the request builder (#12
   contract test), not relied on from the API.

### Mechanism observations for #7 / #12 / #13 / §3.6

- **Custom-content citation granularity = the whole content block.** With one
  content block per chunk, every citation's `cited_text` is the **entire chunk
  text** (`start_block_index=0`, `end_block_index=1`) — there is no sub-sentence
  span. The citations docs confirm custom-content blocks are used *as-is with no
  further chunking*. Implication for §3.6 "cited-span highlighting": to get
  sentence-level cited spans, either (a) pass each chunk as a **plain-text**
  document (auto sentence-chunked, char-index citations) or (b) split each chunk
  into multiple content blocks. The one-block-per-chunk shape proven here gives
  **chunk-level** provenance, which is sufficient for the gate but coarser than
  span highlighting. **Decision for #12/#14 to make explicitly.**
- **Context header leaks into `cited_text`.** Because #2's chunker prepends a
  `"Title → Section"` context header *inside* the chunk text, and that text is the
  citable block, every `cited_text` begins with the header. For production, move
  the context header to the block's `context` field (passed to the model, not
  cited) so citations quote only the evidence body. Recorded as a #7/#14 test case.
- **Response shape** (for #24 replay fixtures): cited answers return multiple
  `text` blocks; cited blocks carry a `citations` list of
  `type: "content_block_location"` objects with `document_index`,
  `document_title`, `start_block_index`, `end_block_index`, `cited_text`.
  `cited_text` is not billed as output tokens.

### Qualifier-splitting observations (for #7's chunker tests)

No harmful qualifier-splitting was observed in the answers (calibrated phrases
came through intact, above). One retrieval-relevant note: the #2 chunker's
one-sentence overlap duplicated a fact across adjacent chunks in exactly one probe
case (q15 Grande Coupure — the intro sentence lands in both `esd…c0031` tail and
`c0032` head), which is why q15's gold set has two members. This is benign for
citations (either resolves correctly) but confirms #7 should make overlap
semantics explicit.

---

## Cost discipline (binding — `reviews/dev-cost-plan-2026-08.md`)

**Model:** `claude-haiku-4-5` only. **Cap:** $1.00 (issue #3). Pricing (per MTok):
Haiku $1 in / $5 out; Batches API = 0.5×.

**Pre-submit estimate (shown before any billed call).** The free `count_tokens`
endpoint was returning `500 Internal Server Error` throughout this session (a
transient server-side incident affecting `count_tokens` **and** the Batches API —
see Deviations), so the estimate used a conservative local char-based token count
(`≈ chars/3.5 + 350` framing tokens/request):

```
input  ≈ 118,982 tokens ;  assumed output ≈ 6,820 tokens (generous: max_tokens/3 × 20)
batched: (118,982×$1 + 6,820×$5)/1e6 × 0.5  = $0.0595 + $0.0170 = $0.0765
live   : (118,982×$1 + 6,820×$5)/1e6 × 1.0  = $0.1190 + $0.0341 = $0.1531
```

$0.153 (live worst case) ≤ $1.00 → **proceeded** (the runner STOPs and reports if
the estimate exceeds the cap).

**Actual usage (from `response.usage`), 20 live calls:**

```
input_tokens = 135,570 ;  output_tokens = 1,456  (answers are concise)
cost = (135,570×$1 + 1,456×$5)/1e6 × 1.0 = $0.13557 + $0.00728 = $0.14285
```

**Actual spend: $0.1429** (well under the $1.00 cap). Logged to
`evals/spend-ledger.csv` (created with the spec header; `activity=spike-probe`,
`mode=live`, `cumulative_usd=$0.14285`). Actual input (135.6k) exceeded the local
estimate (119k) because the citations feature adds a system-prompt/chunking
overhead the char heuristic under-counts — the estimate stayed the right side of
the cap regardless.

---

## Deviations

1. **Batches API unavailable → live fallback (noted per cost-plan M3).** The
   Batches API `messages.batches.create` returned `500 Internal Server Error` on
   every attempt across the whole session (≥6 tries over several minutes, incl. a
   trivial 1-request batch), while plain `messages.create` worked perfectly. The
   cost plan permits live calls "unless a technical blocker makes it impractical
   (then note why)" — this is that blocker. The 20 probe calls therefore ran
   **live (non-batch)** at full price. Cost impact: $0.143 vs ~$0.072 batched —
   both far under the $1.00 cap. The runner keeps the batch path as default
   (`--run`) and the live path behind `--run --live`; re-run under batch once the
   API recovers to regenerate batched fixtures for #24.
2. **`count_tokens` unavailable → local estimate.** Same incident; the pre-submit
   estimate used a conservative local char-based count instead. Actual token usage
   was taken from `response.usage` and is authoritative in the ledger.
3. **`.secrets` is a bare API key** (single line, `sk-ant-…`, no `NAME=`), so it
   was loaded as `ANTHROPIC_API_KEY="$(cat .secrets)"` rather than sourced — the
   file format differs from the letter's example `grep`/`source` recipe. The key
   was never printed, echoed, logged, or committed (`.secrets` is gitignored; the
   session used it only via the env var).
4. **Reranker via `transformers` directly, not `FlagReranker`.** FlagEmbedding's
   `FlagReranker` calls the removed `tokenizer.prepare_for_model` and 400s under
   the installed `transformers==5.15`. Swapped to
   `AutoModelForSequenceClassification` + `AutoTokenizer` with `sigmoid(logit)`
   scores — identical model (`BAAI/bge-reranker-v2-m3`) and semantics to the model
   card; bge-m3 dense embeddings still use FlagEmbedding's `BGEM3FlagModel`.
5. **Spike deps not pinned in `pyproject.toml`.** `docling`, `pymupdf`,
   `FlagEmbedding`, `anthropic` were installed ad hoc for the spike (as in #2);
   #7/#12/#14 own productionising them. Models (`bge-m3`, `bge-reranker-v2-m3`,
   Docling layout/OCR) download from HuggingFace on first use and are kept out of
   git.
6. **No GPU per `nvidia-smi`** (NVML driver/library mismatch), but PyTorch's CUDA
   backend worked — embeddings and reranking ran on GPU; the mismatch only affects
   the `nvidia-smi` CLI.
