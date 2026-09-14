# Post-Launch Iteration Log — 2026-09-13/14

*A teaching narrative of the first 48 hours after "Let's Talk About the Climate
Emergency" went live. Written for future reference: for each problem it records
the symptom, what we tried, the numbers we measured, the decision we took, the
alternatives we rejected **and why**, and the lesson. The value is in the
reasoning and the data — the changelog is the least of it.*

---

## Context — where we started

On 2026-09-13 the site launched as **v1.1.2-launch** at
[climateemergency.chat](https://climateemergency.chat): a non-commercial,
educational, privacy-first RAG chatbot on a **Hetzner CX33** box
(~£6/mo infra; the service caps at ~£10/mo on a cost-recovery model).
Architecture per the ADRs: hybrid retrieval (dense bge-m3 + sparse, fused by
RRF into a top-40) → cross-encoder reranker → **Haiku** generation, with a
structured refusal path so the bot declines honestly when the corpus doesn't
cover a question. Everything runs **on-box** — no off-box inference — because
the privacy model is load-bearing for the product's credibility.

The owner began real-world testing immediately and surfaced a sequence of
problems. This log covers the fixes shipped as **v1.1.3 → v1.1.6** plus the
retrieval-quality investigation still in flight at the end of the session.

The whole run was executed as an orchestrator + Opus subagents (Fable was
rate-limited, so Opus per the standing model policy), with **data-driven
decisions throughout**: offline benchmarks, LLM-judge eval gates, and
adversarial reviews. Each fix shipped as a small PR with a tagged release and
end-to-end wire verification on the box.

---

## 1. The footer crash — a green test that pinned a bug (#381 → v1.1.3)

### Symptom
The very first page load of the live site **500'd**. The owner reported a
`streamlit.runtime.media_file_storage.MediaFileStorageError` from the footer
render:

```
Error opening '<!-- Rusty Data brand mark ... <svg ...> ... '
  File "/app/ui/app.py", line 93, in _render_footer
    st.image(str(STEWARD_MARK_PATH), width=20)
```

### Investigation
The crashing call was `st.image()` on a **local `.svg`** file. Under the pinned
Streamlit range (`>=1.38,<2`), `st.image` reads the SVG *markup* into memory
and then the media-file store tries to `open()` that markup **as a filename** —
which of course fails. `st.image` simply cannot render a local `.svg` in this
version.

The damning detail: the shell-hygiene test suite **asserted that `st.image`
was used** for the mark (`test_shell_renders_the_mark_via_st_image`). The test
pinned the broken call in place, so the bug shipped **green**.

### Decision
Replace the media-file round-trip entirely. A new pure helper
`ui.footer.steward_mark_img_tag()` base64-encodes the checked-in asset into a
self-contained `data:image/svg+xml` `<img>` tag, rendered inline on the ADR-018
credit line via `st.markdown(..., unsafe_allow_html=True)`. The asset is
repo-controlled and self-contained (no scripts, no external refs), so inlining
raw HTML adds no injection surface. ADR-018 pins (the credit / non-commercial
pairing and the rustydata.ai link) were left untouched.

We verified **end-to-end through Streamlit's `AppTest` runtime** (the real
media-file layer), not just at the unit level:

| Path | Result |
|---|---|
| OLD `st.image(local .svg)` | RAISES `Error opening '<!--...` (reproduces the live crash) |
| NEW data-URI `<img>` via `st.markdown` | no exception; emits the `<img>` |

The hygiene guard was rewritten to pin the **data-URI/markdown contract and
forbid `st.image` for the mark**. Opus adversarial review = **PASS** (no
blockers/majors; 4 minor nits, e.g. tests prove the helper's string shape not
the render, and a missing asset would still crash — a test-hardening follow-up).
Tagged **v1.1.3-launch**, redeployed UI-only (`--no-deps`), verified: UI
healthy, render-proof through the running image, public HTTPS 200, zero
`MediaFileStorageError` in logs.

### Alternatives considered & rejected
- **Convert the SVG to PNG and keep `st.image`.** Rejected: loses vector
  crispness at footer scale and still keeps the fragile media-file path for a
  trivial 20px mark.
- **Drop the mark.** Rejected: ADR-018 requires the "Built by Rusty Data"
  credit to be present and paired with the non-commercial note.

### Lesson
**A test that pins broken behavior is worse than no test** — it manufactures
false confidence and lets the bug ship green. Assert the *outcome the user
sees* (does the mark render?), not the *mechanism* (is `st.image` called?).
And verify UI fixes through the framework's real runtime, not a mock that can't
reproduce the failure.

---

## 2. The latency crisis — root-causing "unusable" by elimination

### Symptom
Novel (free-text, non-starter) questions took **~2 minutes** to answer, and
sometimes appeared to hang. The owner's verdict: **"unusable."**

### Investigation — measure each stage, eliminate one at a time
Rather than guess, we instrumented and eliminated candidates on the box:

- **Not the LLM.** Generation streams in ~1s once it starts.
- **Not hanging.** No errors, no retries, no stuck connections in the logs.
- **Not memory.** No swap; RAM was fine.
- **The SSE stage timing exposed it:** a ~110s wall sat between the request and
  "retrieval-complete", and only then did generation stream fast.

Isolating that wall pointed straight at the **CPU cross-encoder reranker**.
Measured directly in the api container:

| candidates reranked | wall time |
|---|---|
| 1 | ~4.0 s |
| 8 | ~33 s |
| 40 (`DEFAULT_TOP_K`, the default) | **~159 s** |

The reranker was the **entire** wall: ~4s/candidate × 40 candidates ≈ 110–160s.

The kicker: **ADR-006 had budgeted "~100 ms on CPU for 40 pairs"** for this
exact step. On the CX33's CPU, running the 560M `bge-reranker-v2-m3` with #175
windowed full-coverage scoring, it was off by **~1,600×**.

### Lesson
**Measure each pipeline stage before theorising.** A one-line stage timing
found in minutes what speculation would have chased for hours. And **a
documented performance budget written for a fast machine can be catastrophically
wrong on the deploy hardware** — ADR-006's "~100 ms" was plausible on a
developer GPU/fast CPU and lethal on a €6/mo shared vCPU. Treat budgets as
hypotheses to re-measure on the target, not as facts.

*(This crisis drove the next three fixes: the starter-cache carve-out (#382)
for the flagship questions, and the reranker swap (#383) for everything else.)*

---

## 3. Starters slow AND serving a bad decline (#382 → v1.1.4)

### Symptom
The 13 flagship "starter" questions were **both** slow (they hit the ~2-minute
reranker wall) **and** the flagship *"Why are scientists calling this an
emergency?"* served an unhelpful **live-generated decline** — worse than the
curated cache answer that already existed for it.

### Investigation — two bugs, one screen
1. **Starters only served the curated cache in *paused* mode.** The curated,
   pre-vetted starter answers (decision-6 carve-out) were served when the site
   was paused, but the **live** path ran the full pipeline for them anyway — so
   every starter paid the full reranker cost.
2. **Declines are non-cacheable**, so the semantic cache never warmed for the
   flagship: a live decline can't be cached, meaning the site re-generated the
   same bad decline every time.

### Decision
Extend the existing paused-mode carve-out to the **live path**: an exact
canonical starter match on a first turn serves the curated, pre-vetted answer
with **zero adapter calls** — instant and deterministic. Gated behind a new flag
**`CLIMATE_CHAT_LIVE_STARTER_CACHE`** (default **ON**), plumbed through
docker-compose. The replay/smoke stack pins it **OFF** so the live
retrieval/chart **wire** is still exercised by a starter probe.

Result: **flagship ~110s → ~1s.** Full unit suite: 2594 passed, 3 skipped; CI
all-green including smoke. Tagged **v1.1.4-launch**.

### Alternatives considered & rejected
- **Rework the replay-fixture seeder so smoke could use the cache directly.**
  Rejected: the smoke stack *already* disables the semantic cache the same way
  (a default-on flag pinned off in test), so a new flag **mirroring that exact
  pattern** was far lower-risk than reworking test infrastructure. We chose the
  flag whose shape the codebase already trusted.
- **Just fix the reranker and let starters run the full pipeline.** Rejected as
  the *sole* fix: even a fast reranker leaves starters slower and
  non-deterministic than serving vetted editorial answers, and the flagship
  answer quality was a separate problem (see §7). The cache carve-out is the
  right home for curated, high-traffic questions regardless of reranker speed.

### Lesson
**A config flag that mirrors an existing, trusted pattern beats reworking test
infrastructure.** When you need to special-case a path in production but keep
it exercised in test, copy the mechanism your codebase already uses rather than
inventing a new seam.

---

## 4. The reranker swap — the real fix for novel questions (#383 → v1.1.5)

### Symptom
Every novel/free-text question — the ones the starter cache can't cover — still
took ~2 minutes because of the 560M `bge-reranker-v2-m3` cross-encoder on CPU.

### Investigation — an offline, $0 quality+latency benchmark
We built an **offline benchmark** (later filed as #391) that costs nothing to
run: index the real corpus (795 chunks), retrieve → rerank → compare the top-8
chunk-ids against the eval gold, and report **recall@8 / MRR / nDCG@8** per
candidate reranker **plus per-model latency** for 40 candidates.

| model | params | latency / 40 | recall@8 | MRR | nDCG@8 |
|---|---|---|---|---|---|
| bge-reranker-v2-m3 (current) | 560M | 32.6 s | 0.640 | 0.373 | 0.422 |
| bge-reranker-base | 278M | 11.1 s | **0.240** ❌ | — | — |
| **ms-marco-MiniLM-L-6-v2** | 22M | **2.24 s** | 0.600 | 0.376 | 0.418 |
| ms-marco-MiniLM-L-12-v2 | 33M | 4.28 s | 0.560 | 0.405 | 0.432 |

*(The offline harness measured 32.6s/40 for bge; the live in-container figure
was worse (~159s) because production runs #175 windowed multi-pass scoring over
longer chunks. Both point the same way.)*

Reading the table:
- **`bge-reranker-base` collapses** — recall@8 0.240. A smaller model from the
  *same family* is not automatically a safe downgrade; it was **rejected**
  outright despite being 3× faster.
- **`ms-marco-MiniLM-L-6-v2` is the winner:** **14.5× faster** than bge, with
  recall@8 **within one item of baseline** (0.600 vs 0.640 on 25 gold items)
  and MRR/nDCG tied.
- **L-12** is more accurate on ranking metrics but slower and *lower* on
  recall@8 — not worth it for this budget.

### Decision
Swap to **`cross-encoder/ms-marco-MiniLM-L-6-v2`**. The scoring contract is
unchanged: single-logit `sigmoid(logit)` giving query-comparable scores, same
512-token windowed coverage (#175), same revision-pin discipline. This was
**low-risk** because — post-#313 — the reranker is a **soft, disabled-in-prod
pre-filter**, not the authoritative refusal gate (the structured
generation-level decline is; prod runs `refusal_threshold=None`). A small
recall@8 change does not move the honesty guarantee. Caveat: MiniLM-L-6 is
**English-only**, acceptable because the scope classifier routes non-English
queries away before retrieval.

Result: **novel questions ~160s → ~6s warm** (~93% cut). Tagged
**v1.1.5-launch**. A ~40s one-time cold model-load remains on the first query
after any restart — filed as **#388**.

A retrieval-ceiling diagnostic confirmed the reranker was **never the recall
bottleneck**: most misses are the gold chunk **not reaching the top-40 at all**
(a corpus-coverage limit), not the reranker mis-ordering the pool. Swapping the
reranker was safe on quality precisely because quality wasn't coming from it.

### Lesson
**Benchmark latency AND quality together, on the target hardware, against your
own gold.** The offline harness cost $0 and turned a scary model swap into a
one-line table decision. And **know which component actually owns the metric
you're worried about** — the reranker owned latency, not recall, so trading a
sliver of one for a 14.5× win on the other was obviously correct once measured.

---

## 5. The GPU option — analysed and rejected on the numbers

### Symptom
An obvious "fix" for CPU reranker latency: put the box on a GPU.

### Investigation
Hetzner's cheapest GPU offering, the **GEX44**, is **€184/mo + €79 setup** —
roughly **30×** the current ~€6/mo box. And moving inference to a GPU service
that isn't the app box (or a managed inference API) **breaks the privacy model**
that the product's credibility rests on (on-box inference, nothing leaves the
machine).

### Decision
**Rejected.** The numbers argued *for* the CPU-lightening path (§4), not for
hardware. A 14.5× software speedup on a €6/mo box beat a 30×-cost GPU that also
compromised the privacy story.

### Lesson
Cost and architecture constraints can make the "obvious" performance fix the
wrong one. When the cheap software lever exists, the expensive hardware lever
is usually solving the wrong problem.

---

## 6. The flagship's answer quality — separating model from prompt with a 2×2 (v1.1.6, #394)

### Symptom
Even with a reasonable retrieved passage set, the flagship *"Why are scientists
calling this an emergency?"* served an **unhelpful full decline** — it emitted
the whole `[[NO-ANSWER-DECLINE]]` marker when it actually had consensus severity
findings in hand **plus** Hansen's attributed "climate emergency" framing.

### Investigation — a controlled 2×2 experiment
The tempting move is "use a smarter model." Instead we ran a **2×2**: {shipped
prompt, tuned prompt} × {Haiku, Opus}, on the **real retrieved passages** for
the flagship. The result cleanly separated the two variables:

- **It was NOT the model.** Given a reasonable passage set, **even Haiku
  answers** — it assembles a grounded, honestly-attributed answer with no
  fabrication.
- **It WAS the prompt.** The shipped prompt **over-declined**: it treated
  "no single passage restates the *framing*" as grounds to decline, even though
  the passages carried the *findings* (human-caused rapid warming, present
  impacts, tipping thresholds) and an attributed emergency statement.

### Decision — three surgical prompt additions (PR #394)
Facts stay 100% cited; ADR-018 and every existing rule intact.
1. **Rule 5 — assemble, don't decline.** Thematic "why / how bad" questions are
   answered by **assembling** the cited severe findings into a severity-led
   synthesis, not declining for want of a passage that restates the framing.
   (This is reporting what each passage states, not gap-filling by inference,
   which Rule 1 still forbids.)
2. **Rule 5 — the Socratic close (owner's design ruling).** Present the cited
   findings, then **invite the reader to weigh them** with a genuine question
   ("…is that a situation you would treat as anything less than an emergency?")
   — never assert an uncited conclusion. "This is an emergency" is a *judgement*,
   not a *finding*; asserting it uncited would break Rule 1 and the reader's
   trust. Drawing the conclusion **with** the reader is both honest and
   **stronger than an assertion** — it preserves the credibility model.
3. **Rule 8 — attributed beyond-range framing is usable evidence.** An
   attributed, beyond-assessed-range "emergency" statement (e.g. Hansen's) is
   citable evidence to report *as attributed* alongside the consensus, not a
   reason to decline.

### Results — eval gates held
Validated on the real Haiku pipeline (48 gold items × 2 prompts, ~$0.75):

| gate | result | threshold |
|---|---|---|
| refusal_gate | **20/20 = 1.00** | ≥ 0.90 |
| false_refusal_gate | **0/25 = 0.00** | ≤ 0.05 |
| dangerous over-answers | **0** | 0 |
| behaviour vs shipped (gate subset) | neutral (no regression) | — |

**The Socratic close lands reliably on Opus, not Haiku.** Two Haiku runs
confirmed: Haiku assembles and attributes but **omits the closing question**;
sharpening the directive to force it merely **regressed Haiku into hedging**.

### Resolution — a model split, not a model switch
Live generation **stays Haiku** (cost — see §8). The **starter cache is
regenerated with Opus** (a one-time **$0.31**, prompt-cached, `max_tokens=2048`)
so the flagship and other cached answers carry the full Socratic treatment,
served instantly at zero ongoing cost. After the fix, **11/13 starters answer**
(the launch had 3 honest-decline starters); the two remaining are known corpus
gaps (the emergency-response question, a full decline, and get-involved, a
partial). Tagged **v1.1.6-launch**.

### Alternatives considered & rejected
- **Switch live generation to Opus to get the Socratic close everywhere.**
  Rejected on cost (§8) — recorded as a future option in #390.
- **Force the Socratic close on Haiku with a sharper directive.** Rejected:
  measured to *regress* Haiku into hedging. The capability isn't there to
  command into existence.
- **Fix it by improving retrieval / swapping the model first.** Rejected as the
  primary fix: the 2×2 proved the passages were already good enough and Haiku
  was already capable — the defect was the prompt over-declining. (Retrieval is
  still the right lever for *other* questions — see §10.)

### Lesson
**Separate the model-capability question from the prompt/retrieval question
with a controlled experiment** (here a 2×2) before spending money on a bigger
model. The 2×2 showed the fix was a free prompt edit, not an expensive model
upgrade. And on product philosophy: **"draw the conclusion WITH the reader, not
FOR them"** resolves the advocacy-vs-credibility tension — for a truth-critical
product, an invitation grounded in cited evidence is more persuasive *and* more
honest than an uncited assertion.

---

## 7. The Opus cost analysis — why Haiku stays live (#390)

### Symptom
If Opus writes better answers (§6), why not run it live for everything?

### Investigation — measured per-question cost
Opus 4.8 is priced at **$5 / $25 per MTok** (input / output). Measured
per-question cost:

| configuration | cost / question | questions/mo at £10 |
|---|---|---|
| Haiku (current live) | ~$0.012 | (well within cap) |
| Opus for generation only | ~$0.045 | ~275 |
| Opus for the whole pipeline | ~$0.069 | ~180 |

At £10/mo cost-recovery, Opus-live supports only ~180–275 questions/month, and
the **daily cap would pause the site after ~6–10 Opus questions/day** — it
breaks the cost-recovery model at any real traffic.

### Decision
**Haiku stays live; Opus is used only for the one-time starter-cache
generation.** A **selective-escalation** middle path — Haiku by default,
escalate to Opus only when Haiku's answer is thin or declines — is recorded as
the cheap way to capture most of the quality lift if the budget ever rises.
Also noted: the retrieval upgrade (§10) lifts Haiku's answers **at Haiku cost**,
which is the better lever regardless of model.

### Lesson
**Put a number on it before committing to an expensive default.** "Opus is
better" is true and irrelevant until you know it costs 4–6× per question and
would pause the site after ~8 questions a day. The one-time cache captures the
quality where it matters (curated flagship answers) at negligible ongoing cost.

---

## 8. VC cleanup — the missing teardown owner (#393 → #395)

### Symptom
The autonomous build had left **~300 stale local branches** and **~150 leftover
worktrees** under `.claude/worktrees/`.

### Investigation — root cause, not just cleanup
The per-issue orchestration loop creates an isolated **worktree + `issue-<n>`
branch per agent**, but the lifecycle was **create → work → PR → merge →
(nothing)**. Teardown was **nobody's named responsibility** in ORCHESTRATION.md,
CONTRIBUTING.md, or AGENTS.md — so **159 merges left 159+ workspaces behind.**
Compounding factors:
- review branches spawn `-fixes` / `-fixes-impl` pairs;
- rebased duplicates (`*-rebased-local`);
- **stale worktree locks from killed agents** — a lock held by a **dead pid**
  permanently pins the worktree (the harness's own auto-clean only removes
  *unchanged, unlocked* worktrees, so it structurally never reclaims merged
  agent work);
- **GitHub "delete head branch on merge" was OFF** (140 merged branches still
  on origin).

### Decision — ship all three parts (PR #395)
1. **`scripts/git-cleanup.sh`** — an idempotent sweep run at session
   start/end: `git fetch --prune`; release locks held by **dead** pids only;
   `git worktree remove --force` for worktrees whose branch is **merged into
   main**; `git worktree prune`; `git branch -d` (never `-D`) for merged
   branches; optional `PRUNE_REMOTE=1`; report unmerged strays for a human.
   Every deletion is gated by `--merged` / `git merge-base --is-ancestor` /
   `-d`, so it **can never drop unmerged work** (also supports `DRY_RUN=1`).
2. **ORCHESTRATION.md + CONTRIBUTING.md** — post-merge teardown is now a
   **named orchestrator duty** at the merge step (remove the worktree + delete
   the branch, including the `-impl` sibling). The script is the safety net,
   not a substitute.
3. **Enabled GitHub "Automatically delete head branches"** (`delete_branch_on_merge`
   now `true`).

The one-off cleanup this session took **worktrees 150 → 4** and **local branches
~300 → 7**. Acceptance: after a build session, `git worktree list` shows only
the main checkout + live agents, and `git branch --merged main` lists only
`main`.

### Lesson
**Any create-per-unit workflow needs an explicit teardown *owner*, or it
leaks.** The failure wasn't a bug in any step — it was an *unowned* step. Name
the responsibility, automate a safe idempotent sweep as backstop, and turn on
the platform's own garbage collection.

---

## 9. The retrieval-quality investigation — benchmarking hypotheses before building (in progress)

The owner wanted **better evidence retrieval for ALL questions without slowing
it down** — a genuine live-quality lever that lifts Haiku's answers at Haiku
cost. This workstream was still in flight at session close; its value here is
the **method**: benchmark each hypothesis cheaply *before* building it.

### 9a. Reranker speedup via int8 quantization (in progress, feeds #385)
Benchmarked **int8 dynamic quantization** of the MiniLM cross-encoder on CPU:
**~1.43× faster, recall held.** That headroom would let us widen the candidate
net (40 → ~57) within the latency budget — but it **needs the #313 refusal
threshold recalibrated** for the new candidate count first.

### 9b. Multi-query facet expansion — TESTED and REJECTED (#384, closed)
The intuitive fix for broad thematic questions: expand the question into 2–4
facet sub-queries, retrieve for each, union + dedupe, rerank. We benchmarked it
on the box (real ~809-chunk corpus, 25 gold answerable items) **before building
the production path**:

| approach | pool-coverage | recall@8 | retrieval cost |
|---|---|---|---|
| single query (baseline) | 0.80 | 0.600 | 1× |
| multi-query facet union | 0.88 | 0.640 | ~5× + a new Haiku dependency |

**Rejected.** Only **+0.04 recall@8** (one item of 25) for a ~5× retrieval-cost
multiplier. And critically, **the mechanism was wrong**: the bottleneck is *not*
retrieval coverage. The single query already puts gold in the pool 80% of the
time (multi-query lifts that to 88%), but the **MiniLM reranker drops gold out
of the top-8 even when it's in the pool** — pool-coverage 0.88 vs recall@8 0.64
is a **24-point gap the reranker leaves on the table**. Multi-query surfaced
more gold into the pool; the reranker just buried it again.

### 9c. The real levers being benchmarked
- **Raise `GENERATION_TOP_K` 8 → ~12** — recover gold that ranks just outside
  the top-8 (cheap; needs a DESIGN §3.4 ruling).
- **Better fast rerankers** (jina / mxbai / gte class) — close the pool→top-8
  conversion gap, which is where the recall is actually lost.

### Lesson
**Benchmark the hypothesis cheaply before building it.** The multi-query
benchmark cost a few dollars of eval and **saved us from shipping a 5×-cost
change with a new LLM dependency that didn't address the actual bottleneck.**
Measuring *pool-coverage vs recall@8 separately* was what revealed the truth:
gold was reaching the pool fine; the reranker was the leak. You cannot fix the
right thing until you've localised where the metric is actually lost.

---

## Current state (end of 2026-09-14)

### Shipped and live
| release | fix |
|---|---|
| **v1.1.3-launch** | footer crash (`st.image` on local `.svg`) — #381 |
| **v1.1.4-launch** | serve curated starter cache on the live path — #382 |
| **v1.1.5-launch** | reranker swap bge-560M → ms-marco-MiniLM-L-6-v2 — #383 |
| **v1.1.6-launch** | Socratic answer-prompt tune + Opus-regenerated starter cache — #394 |

### What's fast / good now
- **Starters ~1s** (served from the curated cache, live path included).
- **Novel questions ~6s warm** (MiniLM reranker; ~40s one-time cold-load after
  a restart).
- **Flagship "why an emergency"** now serves a proper grounded **Socratic**
  answer — cited findings then a reader-invitation, with "emergency" attributed
  to Hansen (Rule 8), never asserted uncited.
- **11/13 starters answer** (was 3 declines at launch).
- Live generation on **Haiku** (cost-recovery); **Opus** used only for the
  one-time starter-cache regeneration.

### Still open
- **#385** — reranker: further CPU speedup (int8/ONNX, candidate tuning) to
  afford a wider net. *(int8 quant benchmarked at ~1.43×, recall held.)*
- **#386** — evaluate HyDE (hypothetical-document embeddings) for thematic
  questions.
- **#387** — latency knob: trim reranked candidates 40 → ~20 (opposite
  direction to the wider net; decide together).
- **#388** — eliminate the ~40s cold-start on first query after api restart
  (model warmup).
- **#389** — add IPCC AR6 Synthesis Report as a consensus "why it's serious"
  source (licence-gated; the flagship currently rests mainly on single Hansen
  paper).
- **#390** — future option: Opus + Socratic close live across the board
  (recorded; Haiku live for now).
- **#391** — formalise `scripts/bench_reranker.py` as a committed
  retrieval-benchmark tool (currently untracked; fails repo lint E501/B905).
- **In-flight retrieval-quality work** — `GENERATION_TOP_K` 8→12 and alternative
  fast rerankers (the real levers per §9); wider net gated on the #313 threshold
  recalibration.

*(#384 multi-query and #393 VC teardown are **closed** — resolved as documented
in §9b and §8.)*

---

## Methodology lessons, distilled

1. **A test that pins broken behavior is worse than no test.** Assert the
   outcome the user sees, not the mechanism.
2. **Measure each pipeline stage before theorising** — stage timing found the
   latency wall in minutes.
3. **A documented performance budget written for fast hardware can be off by
   ~1,600× on the deploy box.** Re-measure budgets on the target.
4. **Benchmark latency AND quality together, offline, on target hardware,
   against your own gold** — a $0 harness turned a scary model swap into a table
   lookup.
5. **Know which component owns the metric you're worried about** — the reranker
   owned latency, not recall; the reranker (not coverage) was where recall
   leaked. Localise the loss before fixing.
6. **Separate model-capability from prompt/retrieval with a controlled
   experiment** (a 2×2) before paying for a bigger model.
7. **Put a number on the expensive default before committing** — Opus was 4–6×
   Haiku and would pause the site after ~8 questions/day.
8. **A config flag mirroring an existing trusted pattern beats reworking test
   infrastructure.**
9. **Any create-per-unit workflow needs an explicit teardown owner, or it
   leaks** — automate a safe idempotent sweep as backstop.
10. **Benchmark the hypothesis cheaply before building it** — the multi-query
    benchmark saved a 5×-cost change that didn't address the real bottleneck.
11. **Draw the conclusion WITH the reader, not FOR them** — for a truth-critical
    product, a cited invitation beats an uncited assertion on both honesty and
    persuasion.
12. **Ship each fix as a small PR with a tagged release and end-to-end wire
    verification on the box** — small, reversible, verifiable increments under
    an orchestrator + adversarial-review discipline.
