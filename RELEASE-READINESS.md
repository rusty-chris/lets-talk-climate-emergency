# RELEASE-READINESS.md — offline gate dry-run + owner launch checklist

> Compiled by the release-readiness operator on 2026-09-04 against `main` at
> commit `5a2d3c7`; issue/queue state refreshed against `main` at `2048be1`
> (the #296 merge). Every claim below is checked against the repo; no
> aspirational statements. This is a
> **dry-run report and runbook**, not a release certification — a release is
> certified only by the live eval run of §3, which this document cannot and
> does not perform (zero live API calls were made producing it).

---

## (a) Build-state summary

**Features / issue queue.** All 24 core build issues are merged. The 15
issues still open on GitHub are *not* build blockers:

| # | Kind | Status for launch |
|---|---|---|
| 162 | review-finding, **major** | Open. Two #16 acceptance-criterion replay tests unauthored (recorded in PR prose). Planner-replay half is now landed (#283 wire-schema fix; fixture committed in the #162 session-5 ledger row); flagship replay stays skip-marked, blocked on #23 / #281 (curated-only ruling). Non-blocking for offline gates; tracked. |
| 291 | review-finding, minor (evals) | Open. Deferred **live-eval obligation**, not a code defect: live-eval the semantic-cache 0.95 cosine threshold against the REAL bge-m3 embedder (fakes cannot pin real geometry). Belongs to the live eval tier (§3). |
| 294 | review-finding, minor | **CLOSED** 2026-09-04 — PR #296 merged the unit-tier full-seam guard for the shared embedder (`main` at `2048be1`). Out of the queue. |
| 281 | documentation | Open. Records the design ruling that splice charts are curated-only for MVP (decoder-authored splices foreclosed). Decision doc, not code. |
| 282 | enhancement | Open. `thinking:{type:disabled}` output-budget optimisation. Deferred. |
| 58, 59 | SotA recommendations | Open. Contextual-retrieval A/B (58) and read-only MCP server (59). Phase-2 deferred. |
| 23 | phase-1.5, licensing | Open. **Owner-gated** permission/NC-confirmation letters + NEB outreach. Blocks Tier-C ingestion (Ripple/WMO/IPCC full text) and the decoder-authored flagship splice. See §2 and §4. |
| 297–304 | refactor (×8) | Open. The phase-gate logic-efficiency refactor queue (ORCHESTRATION.md recurring audit): 8 clusters, ~-300 LOC total. Non-blocking for launch; queued behind release-critical work. **Exception — #303 flags a release-relevant gap** (see below and the §(d) precondition). |

**#303 gate-wiring gap (release-relevant, in progress).** #303's audit
(verified against the code) found the release gate battery exists twice with
drifted membership — `evals/harness.py:_arm_gate_battery` lacks
`chart_faithfulness_gate`, which only the offline CLI battery includes — and
that three built-and-unit-tested gates (`citation_support_gate`,
`route_accuracy_gate`, `opus_escalation_allowed`) are wired into **no**
battery, so DESIGN §10's citation-support number is currently un-gated on
every path. Per the orchestrator (2026-09-04) this is being fixed red-first
on branch `review-303-gate-wiring` (not yet pushed to origin at refresh
time). The §(d) live release run is conditioned on the post-#303 unified
battery.

**Review queue.** Not fully dry: 2 open `review-finding` issues — one major
(#162), one minor (#291 deferred-live). None is a `blocker`. Per
ORCHESTRATION.md §"Definition of ready", these are the "explicitly deferred
with recorded reasons" tail, not unresolved blockers.

**Spend ledger** (`evals/spend-ledger.csv`, single source of truth).
Cumulative API spend to date: **$0.338359** across 9 recorded sessions
(spike probe, #162 planner-recording sessions, #276 Sonnet capability probe,
#262 schema probe). Well under any per-run cap. Producing *this* document
made **zero** live API calls (`ANTHROPIC_API_KEY` unused; offline suite is
$0 by design).

---

## (b) Offline eval gate dry-run

Ran `uv run python scripts/run_evals.py --offline --out-dir <scratch>` on the
**real committed gold sets** (`evals/gold/climate_qa.yaml` +
`evals/gold/chart_requests.yaml`, both validated OK). Artefacts written to a
scratch dir only (post-#242: default never publishes to `evals/`), banner
`OFFLINE / SIMULATED RESULTS` in the RESULTS.md, payload `mode:
offline-simulated`. **Release verdict: `BLOCKED`; process exit code 1**
(fail-closed — a blocked owner audit must block the release build).

| Gate | Status | Count | What it certifies here / caveat |
|---|---|---|---|
| refusal | PASSED | 20/20 (≥0.9) | Simulated on the offline seam — refusals are asserted True for the gold refusal ids, not measured against a live classifier. Certifies the gate arithmetic + gold wiring, **not** live refusal behaviour. |
| false_refusal | PASSED | 0/55 (<0.05) | Derived from the actual offline answer-path run over the 55 answerable gold items (no false refusals). Plumbing + arithmetic real; answer *quality* not judged. |
| canned_out_of_scope | PASSED | 9/9 | Simulated (asserted True for the canned ids). |
| **severity** | **BLOCKED** | — | **Expected and correct.** `evals/gold/severity-audit-packet.md` header is `owner_severity_audit: pending`; the gate calls `assert_owner_severity_audit_complete()` and refuses to score the 15 agent-authored severity labels until the owner audits them. This is the single gate driving the `BLOCKED` verdict. |
| chart_spec | PASSED | 11/11 | Derived by comparing the gold-driven planner's ACTUAL output to gold (a wrong planner would fail). 1 flagship item visibly SKIPPED with recorded reason (Binding #117 / #23: flagship expected-values excluded until #23 confirmations land; splice arithmetic kept under synthetic-data fixtures meanwhile). |
| chart_faithfulness | PASSED | 1320/1320 | Real: the independent fixture generator (`compute_chart_fixtures`) is re-run over the committed synthetic CSVs and every rendered value compared to committed `chart_fixtures.json`. Catches drift in fixtures or synthetic data. |
| chart_refusal | PASSED | 3/3 | Derived from actual planner refusal outputs vs gold. |
| voices_separation | PASSED | 0/0 (structural) | No voices/evidence separation violations in the offline answer records. |

**What the offline PASS gates do NOT certify:** none of the live LLM-judge
release gates (faithfulness, citation-support rate, confidence-level
fidelity, severity fidelity, adversarial rubric) run offline — those are
non-deterministic, cost money, and require the live model (IMPLEMENTATION.md
§4.4). The offline suite exercises the classify→route→generate→gate
*plumbing*, the deterministic metric *arithmetic*, and the chart
data-faithfulness path against synthetic data. A green offline run means the
harness and gates are wired correctly and the deterministic checks hold; it
is **not** a release pass. The real release verdict comes only from §3.

---

## (b′) Release-artefact path verification (§2 of the tasking)

- **What a real release run produces.** `scripts/run_evals.py` with
  `--live`/`--record` refuses without `ANTHROPIC_API_KEY` (verified: it
  names the missing credential and exits 1 rather than fabricating a run).
  Live/recording arms are driven by the recorded-run tooling, not this entry
  point. A real release run writes `results.json` + `RESULTS.md`; publishing
  to `evals/` is the explicit `--out-dir evals` opt-in.
- **Where `evals/RESULTS.md` must land (the #249 live-boot gate).**
  `service/main.py:validate_deployment_artifacts` enforces #249: when the
  deployed index has a recorded corpus version (a real ingested/live deploy)
  and the provider is not the `replay` smoke stack, the repo's
  `evals/RESULTS.md` **must be a readable file** or the service **refuses to
  boot**, naming the path. Only the un-ingested dev/compose-smoke stack
  (`index_corpus_version is None`, the #215 zero-config boundary) and the
  explicit `replay` stack may boot without it and serve the honestly-marked
  interim placeholder pages. Confirmed `evals/RESULTS.md` is **not** committed
  today (correct — it must come from the real live run, not an offline
  simulation).
- **Dry-check of `build_transparency_pages`.** Ran
  `service.transparency.build_transparency_pages` against a **temp copy** of
  the offline `RESULTS.md` (not committed). Result: builds successfully;
  `/about` carries the `OFFLINE / SIMULATED` banner through from the results
  text (proving the transparency build faithfully renders whatever results
  file it is given — so shipping an offline file would visibly mislabel the
  public page, which is exactly why #249 requires the real one); `/privacy`
  still renders the `PENDING-owner-decision` contact placeholder;
  `/voices` renders the signed-off `voices.yaml` (16 KB). **RESULTS.md was
  not committed to `evals/`.**

---

## (c) Owner-gate checklist — exact actions and current states

Each item below is a stop-and-ask owner action (ORCHESTRATION.md §Stop-and-ask
/ DEPLOYMENT.md §8). The agent has prepared everything; the owner performs
the act and flips the recorded state.

| Gate | Current state | Exact owner action |
|---|---|---|
| **Severity audit** | `owner_severity_audit: pending` (`evals/gold/severity-audit-packet.md` line 1). Blocks the release severity gate → offline verdict BLOCKED. | Read `evals/gold/severity-rubric.md`; review the 15 annotations (load-bearing: qa-sev-07/08/09/11); correct any label in `climate_qa.yaml` + the packet; set header to `owner_severity_audit: complete <YYYY-MM-DD>`, commit, then regenerate COVERAGE.md (`python evals/scripts/gold_coverage.py`). |
| **Permission letters** | `permission_letters_sent: pending` (`letters/SENDING-RECORD.md` line 1). Drives the `/about` Ripple exclusion wording ("permission to be requested"). **Six** letter drafts prepared: `letters/01-ipcc.md`…`06-neb-campaign.md` (recounted 2026-09-04 — `letters/` holds 8 files, but two are records, not letters: `ADDRESSEES.md` and `SENDING-RECORD.md`; the SENDING-RECORD itself names the range 01–06). | Send the letters under the owner's name (IPCC, OUP/Ripple, WMO, Carbon Brief + Berkeley Earth NC-confirmations, NEB outreach); flip the header to `sent <YYYY-MM-DD>` and commit. Part of issue #23. |
| **Privacy contact email** | `PRIVACY_CONTACT_EMAIL = "privacy-contact-PENDING-owner-decision@example.invalid"` (`service/transparency.py:204`). Rendered verbatim on `/privacy`. | Replace with the real published UK-GDPR contact address (one-line change at that constant; the page renders it and nowhere else). |
| **#260 voices manual review** | Issue **CLOSED** 2026-09-03 — owner ruled the voices layer ships in the prototype *with* the 4 unverified claims retained, behind the published `VOICES_PROTOTYPE_NOTE` ("still under editorial review"). Not a boot blocker. | Follow-up (before/shortly after launch): verify the 4 retained claims — NEB "ten experts" count, Mann/Haigh "supporter" vs "signatory", Oldridge brothers convening claim, 7 Apr 2026 film date — and correct `voices/voices.yaml` as needed. |
| **Voices content sign-off** | Signed-off content merged (#198/#292); `voices/voices.yaml` is the build source of truth; placeholder retired. | No blocking action; covered by the #260 follow-up above. |
| **#23 / Tier-C + flagship** | Open, owner-gated (licensing). Blocks Tier-C full-text ingestion (Ripple/WMO/IPCC) and the decoder-authored flagship splice (Kaufman/Bereiter are open-provisional per Binding #117). The offline chart_spec gate already skips the flagship expected-values with this recorded reason. | Obtain the written affirmative permissions (via the letters above), then land Tier-C sources / flagship fixtures. Until then flagship ships **curated, not decoder-recorded** (#281 ruling). |

---

## (d) Launch sequence (distilled from `service/DEPLOYMENT.md`)

**Owner provisions first (stop-and-ask; DEPLOYMENT.md §8):**
1. **ICO registration self-assessment** for processing personal data (the
   service logs conversation text under legitimate interests + short-lived
   hashed rate-limit counts). Record the reference or documented exemption
   with a date before the repo/site goes public.
2. Create the hosting account (Fly.io / Railway, <£20/month target) and
   register the domain.
3. Provide the real `ANTHROPIC_API_KEY` via the platform secrets store
   (presence-checked only; never stored on config or logged).
4. Confirm the monthly spend cap value against the <£20/month target.
5. Complete the owner gates in §(c): severity audit, send letters, privacy
   email, voices follow-up.

**Orchestrator then runs (autonomous, once the owner gates clear):**
1. **Live eval release run** via the recorded-run tooling (requires the key
   + a passing budget pre-flight). **Precondition: the #303 gate-wiring fix
   must be merged first** — the live run MUST use the post-#303 unified gate
   battery (one shared battery builder; `chart_faithfulness_gate` reconciled;
   the citation-support / route-accuracy / opus-escalation gates wired per
   the orchestrator's #303 ruling), not either of today's diverged batteries. Every DESIGN §10 / IMPLEMENTATION §6 gate
   must pass: faithfulness / citation-support targets, refusal >90% /
   false-refusal <5%, **severity ≥90% exact-or-adjacent with zero two-level
   errors** (only scorable once the audit is `complete`), severity-retrieval
   recall, chart data-faithfulness 100% vs fixtures, voices-separation 100%,
   fail-closed cut-off verified. Includes the **#291 threshold gate** — the
   deferred live-eval of the 0.95 semantic-cache cosine threshold against the
   real bge-m3 embedder (ratify whether 0.95 stands or needs a lexical veto).
   **STOP-POINT:** any failed/blocked gate → non-zero exit → release blocks.
2. **Publish `evals/RESULTS.md` + `results.json`** to `evals/` (`--out-dir
   evals`) and commit — this is the artefact the #249 live-boot gate and the
   #19 `/about` transparency build read. (Must be the live-run file, never an
   offline-simulated one.)
3. **Build the release corpus + index** so `CLIMATE_CHAT_CORPUS_VERSION`
   matches the deployed index; set `CLIMATE_CHAT_CORPUS_VINTAGE`.
4. **Regenerate the starter-answer cache + flagship chart specs** through the
   real pipeline against the release corpus (DEPLOYMENT.md §3); point
   `CLIMATE_CHAT_STARTER_CACHE_DIR` at it. Never ship the synthetic
   `service/dev_starter_cache`.
5. **Configure the environment** (DEPLOYMENT.md §2): all `CRITICAL_ENV_VARS`
   (no defaults — the service names every missing one and refuses to boot),
   plus `CLIMATE_CHAT_THRESHOLD_ARTIFACT`, `CLIMATE_CHAT_DATASET_MANIFEST`,
   `CLIMATE_CHAT_CHART_PACK_DIR` for a live/permalink-serving stack.
6. **`docker compose up -d --build`**; point ingress at `api:8000` (publish
   `api`/`ui` only, keep `qdrant` internal).
7. **Verify health** (`/health` → `{"status":"ok"}` in both live and paused
   modes; `/about`, `/privacy`, `/sources`, `/voices` 200) and the
   **fail-closed budget cut-off** (`tests/smoke/test_cutoff_fails_closed.py`
   convention: a simulated breach flips to the read-only paused state, zero
   LLM calls, permalinks/static pages keep serving).

**Compose-stack sanity (§3 of the tasking).** Docker is **not available** in
this operator environment (`docker: command not found`), so the smoke tier
could not be run locally. Verified instead via CI: the latest **completed**
`main` run (#292, `33830837653`) shows **smoke (docker compose up + health
checks) ✓ success** (6m36s), alongside lint ✓, unit ✓, integration ✓. The
#293 push run (`33833481577`) subsequently completed **success on all four
jobs including smoke**; the #296 merge run (`33834604068`) was still
in-progress at refresh time.

---

## (e) Standing non-commercial / licensing constraints (do not weaken)

- **ADR-018 — non-commercial educational public-benefit framing.** Supersedes
  the old commercial-portfolio framing. The corpus is tiered: **Tier A**
  (open/public-domain), **Tier B** (NC licences, unlocked by ADR-018),
  **Tier C** (permission-pending, link-only until a written affirmative reply
  is on file — Ripple/OUP, WMO, IPCC full text).
- **Tier-B usage rule.** `permitted_context` is load-bearing:
  Tier-B (e.g. Carbon Brief CC BY-NC-ND) may be used only in the
  non-commercial educational context. **Commercial use requires dropping
  Tier B from the manifest first** — never "it's basically non-commercial,
  use it in a paid demo". The manifest makes the drop mechanical.
- **Free-to-read ≠ licensed.** Ripple et al. / BioScience are free-to-read
  but all-rights-reserved OUP (verified 2026-08) → Tier C, link-only until
  #23's permission lands. The hardened licence gate exists to catch exactly
  this trap.
- **Voices/evidence separation** is a tested invariant: voices-layer prose is
  never cited for scientific claims; the voices-separation gate enforces it.
- **Charts** are rendered server-side from a closed ChartSpec vocabulary over
  named datasets (ADR-020) — the model writes neither the numbers nor the
  pixels; no model-authored plotting code, no open-web data (ADR-021).

---

## Discrepancies found (docs vs reality)

1. **#260 is CLOSED, not an open "pending manual review" gate.** The owner
   ruled (2026-09-03) that the voices layer ships in the prototype with the
   unverified claims retained behind the published under-review note. The 4
   verification items survive as owner *follow-ups* (before/shortly after
   launch), not as a boot/release blocker. Represented that way in §(c).
2. **The release gate battery exists twice with drifted membership (#303).**
   The offline gate table in §(b) was produced by `run_offline_suite`'s
   inline battery, which includes `chart_faithfulness_gate`; the harness's
   `_arm_gate_battery` (the live-run path) lacks it, and the built
   citation-support / route-accuracy / opus-escalation gates are wired into
   neither. So today's §(b) table is one battery's view, and a live run on
   the un-fixed harness battery would silently gate *less* than the offline
   run did. Being fixed red-first (branch `review-303-gate-wiring`); the
   §(d) precondition holds the live run until the unified battery is merged.
3. **Letters count: six drafts, not eight.** `letters/` holds 8 files, but
   `ADDRESSEES.md` and `SENDING-RECORD.md` are records; the letter drafts
   are exactly `01-ipcc.md`–`06-neb-campaign.md`, matching the range the
   SENDING-RECORD itself names.
4. **Everything else matched the docs.** The offline verdict is BLOCKED for
   exactly the documented reason (pending severity audit); the #249 gate,
   the SENDING-RECORD, the PRIVACY_CONTACT_EMAIL placeholder, and the ADR-018
   tiering are all in the state their docstrings/records claim.
