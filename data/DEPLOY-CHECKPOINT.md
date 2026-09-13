# Production deploy checkpoint — 2026-09-13

Server: `root@95.217.167.100` (Hetzner CX33). Repo checkout: `/opt/climate-chat`
at tag **`v1.1.0-launch`** (`c04968e`). Session:
`deploy-starter-cache-2026-09-13`.

This checkpoint records the state a Fable operator reached before it was
stopped (~09:25–10:25 UTC) to conserve quota, and the Opus operator's
survey/decisions afterwards. It is a factual record, not a sign-off.

## TL;DR — go-public status: **BLOCKED (owner decision required)**

The infrastructure is essentially ready, but the production `api` container
**cannot boot** and the site **cannot go public** until the release
starter-answer cache holds all 13 valid entries. Completing it needs
**~$0.15 more of live spend**, but only **~$0.12 remains** under the
**$0.50 hard deploy cap** — and $0.38 of that cap was already consumed by
six non-idempotent generation restarts. The repo (`service/DEPLOYMENT.md §3`)
**forbids** shipping the synthetic dev cache as the real one, and
`load_starter_cache` refuses to start on a partial cache. So the deploy is
parked at a genuine budget wall that only the owner can clear (raise the cap
or accept an alternative). Details in "The blocker" below.

## Per-step state (inherited from Fable vs. done/verified by Opus)

| Step | State | By |
|---|---|---|
| apt upgrade + hardening | done | Fable |
| UFW 22/80/443 (+443/udp, v6) | **verified active** | Opus |
| SSH password auth disabled; root prohibit-password | **verified** (`sshd -T`) | Opus |
| Docker 29.8.0 + Compose v5.5.1 | verified installed | Fable/Opus |
| Repo cloned at `/opt/climate-chat` @ `v1.1.0-launch` | **verified matches tag** (`c04968e` == `refs/tags/v1.1.0-launch^{}`) | Opus |
| `v1.1.0-launch` tag created & pushed to origin | already present at origin — no action needed | Fable |
| `qdrant` container up healthy | verified (restarts=0; started 10:32 after a benign restart, exit 0) | Opus |
| `climate-chat-api` image built | log ends "Image climate-chat-api Built" | Fable |
| Corpus + datasets built & indexed | **verified**: 809 chunks (795 evidence + 14 voices), 32 docs, `gold_parity_exact: true`, version `v1.1.0-launch-2026-09-13` | Fable built / Opus verified |
| Secrets `/root/climate-chat.env` (mode 600) | **verified present** (1363 B, 17 vars incl. `ANTHROPIC_API_KEY`) — API key NOT re-sent | Fable |
| Starter-cache generation | **INCOMPLETE — 3/13 entries; blocked (see below)** | Fable ran; Opus surveyed |
| Config PRs (Caddy, backup, cap map, transparency, repins) | **5 PRs already open** (#370–#374, base `main`) | Fable |
| Chart-spec store / `/chart/<hash>` permalinks | not yet populated (part of §3, blocked with starter cache) | — |
| `docker compose --profile production up` | **NOT run** (api can't boot without a valid cache) | — |
| Caddy certs (both domains) / redirects | **not reached** (stack not up) | — |
| Verification battery (health, transparency, live Q, charts, feedback, budget drill, IP-log check) | **not reached** | — |
| Nightly backup cron + proof pull | **not reached** | — |

## The blocker — starter-answer cache (the one thing standing between here and live)

`service/DEPLOYMENT.md §3` + `service.starter_cache.load_starter_cache`:
the production api refuses to start unless `starter_answers.json` contains a
valid, cited entry for **every** question in `STARTER_QUESTIONS` (13), and
the committed `service/dev_starter_cache` synthetic content must **never** be
shipped as the real cache. So a complete real cache is a hard gate for go-live.

**What the Fable operator's runs actually did** (`/root/release-build/`):
- Only **3 of 13** entries were written (`entries/00.json`, `01.json`,
  `02.json`). `02` was the last; run 6 died mid-Q3 when its SSH parent
  exited. Entry `00` ("Why are scientists calling this an emergency?") is a
  **persistent decline** (validated=false, shipped as honest-decline prose
  after 4 retries).
- The generator (`/root/release-build/generate_starter_cache.py`, a
  server-only script — **not in the repo**) iterates
  `for … in enumerate(STARTER_QUESTIONS)` from index 0 every run. It has
  **no resume/skip** for already-written entries, so each of the **6 runs**
  re-ran Q0 (4 declining retries each) before making any new progress. That
  non-idempotent restart loop is what burned the budget.
- It also carries its own `cap $0.5` meter, so a fresh completion run would
  **halt mid-cache** on the cap rather than finish.

**Why some questions decline is a known corpus gap, not a deploy bug.**
`evals/gold/COVERAGE.md` records that flagship-type questions (why-emergency,
the "hasn't warming paused" hiatus item, CO₂-fertilisation, 1970s-cooling)
are *blocked on corpus-expansion* — the pending Tier A explainer pages
(`nasa_climate_explainers`, `noaa_climate_explainers`, `metoffice_explainers`)
are not yet ingested (corpus is a deliberately minimal 32 docs). So even a
fully-funded completion run would ship **honest-decline** answers for the
first flagship starter question and the CO₂/temperature chart question. That
is a product/content decision the owner should see, which is a second reason
this is owner-gated rather than operator-fixable.

### Options for the owner (recommended first)
1. **Raise the deploy cap by ~$0.20** and let the operator complete one
   generation run. Pair with making the generator resumable (skip the 3
   written valid entries) so completion costs only the remaining questions
   (~$0.10), not another full restart. *Recommended.*
2. Expand the corpus with the pending Tier A explainers first, then
   regenerate — yields grounded (non-decline) flagship answers. Larger scope.
3. Accept that flagship starter answers ship as honest declines (option 1's
   content outcome) — an explicit product ruling.
4. **Not acceptable per the repo:** shipping the synthetic dev cache publicly
   (`§3` forbids it) or booting on a partial cache (`load_starter_cache`
   refuses).

## Spend reconciliation (HARD CAP $0.50 total, incl. Fable's spend)

Authoritative server meter (`/root/release-build/usage_tally.json`, confirmed
against `usage_tally.run6.json`): **total = $0.380284**, all
`claude-haiku-4-5` (classify/generation/validator), across 6 runs
(runs 1–5 carried prior = $0.315895; run 6 measured = $0.064389).

- **Spent so far: $0.380 of $0.50** → **~$0.12 headroom.**
- Ledgered to `evals/spend-ledger.csv` as two honest rows under session
  `deploy-starter-cache-2026-09-13` (prior-runs aggregate + measured run 6),
  appended to the launch lineage; cumulative advances
  $9.334428 → **$9.714712**.
- **No further spend was incurred by the Opus operator** — the mandated one
  live verification question was **deliberately not run**, because with the
  deploy blocked on the cache there is nothing yet to verify end-to-end and
  the remaining headroom must be preserved for the completion decision.

## Other findings / notes
- **Env-file quoting bug (latent):** `/root/climate-chat.env` has an unquoted
  multi-value line (`CLIMATE_CHAT_REDIRECT_DOMAINS=letstalkclimateemergency.org, …`)
  that breaks POSIX `sh` sourcing — the generator logged
  `sh: 5: /root/climate-chat.env: letstalkclimateemergency.org,: not found`.
  Docker Compose reads env files literally (does not shell-source), so this
  does **not** affect `compose up`; fix (quote the value) before any tooling
  that `source`s the file. Verify the service's comma-split trims the leading
  space in the second domain.
- **`#368 getrusage`: deferred — not implemented** (per brief).
- **No code was patched on the server.**
- Open config PRs cover the known repo-truth gaps: #370 Caddy both domains,
  #371 transparency source mounts, #372 nightly backup cron, #373 £10/mo→daily
  cap mapping, #374 launch-corpus repins. These are the Fable operator's; they
  await orchestrator review/merge.

## Remaining sequence once the cache is unblocked
Complete starter cache (13 entries) + chart-spec store → full env verify →
`docker compose --profile production up -d --build` → Caddy certs both domains
+ `.org`/www redirects → verification battery (health both modes; 5
transparency pages incl. `/footprint`; one live grounded question, ledgered;
chart starter + permalink + .csv/.svg; feedback endpoint; budget cut-off
drill; no raw IPs in logs) → nightly backup cron + one proof-pull → full
deploy report.
