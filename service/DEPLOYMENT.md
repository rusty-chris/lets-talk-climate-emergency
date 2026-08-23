# service/DEPLOYMENT.md — deployment runbook (issue #22)

The one-command deploy and operating runbook for the FastAPI service
(DESIGN §9, ADR-015). Committed evidence, not a test. The service fails
**closed to a read-only paused state** on budget breach — it must never
take the briefing fully offline.

Companion: `service/README`-level detail lives in the module docstrings;
this file is the operator's checklist.

---

## 1. What runs

`docker compose up` brings up three services (see `docker-compose.yml`):

| Service | Image / command | Port (loopback) |
|---|---|---|
| `api` | `uvicorn service.main:app` | 8000 |
| `qdrant` | `qdrant/qdrant` | 6333 / 6334 |
| `ui` | `streamlit run ui/app.py` | 8501 |

`service.main` is the composition root: it reads the environment
(`service.config.load_service_config`), builds the real dependencies
(the live `AnthropicAdapter`, the lazily-loaded retrieval/chart seams, and
the #13 citation-support validator) and serves the app. Importing the
service never loads torch/docling/fitz — model weights load lazily on the
first live retrieval, so startup, `/health` and the paused state stay
cheap (issue #125).

## 2. Configure the environment

Critical variables (no defaults — the service refuses to boot, naming
every missing/invalid one at once; the list is `service.config.CRITICAL_ENV_VARS`):

| Variable | Meaning |
|---|---|
| `CLIMATE_CHAT_DAILY_BUDGET_USD` | Hard daily spend cap (USD). Breach → paused. |
| `CLIMATE_CHAT_OPUS_SUBCAP_USD` | Lower sub-cap for Opus "best" mode (≤ daily cap). |
| `CLIMATE_CHAT_CORPUS_VERSION` | Corpus version the deployed index must match. |
| `CLIMATE_CHAT_CORPUS_VINTAGE` | Vintage date shown in answer footers. |
| `CLIMATE_CHAT_SITE_URL` | Public base URL (chart attribution/CSV links). |
| `CLIMATE_CHAT_QDRANT_URL` | Qdrant endpoint (e.g. `http://qdrant:6333`). |
| `CLIMATE_CHAT_STARTER_CACHE_DIR` | Directory holding `starter_answers.json`. |
| `CLIMATE_CHAT_LOG_DIR` | Exchange-log + chart-spec store directory. |
| `ANTHROPIC_API_KEY` | Presence-checked only; never stored on config or logged. |

Optional variables (safe defaults): `CLIMATE_CHAT_RATE_LIMIT_PER_MINUTE`
(10), `CLIMATE_CHAT_BEST_MODE` (off), `CLIMATE_CHAT_TRUSTED_PROXY`
(off — set to `1` only behind a trusted ingress such as Fly/Railway, so
the first `X-Forwarded-For` entry is honoured), `CLIMATE_CHAT_COLLECTION`.

Live-generation-only (optional; needed for retrieval / chart generation,
not for the paused stack): `CLIMATE_CHAT_THRESHOLD_ARTIFACT` (calibrated
refusal-threshold artifact), `CLIMATE_CHAT_DATASET_MANIFEST`,
`CLIMATE_CHAT_CHART_PACK_DIR`, `CLIMATE_CHAT_CHART_STORE_DIR`.

`docker-compose.yml` passes each `CLIMATE_CHAT_*`/`ANTHROPIC_API_KEY`
through from the host with a dev-safe default, so a plain
`docker compose up` boots against the committed synthetic dev starter
cache at `/app/service/dev_starter_cache`. A real deploy sets the values
explicitly (a `.env` file or the platform's secrets UI for
`ANTHROPIC_API_KEY`).

## 3. Release-time starter-cache generation

The read-only paused state is only honest if there is something to serve.
**At each release**, regenerate the starter-topic answer cache and the
flagship chart specs through the real pipeline against the release corpus:

1. Ensure the release index is built and `CLIMATE_CHAT_CORPUS_VERSION`
   matches it.
2. Run the release cache-generation step (the pipeline once per starter
   question in `service.starter_cache.STARTER_QUESTIONS`, writing
   `starter_answers.json` with a fresh `generated_on` date and each entry's
   `answer_text`, `citations`, `footer`, and any `chart_spec_hash`).
3. Store the flagship chart specs in the chart-spec store so their
   `/chart/<hash>` permalinks serve while paused.
4. Point `CLIMATE_CHAT_STARTER_CACHE_DIR` at the generated cache.

`service.starter_cache.load_starter_cache` validates the artifact at
startup and refuses loudly (naming every missing/invalid question) rather
than starting on a silent empty paused state. The committed
`service/dev_starter_cache/starter_answers.json` is synthetic dev/smoke
content only — never ship it as the real cache.

## 4. One-command deploy

```
# From the repo root, with the environment configured (§2):
docker compose up -d --build
```

Point the platform ingress (Fly.io / Railway small VM, <£20/month target)
at the `api` service's port 8000. Publish only the `api` (and `ui`) ports;
keep `qdrant` internal.

## 5. Verify health

```
curl -sf http://<host>:8000/health          # -> {"status": "ok"} in BOTH modes
curl -sf http://<host>:8000/about            # static surface, 200
curl -sf http://<host>:8000/privacy          # carries the logging disclosure + lawful basis
```

`/health` returns `{"status": "ok"}` while live AND while paused — a paused
service is alive, not down. It is never rate-limited, so monitoring can
always tell pause from outage.

## 6. Budget cut-off behaviour (the GATE)

- Spend is tracked server-side per UTC day from every adapter-reported
  usage record, priced by the single source `evals.pricing`.
- `spend >= daily cap` (boundary included) → the service switches to the
  **paused** read-only state; `spend` cannot be read → also paused (fail
  closed). It resets at **midnight UTC**.
- While paused: `/chat` answers with a dated "paused for today" response
  (or the cached starter answer for a starter-topic question), zero LLM
  calls; charts, permalinks, `/about`, `/privacy`, `/sources`, `/voices`
  all keep serving. Opus "best" mode sits behind its own lower sub-cap;
  when the sub-cap is spent but the daily cap has room, queries fall back
  to the default model rather than refusing.
- Verified end-to-end by `tests/smoke/test_cutoff_fails_closed.py` (a
  simulated breach against the composed stack, checked from outside).

## 7. Backup & restore

**Exchange logs** (`CLIMATE_CHAT_LOG_DIR/exchanges.jsonl`): append-only
JSONL, one record per exchange, no identifiers (no IP, hash, user-agent,
cookie, session — `service.exchange_log.FORBIDDEN_IDENTIFIER_FIELDS`).
Records are retained 90 days, then deleted by the retention job
(`ExchangeLog.purge_expired`). Back up by copying the directory; restore by
replacing it. Run the 90-day retention job on a daily schedule (e.g. cron
calling a small script over `ExchangeLog.purge_expired`).

**Spend state**: tracked in-process per UTC day (resets at midnight UTC);
no restore needed across a restart within a day beyond re-reading it if a
persistence seam is configured. A restart mid-day starts the day's
accumulator from zero — size the cap with that headroom in mind, or wire a
persistent `spend_reader` (fails closed to paused on read error).

**Rate-limit store** (`service.rate_limit`): hashed-IP request counts with
a rotating daily salt, held ≤7 days (`RateLimiter.purge_expired`), stored
separately from the exchange logs with no field that can join the two.
Ephemeral — no backup needed; run the 7-day purge on a schedule.

**Chart-spec store** (`CLIMATE_CHAT_CHART_STORE_DIR`): ~1 KB JSON specs
addressed by content hash; back up with the log directory.

## 8. Owner actions (STOP-and-ask items — the owner's act, not the agent's)

These are gated per ORCHESTRATION.md §"Stop-and-ask points". Present them;
do not perform them.

- [ ] **ICO registration self-assessment.** Before public launch, complete
  the ICO registration self-assessment for processing personal data (the
  service logs conversation text under legitimate interests and holds
  short-lived hashed request counts for rate-limiting). Self-assessment:
  <https://ico.org.uk/for-organisations/data-protection-fee/self-assessment/>
  Record the outcome (registration reference or documented exemption) with
  a date, before the repo/site goes public.
- [ ] Create the hosting account (Fly.io / Railway) and register the domain.
- [ ] Provide the real `ANTHROPIC_API_KEY` via the platform's secrets store.
- [ ] Approve the voices-layer content (first-party prose about real people).
- [ ] Approve making the repository / deployment public.
- [ ] Confirm the monthly spend cap value against the <£20/month target.
