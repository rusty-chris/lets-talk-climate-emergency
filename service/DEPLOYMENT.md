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
(off — set to `1` only behind the trusted TLS-terminating Caddy ingress
of the `production` compose profile (§9), so the first `X-Forwarded-For`
entry is honoured; see §9 for why that is spoof-proof there and unsafe
anywhere else), `CLIMATE_CHAT_COLLECTION`,
`CLIMATE_CHAT_CHART_STORE_DIR` (defaults under the log dir).

Required to serve stored chart permalinks — the paused/read-only stack
**included** (#214): `CLIMATE_CHAT_DATASET_MANIFEST` and
`CLIMATE_CHAT_CHART_PACK_DIR`. Re-rendering a stored ~1 KB spec needs the
dataset manifest and the landed chart pack (the spec is the chart
definition, not the data), so any stack whose chart-spec store holds
specs — including a paused deploy serving the flagship permalinks that the
cached starter answers link to — must set both. The service validates
this at boot and refuses loudly, naming each missing one, rather than
500-ing the first permalink; a stack with an empty spec store needs
neither and returns clean 404s.

Live-generation-only (needed to answer live queries; validated at boot
whenever an index is recorded — #216, so a live deploy that skipped one
refuses to start instead of 500-ing the first query):
`CLIMATE_CHAT_THRESHOLD_ARTIFACT` (the calibrated refusal-threshold
artifact). A live deploy also needs the render inputs listed above.

`docker-compose.yml` passes each `CLIMATE_CHAT_*`/`ANTHROPIC_API_KEY`
through from the host with a dev-safe default, so a plain
`docker compose up` boots against the committed synthetic dev starter
cache at `/app/service/dev_starter_cache`. A real deploy sets every
value explicitly in a root-only env file on the server (template:
`deploy/production.env.example`; conventions in §9 — the filled file
holds `ANTHROPIC_API_KEY` and is never committed).

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

That is the dev/CI stack: every port loopback-only, no ingress. The
production deploy is the same compose file plus the `production` profile
(the Caddy TLS ingress) and the restart-policy overlay — the full
platform runbook is §9. In every deployment shape, only the ingress is
public and `qdrant` stays internal (never routed; see `deploy/Caddyfile`).

## 5. Verify health

```
curl -sf http://<host>:8000/health          # -> {"status": "ok"} in BOTH modes
curl -sf http://<host>:8000/about            # static surface, 200
curl -sf http://<host>:8000/privacy          # carries the logging disclosure + lawful basis
```

`/health` returns `{"status": "ok"}` while live AND while paused — a paused
service is alive, not down. It is never rate-limited, so monitoring can
always tell pause from outage.

The api is served `uvicorn service.main:app … --no-access-log
--no-proxy-headers` (compose command and Dockerfile CMD). `--no-access-log`
keeps uvicorn's default access line — which carries the raw client IP
(`client_addr`) — out of the container logs (DESIGN §9 / issue #212); the
app's hashed rate-limit records are the sanctioned request telemetry.
`--no-proxy-headers` leaves all `X-Forwarded-For` trust to
`resolve_client_ip` / `CLIMATE_CHAT_TRUSTED_PROXY` (the single XFF trust
point), so uvicorn never rewrites `request.client` from a spoofable header
behind the app's back. Do not re-enable uvicorn access logging.

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

**Retention runs in two places (#213).** The in-memory rate-limit store
can only be purged by the process that owns it, so the service itself
runs the purges: a FastAPI lifespan background task drives
`service.retention.run_retention_pass` (both `ExchangeLog.purge_expired`
— 90 days — and `RateLimiter.purge_expired` — 7 days) once at startup and
then every `RETENTION_PURGE_INTERVAL` (6 hours) for the process lifetime.
`RateLimiter.allow` also self-trims records outside the 7-day window on
each call, so the store stays bounded between passes. **No external cron
job can reach the rate-limit store** — the earlier runbook's "cron over
`RateLimiter.purge_expired`" was architecturally impossible (a separate
process sees an empty limiter).

**Exchange logs** (`CLIMATE_CHAT_LOG_DIR/exchanges.jsonl`): append-only
JSONL, one record per exchange, no identifiers (no IP, hash, user-agent,
cookie, session — `service.exchange_log.FORBIDDEN_IDENTIFIER_FIELDS`).
Retained 90 days. The in-process pass above already purges them; for an
out-of-process daily cron (e.g. if the service is idle), the shipped
runner covers the file-backed log:

```
uv run python scripts/run_retention.py "$CLIMATE_CHAT_LOG_DIR"
```

Because the serving process and this cron both rewrite the same file, every
read-modify-write holds an OS-level exclusive lock (`fcntl.flock` on a
`<path>.lock` sidecar, `service.exchange_log.exchange_file_lock`, #264), so
a purge can never lose-update a concurrent thumbs-verdict write and vice
versa. **flock is unreliable over network-mounted volumes (NFS): the log
volume MUST be host-local** — the compose `api_data` volume is. Rewrites are
atomic (temp file + fsync + `os.replace`, #265), so a crash mid-write leaves
the previous log byte-identical rather than destroying up to 90 days of
records; a torn trailing line left by an interrupted write is quarantined to
a `<path>.corrupt` sidecar with a WARNING and skipped, so retention and
reads keep working.

Back up by copying the directory; restore by replacing it.

**Eval-harvest triage** (#56, #266): to review thumbed-down exchanges and
promote one into a gold set through the irreversible detachment step, run
the shipped triage entrypoint next to the retention runner:

```
# List the queue (thumbs-down first; unsafe exclusions never shown):
uv run python scripts/run_harvest_triage.py "$CLIMATE_CHAT_LOG_DIR"
# Emit one exchange's detached, content-only promotion payload:
uv run python scripts/run_harvest_triage.py "$CLIMATE_CHAT_LOG_DIR" --detach <exchange_id>
```

**Spend state** (#217): journalled per UTC day to
`CLIMATE_CHAT_LOG_DIR/spend-state/` on every recorded usage and read back
at startup, so a restart or crash-loop cannot forget the day's spend and
un-pause the cap. Mount the log dir as a persistent volume (the compose
`api_data` volume does this) so the journal outlives the container. A
corrupt/unreadable journal fails closed to paused; a new UTC day starts
clean. The day still resets at midnight UTC.

**Rate-limit store** (`service.rate_limit`): hashed-IP request counts with
a rotating daily salt, held ≤7 days (`RateLimiter.purge_expired`, driven
by the in-process pass above), stored separately from the exchange logs
with no field that can join the two. Ephemeral — no backup needed.

**Chart-spec store** (`CLIMATE_CHAT_CHART_STORE_DIR`): ~1 KB JSON specs
addressed by content hash; back up with the log directory.

## 8. Owner actions (STOP-and-ask items — the owner's act, not the agent's)

These are gated per ORCHESTRATION.md §"Stop-and-ask points". Present them;
do not perform them.

- [ ] **ICO registration — covered by the existing Rusty Data registration
  (owner confirmation 2026-09-12).** The owner already pays the ICO annual
  data-protection fee as Rusty Data, and one registration covers all
  processing by the controller — so this service (conversation-text logging
  under legitimate interests, short-lived hashed request counts for
  rate-limiting) needs no fresh assessment or fee. Before public launch:
  confirm the `/privacy` page names the controller consistently with that
  registration, and add this service to the internal record of processing
  activities (an Article 30 note — kept internally, nothing filed). A fresh
  assessment/fee is only needed if the operating entity changes.
- [ ] Create the Hetzner account and register the domain (platform decision
  made 2026-09-12: Hetzner CX32, Ubuntu 24.04, EU DC — §9; account
  creation and DNS remain the owner's act).
- [ ] Provide the real `ANTHROPIC_API_KEY` in the root-only server env file
  (§9.5 — there is no platform secrets UI on a bare VPS; the env file IS
  the secrets store, which is why it is root-only and never committed).
- [ ] Approve the voices-layer content (first-party prose about real people).
- [ ] Approve making the repository / deployment public.
- [ ] Confirm the monthly spend cap value against the <£20/month target.

## 9. Hetzner deployment (chosen platform, owner decision 2026-09-12)

The owner chose **Hetzner Cloud** — a **CX32** (4 vCPU / 8 GB / 80 GB),
**Ubuntu 24.04**, an EU data centre (fsn1/nbg1/hel1), running the
committed compose stack with the `production` profile: **Caddy**
terminates TLS and is the only public entry point. Contract pinned by
`tests/unit/test_production_ingress.py`; routing in `deploy/Caddyfile`.

### 9.1 Provision the server (owner account, agent-scriptable after)

1. Create the CX32 with Ubuntu 24.04 in an EU DC, SSH **key-only** auth
   (no password login; disable root password auth in
   `/etc/ssh/sshd_config` if the image did not).
2. Firewall — both layers, because they are not the same wall:
   - Hetzner Cloud Firewall (or `ufw` on-host, or both): allow inbound
     **22/tcp** (SSH), **80/tcp** (ACME + https redirect), **443/tcp**
     and **443/udp** (HTTPS + HTTP/3); deny the rest.
     ```
     ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw allow 443/udp
     ufw enable
     ```
   - **Docker-published ports bypass ufw** (Docker programs iptables
     directly — finding #36 at host level). The stack's actual wall is
     that every base service binds `127.0.0.1` only and **caddy is the
     sole `0.0.0.0` publisher** (80/443). Never "fix" a connectivity
     problem by unprefixing a loopback port binding.
3. Point the domain's DNS A/AAAA records at the server **before** first
   boot — Caddy's ACME issuance needs the domain resolving to it.

### 9.2 Install Docker

Docker Engine + the compose plugin from Docker's apt repository (the
Ubuntu 24.04 default repo's docker.io lags):
<https://docs.docker.com/engine/install/ubuntu/> — then `docker compose
version` to confirm the v2 plugin.

### 9.3 Clone at the release tag

```
git clone https://github.com/rusty-chris/lets-talk-climate-emergency.git /opt/climate-chat
cd /opt/climate-chat && git checkout v1.0.0-mvp
```

Deploy from the tag, never from `main` tip: the tag is what the release
gates certified.

### 9.4 Build the release artifacts

Run the release steps **in the checkout, before the image build** — the
Dockerfile `COPY . .` bakes them in, and the env file (§9.5) points at
their in-container `/app/...` paths:

1. Corpus + index: `scripts/make_corpus.py`, `scripts/ingest_corpus.py`
   (and `scripts/ingest_voices.py`), so the recorded index version
   matches `CLIMATE_CHAT_CORPUS_VERSION` (§2, boot-checked).
2. Datasets + chart pack: `scripts/make_datasets.py` →
   `CLIMATE_CHAT_DATASET_MANIFEST` / `CLIMATE_CHAT_CHART_PACK_DIR`
   (required even for a paused stack — §2).
3. Calibrated refusal-threshold artifact + the published live eval
   results (`scripts/run_evals.py` per RELEASE-READINESS.md §(d) —
   `evals/RESULTS.md` must be the real live-run file; the #249 boot gate
   refuses an ingested deploy without it).
4. Starter cache + flagship chart specs: §3 of this runbook.

### 9.5 The env file (root-only, never committed)

```
install -m 600 -o root -g root deploy/production.env.example /root/climate-chat.env
$EDITOR /root/climate-chat.env    # fill every REPLACE-ME (§2 lists the semantics)
```

`/root/climate-chat.env` is the secrets store: root-only (`install -m 600`
above is exactly `chmod 600` + `chown root:root`), outside the checkout,
**never committed** (it holds `ANTHROPIC_API_KEY`).
Two values are ingress-specific and both matter:

- `CLIMATE_CHAT_SITE_URL=https://<domain>` — one path-routed origin
  (`deploy/Caddyfile`): the api's public routes (`/chart/...`, `/about`,
  `/privacy`, `/sources`, `/voices`, `/chat`, `/feedback`, `/health`)
  are served from the same domain as the UI, so every permalink the app
  renders off `SITE_URL` resolves as-is.
- `CLIMATE_CHAT_TRUSTED_PROXY=1` — behind Caddy the api's socket peer is
  always the caddy container, so `resolve_client_ip` must key the rate
  limiter on the first `X-Forwarded-For` entry. Spoof-proof **only**
  because the Caddyfile configures no `trusted_proxies`: Caddy replaces
  any client-supplied `X-Forwarded-For` with the real client address.
  Keep it `0` in any stack not fronted by this ingress.

### 9.6 Bring the stack up

```
cd /opt/climate-chat
docker compose -f docker-compose.yml -f deploy/compose.production.yml \
  --env-file /root/climate-chat.env --profile production up -d --build
```

`--profile production` adds the Caddy ingress; the
`deploy/compose.production.yml` overlay adds `restart: unless-stopped`
to the long-running services (api/qdrant/ui — deliberately not the
one-shot smoke-seeder) so a daemon or server restart brings the whole
stack back. Use the same `-f ... --profile production` flags for every
subsequent compose command (`ps`, `logs`, `down`), or compose will not
see the ingress service.

### 9.7 Verification battery

From anywhere (the public surface, through Caddy):

```
curl -sf https://<domain>/health     # {"status":"ok"} — live AND paused
curl -sf https://<domain>/about      # transparency page, 200
curl -sf https://<domain>/privacy    # logging disclosure + lawful basis
curl -sf https://<domain>/sources    # 200
curl -sf https://<domain>/voices     # 200
curl -sfI https://<domain>/          # Streamlit shell, 200
curl -sf http://<domain>/health -o /dev/null -w '%{http_code} %{redirect_url}\n'
                                     # 308 -> https:// (the port-80 redirect)
```

Then in a browser: load `https://<domain>/`, ask a starter question and
watch it stream (that exercises the Streamlit WebSocket through Caddy —
`reverse_proxy` upgrades it natively; a page that loads but never
streams is a WS failure), and open a chart permalink + its `.csv` link
from an answer footer (they render off `SITE_URL` and must resolve on
this origin). On the server: `docker compose ... ps` shows every service
healthy, and `qdrant` ports answer **only** on `127.0.0.1`
(`curl -sf http://127.0.0.1:6333/readyz` on-host works,
`https://<domain>:6333` from outside must not connect).

### 9.8 Backup cron

Everything stateful the service cannot regenerate lives in the
`api_data` volume (exchange log, chart-spec store, the #217 spend
journal — §7); the qdrant index is rebuildable from the corpus. Nightly
on-host cron (03:17 UTC, 14 kept), volume → tarball:

```
17 3 * * * docker run --rm -v climate_chat_api_data:/data:ro -v /root/backups:/backup alpine \
  tar czf /backup/api-data-$(date +\%F).tar.gz -C /data . \
  && ls -1t /root/backups/api-data-*.tar.gz | tail -n +15 | xargs -r rm
```

(Adjust the volume name prefix to `docker volume ls`'s output — compose
prefixes it with the project directory name.) Restore per §7: replace
the volume contents and restart. This cron is host-side backup only; the
retention purges themselves run in-process (§7).

### 9.9 Uptime ping

Point any external monitor (e.g. UptimeRobot / Uptime Kuma) at
`GET https://<domain>/health` expecting HTTP 200 and body
`{"status":"ok"}`. `/health` is never rate-limited and returns 200 in
the paused state too — so this ping distinguishes *outage* (alert) from
*budget pause* (by design, no alert). Alert on non-200/timeout only.
