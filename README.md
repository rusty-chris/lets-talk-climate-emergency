# Let's Talk About the Climate Emergency

A free, open-source, public-benefit chatbot that gives people the emergency briefing on climate they have never had: straight answers grounded strictly in authoritative publications, with inline citations to the exact source passages — plus the ability to generate shareable, source-stamped charts from canonical climate datasets.

## What this is

This is an **educational piece of software for public benefit**. It is free to use, carries no advertising, sells nothing, and its code and evaluation results are public. Rusty Data builds and stewards it and may point to it as work it has done, but the product itself is **non-commercial** — several key sources are ingested only on that basis (see `DESIGN.md` §2.1 for the licensing consequence: if the project ever becomes commercial, every non-commercial-licensed document must be removed from the corpus).

**Not affiliated with or endorsed by** the National Emergency Briefing campaign, NASA, NOAA, the Met Office, Copernicus, USGCRP, UNEP, or the IPCC. All sources cited and linked.

## Status

Phase 0 — repo scaffolding. See `issues/` for the full build plan and `DESIGN.md` §10 for the roadmap and release gates.

## Documentation

- [`DESIGN.md`](DESIGN.md) — design document: mission, corpus & ingestion, RAG pipeline, guardrails, evaluation, tech stack, deployment (source of truth for scope and architecture).
- [`DECISIONS.md`](DECISIONS.md) — architecture decision record (ADR) log.
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — TDD implementation design: module boundaries, test-first workflow, test pyramid, CI stages.
- [`ORCHESTRATION.md`](ORCHESTRATION.md) — autonomous build methodology: roles, the per-issue loop, review process.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — version-control protocol.

## Repo layout

| Path | Purpose |
|---|---|
| `ingestion/` | Manifest validation, licensing gate, fetch, parse, chunk (DESIGN §2) |
| `rag/` | Embedding, indexing, retrieval, rerank, refusal gate, provider adapter (DESIGN §3) |
| `charts/` | Chart data pack, transforms, ChartSpec validation, planner, renderer (DESIGN §3.7) |
| `ui/` | Streamlit UI (DESIGN §7) |
| `service/` | FastAPI service: routes, budget tracker, rate limiter (DESIGN §9) |
| `evals/` | Evaluation harness (DESIGN §6) |
| `corpus/` | Corpus manifest + fetch scripts (manifest only — see `corpus/README.md`) |
| `datasets/` | Chart dataset manifest + fetch scripts (see `datasets/README.md`) |
| `voices/` | The voices layer — campaigns and people communicating the emergency (DESIGN §2.5) |
| `reviews/` | Design-review reports |
| `scripts/` | Repo maintenance scripts (e.g. `publish_issues.py`) |
| `tests/` | pytest suite: unit, `integration`, `smoke`, `live` (IMPLEMENTATION.md §3) |

## Development

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync                    # install dependencies into .venv
uv run pre-commit install  # install git hooks (ruff lint/format)
uv run pytest              # unit tests (default); add -m integration / -m smoke for the other tiers
docker compose up          # start the api, qdrant and ui stub services
```

## Licence

Code is licensed under [Apache-2.0](LICENSE). Corpus and dataset text are governed by the per-document licensing terms recorded in the manifests (`corpus/`, `datasets/`) — see `DESIGN.md` §2.1; nothing outside the `open` permission tier is redistributed in this repository.
