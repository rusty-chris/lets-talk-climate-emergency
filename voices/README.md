# voices/

The voices layer (DESIGN.md §2.5) — **not part of the RAG evidence corpus**.
A curated, hand-written library (`voices.yaml` + first-party descriptive
text authored by this project, and therefore freely ingestable) connecting
users to the people and campaigns publicly communicating the climate
emergency, without ever citing them as scientific evidence.

Answers grounded in this layer are labelled `source_type: voices` and
rendered as "About the movement" — structurally separated from
science-citation answers (DESIGN.md §4 item 10).

## Contents (issue #8)

- `voices.yaml` — the entities: the National Emergency Briefing campaign
  and film, the named briefing experts, Chris Packham, the Alliance of
  World Scientists / Ripple warnings (link-only), Ed Hawkins' warming
  stripes, the Climate Majority Project & SAFER, Covering Climate Now, and
  Sir David King / CCAG. First-party prose per entity, links, and snapshot
  facts (petition / MP / screening counts) that each carry an `as_of` date
  and are **rendered with it**.
- `render.py` — loads and validates `voices.yaml`, renders each entity for
  ingestion, and ingests it through the #7 chunker into
  `source_type: voices` chunks under the "About the movement" attribution.
  Voices text is first-party and in-repo, so it does NOT go through the
  sha256-pinned external-fetch path.
- `EDITORIAL_CHECKLIST.md` — the client's per-entity sign-off surface.

Ingest it with `make voices` (or `make ingest`, which runs it after the
corpus). Real, named people are described here, so **the content requires
the client's editorial sign-off before merge** (ORCHESTRATION.md).
