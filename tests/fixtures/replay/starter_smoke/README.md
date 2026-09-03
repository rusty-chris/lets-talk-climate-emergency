# Replay fixtures for the live-path smoke tier (review finding #231)

This directory is the `CLIMATE_CHAT_REPLAY_DIR` the replay-composed smoke
stack (`tests/smoke/test_starter_live_replay.py`) points the api's
`ReplayAdapter` at. It is committed **empty** (this README only), on
purpose:

- The fixtures are **generated at seed time, inside the composed stack**,
  by `scripts/seed_smoke_stack.py` (the `smoke-seeder` compose service) —
  a `FakeAdapter`-driven `RecordingAdapter` writes each canonical
  request-hash → synthetic-response pair. Zero live API calls, no key.
- They are generated rather than committed because the `generate_stream`
  and validator request hashes embed the **retrieved passages** — the
  output of real bge-m3 + bge-reranker inference over the seeded
  synthetic corpus. That computation is deterministic *within* one
  environment (the seeder and the api run the same image against the
  same qdrant), but pre-baked hashes recorded on a different host could
  drift on ulp-level score reordering. Generate-then-serve in the same
  stack removes the gamble; the seeder asserts the round-trip hash
  matches before the api ever boots.
- In compose, this path is a shared named volume (`replay_fixtures`)
  mounted into both the seeder and the api, so the generated fixtures
  never land in the host checkout.

All content is synthetic (the Aurelian-Basin fixture universe; invented
answers and verdicts) — nothing derives from a live model response.
