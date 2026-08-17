"""PROTOTYPE (issue #3 spike) — minimal retrieve -> rerank -> native-citations loop.

*** Spike / de-risking code, NOT production. ***
The production RAG pipeline is issues #7/#12/#13/#14 and starts from red test
suites (IMPLEMENTATION.md §2 item 6). This package exists only to prove the core
mechanism — Claude native citations over custom-content document blocks — end to
end on the two real spike documents from #2, and to run the 20-question probe
that is the Phase-0 gate (DESIGN §10).

Modules:
* ``chunk_corpus``  — parse+chunk both spike PDFs via the #2 prototype chunker,
  keeping chunk text (spike_run.py drops it); writes the gitignored chunk corpus.
* ``probe``         — the committed 20-question probe (questions + expected
  source chunks); the gate evidence input.
* ``run_probe``     — bge-m3 embed, naive top-k retrieval, bge-reranker-v2-m3
  rerank to top-8, build the custom-content-block generation request, submit the
  20 questions via the Batches API on ``claude-haiku-4-5``, resolve citations to
  source blocks, and write the findings note + spend-ledger row.
"""
