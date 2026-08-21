# Grounded-generation system prompt (issue #12)

RED-PHASE STUB — the green-phase implementer authors the real prompt here.

This committed artifact IS the generation system prompt (prompt text is
data, IMPLEMENTATION.md §1). The failing suite in
`tests/unit/test_generation_system_prompt.py` pins what it must carry:

- the eight DESIGN §3.3 rules (answer only from passages; cite every
  claim; preserve calibrated language verbatim; lead with the headline
  finding at source severity; say plainly when passages don't answer;
  plain language; voices labelled, never evidence; beyond-assessed-range
  attributed to its authors, never as consensus);
- enough deliberate bulk that the STATIC prefix clears Haiku 4.5's
  4096-token minimum cacheable prefix (caching fails silently below it).

Nothing in this stub satisfies those tests; that is what red means.
