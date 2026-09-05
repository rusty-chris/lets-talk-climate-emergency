"""Citation density + relevance in the generation prompt (finding #311) — RED.

The release run's haiku arm produced 524 factual sentences against only
256 citation events: even under a perfected span-aware validator (#310),
roughly half the factual sentences carry no citation event at all and
score unsupported. Separately, several answers cited chunks unrelated to
the sentence they accompany (drought-definition text cited for observed
warming; sea-level text cited for CO2-greening sentences).

The fix is prompt-level first (issue #311): the committed generation
system prompt must carry

1. an explicit EVERY-FACTUAL-SENTENCE-CITES instruction — the anchor
   phrase "every factual sentence" (or "each factual sentence") tied to
   the citation duty, sentence-level rather than the current
   claim-level phrasing, since the §10 target and the #13 validator
   both count per SENTENCE;
2. the omission preference — prefer OMITTING a claim over asserting it
   uncited (anchor: omit/leave out/drop … rather than/over/instead
   of … uncited/without a citation);
3. the relevance rule in prohibition form — NEVER cite a passage/
   block/document that does not support the sentence (anchor:
   never/do not … cite … does not … support).

These are characterisation guards in the style of
``tests/unit/test_generation_system_prompt.py``: each pins a
load-bearing invariant textually (anchor language + non-contradiction),
never exact phrasing — the green-phase author is free to write good
prose around the anchors.

Caching coherence (the static-prefix cache floor rules): the density
instructions must live in the STATIC committed artifact — block 0 of
the system channel — never in a volatile block, so the strengthened
prompt still shares one cached prefix across all traffic and still
clears Haiku 4.5's 4096-token minimum cacheable prefix. (A prompt
change invalidates replay recordings by design — IMPLEMENTATION.md
§4.2; the request hash changing IS the mechanism that retires stale
recordings.)
"""

from __future__ import annotations

import re

from rag.generation import (
    HAIKU_MIN_CACHEABLE_PREFIX_TOKENS,
    SYSTEM_PROMPT_PATH,
    GenerationConfig,
    build_generation_request,
    estimate_tokens_lower_bound,
    load_system_prompt,
)
from tests._generation_fixtures import make_retrieved


def _prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _has(pattern: str) -> bool:
    return re.search(pattern, _prompt(), flags=re.IGNORECASE | re.DOTALL) is not None


# The three #311 anchors, shared with the static-prefix coherence test.
EVERY_SENTENCE_ANCHOR = r"(every|each) factual sentence.{0,200}(citation|cite[sd]?)"
OMISSION_ANCHOR = (
    r"(omit\w*|leave\w* out|drop\w*).{0,160}(rather than|over|instead of|before)"
    r".{0,160}(uncited|without (a |its )?citation|no citation)"
)
IRRELEVANT_CITE_ANCHOR = (
    r"(never|do not|don'?t).{0,40}cite.{0,160}"
    r"(does not|doesn'?t|cannot|can'?t|fails? to).{0,60}support"
)


def test_prompt_demands_a_citation_on_every_factual_sentence():
    """#311 anchor 1: the citation duty is stated at SENTENCE level —
    "every factual sentence" carries a citation. The current claim-level
    phrasing licenses the observed per-claim-cluster citing style (256
    events over 524 sentences)."""
    assert _has(EVERY_SENTENCE_ANCHOR), (
        "the generation system prompt must state the citation duty per "
        "factual SENTENCE — anchor: 'every factual sentence' (or 'each "
        "factual sentence') tied to the citation duty"
    )


def test_prompt_prefers_omission_over_uncited_assertion():
    """#311 anchor 2: an uncitable sentence is dropped, not asserted —
    the model must prefer omitting a claim over stating it uncited."""
    assert _has(OMISSION_ANCHOR), (
        "the generation system prompt must state the omission preference "
        "— omit/drop a claim rather than assert it uncited"
    )


def test_prompt_forbids_citing_a_non_supporting_passage():
    """#311 anchor 3: relevance in prohibition form — never cite a
    passage that does not support the sentence (the qa-sp-01/qa-adv-04
    irrelevant-chunk citations are a generation-quality defect even when
    the judge correctly scores them unsupported)."""
    assert _has(IRRELEVANT_CITE_ANCHOR), (
        "the generation system prompt must forbid citing a passage that "
        "does not support the sentence — anchor: never/do not cite ... "
        "does not ... support"
    )


def test_prompt_does_not_licence_sparse_or_padded_citing():
    """Non-contradiction guards: no instruction may licence per-paragraph
    citing, one-citation-per-claim-cluster style, or citing passages for
    coverage rather than support."""
    text = _prompt().lower()
    for poison in (
        "one citation per paragraph",
        "cite once per paragraph",
        "a single citation is enough",
        "cite the most relevant passage once",
        "cite every passage you were given",
    ):
        assert poison not in text, poison


class TestDensityInstructionsRideTheStaticCachedPrefix:
    """The strengthened prompt composes with the caching contract: the
    density anchors live in the committed artifact (system block 0 — the
    static, cacheable prefix), and the artifact still clears Haiku 4.5's
    silent 4096-token cache floor."""

    def test_anchors_live_in_the_static_system_block(self):
        request = build_generation_request(
            make_retrieved(3),
            "How much has the invented basin warmed?",
            config=GenerationConfig(),
        )
        static_prefix = request["system"][0]["text"]
        assert static_prefix == load_system_prompt(), (
            "system block 0 must BE the committed artifact — density "
            "instructions never ride a volatile block"
        )
        flags = re.IGNORECASE | re.DOTALL
        for anchor in (EVERY_SENTENCE_ANCHOR, OMISSION_ANCHOR, IRRELEVANT_CITE_ANCHOR):
            assert re.search(anchor, static_prefix, flags=flags), anchor
        # Non-tone traffic still carries exactly the one static block —
        # nothing volatile was added ahead of or after it for density.
        assert len(request["system"]) == 1

    def test_strengthened_prompt_still_clears_the_cache_floor(self):
        assert estimate_tokens_lower_bound(load_system_prompt()) >= (
            HAIKU_MIN_CACHEABLE_PREFIX_TOKENS
        )
