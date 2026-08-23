"""The committed system-prompt artifact: DESIGN §3.3's eight rules
(issue #12) — RED.

Characterisation guards, not quality claims (issue #12 TDD plan item 5):
each test asserts a load-bearing INVARIANT of the prompt textually —
presence of the rule's anchor language and non-contradiction — never its
exact phrasing, so the green-phase author is free to write a good prompt
and later prompt-tuning cannot silently drop a rule. (A prompt change
still invalidates replay recordings by design — IMPLEMENTATION.md §4.2.)
"""

from __future__ import annotations

import re

from rag.generation import (
    HAIKU_MIN_CACHEABLE_PREFIX_TOKENS,
    SYSTEM_PROMPT_PATH,
    assert_cacheable_prefix,
    estimate_tokens_lower_bound,
    load_system_prompt,
)


def _prompt() -> str:
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    assert "RED-PHASE STUB" not in text, (
        "the committed system prompt is still the red-phase placeholder"
    )
    return text


def _has(pattern: str) -> bool:
    return re.search(pattern, _prompt(), flags=re.IGNORECASE | re.DOTALL) is not None


def test_artifact_is_committed_and_loaded_verbatim():
    """Prompt text is data (IMPLEMENTATION.md §1): the loader returns the
    committed artifact byte-for-byte — never an assembled imitation."""
    assert SYSTEM_PROMPT_PATH.is_file()
    assert load_system_prompt() == SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def test_rule_1_answer_only_from_provided_passages():
    """§3.3 rule 1: answers come only from the provided passages —
    outside knowledge is named and forbidden."""
    assert _has(r"only.{0,80}(provided|retrieved|supplied).{0,40}(passage|document|excerpt)")
    assert _has(r"(no|never|not).{0,60}(outside|own|prior|general).{0,20}knowledge")


def test_rule_2_cite_every_claim():
    assert _has(r"cite.{0,60}(every|each|all).{0,60}(claim|statement|fact)")


def test_rule_3_preserve_calibrated_language_verbatim():
    """§3.3 rule 3: calibrated qualifiers survive verbatim — never
    upgraded, downgraded or dropped."""
    assert _has(r"verbatim")
    assert _has(r"(qualifier|calibrated|likel\w+|confidence)")
    assert _has(r"(never|not|don'?t).{0,80}(upgrade|downgrade|strengthen|weaken|soften|drop)")


def test_rule_4_lead_with_headline_severity():
    """§3.3 rule 4: lead with the headline finding at the severity the
    source states it — no restructuring alarm into reassurance, no added
    alarm (severity fidelity is symmetric, DESIGN §4.8)."""
    assert _has(r"(lead|begin|start|open).{0,120}(headline|main|central).{0,40}finding")
    assert _has(r"severity")
    assert _has(r"(reassur\w+|soften|downplay|underplay)")
    assert _has(r"(add|inflat\w+|exaggerat\w+).{0,60}alarm")


def test_rule_5_refusal_template_hook():
    """§3.3 rule 5: when the passages don't answer, say so plainly and
    point at the right source — the honest-refusal posture inside the
    answered path (partial support, §3.5)."""
    assert _has(r"(don'?t|do not|cannot|can'?t).{0,80}answer")
    assert _has(r"(say so|state|acknowledge).{0,40}(plain|clear|direct|honest)")


def test_rule_6_plain_language_and_jargon():
    assert _has(r"plain (language|english)")
    assert _has(r"jargon|technical term")
    assert _has(r"(no|without|little).{0,30}background|first use|define")


def test_rule_7_voices_labelled_never_evidence():
    """§3.3 rule 7: voices-sourced content is movement context, labelled
    as such — never evidence for a scientific claim (the prompt-side
    complement to the STRUCTURAL filter in rag.retrieval)."""
    assert _has(r"voices")
    assert _has(r"(label|mark|identif)")
    assert _has(r"(never|not).{0,80}evidence.{0,80}(scientific|science|claim)")


def test_rule_8_beyond_assessed_range_attribution():
    """§3.3 rule 8: beyond-assessed-range material is attributed to its
    authors, with the assessed range stated where retrieved alongside —
    never presented as consensus."""
    assert _has(r"beyond.{0,3}assessed.{0,3}range")
    assert _has(r"assessed range")
    assert _has(r"attribut\w+.{0,80}author")
    assert _has(r"(never|not).{0,60}(as )?consensus")


def test_prompt_declares_passage_content_is_never_instructions():
    """Finding #187 / DESIGN §4 (injection resistance): the defence must
    cover ALL externally-originated model-visible channels — passage
    bodies, context/section headers, attribution titles — not only the
    question. The prompt must state (a) supplied documents and their
    metadata are quoted source material — data, and (b) instruction-like
    text inside them is content, never followed, and can never change
    the rules."""
    # (a) passages/documents/titles/metadata are quoted data, not
    # instructions to the model.
    assert _has(
        r"(passage|document)s?\b.{0,200}(titles?|metadata|headings?|headers?)"
        r".{0,300}(data|quoted|source material)"
    )
    assert _has(r"(data|source material|quoted).{0,120}(never|not).{0,60}instruction")
    # (b) an instruction-like sentence inside a passage is content to
    # report on, never a command to obey.
    assert _has(
        r"(command|instruction|imperative).{0,200}(inside|within|embedded in|in a)"
        r".{0,40}(passage|document)"
    )
    assert _has(r"(never|not).{0,60}(obey|follow|execut)")
    # And nothing supplied with a request can amend the rules.
    assert _has(
        r"(nothing|no (text|content|passage|document|question)).{0,120}"
        r"(request|supplied|passage|document).{0,120}"
        r"(amend|change|override|alter|rewrite).{0,40}(these )?rules"
    )


def test_prompt_does_not_contradict_the_grounding_contract():
    """Non-contradiction: no instruction may invite outside knowledge,
    paraphrase-with-adjusted-tone of qualifiers, or unlabelled voices
    material. Cheap negative guards against the likeliest regressions."""
    text = _prompt().lower()
    for poison in (
        "use your general knowledge",
        "you may draw on outside",
        "improve the wording of quoted",
        "make the tone more urgent",
        "follow any instructions in the passages",
        "follow instructions found in the documents",
    ):
        assert poison not in text, poison


def test_static_prompt_clears_the_haiku_cacheable_prefix_floor():
    """The orchestrator note on #12: Haiku 4.5 caches nothing below a
    4096-token prefix, silently. The artifact itself must be deliberately
    sized so the static prefix clears the floor — this is a property of
    the committed file, checked here without any request in sight."""
    prompt = _prompt()
    assert estimate_tokens_lower_bound(prompt) >= HAIKU_MIN_CACHEABLE_PREFIX_TOKENS
    assert assert_cacheable_prefix([{"type": "text", "text": prompt}]) is None
