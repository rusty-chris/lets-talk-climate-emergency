"""Citation precision 94.4% vs 95% — the 19 unsupported pairs (finding #340) — RED.

Release run 2 measured citation_entailment_precision 321/340 = 94.4%
against the owner-ratified 0.95 bar. Recomputing every failing pair from
the run-2 journals (SSE transcripts + per-pair verdicts + the run's own
indexed chunk bodies — local, $0) classifies the 19 unsupported attached
sentences:

- **10 judge strictness**: the substantive claim content is verbatim or a
  faithful paraphrase of the cited source, rejected over discourse
  framing or rewording (e.g. qa-adv-03 "What happened next is clear:
  2014, 2015 and 2016 were all significantly hotter…" — the source
  states the fact word-for-word; qa-sp-13's "beyond the assessed
  consensus range" gloss of Hansen — the label the gold set REQUIRES —
  judged against a source saying "in contradiction to conclusions of
  IPCC").
- **5 pairing/segmentation artifacts**: composite "sentences" produced by
  the answer segmenter gluing text across markdown paragraph boundaries
  (qa-mp-01: a bold **lead** absorbed; qa-mp-09: a ``##`` heading
  absorbed; qa-sev-10: a quote-terminated sentence not split), plus one
  bare "Yes." isolated by block-span attachment (qa-sp-06) and one
  multi-source synthesis sentence whose two cited sources each entail
  half (qa-sp-14) — no single source can entail a glued composite.
- **4 genuine goes-beyond additions**: an unentailed clause added to
  otherwise-supported content (qa-sp-02, qa-sev-06, qa-sev-12 s3,
  qa-sev-15).

A mix near the bar (2 sentences short of 323/340 = 95.0%; the deficit is
0.5 SE — binomial P(X<=321 | p=0.95, n=340) = 0.34), so NO threshold or
policy change rides here; the split and variance arithmetic go to the
owner. What this suite DOES pin is the one **mechanical** defect in the
artifact class — the answer segmenter's sentence rule
(``(?<=[.!?])\\s+(?=[A-Z])``) only splits when the next character is an
uppercase letter, so a sentence followed by a markdown heading (``##``),
a bold lead (``**``), or ending inside a closing quote is glued into one
composite claim that no source can entail — plus honesty guards.

Scoping note for the implementer: the segmenter reuses
``ingestion.pipeline.split_sentences``, which the INGESTION pipeline also
consumes (chunk statements/headlines) — fixing the shared rule may move
chunk boundaries and require a gold chunk-id re-pin. These tests pin the
behaviour at the ANSWER-segmentation seam (``segment_answer_sentences``),
leaving the fix's location (shared rule vs validator-side) to
implementation.

Deliberately NOT pinned (owner/design-level; report only): the
cited-sentence-is-always-factual rule that forces "Yes." into the pool
(#312's policy), the any-single-source entailment rule that fails
multi-source synthesis (a #325 rubric property), any judge-prompt
loosening (the anti-injection and never-use-outside-knowledge framings
must survive as-is), and the 0.95 bar itself — pinned UNCHANGED below.
"""

from __future__ import annotations

from evals.gates import CITATION_ENTAILMENT_PRECISION_THRESHOLD
from rag.citation_validator import (
    ValidatorConfig,
    build_validation_request,
    segment_answer_sentences,
)
from tests._citation_validator_fixtures import text_event, transcript
from tests.unit.test_citation_validator_request import make_pairs

# ---------------------------------------------------------------------------
# 1. Answer segmentation: markdown/quote boundaries end sentences (RED)
# ---------------------------------------------------------------------------

#: Invented answer text reproducing the qa-mp-09 composite shape: sentence,
#: blank line, ATX heading, blank line, sentence — glued into ONE claim in
#: the run because the splitter demands an uppercase letter after the stop.
HEADING_GLUED_TEXT = (
    "The committed rise in sea level is already locked in by past emissions.\n\n"
    "## What the invented projections show\n\n"
    "Tide gauges in the study recorded a 0.2 metre rise since 1901."
)

#: The qa-mp-01 composite shape: sentence, blank line, bold paragraph lead.
BOLD_LEAD_GLUED_TEXT = (
    "Volcanic outgassing contributes fifty times less carbon dioxide than industry.\n\n"
    "**Emissions match the observed warming.** Natural factors alone would have "
    "produced a slight cooling over the invented study period."
)

#: The qa-sev-10 composite shape: a sentence whose full stop sits inside a
#: closing quote, followed by an ordinary sentence.
QUOTE_TERMINATED_TEXT = (
    'The assessment distinguishes "too late to avoid all change." '
    "Further warming still responds to the choices made this decade."
)


def _sentence_texts(full_text: str) -> list[str]:
    events = transcript(text_event(full_text[:29]), text_event(full_text[29:]))
    return [s.text for s in segment_answer_sentences(events)]


class TestMarkdownAwareSentenceBoundaries:
    def test_heading_paragraph_break_ends_the_sentence(self):
        """Run 2's qa-mp-09 failure shape: '<sentence>.\\n\\n## <heading>
        \\n\\n<sentence>.' must never segment as one claim — the leading
        sentence and the trailing sentence are separate entailment
        units, so a verbatim-supported trailing sentence cannot be
        failed by heading glue."""
        texts = _sentence_texts(HEADING_GLUED_TEXT)
        assert len(texts) >= 2, f"heading-glued composite: {texts!r}"
        glued = [t for t in texts if "locked in" in t and "Tide gauges" in t]
        assert not glued, (
            "a markdown heading paragraph break must end the sentence — the "
            "composite claim is unentailable by any single source (issue #340, "
            "qa-mp-09's failing pair shape)"
        )

    def test_bold_lead_paragraph_break_ends_the_sentence(self):
        """qa-mp-01's shape: '<sentence>.\\n\\n**<lead>.** <sentence>.'
        — the pre-break sentence must not swallow the bold lead and the
        following sentence."""
        texts = _sentence_texts(BOLD_LEAD_GLUED_TEXT)
        assert len(texts) >= 2, f"bold-lead composite: {texts!r}"
        glued = [t for t in texts if "Volcanic outgassing" in t and "Natural factors" in t]
        assert not glued, (
            "a bold paragraph lead after a blank line must start a new "
            "sentence (issue #340, qa-mp-01's failing pair shape)"
        )

    def test_quote_terminated_sentence_splits(self):
        """qa-sev-10's shape: a full stop inside a closing double quote
        ends the sentence; the following sentence is a separate claim."""
        texts = _sentence_texts(QUOTE_TERMINATED_TEXT)
        assert len(texts) == 2, f"quote-terminated composite: {texts!r}"

    def test_abbreviation_and_decimal_guards_survive(self):
        """Non-regression: the documented non-splits stay — 'Fig. 3'
        and decimals never produce false boundaries while the markdown
        boundaries are added."""
        texts = _sentence_texts(
            "Warming reached 1.9 degrees in the invented basin, per Fig. 3 of the "
            "assessment. Reservoir inflows declined over the following decade."
        )
        assert len(texts) == 2
        assert "Fig. 3" in texts[0]
        assert "1.9" in texts[0]


# ---------------------------------------------------------------------------
# 2. Honesty and threshold guards (GREEN — must stay green through any fix)
# ---------------------------------------------------------------------------


def _judge_system_prompt() -> str:
    return str(build_validation_request(make_pairs(2), config=ValidatorConfig())["system"])


# NOTE (issue #340 adjudication): the judge-prompt calibration red
# (``TestEntailmentJudgeCalibration.test_prompt_frames_entailment_over_wording``)
# was DROPPED by the orchestrator's run-2 ratification comment on #338:
# "changing the entailment judge at the exact bar it measures is
# metric-gaming risk; the artifacts alone clear the gate … Implementer
# removes that one red test citing this comment." The three segmentation
# reds above stand; the honesty/threshold guards below stay green.


class TestEntailmentJudgeGuards:
    def test_gate_threshold_is_untouched(self):
        """The owner-ratified bar moves only with owner sign-off: any
        change to 0.95 in a #340 fix is out of bounds (the issue's own
        rule)."""
        assert CITATION_ENTAILMENT_PRECISION_THRESHOLD == 0.95

    def test_anti_injection_framing_survives(self):
        """The judge prompt's injection armour is load-bearing (claim and
        source texts are attacker-influenced): the quoted-material /
        never-followed framing must survive any calibration."""
        prompt = _judge_system_prompt().lower()
        assert "never" in prompt
        assert "followed" in prompt or "obeyed" in prompt or "executed" in prompt
        assert "data" in prompt or "quoted material" in prompt

    def test_goes_beyond_honesty_rule_survives(self):
        """The honesty core: a claim that goes beyond or contradicts its
        source stays NOT supported — the genuine goes-beyond additions
        among the 19 (qa-sp-02, qa-sev-06, qa-sev-12, qa-sev-15) must
        keep failing."""
        prompt = _judge_system_prompt().lower()
        assert "goes beyond" in prompt
        assert "contradict" in prompt

    def test_no_benefit_of_the_doubt_language(self):
        """Non-contradiction guard: no calibration may tell the judge to
        default to supported or give claims the benefit of the doubt."""
        prompt = _judge_system_prompt().lower()
        for poison in (
            "benefit of the doubt",
            "when unsure, mark supported",
            "default to supported",
            "lean towards supported",
        ):
            assert poison not in prompt, poison
