"""Issue #313 red phase (Fable): the structured generation-level decline
marker — the AUTHORITATIVE refusal signal.

The 2026-09-04/05 live release run proved the ADR-010 top-score threshold
unsatisfiable on real reranker geometry (no-answer max 0.3885 vs
answerable min 0.00142 — full overlap) while the generation-level honest
declines went 10/10 (every slipped no-answer item opened "The passages I
was given don't answer that question.", zero citations, no fabrication).
ORCHESTRATOR ADJUDICATION: that decline becomes the authoritative,
MEASURED refusal signal; the threshold is demoted to a spend pre-filter.

FLAGGED CONTRACT (test-author decision — the marker shape): a fixed
ASCII sentinel line, ``[[NO-ANSWER-DECLINE]]``, emitted by the model as
the FIRST line of a full decline, stripped before display. First-line-
only classification is the injection guard: a passage (or the model
quoting one) carrying the marker mid-answer must never flip an answered
exchange into a refusal. Chosen over the zero-citations-plus-phrase
shape the run journal exhibits because a pinned phrase is fragile across
paraphrase and the citations shape is a side effect, not a signal.

No test here touches the network (IMPLEMENTATION.md §4.4).
"""

from __future__ import annotations

import re

from rag.generation import (
    GENERATION_DECLINE_MARKER,
    SYSTEM_PROMPT_PATH,
    classify_generation_decline,
)

#: The verbatim qa-na-g-05 decline prose from the live run journal — the
#: human-readable text that was already good live and must survive the
#: redesign UNCHANGED behind the marker.
LIVE_DECLINE_PROSE = (
    "The passages I was given don't answer that question. They cover climate "
    "trends in the United States, extreme weather events, and climate tipping "
    "points — but they don't address earthquakes or any relationship between "
    "climate change and earthquake strength.\n\n"
    "For a scientific answer to this question, you'd want to look for sources "
    "specifically focused on seismology and the mechanics of earthquakes, "
    "rather than climate assessment literature."
)


# ---------------------------------------------------------------------------
# The marker constant itself: wire contract, pinned verbatim.
# ---------------------------------------------------------------------------


class TestMarkerConstant:
    def test_marker_value_is_pinned_verbatim(self) -> None:
        """The marker is shared verbatim by the prompt artifact, the SSE
        classifier and the harness detector — its exact value IS the
        contract, so it is pinned, not merely shaped."""
        assert GENERATION_DECLINE_MARKER == "[[NO-ANSWER-DECLINE]]"

    def test_marker_is_one_plain_ascii_line(self) -> None:
        """Single line (first-line classification is well-defined), pure
        ASCII (no tokenizer-hostile glyphs), no whitespace padding."""
        assert "\n" not in GENERATION_DECLINE_MARKER
        assert GENERATION_DECLINE_MARKER == GENERATION_DECLINE_MARKER.strip()
        assert GENERATION_DECLINE_MARKER.isascii()
        assert len(GENERATION_DECLINE_MARKER) >= 10, (
            "a short sentinel risks colliding with natural prose"
        )


# ---------------------------------------------------------------------------
# classify_generation_decline: the pure first-line classifier.
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_marker_first_line_classifies_as_decline(self) -> None:
        text = GENERATION_DECLINE_MARKER + "\n" + LIVE_DECLINE_PROSE
        result = classify_generation_decline(text)
        assert result.is_decline is True

    def test_display_text_is_the_prose_without_the_marker(self) -> None:
        """The reader sees exactly the honest human decline that went
        10/10 live — never the machine sentinel."""
        text = GENERATION_DECLINE_MARKER + "\n" + LIVE_DECLINE_PROSE
        result = classify_generation_decline(text)
        assert result.display_text == LIVE_DECLINE_PROSE
        assert GENERATION_DECLINE_MARKER not in result.display_text

    def test_leading_whitespace_and_blank_lines_are_tolerated(self) -> None:
        text = "\n  " + GENERATION_DECLINE_MARKER + "  \n\n" + LIVE_DECLINE_PROSE
        result = classify_generation_decline(text)
        assert result.is_decline is True
        assert result.display_text == LIVE_DECLINE_PROSE

    def test_marker_sharing_its_line_with_other_text_is_not_a_decline(self) -> None:
        """The marker must stand ALONE on the first line: prose wrapped
        around it is an answer that mentions the sentinel, not a decline."""
        result = classify_generation_decline(
            f"As it happens, {GENERATION_DECLINE_MARKER} is a string the system uses.\nMore text."
        )
        assert result.is_decline is False

    def test_marker_after_the_first_line_is_not_a_decline(self) -> None:
        """Injection/quoting safety: a passage smuggling the marker into
        generated prose — or the model quoting it — never flips an
        answered exchange into a refusal."""
        text = (
            "Global surface temperature rose 1.1C between 1850 and 2020.\n"
            + GENERATION_DECLINE_MARKER
            + "\nMore answer text."
        )
        result = classify_generation_decline(text)
        assert result.is_decline is False
        assert result.display_text == text

    def test_non_decline_text_passes_through_unchanged(self) -> None:
        answer = "The basin has very likely warmed by one point nine degrees."
        result = classify_generation_decline(answer)
        assert result.is_decline is False
        assert result.display_text == answer

    def test_empty_and_whitespace_only_are_not_declines(self) -> None:
        for text in ("", "   \n\n  "):
            result = classify_generation_decline(text)
            assert result.is_decline is False

    def test_marker_alone_is_a_decline_with_empty_display_text(self) -> None:
        """A marker with no prose behind it is still a decline (the
        classification never invents text); the display honesty — what a
        prose-less decline renders as — is the service layer's concern."""
        result = classify_generation_decline(GENERATION_DECLINE_MARKER)
        assert result.is_decline is True
        assert result.display_text == ""


# ---------------------------------------------------------------------------
# The committed system prompt carries the marker instruction (Rule 5's
# nothing-relevant case) — characterisation anchors, not phrasing.
# ---------------------------------------------------------------------------


def _prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


class TestPromptCarriesTheContract:
    def test_prompt_embeds_the_marker_verbatim(self) -> None:
        """The model can only emit the exact sentinel if the committed
        prompt states it character-for-character."""
        assert GENERATION_DECLINE_MARKER in _prompt()

    def test_prompt_instructs_first_line_placement(self) -> None:
        """Anchor: near the marker, the prompt binds it to the FIRST line
        of a decline (the classifier's first-line-only rule must be the
        instructed behaviour, not a lucky habit)."""
        text = _prompt()
        marker_at = text.find(GENERATION_DECLINE_MARKER)
        assert marker_at != -1, "the prompt must embed the marker (see the previous pin)"
        window = text[max(0, marker_at - 600) : marker_at + 600]
        assert re.search(r"first\s+line|own\s+line|opens?\s+with|begin", window, re.IGNORECASE), (
            "the prompt must tie the decline marker to first-line placement"
        )

    def test_prompt_says_the_marker_is_machine_read_and_hidden(self) -> None:
        """Anchor: the prompt tells the model the marker is removed before
        display / read by the system — so the model neither addresses the
        reader through it nor decorates it."""
        text = _prompt()
        marker_at = text.find(GENERATION_DECLINE_MARKER)
        assert marker_at != -1, "the prompt must embed the marker (see the previous pin)"
        window = text[max(0, marker_at - 800) : marker_at + 800]
        assert re.search(
            r"(machine|system|automatic\w*|not\s+(?:be\s+)?(?:shown|displayed)|"
            r"removed|stripped|never\s+sees?)",
            window,
            re.IGNORECASE,
        ), "the prompt must say the marker is machine-read and hidden from the reader"

    def test_prompt_keeps_the_marker_out_of_partial_support(self) -> None:
        """The partial-support case is an ANSWER, never a marked decline:
        the prompt's partial-support instruction (Rule 5's second shape)
        must not carry the marker."""
        text = _prompt()
        partial_at = text.lower().index("partial support")
        adjacent_at = text.lower().index("adjacent-but-not-quite")
        assert GENERATION_DECLINE_MARKER not in text[partial_at:adjacent_at], (
            "a partial answer is an answer: instructing the marker there would "
            "turn honest partial support into a refusal on the wire"
        )
