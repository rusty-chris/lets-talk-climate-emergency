"""Pin: the non-English preamble_note NEVER rides into the generation
prompt (issue #12, orchestrator ratification of the red phase).

The ratification's ruling, verbatim in intent: *preamble_note is
response-surface furniture, NOT model input* — the one-line
answered-in-English note attaches at the service layer (#22) alongside
the footer. This is an additive red-first pin, explicitly authorized by
the orchestrator on issue #12.

Why it matters: the note is user-visible response furniture derived from
a classifier reading user-controlled text (finding #87). Injecting it
into the generation prompt would (a) add a volatile block that varies by
query language, silently splitting the byte-stable cache prefix, and
(b) open a channel from classifier output into the model's instructions.
The generation request builder takes only (RetrievedPassages, question,
config) — no QueryDecision — so the leak is structurally impossible;
this test pins that it STAYS impossible.
"""

from __future__ import annotations

import inspect
import json

from rag.generation import GenerationConfig, build_generation_request
from rag.provider import build_anthropic_messages_request
from rag.query import ENGLISH_ANSWER_NOTE
from tests._generation_fixtures import make_retrieved

QUESTION = "How much has the invented basin warmed?"


def test_preamble_note_never_enters_the_generation_request():
    """Neither the seam payload nor the Anthropic API request may carry
    the answered-in-English note anywhere — system, messages, documents,
    config."""
    seam = build_generation_request(
        make_retrieved(3, tone_flag=True), QUESTION, config=GenerationConfig()
    )
    assert ENGLISH_ANSWER_NOTE not in json.dumps(seam, ensure_ascii=False)

    api = build_anthropic_messages_request(seam)
    assert ENGLISH_ANSWER_NOTE not in json.dumps(api, ensure_ascii=False)


def test_generation_builder_accepts_no_decision_channel():
    """The builder's signature has no parameter through which a
    QueryDecision (and so a preamble_note) could arrive: the separation
    is structural, not behavioural."""
    parameter_names = set(inspect.signature(build_generation_request).parameters)
    assert parameter_names == {"retrieved", "question", "config"}
