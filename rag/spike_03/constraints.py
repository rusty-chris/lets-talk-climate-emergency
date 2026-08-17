"""PROTOTYPE (issue #3 spike) — live §3.4 native-citations constraint probes.

Observes, against the live API, the constraints DESIGN §3.4 asserts and that #12's
contract tests will pin on the request builders. Rejected (400) requests are NOT
billed, so these are ~free. Run::

    uv run python -m rag.spike_03.run_probe --constraints

Records:
* structured output + citations -> 400 (citations incompatible with output_config);
* mixed cited / uncited document blocks -> 400 (all-or-none citations).
"""

from __future__ import annotations

import anthropic

MODEL = "claude-haiku-4-5"

_DOC_A = {
    "type": "document",
    "source": {"type": "content", "content": [{"type": "text", "text": "The sky is blue."}]},
    "title": "A",
    "citations": {"enabled": True},
}
_DOC_B_NOCITE = {
    "type": "document",
    "source": {"type": "content", "content": [{"type": "text", "text": "The grass is green."}]},
    "title": "B",
    # deliberately NO citations key -> mixed cited/uncited in one request
}


def _probe(label: str, **kwargs) -> dict:
    client = anthropic.Anthropic()
    try:
        client.messages.create(model=MODEL, max_tokens=64, **kwargs)
        return {"label": label, "outcome": "UNEXPECTED 200 (request succeeded)", "status": 200}
    except anthropic.BadRequestError as e:
        return {
            "label": label,
            "outcome": "400 as expected",
            "status": 400,
            "message": str(e.message)[:300],
        }
    except anthropic.APIStatusError as e:  # some other status
        return {
            "label": label,
            "outcome": f"{e.status_code}",
            "status": e.status_code,
            "message": str(e.message)[:300],
        }


def run_constraints() -> list[dict]:
    results = []
    # 1. structured output on the citations call -> 400
    results.append(
        _probe(
            "structured_output_with_citations",
            system="Answer from the document.",
            messages=[
                {
                    "role": "user",
                    "content": [_DOC_A, {"type": "text", "text": "What color is the sky?"}],
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"color": {"type": "string"}},
                        "required": ["color"],
                        "additionalProperties": False,
                    },
                }
            },
        )
    )
    # 2. mixed cited/uncited documents -> 400 (all-or-none)
    results.append(
        _probe(
            "mixed_cited_and_uncited_documents",
            messages=[
                {
                    "role": "user",
                    "content": [
                        _DOC_A,
                        _DOC_B_NOCITE,
                        {"type": "text", "text": "Describe both."},
                    ],
                }
            ],
        )
    )
    print("=== §3.4 CONSTRAINT PROBES (live; 400s not billed) ===")
    for r in results:
        print(f"- {r['label']}: {r['outcome']}")
        if r.get("message"):
            print(f"    -> {r['message']}")
    return results


if __name__ == "__main__":
    run_constraints()
