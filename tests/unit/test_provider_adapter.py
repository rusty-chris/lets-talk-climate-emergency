"""Behaviour tests for the fake/replay/recording provider adapters (issue #24).

IMPLEMENTATION.md §4: LLM calls are non-deterministic, slow and cost money, so
nothing in the unit or integration tiers may make one. These tests pin the
seams every LLM-dependent issue (#10, #12, #13, #16, #21) builds on:

- FakeAdapter returns programmed responses, records every call (method + full
  request payload), and supports response *sequences* for retry-path tests.
- ReplayAdapter replays checked-in fixtures keyed by a canonical request hash
  and fails loudly — naming the hash and the re-record command — on any
  unrecorded request, so a changed prompt invalidates its recordings by design.
- RecordingAdapter refuses to run without the explicit env flag AND a live
  key, and scrubs API keys / auth headers from everything it stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.provider import (
    RE_RECORD_COMMAND,
    AnswerWithCitations,
    Citation,
    FakeAdapter,
    FakeAdapterExhaustedError,
    ReplayAdapter,
    ReplayFixtureMissingError,
    canonical_request_hash,
)

GENERATE_PAYLOAD = {
    "messages": [{"role": "user", "content": "How much has the Aurelian Basin warmed?"}],
    "documents": [
        {
            "title": "SYNTHETIC doc",
            "content": [{"type": "text", "text": "Warming of 1.9 C is very likely."}],
            "citations": {"enabled": True},
        }
    ],
    "config": {"model": "claude-haiku-4-5", "max_tokens": 1024},
}

STRUCTURED_PAYLOAD = {
    "messages": [{"role": "user", "content": "classify this"}],
    "schema": {"type": "object", "properties": {"scope": {"type": "string"}}},
    "config": {"model": "claude-haiku-4-5"},
}


def _answer(text: str = "It warmed by 1.9 C.") -> AnswerWithCitations:
    return AnswerWithCitations(
        text=text,
        citations=(Citation(cited_text="Warming of 1.9 C is very likely.", document_index=0),),
    )


class TestFakeAdapter:
    def test_fake_adapter_returns_programmed_response(self):
        """TDD plan item 1: programmed response comes back; the call is recorded

        with its method name and the *full* request payload, so tests can
        assert on exactly what our code sent to the model seam.
        """
        answer = _answer()
        fake = FakeAdapter(generate_results=[answer])

        result = fake.generate(**GENERATE_PAYLOAD)

        assert result is answer
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call.method == "generate"
        assert call.payload == GENERATE_PAYLOAD

    def test_fake_adapter_records_calls_across_methods(self):
        """Recording covers every protocol method, in call order."""
        fake = FakeAdapter(
            generate_results=[_answer()],
            structured_results=[{"scope": "in_scope"}],
            plan_chart_results=[{"chart_type": "line"}],
        )

        fake.structured(**STRUCTURED_PAYLOAD)
        fake.generate(**GENERATE_PAYLOAD)
        fake.plan_chart(request="plot synthetic co2", catalog={"datasets": []})

        assert [c.method for c in fake.calls] == ["structured", "generate", "plan_chart"]
        assert fake.calls[0].payload == STRUCTURED_PAYLOAD
        assert fake.calls[2].payload == {
            "request": "plot synthetic co2",
            "catalog": {"datasets": []},
        }

    def test_fake_adapter_supports_response_sequences(self):
        """TDD plan item 2: sequential programmed responses for retry paths (#16).

        Each call consumes the next programmed response in order; a call past
        the end of the sequence raises, so a retry-once test fails loudly if
        the code under test retries twice.
        """
        fake = FakeAdapter(structured_results=[{"attempt": 1}, {"attempt": 2}])

        assert fake.structured(**STRUCTURED_PAYLOAD) == {"attempt": 1}
        assert fake.structured(**STRUCTURED_PAYLOAD) == {"attempt": 2}
        with pytest.raises(FakeAdapterExhaustedError):
            fake.structured(**STRUCTURED_PAYLOAD)
        # The exhausted call is still recorded — the call log stays truthful.
        assert [c.method for c in fake.calls] == ["structured"] * 3

    def test_fake_adapter_sequence_can_raise_programmed_exceptions(self):
        """A programmed exception instance is raised, for failure-path tests."""
        boom = RuntimeError("synthetic transport failure")
        fake = FakeAdapter(generate_results=[boom, _answer("recovered")])

        with pytest.raises(RuntimeError, match="synthetic transport failure"):
            fake.generate(**GENERATE_PAYLOAD)
        assert fake.generate(**GENERATE_PAYLOAD).text == "recovered"

    def test_fake_adapter_conftest_fixture_injects_unprogrammed_adapter(self, fake_adapter):
        """The shared conftest fixture provides a FakeAdapter for injection;

        unprogrammed, any call fails loudly (zero-call assertions stay honest).
        """
        assert isinstance(fake_adapter, FakeAdapter)
        assert fake_adapter.calls == []
        with pytest.raises(FakeAdapterExhaustedError):
            fake_adapter.structured(**STRUCTURED_PAYLOAD)


def _write_replay_fixture(fixtures_dir: Path, method: str, payload: dict, response: dict) -> Path:
    """Write a replay fixture in the documented on-disk format.

    Written by hand (not through the recorder) so these tests pin the format
    itself: JSON file named `<canonical request hash>.json` containing the
    method, the scrubbed request, and a typed response.
    """
    fixture_path = fixtures_dir / f"{canonical_request_hash(method, payload)}.json"
    fixture_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "marker": "SYNTHETIC FIXTURE — authored for this project's tests",
                    "scrubbed": True,
                },
                "method": method,
                "request": payload,
                "response": response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return fixture_path


class TestCanonicalRequestHash:
    def test_hash_is_stable_under_dict_key_order(self):
        """The hash canonicalises the payload: key order must not matter,

        or a semantically identical request would miss its recording.
        """
        a = {"messages": [{"role": "user", "content": "q"}], "config": {"model": "m"}}
        b = {"config": {"model": "m"}, "messages": [{"content": "q", "role": "user"}]}
        assert canonical_request_hash("structured", a) == canonical_request_hash("structured", b)

    def test_hash_changes_when_prompt_or_method_changes(self):
        """A changed prompt (or method) invalidates its recordings by design."""
        base = {"messages": [{"role": "user", "content": "original prompt"}]}
        changed = {"messages": [{"role": "user", "content": "edited prompt"}]}
        assert canonical_request_hash("structured", base) != canonical_request_hash(
            "structured", changed
        )
        assert canonical_request_hash("structured", base) != canonical_request_hash(
            "generate", base
        )


class TestReplayAdapter:
    def test_replay_adapter_returns_recorded_response_for_matching_request(self, tmp_path):
        """TDD plan item 3: canonical request-hash lookup returns the recording."""
        _write_replay_fixture(
            tmp_path,
            "structured",
            STRUCTURED_PAYLOAD,
            {"type": "dict", "value": {"scope": "in_scope"}},
        )
        replay = ReplayAdapter(tmp_path)

        assert replay.structured(**STRUCTURED_PAYLOAD) == {"scope": "in_scope"}

    def test_replay_adapter_reconstructs_answer_with_citations(self, tmp_path):
        """Recorded generate responses come back as AnswerWithCitations objects."""
        _write_replay_fixture(
            tmp_path,
            "generate",
            GENERATE_PAYLOAD,
            {
                "type": "answer_with_citations",
                "text": "It warmed by 1.9 C.",
                "citations": [
                    {
                        "cited_text": "Warming of 1.9 C is very likely.",
                        "document_index": 0,
                        "document_title": "SYNTHETIC doc",
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            },
        )
        replay = ReplayAdapter(tmp_path)

        answer = replay.generate(**GENERATE_PAYLOAD)

        assert isinstance(answer, AnswerWithCitations)
        assert answer.text == "It warmed by 1.9 C."
        assert answer.citations == (
            Citation(
                cited_text="Warming of 1.9 C is very likely.",
                document_index=0,
                document_title="SYNTHETIC doc",
                start_block_index=0,
                end_block_index=1,
            ),
        )

    def test_replay_adapter_raises_on_unrecorded_request(self, tmp_path):
        """TDD plan item 4: loud failure naming the request hash and the

        re-record command — a changed prompt must invalidate its recordings
        by design, never silently fall through to a stale or generic response.
        """
        replay = ReplayAdapter(tmp_path)
        expected_hash = canonical_request_hash("structured", STRUCTURED_PAYLOAD)

        with pytest.raises(ReplayFixtureMissingError) as excinfo:
            replay.structured(**STRUCTURED_PAYLOAD)

        message = str(excinfo.value)
        assert expected_hash in message
        assert "structured" in message
        assert RE_RECORD_COMMAND in message
        assert "CLIMATE_CHAT_RECORD" in message
