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
    MAX_GENERATE_DOCUMENTS,
    RE_RECORD_COMMAND,
    REPLAY_FIXTURE_FORMAT_VERSION,
    AnswerWithCitations,
    CanonicalisationError,
    Citation,
    FakeAdapter,
    FakeAdapterExhaustedError,
    ProviderContractError,
    RawProviderResponse,
    RecordingAdapter,
    RecordingDisabledError,
    ReplayAdapter,
    ReplayFixtureMissingError,
    ReplayFormatError,
    SecretLeakError,
    StructuredResult,
    canonical_request_hash,
    deserialize_response,
    scrub_payload,
    serialize_response,
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
        )

        fake.structured(**STRUCTURED_PAYLOAD)
        fake.generate(**GENERATE_PAYLOAD)

        assert [c.method for c in fake.calls] == ["structured", "generate"]
        assert fake.calls[0].payload == STRUCTURED_PAYLOAD

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

    def test_recorded_call_payload_is_immune_to_caller_mutation(self):
        """Review finding #69: the call log must stay truthful when code under

        test mutates its message/document structures *after* the call — the
        exact #16 retry pattern (append a correction message between
        attempts) and #10's threaded conversation-context lists. A shallow
        copy would let the second attempt retroactively rewrite the first
        recorded call.
        """
        fake = FakeAdapter(structured_results=[{"attempt": 1}, {"attempt": 2}])
        messages = [{"role": "user", "content": "classify this"}]
        schema = {"type": "object", "properties": {"scope": {"type": "string"}}}
        config = {"model": "claude-haiku-4-5"}

        fake.structured(messages=messages, schema=schema, config=config)

        # The #16 retry shape: append a correction and mutate nested state,
        # then call again.
        messages.append({"role": "user", "content": "your output was invalid; retry"})
        schema["properties"]["scope"]["type"] = "integer"
        config["model"] = "mutated-model"
        fake.structured(messages=messages, schema=schema, config=config)

        first = fake.calls[0].payload
        assert first["messages"] == [{"role": "user", "content": "classify this"}], (
            "first recorded call shows a message that had not been sent yet"
        )
        assert first["schema"]["properties"]["scope"]["type"] == "string"
        assert first["config"]["model"] == "claude-haiku-4-5"

    def test_fake_adapter_conftest_fixture_injects_unprogrammed_adapter(self, fake_adapter):
        """The shared conftest fixture provides a FakeAdapter for injection;

        unprogrammed, any call fails loudly (zero-call assertions stay honest).
        """
        assert isinstance(fake_adapter, FakeAdapter)
        assert fake_adapter.calls == []
        with pytest.raises(FakeAdapterExhaustedError):
            fake_adapter.structured(**STRUCTURED_PAYLOAD)


def _cited_doc(text: str = "Warming of 1.9 C is very likely.") -> dict:
    return {
        "title": "SYNTHETIC doc",
        "content": [{"type": "text", "text": text}],
        "citations": {"enabled": True},
    }


class TestProviderContract:
    """Review finding #62: the fakes must enforce the DESIGN §3.4 constraints

    the live API enforces (or the design demands), so a test suite that is
    green against FakeAdapter/ReplayAdapter can never hide a request that
    would 400 (or silently degrade) against the real AnthropicAdapter. The
    §4.3 builder contract tests (#10/#13) remain; this is the seam-level
    backstop that makes them non-bypassable.
    """

    def test_fake_generate_rejects_more_than_eight_documents(self):
        """§3.4: generation call documents bounded to reranked top-8."""
        fake = FakeAdapter(generate_results=[_answer()])
        nine_docs = [_cited_doc(f"synthetic passage {i}") for i in range(9)]

        with pytest.raises(ProviderContractError, match="8"):
            fake.generate(
                messages=GENERATE_PAYLOAD["messages"],
                documents=nine_docs,
                config=GENERATE_PAYLOAD["config"],
            )
        # The programmed response was NOT consumed: a subsequent valid call
        # still receives it, so the validator fires before the queue.
        assert fake.generate(**GENERATE_PAYLOAD) == _answer(), (
            "contract violation must not consume a programmed response"
        )
        assert MAX_GENERATE_DOCUMENTS == 8

    def test_fake_generate_rejects_uncited_or_mixed_document_blocks(self):
        """§3.4: all-or-none citations — every document block cited.

        The live API 400s on mixed cited/uncited blocks, and the design
        demands every block cited; a single uncited block must raise.
        """
        fake = FakeAdapter(generate_results=[_answer(), _answer()])
        uncited = {"title": "SYNTHETIC doc", "content": [{"type": "text", "text": "x"}]}
        disabled = dict(_cited_doc(), citations={"enabled": False})

        for bad_docs in ([uncited], [_cited_doc(), uncited], [disabled]):
            with pytest.raises(ProviderContractError, match="citations"):
                fake.generate(
                    messages=GENERATE_PAYLOAD["messages"],
                    documents=bad_docs,
                    config=GENERATE_PAYLOAD["config"],
                )

    def test_fake_generate_rejects_structured_output_config(self):
        """§3.4: cited generation is never combined with structured output /

        tool configuration — the very incompatibility the protocol was split
        to prevent.
        """
        fake = FakeAdapter(generate_results=[_answer()] * 3)
        for forbidden in ({"tools": []}, {"tool_choice": {"type": "any"}}, {"output_schema": {}}):
            config = {"model": "claude-haiku-4-5", **forbidden}
            with pytest.raises(ProviderContractError, match=next(iter(forbidden))):
                fake.generate(
                    messages=GENERATE_PAYLOAD["messages"],
                    documents=GENERATE_PAYLOAD["documents"],
                    config=config,
                )

    def test_fake_structured_rejects_citations_config(self):
        """§4.3: structured-output calls never enable citations."""
        fake = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        with pytest.raises(ProviderContractError, match="citations"):
            fake.structured(
                messages=STRUCTURED_PAYLOAD["messages"],
                schema=STRUCTURED_PAYLOAD["schema"],
                config={"model": "claude-haiku-4-5", "citations": {"enabled": True}},
            )

    def test_fake_adapter_rejects_mistyped_programmed_response(self):
        """generate must return AnswerWithCitations; structured a mapping —

        a mis-programmed fake fails at call time, not when the downstream
        code trips over the wrong type (or worse, doesn't).
        """
        fake = FakeAdapter(generate_results=[{"text": "a bare dict"}])
        with pytest.raises(ProviderContractError, match="AnswerWithCitations"):
            fake.generate(**GENERATE_PAYLOAD)

        fake = FakeAdapter(structured_results=[_answer()])
        with pytest.raises(ProviderContractError, match="structured"):
            fake.structured(**STRUCTURED_PAYLOAD)

    def test_structured_seam_carries_system_in_dedicated_top_level_field(self, tmp_path):
        """Finding #91: the seam's canonical request shape for the system prompt.

        `structured` takes an optional top-level `system` string that maps
        1:1 onto the Anthropic Messages API's top-level `system` parameter.
        It is recorded in the payload when given, and OMITTED when None so
        pre-#91 recorded request hashes (fixtures recorded without a system
        field) stay valid. The recorder/replayer round-trips it like any
        other payload key.
        """
        fake = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        fake.structured(system="You are the query-processing stage.", **STRUCTURED_PAYLOAD)
        (call,) = fake.calls
        assert call.payload["system"] == "You are the query-processing stage."

        bare = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        bare.structured(**STRUCTURED_PAYLOAD)
        assert "system" not in bare.calls[0].payload, (
            "system=None must be omitted from the payload, or every pre-#91 "
            "recorded request hash is silently invalidated"
        )

        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(system="You are the query-processing stage.", **STRUCTURED_PAYLOAD)
        replayed = ReplayAdapter(tmp_path).structured(
            system="You are the query-processing stage.", **STRUCTURED_PAYLOAD
        )
        assert replayed == {"scope": "in_scope"}

    def test_structured_seam_returns_usage_like_generate(self):
        """Finding #92: `structured` has a usage channel, like `generate`.

        The seam returns a StructuredResult — a Mapping over the parsed
        structured output (so every existing consumer keeps indexing it like
        the dict it replaces) that also carries `.usage`, mirroring
        AnswerWithCitations.usage (finding #64). Tests may program plain
        mappings for convenience; the fake wraps them with usage None, so
        consumers can always rely on `.usage` existing.
        """
        usage = {"input_tokens": 430, "output_tokens": 58}
        fake = FakeAdapter(
            structured_results=[StructuredResult(value={"scope": "in_scope"}, usage=usage)]
        )
        result = fake.structured(**STRUCTURED_PAYLOAD)
        assert result == {"scope": "in_scope"}, "StructuredResult must equal its mapping value"
        assert result["scope"] == "in_scope"
        assert result.usage == usage

        plain = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        wrapped = plain.structured(**STRUCTURED_PAYLOAD)
        assert isinstance(wrapped, StructuredResult)
        assert wrapped.usage is None

    def test_structured_usage_round_trips_through_record_and_replay(self, tmp_path):
        """Recorded structured usage survives the record/replay round trip.

        Like generate's usage (finding #64): #21/#22 cost accounting must
        read token usage from replayed structured responses, and old
        fixtures without the field keep replaying with usage None.
        """
        usage = {"input_tokens": 430, "output_tokens": 58}
        transport = FakeAdapter(
            structured_results=[StructuredResult(value={"scope": "in_scope"}, usage=usage)]
        )
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**STRUCTURED_PAYLOAD)

        replayed = ReplayAdapter(tmp_path).structured(**STRUCTURED_PAYLOAD)
        assert replayed == {"scope": "in_scope"}
        assert replayed.usage == usage

        # A pre-#92 fixture (bare "dict" response, no usage) still replays.
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        _write_replay_fixture(
            legacy_dir,
            "structured",
            STRUCTURED_PAYLOAD,
            {"type": "dict", "value": {"scope": "in_scope"}},
        )
        legacy = ReplayAdapter(legacy_dir).structured(**STRUCTURED_PAYLOAD)
        assert legacy == {"scope": "in_scope"}
        assert legacy.usage is None

    def test_seam_rejects_system_role_messages(self):
        """Finding #91: `messages` NEVER carries a role: "system" entry.

        The live Messages API rejects role "system" inside `messages` on
        claude-haiku-4-5; the system prompt's only sanctioned channel is the
        dedicated top-level field. Enforced at the seam so a green fake-backed
        suite can never hide a request that would 400 live (finding #62
        principle), on structured and generate alike.
        """
        system_first = [{"role": "system", "content": "instructions"}] + list(
            STRUCTURED_PAYLOAD["messages"]
        )
        fake = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        with pytest.raises(ProviderContractError, match="system"):
            fake.structured(
                messages=system_first,
                schema=STRUCTURED_PAYLOAD["schema"],
                config=STRUCTURED_PAYLOAD["config"],
            )

        gen = FakeAdapter(generate_results=[_answer()])
        with pytest.raises(ProviderContractError, match="system"):
            gen.generate(
                messages=[{"role": "system", "content": "instructions"}]
                + list(GENERATE_PAYLOAD["messages"]),
                documents=GENERATE_PAYLOAD["documents"],
                config=GENERATE_PAYLOAD["config"],
            )

    def test_replay_adapter_applies_same_request_validation(self, tmp_path):
        """ReplayAdapter shares the seam validator: an invalid request raises

        ProviderContractError (naming the constraint), never the misleading
        'no recorded fixture' error.
        """
        replay = ReplayAdapter(tmp_path)
        nine_docs = [_cited_doc(f"synthetic passage {i}") for i in range(9)]
        with pytest.raises(ProviderContractError, match="8"):
            replay.generate(
                messages=GENERATE_PAYLOAD["messages"],
                documents=nine_docs,
                config=GENERATE_PAYLOAD["config"],
            )

    def test_replay_adapter_rejects_mistyped_recorded_response(self, tmp_path):
        """A `dict` fixture replayed through generate is a mistyped recording

        (or a misfiled fixture) and must raise, not leak a dict out of a
        method annotated -> AnswerWithCitations.
        """
        _write_replay_fixture(
            tmp_path,
            "generate",
            GENERATE_PAYLOAD,
            {"type": "dict", "value": {"scope": "in_scope"}},
        )
        with pytest.raises(ProviderContractError, match="AnswerWithCitations"):
            ReplayAdapter(tmp_path).generate(**GENERATE_PAYLOAD)

    def test_recording_adapter_applies_same_request_validation(self, tmp_path):
        """RecordingAdapter validates before delegating: an invalid request

        never reaches a (paid, live) transport and never writes a fixture.
        """
        transport = FakeAdapter(generate_results=[_answer()])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        with pytest.raises(ProviderContractError):
            recorder.generate(
                messages=GENERATE_PAYLOAD["messages"],
                documents=GENERATE_PAYLOAD["documents"],
                config={"model": "claude-haiku-4-5", "tools": []},
            )
        assert transport.calls == [], "invalid request must not reach the transport"
        assert list(tmp_path.glob("*.json")) == []


def _write_replay_fixture(
    fixtures_dir: Path,
    method: str,
    payload: dict,
    response: dict,
    format_version: int | None = REPLAY_FIXTURE_FORMAT_VERSION,
) -> Path:
    """Write a replay fixture in the documented on-disk format.

    Written by hand (not through the recorder) so these tests pin the format
    itself: JSON file named `<canonical request hash>.json` containing the
    versioned envelope, the method, the scrubbed request, and a typed
    response. `format_version=None` omits the field (for tests of the
    version guard).
    """
    meta: dict = {
        "marker": "SYNTHETIC FIXTURE — authored for this project's tests",
        "scrubbed": True,
    }
    if format_version is not None:
        meta["format_version"] = format_version
    fixture_path = fixtures_dir / f"{canonical_request_hash(method, payload)}.json"
    fixture_path.write_text(
        json.dumps(
            {
                "_meta": meta,
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

    def test_canonical_hash_rejects_non_string_dict_keys(self):
        """Review finding #68: json.dumps silently coerces non-str keys to

        strings, so {1: x} and {"1": x} would share one fixture — a silent
        collision that replays the wrong response. Refusal makes collisions
        impossible, matching the repo's fail-loudly style. (Mixed-type keys,
        which json.dumps turns into a TypeError from the key sort, get the
        same dedicated error.)
        """
        for bad in (
            {"schema": {1: "x"}},
            {"schema": {True: "x"}},
            {"schema": {None: "x"}},
            {"schema": {1: "x", "a": "y"}},
        ):
            with pytest.raises(CanonicalisationError, match="key"):
                canonical_request_hash("structured", bad)

    def test_canonical_hash_treats_equal_numbers_equally(self):
        """Decision (finding #68): int-valued floats are normalised to int,

        because JSON (and the API) treat 1 and 1.0 as the same number — an
        innocuous int→float refactor at a call site must not invalidate
        every recording with a message blaming the prompt. Genuinely
        fractional floats still hash distinctly.
        """
        a = {"config": {"temperature": 1}}
        b = {"config": {"temperature": 1.0}}
        assert canonical_request_hash("structured", a) == canonical_request_hash("structured", b)
        c = {"config": {"temperature": 0.7}}
        assert canonical_request_hash("structured", a) != canonical_request_hash("structured", c)

    def test_canonical_hash_accepts_abstract_mappings_and_tuples(self):
        """The protocol signatures promise Mapping/Sequence inputs; a

        MappingProxyType or tuple payload must hash identically to its
        dict/list equivalent instead of raising TypeError.
        """
        from types import MappingProxyType

        as_dict = {"messages": [{"role": "user", "content": "q"}], "config": {"model": "m"}}
        as_abstract = MappingProxyType(
            {
                "messages": (MappingProxyType({"role": "user", "content": "q"}),),
                "config": MappingProxyType({"model": "m"}),
            }
        )
        assert canonical_request_hash("structured", as_dict) == canonical_request_hash(
            "structured", as_abstract
        )

    def test_canonical_hash_rejects_nan_and_infinity(self):
        """NaN/Infinity are not JSON and no real API request can carry them;

        hashing them 'successfully' would mint an identity for an impossible
        request. Reject loudly instead.
        """
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(CanonicalisationError, match="finite"):
                canonical_request_hash("structured", {"config": {"temperature": bad_value}})


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


class TestReplayFixtureFormat:
    """Review finding #64: the on-disk format must carry a version (so it can

    evolve without heuristics), transport usage metadata (the §9 cost model
    and #21/#22 need it), and a raw-response type (so the future
    AnthropicAdapter's parsing of citation deltas / streamed events can be
    replay-pinned per IMPLEMENTATION §4.2, instead of every consumer issue
    improvising its own transport fixtures).
    """

    def test_replay_fixture_format_carries_version(self, tmp_path, replay_fixtures_dir):
        """Every recorder-written and committed fixture declares

        _meta.format_version; replay rejects unknown or missing versions
        loudly instead of guessing.
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**STRUCTURED_PAYLOAD)
        [fixture_path] = tmp_path.glob("*.json")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture["_meta"]["format_version"] == REPLAY_FIXTURE_FORMAT_VERSION == 1

        for path in sorted(replay_fixtures_dir.glob("*.json")):
            committed = json.loads(path.read_text(encoding="utf-8"))
            assert committed["_meta"]["format_version"] == REPLAY_FIXTURE_FORMAT_VERSION, (
                f"{path.name}: committed fixture lacks the current format_version"
            )

    def test_replay_rejects_unknown_or_missing_format_version(self, tmp_path):
        unknown_dir = tmp_path / "unknown"
        unknown_dir.mkdir()
        _write_replay_fixture(
            unknown_dir,
            "structured",
            STRUCTURED_PAYLOAD,
            {"type": "dict", "value": {"scope": "in_scope"}},
            format_version=99,
        )
        with pytest.raises(ReplayFormatError, match="99"):
            ReplayAdapter(unknown_dir).structured(**STRUCTURED_PAYLOAD)

        missing_dir = tmp_path / "missing"
        missing_dir.mkdir()
        _write_replay_fixture(
            missing_dir,
            "structured",
            STRUCTURED_PAYLOAD,
            {"type": "dict", "value": {"scope": "in_scope"}},
            format_version=None,
        )
        with pytest.raises(ReplayFormatError, match="format_version"):
            ReplayAdapter(missing_dir).structured(**STRUCTURED_PAYLOAD)

    def test_answer_with_citations_round_trips_usage_metadata(self):
        """Cache-usage metadata (input/output/cache tokens) must survive the

        record/replay round trip — the §9 cost model and #21/#22 cost
        accounting read it from replayed responses.
        """
        usage = {
            "input_tokens": 1200,
            "output_tokens": 180,
            "cache_creation_input_tokens": 900,
            "cache_read_input_tokens": 300,
        }
        answer = AnswerWithCitations(
            text="It warmed by 1.9 C.",
            citations=(Citation(cited_text="Warming of 1.9 C is very likely.", document_index=0),),
            usage=usage,
        )
        round_tripped = deserialize_response(serialize_response(answer))
        assert round_tripped == answer
        assert round_tripped.usage == usage

        # Absent usage stays absent (old fixtures without the field parse).
        bare = deserialize_response(serialize_response(AnswerWithCitations(text="t")))
        assert bare.usage is None

    def test_serialize_response_round_trips_raw_provider_response(self):
        """The 'raw' response type carries an unparsed provider payload plus

        an optional streaming event sequence, so the transport-level parsing
        inside the future AnthropicAdapter (#13) can be regression-pinned
        through this same fixture machinery.
        """
        raw = RawProviderResponse(
            payload={
                "id": "msg_synthetic",
                "content": [{"type": "text", "text": "It warmed.", "citations": []}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            events=(
                {"type": "message_start"},
                {"type": "content_block_delta", "delta": {"type": "citations_delta"}},
                {"type": "message_stop"},
            ),
        )
        serialized = serialize_response(raw)
        assert serialized["type"] == "raw"
        round_tripped = deserialize_response(serialized)
        assert round_tripped == raw

    def test_replay_methods_reject_raw_fixtures(self, tmp_path):
        """Raw fixtures feed the transport-level replayer built with #13;

        the typed ProviderAdapter methods must refuse them rather than leak
        an unparsed payload out of the seam.
        """
        _write_replay_fixture(
            tmp_path,
            "generate",
            GENERATE_PAYLOAD,
            {"type": "raw", "payload": {"id": "msg_synthetic"}, "events": []},
        )
        with pytest.raises(ProviderContractError):
            ReplayAdapter(tmp_path).generate(**GENERATE_PAYLOAD)


# Built at runtime so no secret-shaped literal is ever committed (gitleaks
# scans the tree; a real-looking token in test source would be a false
# positive at best and a bad example at worst).
FAKE_SECRET = "sk-" + "ant-" + "api03-" + "SYNTHETIC-TEST-VALUE-NOT-A-REAL-KEY"

RECORDING_ENV = {"CLIMATE_CHAT_RECORD": "1", "ANTHROPIC_API_KEY": FAKE_SECRET}

PAYLOAD_WITH_CREDENTIALS = {
    "messages": [{"role": "user", "content": "classify this"}],
    "schema": {"type": "object"},
    "config": {
        "model": "claude-haiku-4-5",
        "api_key": FAKE_SECRET,
        "headers": {
            "x-api-key": FAKE_SECRET,
            "authorization": "Bearer " + FAKE_SECRET,
            "anthropic-version": "2023-06-01",
        },
    },
}


class TestRecordingAdapter:
    def test_recorder_refuses_without_env_flag(self, tmp_path, fake_adapter):
        """TDD plan item 5: no accidental live calls from the test suite.

        Recording requires BOTH the explicit CLIMATE_CHAT_RECORD=1 env flag
        and a live ANTHROPIC_API_KEY; anything less refuses at construction,
        before any call could be delegated to a live transport.
        """
        with pytest.raises(RecordingDisabledError, match="CLIMATE_CHAT_RECORD"):
            RecordingAdapter(fake_adapter, tmp_path, env={})

        with pytest.raises(RecordingDisabledError, match="CLIMATE_CHAT_RECORD"):
            RecordingAdapter(
                fake_adapter, tmp_path, env={"CLIMATE_CHAT_RECORD": "0", "ANTHROPIC_API_KEY": "x"}
            )

        with pytest.raises(RecordingDisabledError, match="ANTHROPIC_API_KEY"):
            RecordingAdapter(fake_adapter, tmp_path, env={"CLIMATE_CHAT_RECORD": "1"})

        with pytest.raises(RecordingDisabledError, match="ANTHROPIC_API_KEY"):
            RecordingAdapter(
                fake_adapter, tmp_path, env={"CLIMATE_CHAT_RECORD": "1", "ANTHROPIC_API_KEY": ""}
            )

    def test_recorder_round_trips_through_replay(self, tmp_path):
        """With the flag and a key, the recorder delegates to its transport,

        writes a fixture keyed by the canonical request hash, and the
        ReplayAdapter replays it byte-for-byte deterministically. The
        transport here is a FakeAdapter: the recorder is buildable and
        testable with no real key on the machine (live recording is a later,
        env-gated act through the same seam).
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)

        result = recorder.structured(**STRUCTURED_PAYLOAD)

        assert result == {"scope": "in_scope"}
        assert [c.method for c in transport.calls] == ["structured"]
        expected = tmp_path / f"{canonical_request_hash('structured', STRUCTURED_PAYLOAD)}.json"
        assert expected.is_file()
        assert ReplayAdapter(tmp_path).structured(**STRUCTURED_PAYLOAD) == {"scope": "in_scope"}

    def test_recorded_fixtures_scrubbed(self, tmp_path, replay_fixtures_dir):
        """TDD plan item 6: no API keys or auth headers in any fixture JSON.

        Two halves: (a) the recorder strips credential-bearing keys before
        writing, verified end-to-end on a payload carrying an api key and
        auth headers; (b) every fixture actually committed under
        tests/fixtures/replay/ is scrubbed and secret-free.
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**PAYLOAD_WITH_CREDENTIALS)

        [fixture_path] = tmp_path.glob("*.json")
        text = fixture_path.read_text(encoding="utf-8")
        assert FAKE_SECRET not in text
        assert "sk-ant-" not in text
        fixture = json.loads(text)
        stored_config = fixture["request"]["config"]
        assert "api_key" not in stored_config
        assert "headers" not in stored_config
        # Non-credential request content is preserved.
        assert stored_config["model"] == "claude-haiku-4-5"
        assert fixture["_meta"]["scrubbed"] is True

        # (b) The committed corpus of replay fixtures.
        committed = sorted(replay_fixtures_dir.glob("*.json"))
        assert committed, "no committed replay fixtures found under tests/fixtures/replay/"
        for path in committed:
            committed_text = path.read_text(encoding="utf-8")
            assert "sk-ant-" not in committed_text, f"{path.name}: api key material"
            assert "authorization" not in committed_text.lower(), f"{path.name}: auth header"
            assert "x-api-key" not in committed_text.lower(), f"{path.name}: api key header"
            committed_fixture = json.loads(committed_text)
            assert committed_fixture["_meta"]["scrubbed"] is True, f"{path.name}: not scrubbed"
            # Finding #63: every committed fixture's filename must equal the
            # canonical hash of its stored request — verifiable from the
            # committed content alone, and pinned against hand-editing drift.
            expected_stem = canonical_request_hash(
                committed_fixture["method"], committed_fixture["request"]
            )
            assert path.stem == expected_stem, (
                f"{path.name}: filename does not match the canonical hash of its "
                f"stored request ({expected_stem}) - tampered, misfiled, or recorded "
                "with an unscrubbed hash"
            )

    def test_recorded_fixture_filename_matches_hash_of_stored_request(self, tmp_path):
        """Review finding #63: the fixture must be keyed by the hash of the

        SCRUBBED payload — the payload it actually stores — so the filename
        is verifiable from the committed content alone. Hashing the raw
        credentialed payload leaves the fixture unverifiable (and derives
        the filename from secret-bearing bytes).
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**PAYLOAD_WITH_CREDENTIALS)

        [fixture_path] = tmp_path.glob("*.json")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture_path.stem == canonical_request_hash("structured", fixture["request"])

    def test_replay_finds_recording_made_with_credentialed_payload(self, tmp_path):
        """Review finding #63: exactly when the scrubber does its job, the

        fixture must stay replayable by keyless CI. The scrubbed equivalent
        payload (what CI sends, holding no header/key material) and the
        credentialed payload (what the recording machine sent) must resolve
        to the same recording — scrub-before-hash makes them one key.
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**PAYLOAD_WITH_CREDENTIALS)
        replay = ReplayAdapter(tmp_path)

        scrubbed_payload = scrub_payload(PAYLOAD_WITH_CREDENTIALS)
        assert "api_key" not in scrubbed_payload["config"], "precondition: scrubber fired"
        assert replay.structured(**scrubbed_payload) == {"scope": "in_scope"}
        assert replay.structured(**PAYLOAD_WITH_CREDENTIALS) == {"scope": "in_scope"}

    def test_recorder_fails_closed_when_secret_survives_in_content(self, tmp_path):
        """A secret pattern inside request *content* (not a credential field)

        means something upstream leaked a key into a prompt. The recorder
        must refuse to write rather than silently redact-and-commit.
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        leaky_payload = {
            "messages": [{"role": "user", "content": f"my key is {FAKE_SECRET}"}],
            "schema": {"type": "object"},
            "config": {"model": "claude-haiku-4-5"},
        }

        with pytest.raises(SecretLeakError):
            recorder.structured(**leaky_payload)
        assert list(tmp_path.glob("*.json")) == [], "fixture must not be written on a leak"

    def test_recorder_scrubs_common_credential_keys_and_shapes(self, tmp_path):
        """Review finding #65: the recorder wraps "any transport", so the

        scrub must cover the credential keys that realistically ride in
        provider/proxy configs (apikey, token, access_token, cookie...) and
        the fail-closed scan must know non-Anthropic secret shapes (AWS
        AKIA..., HuggingFace hf_..., GitHub ghp_...) — not only sk-ant- and
        bearer. All values are synthetic, built by concatenation so no
        secret-shaped literal sits in the source tree.
        """
        aws_style = "AKIA" + "IOSFODNN7EXAMPLE"
        hf_style = "hf_" + "SYNTHETICxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        gh_style = "ghp_" + "SYNTHETICxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(
            messages=[{"role": "user", "content": "classify this"}],
            schema={"type": "object"},
            config={
                "model": "claude-haiku-4-5",
                "max_tokens": 256,
                "apikey": aws_style,
                "token": hf_style,
                "access_token": gh_style,
                "cookie": "session=" + "SYNTHETICCOOKIEVALUE",
                "client_secret": "SYNTHETIC" + "CLIENTSECRET",
            },
        )

        [fixture_path] = tmp_path.glob("*.json")
        text = fixture_path.read_text(encoding="utf-8")
        for secret in (aws_style, hf_style, gh_style, "SYNTHETICCOOKIEVALUE", "CLIENTSECRET"):
            assert secret not in text, f"credential value {secret[:8]}... written to fixture"
        stored_config = json.loads(text)["request"]["config"]
        # Non-credential request content is preserved untouched.
        assert stored_config["model"] == "claude-haiku-4-5"
        assert stored_config["max_tokens"] == 256

    def test_recorder_fails_closed_on_non_anthropic_secret_shapes_in_content(self, tmp_path):
        """Finding #65 keeps the fail-closed rule: a secret *shape* surviving

        into content (where key-based scrubbing cannot reach) aborts the
        write for AWS/HF/GitHub-shaped material exactly as for sk-ant-.
        """
        for leaked in (
            "AKIA" + "IOSFODNN7EXAMPLE",
            "hf_" + "SYNTHETICxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "ghp_" + "SYNTHETICxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        ):
            transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
            recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
            with pytest.raises(SecretLeakError):
                recorder.structured(
                    messages=[{"role": "user", "content": f"my credential is {leaked}"}],
                    schema={"type": "object"},
                    config={"model": "claude-haiku-4-5"},
                )
            assert list(tmp_path.glob("*.json")) == [], "fixture must not be written on a leak"

    def test_secret_leak_error_never_contains_secret_material(self, tmp_path):
        """Review finding #66: the exception raised on a leak lands in

        terminal scrollback and CI logs for live recording jobs — it must
        never echo any fragment of the matched secret (the old message
        embedded up to 5 characters of a bearer token). Trigger every
        pattern with content carrying a distinctive synthetic token and
        assert no part of it reaches the message.
        """
        token = "SYNTH" + "TOKENVALUE12345"  # 20 alnum chars, built by concatenation
        leaky_contents = [
            "header was Bearer " + token,
            "x-api-key: " + token,
            "authorization header " + token,
            "sk-ant-" + token,
            "sk-" + token,
            "hf_" + token,
            "ghp_" + token,
            "AKIA" + "0123456789ABCDEF" + " " + token,
        ]
        for content in leaky_contents:
            transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
            recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
            with pytest.raises(SecretLeakError) as excinfo:
                recorder.structured(
                    messages=[{"role": "user", "content": content}],
                    schema={"type": "object"},
                    config={"model": "claude-haiku-4-5"},
                )
            message = str(excinfo.value)
            # No fragment of the secret, however short: the old message's
            # 12-char echo leaked "Bearer SYNTH" / "hf_SYNTHTOKE" etc.
            for fragment in (token, "TOKENVALUE", "SYNTH", "01234"):
                assert fragment not in message, (
                    f"secret fragment {fragment!r} echoed for content {content[:6]!r}..."
                )
            assert "NOT written" in message, "the fail-closed outcome must stay explicit"
            assert list(tmp_path.glob("*.json")) == []

    def test_scrub_payload_key_list_covers_common_credential_names(self):
        """Table-driven (finding #65): every realistic credential-bearing key

        is dropped, in any casing/separator style; ordinary request keys —
        including 'max_tokens', whose normalised form contains 'token' as a
        substring — are kept.
        """
        scrubbed_keys = [
            "api_key",
            "apikey",
            "x-api-key",
            "authorization",
            "proxy_authorization",
            "auth_token",
            "bearer_token",
            "token",
            "access_token",
            "refresh_token",
            "proxy_token",
            "id_token",
            "secret",
            "client_secret",
            "aws_secret_access_key",
            "session_key",
            "password",
            "cookie",
            "set-cookie",
            "headers",
            "extra_headers",
            "credentials",
            "Authorization",
            "ACCESS_TOKEN",
        ]
        for key in scrubbed_keys:
            scrubbed = scrub_payload({"config": {key: "synthetic-credential", "model": "m"}})
            assert scrubbed == {"config": {"model": "m"}}, f"key {key!r} was not scrubbed"

        kept_keys = ["model", "max_tokens", "max_tokens_to_sample", "temperature", "top_k"]
        for key in kept_keys:
            scrubbed = scrub_payload({"config": {key: 7}})
            assert scrubbed == {"config": {key: 7}}, f"non-credential key {key!r} was scrubbed"

    def test_committed_replay_fixtures_declare_provenance(self, replay_fixtures_dir):
        """Review finding #67: recorded fixtures embed full request/response

        content, so tests/fixtures/replay/ needs the same only-synthetic
        guard as corpus/ and charts/ (DESIGN 2.1) — here as _meta fields,
        since JSON has no first-line marker. Every committed fixture must
        carry a non-empty provenance and a filled content_signoff
        ({who, date, note}, mirroring the manifest's human_signoff)
        attesting the embedded content is synthetic or licensed-open. The
        recorder deliberately writes content_signoff: null, so committing a
        fresh recording fails this test until a human fills in the
        sign-off — commit-time provenance is an explicit human act.
        """
        committed = sorted(replay_fixtures_dir.glob("*.json"))
        assert committed, "no committed replay fixtures found under tests/fixtures/replay/"
        for path in committed:
            meta = json.loads(path.read_text(encoding="utf-8"))["_meta"]
            assert meta.get("provenance"), f"{path.name}: missing/empty _meta.provenance"
            signoff = meta.get("content_signoff")
            assert signoff, (
                f"{path.name}: missing/empty _meta.content_signoff - a human must attest "
                "the embedded request/response content is synthetic or licensed-open "
                "(DESIGN 2.1) before a recording is committed"
            )
            for field in ("who", "date", "note"):
                assert signoff.get(field), f"{path.name}: content_signoff.{field} empty"

    def test_recorder_output_matches_committed_fixture_format(self, tmp_path, replay_fixtures_dir):
        """Review finding #67: the committed exemplar claimed to be recorder

        output while carrying _meta keys the recorder never writes. The
        recorder's _meta key-set must equal every committed fixture's
        _meta key-set, killing exemplar/recorder drift in either direction.
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**STRUCTURED_PAYLOAD)
        [recorded_path] = tmp_path.glob("*.json")
        recorded_meta_keys = set(json.loads(recorded_path.read_text(encoding="utf-8"))["_meta"])

        for path in sorted(replay_fixtures_dir.glob("*.json")):
            committed_meta_keys = set(json.loads(path.read_text(encoding="utf-8"))["_meta"])
            assert committed_meta_keys == recorded_meta_keys, (
                f"{path.name}: committed _meta keys {sorted(committed_meta_keys)} != "
                f"recorder-written _meta keys {sorted(recorded_meta_keys)} - the fixture "
                "is not honest recorder output (or the recorder format drifted)"
            )

    def test_recorder_writes_null_content_signoff(self, tmp_path):
        """The recorder cannot sign off content itself: it writes

        content_signoff: null so a freshly recorded fixture fails the
        committed-provenance guard until a human fills it in (finding #67).
        """
        transport = FakeAdapter(structured_results=[{"scope": "in_scope"}])
        recorder = RecordingAdapter(transport, tmp_path, env=RECORDING_ENV)
        recorder.structured(**STRUCTURED_PAYLOAD)
        [recorded_path] = tmp_path.glob("*.json")
        meta = json.loads(recorded_path.read_text(encoding="utf-8"))["_meta"]
        assert "content_signoff" in meta
        assert meta["content_signoff"] is None

    def test_scrub_payload_removes_credential_keys_recursively(self):
        """scrub_payload strips credential-bearing keys at any nesting depth."""
        scrubbed = scrub_payload(
            {
                "config": {"nested": {"Authorization": "Bearer " + FAKE_SECRET}},
                "headers": {"x-api-key": FAKE_SECRET},
                "messages": [{"role": "user", "content": "kept"}],
            }
        )
        assert scrubbed == {
            "config": {"nested": {}},
            "messages": [{"role": "user", "content": "kept"}],
        }
