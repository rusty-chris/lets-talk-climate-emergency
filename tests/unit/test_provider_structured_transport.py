"""AnthropicAdapter structured-transport twins (review finding #206).

Unit tier, keyless, zero network: the structured transport landed in
issue #13 pinned only by the skip-marked live round-trip — request
mapping (``build_anthropic_structured_request``), the JSON/usage fold
(``_structured_result_from_message`` via ``_structured_call``) and the
#184 ``ProviderTransportError`` mapping had no FakeAdapter/fake-SDK-
client twin at all, unlike the generate/streaming sibling
(tests/unit/test_generation_streaming.py). These tests mirror that
suite's fake-SDK-client structure through the existing
``AnthropicAdapter(client=...)`` injection seam.

These pin EXISTING landed behaviour (the #206 exemption from the red
rule): they are regression pins, expected green as written — the red
commit body records which, per the finding. The keyless-before-network
pin already exists
(test_generation_request_builders.py::
test_anthropic_adapter_structured_transport_keyless_raises_before_network)
and is not duplicated here.
"""

from __future__ import annotations

import copy
import json

import pytest

from rag.provider import (
    AnthropicAdapter,
    ProviderContractError,
    ProviderTransportError,
    StructuredResult,
    build_anthropic_structured_request,
)

#: A canonical seam `structured` payload (the shape the citation
#: validator's builder emits: verdict schema, judge instructions on the
#: top-level system channel, model + max_tokens config only).
STRUCTURED_PAYLOAD = {
    "messages": [{"role": "user", "content": "Judge the synthetic claim/source pairs."}],
    "schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_index": {"type": "integer"},
                        "supported": {"type": "boolean"},
                    },
                    "required": ["pair_index", "supported"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
    "config": {"model": "claude-haiku-4-5", "max_tokens": 512},
    "system": "You are a careful entailment judge for a synthetic test.",
}

VERDICTS_JSON = {"verdicts": [{"pair_index": 0, "supported": True}]}


def _payload(**overrides):
    payload = copy.deepcopy(STRUCTURED_PAYLOAD)
    payload.update(overrides)
    return payload


def _payload_without_system():
    payload = copy.deepcopy(STRUCTURED_PAYLOAD)
    del payload["system"]
    return payload


class _FakeSDKMessage:
    """A stub SDK response message: model_dump() like the real SDK type."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return copy.deepcopy(self._payload)


class _FakeMessages:
    def __init__(self, message, log):
        self._message = message
        self._log = log

    def create(self, **kwargs):
        self._log.append(kwargs)
        return self._message


class _FakeSDKClient:
    """The injected non-streaming twin of the streaming suite's fake."""

    def __init__(self, message):
        self.create_calls: list = []
        self.messages = _FakeMessages(message, self.create_calls)


def _message(text_blocks, usage=None):
    content = [{"type": "text", "text": text} for text in text_blocks]
    payload = {"content": content}
    if usage is not None:
        payload["usage"] = usage
    return _FakeSDKMessage(payload)


class TestBuildAnthropicStructuredRequest:
    def test_build_anthropic_structured_request_maps_the_seam_payload(self):
        """The exact kwargs pin: model/max_tokens from config and nothing
        else, the schema on output_config.format json_schema as a copy
        (mutating the request can never corrupt the seam payload),
        messages copied 1:1, top-level system passed through verbatim."""
        payload = _payload()
        api_request = build_anthropic_structured_request(payload)
        assert set(api_request) == {"model", "max_tokens", "output_config", "messages", "system"}
        assert api_request["model"] == payload["config"]["model"]
        assert api_request["max_tokens"] == payload["config"]["max_tokens"]
        assert api_request["output_config"] == {
            "format": {"type": "json_schema", "schema": payload["schema"]}
        }
        assert api_request["output_config"]["format"]["schema"] is not payload["schema"]
        assert api_request["messages"] == payload["messages"]
        assert api_request["messages"][0] is not payload["messages"][0]
        assert api_request["system"] == payload["system"]

    def test_build_anthropic_structured_request_omits_absent_system(self):
        """The #91 hash-stability rule: no system given -> no system key
        emitted (a None system must never invent request bytes)."""
        api_request = build_anthropic_structured_request(_payload_without_system())
        assert "system" not in api_request
        assert set(api_request) == {"model", "max_tokens", "output_config", "messages"}

    def test_build_anthropic_structured_request_never_emits_citations_config(self):
        """The #62 ordering: a corrupt payload carrying citations config
        raises the seam contract error BEFORE any request dict exists —
        structured calls never enable citations (§3.4/IMPLEMENTATION
        §4.3)."""
        corrupt = _payload(config={**STRUCTURED_PAYLOAD["config"], "citations": {"enabled": True}})
        with pytest.raises(ProviderContractError):
            build_anthropic_structured_request(corrupt)

    def test_build_anthropic_structured_request_rejects_system_role_messages(self):
        """Finding #91 at this builder too: role 'system' inside messages
        raises — the dedicated top-level channel is the only one."""
        corrupt = _payload(messages=[{"role": "system", "content": "smuggled instructions"}])
        with pytest.raises(ProviderContractError):
            build_anthropic_structured_request(corrupt)


class TestStructuredCallFold:
    def test_structured_call_folds_json_text_and_integer_usage(self):
        """The transport fold: text blocks concatenate into the JSON
        value, usage keeps INTEGER fields only (bools and strings never
        pollute spend arithmetic — the accumulate_answer convention), and
        the seam returns the contract type. The kwargs on the wire ARE
        the pure builder's output — no drift between the tested builder
        and the live call."""
        usage = {
            "input_tokens": 2500,
            "output_tokens": 300,
            "service_tier": "standard",  # non-int: dropped
            "truncated": True,  # bool: dropped (bool is an int subclass)
        }
        verdicts_text = json.dumps(VERDICTS_JSON)
        split_at = len(verdicts_text) // 2
        client = _FakeSDKClient(
            _message([verdicts_text[:split_at], verdicts_text[split_at:]], usage=usage)
        )
        adapter = AnthropicAdapter(client=client)
        payload = _payload()
        result = adapter.structured(**payload)
        assert isinstance(result, StructuredResult)
        assert result["verdicts"] == VERDICTS_JSON["verdicts"]
        assert result.usage == {"input_tokens": 2500, "output_tokens": 300}
        assert client.create_calls == [build_anthropic_structured_request(payload)]
        assert "stream" not in client.create_calls[0]

    def test_structured_call_without_usage_reports_usage_none(self):
        client = _FakeSDKClient(_message([json.dumps(VERDICTS_JSON)]))
        result = AnthropicAdapter(client=client).structured(**_payload())
        assert result.usage is None
        assert result["verdicts"] == VERDICTS_JSON["verdicts"]

    def test_structured_non_json_response_is_a_provider_contract_error(self):
        """output_config.format json_schema guarantees JSON; a response
        that is not is a transport-contract violation surfaced as the
        seam's typed error — never a bare json.JSONDecodeError."""
        client = _FakeSDKClient(_message(["this is not JSON at all"]))
        with pytest.raises(ProviderContractError):
            AnthropicAdapter(client=client).structured(**_payload())

    def test_structured_non_object_json_is_a_provider_contract_error(self):
        """The other branch: valid JSON that is not an object (an array)
        violates the structured contract identically."""
        client = _FakeSDKClient(_message([json.dumps([1, 2, 3])]))
        with pytest.raises(ProviderContractError):
            AnthropicAdapter(client=client).structured(**_payload())

    def test_structured_call_validates_request_before_transport(self):
        """The #62 ordering through the adapter: a contract-violating
        payload raises with the transport untouched."""

        class _ExplodingClient:
            def __getattr__(self, name):
                raise AssertionError("transport touched on a contract-violating request")

        adapter = AnthropicAdapter(client=_ExplodingClient())
        corrupt_config = {**STRUCTURED_PAYLOAD["config"], "citations": {"enabled": True}}
        with pytest.raises(ProviderContractError):
            adapter.structured(
                messages=STRUCTURED_PAYLOAD["messages"],
                schema=STRUCTURED_PAYLOAD["schema"],
                config=corrupt_config,
            )


class TestStructuredTransportErrorsAreSeamTyped:
    """The #184 mapping on the structured path, mirroring the streaming
    suite's pins: raw anthropic.* exception types never cross the seam;
    connection/timeout and 408/429/5xx are retryable, other 4xx are
    permanent."""

    def _adapter_with_raising_client(self, exc: BaseException) -> AnthropicAdapter:
        class _RaisingMessages:
            def create(self, **kwargs):
                raise exc

        class _RaisingClient:
            messages = _RaisingMessages()

        return AnthropicAdapter(client=_RaisingClient())

    @staticmethod
    def _status_error(status_code: int, error_type: str):
        anthropic = pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        response = httpx.Response(
            status_code,
            request=httpx.Request("POST", "https://api.example.invalid/v1/messages"),
        )
        return anthropic.APIStatusError(
            f"synthetic {error_type}",
            response=response,
            body={"type": "error", "error": {"type": error_type}},
        )

    def test_connection_error_crosses_the_seam_retryable(self):
        anthropic = pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        sdk_error = anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.example.invalid/v1/messages")
        )
        adapter = self._adapter_with_raising_client(sdk_error)
        with pytest.raises(ProviderTransportError) as excinfo:
            adapter.structured(**_payload())
        assert not isinstance(excinfo.value, anthropic.APIError)
        assert excinfo.value.retryable is True
        assert excinfo.value.status_code is None

    @pytest.mark.parametrize(
        ("status_code", "error_type", "retryable"),
        [
            pytest.param(429, "rate_limit_error", True, id="429-retryable"),
            pytest.param(500, "api_error", True, id="500-retryable"),
            pytest.param(529, "overloaded_error", True, id="529-retryable"),
            pytest.param(400, "invalid_request_error", False, id="400-permanent"),
        ],
    )
    def test_status_errors_cross_the_seam_with_pinned_retryability(
        self, status_code, error_type, retryable
    ):
        """4xx contract errors (a rejected schema above all — the #203
        failure class) must surface as PERMANENT: retrying them is pure
        spend. Transient statuses stay retryable."""
        anthropic = pytest.importorskip("anthropic")
        adapter = self._adapter_with_raising_client(self._status_error(status_code, error_type))
        with pytest.raises(ProviderTransportError) as excinfo:
            adapter.structured(**_payload())
        assert not isinstance(excinfo.value, anthropic.APIError)
        assert excinfo.value.status_code == status_code
        assert excinfo.value.error_type == error_type
        assert excinfo.value.retryable is retryable
