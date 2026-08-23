"""The batched entailment request + verdict parsing (issue #13) — RED.

Unit tier, pure: ``build_validation_request`` is a pure builder over the
pairs; ``parse_validation_output`` is pure validation of the structured
output. The §3.4 contract is enforced here at the builder before any
network is possible (IMPLEMENTATION.md §4.3), with the shared seam
validator (``rag.provider.validate_request``) as the backstop.
"""

from __future__ import annotations

import json

import pytest

from rag.citation_validator import (
    VALIDATOR_MODEL_DEFAULT,
    EntailmentPair,
    MalformedValidationOutputError,
    ValidatorConfig,
    build_validation_request,
    parse_validation_output,
)
from rag.provider import canonical_request_hash, validate_request


def make_pairs(count: int) -> list[EntailmentPair]:
    """`count` synthetic pairs over distinct sentences and blocks."""
    return [
        EntailmentPair(
            pair_index=i,
            sentence_index=i + 1,  # sentence 0 is furniture in the fixtures
            sentence_text=f"Synthetic factual sentence {i} about invented basin warming.",
            document_index=i,
            block_text=f"Synthetic passage {i}: the invented basin very likely warmed.",
        )
        for i in range(count)
    ]


class TestBuildValidationRequest:
    def test_single_request_carries_all_pairs(self):
        """ONE batched request carries EVERY sentence<->block pair: all
        sentence texts and all block texts ride the request messages —
        the model sees the whole batch at once (§3.3: a single batched
        call, never per-sentence calls)."""
        pairs = make_pairs(5)
        request = build_validation_request(pairs, config=ValidatorConfig())
        message_text = json.dumps(request["messages"], ensure_ascii=False)
        for pair in pairs:
            assert pair.sentence_text in message_text
            assert pair.block_text in message_text

    def test_request_config_is_model_and_max_tokens_only(self):
        """Model id is config (Haiku default); nothing else leaks into
        the seam config — no citations, no tool keys, nothing to 400
        the live structured call."""
        default = build_validation_request(make_pairs(2), config=ValidatorConfig())
        assert default["config"] == {
            "model": VALIDATOR_MODEL_DEFAULT,
            "max_tokens": ValidatorConfig().max_tokens,
        }
        assert VALIDATOR_MODEL_DEFAULT == "claude-haiku-4-5"

        override = ValidatorConfig(model="claude-haiku-9-9", max_tokens=256)
        overridden = build_validation_request(make_pairs(2), config=override)
        assert overridden["config"] == {"model": "claude-haiku-9-9", "max_tokens": 256}

    def test_request_satisfies_the_seam_contract(self):
        """The built payload passes the shared §3.4 seam validator: no
        citations configuration, no role:'system' message (the
        instructions ride the dedicated top-level system channel,
        finding #91)."""
        request = build_validation_request(make_pairs(3), config=ValidatorConfig())
        assert set(request) == {"messages", "system", "schema", "config"}
        assert "citations" not in request["config"]
        assert request["system"], "instructions must ride the top-level system channel"
        for message in request["messages"]:
            assert message["role"] != "system"
        # Backstop: the shared seam validator accepts it as-is.
        validate_request("structured", request)

    def test_schema_demands_one_verdict_per_pair(self):
        """The output schema constrains the decoder to exactly one
        ``{pair_index, supported}`` verdict per pair — supported is a
        BOOLEAN, and the verdicts array is required and exact-length
        (a schema is steering; the parser re-enforces on the way back)."""
        pairs = make_pairs(4)
        schema = build_validation_request(pairs, config=ValidatorConfig())["schema"]
        assert schema["type"] == "object"
        assert "verdicts" in schema["required"]
        verdicts_schema = schema["properties"]["verdicts"]
        assert verdicts_schema["type"] == "array"
        assert verdicts_schema["minItems"] == len(pairs)
        assert verdicts_schema["maxItems"] == len(pairs)
        item_schema = verdicts_schema["items"]
        assert set(item_schema["required"]) >= {"pair_index", "supported"}
        assert item_schema["properties"]["supported"]["type"] == "boolean"
        assert item_schema["properties"]["pair_index"]["type"] == "integer"

    def test_request_is_deterministic_and_canonically_hashable(self):
        """Identical pairs produce an identical payload with a stable
        canonical request hash — the property replay fixtures key on
        (IMPLEMENTATION.md §4.2)."""
        first = build_validation_request(make_pairs(3), config=ValidatorConfig())
        second = build_validation_request(make_pairs(3), config=ValidatorConfig())
        assert first == second
        assert canonical_request_hash("structured", first) == canonical_request_hash(
            "structured", second
        )


class TestParseValidationOutput:
    def test_happy_path_joins_verdicts_back_to_pairs(self):
        pairs = make_pairs(3)
        raw = {
            "verdicts": [
                {"pair_index": 0, "supported": True},
                {"pair_index": 2, "supported": False},
                {"pair_index": 1, "supported": True},
            ]
        }
        verdicts = parse_validation_output(raw, pairs)
        assert len(verdicts) == 3
        by_pair = {v.pair_index: v for v in verdicts}
        assert by_pair[0].supported is True
        assert by_pair[2].supported is False
        # The join carries the pair's sentence/block identities through.
        assert by_pair[2].sentence_index == pairs[2].sentence_index
        assert by_pair[2].document_index == pairs[2].document_index

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param({}, id="missing-verdicts"),
            pytest.param({"verdicts": "yes"}, id="verdicts-not-a-list"),
            pytest.param(
                {"verdicts": [{"pair_index": 0, "supported": True}]},
                id="missing-a-pair",
            ),
            pytest.param(
                {
                    "verdicts": [
                        {"pair_index": 0, "supported": True},
                        {"pair_index": 1, "supported": True},
                        {"pair_index": 1, "supported": False},
                    ]
                },
                id="duplicate-pair-index",
            ),
            pytest.param(
                {
                    "verdicts": [
                        {"pair_index": 0, "supported": True},
                        {"pair_index": 7, "supported": True},
                    ]
                },
                id="unknown-pair-index",
            ),
            pytest.param(
                {
                    "verdicts": [
                        {"pair_index": 0, "supported": "yes"},
                        {"pair_index": 1, "supported": True},
                    ]
                },
                id="non-boolean-supported",
            ),
            pytest.param(
                {
                    "verdicts": [
                        {"pair_index": 0},
                        {"pair_index": 1, "supported": True},
                    ]
                },
                id="missing-supported",
            ),
        ],
    )
    def test_malformed_output_raises_the_typed_error(self, raw):
        """Anything outside the schema raises the typed error that
        drives the retry-once path — never a bare KeyError/ValueError
        crash (IMPLEMENTATION.md §4.3, the #10/#16 convention)."""
        with pytest.raises(MalformedValidationOutputError):
            parse_validation_output(raw, make_pairs(2))
