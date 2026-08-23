"""Generation request builders: seam payload + Anthropic API mapping
(issue #12) — RED.

Unit tier, pure: the §3.4 contract tests live against the request
builders, not the network (IMPLEMENTATION.md §1/§4.3) — every violation
raises before any transport could exist to bill. The prompt-caching
contract (static-prefix-first ordering, byte-stable prefix, Haiku 4.5's
4096-token cacheable-prefix floor) is pinned here too: it is a property
of request CONSTRUCTION, provable without a key.
"""

from __future__ import annotations

import json

import pytest

from rag import generation
from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    HAIKU_MIN_CACHEABLE_PREFIX_TOKENS,
    OPUS_BEST_MODEL,
    BestModeNotEnabledError,
    GenerationConfig,
    assert_cacheable_prefix,
    build_generation_request,
    estimate_tokens_lower_bound,
    load_system_prompt,
)
from rag.provider import (
    GENERATION_CACHE_CONTROL,
    MAX_GENERATE_DOCUMENTS,
    AnswerWithCitations,
    AnthropicAdapter,
    FakeAdapter,
    ProviderContractError,
    build_anthropic_messages_request,
)
from rag.retrieval import GENERATION_TOP_K
from tests._generation_fixtures import make_retrieved

QUESTION = "How much has the invented basin warmed?"
OTHER_QUESTION = "What do the invented attribution studies conclude?"

# Structured-output / tool configuration must never appear anywhere in a
# citations request (DESIGN §3.4; the live API 400s — spike-03 probe 1).
FORBIDDEN_GENERATION_KEYS = (
    "tools",
    "tool_choice",
    "output_schema",
    "output_config",
    "output_format",
    "response_format",
    "structured_output",
)


def _request(count=3, *, question=QUESTION, config=None, **retrieved_kwargs):
    retrieved = make_retrieved(count, **retrieved_kwargs)
    return build_generation_request(
        retrieved, question, config=config or GenerationConfig()
    ), retrieved


# ---------------------------------------------------------------------------
# Seam payload: documents in the spike-proven custom-content shape
# ---------------------------------------------------------------------------


class TestSeamDocuments:
    def test_one_custom_content_block_per_passage_in_rank_order(self):
        """Spike-03's proven shape, one block per citable unit (§3.3):
        type document, source.type content, the passage BODY as the sole
        citable text (the context header travels in `context`, never
        cited — the spike lesson made contractual in ingestion.blocks)."""
        request, retrieved = _request(3)
        documents = request["documents"]
        assert len(documents) == 3
        for block, passage in zip(documents, retrieved.passages, strict=True):
            assert block["type"] == "document"
            assert block["source"]["type"] == "content"
            assert block["source"]["content"] == [{"type": "text", "text": passage.payload["body"]}]
            # Traceability rides `context` (model-visible, never cited):
            # chunk id resolution and the #143 provenance flags.
            assert passage.chunk_id in block["context"]
            assert passage.payload["context_header"] not in block["source"]["content"][0]["text"]

    def test_block_titles_carry_attribution(self):
        request, retrieved = _request(2)
        for block, passage in zip(request["documents"], retrieved.passages, strict=True):
            assert block["title"] == passage.payload["citation_metadata"]["attribution_text"]

    def test_every_block_enables_citations_never_a_mixed_set(self):
        """§3.4 all-or-none, pinned at CONSTRUCTION: the builder cannot
        emit an uncited block, so a mixed cited/uncited set is not merely
        rejected downstream — it is unbuildable (spike-03 probe 2: the
        live API 400s on a mixture)."""
        request, _ = _request(GENERATION_TOP_K)
        assert request["documents"], "builder produced no documents"
        for block in request["documents"]:
            assert block["citations"] == {"enabled": True}

    def test_more_than_top_k_passages_refuses_before_any_request_exists(self):
        """§3.4: the generation call is bounded to the reranked top-8.
        The bound is OUR design rule (the API imposes none — spike-03
        probe 3), so the builder enforces it itself rather than relying
        on the seam validator."""
        oversized = make_retrieved(GENERATION_TOP_K + 1)
        with pytest.raises(generation.GenerationContractError):
            build_generation_request(oversized, QUESTION, config=GenerationConfig())

    def test_top_k_bound_matches_the_seam_constant(self):
        """One design number, two modules: retrieval's top-8 IS the
        provider seam's document bound."""
        assert GENERATION_TOP_K == MAX_GENERATE_DOCUMENTS == 8


# ---------------------------------------------------------------------------
# Seam payload: config and the §3.4 structured-output ban
# ---------------------------------------------------------------------------


class TestSeamConfig:
    def test_default_model_is_haiku_and_config_is_data(self):
        request, _ = _request(2)
        assert request["config"]["model"] == GENERATION_MODEL_DEFAULT == "claude-haiku-4-5"
        assert request["config"]["max_tokens"] == GenerationConfig().max_tokens

    def test_config_never_carries_structured_output_configuration(self):
        """§4.3: the generation request NEVER sets structured output —
        the builder does not construct the keys at all."""
        request, _ = _request(3)
        for key in FORBIDDEN_GENERATION_KEYS:
            assert key not in request["config"], key
            assert key not in request, key

    def test_opus_requires_best_mode_flag(self):
        """DESIGN §3.3/§9: opus is 'best' mode behind the budget sub-cap;
        without the enabling flag the builder refuses loudly."""
        with pytest.raises(BestModeNotEnabledError):
            _request(2, config=GenerationConfig(model=OPUS_BEST_MODEL))

    def test_best_mode_enabled_selects_opus(self):
        def no_op_budget_guard_for_this_test(model_id: str) -> None:
            """Explicit unit-tier opt-out (finding #186): no budget
            behaviour wanted HERE; None would fail closed."""

        request, _ = _request(
            2,
            config=GenerationConfig(
                model=OPUS_BEST_MODEL,
                best_mode_enabled=True,
                budget_guard=no_op_budget_guard_for_this_test,
            ),
        )
        assert request["config"]["model"] == OPUS_BEST_MODEL

    def test_best_mode_invokes_budget_guard_before_building(self):
        """The #22 hook point: the sub-cap guard sees the opus model id
        before any request exists; a raising guard aborts the build."""
        seen: list[str] = []
        cfg = GenerationConfig(
            model=OPUS_BEST_MODEL, best_mode_enabled=True, budget_guard=seen.append
        )
        _request(2, config=cfg)
        assert seen == [OPUS_BEST_MODEL]

        class SubCapExceeded(RuntimeError):
            pass

        def refusing_guard(model_id: str) -> None:
            raise SubCapExceeded(model_id)

        with pytest.raises(SubCapExceeded):
            _request(
                2,
                config=GenerationConfig(
                    model=OPUS_BEST_MODEL, best_mode_enabled=True, budget_guard=refusing_guard
                ),
            )

    def test_default_haiku_path_never_consults_the_budget_guard(self):
        """The sub-cap gates opus only (§9); the default path must not
        acquire a hidden dependency on the guard."""
        seen: list[str] = []
        _request(2, config=GenerationConfig(budget_guard=seen.append))
        assert seen == []


class TestModelPolicyClosedVocabulary:
    """Finding #186: the model gate must be a closed vocabulary matched
    by FAMILY, and best mode must fail CLOSED (ADR-015) when the budget
    guard is missing — a single config value must never be the whole
    distance between gated demo mode and ungated premium spend."""

    def test_model_id_outside_the_allowlist_is_refused_before_any_request_exists(self):
        """Arbitrary and near-miss model ids refuse with zero adapter
        calls: model id stays config (DESIGN §3.3), but config drawn
        from a closed, gated vocabulary."""
        from rag.generation import generate_grounded_answer
        from tests._generation_fixtures import CORPUS_VINTAGE

        for model in (
            "gpt-5",
            "claude-opus-4-1",  # older, still premium-priced Opus
            "claude-sonnet-4-6",
            "claude-opus-4-88",  # prefix near-miss, not a 4-8 snapshot
            "",
        ):
            fake = FakeAdapter()
            with pytest.raises(generation.GenerationContractError):
                generate_grounded_answer(
                    fake,
                    make_retrieved(2),
                    QUESTION,
                    config=GenerationConfig(model=model),
                    corpus_vintage=CORPUS_VINTAGE,
                )
            assert fake.calls == [], model

    def test_opus_snapshot_alias_cannot_bypass_the_subcap(self):
        """A dated snapshot of the gated model (the id form Anthropic
        actually publishes) is the same premium model: family-matched,
        so it hits the best-mode gate, and the guard sees the exact id
        that will reach the API."""
        snapshot = f"{OPUS_BEST_MODEL}-20260115"
        with pytest.raises(BestModeNotEnabledError):
            _request(2, config=GenerationConfig(model=snapshot))

        seen: list[str] = []
        request, _ = _request(
            2,
            config=GenerationConfig(
                model=snapshot, best_mode_enabled=True, budget_guard=seen.append
            ),
        )
        assert seen == [snapshot]
        assert request["config"]["model"] == snapshot

    def test_best_mode_with_no_guard_installed_refuses(self):
        """ADR-015: the budget cut-off fails CLOSED. Best mode with no
        guard installed is a half-configured deployment, refused loudly;
        a tier that genuinely wants no budget behaviour passes an
        explicit no-op with a name that says so - never None."""
        with pytest.raises(generation.GenerationContractError):
            _request(
                2,
                config=GenerationConfig(
                    model=OPUS_BEST_MODEL, best_mode_enabled=True, budget_guard=None
                ),
            )

    def test_default_model_snapshot_alias_stays_ungated(self):
        """Family matching cuts both ways: a dated snapshot of the
        DEFAULT model is still the default family - allowed, ungated,
        guard untouched."""
        seen: list[str] = []
        snapshot = f"{GENERATION_MODEL_DEFAULT}-20260115"
        request, _ = _request(2, config=GenerationConfig(model=snapshot, budget_guard=seen.append))
        assert request["config"]["model"] == snapshot
        assert seen == []


# ---------------------------------------------------------------------------
# Seam payload: the system channel (finding #91) and the tone block
# ---------------------------------------------------------------------------


class TestSeamSystemChannel:
    def test_static_system_prompt_is_block_zero_verbatim(self):
        """The committed artifact rides the dedicated system channel
        (finding #91), verbatim, as the FIRST block — the static prefix
        every query shares."""
        request, _ = _request(2)
        system = list(request["system"])
        assert system, "seam payload carries no system channel"
        assert system[0]["type"] == "text"
        assert system[0]["text"] == load_system_prompt()

    def test_question_and_documents_never_leak_into_system(self):
        request, retrieved = _request(2)
        system_text = json.dumps(list(request["system"]))
        assert QUESTION not in system_text
        for passage in retrieved.passages:
            assert passage.payload["body"] not in system_text

    def test_messages_carry_the_question_and_never_role_system(self):
        request, _ = _request(2)
        messages = list(request["messages"])
        assert messages, "seam payload carries no messages"
        assert all(m["role"] != "system" for m in messages)
        assert QUESTION in json.dumps(messages)

    def test_tone_flag_alters_only_the_tone_block(self):
        """Adversarial routing (§3.1) may add ONE tone-instruction block,
        pinned to the volatile region directly after the static prefix —
        everything else in the request is byte-identical, so the cache
        prefix survives tone-flagged traffic."""
        flagged, _ = _request(3, tone_flag=True)
        unflagged, _ = _request(3, tone_flag=False)

        assert flagged["messages"] == unflagged["messages"]
        assert flagged["documents"] == unflagged["documents"]
        assert flagged["config"] == unflagged["config"]

        flagged_system = list(flagged["system"])
        unflagged_system = list(unflagged["system"])
        # Static prefix identical; exactly one extra block, at position 1.
        assert flagged_system[0] == unflagged_system[0]
        assert len(flagged_system) == len(unflagged_system) + 1
        assert flagged_system[2:] == unflagged_system[1:]
        tone_block = flagged_system[1]
        assert tone_block["type"] == "text"
        assert tone_block["text"].strip(), "tone block must carry an instruction"
        assert tone_block not in unflagged_system


# ---------------------------------------------------------------------------
# Anthropic Messages API mapping (the AnthropicAdapter's pure builder)
# ---------------------------------------------------------------------------


def _api_request(count=3, *, question=QUESTION, **retrieved_kwargs):
    seam, retrieved = _request(count, question=question, **retrieved_kwargs)
    return build_anthropic_messages_request(seam), seam, retrieved


class TestAnthropicRequestBuilder:
    def test_model_and_max_tokens_come_from_config(self):
        api, seam, _ = _api_request(2)
        assert api["model"] == seam["config"]["model"] == GENERATION_MODEL_DEFAULT
        assert api["max_tokens"] == seam["config"]["max_tokens"]

    def test_system_is_the_top_level_param_never_a_message(self):
        """Finding #91 live-verified: the Messages API rejects
        role:'system' entries; the system prompt's only channel is the
        top-level `system` param."""
        api, seam, _ = _api_request(2)
        assert list(api["system"]) == list(seam["system"]) or (
            # cache_control stamping is the ONLY permitted difference
            [{k: v for k, v in block.items() if k != "cache_control"} for block in api["system"]]
            == [dict(block) for block in seam["system"]]
        )
        assert all(message["role"] != "system" for message in api["messages"])

    def test_cache_breakpoint_on_last_static_block_only(self):
        """The static prefix ends at block 0: it carries the ephemeral
        cache_control breakpoint; nothing after it (volatile region) may
        carry one, or the 'stable prefix' would include per-query bytes."""
        api, _, _ = _api_request(2, tone_flag=True)
        system = list(api["system"])
        assert system[0].get("cache_control") == GENERATION_CACHE_CONTROL
        for block in system[1:]:
            assert "cache_control" not in block

    def test_prompt_prefix_is_byte_stable_across_different_questions(self):
        """The 4096-floor prerequisite: caching is an exact byte-prefix
        match, so two requests for DIFFERENT questions (and different
        passages) must serialise an identical prefix up to and including
        the cache breakpoint — any leak of volatile content into the
        prefix silently zeroes the cache."""
        api_a, _, _ = _api_request(3, question=QUESTION)
        api_b, _, _ = _api_request(4, question=OTHER_QUESTION)
        prefix_a = json.dumps(api_a["system"][0], sort_keys=True)
        prefix_b = json.dumps(api_b["system"][0], sort_keys=True)
        assert prefix_a == prefix_b
        assert api_a["model"] == api_b["model"]
        # And the volatile content really is volatile — the requests differ.
        assert api_a["messages"] != api_b["messages"]

    def test_static_prefix_clears_the_haiku_4096_token_floor(self):
        """Haiku 4.5 silently writes NO cache entry below a 4096-token
        prefix (orchestrator note on #12): the shipped static prefix must
        clear the floor by conservative token arithmetic, or the §9
        'less with prompt caching' cost model is fiction."""
        api, _, _ = _api_request(2)
        static_prefix = api["system"][0]["text"]
        assert estimate_tokens_lower_bound(static_prefix) >= HAIKU_MIN_CACHEABLE_PREFIX_TOKENS, (
            "static system prompt is under Haiku 4.5's 4096-token cacheable-prefix floor"
        )
        # The always-on guard agrees (no raise).
        assert assert_cacheable_prefix(api["system"]) is None

    def test_documents_ride_the_user_turn_before_the_question(self):
        """One user turn: the document blocks in seam order, then the
        question text — documents and question are both AFTER the cached
        prefix, in the volatile region."""
        api, seam, _ = _api_request(3)
        user_turns = [m for m in api["messages"] if m["role"] == "user"]
        assert user_turns, "no user turn in the API request"
        content = list(user_turns[0]["content"])
        document_blocks = [b for b in content if b.get("type") == "document"]
        assert document_blocks == list(seam["documents"])
        assert content[-1] == {"type": "text", "text": QUESTION}
        assert content.index(document_blocks[0]) < content.index(content[-1])

    def test_builder_never_emits_structured_output_config(self):
        """§4.3 heart: the builder does not merely omit `output_config` —
        no structured-output/tool key exists anywhere in the emitted
        kwargs (the live API 400s the combination; spike-03 probe 1)."""
        api, _, _ = _api_request(3)
        for key in FORBIDDEN_GENERATION_KEYS:
            assert key not in api, key

    def test_builder_refuses_over_bound_document_sets(self):
        """Even handed a corrupt seam payload, the API builder never
        CONSTRUCTS an over-bound request (§3.4: ours is a design bound,
        not an API one)."""
        seam, _ = _request(GENERATION_TOP_K)
        corrupt = dict(seam)
        corrupt["documents"] = list(seam["documents"]) * 2
        with pytest.raises(ProviderContractError):
            build_anthropic_messages_request(corrupt)

    def test_builder_refuses_mixed_cited_uncited_blocks(self):
        seam, _ = _request(2)
        corrupt = dict(seam)
        documents = [dict(block) for block in seam["documents"]]
        documents[1]["citations"] = {"enabled": False}
        corrupt["documents"] = documents
        with pytest.raises(ProviderContractError):
            build_anthropic_messages_request(corrupt)

    def test_builder_refuses_structured_output_in_config(self):
        seam, _ = _request(2)
        corrupt = dict(seam)
        corrupt["config"] = {**seam["config"], "output_schema": {"type": "object"}}
        with pytest.raises(ProviderContractError):
            build_anthropic_messages_request(corrupt)


# ---------------------------------------------------------------------------
# Conservative token floor arithmetic
# ---------------------------------------------------------------------------


class TestTokenFloorArithmetic:
    def test_lower_bound_is_conservative_and_monotone(self):
        """The estimate must UNDER-estimate (≤ one token per four chars —
        passing the floor check must guarantee the real tokenizer count
        clears it) and grow with the text."""
        sample = "calibrated evidence for the invented basin " * 50
        assert estimate_tokens_lower_bound("") == 0
        assert estimate_tokens_lower_bound(sample) <= len(sample) // 4
        assert estimate_tokens_lower_bound(sample * 2) >= estimate_tokens_lower_bound(sample)

    def test_short_static_prefix_refuses_loudly_naming_the_floor(self):
        """Silent cache failure is the enemy: an under-floor prefix must
        raise, and the message must name the 4096 floor so the failure
        is self-explaining."""
        with pytest.raises(generation.GenerationContractError, match="4096"):
            assert_cacheable_prefix([{"type": "text", "text": "far too short to cache"}])

    def test_comfortably_long_prefix_passes(self):
        long_text = "the invented basin assessment repeats itself deliberately " * 800
        assert assert_cacheable_prefix([{"type": "text", "text": long_text}]) is None


# ---------------------------------------------------------------------------
# Seam scaffolding pins (contract-validated skeleton, system channel)
# ---------------------------------------------------------------------------


class TestSeamScaffolding:
    """These pin the issue-12 provider scaffolding (skeleton + system
    channel). They exercise shared-validator behaviour and may already
    pass at red — the red commit body records which."""

    def _seam_kwargs(self, count=2):
        retrieved = make_retrieved(count)
        documents = [
            {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": [{"type": "text", "text": p.payload["body"]}],
                },
                "title": "t",
                "context": p.chunk_id,
                "citations": {"enabled": True},
            }
            for p in retrieved.passages
        ]
        return {
            "messages": [{"role": "user", "content": QUESTION}],
            "documents": documents,
            "config": {"model": GENERATION_MODEL_DEFAULT, "max_tokens": 64},
        }

    def test_fake_adapter_records_the_generate_system_channel(self):
        """`generate` grew the #91 system channel: recorded when given,
        OMITTED from the payload when None so pre-#12 recorded request
        hashes stay valid."""
        fake = FakeAdapter(generate_results=[AnswerWithCitations(text="ok")] * 2)
        kwargs = self._seam_kwargs()
        system = [{"type": "text", "text": "static prompt"}]
        fake.generate(**kwargs, system=system)
        fake.generate(**kwargs)
        with_system, without_system = fake.calls_to("generate")
        assert with_system.payload["system"] == system
        assert "system" not in without_system.payload

    def test_anthropic_adapter_skeleton_validates_before_transport(self):
        """'Stub, contract-validated': the skeleton runs the shared §3.4
        validator FIRST, so a violating request raises the contract error
        — never the not-implemented transport error."""
        adapter = AnthropicAdapter()
        kwargs = self._seam_kwargs()
        kwargs["documents"] = kwargs["documents"] * 5  # 10 > 8
        with pytest.raises(ProviderContractError):
            adapter.generate(**kwargs)

    def test_anthropic_adapter_keyless_generate_raises_before_any_network(self, monkeypatch):
        """Temporal pin, advanced at green (the ratified #10 pattern —
        disclosed in the green commit body): the red phase pinned
        'transport not implemented yet'; the transport now exists, so
        the pin advances to the invariant that outlives it — a keyless
        environment (this tier, keyless CI) fails loudly and LOCALLY,
        before any SDK client or network I/O exists."""
        from rag.provider import AnthropicKeyMissingError

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        adapter = AnthropicAdapter()
        with pytest.raises(AnthropicKeyMissingError):
            adapter.generate(**self._seam_kwargs())

    def test_anthropic_adapter_structured_transport_is_not_implemented_yet(self):
        """The structured transport is issue #13 work: still a loud
        NotImplementedError (never a silent fake) — the #10 classifier
        release-gate pins (test_classifier_accuracy) rely on every
        --no-batch item stopping exactly here."""
        adapter = AnthropicAdapter()
        with pytest.raises(NotImplementedError):
            adapter.structured(
                messages=[{"role": "user", "content": QUESTION}],
                schema={"type": "object"},
                config={"model": GENERATION_MODEL_DEFAULT, "max_tokens": 64},
            )
