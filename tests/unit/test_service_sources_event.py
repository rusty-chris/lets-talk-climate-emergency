"""Issue #220 RED — the SSE ``sources`` event and its §2.1 licensing wall.

The #22 wire carries citations, badges and a footer but no event exposing
the retrieved passages themselves, so the DESIGN §3.6/§7.2 sources panel
is not representable in the UI. This suite pins the missing surface:

- ``service.app.SOURCES_EVENT`` ("sources"), a member of the declared
  ``SSE_EVENT_NAMES`` vocabulary (the #230 parity guard carries it to
  the UI's HANDLED_EVENTS).
- The pure licensing wall :func:`service.app.bounded_excerpt`, keyed on
  the manifest's ``permitted_context`` (the §2.1 invariant rule — never
  tier labels): ``open`` documents carry a fuller bounded excerpt,
  ``non-commercial-educational`` / ``permission-on-file`` a strictly
  tighter one, and ANY unrecognised context fails CLOSED (no excerpt at
  all). NO wire frame ever carries full Tier-B text.
- The pure, seam-injectable builder
  :func:`service.app.build_sources_event` — server-side composition
  over the retrieval result only, so provider requests (and therefore
  the replay fixtures' request hashes) and the exchange log are both
  provably untouched.
- The /chat pipeline: exactly one sources event per grounded exchange,
  after ``meta`` and before the first ``text``; refusal / canned /
  chart / paused / cached-starter exchanges emit NONE.

FLAGGED DECISIONS (for orchestrator ratification, pinned here):

1. Excerpt bounds are CHARACTER counts over a verbatim body prefix:
   ``open`` <= 600 chars, ``non-commercial-educational`` and
   ``permission-on-file`` <= 300 chars. Characters (not sentences) keep
   the wall deterministic and language-agnostic; a verbatim prefix is
   the only unadapted truncation (the Carbon Brief ND rule).
2. The wire ``source_tier`` is a DISPLAY label derived from
   ``permitted_context`` (open->A, non-commercial-educational->B,
   permission-on-file->C): the chunk payload does not carry the
   manifest's ``source_tier`` today, and deriving display from the same
   key the wall enforces on can never disagree with enforcement.
3. Entries carry ``canonical_url`` (the §3.6 deep link) beyond the
   issue's minimum field list.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from fastapi.testclient import TestClient

from ingestion.manifest import DOCUMENT_PERMITTED_CONTEXTS
from rag.generation import (
    GENERATION_MODEL_DEFAULT,
    GenerationConfig,
    build_generation_request,
)
from rag.provider import canonical_request_hash
from rag.retrieval import RerankedPassage, RetrievedPassages
from service.app import (
    EXCERPT_BOUNDS,
    META_EVENT,
    OPEN_EXCERPT_MAX_CHARS,
    RESTRICTED_EXCERPT_MAX_CHARS,
    SOURCE_ENTRY_KEYS,
    SOURCE_TIER_LABELS,
    SOURCES_EVENT,
    SSE_EVENT_NAMES,
    bounded_excerpt,
    build_sources_event,
    create_app,
    format_sse_event,
)
from service.budget import ServiceMode
from tests._generation_fixtures import make_payload, transport_stream_events
from tests._service_fixtures import (
    FakeRetrieve,
    classifier_output,
    events_named,
    make_harness,
    post_chat,
)

# ---------------------------------------------------------------------------
# Synthetic passages with explicit licensing contexts (the fixture default
# carries the non-canonical "public-noncommercial", which this suite pins
# as the FAIL-CLOSED path — a useful honesty check, not an accident).
# ---------------------------------------------------------------------------

#: A body comfortably longer than every bound (no substring collisions).
LONG_BODY = " ".join(
    f"Invented Aurelian sentence number {i} about entirely fictional basin warming."
    for i in range(40)
)
assert len(LONG_BODY) > OPEN_EXCERPT_MAX_CHARS


def make_source_passage(
    index: int,
    *,
    permitted_context: Any = "open",
    source_type: str = "evidence",
    body: str | None = None,
    title: Any = "Meridian Assessment, Chapter 1 (invented)",
    attribution: Any = "Meridian Climate Assessment Cycle 3 (invented)",
    drop_metadata: bool = False,
) -> RerankedPassage:
    payload = make_payload(index, source_type=source_type)
    if body is not None:
        payload["body"] = body
    if drop_metadata:
        payload.pop("citation_metadata")
    else:
        payload["citation_metadata"] = {
            "title": title,
            "licence": "synthetic-licence",
            "attribution_text": attribution,
            "canonical_url": f"https://example.invalid/meridian/{index}",
            "permitted_context": permitted_context,
        }
    return RerankedPassage(
        chunk_id=payload["chunk_id"],
        rerank_score=0.95 - index * 0.05,
        clears_threshold=True,
        payload=payload,
    )


def retrieved_with(*passages: RerankedPassage) -> RetrievedPassages:
    return RetrievedPassages(passages=tuple(passages))


class TestVocabulary:
    def test_sources_event_name_and_membership(self) -> None:
        assert SOURCES_EVENT == "sources"
        assert SOURCES_EVENT in SSE_EVENT_NAMES, (
            "the #230 vocabulary must declare the sources event so the "
            "bidirectional UI parity guard carries it"
        )

    def test_excerpt_bounds_cover_exactly_the_manifest_contexts(self) -> None:
        """The wall is keyed on the manifest's permitted_context vocabulary
        — a NEW manifest context value fails HERE until a bound is decided
        for it (fail-closed by construction, never by accident)."""
        assert set(EXCERPT_BOUNDS) == set(DOCUMENT_PERMITTED_CONTEXTS)
        assert set(SOURCE_TIER_LABELS) == set(DOCUMENT_PERMITTED_CONTEXTS)

    def test_the_restricted_bound_is_strictly_tighter(self) -> None:
        assert RESTRICTED_EXCERPT_MAX_CHARS < OPEN_EXCERPT_MAX_CHARS
        assert EXCERPT_BOUNDS["open"] == OPEN_EXCERPT_MAX_CHARS
        assert EXCERPT_BOUNDS["non-commercial-educational"] == RESTRICTED_EXCERPT_MAX_CHARS
        assert EXCERPT_BOUNDS["permission-on-file"] == RESTRICTED_EXCERPT_MAX_CHARS


class TestBoundedExcerpt:
    """The pure wall: verbatim bounded prefix, or nothing at all."""

    def test_open_short_body_rides_whole(self) -> None:
        body = "A short invented passage."
        excerpt = bounded_excerpt(body, "open")
        assert excerpt == body

    def test_open_long_body_is_the_verbatim_bounded_prefix(self) -> None:
        excerpt = bounded_excerpt(LONG_BODY, "open")
        assert excerpt == LONG_BODY[:OPEN_EXCERPT_MAX_CHARS]

    def test_restricted_contexts_share_the_tight_bound(self) -> None:
        for context in ("non-commercial-educational", "permission-on-file"):
            excerpt = bounded_excerpt(LONG_BODY, context)
            assert excerpt == LONG_BODY[:RESTRICTED_EXCERPT_MAX_CHARS], context

    def test_unknown_context_fails_closed(self) -> None:
        """Anything outside the manifest vocabulary — including the
        legacy fixture string and near-misses — yields NO excerpt."""
        for context in (
            None,
            "",
            "Open",  # miscased: no normalisation, no benefit of the doubt
            "OPEN",
            "public-noncommercial",  # the pre-#220 fixture legacy value
            "commercial",
            "non-commercial",
            42,
        ):
            assert bounded_excerpt(LONG_BODY, context) is None, (
                f"permitted_context {context!r} must fail CLOSED (no excerpt)"
            )

    def test_never_exceeds_the_bound_for_any_context(self) -> None:
        """The invariant behind 'no full Tier-B text on the wire': for
        every context the wall knows, the excerpt is a prefix within the
        bound; for every context it does not, the excerpt is None."""
        huge = "x" * 10_000
        for context, bound in EXCERPT_BOUNDS.items():
            excerpt = bounded_excerpt(huge, context)
            assert excerpt is not None
            assert len(excerpt) <= bound
            assert huge.startswith(excerpt)
        assert bounded_excerpt(huge, "anything-else") is None


class TestBuildSourcesEvent:
    """The pure builder: retrieval result -> the one wire event."""

    def test_event_shape_one_entry_per_passage_in_order(self) -> None:
        retrieved = retrieved_with(
            make_source_passage(0),
            make_source_passage(1, permitted_context="non-commercial-educational"),
            make_source_passage(2, source_type="voices"),
        )
        event = build_sources_event(retrieved)
        assert event["event"] == SOURCES_EVENT
        entries = event["data"]["sources"]
        assert [entry["chunk_id"] for entry in entries] == [
            passage.chunk_id for passage in retrieved.passages
        ], "wire order is retrieval order (best first — the document-index order)"

    def test_entry_keys_are_exactly_the_closed_wire_shape(self) -> None:
        """No extra key can ever smuggle unbounded source text past the
        excerpt wall — the entry shape is CLOSED."""
        retrieved = retrieved_with(
            make_source_passage(0),
            make_source_passage(1, permitted_context="does-not-exist"),
        )
        for entry in build_sources_event(retrieved)["data"]["sources"]:
            assert set(entry) == set(SOURCE_ENTRY_KEYS)

    def test_metadata_carriage_and_title_fallbacks(self) -> None:
        titled = make_source_passage(0, title="An Invented Chapter Title")
        untitled = make_source_passage(1, title=None, attribution="Fallback Attribution (invented)")
        bare = make_source_passage(2, title=None, attribution=None)
        entries = build_sources_event(retrieved_with(titled, untitled, bare))["data"]["sources"]

        assert entries[0]["title"] == "An Invented Chapter Title"
        assert entries[0]["doc_id"] == titled.payload["doc_id"]
        assert entries[0]["attribution_text"] == "Meridian Climate Assessment Cycle 3 (invented)"
        assert entries[0]["canonical_url"] == "https://example.invalid/meridian/0"
        assert entries[0]["source_type"] == "evidence"
        assert entries[0]["permitted_context"] == "open"

        # Title falls back to the attribution, then to the doc id.
        assert entries[1]["title"] == "Fallback Attribution (invented)"
        assert entries[2]["title"] == bare.payload["doc_id"]

    def test_source_tier_display_labels(self) -> None:
        retrieved = retrieved_with(
            make_source_passage(0, permitted_context="open"),
            make_source_passage(1, permitted_context="non-commercial-educational"),
            make_source_passage(2, permitted_context="permission-on-file"),
            make_source_passage(3, permitted_context="never-heard-of-it"),
        )
        tiers = [e["source_tier"] for e in build_sources_event(retrieved)["data"]["sources"]]
        assert tiers == ["A", "B", "C", None]

    def test_excerpts_bounded_per_context(self) -> None:
        short_body = "One short invented sentence."
        retrieved = retrieved_with(
            make_source_passage(0, permitted_context="open", body=LONG_BODY),
            make_source_passage(1, permitted_context="non-commercial-educational", body=LONG_BODY),
            make_source_passage(2, permitted_context="permission-on-file", body=short_body),
        )
        entries = build_sources_event(retrieved)["data"]["sources"]

        assert entries[0]["excerpt"] == LONG_BODY[:OPEN_EXCERPT_MAX_CHARS]
        assert entries[0]["excerpt_truncated"] is True

        assert entries[1]["excerpt"] == LONG_BODY[:RESTRICTED_EXCERPT_MAX_CHARS]
        assert entries[1]["excerpt_truncated"] is True

        # A body within the bound rides whole, unmarked.
        assert entries[2]["excerpt"] == short_body
        assert entries[2]["excerpt_truncated"] is False

    def test_fail_closed_entry_is_metadata_only(self) -> None:
        """An unrecognised permitted_context (or missing citation
        metadata) still LISTS the passage — attribution honesty — but
        with no excerpt at all and no tier label."""
        unknown = make_source_passage(0, permitted_context="mystery-licence", body=LONG_BODY)
        stripped = make_source_passage(1, body=LONG_BODY, drop_metadata=True)
        entries = build_sources_event(retrieved_with(unknown, stripped))["data"]["sources"]
        for entry in entries:
            assert entry["excerpt"] is None
            assert entry["excerpt_truncated"] is False
            assert entry["source_tier"] is None
            assert entry["chunk_id"]  # metadata-honest: the passage IS listed
        assert LONG_BODY not in json.dumps(entries)

    def test_no_wire_frame_ever_carries_full_tier_b_text(self) -> None:
        """THE licensing wall, at the serialised frame: a Tier-B body
        longer than the tight bound never appears whole in the SSE wire
        text — not in the excerpt, not in any other field."""
        for context in ("non-commercial-educational", "permission-on-file"):
            retrieved = retrieved_with(
                make_source_passage(0, permitted_context=context, body=LONG_BODY)
            )
            wire_text = format_sse_event(build_sources_event(retrieved))
            assert LONG_BODY not in wire_text, (
                f"full {context} text leaked onto the wire — the §2.1 wall is breached"
            )
            # The bounded excerpt itself IS on the wire (JSON-escaped or
            # not, the raw prefix contains no escapables by construction).
            assert LONG_BODY[:RESTRICTED_EXCERPT_MAX_CHARS] in wire_text


def retrieval_harness(tmp_path, **kwargs):
    """A harness programmed for one full retrieval-route exchange."""
    harness = make_harness(tmp_path, **kwargs)
    harness.adapter.queue("structured", classifier_output())
    harness.adapter.queue("generate_stream", transport_stream_events())
    return harness


class TestChatPipelineSourcesEmission:
    def test_grounded_exchange_emits_one_sources_event_after_meta_before_text(
        self, tmp_path
    ) -> None:
        """The pinned ordering: meta, then the sources event, then the
        generation stream — the panel is renderable before the first
        token lands."""
        harness = retrieval_harness(tmp_path)
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        names = [event["event"] for event in events]

        assert names.count(SOURCES_EVENT) == 1, "exactly one sources event per grounded exchange"
        assert names[0] == META_EVENT
        assert names[1] == SOURCES_EVENT, "the sources event follows meta immediately"
        assert names.index(SOURCES_EVENT) < names.index("text")

    def test_sources_entries_mirror_the_retrieved_passages(self, tmp_path) -> None:
        retrieved = retrieved_with(
            make_source_passage(0),
            make_source_passage(1, source_type="voices"),
        )
        harness = retrieval_harness(tmp_path, retrieve=FakeRetrieve(result=retrieved))
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        (sources_event,) = events_named(events, SOURCES_EVENT)
        assert [entry["chunk_id"] for entry in sources_event["data"]["sources"]] == [
            passage.chunk_id for passage in retrieved.passages
        ]

    def test_sources_builder_seam_is_injectable_and_emitted_verbatim(self, tmp_path) -> None:
        """The pure-builder seam: an injected build_sources is called
        exactly once with the retrieval result, and its returned mapping
        is emitted verbatim."""
        sentinel = {
            "event": SOURCES_EVENT,
            "data": {"sources": [{"chunk_id": "seam-sentinel", "excerpt": None}]},
        }
        calls: list[Any] = []

        def fake_build_sources(retrieved: RetrievedPassages) -> dict[str, Any]:
            calls.append(retrieved)
            return sentinel

        pinned_result = retrieved_with(make_source_passage(0))
        harness = retrieval_harness(tmp_path, retrieve=FakeRetrieve(result=pinned_result))
        deps = replace(harness.deps, build_sources=fake_build_sources)
        app = create_app(harness.config, deps)
        events = post_chat(TestClient(app), "Why is the basin warming?")

        # The seam receives the SAME RetrievedPassages object the retrieve
        # dep produced, exactly once, and its mapping is emitted verbatim.
        assert len(calls) == 1
        assert calls[0] is pinned_result
        (sources_event,) = events_named(events, SOURCES_EVENT)
        assert sources_event["data"] == sentinel["data"]

    def test_refusal_emits_no_sources_event(self, tmp_path) -> None:
        """An honest refusal retrieved nothing above threshold: dressing
        it with a sources panel would fabricate grounding."""
        from tests._generation_fixtures import make_refusal

        harness = make_harness(tmp_path, retrieve=FakeRetrieve(result=make_refusal()))
        harness.adapter.queue("structured", classifier_output())
        events = post_chat(TestClient(harness.app), "Why is the basin warming?")
        assert events_named(events, SOURCES_EVENT) == []

    def test_canned_route_emits_no_sources_event(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
        events = post_chat(TestClient(harness.app), "Recommend me a laptop")
        assert events_named(events, SOURCES_EVENT) == []

    def test_chart_route_emits_no_sources_event(self, tmp_path) -> None:
        harness = make_harness(tmp_path)
        harness.adapter.queue(
            "structured", classifier_output(scope="chart_request", rewritten="plot basin co2")
        )
        events = post_chat(TestClient(harness.app), "Plot the basin CO2")
        assert events_named(events, SOURCES_EVENT) == []

    def test_paused_and_cached_starter_emit_no_sources_event(self, tmp_path) -> None:
        """PAUSED serves furniture and dated cache — zero adapter calls
        and zero sources events (a cached answer's citations already ride
        the answer event; no fresh retrieval happened)."""
        from service.starter_cache import STARTER_QUESTIONS

        harness = make_harness(tmp_path)
        harness.tracker.record_usage(
            "claude-haiku-4-5", {"input_tokens": 2_000_000, "output_tokens": 200_000}
        )
        assert harness.tracker.mode() is ServiceMode.PAUSED
        client = TestClient(harness.app)

        paused_events = post_chat(client, "Why is the basin warming?")
        assert events_named(paused_events, SOURCES_EVENT) == []

        cached_events = post_chat(client, STARTER_QUESTIONS[0])
        assert events_named(cached_events, SOURCES_EVENT) == []

    def test_provider_request_unchanged_by_the_sources_surface(self, tmp_path) -> None:
        """The sources event is server-side composition over the
        retrieval result: the generation request the adapter sees is
        byte-identical to rag.generation.build_generation_request over
        the same inputs, so recorded replay fixtures' request hashes
        (rag.provider.canonical_request_hash) are unaffected."""
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")

        (call,) = harness.adapter.calls_to("generate_stream")
        seen_payload = dict(call.payload)
        assert '"sources"' not in json.dumps(seen_payload), (
            "no sources-derived key may ride the provider request"
        )
        # The default FakeRetrieve serves make_retrieved() (deterministic),
        # so an independently built request must hash identically to the
        # one the adapter actually saw.
        expected = build_generation_request(
            _default_retrieved(),
            "Why is the basin warming?",
            config=GenerationConfig(model=GENERATION_MODEL_DEFAULT),
        )
        assert canonical_request_hash("generate_stream", seen_payload) == canonical_request_hash(
            "generate_stream", expected
        )

    def test_exchange_log_gains_no_new_surface(self, tmp_path) -> None:
        """Exchange logging is UNCHANGED: the record's closed key set
        (the §9 schema) gains nothing — no sources key, no excerpt
        carriage, no new identifier surface."""
        harness = retrieval_harness(tmp_path)
        post_chat(TestClient(harness.app), "Why is the basin warming?")

        log_path = tmp_path / "logs" / "exchanges.jsonl"
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        record = json.loads(lines[-1])
        assert set(record) == {
            "exchange_id",
            "timestamp",
            "question",
            "route",
            "answer_text",
            "retrieved_chunk_ids",
            "citations",
            "validation",
            "usage_records",
            "exclude_from_harvest",
            "feedback",
            # Issue #57: the semantic-cache linkage key (None on this route).
            "cached_from",
        }


def _default_retrieved() -> RetrievedPassages:
    from tests._generation_fixtures import make_retrieved

    return make_retrieved()
