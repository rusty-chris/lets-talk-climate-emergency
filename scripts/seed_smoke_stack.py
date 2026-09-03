"""Seed the replay-composed smoke stack (review finding #231).

Runs as the `smoke-seeder` compose service before the api starts. When
`CLIMATE_CHAT_PROVIDER` is not `replay` (the default dev/paused stacks)
it exits 0 immediately — the seeder is inert everywhere except the
live-path smoke environment (tests/smoke/test_starter_live_replay.py).

Under the replay environment it makes the composed stack able to serve a
real grounded answer and the flagship chart with ZERO live API calls:

1. **Index the synthetic corpus.** The committed Aurelian-Basin fixture
   corpus (real `chunk_document` output over synthetic docs,
   tests/_indexing_fixtures.py) is embedded with the REAL pinned bge-m3
   and indexed into the composed qdrant under the smoke corpus version —
   so the api's boot-time version check and its real retrieval path
   (`service.main._LazyRetrieval`: bge-m3 query embedding, hybrid query,
   bge-reranker, threshold gate) run exactly as a live deploy would.
2. **Generate the replay fixtures, generate-then-serve.** Every LLM call
   the two smoke exchanges will make (classifier x2, generation stream,
   validator, chart planner) is driven through the REAL pipeline
   builders with a programmed `FakeAdapter` behind the
   `RecordingAdapter` writer — synthetic responses, recorded under the
   canonical request hash into the shared `CLIMATE_CHAT_REPLAY_DIR`
   volume the api's `ReplayAdapter` reads. The fixtures are generated
   HERE, inside the same image and against the same qdrant that will
   serve them, because the generation/validation request hashes embed
   the retrieved passages: real-model retrieval is deterministic within
   one environment, and generating in the serving environment removes
   any cross-host reproducibility gamble (the determinism decision
   recorded in tests/fixtures/replay/starter_smoke/README.md).
3. **Prove the round trip before the api boots.** Retrieval is re-run
   and the rebuilt generation request hash must resolve to a written
   fixture; the planner's spec is rendered through the SAME frame loader
   the api uses (`service.main.load_chart_pack_frames`). Any drift fails
   the seeder — and `docker compose up` — loudly.

Zero live network calls by construction: the inner adapter is a
`FakeAdapter`; the recorder env is injected (no real key anywhere).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROVIDER_REPLAY = "replay"

#: Synthetic usage riding each recorded response — small, honest numbers
#: so the spend tracker meters real-looking (micro-dollar) costs.
_CLASSIFIER_USAGE = {"input_tokens": 420, "output_tokens": 40}
_GENERATION_USAGE = {"input_tokens": 5200, "output_tokens": 90, "cache_read_input_tokens": 4096}
_VALIDATOR_USAGE = {"input_tokens": 700, "output_tokens": 30}
_PLANNER_USAGE = {"input_tokens": 900, "output_tokens": 120}


def _wait_for_qdrant(client, deadline_s: float = 120.0) -> None:
    """Belt over compose's depends_on: wait until qdrant answers."""
    deadline = time.monotonic() + deadline_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.get_collections()
            return
        except Exception as error:  # noqa: BLE001 - readiness probe
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"qdrant never became ready for seeding: {last_error}")


def _synthetic_answer_stream(retrieved) -> list[dict]:
    """A synthetic Anthropic-shaped stream: two cited factual sentences.

    The sentences are invented (Aurelian-Basin universe); each citation
    quotes the leading words of the retrieved passage it points at, so
    the cited text is honest against the document blocks the request
    carries. Includes a calibrated term ("very likely") so the UI's
    annotation surface is exercised end to end.
    """

    def _passage_quote(passage) -> str:
        return " ".join(str(passage.payload.get("body", "")).split()[:12])

    def _passage_title(passage) -> str:
        metadata = passage.payload.get("citation_metadata") or {}
        return str(metadata.get("attribution_text") or passage.chunk_id)

    def _citation(index: int) -> dict:
        passage = retrieved.passages[index]
        return {
            "type": "content_block_delta",
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": _passage_quote(passage),
                    "document_index": index,
                    "document_title": _passage_title(passage),
                },
            },
        }

    def _text(text: str) -> dict:
        return {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}

    return [
        {"type": "message_start", "message": {"role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        _text(
            "Surface temperatures across the Aurelian Basin have very likely "
            "risen by one point nine degrees since the fictional baseline "
            "period. "
        ),
        _citation(0),
        _text(
            "Attribution studies comparing the observed decline against "
            "counterfactual simulations find the basin drying signal is very "
            "likely human-driven."
        ),
        _citation(1),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": dict(_GENERATION_USAGE),
        },
        {"type": "message_stop"},
    ]


def _smoke_chart_spec() -> dict:
    """The planner-fixture ChartSpec: a full-range line over the one smoke
    pack dataset. Validated at seed time by the real planner flow AND
    rendered through the api's own frame loader before the api boots."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "smoke-co2-line",
        "chart_type": "line",
        "title": "Synthetic CO2 record (invented)",
        "time_range_ce": [1959, 2024],
        "series": [
            {
                "id": "co2",
                "label": "CO2 (ppm, invented)",
                "unit": "ppm",
                "dataset": "syn_smoke_co2",
            }
        ],
    }


def seed(client, env) -> None:
    """Index the corpus and generate the replay fixtures (see module docs)."""
    from charts.pack import load_chart_pack_frames
    from charts.planner import PlannedChart, plan_chart_request
    from charts.render import render_chart
    from rag.citation_validator import (
        ValidatorConfig,
        build_entailment_pairs,
        segment_answer_sentences,
        validate_exchange,
    )
    from rag.generation import (
        GenerationConfig,
        build_generation_request,
        stream_grounded_answer,
    )
    from rag.indexing import Bgem3EmbeddingModel, build_index
    from rag.provider import (
        FakeAdapter,
        RawProviderResponse,
        RecordingAdapter,
        StructuredResult,
        canonical_request_hash,
    )
    from rag.query import Route, process_query
    from rag.retrieval import (
        BgeRerankerV2M3,
        RetrievalConfig,
        RetrievedPassages,
        load_threshold_artifact,
        retrieve,
    )
    from service.app import _grounded_answer_from_sse
    from service.config import DEFAULT_COLLECTION_NAME
    from service.starter_cache import STARTER_QUESTIONS
    from tests._indexing_fixtures import fixture_corpus

    corpus_version = env["CLIMATE_CHAT_CORPUS_VERSION"]
    corpus_vintage = env["CLIMATE_CHAT_CORPUS_VINTAGE"]
    collection = (env.get("CLIMATE_CHAT_COLLECTION") or "").strip() or DEFAULT_COLLECTION_NAME
    replay_dir = Path(env["CLIMATE_CHAT_REPLAY_DIR"])
    manifest_path = Path(env["CLIMATE_CHAT_DATASET_MANIFEST"])
    pack_dir = Path(env["CLIMATE_CHAT_CHART_PACK_DIR"])
    threshold_path = Path(env["CLIMATE_CHAT_THRESHOLD_ARTIFACT"])
    site_url = env.get("CLIMATE_CHAT_SITE_URL", "")

    grounded_question = STARTER_QUESTIONS[0]
    chart_question = next(q for q in STARTER_QUESTIONS if q.startswith("Show me"))

    # 1. Index the synthetic corpus with the REAL pinned embedder, under
    # the smoke corpus version the api is configured to expect.
    print(f"seed_smoke_stack: indexing the fixture corpus into {collection!r}…", flush=True)
    chunks, records = fixture_corpus()
    embedder = Bgem3EmbeddingModel()
    report = build_index(
        client,
        collection,
        chunks,
        records,
        embedding_model=embedder,
        corpus_version=corpus_version,
    )
    print(f"seed_smoke_stack: indexed {report.indexed_chunk_count} chunks.", flush=True)

    # 2. Retrieval exactly as service.main._LazyRetrieval will run it.
    calibration = load_threshold_artifact(threshold_path)
    reranker = BgeRerankerV2M3()
    retrieval_config = RetrievalConfig(
        refusal_threshold=calibration.threshold,
        corpus_coverage=(),
    )

    def run_retrieval(decision):
        return retrieve(
            client,
            collection,
            decision,
            embedding_model=embedder,
            reranker=reranker,
            config=retrieval_config,
            expected_corpus_version=corpus_version,
        )

    # 3. The FakeAdapter-driven recording writer: zero live calls by
    # construction (the inner adapter is a fake; the env is injected).
    fake = FakeAdapter()
    recorder = RecordingAdapter(
        fake,
        replay_dir,
        env={"CLIMATE_CHAT_RECORD": "1", "ANTHROPIC_API_KEY": "seed-synthetic-not-a-real-key"},
    )

    # Grounded starter: classifier -> retrieval -> generation stream ->
    # validator, each request built by the REAL pipeline code.
    fake.queue(
        "structured",
        StructuredResult(
            {"scope": "in_scope", "rewritten_query": grounded_question, "language": "en"},
            usage=dict(_CLASSIFIER_USAGE),
        ),
    )
    decision = process_query(recorder, grounded_question, [])
    assert decision.route is Route.RETRIEVAL, decision

    retrieved = run_retrieval(decision)
    assert isinstance(retrieved, RetrievedPassages), (
        f"seeded retrieval refused: {retrieved!r} — the smoke threshold must admit "
        "the synthetic corpus"
    )
    assert len(retrieved.passages) >= 2, "the citation fixture cites two passages"

    fake.queue(
        "generate_stream",
        RawProviderResponse(payload={}, events=tuple(_synthetic_answer_stream(retrieved))),
    )
    generation_config = GenerationConfig()
    transcript = list(
        stream_grounded_answer(
            recorder,
            retrieved,
            grounded_question,
            config=generation_config,
            corpus_vintage=corpus_vintage,
        )
    )
    assert transcript[-1]["event"] == "footer", (
        f"the synthetic stream must complete with the footer, ended with {transcript[-1]!r}"
    )

    answer = _grounded_answer_from_sse(transcript, retrieved, corpus_vintage)
    pairs = build_entailment_pairs(segment_answer_sentences(transcript), answer.cited_passages)
    assert pairs, "the grounded fixture must produce entailment pairs"
    fake.queue(
        "structured",
        StructuredResult(
            {"verdicts": [{"pair_index": pair.pair_index, "supported": True} for pair in pairs]},
            usage=dict(_VALIDATOR_USAGE),
        ),
    )
    outcome = validate_exchange(recorder, answer, transcript, config=ValidatorConfig())
    assert outcome.validated, outcome

    # Chart starter: classifier -> planner (validates the spec for real).
    fake.queue(
        "structured",
        StructuredResult(
            {"scope": "chart_request", "rewritten_query": chart_question, "language": "en"},
            usage=dict(_CLASSIFIER_USAGE),
        ),
    )
    chart_decision = process_query(recorder, chart_question, [])
    assert chart_decision.route is Route.CHART, chart_decision
    fake.queue(
        "structured",
        StructuredResult(
            {"outcome": "spec", "spec": _smoke_chart_spec()},
            usage=dict(_PLANNER_USAGE),
        ),
    )
    planned = plan_chart_request(recorder, chart_decision.chart_request or "", manifest_path)
    assert isinstance(planned, PlannedChart), planned

    # 4. Round-trip proofs, before the api ever boots.
    # (a) Retrieval determinism: a fresh retrieval must rebuild the exact
    # generation request the fixture was recorded under.
    replayed_retrieval = run_retrieval(decision)
    assert isinstance(replayed_retrieval, RetrievedPassages)
    rebuilt = build_generation_request(
        replayed_retrieval, grounded_question, config=generation_config
    )
    rebuilt_hash = canonical_request_hash("generate_stream", rebuilt)
    assert (replay_dir / f"{rebuilt_hash}.json").is_file(), (
        "re-running retrieval produced a different generation request hash — "
        "the environment is not deterministic; the replay smoke would miss its fixture"
    )
    # (b) The planner's spec renders through the SAME frame loader the api
    # uses (service.main.load_chart_pack_frames / _LazyRenderer).
    raw_manifest, frames = load_chart_pack_frames(manifest_path, pack_dir)
    artifact = render_chart(planned.spec, frames=frames, manifest=raw_manifest, site_url=site_url)
    assert artifact.alt_text.strip(), "the rendered chart must carry alt text"

    written = sorted(path.name for path in replay_dir.glob("*.json"))
    print(f"seed_smoke_stack: wrote {len(written)} replay fixtures: {written}", flush=True)


def main() -> int:
    provider = os.environ.get("CLIMATE_CHAT_PROVIDER", "").strip().lower()
    if provider != PROVIDER_REPLAY:
        print(
            "seed_smoke_stack: CLIMATE_CHAT_PROVIDER is not 'replay' — nothing to seed.",
            flush=True,
        )
        return 0

    # Diagnosability first: the seeder runs inside `docker compose up`,
    # whose failure hides container logs — name the environment loudly.
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"

    def _status(present: bool) -> str:
        return "present" if present else "MISSING"

    print(
        f"seed_smoke_stack: python {sys.version.split()[0]}; cwd {Path.cwd()}; "
        f"tests pkg {_status(Path('tests/__init__.py').is_file())}; "
        f"smoke fixtures {_status(Path('tests/fixtures/smoke').is_dir())}; "
        f"hub cache {hub_cache} {_status(hub_cache.is_dir())}",
        flush=True,
    )

    # Heavy imports only past the gate: the inert (default/paused) runs
    # must exit in milliseconds, and only the replay stack pays for torch.
    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.environ["CLIMATE_CHAT_QDRANT_URL"])
    _wait_for_qdrant(client)
    seed(client, os.environ)
    print("seed_smoke_stack: done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
