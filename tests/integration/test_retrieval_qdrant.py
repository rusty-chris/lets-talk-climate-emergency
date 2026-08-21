"""The include-list fail-closed contract against the REAL composed Qdrant
server (review finding #174) — integration tier.

The entire #11 fail-closed proof previously ran on the `:memory:` client
emulation; finding #174 requires the same contract exercised against the
pinned qdrant/qdrant:v1.19.0 server, whose documented match semantics
satisfy a condition when ANY element of an array payload value matches —
the behaviour that lets `source_type: ["voices", "evidence"]` through an
include `MatchAny(["evidence"])` on science routes.

The embedder stays the deterministic hash fake (the #9 verdict: server
tests pin the MECHANISM; real weights live in the model smokes), and the
reranker is rigged to score rogue text top so any leak wins the ranking
and fails loudly.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from rag.query import ScopeClass
from rag.retrieval import RetrievalError, RetrievedPassages
from tests._docker import require_docker
from tests._indexing_fixtures import HashEmbeddingModel, build, fixture_corpus
from tests._retrieval_fixtures import (
    VOICES_MARKER,
    VOICES_PULL_QUERY,
    TableReranker,
    config,
    decision,
    implant_rogue_chunk,
    run_retrieve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QDRANT_URL = "http://127.0.0.1:6333"

ROGUE_MARKER = "lanternwood briefing organisers"

#: Every rogue source_type shape from findings #158 and #174: the original
#: mis-cased/unknown/null/missing matrix plus arrays and non-string scalars.
ROGUE_SOURCE_TYPES = (
    ("rogue-array-evidence", ["evidence"], True),
    ("rogue-array-both", ["voices", "evidence"], True),
    ("rogue-array-voices", ["voices"], True),
    ("rogue-int", 42, True),
    ("rogue-bool", True, True),
    ("rogue-object", {"v": "evidence"}, True),
    ("rogue-empty", "", True),
    ("rogue-padded", "evidence ", True),
    ("rogue-cased", "Voices", True),
    ("rogue-unknown", "campaign-material", True),
    ("rogue-null", None, True),
    ("rogue-missing", None, False),
)


@pytest.fixture(scope="module")
def qdrant_server():
    """The composed Qdrant service, healthy; stopped again afterwards
    unless it was already running before the tests (same convention as
    tests/integration/test_indexing_qdrant.py)."""
    require_docker()
    was_running = bool(
        subprocess.run(
            ["docker", "compose", "ps", "-q", "--status", "running", "qdrant"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    up = subprocess.run(
        ["docker", "compose", "up", "-d", "--wait", "qdrant"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert up.returncode == 0, up.stdout + up.stderr
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        yield client
        client.close()
    finally:
        if not was_running:
            subprocess.run(
                ["docker", "compose", "stop", "qdrant"], cwd=REPO_ROOT, capture_output=True
            )


@pytest.fixture
def collection(qdrant_server):
    """A unique collection name per test, deleted on the way out."""
    name = f"test-retrieval-{uuid.uuid4().hex[:12]}"
    yield name
    if qdrant_server.collection_exists(name):
        qdrant_server.delete_collection(name)
    meta = f"{name}__rag_meta"
    if qdrant_server.collection_exists(meta):
        qdrant_server.delete_collection(meta)


def test_include_filter_fail_closed_against_real_qdrant(qdrant_server, collection) -> None:
    """Finding #174: build the fixture index against the real pinned
    server, implant the full rogue source_type matrix (arrays included)
    past build_index, and drive retrieve() on every route with the
    reranker rigged to score rogue text top. Every rogue chunk either
    never appears in a result, or the run refuses with a typed
    RetrievalError naming it — never served evidence."""
    model = HashEmbeddingModel()
    chunks, records = fixture_corpus()
    build(qdrant_server, chunks, records, model=model, collection=collection)

    rogue_ids = set()
    for chunk_id, value, include_key in ROGUE_SOURCE_TYPES:
        rogue_ids.add(
            implant_rogue_chunk(
                qdrant_server,
                model,
                chunk_id=chunk_id,
                body=f"{chunk_id} {ROGUE_MARKER} chunk reading aloud in the market square.",
                source_type_value=value,
                include_source_type_key=include_key,
                collection_name=collection,
            )
        )
    reranker = TableReranker([(ROGUE_MARKER, 0.999), (VOICES_MARKER, 0.99)], default=0.1)

    scopes = (ScopeClass.IN_SCOPE, ScopeClass.ADVERSARIAL_IN_SCOPE, ScopeClass.VOICES)
    for scope in scopes:
        try:
            result = run_retrieve(
                qdrant_server,
                decision(VOICES_PULL_QUERY, scope=scope),
                model=model,
                reranker=reranker,
                cfg=config(),
                collection_name=collection,
            )
        except RetrievalError as error:
            assert any(rogue_id in str(error) for rogue_id in rogue_ids), (
                "the typed fail-closed refusal must name an offending chunk"
            )
            continue
        assert isinstance(result, RetrievedPassages)
        leaked = {p.chunk_id for p in result.passages} & rogue_ids
        assert leaked == set(), (
            f"rogue source_type chunks must fail closed against the real "
            f"server on every route; leaked on {scope}: {leaked}"
        )
        for passage in result.passages:
            assert isinstance(passage.payload.get("source_type"), str)
