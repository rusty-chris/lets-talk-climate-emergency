"""Gold chunk-id resolution against the LIVE ingest output (issue #20,
TDD step 9). Integration tier (auto-marked by this directory's conftest).

Runs over the artefacts `make ingest` writes (data/ingest/chunks.jsonl):
every gold chunk_id must resolve in the current corpus version, the
committed snapshot (evals/gold/ingest_chunk_ids.txt) must equal the live
id set exactly, and every severity source-passage quote must be a
verbatim (whitespace-normalised) substring of its chunk's body. Chunk ids
are content-hash based, so ANY corpus or chunker change that moves gold
text shows up here first — rerun on every corpus-version bump.

When the ingest artefacts are absent the test SKIPS LOUDLY with the
command to produce them (the ingest itself needs network + Docling and is
deliberately not run per-PR; the unit-tier snapshot test in
tests/unit/test_gold_sets.py still guards every PR). The skip is a
recorded cap, not a silent one — evals/gold/COVERAGE.md names it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_JSONL = REPO_ROOT / "data" / "ingest" / "chunks.jsonl"
QA_PATH = REPO_ROOT / "evals" / "gold" / "climate_qa.yaml"
SNAPSHOT_PATH = REPO_ROOT / "evals" / "gold" / "ingest_chunk_ids.txt"


def _normalised(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def live_chunks() -> dict[str, str]:
    """chunk_id -> whitespace-normalised body from the live ingest run."""
    if not CHUNKS_JSONL.is_file():
        pytest.skip(
            "data/ingest/chunks.jsonl not found — run `make ingest` (network + "
            "Docling) to produce the live ingest artefacts; the unit-tier "
            "snapshot test still guards gold chunk ids on every PR run"
        )
    chunks: dict[str, str] = {}
    for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            chunks[record["chunk_id"]] = _normalised(record["body"])
    return chunks


@pytest.fixture(scope="module")
def qa_items() -> list[dict]:
    return yaml.safe_load(QA_PATH.read_text(encoding="utf-8"))["items"]


def test_gold_chunk_ids_exist_in_index(live_chunks, qa_items):
    missing = []
    for item in qa_items:
        for chunk_id in item.get("gold_chunk_ids") or []:
            if chunk_id not in live_chunks:
                missing.append((item["id"], chunk_id))
    assert not missing, (
        "gold chunk ids no longer resolve against the live ingest (corpus or "
        f"chunker changed?): {missing} — re-select gold chunks, then refresh "
        "the snapshot with `python evals/scripts/gold_coverage.py --write-snapshot`"
    )


def test_snapshot_matches_live_ingest(live_chunks):
    snapshot = {
        line.strip()
        for line in SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert snapshot == set(live_chunks), (
        "evals/gold/ingest_chunk_ids.txt is stale relative to the live ingest "
        "— regenerate with `make ingest && python evals/scripts/gold_coverage.py "
        "--write-snapshot` and re-check every gold chunk id"
    )


def test_severity_source_quotes_are_verbatim(live_chunks, qa_items):
    for item in qa_items:
        severity = item.get("severity")
        if not severity:
            continue
        passage = severity["source_passage"]
        chunk_id = passage.get("chunk_id")
        if chunk_id is None:
            continue  # blocked item: pending_doc_id recorded instead
        assert chunk_id in live_chunks, item["id"]
        assert _normalised(passage["quote"]) in live_chunks[chunk_id], (
            f"{item['id']}: severity source quote is not a verbatim substring "
            f"of {chunk_id} — the annotation must derive from the passage it "
            "cites (DESIGN §6.1)"
        )
