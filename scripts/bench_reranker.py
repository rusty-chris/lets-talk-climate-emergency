"""OFFLINE ($0, no LLM calls) reranker benchmark for the climate-chat RAG.

Compares cross-encoder rerankers to replace the slow BAAI/bge-reranker-v2-m3
on CPU. Measures BOTH retrieval quality (recall@8 / MRR / nDCG@8 over the
gold answer items) and per-model rerank latency on the fixed realistic
40-candidate perf set.

Nothing here touches production code paths destructively: it reuses the REAL
bge-m3 embedder and rag.indexing / rag.retrieval building blocks, indexes the
real corpus ONCE into an in-memory qdrant, and swaps only the reranker.

Prerequisite: a landed corpus at ``data/ingest/chunks.jsonl`` (the documented
ingest-output convention — produce it with
``.venv/bin/python scripts/ingest_corpus.py --out-dir data/ingest``). Override
the location with ``BENCH_CHUNKS=/path/to/chunks.jsonl``.

Run:  .venv/bin/python scripts/bench_reranker.py
      BENCH_ONLY=ms-marco-MiniLM-L-6-v2 BENCH_SUBSET=10 \
        .venv/bin/python scripts/bench_reranker.py   # quick single-model pass
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Real building blocks (only the RERANKER varies across runs).
from evals.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from ingestion.pipeline import ChunkRecord, DocumentIngestRecord  # noqa: E402
from rag.indexing import Bgem3EmbeddingModel, build_index, hybrid_query  # noqa: E402
from rag.retrieval import (  # noqa: E402
    EVIDENCE_SOURCE_TYPES,
    GENERATION_TOP_K,
    RERANK_CANDIDATE_K,
    reranker_window_bounds,
)

CORPUS_VERSION = "bench"
COLLECTION = "bench_corpus"
# The landed ingest output (scripts/ingest_corpus.py --out-dir data/ingest);
# override with BENCH_CHUNKS for an ad-hoc corpus location.
CHUNKS_JSONL = Path(
    os.environ.get("BENCH_CHUNKS", str(REPO_ROOT / "data" / "ingest" / "chunks.jsonl"))
)
GOLD = REPO_ROOT / "evals/gold/climate_qa.yaml"


# ---------------------------------------------------------------------------
# Candidate reranker: our OWN loader (plain transformers, download allowed),
# mirroring rag.retrieval.CrossEncoderReranker semantics exactly — windowed full-coverage
# scoring, max over windows, sigmoid(logit). English-only MiniLM cross
# encoders are AutoModelForSequenceClassification-shaped (single logit) too.
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    _MAX_PAIR_TOKENS = 512

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision
        )
        self._model.eval()
        self._model_id = model_id
        # Some cross-encoders (e.g. certain MiniLM configs) cap positions <512.
        model_max = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(model_max, int) and 0 < model_max < self._MAX_PAIR_TOKENS:
            self._MAX_PAIR_TOKENS = model_max

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch

        passages = list(passages)
        if not passages:
            return []
        special = self._tokenizer.num_special_tokens_to_add(pair=True)
        q_tokens = len(self._tokenizer(query, add_special_tokens=False)["input_ids"])
        window_budget = self._MAX_PAIR_TOKENS - q_tokens - special
        if window_budget <= 0:
            raise RuntimeError(f"query consumes whole pair cap for {self._model_id}")

        pair_texts: list[list[str]] = []
        owner: list[int] = []
        for pi, passage in enumerate(passages):
            enc = self._tokenizer(passage, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            for start, end in reranker_window_bounds(len(offsets), window_budget):
                text = "" if start == end else passage[offsets[start][0] : offsets[end - 1][1]]
                pair_texts.append([query, text])
                owner.append(pi)

        inputs = self._tokenizer(
            pair_texts,
            padding=True,
            truncation=True,
            max_length=self._MAX_PAIR_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.view(-1).float()
            window_scores = torch.sigmoid(logits)
        best = [0.0] * len(passages)
        for pi, s in zip(owner, window_scores, strict=True):
            best[pi] = max(best[pi], float(s))
        return best


# ---------------------------------------------------------------------------
# Corpus loading + index build (ONCE, reused across all rerankers).
# ---------------------------------------------------------------------------
def load_chunks(path: Path) -> tuple[list[ChunkRecord], dict[str, DocumentIngestRecord]]:
    chunks: list[ChunkRecord] = []
    doc_backend: dict[str, str] = {}
    for line in path.open():
        d = json.loads(line)
        chunks.append(
            ChunkRecord(
                chunk_id=d["chunk_id"],
                doc_id=d["doc_id"],
                section_path=tuple(d["section_path"]),
                context_header=d["context_header"],
                body=d["body"],
                token_count=d["token_count"],
                confidence_markers=tuple(d["confidence_markers"]),
                consensus_position=d["consensus_position"],
                source_type=d["source_type"],
                citation_metadata=d["citation_metadata"],
                block_types=tuple(d.get("block_types", ())),
                oversized_atomic=d.get("oversized_atomic", False),
                parse_backend=d.get("parse_backend", "unknown"),
                needs_hand_review=d.get("needs_hand_review", False),
            )
        )
        doc_backend[d["doc_id"]] = d.get("parse_backend", "unknown")
    records = {
        doc_id: DocumentIngestRecord(doc_id=doc_id, parse_backend=backend, needs_hand_review=False)
        for doc_id, backend in doc_backend.items()
    }
    return chunks, records


def load_gold_items() -> list[dict[str, Any]]:
    data = yaml.safe_load(GOLD.open())
    items = data["items"] if isinstance(data, dict) else data
    gold = []
    for it in items:
        if (
            it.get("category") in ("single_passage", "multi_passage")
            and it.get("expected_behaviour") == "answer"
            and it.get("gold_chunk_ids")
        ):
            gold.append(it)
    return gold


# ---------------------------------------------------------------------------
# Latency: reuse the perf-harness's realistic 40-candidate set construction.
# ---------------------------------------------------------------------------
def realistic_candidate_set() -> list[str]:
    from tests.integration.test_reranker_smoke import _realistic_candidate_set

    return _realistic_candidate_set()


LATENCY_QUERY = "How much have surface temperatures risen across the Aurelian Basin?"


def measure_latency(reranker: Any) -> tuple[float, float]:
    passages = realistic_candidate_set()
    assert len(passages) == RERANK_CANDIDATE_K
    reranker.score(LATENCY_QUERY, passages[:2])  # warm-up: exclude model init
    started = time.perf_counter()
    reranker.score(LATENCY_QUERY, passages)
    elapsed = time.perf_counter() - started
    return elapsed, elapsed / RERANK_CANDIDATE_K


# ---------------------------------------------------------------------------
# Quality: retrieve 40 -> rerank -> top-8 -> recall@8 / MRR / nDCG@8.
# ---------------------------------------------------------------------------
def eval_quality(reranker: Any, client: Any, embedder: Any, gold: list[dict[str, Any]]) -> dict:
    recalls, mrrs, ndcgs = [], [], []
    per_item = []
    for it in gold:
        query = it["question"]
        gold_ids = list(it["gold_chunk_ids"])
        semantics = it.get("recall_semantics", "any_gold")  # single_passage: 1 gold id
        candidates = hybrid_query(
            client,
            COLLECTION,
            query,
            embedding_model=embedder,
            expected_corpus_version=CORPUS_VERSION,
            top_k=RERANK_CANDIDATE_K,
            include_source_types=EVIDENCE_SOURCE_TYPES,
        )
        bodies = [c.payload["body"] for c in candidates]
        scores = reranker.score(query, bodies)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        top_ids = [c.chunk_id for c, _ in ranked[:GENERATION_TOP_K]]
        r = recall_at_k(top_ids, gold_ids, k=GENERATION_TOP_K, semantics=semantics)
        m = mrr(top_ids, gold_ids)
        n = ndcg_at_k(top_ids, gold_ids, k=GENERATION_TOP_K)
        recalls.append(1.0 if r else 0.0)
        mrrs.append(m)
        ndcgs.append(n)
        per_item.append(
            {
                "id": it["id"],
                "recall": r,
                "mrr": m,
                "ndcg": n,
                "gold": gold_ids,
                "top8": top_ids,
                "semantics": semantics,
            }
        )
    n_items = len(gold)
    return {
        "n": n_items,
        "recall@8": sum(recalls) / n_items,
        "mrr": sum(mrrs) / n_items,
        "ndcg@8": sum(ndcgs) / n_items,
        "per_item": per_item,
    }


def main() -> int:
    only = os.environ.get("BENCH_ONLY")  # comma-separated model keys, optional
    subset = os.environ.get("BENCH_SUBSET")  # int: run first N gold items only

    if not CHUNKS_JSONL.is_file():
        print(
            f"[bench] no corpus at {CHUNKS_JSONL} — run "
            "`scripts/ingest_corpus.py --out-dir data/ingest` first, or point "
            "BENCH_CHUNKS at an existing chunks.jsonl.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    print(f"[bench] loading chunks from {CHUNKS_JSONL}", flush=True)
    chunks, records = load_chunks(CHUNKS_JSONL)
    print(f"[bench] {len(chunks)} chunks / {len(records)} documents", flush=True)

    gold = load_gold_items()
    if subset:
        gold = gold[: int(subset)]
    print(f"[bench] {len(gold)} gold answer items (single_passage + multi_passage)", flush=True)

    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    print("[bench] loading real bge-m3 embedder + building index (ONCE)…", flush=True)
    t0 = time.perf_counter()
    embedder = Bgem3EmbeddingModel()
    report = build_index(
        client, COLLECTION, chunks, records, embedding_model=embedder, corpus_version=CORPUS_VERSION
    )
    print(
        f"[bench] indexed {report.indexed_chunk_count} chunks in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    candidates = {
        # Baseline = the PRE-swap production reranker, loaded explicitly by
        # id/revision via this script's own generic loader (the production
        # class now pins MiniLM per ADR-006, so it can no longer stand in for
        # the old 560M model here).
        "bge-reranker-v2-m3": (
            "baseline",
            lambda: CrossEncoderReranker(
                "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
            ),
            "~560M",
        ),
        "bge-reranker-base": (
            "candidate",
            lambda: CrossEncoderReranker("BAAI/bge-reranker-base"),
            "~278M",
        ),
        "ms-marco-MiniLM-L-6-v2": (
            "candidate",
            lambda: CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "~22M",
        ),
        "ms-marco-MiniLM-L-12-v2": (
            "candidate",
            lambda: CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2"),
            "~33M",
        ),
    }
    if only:
        keys = [k.strip() for k in only.split(",")]
        candidates = {k: v for k, v in candidates.items() if k in keys}

    results = {}
    for key, (kind, factory, params) in candidates.items():
        print(f"\n[bench] === {key} ({kind}, {params}) ===", flush=True)
        try:
            reranker = factory()
        except Exception as e:  # noqa: BLE001
            print(f"[bench] FAILED to load {key}: {type(e).__name__}: {e}", flush=True)
            results[key] = {"error": f"load: {type(e).__name__}: {e}"}
            continue
        try:
            lat40, latper = measure_latency(reranker)
            print(
                f"[bench] latency: {lat40:.3f}s / 40 cands  ({latper * 1000:.1f} ms/cand)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bench] latency FAILED for {key}: {type(e).__name__}: {e}", flush=True)
            lat40 = latper = float("nan")
        try:
            tq = time.perf_counter()
            q = eval_quality(reranker, client, embedder, gold)
            print(
                f"[bench] quality: recall@8={q['recall@8']:.3f} mrr={q['mrr']:.3f} "
                f"ndcg@8={q['ndcg@8']:.3f}  ({time.perf_counter() - tq:.1f}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bench] quality FAILED for {key}: {type(e).__name__}: {e}", flush=True)
            q = {"error": f"{type(e).__name__}: {e}"}
        results[key] = {
            "kind": kind,
            "params": params,
            "latency_40s": lat40,
            "latency_per_cand": latper,
            "quality": q,
        }
        # free memory between models
        del reranker

    out = REPO_ROOT / "scratchpad_bench_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[bench] wrote {out}", flush=True)

    # Summary table
    print("\n" + "=" * 100)
    print(
        f"{'model':<26}{'params':>8}{'40-cand s':>12}{'ms/cand':>10}"
        f"{'recall@8':>10}{'MRR':>8}{'nDCG@8':>9}"
    )
    print("-" * 100)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<26}  ERROR: {r['error']}")
            continue
        q = r["quality"]
        if "error" in q:
            print(
                f"{key:<26}{r['params']:>8}{r['latency_40s']:>12.3f}"
                f"{r['latency_per_cand'] * 1000:>10.1f}   quality ERROR: {q['error']}"
            )
            continue
        print(
            f"{key:<26}{r['params']:>8}{r['latency_40s']:>12.3f}"
            f"{r['latency_per_cand'] * 1000:>10.1f}{q['recall@8']:>10.3f}"
            f"{q['mrr']:>8.3f}{q['ndcg@8']:>9.3f}"
        )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
