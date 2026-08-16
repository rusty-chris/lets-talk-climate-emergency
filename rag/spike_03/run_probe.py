"""PROTOTYPE (issue #3 spike) — retrieve -> rerank -> native-citations probe runner.

The Phase-0 gate mechanism end to end (DESIGN §3.2-§3.4, §10):

1. Load the chunk corpus (``rag.spike_03.chunk_corpus`` output).
2. bge-m3 dense embeddings; naive cosine top-k retrieval (top-40).
   (Spike scope says "naive top-k"; production §3.2 is hybrid dense+sparse RRF.)
3. bge-reranker-v2-m3 cross-encoder rerank -> top-8.
4. Build ONE custom-content document block per top-8 chunk with
   ``citations: {enabled: true}`` (all-or-none, §3.4), no structured output on the
   generation call (§3.4). document_index 0..7 == retrieval rank.
5. Submit the 20 probe questions via the Batches API (50% off) on
   ``claude-haiku-4-5`` (cost plan M2/M3).
6. Resolve each answer's citations back to source chunks; grade vs the committed
   gold blocks; write ``reviews/spike-03-probe-findings.md`` and append the spend
   row to ``evals/spend-ledger.csv``.

Usage::

    # dry-run: chunk+retrieve+rerank+cost estimate, NO API calls
    uv run python -m rag.spike_03.run_probe --estimate
    # full probe (needs ANTHROPIC_API_KEY; submits the batch)
    uv run python -m rag.spike_03.run_probe --run
    # §3.4 constraint probes (deliberate 400s; not billed)
    uv run python -m rag.spike_03.run_probe --constraints
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rag.spike_03.probe import PROBE

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "spike"
CHUNKS_PATH = DATA_DIR / "spike03_chunks.jsonl"
FINDINGS_PATH = ROOT / "reviews" / "spike-03-probe-findings.md"
LEDGER_PATH = ROOT / "evals" / "spend-ledger.csv"
RETRIEVAL_JSON = DATA_DIR / "spike03_retrieval.json"
RESULTS_JSON = DATA_DIR / "spike03_results.json"

MODEL = "claude-haiku-4-5"
TOP_K = 40
TOP_N = 8
MAX_TOKENS = 1024

# Haiku 4.5 pricing per MTok (cost plan / claude-api skill): $1 in, $5 out.
PRICE_IN_PER_MTOK = 1.0
PRICE_OUT_PER_MTOK = 5.0
BATCH_MULT = 0.5  # Batches API = 50% off

SYSTEM_PROMPT = (
    "You answer questions about climate science using ONLY the provided source "
    "passages. Rules:\n"
    "1. Use only the provided passages; no outside knowledge.\n"
    "2. Cite every factual claim from the passages.\n"
    "3. Preserve calibrated language verbatim (e.g. 'very likely', 'high "
    "confidence') — never upgrade, downgrade, or drop a qualifier.\n"
    "4. Lead with the headline finding at the severity the source states it.\n"
    "5. If the passages do not answer the question, say so plainly.\n"
    "6. Use plain language for a general reader; define jargon on first use."
)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"MISSING {CHUNKS_PATH}. Run: uv run python -m rag.spike_03.chunk_corpus")
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# Retrieval + rerank (real local models; slow first run downloads weights)
# --------------------------------------------------------------------------- #
@dataclass
class Ranked:
    question_id: str
    question: str
    gold_chunk_ids: list[str]
    ranked_chunk_ids: list[str] = field(default_factory=list)  # top-N after rerank
    rerank_scores: list[float] = field(default_factory=list)


def retrieve_and_rerank(chunks: list[dict]) -> list[Ranked]:
    """bge-m3 dense embeddings (FlagEmbedding) + bge-reranker-v2-m3 cross-encoder.

    The reranker uses transformers directly (AutoModelForSequenceClassification),
    not FlagEmbedding's ``FlagReranker`` — the latter calls the removed
    ``tokenizer.prepare_for_model`` and 400s on transformers 5.x. The model,
    scoring (sigmoid(logit)) and semantics are identical to the model card.
    """
    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading bge-m3 (dense embeddings) on {device} ...", flush=True)
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    doc_emb = np.asarray(embedder.encode(texts, batch_size=8, max_length=1024)["dense_vecs"])
    q_texts = [q["question"] for q in PROBE]
    q_emb = np.asarray(embedder.encode(q_texts, batch_size=8, max_length=256)["dense_vecs"])
    # bge-m3 dense vecs are L2-normalised already; cosine == dot product.
    sims = q_emb @ doc_emb.T  # (Q, N)

    print(f"Loading bge-reranker-v2-m3 (cross-encoder rerank) on {device} ...", flush=True)
    rk_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    rk_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    rk_model = rk_model.to(device).eval()

    def rerank(query: str, cand_texts: list[str]) -> list[float]:
        pairs = [[query, t] for t in cand_texts]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), 16):
                batch = pairs[i : i + 16]
                enc = rk_tok(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                logits = rk_model(**enc).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores

    ranked: list[Ranked] = []
    for qi, q in enumerate(PROBE):
        topk_idx = np.argsort(-sims[qi])[:TOP_K]
        cand_texts = [texts[i] for i in topk_idx]
        scores = rerank(q["question"], cand_texts)
        order = np.argsort(-np.asarray(scores))[:TOP_N]
        top_ids = [ids[topk_idx[o]] for o in order]
        top_scores = [float(scores[o]) for o in order]
        ranked.append(
            Ranked(
                question_id=q["id"],
                question=q["question"],
                gold_chunk_ids=list(q["gold_chunk_ids"]),
                ranked_chunk_ids=top_ids,
                rerank_scores=top_scores,
            )
        )
    return ranked


# --------------------------------------------------------------------------- #
# Request construction (§3.3 / §3.4)
# --------------------------------------------------------------------------- #
def build_documents(ranked: Ranked, chunk_by_id: dict[str, dict]) -> list[dict]:
    """One custom-content document per top-N chunk; document_index == rank.

    citations enabled on EVERY document (§3.4 all-or-none). Chunk id is carried in
    `context` (passed to the model, not cited) so the response is traceable.
    """
    docs = []
    for cid in ranked.ranked_chunk_ids:
        c = chunk_by_id[cid]
        title = f"{c['title']}"[:200]
        docs.append(
            {
                "type": "document",
                "source": {"type": "content", "content": [{"type": "text", "text": c["text"]}]},
                "title": title,
                "context": f"chunk_id={cid}; section={'/'.join(c['section_path'])}",
                "citations": {"enabled": True},
            }
        )
    return docs


def build_params(ranked: Ranked, chunk_by_id: dict[str, dict]) -> dict:
    docs = build_documents(ranked, chunk_by_id)
    user_content = docs + [{"type": "text", "text": ranked.question}]
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }


# --------------------------------------------------------------------------- #
# Cost estimate (before any billed call)
# --------------------------------------------------------------------------- #
def _local_input_estimate(params: dict) -> int:
    """Conservative local token estimate from assembled request text.

    ~1 token / 3.5 chars for English (slight overestimate) + ~350 tokens for the
    citations system-prompt / per-document-chunking framing the API adds.
    """
    chars = len(params["system"])
    for block in params["messages"][0]["content"]:
        if block["type"] == "document":
            chars += len(block["source"]["content"][0]["text"]) + len(block.get("title", ""))
        elif block["type"] == "text":
            chars += len(block["text"])
    return int(chars / 3.5) + 350


def estimate_cost(ranked: list[Ranked], chunk_by_id: dict[str, dict]) -> dict:
    """Pre-compute expected input tokens before any billed call.

    Primary source is the free count_tokens endpoint (one attempt, no retry storm);
    on any failure (it was 500-ing during this spike) it falls back to a
    conservative local char-based estimate so the cap check never blocks on a
    server hiccup. Actual usage is recorded from response.usage after the batch.
    """
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    total_in = 0
    per_q = []
    method = "count_tokens"
    for r in ranked:
        params = build_params(r, chunk_by_id)
        try:
            ct = client.messages.count_tokens(
                model=MODEL, system=params["system"], messages=params["messages"]
            )
            n = ct.input_tokens
        except anthropic.APIError:
            method = "local-fallback (count_tokens unavailable)"
            n = _local_input_estimate(params)
        total_in += n
        per_q.append((r.question_id, n))
    est_out = MAX_TOKENS // 3 * len(ranked)  # generous: assume ~1/3 of cap per answer
    cost_in = total_in / 1e6 * PRICE_IN_PER_MTOK * BATCH_MULT
    cost_out = est_out / 1e6 * PRICE_OUT_PER_MTOK * BATCH_MULT
    return {
        "total_input_tokens": total_in,
        "assumed_output_tokens": est_out,
        "cost_in_usd": cost_in,
        "cost_out_usd": cost_out,
        "total_usd": cost_in + cost_out,
        "token_count_method": method,
        "per_question_input": per_q,
    }


# --------------------------------------------------------------------------- #
# Response parsing (shared by batch and live paths)
# --------------------------------------------------------------------------- #
def _parse_message(msg) -> tuple[list[dict], dict]:
    """Extract cited text blocks + a usage row from an Anthropic message."""
    blocks = []
    for b in msg.content:
        if b.type == "text":
            cites = []
            for cit in b.citations or []:
                cites.append(
                    {
                        "type": getattr(cit, "type", None),
                        "document_index": getattr(cit, "document_index", None),
                        "document_title": getattr(cit, "document_title", None),
                        "start_block_index": getattr(cit, "start_block_index", None),
                        "end_block_index": getattr(cit, "end_block_index", None),
                        "cited_text": (getattr(cit, "cited_text", "") or "")[:200],
                    }
                )
            blocks.append({"text": b.text, "citations": cites})
    u = msg.usage
    usage_row = {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    return blocks, usage_row


# --------------------------------------------------------------------------- #
# Batch submit + collect
# --------------------------------------------------------------------------- #
def submit_batch(ranked: list[Ranked], chunk_by_id: dict[str, dict]):
    import time

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    requests = [
        Request(
            custom_id=r.question_id,
            params=MessageCreateParamsNonStreaming(**build_params(r, chunk_by_id)),
        )
        for r in ranked
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id}; polling ...", flush=True)
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(
            f"  status={batch.processing_status} processing={batch.request_counts.processing}",
            flush=True,
        )
        time.sleep(20)
    results = {}
    usage_rows = []
    for res in client.messages.batches.results(batch.id):
        cid = res.custom_id
        if res.result.type != "succeeded":
            results[cid] = {"error": res.result.type}
            continue
        blocks, usage_row = _parse_message(res.result.message)
        usage_rows.append(usage_row)
        results[cid] = {"blocks": blocks}
    return batch.id, results, usage_rows


# --------------------------------------------------------------------------- #
# Live (non-batch) submit — fallback when the Batches API is unavailable
# --------------------------------------------------------------------------- #
def submit_live(ranked: list[Ranked], chunk_by_id: dict[str, dict]):
    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    results = {}
    usage_rows = []
    for r in ranked:
        params = build_params(r, chunk_by_id)
        msg = client.messages.create(**params)
        blocks, usage_row = _parse_message(msg)
        usage_rows.append(usage_row)
        results[r.question_id] = {"blocks": blocks}
        n_cit = sum(len(b["citations"]) for b in blocks)
        print(f"  {r.question_id}: {n_cit} citation(s)", flush=True)
    return "live-nonbatch", results, usage_rows


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def grade(ranked: list[Ranked], results: dict) -> list[dict]:
    graded = []
    for r in ranked:
        res = results.get(r.question_id, {})
        gold = set(r.gold_chunk_ids)
        gold_in_top8 = sorted(gold & set(r.ranked_chunk_ids))
        if "error" in res:
            graded.append(
                {
                    "id": r.question_id,
                    "question": r.question,
                    "gold": r.gold_chunk_ids,
                    "gold_in_top8": gold_in_top8,
                    "cited_chunk_ids": [],
                    "n_citations": 0,
                    "pass": False,
                    "note": f"API error: {res['error']}",
                }
            )
            continue
        cited_idx = []
        for blk in res["blocks"]:
            for cit in blk["citations"]:
                if cit["document_index"] is not None:
                    cited_idx.append(cit["document_index"])
        cited_chunk_ids = [
            r.ranked_chunk_ids[i] for i in cited_idx if 0 <= i < len(r.ranked_chunk_ids)
        ]
        cited_set = set(cited_chunk_ids)
        n_cit = len(cited_idx)
        correct = bool(cited_set & gold)
        passed = correct and n_cit > 0
        note = ""
        if not gold_in_top8:
            note = "gold chunk NOT retrieved into top-8 (retrieval miss)"
        elif n_cit == 0:
            note = "no citations produced"
        elif not correct:
            note = "cited only non-gold blocks"
        graded.append(
            {
                "id": r.question_id,
                "question": r.question,
                "gold": r.gold_chunk_ids,
                "gold_in_top8": gold_in_top8,
                "cited_chunk_ids": sorted(cited_set),
                "n_citations": n_cit,
                "pass": passed,
                "note": note,
            }
        )
    return graded


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
LEDGER_HEADER = [
    "date",
    "session_id",
    "activity",
    "issue",
    "model",
    "mode",
    "calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "cost_usd",
    "cumulative_usd",
    "notes",
]


def append_ledger(
    session_id: str, usage_rows: list[dict], notes: str, mode: str = "batch"
) -> float:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    mult = BATCH_MULT if mode == "batch" else 1.0
    calls = len(usage_rows)
    tin = sum(u["input_tokens"] for u in usage_rows)
    tout = sum(u["output_tokens"] for u in usage_rows)
    cread = sum(u["cache_read"] for u in usage_rows)
    ccreate = sum(u["cache_creation"] for u in usage_rows)
    cost = (tin / 1e6 * PRICE_IN_PER_MTOK + tout / 1e6 * PRICE_OUT_PER_MTOK) * mult
    # cumulative = prior maximum cumulative across existing data rows + this cost.
    prior = 0.0
    existed = LEDGER_PATH.exists()
    if existed:
        with LEDGER_PATH.open(encoding="utf-8") as fh:
            reader = csv.DictReader(row for row in fh if not row.startswith("#"))
            for row in reader:
                try:
                    prior = max(prior, float(row["cumulative_usd"]))
                except (KeyError, ValueError):
                    pass
    cumulative = round(prior + cost, 6)
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if not existed:
            fh.write("# evals/spend-ledger.csv — one row per API-touching session or batch\n")
            w.writerow(LEDGER_HEADER)
        w.writerow(
            [
                _dt.date.today().isoformat(),
                session_id,
                "spike-probe",
                "3",
                MODEL,
                mode,
                calls,
                tin,
                tout,
                cread,
                ccreate,
                round(cost, 6),
                cumulative,
                notes,
            ]
        )
    return cost, cumulative


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _prepare(use_cache: bool = True):
    chunks = load_chunks()
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    if use_cache and RETRIEVAL_JSON.exists():
        cached = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))
        ranked = [Ranked(**row) for row in cached]
        print(f"Loaded cached retrieval for {len(ranked)} questions ({RETRIEVAL_JSON.name}).")
    else:
        ranked = retrieve_and_rerank(chunks)
        RETRIEVAL_JSON.write_text(
            json.dumps([r.__dict__ for r in ranked], indent=2, ensure_ascii=False)
        )
    return chunks, chunk_by_id, ranked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--estimate", action="store_true", help="chunk+retrieve+rerank+cost, no billed calls"
    )
    ap.add_argument("--run", action="store_true", help="submit the 20-question probe and grade")
    ap.add_argument(
        "--live",
        action="store_true",
        help="submit as live non-batch calls (fallback when the Batches API is down)",
    )
    ap.add_argument(
        "--constraints", action="store_true", help="live §3.4 constraint probes (400s, not billed)"
    )
    ap.add_argument(
        "--refresh", action="store_true", help="recompute retrieval instead of using the cache"
    )
    args = ap.parse_args()

    if args.constraints:
        from rag.spike_03.constraints import run_constraints

        run_constraints()
        return 0

    mode = "live" if args.live else "batch"
    mult = 1.0 if mode == "live" else BATCH_MULT

    chunks, chunk_by_id, ranked = _prepare(use_cache=not args.refresh)
    n_gold_missed = sum(1 for r in ranked if not (set(r.gold_chunk_ids) & set(r.ranked_chunk_ids)))
    print(f"\nRetrieval: {len(ranked)} questions; gold-in-top8 misses: {n_gold_missed}")

    est = estimate_cost(ranked, chunk_by_id)
    # estimate_cost prices at BATCH_MULT; rescale to the chosen mode.
    total_est = est["total_usd"] / BATCH_MULT * mult
    print(f"\n=== COST ESTIMATE ({mode}, before submit) ===")
    print(f"  total input tokens : {est['total_input_tokens']} ({est['token_count_method']})")
    print(f"  assumed output toks: {est['assumed_output_tokens']}")
    print(f"  TOTAL (est) : ${total_est:.4f}   (cap $1.00)")
    (DATA_DIR / "spike03_estimate.json").write_text(json.dumps(est, indent=2))

    if total_est > 1.00:
        print("\nSTOP: estimate exceeds $1.00 cap. Not submitting.")
        return 2

    if not args.run:
        print("\n(--estimate only; not submitting. Re-run with --run [--live] to submit.)")
        return 0

    if mode == "live":
        print("\nSubmitting 20 LIVE (non-batch) calls (Batches API unavailable) ...", flush=True)
        run_id, results, usage_rows = submit_live(ranked, chunk_by_id)
    else:
        run_id, results, usage_rows = submit_batch(ranked, chunk_by_id)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    graded = grade(ranked, results)
    n_pass = sum(1 for g in graded if g["pass"])
    notes = f"{mode}={run_id}; gate {n_pass}/20; retrieval_misses={n_gold_missed}"
    cost, cumulative = append_ledger(
        f"spike03-{_dt.date.today().isoformat()}", usage_rows, notes, mode=mode
    )
    print(f"\n=== GATE: {n_pass}/20 questions' citations resolve to a gold source block ===")
    print(f"actual spend this run ({mode}): ${cost:.4f}; cumulative ledger: ${cumulative:.4f}")
    (DATA_DIR / "spike03_graded.json").write_text(json.dumps(graded, indent=2, ensure_ascii=False))
    print(f"graded detail -> {DATA_DIR / 'spike03_graded.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
