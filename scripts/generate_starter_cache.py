"""Resumable release step: generate the REAL starter-answer cache
(service/DEPLOYMENT.md §3) through the live pipeline against the release
index.

Promoted into the repo after the 2026-09-13 deploy incident
(data/DEPLOY-CHECKPOINT.md): the previous server-only script restarted from
question 0 every run, wrote its entries non-atomically, and rebuilt its
cross-run spend from a glob of per-run tallies plus a hardcoded constant, so
six non-idempotent restarts burned $0.38 of a $0.50 hard cap while writing
only 3 of 13 entries. This version is idempotent and resumable:

  * :class:`SpendMeter` carries prior spend across runs through ONE ledger
    file (``carried_spend.json``) beside the cache, so the $0.50 cap bounds
    the WHOLE deploy step, not one process. It refuses at a QUESTION boundary
    (before spending on a new question) as well as fail-closed before each
    call — so a cap hit stops mid-question, never mid-file.
  * :func:`generate_starter_cache` processes ``STARTER_QUESTIONS`` in
    deterministic order, VALIDATES each already-written entry and skips only
    the complete ones (regenerating anything malformed rather than trusting
    it), and writes each entry ATOMICALLY on completion. A killed run loses
    at most the single in-flight question; the aggregate cache file is only
    written once every entry exists, so a partial run never emits a
    half-written cache.

The pure orchestration/spend/validation core (this module's top half) imports
nothing heavy and is unit-tested with fakes at ZERO API cost; the live
pipeline wiring in :func:`main` performs the one funded generation run.

Live failure policy (unchanged from the certified pipeline): a pre-filter
refusal HALTS the build; a generation DECLINE is retried and, if persistent,
ships the model's HONEST decline prose (marker stripped) with whatever it
cited — a persistent decline is the pipeline's real, gate-accepted behaviour
(release-run-5 certified it inside the <5% false-refusal budget). A final
attempt with ZERO citations still halts. Declined entries are flagged in the
build report for the deploy report.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from evals.pricing import estimate_cost_usd
from service.atomic_write import atomic_write_text
from service.starter_cache import (
    STARTER_CACHE_FILENAME,
    STARTER_QUESTIONS,
    load_starter_cache,
)

#: The cross-run carried-spend ledger, written beside the cache/entries.
CARRIED_SPEND_FILENAME = "carried_spend.json"

HARD_CAP_USD = 0.50
#: Stop BEFORE a call/question once spend passes this line (worst-case single
#: call is a cached-prompt Haiku generation, well under $0.02).
PRE_CALL_LINE_USD = 0.45

#: Env overrides for the deploy-step caps. The incident's $0.50/$0.45 lines
#: are the defaults; the owner can approve a higher whole-deploy-step cap
#: (e.g. to finish a resumed cache that already carries prior spend) and the
#: deploy finisher raises it through these WITHOUT patching code on the box.
HARD_CAP_ENV = "STARTER_CACHE_HARD_CAP_USD"
PRE_CALL_LINE_ENV = "STARTER_CACHE_PRE_CALL_LINE_USD"


def resolve_caps(environ: Mapping | None = None) -> tuple[float, float]:
    """Resolve ``(hard_cap_usd, pre_call_line_usd)`` from the environment.

    Defaults are the module constants (the incident's $0.50/$0.45 lines). An
    owner-approved raise sets :data:`HARD_CAP_ENV` / :data:`PRE_CALL_LINE_ENV`
    so the whole deploy step — carried prior spend included — is bounded by
    the higher line without a code change on the server. The pre-call line
    must sit strictly below the hard cap (its whole job is to stop one
    worst-case call short of it), so an inverted pair is a loud error rather
    than a cap that never guards.
    """
    environ = os.environ if environ is None else environ
    hard = float(environ.get(HARD_CAP_ENV, HARD_CAP_USD))
    pre = float(environ.get(PRE_CALL_LINE_ENV, PRE_CALL_LINE_USD))
    if pre >= hard:
        raise ValueError(
            f"{PRE_CALL_LINE_ENV}=${pre} must be strictly below "
            f"{HARD_CAP_ENV}=${hard} — the pre-call line stops one call short "
            "of the hard cap; an inverted pair would never guard"
        )
    return hard, pre


def _normalise_question(question: str) -> str:
    """Collapse surrounding/interior whitespace (service.starter_cache parity)."""
    return " ".join(str(question).split())


class SpendCapReached(RuntimeError):
    """A call or question would cross the pre-call line or the hard cap.

    Carries the fail-closed intent of the deploy's $0.50 cap: the meter has
    already persisted the spend before raising, so the next run inherits it.
    """


class SpendMeter:
    """Per-call fail-closed spend meter with cross-run carry.

    ``spent`` is seeded from the ledger file's cumulative ``total_usd`` and
    keeps counting, so the pre-call line and hard cap bound the WHOLE deploy
    step across every process that shares the ledger — not one run.
    """

    def __init__(
        self,
        ledger_path,
        *,
        hard_cap_usd: float = HARD_CAP_USD,
        pre_call_line_usd: float = PRE_CALL_LINE_USD,
        cost_fn=estimate_cost_usd,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.hard_cap_usd = hard_cap_usd
        self.pre_call_line_usd = pre_call_line_usd
        self.cost_fn = cost_fn
        self.rows: list[dict] = []
        prior = 0.0
        if self.ledger_path.is_file():
            try:
                prior = float(
                    json.loads(self.ledger_path.read_text(encoding="utf-8")).get("total_usd", 0.0)
                    or 0.0
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                prior = 0.0
        self.prior = prior
        self.spent = prior

    def check(self, label: str) -> None:
        """Refuse BEFORE spending when already at/over the pre-call line."""
        if self.spent >= self.pre_call_line_usd:
            self.persist()
            raise SpendCapReached(
                f"HARD-CAP GUARD before {label}: spent ${self.spent:.4f} >= "
                f"${self.pre_call_line_usd} pre-call line (cap ${self.hard_cap_usd}) — halting"
            )

    def record(self, segment: str, model: str, usage: Mapping | None) -> None:
        """Price one billed call, add it, persist, then fail closed on the cap."""
        usage = dict(usage or {})
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(
            usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0)) or 0
        )
        cache_write = int(
            usage.get("cache_creation_input_tokens", usage.get("cache_creation_tokens", 0)) or 0
        )
        cost = self.cost_fn(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_write,
        )
        self.spent += cost
        self.rows.append(
            {
                "segment": segment,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read,
                "cache_creation_tokens": cache_write,
                "cost_usd": cost,
            }
        )
        self.persist()
        if self.spent >= self.hard_cap_usd:
            raise SpendCapReached(
                f"HARD CAP BREACHED after {segment}: ${self.spent:.4f} >= ${self.hard_cap_usd}"
            )

    def persist(self) -> None:
        """Atomically rewrite the carried-spend ledger (crash-survivable)."""
        by_segment: dict[str, dict] = {}
        for row in self.rows:
            agg = by_segment.setdefault(
                row["segment"],
                {
                    "model": row["model"],
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            agg["calls"] += 1
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_creation_tokens",
                "cost_usd",
            ):
                agg[key] += row[key]
        atomic_write_text(
            self.ledger_path,
            json.dumps(
                {
                    "total_usd": self.spent,
                    "run_usd": self.spent - self.prior,
                    "prior_usd": self.prior,
                    "by_segment": by_segment,
                    "rows": self.rows,
                },
                indent=2,
            )
            + "\n",
        )


def entry_is_valid(saved_entry: Mapping, question: str) -> tuple[bool, str]:
    """Whether an already-written entry is complete and matches ``question``.

    Mirrors ``service.starter_cache.load_starter_cache``'s per-entry checks so
    a resumed entry is trusted only if the service would accept it: matching
    question, and non-empty ``answer_text``, ``citations`` and ``footer``. A
    malformed leftover is regenerated, never shipped.
    """
    if not isinstance(saved_entry, Mapping):
        return False, "entry is not a JSON object"
    stored_q = saved_entry.get("question")
    if not isinstance(stored_q, str) or _normalise_question(stored_q) != _normalise_question(
        question
    ):
        return False, f"question mismatch (stored {stored_q!r}, expected {question!r})"
    answer_text = saved_entry.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return False, "answer_text is missing/empty"
    if not saved_entry.get("citations"):
        return False, "citations are missing/empty"
    footer = saved_entry.get("footer")
    if not isinstance(footer, str) or not footer.strip():
        return False, "footer is missing/empty"
    return True, ""


def generate_starter_cache(
    questions,
    *,
    entries_dir: Path,
    out_cache_dir: Path,
    meter: SpendMeter,
    answer_fn,
    generated_on: str,
) -> dict:
    """Resumably generate every starter entry, then the aggregate cache.

    For each question in deterministic order: skip a valid already-written
    entry ($0); regenerate a malformed one; otherwise refuse at the cap
    BEFORE spending (``meter.check``) and generate through ``answer_fn(index,
    question, meter=meter)``, writing the entry atomically on completion. The
    aggregate ``starter_answers.json`` is written — and self-validated exactly
    as the service will at boot — only once all entries exist, so a cap halt
    or crash never leaves a partial aggregate behind.

    ``answer_fn`` returns ``{"entry": <entry dict>, "report": <report dict>}``;
    it does the live retrieval/generation/validation and its own per-call
    metering. Raises :class:`SpendCapReached` on a cap halt (completed entries
    stay on disk for the next run).
    """
    entries_dir = Path(entries_dir)
    out_cache_dir = Path(out_cache_dir)
    entries_dir.mkdir(parents=True, exist_ok=True)

    collected: list[dict] = []
    resumed = 0
    written = 0
    for index, question in enumerate(questions):
        entry_path = entries_dir / f"{index:02d}.json"
        if entry_path.is_file():
            saved: object = None
            try:
                saved = json.loads(entry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                saved = None
            if isinstance(saved, Mapping):
                saved_entry = saved.get("entry", saved)
                ok, reason = entry_is_valid(saved_entry, question)
                if ok:
                    collected.append(dict(saved_entry))
                    resumed += 1
                    print(f"== {question} — RESUMED from {entry_path.name} ($0)", flush=True)
                    continue
                print(
                    f"== {question} — REGENERATING {entry_path.name} "
                    f"(existing entry invalid: {reason})",
                    flush=True,
                )
            else:
                print(
                    f"== {question} — REGENERATING {entry_path.name} (unreadable)",
                    flush=True,
                )

        # Refuse at the QUESTION boundary before spending anything on it, so a
        # cap halt stops between questions, never mid-file.
        meter.check(f"question {index}: {question!r}")
        print(f"== {question}", flush=True)
        result = answer_fn(index, question, meter=meter)
        entry = dict(result["entry"])
        report = dict(result.get("report", {}))
        atomic_write_text(
            entry_path,
            json.dumps({"entry": entry, "report": report}, indent=1, ensure_ascii=False),
        )
        collected.append(entry)
        written += 1

    out_cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_cache_dir / STARTER_CACHE_FILENAME,
        json.dumps(
            {"generated_on": generated_on, "entries": collected}, indent=2, ensure_ascii=False
        )
        + "\n",
    )
    # Self-validate exactly as the service will at boot (raises on any gap).
    cache = load_starter_cache(out_cache_dir)
    print(
        f"starter cache written + validated: {out_cache_dir} "
        f"({len(cache.entries)} entries, {resumed} resumed, {written} generated, "
        f"generated_on={generated_on}); total spend ${meter.spent:.4f}",
        flush=True,
    )
    return {
        "generated_on": generated_on,
        "entries": collected,
        "resumed": resumed,
        "written": written,
        "total_usd": meter.spent,
    }


# ---------------------------------------------------------------------------
# Live pipeline wiring (the one funded generation run). Everything below is
# import-heavy (qdrant + model weights + the rag stack) and is deliberately
# imported INSIDE main(), so importing this module for the unit tests never
# loads torch/qdrant and spends nothing.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("CLIMATE_CHAT_REPO_ROOT", "/opt/climate-chat"))
RUN = Path(os.environ.get("STARTER_CACHE_RUN_DIR", "/root/release-build"))
OUT_CACHE_DIR = Path(
    os.environ.get("STARTER_CACHE_OUT_DIR", str(REPO_ROOT / "release-starter-cache"))
)
QDRANT_URL = os.environ.get("CLIMATE_CHAT_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("CLIMATE_CHAT_COLLECTION", "climate_chunks")
CORPUS_VERSION = os.environ.get("CLIMATE_CHAT_CORPUS_VERSION", "v1.1.0-launch-2026-09-13")
CORPUS_VINTAGE = os.environ.get("CLIMATE_CHAT_CORPUS_VINTAGE", "2026-09-13")
THRESHOLD_ARTIFACT = Path(
    os.environ.get("CLIMATE_CHAT_THRESHOLD_ARTIFACT", str(RUN / "threshold.json"))
)
DATASET_MANIFEST = REPO_ROOT / "datasets" / "manifest.yaml"
CHART_PACK_DIR = REPO_ROOT / "data" / "datasets"


def _seed_ledger_from_legacy_tally(ledger_path: Path) -> None:
    """Bridge the pre-promotion meter: if the new carried-spend ledger does
    not yet exist but the incident's ``usage_tally.json`` does, seed the
    ledger with its cumulative ``total_usd`` so the resume inherits the
    already-spent $0.38 rather than restarting the cap at $0."""
    if ledger_path.is_file():
        return
    legacy = RUN / "usage_tally.json"
    if not legacy.is_file():
        return
    try:
        total = float(json.loads(legacy.read_text(encoding="utf-8")).get("total_usd", 0.0) or 0.0)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return
    if total > 0.0:
        atomic_write_text(
            ledger_path,
            json.dumps({"total_usd": total, "seeded_from": str(legacy)}, indent=2) + "\n",
        )
        print(f"meter: seeded carried spend ${total:.4f} from {legacy}", flush=True)


def main() -> int:
    import dataclasses
    from datetime import UTC, datetime

    from qdrant_client import QdrantClient

    from charts.pack import load_chart_pack_frames
    from rag.citation_validator import ValidatorConfig, validate_exchange
    from rag.generation import (
        CITATION_EVENT,
        ERROR_EVENT,
        FOOTER_EVENT,
        GENERATION_MODEL_DEFAULT,
        TEXT_EVENT,
        USAGE_EVENT,
        GenerationConfig,
        classify_generation_decline,
        stream_grounded_answer,
    )
    from rag.provider import AnthropicAdapter
    from rag.query import Route, process_query
    from rag.retrieval import (
        BgeRerankerV2M3,
        RetrievalConfig,
        RetrievedPassages,
        load_prefilter_artifact,
        retrieve,
    )
    from service.app import _grounded_answer_from_sse

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not in the environment — refusing")

    hard_cap_usd, pre_call_line_usd = resolve_caps()
    ledger_path = RUN / CARRIED_SPEND_FILENAME
    _seed_ledger_from_legacy_tally(ledger_path)
    meter = SpendMeter(ledger_path, hard_cap_usd=hard_cap_usd, pre_call_line_usd=pre_call_line_usd)
    print(
        f"meter: prior deploy-step spend ${meter.prior:.4f} "
        f"(cap ${hard_cap_usd}, pre-call line ${pre_call_line_usd})",
        flush=True,
    )

    adapter = AnthropicAdapter(api_key=os.environ["ANTHROPIC_API_KEY"])
    client = QdrantClient(url=QDRANT_URL)
    calibration = load_prefilter_artifact(THRESHOLD_ARTIFACT)
    print(
        f"pre-filter: enabled={calibration.enabled} threshold={calibration.threshold} "
        f"({calibration.reason or 'release-run-5 calibrated floor, reused'})",
        flush=True,
    )
    embedder = None  # built lazily below (multi-GB weights)
    reranker = None
    retrieval_config = RetrievalConfig(refusal_threshold=calibration.threshold, corpus_coverage=())

    def run_retrieval(decision):
        nonlocal embedder, reranker
        if embedder is None:
            from rag.indexing import Bgem3EmbeddingModel

            embedder = Bgem3EmbeddingModel()
            reranker = BgeRerankerV2M3()
        return retrieve(
            client,
            COLLECTION,
            decision,
            embedding_model=embedder,
            reranker=reranker,
            config=retrieval_config,
            expected_corpus_version=CORPUS_VERSION,
        )

    # ---- Sanity ($0): the render inputs load — the #214/#216 boot gate will
    # demand them, and a live chart exchange renders through them.
    raw_manifest, frames = load_chart_pack_frames(DATASET_MANIFEST, CHART_PACK_DIR)
    assert frames, "the landed chart pack must parse"
    print(f"chart pack frames loaded: {sorted(frames)}", flush=True)

    generation_config = GenerationConfig()  # Haiku default, no best mode
    generated_on = datetime.now(UTC).date().isoformat()

    def grounded_answer(decision, question: str, meter, attempt: int = 1, best: dict | None = None):
        """retrieval -> streamed generation -> validator; returns the entry
        fields. Halts on pre-filter refusal; retries on decline/error."""
        best = best if best is not None else {}
        retrieved = run_retrieval(decision)
        if not isinstance(retrieved, RetrievedPassages):
            raise SystemExit(
                f"HALT: pre-filter refused starter question {question!r}: {retrieved!r}"
            )
        meter.check(f"generation for {question!r}")
        transcript = []
        text_parts: list[str] = []
        citations: list[dict] = []
        footer = ""
        saw_error = False
        # Service parity: generation gets the visitor's RAW question; only
        # retrieval used the rewrite.
        for event in stream_grounded_answer(
            adapter, retrieved, question, config=generation_config, corpus_vintage=CORPUS_VINTAGE
        ):
            transcript.append(event)
            name = event["event"]
            if name == TEXT_EVENT:
                text_parts.append(event["data"].get("text", ""))
            elif name == CITATION_EVENT:
                data = dict(event["data"])
                data.setdefault("attribution_text", data.get("document_title", ""))
                citations.append(data)
            elif name == FOOTER_EVENT:
                footer = event["data"].get("text", "")
            elif name == USAGE_EVENT:
                meter.record("generation", GENERATION_MODEL_DEFAULT, event["data"])
            elif name == ERROR_EVENT:
                saw_error = True
        answer_text = "".join(text_parts)
        # Debug evidence per attempt (why a decline happened).
        debug_dir = RUN / "debug"
        debug_dir.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() else "-" for c in question)[:48]
        (debug_dir / f"{safe}-attempt{attempt}.json").write_text(
            json.dumps(
                {
                    "question": question,
                    "retrieval_query": decision.retrieval_query,
                    "answer_text": answer_text,
                    "citation_count": len(citations),
                    "retrieved": [
                        {
                            "chunk_id": p.chunk_id,
                            "score": getattr(p, "rerank_score", None) or getattr(p, "score", None),
                            "clears_threshold": getattr(p, "clears_threshold", None),
                            "head": " ".join(str(p.payload.get("body", "")).split()[:20]),
                        }
                        for p in retrieved.passages
                    ],
                },
                indent=1,
                ensure_ascii=False,
            )
        )
        decline = classify_generation_decline(answer_text)
        declined = decline.is_decline
        clean = citations and not saw_error and not declined and answer_text.strip()
        usable = citations and not saw_error and answer_text.strip()
        if usable and (best.get("rank", -1) < (2 if clean else 1)):
            best.update(
                rank=2 if clean else 1,
                answer_text=answer_text,
                citations=list(citations),
                footer=footer,
                declined=declined,
                decline_display=decline.display_text if declined else "",
                transcript=list(transcript),
                retrieved=retrieved,
            )
        retry = (best.get("rank", -1) < 2 and attempt < 3) or (
            best.get("rank", -1) < 1 and attempt < 5
        )
        if not clean and retry:
            print(
                f"  retrying {question!r} (attempt {attempt}): error={saw_error} "
                f"declined={declined} citations={len(citations)}",
                flush=True,
            )
            return grounded_answer(decision, question, meter, attempt=attempt + 1, best=best)
        if best.get("rank", -1) < 1:
            raise SystemExit(
                f"HALT: starter question {question!r} produced no usable cited output after "
                f"{attempt} attempts (error={saw_error}, declined={declined}, "
                f"citations={len(citations)})"
            )
        answer_text = best["answer_text"]
        citations = best["citations"]
        footer = best["footer"]
        declined = best["declined"]
        transcript = best["transcript"]
        retrieved = best["retrieved"]
        if declined:
            print(
                f"  DECLINED (persistent, shipping honest decline prose): {question!r}", flush=True
            )
            answer_text = best["decline_display"]
        outcome = None
        if not declined:
            meter.check(f"validator for {question!r}")
            answer = _grounded_answer_from_sse(transcript, retrieved)
            outcome = validate_exchange(adapter, answer, transcript, config=ValidatorConfig())
            usage = getattr(outcome, "usage", None)
            if usage:
                meter.record(
                    "validator", getattr(outcome, "model", GENERATION_MODEL_DEFAULT), usage
                )
        return {
            "answer_text": answer_text,
            "citations": citations,
            "footer": footer,
            "validated": bool(getattr(outcome, "validated", False)),
            "declined": bool(declined),
            "attempts": attempt,
            "retrieved_chunk_ids": [p.chunk_id for p in retrieved.passages],
        }

    def answer_fn(index: int, question: str, meter: SpendMeter) -> dict:
        decision = None
        for classify_attempt in range(1, 4):
            meter.check(f"classifier for {question!r}")
            decision = process_query(adapter, question, [])
            usage = getattr(decision.classification, "usage", None)
            meter.record("classify", GENERATION_MODEL_DEFAULT, usage)
            if decision.route is not Route.CANNED:
                break
            print(
                f"  classify attempt {classify_attempt}: CANNED "
                f"(scope={decision.classification.scope!r}) — canned_response head: "
                f"{(decision.canned_response or '')[:120]!r}",
                flush=True,
            )
        route = decision.route
        forced_route = False
        print(f"  route={route}", flush=True)
        chart_spec_hash = None
        if route is Route.CANNED:
            # Persistent canning of a canonical starter: force retrieval for
            # CACHE GENERATION ONLY (retrieval/generation/citations stay real).
            forced_route = True
            print(
                f"  FORCED-ROUTE retrieval for persistent-canned starter: {question!r}", flush=True
            )
            decision = dataclasses.replace(
                decision,
                route=Route.RETRIEVAL,
                retrieval_query=decision.classification.rewritten_query or question,
                canned_response=None,
            )
            route = decision.route
        if route is Route.CHART:
            # Flagship licence-blocked on #23: serve a REAL cited textual
            # answer for the 10k starter instead — re-route to retrieval.
            decision = dataclasses.replace(
                decision,
                route=Route.RETRIEVAL,
                retrieval_query=decision.chart_request or decision.classification.rewritten_query,
                chart_request=None,
            )
        elif route is not Route.RETRIEVAL:
            raise SystemExit(
                f"HALT: starter question {question!r} routed {route!r} — a starter cache entry "
                "cannot be unsafe/unknown; investigate before shipping"
            )
        fields = grounded_answer(decision, question, meter)
        entry = {
            "question": question,
            "answer_text": fields["answer_text"],
            "citations": fields["citations"],
            "footer": fields["footer"],
            "chart_spec_hash": chart_spec_hash,
        }
        report = {
            "question": question,
            "route": str(route),
            "forced_route": forced_route,
            "validated": fields["validated"],
            "declined": fields["declined"],
            "attempts": fields["attempts"],
            "citation_count": len(fields["citations"]),
            "retrieved_chunk_ids": fields["retrieved_chunk_ids"],
            "chart_spec_hash": chart_spec_hash,
        }
        return {"entry": entry, "report": report}

    summary = generate_starter_cache(
        STARTER_QUESTIONS,
        entries_dir=RUN / "entries",
        out_cache_dir=OUT_CACHE_DIR,
        meter=meter,
        answer_fn=answer_fn,
        generated_on=generated_on,
    )
    # Aggregate per-entry reports for the deploy report.
    reports = []
    for index in range(len(STARTER_QUESTIONS)):
        entry_path = RUN / "entries" / f"{index:02d}.json"
        if entry_path.is_file():
            saved = json.loads(entry_path.read_text(encoding="utf-8"))
            if isinstance(saved, Mapping) and "report" in saved:
                reports.append(saved["report"])
    atomic_write_text(
        RUN / "starter_build_report.json", json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
    )
    return 0 if summary["written"] >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
