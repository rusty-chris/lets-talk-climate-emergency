"""Retrieval service: rerank, structural voices filter, refusal gate (issue #11).

RED-phase contract stubs: every behaviour raises ``NotImplementedError``;
the failing suite in ``tests/unit/test_retrieval_*.py`` and
``tests/integration/test_reranker_smoke.py`` pins the contract below.

DESIGN §3.2: the #9 hybrid top-40 (``rag.indexing.hybrid_query``) is cut
to the top-8 fed to generation by the ``bge-reranker-v2-m3``
cross-encoder. ADR-006, stated precisely: cross-encoder logits are not
calibrated probabilities either — they are **query-comparable relevance
scores**, which is the property thresholding needs and RRF lacks (RRF
scores are rank-fusion artefacts; the top RRF score for a query with
excellent matches and a query with garbage matches can be identical).
The refusal gate (§3.5 / ADR-010) therefore thresholds on the reranker's
scores, with the threshold set empirically against refusal/false-refusal
targets — never hand-tuned in code.

Contract points the red suite pins:

- **Reranker seam.** All scoring flows through the :class:`Reranker`
  protocol. :class:`BgeRerankerV2M3` (real weights) is ONE
  implementation, used only at integration tier; every unit test injects
  a deterministic fake (``tests/_retrieval_fixtures``). Importing this
  module never imports torch/transformers (the heavy stack loads lazily
  inside ``BgeRerankerV2M3`` only), and never imports ``rag.provider``
  — nothing in this module can make an LLM call, so the honest-refusal
  path is structurally template-only.
- **Real reranker via ``transformers`` directly**, not FlagEmbedding's
  ``FlagReranker``: the wrapper calls the removed
  ``tokenizer.prepare_for_model`` and breaks under transformers 5.x
  (spike-03 finding, ``reviews/spike-03-probe-findings.md`` deviation 4).
  ``AutoModelForSequenceClassification`` + ``AutoTokenizer`` with
  ``sigmoid(logit)`` scores — identical model and semantics to the model
  card. Scores land in (0, 1), the scale the threshold artifact is
  calibrated in.
- **Structural voices filter (§3.2 / §2.5) — include-list-first.**
  ``source_type`` is (today) an unvalidated free string upstream
  (review finding #158): case variants, null, or a missing key would
  all slip past a ``must_not "voices"`` blocklist, and Qdrant's
  ``MatchAny`` is exact and case-sensitive. The policy therefore never
  blocklists: every retrieval route carries an INCLUDE list of
  known-good source types, applied IN THE STORE QUERY via the #9
  Prefetch-level ``include_source_types`` capability — never by
  post-hoc trimming of results. Science (non-voices) routes include
  ONLY :data:`EVIDENCE_SOURCE_TYPES`; voices-classified routes
  (``QueryDecision.voices_bias``) include evidence AND voices — bias,
  never exclusion of evidence. A chunk whose ``source_type`` is
  unknown, mis-cased, null or absent matches no include list and so
  fails CLOSED: it is never served on any route, whatever the parallel
  data-side hardening does. The pure policy is
  :func:`permitted_source_types`; the reranker (and therefore
  generation) never sees a voices or unknown-typed chunk on a science
  route.
- **Refusal gate (§3.5 / ADR-010).** Top reranked score below the
  configured threshold -> :class:`HonestRefusal`, a typed result whose
  text is a pure template (:func:`build_refusal_text`) naming what the
  corpus DOES cover — no LLM call. At-or-above threshold answers.
  Mixed above/below in the top-8 -> partial support: each passage
  carries ``clears_threshold`` so generation can name what is and is
  not supported.
- **Threshold is eval-derived config, never a magic constant.** The
  value is calibrated by #20/#21 on the gold set's no-answer calibration
  items (disjoint from the gate items, DESIGN §6.1) and carried in a
  committed artifact (:func:`save_threshold_artifact` /
  :func:`load_threshold_artifact`); :class:`RetrievalConfig` requires it
  explicitly, with no default anywhere in this module.
- **Metadata intact.** Every returned passage carries its full stored
  payload (§2.4 metadata: section path, attribution, ``source_type``,
  ``consensus_position``, and the #143 ``parse_backend`` /
  ``degraded_fallback`` / ``needs_hand_review`` flags).
- **Route flags carried untouched**: ``tone_flag`` (adversarial routes)
  rides the result — refusal or passages — unmodified.
- **Latency budget**: the ~100 ms CPU budget for reranking 40 pairs
  (ADR-006) is documented as :data:`RERANK_LATENCY_BUDGET_SECONDS`,
  recorded per run by :func:`record_rerank_latency`, and asserted only
  on the demo hardware profile (the ``perf``-marked integration test),
  never on CI hardware.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.indexing import (
    DEFAULT_TOP_K,
    EmbeddingModel,
    _hf_snapshot_dir,
    hybrid_query,
)
from rag.query import QueryDecision, Route

__all__ = [
    "BGE_RERANKER_MODEL_ID",
    "BGE_RERANKER_REVISION",
    "THRESHOLD_ARTIFACT_SCHEMA_VERSION",
    "RERANK_CANDIDATE_K",
    "GENERATION_TOP_K",
    "RERANK_LATENCY_BUDGET_SECONDS",
    "VOICES_SOURCE_TYPE",
    "EVIDENCE_SOURCE_TYPES",
    "KNOWN_SOURCE_TYPES",
    "RetrievalError",
    "CalibrationGateOverlapError",
    "Reranker",
    "BgeRerankerV2M3",
    "RerankedPassage",
    "RetrievedPassages",
    "HonestRefusal",
    "RetrievalConfig",
    "ThresholdCalibration",
    "permitted_source_types",
    "reranker_window_bounds",
    "build_refusal_text",
    "retrieve",
    "calibrate_refusal_threshold",
    "save_threshold_artifact",
    "load_threshold_artifact",
    "PREFILTER_ARTIFACT_SCHEMA_VERSION",
    "PrefilterCalibration",
    "calibrate_prefilter_floor",
    "save_prefilter_artifact",
    "load_prefilter_artifact",
    "check_calibration_gate_split",
    "PERF_LOG_PATH_ENV",
    "default_perf_log_path",
    "record_rerank_latency",
]

#: The pinned cross-encoder (DESIGN §3.2 / ADR-006).
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

#: Finding #178 (the #163 rule applied to the reranker): the FULL commit
#: hash of the Hugging Face hub revision the refusal threshold is
#: calibrated against, verified against both the hub and the local cached
#: snapshot on 2026-08-21. Under an unpinned load, a fresh machine could
#: fetch different weights under the same recorded model id — and scores
#: from different weights share no scale, silently invalidating the
#: calibrated threshold. Bump deliberately, together with a full
#: re-calibration (#20/#21).
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

#: The threshold artifact's on-disk schema version (finding #173). Written
#: by :func:`save_threshold_artifact` and REQUIRED verbatim by
#: :func:`load_threshold_artifact`: a document missing it (hand-built, or
#: from some other tool) or carrying a different version refuses loudly
#: instead of being guessed at. Bump together with any change to the
#: artifact's shape or semantics.
THRESHOLD_ARTIFACT_SCHEMA_VERSION = 1

#: DESIGN §3.2: hybrid top-40 in … (the #9 fused result set feeds the reranker)
RERANK_CANDIDATE_K = DEFAULT_TOP_K

#: … reranked top-8 out (the §3.4 bound on the generation call's documents).
GENERATION_TOP_K = 8

#: ADR-006: ~100 ms on CPU for 40 (query, passage) pairs. Recorded every
#: perf run; ASSERTED only on the demo hardware profile before release,
#: never on CI hardware (issue #11 acceptance criteria).
RERANK_LATENCY_BUDGET_SECONDS = 0.1

#: The §2.5 source_type label whose separation from evidence is structural.
VOICES_SOURCE_TYPE = "voices"

#: The closed include-list of source types science (non-voices) routes may
#: serve (finding #158: include-list-first — an unknown, mis-cased, null
#: or missing source_type matches no include list and fails CLOSED).
EVIDENCE_SOURCE_TYPES = ("evidence",)

#: The full known source_type vocabulary retrieval will ever serve, on any
#: route. Nothing outside it reaches a generation document set.
KNOWN_SOURCE_TYPES = EVIDENCE_SOURCE_TYPES + (VOICES_SOURCE_TYPE,)


def _is_finite_number(value: Any) -> bool:
    """True for a real, finite int/float — never bool (True would coerce
    to 1.0 and silently satisfy numeric checks), never NaN/inf."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reranker_weights_cached(model_id: str, revision: str) -> bool:
    """True when a non-empty local snapshot of ``model_id`` at exactly the
    pinned ``revision`` is cached — the cheap, download-free probe
    :class:`BgeRerankerV2M3` guards its construction with (never triggers
    a multi-GB fetch itself). Any OTHER cached revision does not count
    (findings #163/#178: different weights under the same model id).

    Delegates to the shared ``rag.indexing._hf_snapshot_dir`` so the query
    path and the index-build path probe the cache byte-identically."""
    return _hf_snapshot_dir(model_id, revision) is not None


class RetrievalError(RuntimeError):
    """Base class for loud retrieval refusals (never warnings, never silent)."""


def reranker_window_bounds(passage_token_count: int, window_budget: int) -> list[tuple[int, int]]:
    """Pure windowing arithmetic (finding #175): the [start, end) token
    spans the reranker scores a passage in, whose max wins.

    The chunker budgets ~500 whitespace WORDS per chunk
    (``ingestion.chunk.ChunkConfig.max_tokens``), which tokenise to far
    more XLM-R subwords than the 512-token joint pair cap — a single
    head-only truncated read left everything past ~the first half of a
    real-size chunk invisible (a relevant sentence in the tail scored
    identically to pure filler). Windows guarantee coverage instead:

    - the first window starts at token 0;
    - each window spans at most ``window_budget`` tokens;
    - consecutive windows leave no gap (the final window is
      right-aligned at the passage end, so it is always full-width and
      may overlap its predecessor — a sentence straddling the last
      boundary is still seen whole);
    - the last window ends exactly at ``passage_token_count``.

    A non-positive budget (a query longer than the whole pair cap)
    raises :class:`RetrievalError` — scoring nothing silently is the
    fail-open shape this module refuses everywhere.
    """
    if window_budget <= 0:
        raise RetrievalError(
            f"reranker window budget must be positive, got {window_budget} — "
            "the query (plus special tokens) consumed the whole pair cap, "
            "leaving no room to score any passage text (finding #175)"
        )
    if passage_token_count <= window_budget:
        return [(0, passage_token_count)]
    bounds: list[tuple[int, int]] = []
    start = 0
    while start + window_budget < passage_token_count:
        bounds.append((start, start + window_budget))
        start += window_budget
    bounds.append((passage_token_count - window_budget, passage_token_count))
    return bounds


class CalibrationGateOverlapError(RetrievalError):
    """A gold-set item id appears in BOTH the threshold-calibration subset
    and the refusal-gate eval subset.

    DESIGN §6.1: threshold-calibration items are disjoint from gate items
    — a threshold tuned on the very items the gate is scored on would
    make the >90%/<5% refusal gates circular. The check refuses loudly,
    naming the shared id(s), as #20's data lands.
    """


class Reranker(Protocol):
    """The reranker seam (IMPLEMENTATION.md §1: ``Reranker`` protocol).

    Implementations: :class:`BgeRerankerV2M3` (real weights,
    integration/production only) and the unit tier's deterministic fakes
    (``tests/_retrieval_fixtures``). ``score`` returns exactly one float
    per passage, in passage order, for ONE query — the scores are
    query-comparable relevance scores (ADR-006), the scale the refusal
    threshold lives in.

    ``model_id`` identifies the producing model; threshold calibration is
    model-coupled (scores from a different reranker share no scale).
    """

    @property
    def model_id(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class BgeRerankerV2M3:
    """The real local bge-reranker-v2-m3 cross-encoder (ADR-006). CPU.

    Contract (pinned by the single real-model integration smoke,
    ``test_reranker_orders_relevant_fixture_chunk_first``):

    - ``model_id`` == :data:`BGE_RERANKER_MODEL_ID`;
    - ``score`` reads each (query, passage) pair jointly and returns
      ``sigmoid(logit)`` per passage — floats strictly inside (0, 1),
      query-comparable across queries (ADR-006's wording: NOT calibrated
      probabilities; the property thresholding needs);
    - passages longer than one pair-cap window are scored in
      :func:`reranker_window_bounds` windows covering every token, max
      over windows, all windows in ONE batched model call (finding
      #175: the chunker's ~500-word budget tokenises to ~2x the pair
      cap — a head-only truncated read left tail content scoring as
      filler);
    - loaded via ``transformers`` ``AutoModelForSequenceClassification``
      + ``AutoTokenizer`` directly — FlagEmbedding's ``FlagReranker`` is
      broken under transformers 5.x (spike-03 deviation 4);
    - the heavy stack is imported lazily inside this class only;
    - construction fails with a clear :class:`RetrievalError` when the
      weights are not cached locally; it never silently downloads
      multi-GB weights inside a test run (same rule as
      ``rag.indexing.Bgem3EmbeddingModel``).
    """

    #: Cross-encoder input cap per WINDOW (bge-reranker-v2-m3, ADR-006 /
    #: finding #175). Each scored sequence (query + one passage window +
    #: special tokens) stays within this many tokens; a passage longer than
    #: one window's budget is scored in :func:`reranker_window_bounds`
    #: windows covering every token, and the passage's score is the max
    #: over its windows — never a silent head-only truncated read. Kept at
    #: 512 (not the model's 8192 ceiling) because cross-encoder cost grows
    #: superlinearly with sequence length: two 512-token windows batch
    #: cheaper than one 1024-token sequence, and the windows are
    #: embarrassingly batchable.
    _MAX_PAIR_TOKENS = 512

    def __init__(
        self, model_id: str = BGE_RERANKER_MODEL_ID, revision: str = BGE_RERANKER_REVISION
    ) -> None:
        if not _reranker_weights_cached(model_id, revision):
            raise RetrievalError(
                f"{model_id} weights at the PINNED revision {revision} are "
                "not cached locally under the Hugging Face hub cache "
                "(HF_HUB_CACHE / HF_HOME) — fetch them first (e.g. "
                f"`huggingface-cli download {model_id} --revision {revision}`); "
                "any other cached revision is different weights under the "
                "same model id (findings #163/#178), and BgeRerankerV2M3 "
                "never triggers an implicit multi-GB download on "
                "construction (same rule as rag.indexing.Bgem3EmbeddingModel)."
            )

        # Lazy: the heavy stack (torch / transformers) loads only inside this
        # real implementation, so `import rag.retrieval` stays weight-free
        # (IMPLEMENTATION.md §1/§3). transformers directly, NOT FlagEmbedding's
        # FlagReranker — the wrapper calls the removed
        # tokenizer.prepare_for_model and is broken under transformers 5.x
        # (spike-03 deviation 4).
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Force offline for the load itself: a partially-cached snapshot fails
        # loudly instead of silently fetching the missing pieces over the net.
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            # transformers forwards `revision` to the hub cache resolution
            # (unlike FlagEmbedding — the #163 workaround of passing the
            # snapshot path is not needed here): the load is pinned to
            # exactly the revision the guard above verified is cached.
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        self._model.eval()
        self._model_id = model_id
        self._revision = revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        """The pinned hub revision (full commit hash) the weights were
        loaded from (finding #178) — part of the loaded identity, so the
        model the threshold was calibrated against is verifiable."""
        return self._revision

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch

        passages = list(passages)
        if not passages:
            return []

        # Finding #175: score every passage COMPLETELY. The per-window
        # passage budget is what remains of the pair cap after the query
        # and the pair's special tokens; windows come from the pure
        # reranker_window_bounds (full coverage, right-aligned final
        # window), are mapped back to character spans via the fast
        # tokenizer's offsets, and ALL windows of ALL passages go through
        # the model in ONE batched call. A passage's score is the max
        # over its windows — a relevant sentence in the tail scores the
        # window that contains it, never the truncation floor.
        special_overhead = self._tokenizer.num_special_tokens_to_add(pair=True)
        query_token_count = len(self._tokenizer(query, add_special_tokens=False)["input_ids"])
        window_budget = self._MAX_PAIR_TOKENS - query_token_count - special_overhead
        if window_budget <= 0:
            raise RetrievalError(
                f"query tokenises to {query_token_count} tokens, consuming "
                f"the whole {self._MAX_PAIR_TOKENS}-token pair cap — no "
                "passage text could be scored at all, so the run refuses "
                "loudly instead of scoring nothing (finding #175)"
            )

        pair_texts: list[list[str]] = []
        window_owner: list[int] = []
        for passage_index, passage in enumerate(passages):
            encoding = self._tokenizer(
                passage, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = encoding["offset_mapping"]
            for start, end in reranker_window_bounds(len(offsets), window_budget):
                if start == end:  # empty passage: one empty window
                    window_text = ""
                else:
                    window_text = passage[offsets[start][0] : offsets[end - 1][1]]
                pair_texts.append([query, window_text])
                window_owner.append(passage_index)

        inputs = self._tokenizer(
            pair_texts,
            padding=True,
            truncation=True,
            max_length=self._MAX_PAIR_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.view(-1).float()
            # ADR-006: sigmoid(logit) -> query-comparable relevance scores
            # strictly inside (0, 1), the scale the refusal threshold lives in.
            # NOT calibrated probabilities.
            window_scores = torch.sigmoid(logits)

        best: list[float] = [0.0] * len(passages)
        for passage_index, window_score in zip(window_owner, window_scores, strict=True):
            best[passage_index] = max(best[passage_index], float(window_score))
        return best


@dataclass(frozen=True)
class RerankedPassage:
    """One reranked hit destined for the generation document set.

    ``payload`` is the chunk's full stored §2.4 payload, intact from the
    index (including the #143 ``parse_backend`` / ``degraded_fallback``
    / ``needs_hand_review`` flags). ``clears_threshold`` is the
    partial-support flag: whether THIS passage's ``rerank_score``
    cleared the configured refusal threshold.
    """

    chunk_id: str
    rerank_score: float
    clears_threshold: bool
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievedPassages:
    """The successful retrieval result: the generation call's document set.

    ``passages`` is the reranked top-8 (at most
    :data:`GENERATION_TOP_K`), best first by ``rerank_score``. This
    tuple IS the seam the §6.2 voices-separation eval checks: on a
    non-voices route it provably never contains a
    ``source_type: voices`` payload. ``partial_support`` is True when
    the passages are split above/below the threshold (§3.5: partial
    support named). ``tone_flag`` carries the adversarial routing flag
    through untouched.
    """

    passages: tuple[RerankedPassage, ...]
    tone_flag: bool = False
    partial_support: bool = False


@dataclass(frozen=True)
class HonestRefusal:
    """The typed below-threshold result (§3.5 / ADR-010).

    ``refusal_text`` is built by the pure template
    :func:`build_refusal_text` over ``covered_topics`` — what the corpus
    DOES cover — with no LLM call anywhere on the path (this module
    cannot import ``rag.provider``). ``top_score`` and ``threshold``
    record why the gate fired, for logging (#22) and the eval harness
    (#21). ``tone_flag`` carries the adversarial routing flag through
    untouched.
    """

    refusal_text: str
    covered_topics: tuple[str, ...]
    top_score: float
    threshold: float
    tone_flag: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """Injected retrieval configuration.

    ``refusal_threshold`` has NO default, deliberately: the value is
    eval-derived (#20's calibration items via
    :func:`calibrate_refusal_threshold`, carried in the committed
    threshold artifact) and re-calibrated per corpus version (ADR-010).
    A hard-coded default here would be exactly the hand-tuned magic
    constant the issue forbids. ``corpus_coverage`` names what the
    corpus covers, for the honest-refusal template.
    """

    refusal_threshold: float | None
    corpus_coverage: tuple[str, ...]
    candidate_top_k: int = RERANK_CANDIDATE_K
    final_top_k: int = GENERATION_TOP_K

    def __post_init__(self) -> None:
        # Issue #313: the threshold is DEMOTED to a cost-saving pre-filter, so
        # ``None`` is now a legal, deliberate state — the pre-filter DISABLED,
        # with the authoritative refusal signal living in generation. It is
        # NOT the finding-#172 hazard (that was a silently-disabled gate); a
        # None here is an explicit opt-out and skips the finite-number guard.
        if self.refusal_threshold is None:
            return
        # Finding #172: a NaN threshold makes every gate comparison False —
        # the gate never fires again, silently; ±inf pins it permanently
        # open or shut. A non-finite (or non-numeric/bool) threshold is a
        # typed configuration error at construction, never a silent
        # behaviour change.
        if not _is_finite_number(self.refusal_threshold):
            raise RetrievalError(
                f"refusal_threshold must be a finite real number; got "
                f"{self.refusal_threshold!r} — a non-finite threshold makes "
                "the refusal gate's comparison undefined (NaN compares False "
                "against everything), silently disabling honest refusal "
                "(review finding #172)"
            )


@dataclass(frozen=True)
class ThresholdCalibration:
    """The eval-derived threshold artifact (#20/#21 set the real value).

    ``calibration_item_ids`` records every gold-set item id the
    calibration consumed, so :func:`check_calibration_gate_split` can
    enforce the §6.1 disjointness rule against the gate items.
    """

    threshold: float
    calibration_item_ids: tuple[str, ...]


def permitted_source_types(decision: QueryDecision) -> tuple[str, ...]:
    """Pure policy: the INCLUDE list of ``source_type`` values this query
    may serve, applied IN THE STORE QUERY (the #9 Prefetch-level
    ``include_source_types`` capability).

    Non-voices retrieval routes -> :data:`EVIDENCE_SOURCE_TYPES` — the
    §3.2 structural invariant, include-list-first (finding #158): voices
    chunks are outside the list, and so is every unknown/mis-cased/null
    ``source_type``, which therefore fails CLOSED for science answers.
    Voices-biased routes (``decision.voices_bias``) ->
    :data:`KNOWN_SOURCE_TYPES` — evidence AND voices: bias toward the
    voices source, never exclusion of evidence, and still a closed
    vocabulary (unknown types serve nowhere).
    """
    if decision.voices_bias:
        return KNOWN_SOURCE_TYPES
    return EVIDENCE_SOURCE_TYPES


def build_refusal_text(covered_topics: Sequence[str]) -> str:
    """Pure template: the §3.5 honest-refusal text.

    States plainly that the corpus does not support an answer, then
    names every entry of ``covered_topics`` as what the corpus DOES
    cover. Template only — never an LLM call, never model-derived text.
    """
    lines = [
        "I can't answer that from the evidence in this corpus.",
        "Here is what the corpus does cover:",
    ]
    lines.extend(f"- {topic}" for topic in covered_topics)
    return "\n".join(lines)


def retrieve(
    client: Any,
    collection_name: str,
    decision: QueryDecision,
    *,
    embedding_model: EmbeddingModel,
    reranker: Reranker,
    config: RetrievalConfig,
    expected_corpus_version: str,
) -> RetrievedPassages | HonestRefusal:
    """The retrieval service: hybrid top-40 -> rerank -> gate -> top-8.

    Contract (the red suite pins each point):

    - ``decision`` must be a ``Route.RETRIEVAL`` decision; CHART/CANNED
      decisions raise :class:`RetrievalError` (those routes never reach
      retrieval). ``decision.retrieval_query`` is the query text; every
      other decision field (``preamble_note`` in particular) is opaque
      to this layer — read nothing else, rewrite nothing.
    - Candidates come from store queries restricted at the Prefetch
      level (``rag.indexing.hybrid_query``'s ``include_source_types``
      hook, ``top_k=config.candidate_top_k``) to
      :func:`permitted_source_types` for the route — include-list-first,
      never post-hoc trimming, so out-of-list chunks (voices on science
      routes, unknown/mis-cased/null source types on every route)
      neither occupy fused ranks nor ever reach the reranker.
    - The reranker is called exactly ONCE, with every candidate's text
      (the passage text contains the chunk body), and its scores order
      the result: top ``config.final_top_k`` passages by score,
      metadata intact.
    - Refusal gate: top reranked score < ``config.refusal_threshold``
      -> :class:`HonestRefusal` (template text over
      ``config.corpus_coverage``); top score >= threshold ->
      :class:`RetrievedPassages` with per-passage ``clears_threshold``
      flags and ``partial_support`` set when the top-8 straddles the
      threshold.
    - ``decision.tone_flag`` is carried onto the result untouched.
    """
    if decision.route is not Route.RETRIEVAL:
        raise RetrievalError(
            f"retrieve() received a {decision.route.value!r} decision; only "
            "Route.RETRIEVAL decisions reach retrieval — CHART and CANNED "
            "routes never query the store, so handing one in is a wiring bug"
        )

    # Include-list-first (finding #158): the structural voices/unknown-type
    # filter is applied IN THE STORE QUERY via the #9 Prefetch-level
    # include_source_types hook — never post-hoc trimming — so out-of-list
    # chunks neither occupy fused ranks nor ever reach the reranker.
    include_source_types = permitted_source_types(decision)
    candidates = hybrid_query(
        client,
        collection_name,
        decision.retrieval_query,
        embedding_model=embedding_model,
        expected_corpus_version=expected_corpus_version,
        top_k=config.candidate_top_k,
        include_source_types=include_source_types,
    )

    # Finding #174, belt-and-braces over the store filter: Qdrant match
    # conditions are satisfied when ANY element of an array payload value
    # matches, so a chunk stored with source_type ["voices", "evidence"]
    # passes the include MatchAny on science routes — a scalar-string
    # equality the store filter cannot express. Every candidate the store
    # hands back must therefore carry a scalar string source_type inside
    # the route's include list; anything else is a loud, named failure of
    # the run (an assertion, not post-hoc trimming — the include-list-first
    # stance is untouched, and legitimate data never trips this).
    for candidate in candidates:
        source_type = candidate.payload.get("source_type")
        if not isinstance(source_type, str) or source_type not in include_source_types:
            raise RetrievalError(
                f"chunk {candidate.chunk_id!r} came back from the store with "
                f"source_type {source_type!r}, which is not a scalar string "
                f"in this route's include list {include_source_types!r} — "
                "array payload values match ANY element under Qdrant's "
                "filter semantics, so malformed store data fails CLOSED as "
                "a named error instead of being served as evidence "
                "(review finding #174)"
            )

    # ONE batched rerank call over every candidate's body text (the
    # cross-encoder reads real evidence text). Latency-budget critical:
    # 40 pairs in one call, not 40 model invocations.
    passage_texts = [candidate.payload["body"] for candidate in candidates]
    scores = list(reranker.score(decision.retrieval_query, passage_texts))

    # Finding #172: the gate can only be honest over a sane signal. A
    # wrong-length score list or any non-finite score is a broken seam
    # contract — refuse the whole run loudly (fail CLOSED), never answer
    # from (or sort by) garbage: NaN passes `top < threshold` (every NaN
    # comparison is False) and seizes rank #1 under sorted(reverse=True).
    if len(scores) != len(candidates):
        raise RetrievalError(
            f"reranker {reranker.model_id!r} returned {len(scores)} score(s) "
            f"for {len(candidates)} candidate passage(s) — the Reranker seam "
            "contract is exactly one score per passage, in passage order; "
            "refusing the run rather than mis-attributing scores "
            "(review finding #172)"
        )
    corrupt = [
        (candidate.chunk_id, score)
        for candidate, score in zip(candidates, scores, strict=True)
        if not _is_finite_number(score)
    ]
    if corrupt:
        described = ", ".join(f"{chunk_id!r} scored {score!r}" for chunk_id, score in corrupt)
        raise RetrievalError(
            f"reranker {reranker.model_id!r} produced non-finite relevance "
            f"score(s): {described}. The refusal gate cannot threshold a "
            "non-finite signal (NaN compares False against every threshold, "
            "failing the gate OPEN), so the run refuses loudly instead of "
            "answering (review finding #172)"
        )

    # Reranker scores govern the order (ADR-006), not the RRF fused order.
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    top = ranked[: config.final_top_k]

    threshold = config.refusal_threshold
    top_score = top[0][1] if top else 0.0

    # Zero candidates: there is literally nothing to generate from, so the
    # honest refusal template applies on EVERY pre-filter state — including
    # disabled (issue #313: an empty document set is not an answer).
    if not top:
        return HonestRefusal(
            refusal_text=build_refusal_text(config.corpus_coverage),
            covered_topics=tuple(config.corpus_coverage),
            top_score=top_score,
            threshold=threshold if threshold is not None else 0.0,
            tone_flag=decision.tone_flag,
        )

    # Pre-filter gate (§3.5, DEMOTED by issue #313): a floor fires ONLY when
    # one is configured, refusing below it WITHOUT a generation call — the
    # cost optimisation the demoted threshold survives to provide. With the
    # pre-filter disabled (threshold None) every candidate proceeds to
    # generation, whose structured decline is now the authoritative refusal
    # signal — there is no score-based refusal in that state.
    if threshold is not None and top_score < threshold:
        return HonestRefusal(
            refusal_text=build_refusal_text(config.corpus_coverage),
            covered_topics=tuple(config.corpus_coverage),
            top_score=top_score,
            threshold=threshold,
            tone_flag=decision.tone_flag,
        )

    passages = tuple(
        RerankedPassage(
            chunk_id=candidate.chunk_id,
            rerank_score=score,
            # Disabled pre-filter: nothing straddles, every served passage
            # clears (partial_support stays False below).
            clears_threshold=threshold is None or score >= threshold,
            payload=candidate.payload,
        )
        for candidate, score in top
    )
    # Finding #172, enforced invariant: an answered result ALWAYS leads with
    # a passage that cleared the threshold. With finite scores and a finite
    # threshold this holds arithmetically; the check makes it a loud
    # guarantee rather than an accident of the arithmetic above.
    if not passages or not passages[0].clears_threshold:
        raise RetrievalError(
            "internal invariant violated: a RetrievedPassages result must "
            "lead with a passage whose score cleared the refusal threshold "
            f"(top {passages[0].rerank_score if passages else None!r} vs "
            f"threshold {threshold!r}) — refusing rather than serving an "
            "answer the gate never approved (review finding #172)"
        )
    # Partial support: the answered top-8 straddles the threshold (top cleared,
    # but at least one passage did not) — generation names what is and isn't
    # supported via each passage's clears_threshold flag (§3.5).
    partial_support = any(not passage.clears_threshold for passage in passages)
    return RetrievedPassages(
        passages=passages,
        tone_flag=decision.tone_flag,
        partial_support=partial_support,
    )


def calibrate_refusal_threshold(
    no_answer_top_scores: Mapping[str, float],
    answerable_top_scores: Mapping[str, float],
) -> ThresholdCalibration:
    """Pure, reproducible threshold calibration (ADR-010; scripted, never
    hand-tuned).

    Inputs: per-item TOP reranked score for the gold set's no-answer
    calibration items and for answerable calibration items (item id ->
    score). Identical inputs yield an identical threshold — no
    randomness, no hand adjustment. For separable inputs (every
    no-answer score below every answerable score) the threshold lands
    strictly above every no-answer score and at-or-below every
    answerable score, so the gate refuses all no-answer items
    (score < threshold) and passes all answerable ones
    (score >= threshold). The returned artifact records every consumed
    item id for the §6.1 disjointness check.

    Degenerate inputs refuse loudly (finding #177) instead of returning
    a confidently wrong artifact: both maps must be non-empty, every
    score a finite non-bool number strictly inside (0, 1) — the sigmoid
    scale real reranker scores live in (a NaN would propagate through
    max()/min() into a NaN threshold, silently killing the gate) — the
    two id sets must be disjoint, and the inputs must be separable
    (``max(no-answer) < min(answerable)``; at equality no threshold can
    split them under the gate's refuse-strictly-below arithmetic). The
    typed error names the offending item ids and, for non-separable
    inputs, the overlapping score range.
    """
    if not no_answer_top_scores or not answerable_top_scores:
        raise RetrievalError(
            "threshold calibration requires non-empty score maps for BOTH "
            "the no-answer and the answerable calibration items; got "
            f"{len(no_answer_top_scores)} no-answer and "
            f"{len(answerable_top_scores)} answerable item(s) — refusing "
            "rather than calibrating on nothing (finding #177)"
        )
    bad_scores = sorted(
        (item_id, score)
        for subset in (no_answer_top_scores, answerable_top_scores)
        for item_id, score in subset.items()
        if not _is_finite_number(score) or not 0.0 < score < 1.0
    )
    if bad_scores:
        described = ", ".join(f"{item_id!r} scored {score!r}" for item_id, score in bad_scores)
        raise RetrievalError(
            "threshold calibration scores must be finite numbers strictly "
            "inside (0, 1) — the sigmoid scale real reranker scores live in; "
            f"got: {described}. A NaN here would propagate into a NaN "
            "threshold and silently kill the refusal gate (finding #177)"
        )
    shared_ids = sorted(set(no_answer_top_scores) & set(answerable_top_scores))
    if shared_ids:
        raise RetrievalError(
            "the same gold-set item id(s) appear in BOTH the no-answer and "
            "the answerable calibration subsets — a §6.1 bookkeeping bug in "
            f"the caller, refused rather than double-recorded: "
            f"{', '.join(shared_ids)} (finding #177)"
        )
    max_no_answer_value = max(no_answer_top_scores.values())
    min_answerable_value = min(answerable_top_scores.values())
    if max_no_answer_value >= min_answerable_value:
        overlapping_no_answer = sorted(
            item_id
            for item_id, score in no_answer_top_scores.items()
            if score >= min_answerable_value
        )
        overlapped_answerable = sorted(
            item_id
            for item_id, score in answerable_top_scores.items()
            if score <= max_no_answer_value
        )
        raise RetrievalError(
            "calibration inputs are not separable: the no-answer and "
            f"answerable score distributions overlap in "
            f"[{min_answerable_value!r}, {max_no_answer_value!r}] — any "
            "midpoint threshold would pass known no-answer item(s) "
            f"{', '.join(overlapping_no_answer)} and/or refuse known "
            f"answerable item(s) {', '.join(overlapped_answerable)}. Fix "
            "the calibration data (or the retrieval defect it exposes) "
            "rather than shipping a mis-set gate (finding #177)"
        )
    # Midpoint between the highest no-answer score and the lowest answerable
    # score: for separable inputs it lands strictly above every no-answer
    # score and at-or-below every answerable score, so the gate refuses all
    # no-answer items and passes all answerable ones. Pure arithmetic — no
    # randomness, no hand adjustment; identical inputs -> identical output.
    max_no_answer = max(no_answer_top_scores.values())
    min_answerable = min(answerable_top_scores.values())
    threshold = (max_no_answer + min_answerable) / 2
    item_ids = tuple(no_answer_top_scores) + tuple(answerable_top_scores)
    return ThresholdCalibration(threshold=threshold, calibration_item_ids=item_ids)


def save_threshold_artifact(calibration: ThresholdCalibration, path: Path) -> None:
    """Write the eval-derived threshold artifact (JSON) — the config source
    :class:`RetrievalConfig` is fed from, committed by the #20/#21 eval
    run, never edited by hand."""
    document = {
        "schema_version": THRESHOLD_ARTIFACT_SCHEMA_VERSION,
        "threshold": calibration.threshold,
        "calibration_item_ids": list(calibration.calibration_item_ids),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _reject_json_constant(literal: str) -> float:
    """``json.loads`` hook: the non-standard ``NaN``/``Infinity``/
    ``-Infinity`` literals are exactly the tampered content finding #173
    demands the loader refuse — never parse them into a live float."""
    raise ValueError(f"non-standard JSON literal {literal!r} is not a valid threshold value")


def load_threshold_artifact(path: Path) -> ThresholdCalibration:
    """Read the threshold artifact back; round-trips
    :func:`save_threshold_artifact` exactly.

    A missing or malformed artifact raises :class:`RetrievalError` — the
    service never falls back to a built-in threshold (there is none).
    Validation is strict (finding #173): the document must be a JSON
    object carrying ``schema_version`` ==
    :data:`THRESHOLD_ARTIFACT_SCHEMA_VERSION`, a ``threshold`` that is a
    finite non-bool number strictly inside (0, 1) — the sigmoid scale
    every real calibration lives in; any value at or outside the bounds
    cannot have come from :func:`calibrate_refusal_threshold` over real
    scores, and anything <= 0 would disable refusal outright — and
    ``calibration_item_ids`` as a JSON array of strings (a plain string
    would be silently shredded into characters, vacuously defeating the
    §6.1 disjointness guard). Nothing is coerced.
    """
    try:
        raw = path.read_text()
    except OSError as error:
        raise RetrievalError(
            f"threshold artifact {str(path)!r} could not be read — the "
            "service never invents a fallback threshold, so a missing "
            f"artifact refuses loudly: {error}"
        ) from error

    def _malformed(reason: str) -> RetrievalError:
        return RetrievalError(
            f"threshold artifact {str(path)!r} is malformed — {reason}; the "
            "service never falls back to a built-in threshold (finding #173)"
        )

    try:
        document = json.loads(raw, parse_constant=_reject_json_constant)
    except ValueError as error:
        raise _malformed(f"not parseable as strict JSON: {error}") from error
    if not isinstance(document, dict):
        raise _malformed(f"expected a JSON object, got {type(document).__name__}")

    schema_version = document.get("schema_version")
    if schema_version != THRESHOLD_ARTIFACT_SCHEMA_VERSION:
        raise _malformed(
            f"schema_version must be {THRESHOLD_ARTIFACT_SCHEMA_VERSION}, "
            f"got {schema_version!r} — an artifact without the expected "
            "schema marker was not written by save_threshold_artifact"
        )

    if "threshold" not in document or "calibration_item_ids" not in document:
        raise _malformed("missing 'threshold' and/or 'calibration_item_ids'")
    threshold = document["threshold"]
    if not _is_finite_number(threshold):
        raise _malformed(
            f"'threshold' must be a finite number, got {threshold!r} — a "
            "non-finite threshold silently kills the refusal gate"
        )
    if not 0.0 < threshold < 1.0:
        raise _malformed(
            f"'threshold' must lie strictly inside (0, 1) — the sigmoid "
            f"scale reranker scores live in — got {threshold!r}; a value at "
            "or outside the bounds cannot come from a real calibration and "
            "would pin the gate permanently open or shut"
        )

    item_ids = document["calibration_item_ids"]
    if not isinstance(item_ids, list) or not all(isinstance(i, str) for i in item_ids):
        raise _malformed(
            f"'calibration_item_ids' must be a JSON array of strings, got "
            f"{item_ids!r} — a plain string would be shredded into single "
            "characters, vacuously defeating the §6.1 disjointness guard"
        )

    return ThresholdCalibration(threshold=float(threshold), calibration_item_ids=tuple(item_ids))


# ---------------------------------------------------------------------------
# Issue #313: the threshold demoted to a cost-saving PRE-FILTER.
#
# The 2026-09 live release run proved ADR-010's single top-score threshold
# unsatisfiable on real reranker geometry (a no-answer item at 0.3885 vs
# answerable items at 0.0014–0.05 — the distributions fully overlap), while
# the generation-level honest declines went 10/10. ORCHESTRATOR ADJUDICATION
# (issue #313): the structured generation-level decline
# (rag.generation.GENERATION_DECLINE_MARKER) is now the AUTHORITATIVE
# refusal signal; the reranker score is retained ONLY as a conservative
# pre-filter that skips the generation spend when retrieval is hopeless.
# Consequences pinned by tests/unit/test_review_313_prefilter_demotion.py:
#
# - ``RetrievalConfig.refusal_threshold`` accepts ``None`` — pre-filter
#   DISABLED: ``retrieve`` then always returns RetrievedPassages when any
#   candidate exists (every passage ``clears_threshold`` True, never
#   ``partial_support``), and still returns HonestRefusal when the store
#   returns ZERO candidates (there is literally nothing to generate from).
# - Calibration no longer requires separable distributions: the floor is
#   derived from the ANSWERABLE side alone, so an inseparable corpus no
#   longer bricks the release (finding #177's REFUSED is reserved for
#   genuinely degenerate inputs — empty maps, out-of-scale scores,
#   overlapping ids — never for honest overlap).
# - A missing or unreadable/malformed pre-filter artifact DEGRADES to
#   pre-filter-disabled with a recorded reason (and a warning at the call
#   site) instead of blocking startup — the fail-safe direction is now
#   "spend a generation call and let the model decline honestly", never
#   "refuse to boot" (#216 interplay: see service.main).
# ---------------------------------------------------------------------------

#: The pre-filter artifact's on-disk schema version (issue #313). Distinct
#: from THRESHOLD_ARTIFACT_SCHEMA_VERSION (=1, the retired arbiter shape):
#: a v1 artifact is not silently reinterpreted as a pre-filter.
PREFILTER_ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PrefilterCalibration:
    """The eval-derived PRE-FILTER floor artifact (issue #313).

    ``enabled`` False means the pre-filter is OFF (threshold None): every
    retrieval-route query proceeds to generation, whose structured decline
    is the authoritative refusal signal; ``reason`` records why (missing/
    malformed artifact, or a deliberately disabled calibration).
    ``separable`` is a DIAGNOSTIC record of whether the calibration
    distributions were separable — it never gates anything (issue #313:
    inseparability is expected on real geometry and no longer a failure).
    ``calibration_item_ids`` feeds the §6.1 disjointness check exactly as
    the v1 artifact did (:func:`check_calibration_gate_split`).
    """

    threshold: float | None
    enabled: bool
    calibration_item_ids: tuple[str, ...] = ()
    separable: bool | None = None
    reason: str | None = None


def calibrate_prefilter_floor(
    no_answer_top_scores: Mapping[str, float],
    answerable_top_scores: Mapping[str, float],
) -> PrefilterCalibration:
    """Pure, reproducible pre-filter floor calibration (issue #313).

    RED-phase contract stub; the failing suite in
    ``tests/unit/test_review_313_prefilter_demotion.py`` pins:

    - The floor is CONSERVATIVE by construction: derived from the
      answerable side alone as ``min(answerable_top_scores.values()) / 2``
      — strictly below EVERY answerable calibration score, so the
      pre-filter can never refuse a calibration-answerable query (zero
      false pre-filter refusals by arithmetic), and strictly above 0.
      It fires only when retrieval is hopeless, saving the generation
      spend (§3.5's cost goal, now a cost optimisation rather than the
      refusal arbiter).
    - Separability is NOT required: on the live 2026-09 geometry
      (no-answer max 0.3885 >= answerable min 0.00142) the calibration
      still returns an ENABLED floor, with ``separable`` False recorded
      as a diagnostic. An inseparable-distribution corpus no longer
      bricks the release.
    - Finding #177's discipline is retained for genuinely degenerate
      inputs: empty maps (either side), any non-finite/bool score or a
      score outside (0, 1), or item ids shared between the two maps
      raise :class:`RetrievalError` naming the offenders.
    - Deterministic: identical inputs -> identical output; the returned
      ``calibration_item_ids`` records every consumed id (no-answer then
      answerable) for the §6.1 disjointness check; ``enabled`` True,
      ``reason`` None.
    """
    # Finding #177's discipline is RETAINED for genuinely degenerate inputs
    # (garbage), not for honest overlap: empty maps, non-finite/out-of-scale
    # scores, or shared ids are calibration BUGS, refused with a typed error.
    if not no_answer_top_scores or not answerable_top_scores:
        raise RetrievalError(
            "pre-filter calibration requires non-empty score maps for BOTH "
            "the no-answer and the answerable calibration items; got "
            f"{len(no_answer_top_scores)} no-answer and "
            f"{len(answerable_top_scores)} answerable item(s) — refusing "
            "rather than calibrating on nothing (finding #177)"
        )
    bad_scores = sorted(
        (item_id, score)
        for subset in (no_answer_top_scores, answerable_top_scores)
        for item_id, score in subset.items()
        if not _is_finite_number(score) or not 0.0 < score < 1.0
    )
    if bad_scores:
        described = ", ".join(f"{item_id!r} scored {score!r}" for item_id, score in bad_scores)
        raise RetrievalError(
            "pre-filter calibration scores must be finite numbers strictly "
            "inside (0, 1) — the sigmoid scale real reranker scores live in; "
            f"got: {described} (finding #177)"
        )
    shared_ids = sorted(set(no_answer_top_scores) & set(answerable_top_scores))
    if shared_ids:
        raise RetrievalError(
            "the same gold-set item id(s) appear in BOTH the no-answer and "
            "the answerable calibration subsets — a §6.1 bookkeeping bug in "
            f"the caller: {', '.join(shared_ids)} (finding #177)"
        )

    # The CONSERVATIVE floor: half the lowest answerable calibration score.
    # By construction it sits strictly below EVERY answerable score (so the
    # pre-filter can never refuse a calibration-answerable query — zero false
    # pre-filter refusals) and strictly above 0. Separability is recorded as
    # a DIAGNOSTIC only — an inseparable real corpus (no-answer max >=
    # answerable min) still yields an enabled floor and no longer bricks the
    # release (issue #313 adjudication).
    min_answerable = min(answerable_top_scores.values())
    threshold = min_answerable / 2
    separable = max(no_answer_top_scores.values()) < min_answerable
    item_ids = tuple(no_answer_top_scores) + tuple(answerable_top_scores)
    return PrefilterCalibration(
        threshold=threshold,
        enabled=True,
        calibration_item_ids=item_ids,
        separable=separable,
        reason=None,
    )


def save_prefilter_artifact(calibration: PrefilterCalibration, path: Path) -> None:
    """Write the pre-filter artifact (JSON, schema v2) — the config source
    the service's retrieval seam is fed from (issue #313).

    RED-phase contract stub; pinned: round-trips
    :func:`load_prefilter_artifact` exactly for BOTH enabled and disabled
    calibrations (a deliberately-disabled pre-filter is a committable,
    honest artifact), writing ``schema_version`` ==
    :data:`PREFILTER_ARTIFACT_SCHEMA_VERSION`.
    """
    document = {
        "schema_version": PREFILTER_ARTIFACT_SCHEMA_VERSION,
        "threshold": calibration.threshold,
        "enabled": calibration.enabled,
        "calibration_item_ids": list(calibration.calibration_item_ids),
        "separable": calibration.separable,
        "reason": calibration.reason,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def load_prefilter_artifact(path: Path) -> PrefilterCalibration:
    """Read the pre-filter artifact back — DEGRADING, never blocking.

    RED-phase contract stub (issue #313); the failing suite pins:

    - A well-formed schema-v2 document round-trips
      :func:`save_prefilter_artifact` exactly.
    - A MISSING/unreadable file returns a DISABLED
      :class:`PrefilterCalibration` (``enabled`` False, ``threshold``
      None) whose ``reason`` names the path — it NEVER raises. The
      pre-filter is a cost optimisation; its absence must not block a
      deploy (#216 interplay — contrast :func:`load_threshold_artifact`,
      the retired v1 arbiter loader, whose loud refusal is unchanged).
    - A malformed document (unparseable JSON, wrong/missing
      schema_version, non-finite or out-of-(0,1) threshold on an enabled
      record) ALSO returns a DISABLED calibration with a reason naming
      the defect — degraded honestly, never a live NaN threshold and
      never a crash. Non-standard JSON constants (NaN/Infinity) are
      malformed, not values.
    """

    def _disabled(reason: str) -> PrefilterCalibration:
        # The fail-safe direction under issue #313 is "spend a generation call
        # and let the model decline honestly", never "refuse to boot": a
        # missing/malformed artifact DEGRADES the pre-filter to off, carrying
        # its reason, and never raises.
        return PrefilterCalibration(threshold=None, enabled=False, reason=reason)

    try:
        raw = path.read_text()
    except OSError as error:
        return _disabled(
            f"pre-filter artifact {str(path)!r} could not be read, so the "
            "cost-saving pre-filter is DISABLED (its absence is not a deploy "
            f"blocker under issue #313): {error}"
        )

    try:
        document = json.loads(raw, parse_constant=_reject_json_constant)
    except ValueError as error:
        return _disabled(f"pre-filter artifact {str(path)!r} is not strict JSON: {error}")
    if not isinstance(document, dict):
        return _disabled(
            f"pre-filter artifact {str(path)!r} is not a JSON object, got {type(document).__name__}"
        )
    if document.get("schema_version") != PREFILTER_ARTIFACT_SCHEMA_VERSION:
        return _disabled(
            f"pre-filter artifact {str(path)!r} carries schema_version "
            f"{document.get('schema_version')!r}, not the expected "
            f"{PREFILTER_ARTIFACT_SCHEMA_VERSION} — a v1 arbiter artifact is "
            "never silently reinterpreted as a pre-filter"
        )

    item_ids_raw = document.get("calibration_item_ids", [])
    item_ids = (
        tuple(item_ids_raw)
        if isinstance(item_ids_raw, list) and all(isinstance(i, str) for i in item_ids_raw)
        else ()
    )
    separable_raw = document.get("separable")
    separable = separable_raw if isinstance(separable_raw, bool) else None
    reason_raw = document.get("reason")
    reason = reason_raw if isinstance(reason_raw, str) else None

    if not document.get("enabled"):
        # A deliberately-disabled (committable) artifact round-trips as such.
        return PrefilterCalibration(
            threshold=None,
            enabled=False,
            calibration_item_ids=item_ids,
            separable=separable,
            reason=reason,
        )

    threshold = document.get("threshold")
    if not _is_finite_number(threshold) or not 0.0 < threshold < 1.0:
        return _disabled(
            f"pre-filter artifact {str(path)!r} enables a threshold "
            f"{threshold!r} outside the (0, 1) sigmoid scale real reranker "
            "scores live in — degrading to disabled rather than gating on a "
            "nonsense floor"
        )
    return PrefilterCalibration(
        threshold=float(threshold),
        enabled=True,
        calibration_item_ids=item_ids,
        separable=separable,
        reason=reason,
    )


def check_calibration_gate_split(
    calibration_item_ids: Sequence[str],
    gate_item_ids: Sequence[str],
) -> None:
    """Refuse (``CalibrationGateOverlapError``, naming the shared ids) when
    any item id appears in both subsets; return None when disjoint.
    Guards DESIGN §6.1 as #20's gold-set data lands."""
    shared = sorted(set(calibration_item_ids) & set(gate_item_ids))
    if shared:
        raise CalibrationGateOverlapError(
            "threshold-calibration items and refusal-gate items must be "
            "disjoint (DESIGN §6.1) — a threshold tuned on the gate's own "
            "items makes the release gates circular; shared id(s): "
            f"{', '.join(shared)}"
        )
    return None


#: Env var overriding where the rerank perf log is written (finding #176):
#: the CI workflow and the release runbook both point it (or leave the
#: default) somewhere that OUTLIVES the run — evidence is only evidence
#: if it survives the process that produced it.
PERF_LOG_PATH_ENV = "CLIMATE_CHAT_PERF_LOG"


def perf_log_home(filename: str, *, env_override: str | None = None) -> Path:
    """The persistent home for a perf log named ``filename`` (finding #176).

    The single definition of the #176 convention, shared by every perf log
    in the stack (rerank latency here, badge latency in
    ``rag.citation_validator``): when ``env_override`` names an env var that
    is set, that value is the full log path (CI / the runbook point it
    somewhere that outlives the run); otherwise the committed ``evals/perf/``
    directory at the repo root — never a temp dir a test run deletes on the
    way out."""
    if env_override:
        override = os.environ.get(env_override)
        if override:
            return Path(override)
    return Path(__file__).resolve().parents[1] / "evals" / "perf" / filename


def _append_perf_row(log_path: Path, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Append one perf-log record dict to ``log_path`` as a CSV row (finding
    #176), creating the file (and its parents) with a header row when absent
    and appending otherwise; the written record is returned. The one shared
    DictWriter/header-on-create choreography behind every perf-log wrapper."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(record)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return record


def default_perf_log_path() -> Path:
    """The rerank perf log's resolved home (finding #176): the
    :data:`PERF_LOG_PATH_ENV` env var when set, else the committed
    ``evals/perf/`` directory at the repo root."""
    return perf_log_home("rerank-latency.csv", env_override=PERF_LOG_PATH_ENV)


def record_rerank_latency(
    log_path: Path | None = None,
    *,
    passage_count: int,
    wall_clock_seconds: float,
    hardware_profile: str,
) -> Mapping[str, Any]:
    """Append one rerank-latency measurement to the perf log (CSV).

    ``log_path`` defaults to :func:`default_perf_log_path` (finding
    #176: the log lives at a persistent, documented home — env-var
    overridable — not in whatever temp dir the caller had handy).
    Every record carries at least ``passage_count``,
    ``wall_clock_seconds``, ``budget_seconds``
    (== :data:`RERANK_LATENCY_BUDGET_SECONDS`), ``within_budget`` and
    ``hardware_profile``, and the written record is returned. The
    budget itself is asserted only on the demo hardware profile (issue
    #11 acceptance criteria), so CI records evidence without gating on
    CI hardware speed.
    """
    record: dict[str, Any] = {
        "passage_count": passage_count,
        "wall_clock_seconds": wall_clock_seconds,
        "budget_seconds": RERANK_LATENCY_BUDGET_SECONDS,
        "within_budget": wall_clock_seconds <= RERANK_LATENCY_BUDGET_SECONDS,
        "hardware_profile": hardware_profile,
    }
    return _append_perf_row(default_perf_log_path() if log_path is None else log_path, record)
