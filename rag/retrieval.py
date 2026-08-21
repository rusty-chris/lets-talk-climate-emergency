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

from rag.indexing import DEFAULT_TOP_K, EmbeddingModel, hybrid_query
from rag.query import QueryDecision, Route

__all__ = [
    "BGE_RERANKER_MODEL_ID",
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
    "build_refusal_text",
    "retrieve",
    "calibrate_refusal_threshold",
    "save_threshold_artifact",
    "load_threshold_artifact",
    "check_calibration_gate_split",
    "record_rerank_latency",
]

#: The pinned cross-encoder (DESIGN §3.2 / ADR-006).
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

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


def _hf_hub_cache_dir() -> Path:
    """The Hugging Face hub cache directory, honouring HF_HUB_CACHE/HF_HOME
    (same convention as ``tests/_weights.py`` and ``rag.indexing``,
    duplicated here because production code never imports from ``tests/``)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_finite_number(value: Any) -> bool:
    """True for a real, finite int/float — never bool (True would coerce
    to 1.0 and silently satisfy numeric checks), never NaN/inf."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reranker_weights_cached(model_id: str) -> bool:
    """True when a non-empty local snapshot of ``model_id`` is cached — the
    same cheap, download-free probe :class:`BgeRerankerV2M3` guards its
    construction with (never triggers a multi-GB fetch itself)."""
    snapshots = _hf_hub_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(entry.is_dir() and any(entry.iterdir()) for entry in snapshots.iterdir())


class RetrievalError(RuntimeError):
    """Base class for loud retrieval refusals (never warnings, never silent)."""


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
    - loaded via ``transformers`` ``AutoModelForSequenceClassification``
      + ``AutoTokenizer`` directly — FlagEmbedding's ``FlagReranker`` is
      broken under transformers 5.x (spike-03 deviation 4);
    - the heavy stack is imported lazily inside this class only;
    - construction fails with a clear :class:`RetrievalError` when the
      weights are not cached locally; it never silently downloads
      multi-GB weights inside a test run (same rule as
      ``rag.indexing.Bgem3EmbeddingModel``).
    """

    #: Cross-encoder input cap (bge-reranker-v2-m3, ADR-006). The joint
    #: (query, passage) pair is truncated to this many tokens before scoring.
    _MAX_PAIR_TOKENS = 512

    def __init__(self, model_id: str = BGE_RERANKER_MODEL_ID) -> None:
        if not _reranker_weights_cached(model_id):
            raise RetrievalError(
                f"{model_id} weights are not cached locally under the Hugging "
                "Face hub cache (HF_HUB_CACHE / HF_HOME) — fetch them first "
                f"(e.g. `huggingface-cli download {model_id}`); "
                "BgeRerankerV2M3 never triggers an implicit multi-GB download "
                "on construction (same rule as rag.indexing.Bgem3EmbeddingModel)."
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
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        self._model.eval()
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch

        passages = list(passages)
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        inputs = self._tokenizer(
            pairs,
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
            scores = torch.sigmoid(logits)
        return [float(s) for s in scores]


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

    refusal_threshold: float
    corpus_coverage: tuple[str, ...]
    candidate_top_k: int = RERANK_CANDIDATE_K
    final_top_k: int = GENERATION_TOP_K

    def __post_init__(self) -> None:
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

    # Refusal gate: strictly below threshold refuses (at-threshold answers).
    if not top or top_score < threshold:
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
            clears_threshold=score >= threshold,
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
    """
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


def record_rerank_latency(
    log_path: Path,
    *,
    passage_count: int,
    wall_clock_seconds: float,
    hardware_profile: str,
) -> Mapping[str, Any]:
    """Append one rerank-latency measurement to the perf log (CSV).

    Creates ``log_path`` with a header row when absent; every record
    carries at least ``passage_count``, ``wall_clock_seconds``,
    ``budget_seconds`` (== :data:`RERANK_LATENCY_BUDGET_SECONDS`),
    ``within_budget`` and ``hardware_profile``, and the written record
    is returned. The budget itself is asserted only on the demo
    hardware profile (issue #11 acceptance criteria), so CI records
    evidence without gating on CI hardware speed.
    """
    record: dict[str, Any] = {
        "passage_count": passage_count,
        "wall_clock_seconds": wall_clock_seconds,
        "budget_seconds": RERANK_LATENCY_BUDGET_SECONDS,
        "within_budget": wall_clock_seconds <= RERANK_LATENCY_BUDGET_SECONDS,
        "hardware_profile": hardware_profile,
    }
    fieldnames = list(record)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return record
