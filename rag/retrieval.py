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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.indexing import DEFAULT_TOP_K, EmbeddingModel
from rag.query import QueryDecision

__all__ = [
    "BGE_RERANKER_MODEL_ID",
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

    def __init__(self, model_id: str = BGE_RERANKER_MODEL_ID) -> None:
        raise NotImplementedError("issue #11 red phase: BgeRerankerV2M3 not implemented yet")

    @property
    def model_id(self) -> str:
        raise NotImplementedError("issue #11 red phase: BgeRerankerV2M3 not implemented yet")

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise NotImplementedError("issue #11 red phase: BgeRerankerV2M3 not implemented yet")


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
    raise NotImplementedError("issue #11 red phase: permitted_source_types not implemented yet")


def build_refusal_text(covered_topics: Sequence[str]) -> str:
    """Pure template: the §3.5 honest-refusal text.

    States plainly that the corpus does not support an answer, then
    names every entry of ``covered_topics`` as what the corpus DOES
    cover. Template only — never an LLM call, never model-derived text.
    """
    raise NotImplementedError("issue #11 red phase: build_refusal_text not implemented yet")


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
    raise NotImplementedError("issue #11 red phase: retrieve not implemented yet")


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
    raise NotImplementedError(
        "issue #11 red phase: calibrate_refusal_threshold not implemented yet"
    )


def save_threshold_artifact(calibration: ThresholdCalibration, path: Path) -> None:
    """Write the eval-derived threshold artifact (JSON) — the config source
    :class:`RetrievalConfig` is fed from, committed by the #20/#21 eval
    run, never edited by hand."""
    raise NotImplementedError("issue #11 red phase: save_threshold_artifact not implemented yet")


def load_threshold_artifact(path: Path) -> ThresholdCalibration:
    """Read the threshold artifact back; round-trips
    :func:`save_threshold_artifact` exactly.

    A missing or malformed artifact raises :class:`RetrievalError` — the
    service never falls back to a built-in threshold (there is none).
    """
    raise NotImplementedError("issue #11 red phase: load_threshold_artifact not implemented yet")


def check_calibration_gate_split(
    calibration_item_ids: Sequence[str],
    gate_item_ids: Sequence[str],
) -> None:
    """Refuse (``CalibrationGateOverlapError``, naming the shared ids) when
    any item id appears in both subsets; return None when disjoint.
    Guards DESIGN §6.1 as #20's gold-set data lands."""
    raise NotImplementedError(
        "issue #11 red phase: check_calibration_gate_split not implemented yet"
    )


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
    raise NotImplementedError("issue #11 red phase: record_rerank_latency not implemented yet")
