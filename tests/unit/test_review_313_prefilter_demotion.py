"""Issue #313 red phase (Fable): the reranker threshold demoted from
refusal ARBITER to conservative spend PRE-FILTER.

The 2026-09 live calibration proved the arbiter role unsatisfiable on
real geometry — the production-path scores put a no-answer item at
0.3885 while answerable conversational items sat at 0.00142–0.05, so no
threshold can pass both refusal gates (the record's own verdict). The
authoritative refusal signal moves to the structured generation-level
decline (test_review_313_decline_marker / _service_decline_route); the
threshold survives ONLY to skip the generation spend when retrieval is
hopeless. Pinned here:

- ``RetrievalConfig.refusal_threshold`` accepts ``None`` = pre-filter
  DISABLED; NaN/inf/bool are still refused loudly (finding #172).
- ``retrieve`` with the pre-filter disabled never score-refuses (every
  passage ``clears_threshold`` True, no partial_support) but STILL
  refuses on zero candidates — nothing to generate from is not an
  answer.
- ``calibrate_prefilter_floor``: a conservative floor from the
  ANSWERABLE side alone (min/2 — strictly below every answerable
  calibration score, so zero false pre-filter refusals by
  construction); separability recorded as a diagnostic, NEVER required
  — the live inseparable geometry calibrates instead of bricking the
  release; finding #177's refusals retained for genuinely degenerate
  inputs only.
- The schema-v2 pre-filter artifact DEGRADES on missing/malformed
  (disabled + reason, never an exception, never a live NaN) — the #216
  boot-blocking behaviour is retired in
  test_review_313_deploy_degradation.

FLAGGED (test-author decisions): floor arithmetic pinned as
``min(answerable)/2``; missing AND malformed artifacts both degrade to
disabled (the fail-safe direction is now "spend a generation call and
let the model decline honestly"); the v1 arbiter machinery
(``calibrate_refusal_threshold`` / ``load_threshold_artifact``) is left
in place untouched, diagnostic-only.

No test here touches the network (in-memory Qdrant, fake rerankers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.retrieval import (
    PREFILTER_ARTIFACT_SCHEMA_VERSION,
    HonestRefusal,
    PrefilterCalibration,
    RetrievalConfig,
    RetrievalError,
    RetrievedPassages,
    calibrate_prefilter_floor,
    load_prefilter_artifact,
    save_prefilter_artifact,
)
from tests._retrieval_fixtures import (
    HashReranker,
    decision,
    indexed_corpus,
    run_retrieve,
)

#: The LIVE calibration geometry from the issue-#313 evidence
#: (data/release-run/calibration_scores.json, quoted in the issue): the
#: no-answer max (qa-na-c-11) sits far ABOVE the answerable min
#: (qa-adv-05) — fully overlapping distributions.
LIVE_NO_ANSWER_SCORES = {
    "qa-na-c-11": 0.3885,
    "qa-na-c-03": 0.012,
}
LIVE_ANSWERABLE_SCORES = {
    "qa-adv-05": 0.00142,
    "qa-va-03": 0.00186,
    "qa-va-05": 0.0037,
    "qa-sev-02": 0.031,
    "qa-sp-01": 0.65,
}

SEPARABLE_NO_ANSWER = {"syn-na-01": 0.02, "syn-na-02": 0.05}
SEPARABLE_ANSWERABLE = {"syn-ok-01": 0.4, "syn-ok-02": 0.9}


# ---------------------------------------------------------------------------
# RetrievalConfig: None is the legal "pre-filter disabled" state.
# ---------------------------------------------------------------------------


class TestConfigAcceptsDisabledPrefilter:
    def test_none_threshold_constructs(self) -> None:
        config = RetrievalConfig(refusal_threshold=None, corpus_coverage=("warming",))
        assert config.refusal_threshold is None

    def test_non_finite_thresholds_still_refuse_loudly(self) -> None:
        """Finding #172 is not weakened: None means DISABLED, deliberately;
        NaN/inf/bool still mean a broken config and refuse at construction."""
        for bad in (float("nan"), float("inf"), float("-inf"), True):
            with pytest.raises(RetrievalError):
                RetrievalConfig(refusal_threshold=bad, corpus_coverage=())


# ---------------------------------------------------------------------------
# retrieve() with the pre-filter disabled.
# ---------------------------------------------------------------------------


class TestRetrieveWithPrefilterDisabled:
    def test_low_scores_answer_instead_of_refusing(self) -> None:
        """With the pre-filter OFF, arbitrarily low reranker scores reach
        generation — whose structured decline is now the honest refusal
        path. No score-based HonestRefusal exists in this state."""
        client, model, _chunks = indexed_corpus()
        result = run_retrieve(
            client,
            decision("aurelian basin warming"),
            model=model,
            reranker=HashReranker(),
            cfg=RetrievalConfig(refusal_threshold=None, corpus_coverage=("warming",)),
        )
        assert isinstance(result, RetrievedPassages), (
            "pre-filter disabled: retrieval must hand the passages to "
            "generation, never score-refuse"
        )
        assert result.passages, "candidates existed; the top-8 must be served"

    def test_every_passage_clears_and_no_partial_support(self) -> None:
        """clears_threshold/partial_support are pre-filter bookkeeping;
        with no floor there is no straddling: every served passage clears
        and partial_support is False (generation's Rule-5 honesty about
        coverage is prompt-level, not threshold-level)."""
        client, model, _chunks = indexed_corpus()
        result = run_retrieve(
            client,
            decision("aurelian basin warming"),
            model=model,
            reranker=HashReranker(),
            cfg=RetrievalConfig(refusal_threshold=None, corpus_coverage=("warming",)),
        )
        assert isinstance(result, RetrievedPassages)
        assert all(passage.clears_threshold for passage in result.passages)
        assert result.partial_support is False

    def test_zero_candidates_still_refuse_honestly(self) -> None:
        """A store that returns NOTHING for the route's include list has
        nothing to generate from: even with the pre-filter disabled the
        result is an HonestRefusal, never an empty generation call."""
        from tests._indexing_fixtures import HashEmbeddingModel, build, fixture_corpus, fresh_client

        client = fresh_client()
        model = HashEmbeddingModel()
        chunks, records = fixture_corpus()
        voices_only = [chunk for chunk in chunks if chunk.source_type == "voices"]
        assert voices_only, "the fixture corpus must carry its voices doc"
        build(
            client,
            voices_only,
            {"syn-idx-voices": records["syn-idx-voices"]},
            model=model,
        )
        # A science (non-voices) route over a voices-only index: the
        # include list matches nothing, so zero candidates come back.
        result = run_retrieve(
            client,
            decision("aurelian basin warming"),
            model=model,
            reranker=HashReranker(),
            cfg=RetrievalConfig(refusal_threshold=None, corpus_coverage=("warming",)),
        )
        assert isinstance(result, HonestRefusal), (
            "zero candidates is not answerable regardless of the pre-filter "
            "state — the honest refusal template applies"
        )

    def test_enabled_prefilter_still_refuses_below_floor(self) -> None:
        """The pre-filter is DEMOTED, not deleted: with a floor configured,
        a top score below it still refuses without a generation call —
        that is the whole point of keeping it (spend)."""
        client, model, _chunks = indexed_corpus()
        result = run_retrieve(
            client,
            decision("aurelian basin warming"),
            model=model,
            reranker=HashReranker(),  # scores land far below 0.999...
            cfg=RetrievalConfig(refusal_threshold=0.9999999, corpus_coverage=("warming",)),
        )
        assert isinstance(result, HonestRefusal)


# ---------------------------------------------------------------------------
# calibrate_prefilter_floor: conservative, separability-free, #177-strict
# on genuinely degenerate inputs.
# ---------------------------------------------------------------------------


class TestCalibratePrefilterFloor:
    def test_separable_inputs_yield_the_conservative_floor(self) -> None:
        calibration = calibrate_prefilter_floor(SEPARABLE_NO_ANSWER, SEPARABLE_ANSWERABLE)
        assert calibration.enabled is True
        assert calibration.reason is None
        assert calibration.threshold == pytest.approx(min(SEPARABLE_ANSWERABLE.values()) / 2)
        assert calibration.separable is True

    def test_floor_sits_strictly_below_every_answerable_score(self) -> None:
        """Zero false pre-filter refusals BY CONSTRUCTION: the floor is
        derived from the answerable side alone and every answerable
        calibration item clears it."""
        calibration = calibrate_prefilter_floor(LIVE_NO_ANSWER_SCORES, LIVE_ANSWERABLE_SCORES)
        assert calibration.threshold is not None
        assert 0.0 < calibration.threshold < min(LIVE_ANSWERABLE_SCORES.values())

    def test_live_inseparable_geometry_calibrates_instead_of_bricking(self) -> None:
        """THE issue-#313 pin: the exact live geometry (no-answer 0.3885
        over answerable 0.00142) — which REFUSED the v1 calibration and
        blocked the release — now yields an ENABLED conservative floor
        with the inseparability recorded as a diagnostic."""
        calibration = calibrate_prefilter_floor(LIVE_NO_ANSWER_SCORES, LIVE_ANSWERABLE_SCORES)
        assert calibration.enabled is True
        assert calibration.threshold == pytest.approx(0.00142 / 2)
        assert calibration.separable is False, (
            "inseparability is recorded honestly — it is expected on real "
            "geometry and gates nothing"
        )

    def test_calibration_item_ids_recorded_for_the_6_1_split_check(self) -> None:
        calibration = calibrate_prefilter_floor(SEPARABLE_NO_ANSWER, SEPARABLE_ANSWERABLE)
        assert set(calibration.calibration_item_ids) == (
            set(SEPARABLE_NO_ANSWER) | set(SEPARABLE_ANSWERABLE)
        )

    def test_deterministic(self) -> None:
        first = calibrate_prefilter_floor(LIVE_NO_ANSWER_SCORES, LIVE_ANSWERABLE_SCORES)
        second = calibrate_prefilter_floor(
            dict(LIVE_NO_ANSWER_SCORES), dict(LIVE_ANSWERABLE_SCORES)
        )
        assert first == second

    def test_degenerate_inputs_still_refuse_loudly(self) -> None:
        """Finding #177's discipline survives the demotion for GARBAGE
        (not for honest overlap): empty maps, out-of-scale scores and
        shared ids are calibration bugs, refused with a typed error."""
        with pytest.raises(RetrievalError):
            calibrate_prefilter_floor({}, SEPARABLE_ANSWERABLE)
        with pytest.raises(RetrievalError):
            calibrate_prefilter_floor(SEPARABLE_NO_ANSWER, {})
        with pytest.raises(RetrievalError):
            calibrate_prefilter_floor(
                SEPARABLE_NO_ANSWER, {**SEPARABLE_ANSWERABLE, "syn-bad": float("nan")}
            )
        with pytest.raises(RetrievalError):
            calibrate_prefilter_floor(SEPARABLE_NO_ANSWER, {**SEPARABLE_ANSWERABLE, "syn-bad": 1.5})
        with pytest.raises(RetrievalError):
            calibrate_prefilter_floor(
                {**SEPARABLE_NO_ANSWER, "shared-id": 0.03},
                {**SEPARABLE_ANSWERABLE, "shared-id": 0.4},
            )


# ---------------------------------------------------------------------------
# The schema-v2 artifact: round-trips, and DEGRADES instead of blocking.
# ---------------------------------------------------------------------------


class TestPrefilterArtifact:
    def test_schema_version_is_2_and_distinct_from_v1(self) -> None:
        from rag.retrieval import THRESHOLD_ARTIFACT_SCHEMA_VERSION

        assert PREFILTER_ARTIFACT_SCHEMA_VERSION == 2
        assert PREFILTER_ARTIFACT_SCHEMA_VERSION != THRESHOLD_ARTIFACT_SCHEMA_VERSION

    def test_enabled_calibration_round_trips(self, tmp_path: Path) -> None:
        calibration = calibrate_prefilter_floor(LIVE_NO_ANSWER_SCORES, LIVE_ANSWERABLE_SCORES)
        path = tmp_path / "prefilter.json"
        save_prefilter_artifact(calibration, path)
        assert load_prefilter_artifact(path) == calibration

    def test_disabled_calibration_round_trips(self, tmp_path: Path) -> None:
        """A deliberately-disabled pre-filter is a committable, honest
        artifact (the release record of a corpus that ships without one)."""
        disabled = PrefilterCalibration(
            threshold=None,
            enabled=False,
            reason="synthetic: disabled for this corpus release",
        )
        path = tmp_path / "prefilter-disabled.json"
        save_prefilter_artifact(disabled, path)
        loaded = load_prefilter_artifact(path)
        assert loaded.enabled is False
        assert loaded.threshold is None

    def test_missing_artifact_degrades_to_disabled_never_raises(self, tmp_path: Path) -> None:
        """#216 interplay, the load-side half: a missing artifact is a
        DISABLED pre-filter with a reason naming the path — never an
        exception, never a blocked deploy."""
        missing = tmp_path / "nowhere" / "prefilter.json"
        loaded = load_prefilter_artifact(missing)
        assert loaded.enabled is False
        assert loaded.threshold is None
        assert loaded.reason and str(missing) in loaded.reason

    def test_malformed_artifact_degrades_with_a_reason(self, tmp_path: Path) -> None:
        cases = {
            "not-json.json": "{ not json",
            "wrong-schema.json": json.dumps({"schema_version": 1, "threshold": 0.5}),
            "nan-threshold.json": (
                '{"schema_version": 2, "enabled": true, "threshold": NaN, '
                '"calibration_item_ids": []}'
            ),
            "out-of-scale.json": json.dumps(
                {
                    "schema_version": 2,
                    "enabled": True,
                    "threshold": 2.0,
                    "calibration_item_ids": [],
                }
            ),
        }
        for name, content in cases.items():
            path = tmp_path / name
            path.write_text(content)
            loaded = load_prefilter_artifact(path)
            assert loaded.enabled is False, f"{name}: a malformed artifact must degrade, not enable"
            assert loaded.threshold is None, f"{name}: no live threshold from a malformed artifact"
            assert loaded.reason, f"{name}: the degradation must carry its reason"
            if loaded.threshold is not None:  # defensive: never a NaN either
                assert math.isfinite(loaded.threshold)
