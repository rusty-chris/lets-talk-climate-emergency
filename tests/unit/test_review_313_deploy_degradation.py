"""Issue #313 red phase (Fable): a missing pre-filter calibration no
longer blocks a live deploy (#216 interplay).

Under the retired design the threshold artifact was the refusal ARBITER:
booting without it would have served every first query a 500, so #216
rightly made ``validate_deployment_artifacts`` refuse at boot. Under
#313 the artifact only feeds the cost-saving PRE-FILTER, and the
authoritative refusal signal lives in generation — so its absence
DEGRADES (pre-filter off, warning at the retrieval seam; see
``load_prefilter_artifact``'s disabled-with-reason contract) and must
not brick the deploy. Pinned here:

- ``validate_deployment_artifacts`` no longer lists the threshold
  artifact as a required live-deploy artifact: a live (index-recorded)
  env WITHOUT ``CLIMATE_CHAT_THRESHOLD_ARTIFACT`` validates cleanly
  when the still-required artifacts are present.
- The OTHER boot requirements are untouched: the dataset manifest,
  chart pack and published eval results still refuse loudly, named.

No test here touches the network (pure env-mapping validation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service.app import ServiceStartupError
from service.main import (
    ENV_CHART_PACK_DIR,
    ENV_DATASET_MANIFEST,
    ENV_THRESHOLD_ARTIFACT,
    validate_deployment_artifacts,
)
from tests._service_fixtures import full_deploy_env


def _live_env_without_threshold(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A live-deploy env carrying every still-required artifact but NO
    threshold/pre-filter artifact variable at all."""
    env = full_deploy_env(tmp_path)
    manifest = tmp_path / "datasets-manifest.yaml"
    manifest.write_text("synthetic: manifest\n")
    pack_dir = tmp_path / "chart-pack"
    pack_dir.mkdir()
    results = tmp_path / "RESULTS.md"
    results.write_text("# synthetic published eval results\n")
    env[ENV_DATASET_MANIFEST] = str(manifest)
    env[ENV_CHART_PACK_DIR] = str(pack_dir)
    env.pop(ENV_THRESHOLD_ARTIFACT, None)
    return env, results


class TestThresholdArtifactIsNoLongerRequired:
    def test_live_deploy_boots_without_the_threshold_artifact(self, tmp_path) -> None:
        """THE #313 pin: an ingested (live) deploy with no
        CLIMATE_CHAT_THRESHOLD_ARTIFACT validates cleanly — the
        pre-filter degrades to OFF at the retrieval seam instead of the
        deploy refusing to exist."""
        env, results = _live_env_without_threshold(tmp_path)
        validate_deployment_artifacts(
            env,
            index_corpus_version="corpus-2026-08-01",
            stored_chart_specs=False,
            eval_results_path=results,
        )  # must not raise

    def test_a_dangling_threshold_path_does_not_block_boot_either(self, tmp_path) -> None:
        """A configured-but-missing artifact file is the 'failed
        calibration artifact' case: it degrades at load time
        (load_prefilter_artifact → disabled + reason), so boot validation
        must not treat it as an offender."""
        env, results = _live_env_without_threshold(tmp_path)
        env[ENV_THRESHOLD_ARTIFACT] = str(tmp_path / "does-not-exist.json")
        validate_deployment_artifacts(
            env,
            index_corpus_version="corpus-2026-08-01",
            stored_chart_specs=False,
            eval_results_path=results,
        )  # must not raise


class TestOtherBootRequirementsUnchanged:
    def test_missing_render_inputs_still_refuse_loudly(self, tmp_path) -> None:
        """Green guard: the #214/#216 discipline survives for the
        artifacts that DO still gate first-query servability."""
        env, results = _live_env_without_threshold(tmp_path)
        del env[ENV_DATASET_MANIFEST]
        del env[ENV_CHART_PACK_DIR]
        with pytest.raises(ServiceStartupError) as excinfo:
            validate_deployment_artifacts(
                env,
                index_corpus_version="corpus-2026-08-01",
                stored_chart_specs=False,
                eval_results_path=results,
            )
        message = str(excinfo.value)
        assert ENV_DATASET_MANIFEST in message
        assert ENV_CHART_PACK_DIR in message
        assert ENV_THRESHOLD_ARTIFACT not in message, (
            "the retired requirement must not resurface in the refusal"
        )

    def test_missing_published_results_still_refuse_loudly(self, tmp_path) -> None:
        """Green guard: #249 is untouched — a live deploy without the
        published eval results still refuses, naming the path."""
        env, _results = _live_env_without_threshold(tmp_path)
        missing_results = tmp_path / "missing-RESULTS.md"
        with pytest.raises(ServiceStartupError) as excinfo:
            validate_deployment_artifacts(
                env,
                index_corpus_version="corpus-2026-08-01",
                stored_chart_specs=False,
                eval_results_path=missing_results,
            )
        assert str(missing_results) in str(excinfo.value)
