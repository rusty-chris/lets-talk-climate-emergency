"""Release-run-5 RED (round 2) — PR #353 smoke regression, root-caused.

Committing ``evals/RESULTS.md`` broke EVERY compose smoke stack, not just
replay: the api image deliberately excludes the transparency build's
sources of truth from its layers (.dockerignore keeps ``corpus/``,
``datasets/``, ``voices/`` and ``letters/`` out — fetched Tier bytes and
private records never bake into an image), while ``evals/`` IS in the
image. ``service.main._build_transparency_pages`` keyed the real-build
attempt on RESULTS.md presence ALONE, so the paused and healthcheck
stacks (provider=anthropic) crashed boot with::

    TransparencyBuildError: transparency build failed loading a source of
    truth: [Errno 2] No such file or directory: '.../corpus/manifest.yaml'

(reproduced unit-side without Docker). The first fix (141391c) scoped
only the replay provider and merely moved the failure to the paused
stack.

The honest design this suite pins (ratified pattern: #249's two-sided
boundary):

- **Builder degrade:** the real build is attempted only when the FULL
  input set exists — RESULTS.md + corpus manifest + datasets manifest +
  voices.yaml + letters sending-record. Any absent input → ``None`` (the
  honestly-marked placeholders), NO exception, NO partial read. This is
  what every dev/compose stack (healthcheck, paused, replay) gets inside
  the image, whatever the provider.
- **The replay stack stays placeholder-serving by construction** (the
  ratified #249 wording) even where the full input set exists.
- **The live-boot gate is STRENGTHENED, not weakened:** a live ingested
  non-replay deploy now hard-requires the FULL input set at boot
  (``validate_deployment_artifacts`` names every missing source path),
  so the builder degrade can never silently placeholder a public deploy.
- Live provider + full input set → the real build still runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import service.main
import service.transparency
from service.app import ServiceStartupError
from service.config import PROVIDER_ANTHROPIC, PROVIDER_REPLAY, ServiceConfig

LIVE_INDEX_VERSION = "corpus-2026-09-11"

#: The compose smoke environments (provider, daily budget) — healthcheck
#: (defaults), paused (budget 0), replay. Inside the api image all four
#: transparency sources are absent whatever the stack.
SMOKE_ENVS = (
    pytest.param(PROVIDER_ANTHROPIC, 5.0, id="healthcheck"),
    pytest.param(PROVIDER_ANTHROPIC, 0.0, id="paused"),
    pytest.param(PROVIDER_REPLAY, 5.0, id="replay"),
)


def _config(provider: str, daily_budget: float = 5.0) -> ServiceConfig:
    return ServiceConfig(
        daily_budget_usd=daily_budget,
        opus_subcap_usd=min(1.0, daily_budget),
        corpus_version="dev-corpus-0000",
        corpus_vintage="2026-08-01",
        site_url="http://localhost:8000",
        qdrant_url="http://qdrant.invalid:6333",
        starter_cache_dir="/nonexistent/starter",
        log_dir="/nonexistent/logs",
        provider=provider,
        replay_dir="/nonexistent/replay" if provider == PROVIDER_REPLAY else "",
    )


def _point_sources_at(monkeypatch, root: Path) -> dict[str, Path]:
    """Point every transparency source constant below ``root`` (the
    image-layer simulation: absent unless the test creates them)."""
    paths = {
        "_EVAL_RESULTS_PATH": root / "evals" / "RESULTS.md",
        "_CORPUS_MANIFEST_PATH": root / "corpus" / "manifest.yaml",
        "_DATASETS_MANIFEST_PATH": root / "datasets" / "manifest.yaml",
    }
    for name, path in paths.items():
        monkeypatch.setattr(service.main, name, path)
    transparency_paths = {
        "VOICES_CONTENT_PATH": root / "voices" / "voices.yaml",
        "PERMISSION_LETTERS_RECORD_PATH": root / "letters" / "SENDING-RECORD.md",
    }
    for name, path in transparency_paths.items():
        monkeypatch.setattr(service.transparency, name, path)
    return {**paths, **transparency_paths}


def _materialise_full_input_set(root: Path) -> None:
    for rel in (
        "evals/RESULTS.md",
        "corpus/manifest.yaml",
        "datasets/manifest.yaml",
        "voices/voices.yaml",
        "letters/SENDING-RECORD.md",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder-for-shape\n", encoding="utf-8")


class TestComposeImageBootsWithResultsCommitted:
    """The #353 regression, parametrised over the three smoke stacks: the
    image carries RESULTS.md but none of the four sources — boot must get
    the placeholders (None), never a TransparencyBuildError."""

    @pytest.mark.parametrize(("provider", "budget"), SMOKE_ENVS)
    def test_results_present_sources_absent_serves_placeholders(
        self, tmp_path: Path, monkeypatch, provider: str, budget: float
    ) -> None:
        paths = _point_sources_at(monkeypatch, tmp_path)
        results = paths["_EVAL_RESULTS_PATH"]
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text("Release verdict: PASSED\n", encoding="utf-8")
        # No other source exists — exactly the api image's layer contents.
        assert service.main._build_transparency_pages(_config(provider, budget)) is None

    @pytest.mark.parametrize(("provider", "budget"), SMOKE_ENVS)
    def test_no_results_still_serves_placeholders(
        self, tmp_path: Path, monkeypatch, provider: str, budget: float
    ) -> None:
        _point_sources_at(monkeypatch, tmp_path)
        assert service.main._build_transparency_pages(_config(provider, budget)) is None


class TestReplayStaysPlaceholderByConstruction:
    def test_replay_never_builds_even_with_the_full_input_set(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The ratified #249 wording: the replay stack is by construction
        not a public deploy — placeholders even where a full checkout's
        input set exists."""
        _point_sources_at(monkeypatch, tmp_path)
        _materialise_full_input_set(tmp_path)

        def _explode(**_kwargs):  # pragma: no cover - the pin is "not called"
            raise AssertionError("replay stack must never attempt the real build")

        monkeypatch.setattr(service.transparency, "build_transparency_pages", _explode)
        assert service.main._build_transparency_pages(_config(PROVIDER_REPLAY)) is None


class TestLiveProviderStillBuildsOnFullInputSet:
    def test_full_input_set_builds_real_pages(self, tmp_path: Path, monkeypatch) -> None:
        paths = _point_sources_at(monkeypatch, tmp_path)
        _materialise_full_input_set(tmp_path)
        sentinel = object()
        calls: list[dict] = []

        def _fake_build(**kwargs):
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(service.transparency, "build_transparency_pages", _fake_build)
        built = service.main._build_transparency_pages(_config(PROVIDER_ANTHROPIC))
        assert built is sentinel
        assert calls and calls[0]["eval_results_path"] == paths["_EVAL_RESULTS_PATH"]


class TestLiveBootGateRequiresFullInputSet:
    """#249 STRENGTHENED: a live ingested non-replay deploy hard-requires
    the full transparency input set — the builder degrade can never
    silently placeholder a public deploy."""

    def _validate(self, tmp_path: Path):
        env = {
            service.main.ENV_DATASET_MANIFEST: str(tmp_path / "dm.yaml"),
            service.main.ENV_CHART_PACK_DIR: str(tmp_path / "pack"),
        }
        (tmp_path / "dm.yaml").write_text("datasets: []\n", encoding="utf-8")
        (tmp_path / "pack").mkdir(exist_ok=True)
        return env

    @pytest.mark.parametrize(
        "missing",
        [
            "corpus/manifest.yaml",
            "datasets/manifest.yaml",
            "voices/voices.yaml",
            "letters/SENDING-RECORD.md",
        ],
    )
    def test_live_deploy_missing_any_source_refuses_naming_it(
        self, tmp_path: Path, monkeypatch, missing: str
    ) -> None:
        _point_sources_at(monkeypatch, tmp_path)
        _materialise_full_input_set(tmp_path)
        (tmp_path / missing).unlink()
        env = self._validate(tmp_path)
        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.validate_deployment_artifacts(
                env,
                index_corpus_version=LIVE_INDEX_VERSION,
                stored_chart_specs=False,
                eval_results_path=tmp_path / "evals" / "RESULTS.md",
            )
        assert str(tmp_path / missing) in str(excinfo.value)

    def test_live_deploy_with_full_set_passes(self, tmp_path: Path, monkeypatch) -> None:
        _point_sources_at(monkeypatch, tmp_path)
        _materialise_full_input_set(tmp_path)
        env = self._validate(tmp_path)
        service.main.validate_deployment_artifacts(
            env,
            index_corpus_version=LIVE_INDEX_VERSION,
            stored_chart_specs=False,
            eval_results_path=tmp_path / "evals" / "RESULTS.md",
        )

    def test_replay_stack_still_exempt(self, tmp_path: Path, monkeypatch) -> None:
        _point_sources_at(monkeypatch, tmp_path)
        env = self._validate(tmp_path)
        env[service.main.ENV_PROVIDER] = PROVIDER_REPLAY
        service.main.validate_deployment_artifacts(
            env,
            index_corpus_version=LIVE_INDEX_VERSION,
            stored_chart_specs=False,
            eval_results_path=tmp_path / "evals" / "RESULTS.md",
        )
