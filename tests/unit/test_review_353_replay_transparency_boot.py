"""Release-run-5 RED — PR #353 smoke regression: the REPLAY smoke stack
must keep serving the honestly-marked placeholder transparency pages
even now that ``evals/RESULTS.md`` is committed.

``service.main._build_transparency_pages`` keys ONLY on RESULTS.md
presence. That was indistinguishable from the ratified #249 scoping
while the file did not exist — but run 5 published it, so the replay
compose stack (the #231 seeded smoke) takes the REAL-build path inside
the api image for the first time and crashes at boot: the image
deliberately excludes the build's sources of truth (.dockerignore keeps
``corpus/``, ``datasets/``, ``voices/`` and ``letters/`` out of the
layers — fetched Tier bytes and private records never bake into an
image), so ``build_transparency_pages`` raises TransparencyBuildError
and the api never becomes healthy (the #353 CI smoke failure).

The ratified scoping already exists ON THE BOOT GATE:
``validate_deployment_artifacts`` (#249) exempts the explicit
replay-provider stack — "by construction not a public deploy" — and
lets it keep the honestly-marked interim placeholders. This suite pins
the SAME scoping on the page build itself:

- provider=replay → ``_build_transparency_pages`` returns ``None``
  (placeholders) even when RESULTS.md exists, WITHOUT touching the
  manifests/voices/letters sources (absent in the image);
- the live provider's behaviour is UNCHANGED in both directions:
  RESULTS.md present → the real build runs; absent → ``None`` (and the
  #249 boot gate separately refuses a live ingested deploy) — the
  live-boot gate is not weakened.
"""

from __future__ import annotations

from pathlib import Path

import service.main
import service.transparency
from service.config import PROVIDER_ANTHROPIC, PROVIDER_REPLAY, ServiceConfig


def _config(provider: str) -> ServiceConfig:
    return ServiceConfig(
        daily_budget_usd=1.0,
        opus_subcap_usd=0.5,
        corpus_version="corpus-test",
        corpus_vintage="corpus-test-vintage",
        site_url="https://example.invalid",
        qdrant_url="http://qdrant.invalid:6333",
        starter_cache_dir="/nonexistent/starter",
        log_dir="/nonexistent/logs",
        provider=provider,
        replay_dir="/nonexistent/replay" if provider == PROVIDER_REPLAY else "",
    )


class TestReplayStackKeepsPlaceholders:
    def test_replay_provider_returns_none_even_with_results_present(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The #249 replay exemption applies to the PAGE BUILD too: with
        RESULTS.md present, the replay stack still gets ``None`` (the
        honestly-marked placeholders) — never a real-build attempt whose
        sources the smoke image deliberately excludes."""
        results = tmp_path / "RESULTS.md"
        results.write_text("Release verdict: PASSED\n", encoding="utf-8")
        monkeypatch.setattr(service.main, "_EVAL_RESULTS_PATH", results)
        assert service.main._build_transparency_pages(_config(PROVIDER_REPLAY)) is None

    def test_replay_provider_never_reads_the_build_sources(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The replay image has no corpus/datasets/voices/letters in its
        layers — the build must not even be attempted (a call would raise
        against absent sources; the exemption returns before it)."""
        results = tmp_path / "RESULTS.md"
        results.write_text("Release verdict: PASSED\n", encoding="utf-8")
        monkeypatch.setattr(service.main, "_EVAL_RESULTS_PATH", results)

        def _explode(**_kwargs):  # pragma: no cover - the pin is "not called"
            raise AssertionError("build_transparency_pages must not be called for the replay stack")

        monkeypatch.setattr(service.transparency, "build_transparency_pages", _explode)
        assert service.main._build_transparency_pages(_config(PROVIDER_REPLAY)) is None


class TestLiveProviderBehaviourUnchanged:
    def test_live_provider_with_results_builds_the_real_pages(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The live provider still takes the real-build path when the
        published results exist — the #249 live-boot obligation is not
        weakened by the replay scoping."""
        results = tmp_path / "RESULTS.md"
        results.write_text("Release verdict: PASSED\n", encoding="utf-8")
        monkeypatch.setattr(service.main, "_EVAL_RESULTS_PATH", results)
        sentinel = object()
        calls: list[dict] = []

        def _fake_build(**kwargs):
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(service.transparency, "build_transparency_pages", _fake_build)
        built = service.main._build_transparency_pages(_config(PROVIDER_ANTHROPIC))
        assert built is sentinel
        assert calls and calls[0]["eval_results_path"] == results

    def test_live_provider_without_results_still_returns_none(self, monkeypatch) -> None:
        """Absent results → placeholders, exactly as before (the #249
        boot gate is what refuses a live ingested deploy, not this
        builder)."""
        monkeypatch.setattr(service.main, "_EVAL_RESULTS_PATH", Path("/nonexistent/RESULTS.md"))
        assert service.main._build_transparency_pages(_config(PROVIDER_ANTHROPIC)) is None
