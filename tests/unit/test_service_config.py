"""Typed environment configuration for the service (issue #22).

Pins ``service.config.load_service_config``: critical variables refuse
typed and all-at-once when missing; budgets are validated numerically;
the API key is presence-checked but never stored or echoed; and the
startup corpus/index version check fails loudly on mismatch
(``service.app.create_app``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from service.config import (
    CRITICAL_ENV_VARS,
    ENV_ANTHROPIC_API_KEY,
    ENV_BEST_MODE,
    ENV_CORPUS_VERSION,
    ENV_DAILY_BUDGET_USD,
    ENV_LOG_DIR,
    ENV_OPUS_SUBCAP_USD,
    ENV_RATE_LIMIT_PER_MINUTE,
    ServiceConfigError,
    load_service_config,
)


def valid_env() -> dict[str, str]:
    return {
        "CLIMATE_CHAT_DAILY_BUDGET_USD": "1.50",
        "CLIMATE_CHAT_OPUS_SUBCAP_USD": "0.40",
        "CLIMATE_CHAT_CORPUS_VERSION": "corpus-2026-08-01",
        "CLIMATE_CHAT_CORPUS_VINTAGE": "2026-08-01",
        "CLIMATE_CHAT_SITE_URL": "https://climate-chat.example.test",
        "CLIMATE_CHAT_QDRANT_URL": "http://qdrant:6333",
        "CLIMATE_CHAT_STARTER_CACHE_DIR": "/srv/starter-cache",
        "CLIMATE_CHAT_LOG_DIR": "/srv/logs",
        "ANTHROPIC_API_KEY": "synthetic-test-key-not-real",
    }


def test_config_loaded_from_env() -> None:
    config = load_service_config(valid_env())
    assert config.daily_budget_usd == pytest.approx(1.50)
    assert config.opus_subcap_usd == pytest.approx(0.40)
    assert config.corpus_version == "corpus-2026-08-01"
    assert config.corpus_vintage == "2026-08-01"
    assert config.site_url == "https://climate-chat.example.test"
    assert config.qdrant_url == "http://qdrant:6333"
    assert config.starter_cache_dir == "/srv/starter-cache"
    assert config.log_dir == "/srv/logs"
    # Optional flags default off; defaults applied.
    assert config.best_mode_enabled is False
    assert config.trusted_proxy is False
    assert config.rate_limit_per_minute > 0


def test_missing_critical_config_refuses_typed_naming_every_absence() -> None:
    """The typed refusal lists EVERY missing critical variable at once."""
    env = valid_env()
    del env[ENV_DAILY_BUDGET_USD]
    del env[ENV_CORPUS_VERSION]
    del env[ENV_ANTHROPIC_API_KEY]
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert set(excinfo.value.missing) == {
        ENV_DAILY_BUDGET_USD,
        ENV_CORPUS_VERSION,
        ENV_ANTHROPIC_API_KEY,
    }
    message = str(excinfo.value)
    for name in (ENV_DAILY_BUDGET_USD, ENV_CORPUS_VERSION, ENV_ANTHROPIC_API_KEY):
        assert name in message


def test_empty_critical_value_counts_as_missing() -> None:
    env = valid_env()
    env[ENV_CORPUS_VERSION] = "   "
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert ENV_CORPUS_VERSION in excinfo.value.missing


@pytest.mark.parametrize("bad_value", ["not-a-number", "nan", "inf", "-1"])
def test_malformed_budget_is_a_typed_invalid_refusal(bad_value: str) -> None:
    env = valid_env()
    env[ENV_DAILY_BUDGET_USD] = bad_value
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert ENV_DAILY_BUDGET_USD in excinfo.value.invalid


def test_subcap_exceeding_daily_cap_is_invalid() -> None:
    """A sub-cap above the daily cap is a lie about what Opus can spend."""
    env = valid_env()
    env[ENV_DAILY_BUDGET_USD] = "0.50"
    env[ENV_OPUS_SUBCAP_USD] = "0.60"
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert ENV_OPUS_SUBCAP_USD in excinfo.value.invalid


def test_api_key_is_never_stored_on_config_or_echoed_in_errors() -> None:
    """The key is presence-checked only; the config object gets logged."""
    config = load_service_config(valid_env())
    for value in vars(config).values():
        assert "synthetic-test-key-not-real" not in str(value)
    assert "synthetic-test-key-not-real" not in repr(config)

    env = valid_env()
    del env[ENV_OPUS_SUBCAP_USD]
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert "synthetic-test-key-not-real" not in str(excinfo.value)


def test_boolean_flags_parse_strictly() -> None:
    env = valid_env()
    env[ENV_BEST_MODE] = "true"
    assert load_service_config(env).best_mode_enabled is True
    env[ENV_BEST_MODE] = "0"
    assert load_service_config(env).best_mode_enabled is False
    env[ENV_BEST_MODE] = "banana"
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert ENV_BEST_MODE in excinfo.value.invalid


def test_rate_limit_override_parses_as_positive_int() -> None:
    env = valid_env()
    env[ENV_RATE_LIMIT_PER_MINUTE] = "25"
    assert load_service_config(env).rate_limit_per_minute == 25
    env[ENV_RATE_LIMIT_PER_MINUTE] = "0"
    with pytest.raises(ServiceConfigError) as excinfo:
        load_service_config(env)
    assert ENV_RATE_LIMIT_PER_MINUTE in excinfo.value.invalid


def test_critical_env_var_list_matches_the_documented_contract() -> None:
    """The runbook and the refusal share one list — drift fails here."""
    assert set(CRITICAL_ENV_VARS) == set(valid_env())


REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose_api_command() -> list[str]:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return list(compose["services"]["api"]["command"])


def _dockerfile_cmd_line() -> str:
    lines = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    cmd_lines = [line for line in lines if line.startswith("CMD")]
    assert cmd_lines, "the Dockerfile lost its CMD line"
    return cmd_lines[-1]


class TestUvicornInvocation:
    """Issue #212: the server under the app must not log raw client IPs.

    The app hashes IPs (rate limiting) and scrubs the exchange log — but
    uvicorn's DEFAULT access log writes ``client_addr`` (the raw IP) to
    stdout on every request, which the platform log store retains. That
    contradicts DESIGN §9 ("no user identifiers") and the served
    /privacy copy. These are pure text/parse assertions over the
    committed compose file and Dockerfile (no Docker needed); the smoke
    tier verifies the running stack's actual log output.

    DECISION (flagged for ratification): access logging is turned OFF
    entirely (``--no-access-log``) rather than reformatted — the app's
    own hashed rate-limit records are the sanctioned request telemetry,
    and a custom formatter would be one config drift away from leaking
    again. Uvicorn's own proxy-header interpretation is also disabled
    (``--no-proxy-headers``): ``service.rate_limit.resolve_client_ip``
    with ``CLIMATE_CHAT_TRUSTED_PROXY`` is the ONE place X-Forwarded-For
    trust is decided, and uvicorn rewriting ``request.client`` from XFF
    behind the app's back would both double-interpret the header and
    hand the access log a spoofable "client" IP.
    """

    def test_uvicorn_invocation_disables_access_log(self) -> None:
        assert "--no-access-log" in _compose_api_command(), (
            "docker-compose.yml api.command must pass --no-access-log: the "
            "default uvicorn access log writes raw client IPs to the "
            "container logs (DESIGN §9 violation, issue #212)"
        )
        assert "--no-access-log" in _dockerfile_cmd_line(), (
            "the Dockerfile CMD must pass --no-access-log (deploys that run "
            "the image without the compose command override still must not "
            "log raw client IPs)"
        )

    def test_uvicorn_invocation_disables_blanket_proxy_header_trust(self) -> None:
        assert "--no-proxy-headers" in _compose_api_command(), (
            "docker-compose.yml api.command must pass --no-proxy-headers: "
            "X-Forwarded-For trust is decided ONLY by resolve_client_ip / "
            "CLIMATE_CHAT_TRUSTED_PROXY, never by uvicorn's middleware"
        )
        assert "--no-proxy-headers" in _dockerfile_cmd_line()


class TestStartupVersionCheck:
    """create_app consults the recorded index corpus version once."""

    def test_startup_refuses_on_corpus_index_version_mismatch(self, tmp_path) -> None:
        from service.app import ServiceStartupError
        from tests._service_fixtures import make_harness

        with pytest.raises(ServiceStartupError) as excinfo:
            make_harness(tmp_path, index_version="corpus-2025-01-01")
        message = str(excinfo.value)
        assert "corpus-2025-01-01" in message
        assert "corpus-2026-08-01" in message

    def test_startup_proceeds_on_matching_version(self, tmp_path) -> None:
        from tests._service_fixtures import make_harness

        harness = make_harness(tmp_path)  # index_version defaults to the match
        assert harness.app is not None

    def test_absent_index_starts_with_read_only_surfaces(self, tmp_path) -> None:
        """No recorded index (dev compose before ingestion) is not a wrong
        deploy: the app starts and the static surfaces serve."""
        from fastapi.testclient import TestClient

        from tests._service_fixtures import make_harness

        harness = make_harness(tmp_path, index_version=None)
        client = TestClient(harness.app)
        assert client.get("/health").status_code == 200
        assert client.get("/about").status_code == 200


class TestModuleAppFallback:
    """Issue #215: partial/invalid production config must fail loudly.

    service.main._build_module_app swallows ServiceConfigError into a
    health-only fallback whose /health body is byte-identical to a
    healthy deploy — so a deploy with one mistyped CLIMATE_CHAT_* var
    comes up "healthy" while every real route 404s. Pinned fix: the
    fallback is ONLY for the fully-absent env (no CLIMATE_CHAT_* at all
    — the import/unit-test context); a PARTIALLY present or invalid env
    re-raises so the container crashes and the platform reports the
    failed deploy. The ratified /health contract (exactly
    {"status": "ok"} in live AND paused) is untouched — the loudness
    lives at BOOT, not in the health body.
    """

    def test_partial_config_does_not_degrade_to_a_deceptive_health_app(
        self, monkeypatch, tmp_path
    ) -> None:
        """Nine of ten critical vars set (LOG_DIR missing): building the
        module app must raise the name-every-offender refusal, never
        silently become health-only."""
        import service.main
        from tests._service_fixtures import apply_deploy_env, full_deploy_env

        env = full_deploy_env(tmp_path)
        del env[ENV_LOG_DIR]
        apply_deploy_env(monkeypatch, env)

        with pytest.raises(ServiceConfigError) as excinfo:
            service.main._build_module_app()
        assert ENV_LOG_DIR in str(excinfo.value)

    def test_invalid_config_value_crashes_rather_than_degrading(
        self, monkeypatch, tmp_path
    ) -> None:
        """All critical vars present but one malformed: same loud crash
        (an operator typo is a real misconfiguration, not the test/import
        context the fallback exists for)."""
        import service.main
        from tests._service_fixtures import apply_deploy_env, full_deploy_env

        env = full_deploy_env(tmp_path)
        env[ENV_DAILY_BUDGET_USD] = "not-a-number"
        apply_deploy_env(monkeypatch, env)

        with pytest.raises(ServiceConfigError) as excinfo:
            service.main._build_module_app()
        assert ENV_DAILY_BUDGET_USD in str(excinfo.value)

    def test_health_only_fallback_is_only_for_the_fully_absent_env(self, monkeypatch) -> None:
        """With NO CLIMATE_CHAT_* set at all (a bare ANTHROPIC_API_KEY —
        common on dev machines — is not a deploy), the import-safety
        fallback remains: /health answers the ratified body and no real
        route is mounted."""
        import os

        from fastapi.testclient import TestClient

        import service.main

        for name in list(os.environ):
            if name.startswith("CLIMATE_CHAT_"):
                monkeypatch.delenv(name)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key-not-real")

        app = service.main._build_module_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert client.post("/chat", json={"question": "q"}).status_code == 404


class TestLiveGenerationPrereqValidation:
    """Issue #216: a live deploy's first-query dependencies (threshold
    artifact, dataset manifest, chart pack) are validated at STARTUP —
    a boot that would 500 the first real question refuses loudly with
    every missing name instead. Pure over an env mapping (the seam is
    service.main.validate_deployment_artifacts, called by
    create_service_app); no network, no model weights."""

    def _artifacts(self, tmp_path) -> dict[str, str]:
        """Readable stand-ins for the three deployment artifacts."""
        import service.main

        manifest = tmp_path / "dataset-manifest.yaml"
        manifest.write_text("datasets: []\n", encoding="utf-8")
        pack_dir = tmp_path / "chart-pack"
        pack_dir.mkdir(exist_ok=True)
        threshold = tmp_path / "threshold.json"
        threshold.write_text("{}", encoding="utf-8")
        return {
            service.main.ENV_DATASET_MANIFEST: str(manifest),
            service.main.ENV_CHART_PACK_DIR: str(pack_dir),
            service.main.ENV_THRESHOLD_ARTIFACT: str(threshold),
        }

    def test_startup_validates_live_generation_prereqs_when_index_present(self, tmp_path) -> None:
        """An index-recorded (live) deploy missing all three artifact
        variables is refused naming every one of them at once."""
        import service.main
        from service.app import ServiceStartupError

        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.validate_deployment_artifacts(
                {},
                index_corpus_version="corpus-2026-08-01",
                stored_chart_specs=False,
            )
        message = str(excinfo.value)
        assert service.main.ENV_THRESHOLD_ARTIFACT in message
        assert service.main.ENV_DATASET_MANIFEST in message
        assert service.main.ENV_CHART_PACK_DIR in message

    def test_unreadable_artifact_paths_are_refused_by_name(self, tmp_path) -> None:
        """Set-but-dangling paths are as broken as unset ones: the var
        pointing nowhere is named."""
        import service.main
        from service.app import ServiceStartupError

        env = self._artifacts(tmp_path)
        env[service.main.ENV_THRESHOLD_ARTIFACT] = str(tmp_path / "no-such-threshold.json")
        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.validate_deployment_artifacts(
                env,
                index_corpus_version="corpus-2026-08-01",
                stored_chart_specs=False,
            )
        assert service.main.ENV_THRESHOLD_ARTIFACT in str(excinfo.value)

    def test_read_only_start_needs_no_live_artifacts(self, tmp_path) -> None:
        """No recorded index and no stored specs (the dev/read-only
        start): absence of all three artifacts is legitimate."""
        import service.main

        service.main.validate_deployment_artifacts(
            {},
            index_corpus_version=None,
            stored_chart_specs=False,
        )

    def test_complete_live_env_passes(self, tmp_path) -> None:
        """UPDATED for review finding #249 (invariant preserved — a
        COMPLETE live env passes): completeness now includes readable
        published eval results, so this test supplies one explicitly;
        tests/unit/test_service_transparency_boot.py pins the refusal
        when it is missing."""
        import service.main

        results = tmp_path / "RESULTS.md"
        results.write_text("Release verdict: PASSED\n", encoding="utf-8")
        service.main.validate_deployment_artifacts(
            self._artifacts(tmp_path),
            index_corpus_version="corpus-2026-08-01",
            stored_chart_specs=True,
            eval_results_path=results,
        )

    def test_live_boot_without_threshold_artifact_fails_loudly(self, monkeypatch, tmp_path) -> None:
        """The wiring, end to end at this tier: create_service_app over a
        full critical env whose index reader reports a recorded version
        refuses to boot when the threshold artifact is missing — instead
        of today's healthy boot that 500s the first retrieval query."""
        import service.main
        from service.app import ServiceStartupError
        from tests._service_fixtures import (
            CORPUS_VERSION,
            apply_deploy_env,
            full_deploy_env,
        )

        env = full_deploy_env(tmp_path)
        artifacts = self._artifacts(tmp_path)
        del artifacts[service.main.ENV_THRESHOLD_ARTIFACT]
        env.update(artifacts)
        apply_deploy_env(monkeypatch, env)
        # The index reader seam reports a real ingested deploy (matching
        # the configured corpus version) without touching qdrant.
        monkeypatch.setattr(
            service.main,
            "_make_index_version_reader",
            lambda config: lambda: CORPUS_VERSION,
        )

        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.create_service_app()
        assert service.main.ENV_THRESHOLD_ARTIFACT in str(excinfo.value)


class TestProviderSelection:
    """Review finding #231 RED — an explicit composition-root provider switch.

    The smoke tier's live-path coverage (tests/smoke/test_starter_live_replay.py)
    needs the composed api to run on rag.provider.ReplayAdapter over
    checked-in fixtures; today service/main.py can only construct
    AnthropicAdapter. FLAGGED DECISIONS pinned here: env names
    CLIMATE_CHAT_PROVIDER ('anthropic' default / 'replay') and
    CLIMATE_CHAT_REPLAY_DIR; replay mode does not require an
    ANTHROPIC_API_KEY (it makes zero live calls by construction); live
    mode keeps refusing to start without a key exactly as today.
    """

    def test_default_provider_is_anthropic(self) -> None:
        config = load_service_config(valid_env())
        assert hasattr(config, "provider"), (
            "ServiceConfig must carry the provider switch (finding #231)"
        )
        assert config.provider == "anthropic"

    def test_replay_provider_selected_explicitly_with_fixtures_dir(self, tmp_path) -> None:
        env = {
            **valid_env(),
            "CLIMATE_CHAT_PROVIDER": "replay",
            "CLIMATE_CHAT_REPLAY_DIR": str(tmp_path),
        }
        config = load_service_config(env)
        assert config.provider == "replay"
        assert config.replay_dir == str(tmp_path)

    def test_replay_provider_without_a_fixtures_dir_is_refused(self) -> None:
        env = {**valid_env(), "CLIMATE_CHAT_PROVIDER": "replay"}
        with pytest.raises(ServiceConfigError):
            load_service_config(env)

    def test_unknown_provider_is_refused_by_name(self) -> None:
        """Replay is never selected implicitly and typos never fall back
        to the live adapter: an unknown provider value is a typed refusal."""
        env = {**valid_env(), "CLIMATE_CHAT_PROVIDER": "banana"}
        with pytest.raises(ServiceConfigError):
            load_service_config(env)

    def test_replay_provider_does_not_require_an_api_key(self, tmp_path) -> None:
        """Zero live calls by construction: the replay-composed stack must
        boot with no ANTHROPIC_API_KEY at all (CI convention #32) —
        while the live default keeps its presence check unchanged."""
        env = {
            **valid_env(),
            "CLIMATE_CHAT_PROVIDER": "replay",
            "CLIMATE_CHAT_REPLAY_DIR": str(tmp_path),
        }
        del env[ENV_ANTHROPIC_API_KEY]
        config = load_service_config(env)
        assert config.provider == "replay"

    def test_build_service_deps_constructs_the_replay_adapter(self, tmp_path) -> None:
        """The composition root honours the switch: provider 'replay'
        yields a ReplayAdapter over the configured fixtures dir."""
        import service.main as service_main
        from rag.provider import ReplayAdapter

        repo_root = Path(__file__).resolve().parents[2]
        env = {
            **valid_env(),
            "CLIMATE_CHAT_PROVIDER": "replay",
            "CLIMATE_CHAT_REPLAY_DIR": str(tmp_path / "fixtures"),
            "CLIMATE_CHAT_STARTER_CACHE_DIR": str(repo_root / "service" / "dev_starter_cache"),
            "CLIMATE_CHAT_LOG_DIR": str(tmp_path / "logs"),
        }
        (tmp_path / "fixtures").mkdir()
        config = load_service_config(env)
        deps = service_main.build_service_deps(config)
        assert isinstance(deps.adapter, ReplayAdapter)
        assert Path(deps.adapter.fixtures_dir) == tmp_path / "fixtures"
