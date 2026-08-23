"""Typed environment configuration for the service (issue #22).

Pins ``service.config.load_service_config``: critical variables refuse
typed and all-at-once when missing; budgets are validated numerically;
the API key is presence-checked but never stored or echoed; and the
startup corpus/index version check fails loudly on mismatch
(``service.app.create_app``).
"""

from __future__ import annotations

import pytest

from service.config import (
    CRITICAL_ENV_VARS,
    ENV_ANTHROPIC_API_KEY,
    ENV_BEST_MODE,
    ENV_CORPUS_VERSION,
    ENV_DAILY_BUDGET_USD,
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
