"""Service configuration from the environment (issue #22, DESIGN §9).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_service_config.py`` pins the contract.

The composition root (``service.main``) is the ONLY place environment
variables are read; everything below it takes a typed
:class:`ServiceConfig`. Missing or malformed critical configuration is a
typed refusal (:class:`ServiceConfigError`) naming EVERY offending
variable at once — a public fail-closed service never boots on guessed
budgets, a guessed corpus version, or an absent API key, and an operator
fixing config must see the whole list, not one variable per crash.

Secrets policy: ``ANTHROPIC_API_KEY`` is validated for presence here but
NEVER stored on :class:`ServiceConfig` — the config object gets logged,
repr'd and handed around; the key goes straight from the environment to
the transport adapter in the composition root.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "ENV_DAILY_BUDGET_USD",
    "ENV_OPUS_SUBCAP_USD",
    "ENV_CORPUS_VERSION",
    "ENV_CORPUS_VINTAGE",
    "ENV_SITE_URL",
    "ENV_QDRANT_URL",
    "ENV_STARTER_CACHE_DIR",
    "ENV_LOG_DIR",
    "ENV_ANTHROPIC_API_KEY",
    "ENV_COLLECTION_NAME",
    "ENV_RATE_LIMIT_PER_MINUTE",
    "ENV_BEST_MODE",
    "ENV_TRUSTED_PROXY",
    "ENV_PROVIDER",
    "ENV_REPLAY_DIR",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_REPLAY",
    "ALLOWED_PROVIDERS",
    "CRITICAL_ENV_VARS",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_RATE_LIMIT_PER_MINUTE",
    "ServiceConfig",
    "ServiceConfigError",
    "load_service_config",
]

#: Critical variables — no defaults, typed refusal when absent/malformed.
ENV_DAILY_BUDGET_USD = "CLIMATE_CHAT_DAILY_BUDGET_USD"
ENV_OPUS_SUBCAP_USD = "CLIMATE_CHAT_OPUS_SUBCAP_USD"
ENV_CORPUS_VERSION = "CLIMATE_CHAT_CORPUS_VERSION"
ENV_CORPUS_VINTAGE = "CLIMATE_CHAT_CORPUS_VINTAGE"
ENV_SITE_URL = "CLIMATE_CHAT_SITE_URL"
ENV_QDRANT_URL = "CLIMATE_CHAT_QDRANT_URL"
ENV_STARTER_CACHE_DIR = "CLIMATE_CHAT_STARTER_CACHE_DIR"
ENV_LOG_DIR = "CLIMATE_CHAT_LOG_DIR"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

#: Optional variables with safe defaults.
ENV_COLLECTION_NAME = "CLIMATE_CHAT_COLLECTION"
ENV_RATE_LIMIT_PER_MINUTE = "CLIMATE_CHAT_RATE_LIMIT_PER_MINUTE"
ENV_BEST_MODE = "CLIMATE_CHAT_BEST_MODE"
ENV_TRUSTED_PROXY = "CLIMATE_CHAT_TRUSTED_PROXY"

#: The composition-root provider switch (review finding #231): the default
#: live Anthropic transport, or the deterministic ReplayAdapter over
#: checked-in fixtures for the live-path smoke tier. Replay is NEVER
#: selected implicitly and a typo never falls back to the live adapter.
ENV_PROVIDER = "CLIMATE_CHAT_PROVIDER"
ENV_REPLAY_DIR = "CLIMATE_CHAT_REPLAY_DIR"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_REPLAY = "replay"
ALLOWED_PROVIDERS: tuple[str, ...] = (PROVIDER_ANTHROPIC, PROVIDER_REPLAY)

#: The full critical set, in one place so the refusal test and the
#: deployment runbook can enumerate it without drift.
CRITICAL_ENV_VARS: tuple[str, ...] = (
    ENV_DAILY_BUDGET_USD,
    ENV_OPUS_SUBCAP_USD,
    ENV_CORPUS_VERSION,
    ENV_CORPUS_VINTAGE,
    ENV_SITE_URL,
    ENV_QDRANT_URL,
    ENV_STARTER_CACHE_DIR,
    ENV_LOG_DIR,
    ENV_ANTHROPIC_API_KEY,
)

DEFAULT_COLLECTION_NAME = "climate_chunks"
DEFAULT_RATE_LIMIT_PER_MINUTE = 10


class ServiceConfigError(Exception):
    """Critical service configuration is missing or malformed.

    ``missing`` lists every absent critical variable; ``invalid`` lists
    every present-but-malformed one (non-numeric budgets, a sub-cap
    exceeding the daily cap, …). The message names them all — one crash,
    the whole fix list.
    """

    def __init__(
        self,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        invalid: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.missing = missing
        self.invalid = invalid


@dataclass(frozen=True)
class ServiceConfig:
    """Typed service configuration. Carries NO secrets (see module doc)."""

    daily_budget_usd: float
    opus_subcap_usd: float
    corpus_version: str
    corpus_vintage: str
    site_url: str
    qdrant_url: str
    starter_cache_dir: str
    log_dir: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    best_mode_enabled: bool = False
    trusted_proxy: bool = False
    provider: str = PROVIDER_ANTHROPIC
    replay_dir: str = ""


def load_service_config(env: Mapping[str, str]) -> ServiceConfig:
    """Validate ``env`` (typically ``os.environ``) into a ServiceConfig.

    Contract (pinned by ``tests/unit/test_service_config.py``):

    - every :data:`CRITICAL_ENV_VARS` entry must be present and
      non-empty; ANY absence raises :class:`ServiceConfigError` whose
      ``missing`` names every absent variable at once;
    - budgets must parse as finite non-negative floats, and the Opus
      sub-cap must not exceed the daily cap (a sub-cap above the cap is
      a lie about what Opus can spend); violations raise with the
      offending names on ``invalid``;
    - ``ANTHROPIC_API_KEY`` is presence-checked only — its value never
      appears on the returned config NOR in the error message;
    - optional variables apply their defaults; boolean flags accept
      "1"/"true" (case-insensitive) as true, "0"/"false"/absent as
      false; anything else is ``invalid``.
    """
    invalid: list[str] = []

    # The provider switch is resolved first: it decides which critical
    # variables apply. Replay makes zero live calls by construction, so it
    # boots key-less; live keeps the ANTHROPIC_API_KEY presence check. A
    # replay provider REQUIRES its fixtures dir; an unknown provider is a
    # typed refusal (a typo never silently falls back to the live adapter).
    provider = str(env.get(ENV_PROVIDER, "") or PROVIDER_ANTHROPIC).strip() or PROVIDER_ANTHROPIC
    replay_dir = str(env.get(ENV_REPLAY_DIR, "")).strip()
    if provider not in ALLOWED_PROVIDERS:
        invalid.append(ENV_PROVIDER)
        # Fall back to the strictest (live) critical set for the rest of the
        # checks; the invalid provider already fails the load.
        provider = PROVIDER_ANTHROPIC

    # Presence first: every absent/blank critical variable is named at
    # once, so an operator fixing config sees the whole list, not one
    # crash per variable. Replay does not require the API key.
    critical = tuple(
        name
        for name in CRITICAL_ENV_VARS
        if not (provider == PROVIDER_REPLAY and name == ENV_ANTHROPIC_API_KEY)
    )
    missing = tuple(name for name in critical if not str(env.get(name, "")).strip())
    if provider == PROVIDER_REPLAY and not replay_dir:
        missing = missing + (ENV_REPLAY_DIR,)

    daily_budget = _parse_non_negative_float(env, ENV_DAILY_BUDGET_USD, invalid)
    opus_subcap = _parse_non_negative_float(env, ENV_OPUS_SUBCAP_USD, invalid)
    best_mode = _parse_bool(env, ENV_BEST_MODE, invalid)
    trusted_proxy = _parse_bool(env, ENV_TRUSTED_PROXY, invalid)
    rate_limit = _parse_positive_int(
        env, ENV_RATE_LIMIT_PER_MINUTE, DEFAULT_RATE_LIMIT_PER_MINUTE, invalid
    )

    # A sub-cap above the daily cap is a lie about what Opus can spend —
    # only checkable once both parsed cleanly and both are present.
    if (
        daily_budget is not None
        and opus_subcap is not None
        and ENV_OPUS_SUBCAP_USD not in missing
        and ENV_DAILY_BUDGET_USD not in missing
        and opus_subcap > daily_budget
    ):
        invalid.append(ENV_OPUS_SUBCAP_USD)

    if missing or invalid:
        # The API key value NEVER appears in the message (secrets policy):
        # names only.
        parts: list[str] = []
        if missing:
            parts.append(f"missing critical configuration: {', '.join(missing)}")
        if invalid:
            parts.append(f"invalid configuration: {', '.join(dict.fromkeys(invalid))}")
        raise ServiceConfigError(
            "service configuration refused — " + "; ".join(parts),
            missing=missing,
            invalid=tuple(dict.fromkeys(invalid)),
        )

    # ANTHROPIC_API_KEY is presence-checked only (above): it never lands
    # on the returned config — the composition root reads it straight from
    # the environment and hands it to the transport adapter.
    return ServiceConfig(
        daily_budget_usd=float(daily_budget),
        opus_subcap_usd=float(opus_subcap),
        corpus_version=env[ENV_CORPUS_VERSION].strip(),
        corpus_vintage=env[ENV_CORPUS_VINTAGE].strip(),
        site_url=env[ENV_SITE_URL].strip(),
        qdrant_url=env[ENV_QDRANT_URL].strip(),
        starter_cache_dir=env[ENV_STARTER_CACHE_DIR].strip(),
        log_dir=env[ENV_LOG_DIR].strip(),
        collection_name=str(env.get(ENV_COLLECTION_NAME, "")).strip() or DEFAULT_COLLECTION_NAME,
        rate_limit_per_minute=rate_limit,
        best_mode_enabled=bool(best_mode),
        trusted_proxy=bool(trusted_proxy),
        provider=provider,
        replay_dir=replay_dir,
    )


def _parse_non_negative_float(
    env: Mapping[str, str], name: str, invalid: list[str]
) -> float | None:
    """Parse a finite non-negative float; record ``name`` as invalid on failure."""
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return None  # absence handled by the missing-check
    try:
        value = float(raw)
    except (TypeError, ValueError):
        invalid.append(name)
        return None
    if not math.isfinite(value) or value < 0:
        invalid.append(name)
        return None
    return value


def _parse_bool(env: Mapping[str, str], name: str, invalid: list[str]) -> bool | None:
    """Strict boolean flag: 1/true → True, 0/false/absent → False, else invalid."""
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return False
    normalised = str(raw).strip().lower()
    if normalised in ("1", "true"):
        return True
    if normalised in ("0", "false"):
        return False
    invalid.append(name)
    return None


def _parse_positive_int(env: Mapping[str, str], name: str, default: int, invalid: list[str]) -> int:
    """Parse a positive int override; absent → ``default``; non-positive → invalid."""
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        invalid.append(name)
        return default
    if value <= 0:
        invalid.append(name)
        return default
    return value
