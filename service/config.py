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
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")
