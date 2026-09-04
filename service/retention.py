"""The retention jobs that make the §9 GDPR bounds actually operate (#213).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing tests in ``tests/unit/test_service_privacy_logging.py`` and
``tests/unit/test_service_rate_limit.py`` pin the contract.

``ExchangeLog.purge_expired`` (90-day) and ``RateLimiter.purge_expired``
(7-day) exist and are unit-tested, but review finding #213 showed
nothing ever CALLS them in a running service — and the rate-limit store
is in-process memory, so no external cron job can ever reach it. Two
mechanisms close the gap:

- **In-process scheduled purge** (the only mechanism that can bound the
  in-memory rate-limit store): the app's lifespan runs one
  :func:`run_retention_pass` immediately at startup and then repeats it
  every :data:`RETENTION_PURGE_INTERVAL` for the process lifetime.
  DECISION (flagged for ratification): startup + fixed interval via a
  FastAPI lifespan background task, per the issue's prescription —
  never per-request work.
- **A shipped operator entrypoint** for the file-backed exchange log
  (``scripts/run_retention.py`` driving :func:`purge_exchange_log`), so
  the DEPLOYMENT.md §7 cron guidance points at a script that exists.
  It cannot and does not claim to purge the in-process rate-limit
  store.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from service.exchange_log import ExchangeLog
from service.rate_limit import RateLimiter

__all__ = [
    "RETENTION_PURGE_INTERVAL",
    "run_retention_pass",
    "purge_exchange_log",
]

#: How often the in-process purge repeats. Well inside the tightest
#: retention bound (the 7-day IP-hash rule) — a day-scale interval keeps
#: the stores bounded without measurable request-path cost.
RETENTION_PURGE_INTERVAL = timedelta(hours=6)


def run_retention_pass(
    exchange_log: ExchangeLog,
    rate_limiter: RateLimiter,
    semantic_cache: Any | None = None,
) -> dict[str, int]:
    """One purge pass over EVERY retention-bound store.

    Calls ``exchange_log.purge_expired()`` (90-day bound) and
    ``rate_limiter.purge_expired()`` (7-day bound); returns
    ``{"exchange_records_removed": n, "rate_limit_records_removed": m}``.
    Invoked by the app lifespan at startup and every
    :data:`RETENTION_PURGE_INTERVAL` thereafter.

    Issue #57 (RED contract, pinned by
    ``tests/unit/test_service_semantic_cache.py``): when
    ``semantic_cache`` (a ``service.semantic_cache.SemanticCache``) is
    provided, its ``purge_expired()`` also runs — cached exchange
    content follows the same §9 90-day bound as the exchange log it
    mirrors — and the returned mapping gains
    ``"semantic_cache_entries_removed": k``. ``None`` (cache disabled)
    keeps the two-store behaviour and the two-key return shape. The app
    lifespan passes ``deps.semantic_cache`` through on every pass.
    """
    counts = {
        "exchange_records_removed": exchange_log.purge_expired(),
        "rate_limit_records_removed": rate_limiter.purge_expired(),
    }
    if semantic_cache is not None:
        # Cached exchange content follows the same §9 90-day bound as the
        # exchange log it mirrors.
        counts["semantic_cache_entries_removed"] = semantic_cache.purge_expired()
    return counts


def purge_exchange_log(log_dir: Path, *, clock: Callable[[], datetime]) -> int:
    """The operator-facing exchange-log purge (``scripts/run_retention.py``).

    Opens ``<log_dir>/exchanges.jsonl`` and deletes records older than
    the 90-day retention bound at ``clock()``'s now; returns the count
    removed. File-backed, so it is safe to run from cron OUTSIDE the
    serving process — unlike the in-memory rate-limit store, which only
    the in-process scheduled purge can reach.
    """
    log = ExchangeLog(Path(log_dir) / "exchanges.jsonl", clock=clock)
    return log.purge_expired()
