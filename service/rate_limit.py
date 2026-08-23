"""Per-IP rate limiting with rotating-salt hashing (issue #22, DESIGN §9).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_service_rate_limit.py`` pins the
contract.

Privacy contract (DESIGN §9, UK GDPR): IPs are personal data. They are
hashed with a rotating salt before ANY storage, hash records are
retained at most :data:`IP_HASH_RETENTION_DAYS` days, and the rate-limit
store is structurally separate from the query/exchange logs — no field
in either store can join the two (the separation is a schema property
the tests scan for, not a promise).

Contract points the red suite pins:

- **Rotating salt.** The same IP hashes identically within one salt
  period and DIFFERENTLY across periods; the salt is never derivable
  from the clock alone (two independently constructed providers disagree
  — there is secret randomness, so hashes are not globally linkable).
- **No raw IPs, ever.** No serialisation of the limiter's stored state
  contains a raw IP, any of its octet substrings in order, or the salt.
- **429 over threshold.** More than the configured requests per window
  from one client IP → refused; the refusal echoes NOTHING about the IP.
- **≤7-day retention.** ``purge_expired`` removes hash records older
  than :data:`IP_HASH_RETENTION_DAYS` (injected clock).
- **Health is never limited** (monitoring must see a paused-but-alive
  service); the chat endpoint is.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = [
    "IP_HASH_RETENTION_DAYS",
    "SALT_ROTATION_PERIOD",
    "RotatingSaltProvider",
    "hash_ip",
    "resolve_client_ip",
    "RateLimiter",
]

#: DESIGN §9: hashed-IP records live at most 7 days.
IP_HASH_RETENTION_DAYS = 7

#: Salt rotation cadence — daily, well inside the 7-day retention bound,
#: so no salt outlives the records it hashed.
SALT_ROTATION_PERIOD = timedelta(days=1)


class RotatingSaltProvider:
    """Provides the current salt period's salt (injected clock).

    A salt is generated with secret randomness per rotation period and
    stable within it; ``current_salt()`` returns (salt_period_key, salt).
    Salts for expired periods are discarded — after retention there is
    nothing left to re-hash an IP against.
    """

    def __init__(
        self,
        clock: Callable[[], datetime],
        period: timedelta = SALT_ROTATION_PERIOD,
    ) -> None:
        self._clock = clock
        self.period = period
        # A fresh secret salt is minted per rotation period on first use and
        # held only in memory (never serialised) — so two independently
        # constructed providers disagree on the same period's salt, and no
        # calendar-derivable value lets anyone re-hash and re-join IPs.
        self._salts_by_period: dict[str, str] = {}

    def _period_key(self, moment: datetime) -> str:
        elapsed = moment.astimezone(UTC).timestamp()
        return str(int(elapsed // self.period.total_seconds()))

    def current_salt(self) -> tuple[str, str]:
        """(period_key, salt) for the clock's current rotation period."""
        period_key = self._period_key(self._clock())
        salt = self._salts_by_period.get(period_key)
        if salt is None:
            salt = secrets.token_hex(32)
            self._salts_by_period[period_key] = salt
        return period_key, salt

    def discard_periods(self, keep: set[str]) -> None:
        """Drop salts for periods not in ``keep`` (retention: no salt outlives
        the records it hashed)."""
        for period_key in list(self._salts_by_period):
            if period_key not in keep:
                del self._salts_by_period[period_key]


def hash_ip(ip: str, salt: str) -> str:
    """Pure: one-way hash of ``ip`` under ``salt``.

    Deterministic for (ip, salt); the output never contains the raw IP
    or any two consecutive octets of it, and is not reversible without
    the salt (a keyed hash, not an encoding).
    """
    return hmac.new(salt.encode("utf-8"), ip.encode("utf-8"), hashlib.sha256).hexdigest()


def resolve_client_ip(
    client_host: str | None,
    forwarded_for: str | None,
    *,
    trusted_proxy: bool,
) -> str:
    """Pure: the client IP the limiter keys on.

    Direct deployments use the socket peer (``client_host``).
    ``trusted_proxy`` deployments (Fly/Railway ingress) use the FIRST
    ``X-Forwarded-For`` entry — but ONLY when ``trusted_proxy`` is True:
    honouring XFF from an untrusted peer lets any client mint fresh
    identities per request and walk straight through the limiter.
    Falls back to ``client_host`` when the header is absent/blank, and
    to ``"unknown"`` when both are missing (unknown clients share one
    bucket: fail toward limiting, never toward unlimited).
    """
    if trusted_proxy and forwarded_for and forwarded_for.strip():
        # Only a trusted ingress's XFF is honoured: the FIRST entry is the
        # originating client. An untrusted peer's XFF would let any client
        # mint a fresh identity per request and walk through the limiter.
        return forwarded_for.split(",")[0].strip()
    if client_host and client_host.strip():
        return client_host.strip()
    return "unknown"


class RateLimiter:
    """Sliding-window per-IP limiter over hashed IPs (injected clock)."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        salts: RotatingSaltProvider,
        max_requests: int,
        window: timedelta = timedelta(minutes=1),
    ) -> None:
        self._clock = clock
        self._salts = salts
        self.max_requests = max_requests
        self.window = window
        self._lock = threading.Lock()
        # Each record: hashed IP + its salt period + when it was seen. NO
        # raw IP, NO salt, and nothing query-side (question/exchange id) —
        # the store is structurally unjoinable to the exchange logs.
        self._records: list[dict[str, Any]] = []

    def allow(self, ip: str) -> bool:
        """Record one request from ``ip`` (hashed immediately; the raw IP
        is never stored) and return whether it is within the window
        threshold. The (N+1)th request inside ``window`` returns False."""
        now = self._clock()
        period_key, salt = self._salts.current_salt()
        hashed = hash_ip(ip, salt)
        window_start = now - self.window
        with self._lock:
            self._records.append({"ip_hash": hashed, "salt_period": period_key, "recorded_at": now})
            recent = sum(
                1
                for record in self._records
                if record["ip_hash"] == hashed and record["recorded_at"] > window_start
            )
        return recent <= self.max_requests

    def purge_expired(self) -> int:
        """Drop hash records older than :data:`IP_HASH_RETENTION_DAYS`
        (and any salts whose period they belonged to); return the count
        removed."""
        cutoff = self._clock() - timedelta(days=IP_HASH_RETENTION_DAYS)
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r["recorded_at"] > cutoff]
            self._salts.discard_periods({r["salt_period"] for r in self._records})
            return before - len(self._records)

    def stored_records(self) -> Sequence[Mapping[str, Any]]:
        """The COMPLETE persisted state, as serialisable mappings.

        This is the privacy audit surface: the red suite serialises it
        and scans for raw IPs, salts, and any query-log-joinable field
        (question text, exchange ids). Nothing the limiter persists may
        live outside this view.
        """
        with self._lock:
            return [dict(record) for record in self._records]
