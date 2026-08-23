"""Per-IP rate limiting with rotating-salt hashing (issue #22, DESIGN §9).

Pins ``service.rate_limit``: rotating-salt hashing (same IP hashes
differently across periods; raw IPs never stored anywhere), the 429
over-threshold behaviour through the composed app, ≤7-day hash
retention, the trusted-proxy client-IP resolution rule, and the
structural separation between the rate-limit store and the query logs.
"""

from __future__ import annotations

import json
from datetime import timedelta

from fastapi.testclient import TestClient

from service.rate_limit import (
    IP_HASH_RETENTION_DAYS,
    RateLimiter,
    RotatingSaltProvider,
    hash_ip,
    resolve_client_ip,
)
from tests._service_fixtures import (
    FrozenClock,
    classifier_output,
    make_config,
    make_harness,
    post_chat,
)

IP = "203.0.113.77"


def make_limiter(
    *,
    max_requests: int = 5,
    clock: FrozenClock | None = None,
) -> tuple[RateLimiter, FrozenClock]:
    clock = clock or FrozenClock()
    limiter = RateLimiter(
        clock=clock,
        salts=RotatingSaltProvider(clock),
        max_requests=max_requests,
        window=timedelta(minutes=1),
    )
    return limiter, clock


def test_ip_hash_uses_rotating_salt() -> None:
    """The same IP hashes identically within a salt period and
    DIFFERENTLY across periods — hashes cannot track an IP over time."""
    clock = FrozenClock()
    salts = RotatingSaltProvider(clock)
    period_1, salt_1 = salts.current_salt()
    assert hash_ip(IP, salt_1) == hash_ip(IP, salt_1)

    clock.advance(salts.period)
    period_2, salt_2 = salts.current_salt()
    assert period_2 != period_1
    assert salt_2 != salt_1
    assert hash_ip(IP, salt_2) != hash_ip(IP, salt_1)


def test_salt_carries_secret_randomness_not_just_the_clock() -> None:
    """Two independently constructed providers disagree on the same
    period's salt: a salt derivable from the calendar alone would let
    anyone re-hash IPs and join them back."""
    salts_a = RotatingSaltProvider(FrozenClock())
    salts_b = RotatingSaltProvider(FrozenClock())
    assert salts_a.current_salt()[1] != salts_b.current_salt()[1]


def test_hash_never_contains_the_ip() -> None:
    _, salt = RotatingSaltProvider(FrozenClock()).current_salt()
    hashed = hash_ip(IP, salt)
    assert IP not in hashed
    for octet_pair in ("203.0", "0.113", "113.77"):
        assert octet_pair not in hashed


def test_raw_ips_never_stored() -> None:
    """No serialisation of the limiter's complete stored state contains
    a raw IP or the salt itself."""
    limiter, _ = make_limiter()
    for _ in range(3):
        limiter.allow(IP)
    limiter.allow("198.51.100.9")

    serialised = json.dumps(limiter.stored_records(), default=str)
    assert IP not in serialised
    assert "198.51.100.9" not in serialised
    _, salt = limiter._salts.current_salt()
    assert salt not in serialised


def test_limiter_thresholds_per_ip() -> None:
    limiter, _ = make_limiter(max_requests=3)
    assert [limiter.allow(IP) for _ in range(3)] == [True, True, True]
    assert limiter.allow(IP) is False
    # A different IP is unaffected.
    assert limiter.allow("198.51.100.9") is True


def test_window_slides_with_the_injected_clock() -> None:
    limiter, clock = make_limiter(max_requests=2)
    assert limiter.allow(IP) and limiter.allow(IP)
    assert limiter.allow(IP) is False
    clock.advance(timedelta(minutes=2))
    assert limiter.allow(IP) is True


def test_ip_hashes_deleted_after_7_days() -> None:
    """The retention job removes hash records older than 7 days."""
    limiter, clock = make_limiter()
    limiter.allow(IP)
    assert len(limiter.stored_records()) > 0

    clock.advance(timedelta(days=IP_HASH_RETENTION_DAYS, seconds=1))
    removed = limiter.purge_expired()
    assert removed > 0
    assert list(limiter.stored_records()) == []


def test_records_inside_retention_survive_the_purge() -> None:
    limiter, clock = make_limiter()
    limiter.allow(IP)
    clock.advance(timedelta(days=IP_HASH_RETENTION_DAYS - 1))
    limiter.allow("198.51.100.9")
    clock.advance(timedelta(days=1, seconds=1))
    limiter.purge_expired()
    # The older record is gone; the younger one (age 1 day) remains.
    assert len(limiter.stored_records()) == 1


class TestResolveClientIp:
    def test_direct_deployment_uses_the_socket_peer(self) -> None:
        assert resolve_client_ip(IP, "6.6.6.6", trusted_proxy=False) == IP

    def test_trusted_proxy_uses_first_forwarded_entry(self) -> None:
        assert resolve_client_ip("10.0.0.1", f"{IP}, 10.0.0.1", trusted_proxy=True) == IP

    def test_untrusted_forwarded_header_is_ignored(self) -> None:
        """Honouring XFF from an untrusted peer lets any client mint a
        fresh identity per request and walk through the limiter."""
        assert resolve_client_ip("10.0.0.1", f"{IP}", trusted_proxy=False) == "10.0.0.1"

    def test_missing_everything_shares_one_bucket(self) -> None:
        """Fail toward limiting, never toward unlimited."""
        assert resolve_client_ip(None, None, trusted_proxy=True) == "unknown"


class TestRateLimitThroughTheApp:
    def test_rate_limit_returns_429_over_threshold(self, tmp_path) -> None:
        # A tight limit: 2 chat requests per window.
        harness = make_harness(
            tmp_path,
            config=make_config(
                starter_cache_dir=str(tmp_path / "starter-cache"),
                log_dir=str(tmp_path / "logs"),
                rate_limit_per_minute=2,
            ),
        )
        client = TestClient(harness.app)
        # Program enough classifier outputs for the allowed requests
        # (out_of_scope keeps each exchange to one structured call).
        harness.adapter.queue(
            "structured",
            classifier_output(scope="out_of_scope"),
            classifier_output(scope="out_of_scope"),
        )
        assert client.post("/chat", json={"question": "q1"}).status_code == 200
        assert client.post("/chat", json={"question": "q2"}).status_code == 200

        response = client.post("/chat", json={"question": "q3"})
        assert response.status_code == 429
        # The refusal echoes nothing about the client.
        assert "testclient" not in response.text

    def test_health_is_never_rate_limited(self, tmp_path) -> None:
        harness = make_harness(
            tmp_path,
            config=make_config(
                starter_cache_dir=str(tmp_path / "starter-cache"),
                log_dir=str(tmp_path / "logs"),
                rate_limit_per_minute=1,
            ),
        )
        client = TestClient(harness.app)
        for _ in range(10):
            assert client.get("/health").status_code == 200


class TestRetentionOperatesInProcess:
    """Issue #213: the rate-limit store is in-process memory — no cron
    job in another process can ever purge it, and nothing in the app
    invokes ``purge_expired``. The ≤7-day bound (DESIGN §9) must be
    enforced by the service itself."""

    def test_limiter_purges_in_process_on_a_schedule(self, tmp_path) -> None:
        """The app lifespan wires a scheduled purge over BOTH stores:
        entering the lifespan (TestClient context) runs a retention pass
        — observable through recording fakes at the deps seam."""
        from dataclasses import replace

        from service.app import create_app
        from tests._service_fixtures import make_harness

        class RecordingPurgeStore:
            def __init__(self) -> None:
                self.purge_calls = 0

            def purge_expired(self) -> int:
                self.purge_calls += 1
                return 0

        harness = make_harness(tmp_path)
        fake_limiter = RecordingPurgeStore()
        fake_log = RecordingPurgeStore()
        deps = replace(harness.deps, rate_limiter=fake_limiter, exchange_log=fake_log)
        app = create_app(harness.config, deps)

        with TestClient(app):
            pass  # lifespan startup must schedule + run the first pass

        assert fake_limiter.purge_calls >= 1, (
            "the app lifespan never drove RateLimiter.purge_expired — the "
            "7-day hash-retention bound does not operate (issue #213)"
        )
        assert fake_log.purge_calls >= 1, (
            "the app lifespan never drove ExchangeLog.purge_expired — the "
            "90-day retention bound does not operate in-process (issue #213)"
        )

    def test_purge_interval_sits_inside_the_tightest_retention_bound(self) -> None:
        from service.retention import RETENTION_PURGE_INTERVAL

        assert timedelta(0) < RETENTION_PURGE_INTERVAL <= timedelta(days=1)

    def test_records_are_bounded_without_manual_purge(self) -> None:
        """`allow()` self-trims stale records as it goes: after N calls
        spaced beyond the retention window, the store holds only the
        in-window record(s) — never the whole process history (which
        today grows unbounded and makes each call O(total history))."""
        limiter, clock = make_limiter(max_requests=5)
        for call_index in range(10):
            if call_index:
                clock.advance(timedelta(days=IP_HASH_RETENTION_DAYS + 1))
            limiter.allow(IP)

        records = limiter.stored_records()
        assert len(records) == 1, (
            f"the limiter retained {len(records)} records across ten "
            "beyond-retention gaps — the store is unbounded (issue #213)"
        )
        cutoff = clock() - timedelta(days=IP_HASH_RETENTION_DAYS)
        assert all(record["recorded_at"] > cutoff for record in records)


def test_rate_limit_data_never_joined_to_query_logs(tmp_path) -> None:
    """Structural separation (DESIGN §9): after a real exchange, the
    rate-limit store carries no query-side field (question text,
    exchange id) and the exchange log carries no IP-side field (hash,
    client address) — there is nothing to join ON, in either direction."""
    harness = make_harness(tmp_path)
    harness.adapter.queue("structured", classifier_output(scope="out_of_scope"))
    post_chat(TestClient(harness.app), "a question that will be logged")

    log_records = harness.exchange_log.records()
    assert log_records, "the exchange must have been logged for this test to mean anything"
    limiter_state = json.dumps(harness.limiter.stored_records(), default=str)
    log_state = json.dumps(log_records)

    # Query-side values never reach the limiter store.
    assert "a question that will be logged" not in limiter_state
    for record in log_records:
        assert record["exchange_id"] not in limiter_state

    # IP-side values never reach the exchange log.
    for field in ("ip", "ip_hash", "client", "remote_addr"):
        for record in log_records:
            assert field not in record, f"exchange record carries IP-side field {field!r}"
    assert "testclient" not in log_state
