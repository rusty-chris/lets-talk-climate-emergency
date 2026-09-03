"""Review finding #264 RED — cross-PROCESS exclusion for exchanges.jsonl.

``ExchangeLog`` serialises file access with a ``threading.Lock``, which
exists inside one process only. Two processes legitimately rewrite the
same ``exchanges.jsonl``: the serving process (``append`` on every
exchange and, since #56, ``record_feedback`` — a full-file
read-modify-write on every thumbs click) and the DEPLOYMENT.md §7 cron
(``scripts/run_retention.py``), which constructs its OWN ``ExchangeLog``
and does a full-file read-filter-write in ``purge_expired``. The review
probe reproduced both lost-update failures deterministically: a purged
past-retention record resurrected by a stale ``record_feedback`` write
(the §9/GDPR 90-day bound silently un-enforced), and a 204-confirmed
verdict erased by a stale purge write.

This suite pins the FIXED contract:

- ``service.exchange_log.exchange_file_lock(path)`` — a context-manager
  seam acquiring an OS-LEVEL exclusive lock for the log at ``path``,
  held across every ``ExchangeLog`` read-modify-write (``append``,
  ``records``, ``purge_expired``, ``record_feedback``), inside the
  existing thread lock. Both processes construct ``ExchangeLog``, so
  both honour it. The seam is looked up on the module at call time,
  which is what makes the review's probe injectable here.
- DECISION (flagged for ratification): the recommended mechanism is
  ``fcntl.flock`` on a ``<path>.lock`` sidecar. flock is per
  open-file-description, so two separate acquisitions exclude each
  other even inside one process — which is exactly what these tests
  probe, and what ``fcntl.lockf`` (per-process POSIX record locks)
  would NOT provide. Across processes it holds wherever both writers
  share a kernel and a real filesystem — the DEPLOYMENT §7 reality of
  the serving container and the cron sharing the compose ``api_data``
  volume on one host. (Caveat for the runbook: flock over
  network-mounted volumes, e.g. NFS, is not reliable — DEPLOYMENT.md
  should say the log volume must be host-local.) The tests pin the
  exclusion SEMANTICS and the seam, not the syscall: a lockfile-based
  implementation with equivalent blocking semantics would also pass.

Probe technique (deterministic, no scheduler races): the seam is
wrapped so the FIRST acquisition after patching pauses inside its
critical section; the "other process" (a second ``ExchangeLog``
instance — its own thread lock, exactly what the cron constructs) then
attempts its own read-modify-write from another thread. The tests
assert it BLOCKS until the window closes, and that the final file
carries both writers' effects.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from service.exchange_log import (
    EXCHANGE_LOG_RETENTION_DAYS,
    FEEDBACK_DOWN,
    ExchangeLog,
)
from tests._service_fixtures import FrozenClock
from tests.unit.test_service_privacy_logging import make_record

#: How long the "other process" must stay blocked while the first
#: writer's critical section is held open. Long enough that a missing
#: lock reliably completes inside it; short enough to keep the suite
#: fast.
EXCLUSION_PROBE_WINDOW_S = 0.5

JOIN_TIMEOUT_S = 10.0


def _import_file_lock():
    """The pinned cross-process seam — its absence IS the #264 defect."""
    try:
        from service.exchange_log import exchange_file_lock
    except ImportError:
        pytest.fail(
            "service.exchange_log.exchange_file_lock is missing: every "
            "ExchangeLog read-modify-write must hold an OS-level file lock "
            "shared with the retention cron process — a threading.Lock "
            "only excludes writers inside ONE process (finding #264)"
        )
    return exchange_file_lock


def _pausing_seam(real_lock):
    """Wrap the real seam: the FIRST acquisition pauses inside its
    critical section (lock held) until resumed — the review probe's
    widened race window, injected at the seam the lock must close."""
    in_window = threading.Event()
    resume = threading.Event()
    first_taken = threading.Lock()
    state = {"first_seen": False}

    @contextmanager
    def seam(path: Path):
        with real_lock(path):
            with first_taken:
                pause_here = not state["first_seen"]
                state["first_seen"] = True
            if pause_here:
                in_window.set()
                resume.wait(timeout=JOIN_TIMEOUT_S)
            yield

    return seam, in_window, resume


def _read_records_on_disk(path: Path) -> dict[str, dict]:
    return {
        record["exchange_id"]: record
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _two_process_logs(tmp_path: Path) -> tuple[ExchangeLog, ExchangeLog, dict, dict, Path]:
    """One path, two ``ExchangeLog`` instances — exactly what the serving
    process and the §7 cron construct — seeded with one past-retention
    record and one fresh record."""
    clock = FrozenClock()
    path = tmp_path / "exchanges.jsonl"
    serving = ExchangeLog(path, clock=clock)
    cron = ExchangeLog(path, clock=clock)
    expired = make_record(timestamp=clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS + 1))
    fresh = make_record(timestamp=clock())
    serving.append(expired)
    serving.append(fresh)
    return serving, cron, expired, fresh, path


def _run_interleave(paused_action, blocked_action) -> tuple[threading.Thread, threading.Thread]:
    """Run ``paused_action`` until it parks inside its critical section,
    launch ``blocked_action`` from a second thread, assert the second
    writer BLOCKS for the whole probe window, then release and join."""
    seam_holder = threading.Thread(target=paused_action["run"], daemon=True)
    seam_holder.start()
    assert paused_action["in_window"].wait(timeout=JOIN_TIMEOUT_S), (
        "the first writer never entered the file-lock seam — the "
        "read-modify-write must run inside exchange_file_lock (finding #264)"
    )
    contender = threading.Thread(target=blocked_action, daemon=True)
    contender.start()
    contender.join(timeout=EXCLUSION_PROBE_WINDOW_S)
    try:
        assert contender.is_alive(), (
            "the second process's writer ran INSIDE the first writer's "
            "read-modify-write window: the OS-level file lock is not "
            "excluding a concurrent ExchangeLog instance — this is the "
            "exact lost-update interleave of the review probe (finding #264)"
        )
    finally:
        paused_action["resume"].set()
    seam_holder.join(timeout=JOIN_TIMEOUT_S)
    contender.join(timeout=JOIN_TIMEOUT_S)
    assert not seam_holder.is_alive() and not contender.is_alive(), (
        "the interleaved writers never both completed — the file lock "
        "deadlocked instead of serialising the two processes"
    )
    return seam_holder, contender


def test_file_lock_excludes_across_separate_acquisitions(tmp_path) -> None:
    """Two independent acquisitions for ONE log path exclude each other —
    even inside one process (per-open-file-description semantics, e.g.
    fcntl.flock; a per-process fcntl.lockf or a threading.Lock cannot
    pass this, and neither can the cron's separate process evade it)."""
    exchange_file_lock = _import_file_lock()
    path = tmp_path / "exchanges.jsonl"
    held = threading.Event()
    release = threading.Event()
    second_acquired = threading.Event()

    def holder() -> None:
        with exchange_file_lock(path):
            held.set()
            release.wait(timeout=JOIN_TIMEOUT_S)

    def contender() -> None:
        with exchange_file_lock(path):
            second_acquired.set()

    first = threading.Thread(target=holder, daemon=True)
    first.start()
    assert held.wait(timeout=JOIN_TIMEOUT_S)
    second = threading.Thread(target=contender, daemon=True)
    second.start()
    try:
        assert not second_acquired.wait(timeout=EXCLUSION_PROBE_WINDOW_S), (
            "a second exchange_file_lock acquisition succeeded while the "
            "first was held: the lock does not exclude a concurrent "
            "acquisition on the same path (finding #264)"
        )
    finally:
        release.set()
    first.join(timeout=JOIN_TIMEOUT_S)
    assert second_acquired.wait(timeout=JOIN_TIMEOUT_S), (
        "the blocked acquisition never proceeded after release — the lock "
        "serialises by deadlocking, not by queueing"
    )
    second.join(timeout=JOIN_TIMEOUT_S)


def test_cross_process_purge_cannot_resurrect_expired_records(tmp_path, monkeypatch) -> None:
    """The review's scenario 1: record_feedback reads (file still holds a
    past-retention record), the cron purge runs, record_feedback writes.
    Under the file lock the purge must WAIT; afterwards the expired
    record stays gone AND the verdict is on disk — never a stale full
    copy resurrecting what §9 already deleted."""
    exchange_file_lock = _import_file_lock()
    serving, cron, expired, fresh, path = _two_process_logs(tmp_path)
    seam, in_window, resume = _pausing_seam(exchange_file_lock)
    monkeypatch.setattr("service.exchange_log.exchange_file_lock", seam)

    outcome: dict[str, bool] = {}

    def feedback() -> None:
        outcome["recorded"] = serving.record_feedback(fresh["exchange_id"], FEEDBACK_DOWN)

    _run_interleave(
        {"run": feedback, "in_window": in_window, "resume": resume},
        cron.purge_expired,
    )

    on_disk = _read_records_on_disk(path)
    assert expired["exchange_id"] not in on_disk, (
        "a record past the 90-day retention bound is BACK on disk after "
        "purge_expired ran: the concurrent feedback rewrite resurrected "
        "it from its stale read (finding #264, GDPR §9)"
    )
    assert outcome.get("recorded") is True
    assert on_disk[fresh["exchange_id"]]["feedback"] == {"verdict": FEEDBACK_DOWN}, (
        "the confirmed verdict is missing from the surviving record"
    )


def test_concurrent_purge_never_erases_a_recorded_verdict(tmp_path, monkeypatch) -> None:
    """The review's scenario 2, mirrored: the cron purge holds its
    read-filter-write open; the visitor's thumbs click must WAIT, then
    land — the route's 204 ("your rating was recorded") stays true on
    disk, never clobbered by the purge's verdict-free stale copy."""
    exchange_file_lock = _import_file_lock()
    serving, cron, expired, fresh, path = _two_process_logs(tmp_path)
    seam, in_window, resume = _pausing_seam(exchange_file_lock)
    monkeypatch.setattr("service.exchange_log.exchange_file_lock", seam)

    outcome: dict[str, bool] = {}

    def feedback() -> None:
        outcome["recorded"] = serving.record_feedback(fresh["exchange_id"], FEEDBACK_DOWN)

    _run_interleave(
        {"run": cron.purge_expired, "in_window": in_window, "resume": resume},
        feedback,
    )

    on_disk = _read_records_on_disk(path)
    assert outcome.get("recorded") is True, (
        "record_feedback answered False for a retained exchange after the "
        "purge completed — the interleave lost the record itself"
    )
    assert on_disk[fresh["exchange_id"]]["feedback"] == {"verdict": FEEDBACK_DOWN}, (
        "a verdict the visitor was told was recorded (204) is gone from "
        "disk: the purge's stale write erased it (finding #264)"
    )
    assert expired["exchange_id"] not in on_disk


def test_append_is_never_lost_to_a_concurrent_purge_write(tmp_path, monkeypatch) -> None:
    """The review's scenario 3: an appended exchange (whose exchange_id
    already rode out on the meta event) must survive a concurrent purge
    rewrite — append waits for the purge's critical section, so every
    later feedback POST for that id can still find its record."""
    exchange_file_lock = _import_file_lock()
    serving, cron, expired, fresh, path = _two_process_logs(tmp_path)
    seam, in_window, resume = _pausing_seam(exchange_file_lock)
    monkeypatch.setattr("service.exchange_log.exchange_file_lock", seam)

    appended = make_record()

    _run_interleave(
        {"run": cron.purge_expired, "in_window": in_window, "resume": resume},
        lambda: serving.append(appended),
    )

    on_disk = _read_records_on_disk(path)
    assert appended["exchange_id"] in on_disk, (
        "an appended exchange record vanished under a concurrent purge "
        "write — its exchange_id was already on the wire, so every later "
        "feedback POST for it becomes a 404 (finding #264)"
    )
    assert expired["exchange_id"] not in on_disk
    assert fresh["exchange_id"] in on_disk


def test_reads_wait_for_a_writer_in_progress(tmp_path, monkeypatch) -> None:
    """``records()`` (the triage/harvest read path) takes the same lock:
    a read issued inside another process's rewrite window blocks, then
    observes the completed post-purge state — never a half-applied one."""
    exchange_file_lock = _import_file_lock()
    serving, cron, expired, fresh, path = _two_process_logs(tmp_path)
    seam, in_window, resume = _pausing_seam(exchange_file_lock)
    monkeypatch.setattr("service.exchange_log.exchange_file_lock", seam)

    seen: dict[str, list] = {}

    def reader() -> None:
        seen["records"] = serving.records()

    _run_interleave(
        {"run": cron.purge_expired, "in_window": in_window, "resume": resume},
        reader,
    )

    read_ids = [record["exchange_id"] for record in seen["records"]]
    assert expired["exchange_id"] not in read_ids, (
        "a records() read that overlapped the purge's critical section "
        "served the pre-purge state — reads must exclude against a "
        "writer in progress (finding #264)"
    )
    assert read_ids == [fresh["exchange_id"]]
