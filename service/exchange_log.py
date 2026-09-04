"""Privacy-compliant structured exchange logging (issue #22, DESIGN §9).

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suite in ``tests/unit/test_service_privacy_logging.py`` pins the
contract.

DESIGN §9 privacy section, made code:

- **What we log and why:** question text, retrieved chunk ids, answer,
  citations, citation-support rate (the #13 ``exchange_log_record``
  fields), usage/cost — for service operation and evaluation
  improvement. NO identifiers: no IP, no IP hash, no user agent, no
  cookie, no session, no account (there are none).
- **Retention:** raw exchange records live
  :data:`EXCHANGE_LOG_RETENTION_DAYS` days, then a retention job deletes
  them (injected clock).
- **Harvest flow:** exchanges promoted toward the eval sets are
  hand-reviewed; unsafe-classified (and unsafe-suspected, finding #86)
  exchanges NEVER enter the harvest queue; promotion irreversibly
  detaches the exchange from timestamps and any join key.
- **The #56 seam, now consumed:** every record carries a random
  ``exchange_id`` (joins a thumbs-up/down feedback event to the
  exchange — it identifies the exchange, never the person) and a
  ``feedback`` field. Issue #56 writes it: :meth:`ExchangeLog.
  record_feedback` lands ``{"verdict": "up"|"down"}`` on the matched
  record IN PLACE (decision flagged in the #56 red-phase notes: a
  single-store record rewrite under the log's lock, NOT a sidecar
  feedback log — the file is already rewritten wholesale by
  ``purge_expired``, so "append-only" here means no new record kinds
  and no reordering, and the rewrite keeps every guarantee free:
  feedback follows its exchange through the 90-day purge with no
  second retention job, ``harvest_candidates`` sees it inline, and
  ``detach_for_harvest`` already strips it so a verdict can never ride
  into a published eval set as a join key).
- **Disclosure:** :data:`LOGGING_DISCLOSURE` is the one-line notice the
  chat surface must carry (served to the UI in the chat stream's ``meta``
  event and on the /privacy page).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "LOGGING_DISCLOSURE",
    "EXCHANGE_LOG_RETENTION_DAYS",
    "FORBIDDEN_IDENTIFIER_FIELDS",
    "FEEDBACK_UP",
    "FEEDBACK_DOWN",
    "FEEDBACK_VERDICTS",
    "build_exchange_record",
    "exchange_file_lock",
    "ExchangeLog",
    "harvest_candidates",
    "detach_for_harvest",
]


@contextmanager
def exchange_file_lock(path: Path):
    """A cross-PROCESS exclusive lock for the exchange log at ``path``.

    Two processes legitimately rewrite ``exchanges.jsonl`` — the serving
    process (``append``/``record_feedback``) and the DEPLOYMENT §7 cron
    (``purge_expired``) — and a ``threading.Lock`` excludes writers inside
    ONE process only (finding #264). This seam holds an OS-level exclusive
    lock via ``fcntl.flock`` on a ``<path>.lock`` sidecar: flock is
    per-open-file-description, so two acquisitions exclude each other even
    within one process, and hold across any two processes sharing a kernel
    and a real filesystem (the compose ``api_data`` volume reality).

    Caveat for the runbook: flock over a network-mounted volume (NFS) is
    unreliable — the log volume must stay host-local.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


#: The exact one-line disclosure from DESIGN §9.
LOGGING_DISCLOSURE = (
    "Conversations are logged anonymously to improve the service — don't share personal details."
)

#: DESIGN §9: raw exchange logs live 90 days, then deleted.
EXCHANGE_LOG_RETENTION_DAYS = 90

#: The CLOSED #56 feedback vocabulary: a thumbs verdict is one of exactly
#: these two strings — no free text in MVP (the GDPR surface is unchanged),
#: no scores, no other value. The service route 422s anything else; the UI
#: pins its own constants equal to these (the wire-vocabulary parity
#: pattern of ``tests/unit/test_ui_shell_hygiene.py``).
FEEDBACK_UP = "up"
FEEDBACK_DOWN = "down"
FEEDBACK_VERDICTS: frozenset[str] = frozenset({FEEDBACK_UP, FEEDBACK_DOWN})

#: Field names that must NEVER appear in an exchange record, at any
#: nesting depth — the redaction tests scan serialised records for them.
FORBIDDEN_IDENTIFIER_FIELDS = (
    "ip",
    "ip_hash",
    "client",
    "remote_addr",
    "x_forwarded_for",
    "user_agent",
    "cookie",
    "session",
    "authorization",
)


def build_exchange_record(
    *,
    question: str,
    route: str,
    answer_text: str,
    retrieved_chunk_ids: Sequence[str],
    citations: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    usage_records: Sequence[Mapping[str, Any]],
    exclude_from_harvest: bool,
    timestamp: datetime,
    exchange_id: str | None = None,
    cached_from: str | None = None,
) -> dict[str, Any]:
    """Pure: one structured exchange record (the §9 logging schema).

    Returns a JSON-serialisable mapping with EXACTLY these top-level
    keys: ``exchange_id`` (the #56 feedback join key; identifies the
    exchange, never the person), ``timestamp`` (ISO 8601 of
    ``timestamp``), ``question``, ``route``, ``answer_text``,
    ``retrieved_chunk_ids``, ``citations``, ``validation`` (the #13
    ``exchange_log_record`` mapping, carried whole — including
    ``citation_support_rate``), ``usage_records`` (per-call model +
    usage + cost), ``exclude_from_harvest``, ``feedback`` (None
    until a #56 verdict lands), and ``cached_from`` (issue #57 — the
    source exchange_id on a semantic-cache serving, None otherwise). No
    other keys; none of :data:`FORBIDDEN_IDENTIFIER_FIELDS` at any depth.

    ``exchange_id`` (issue #56): ``None`` — the pre-#56 behaviour —
    mints a fresh random UUID hex; a provided id is carried VERBATIM.
    The chat route mints the id once at stream start, rides it to the
    client in the ``meta`` event, and passes the SAME id here, so the
    id the visitor's thumbs click posts back is the id the logged
    record carries — one id, minted once, joining wire to record. The
    id is random either way: nothing about the content or client ever
    derives it.
    """
    return {
        # A fresh random id per exchange: the #56 feedback join key. It
        # identifies the exchange, never the person — nothing about the
        # content or client derives it. The chat route mints it once at
        # stream start and passes the SAME id here (carried verbatim), so
        # the id on the wire IS the id on the record; ``None`` keeps the
        # pre-#56 minting.
        "exchange_id": exchange_id if exchange_id is not None else uuid.uuid4().hex,
        "timestamp": timestamp.isoformat(),
        "question": question,
        "route": route,
        "answer_text": answer_text,
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
        "citations": [dict(citation) for citation in citations],
        "validation": dict(validation),
        "usage_records": [dict(record) for record in usage_records],
        "exclude_from_harvest": bool(exclude_from_harvest),
        # The #56 seam: empty until a later thumbs-up/down event writes it.
        "feedback": None,
        # The #57 seam: the source exchange_id on a semantic-cache serving
        # (route "cached"); None on every other route. detach_for_harvest
        # whitelists content fields, so this join key never rides into a
        # published eval case.
        "cached_from": cached_from,
    }


class ExchangeLog:
    """Append-only JSONL exchange log with a 90-day retention job."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.Lock()

    def _valid_lines(self, text: str) -> tuple[list[str], str | None]:
        """Split JSONL ``text`` into its valid non-empty line strings,
        tolerating a single torn TRAILING line — the residue an
        interrupted rewrite or append leaves. Returns
        ``(valid_lines, torn_line_or_None)``. A NON-trailing decode
        failure raises loudly: a mid-file corruption signals something
        worse than a torn write and is not silently reparsed (finding
        #265)."""
        stripped = [line for line in text.splitlines() if line.strip()]
        for index, line in enumerate(stripped):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                if index == len(stripped) - 1:
                    return stripped[:index], line
                raise
        return stripped, None

    def _read_valid_lines(self) -> tuple[list[str], bool]:
        """Under the caller's lock: ``(valid line strings, torn_tail_seen)``.
        A torn TRAILING line is quarantined LOUDLY (WARNING + sidecar) and
        excluded from the returned lines; the live file is NOT rewritten
        here — callers that rewrite drop the torn tail naturally, and
        ``records()`` heals it explicitly."""
        if not self.path.is_file():
            return [], False
        lines, torn = self._valid_lines(self.path.read_text(encoding="utf-8"))
        if torn is not None:
            self._quarantine(torn)
        return lines, torn is not None

    def _quarantine(self, torn_line: str) -> None:
        """Move a torn TRAILING line out of the live log, LOUDLY: append
        it to the ``<path>.corrupt`` sidecar and log a WARNING. An
        interrupted rewrite or append leaves a truncated JSON record;
        dropping operator data is never silent (finding #265)."""
        corrupt_path = Path(str(self.path) + ".corrupt")
        with corrupt_path.open("a", encoding="utf-8") as handle:
            handle.write(torn_line + "\n")
        _LOGGER.warning(
            "quarantined a torn/truncated trailing line from %s to %s "
            "(%d bytes): an interrupted rewrite or append left a corrupt "
            "JSON record; the valid records are retained (finding #265)",
            self.path,
            corrupt_path,
            len(torn_line),
        )

    def _atomic_write(self, text: str) -> None:
        """Replace the log file atomically: write ``text`` to a temp file
        in the same directory, fsync it to disk, then ``os.replace`` it
        into place. A crash before the replace leaves the OLD file
        byte-identical; the fsync stops a power loss installing unflushed
        (e.g. empty) data behind the rename (finding #265)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(str(self.path) + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(tmp_path, self.path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record as a single JSON line."""
        line = json.dumps(dict(record), ensure_ascii=False)
        # The thread lock (this instance) inside the cross-process file lock
        # (shared with the §7 cron): every read-modify-write holds both
        # (finding #264). The seam is a module global, so tests inject it.
        with self._lock, exchange_file_lock(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def records(self) -> list[dict[str, Any]]:
        """All records currently retained, in append order.

        A torn trailing line is tolerated: it is quarantined out (loud
        WARNING + sidecar) and the live file healed, never raised as a
        JSONDecodeError that would kill the triage/harvest read path
        (finding #265)."""
        with self._lock, exchange_file_lock(self.path):
            lines, torn_seen = self._read_valid_lines()
            if torn_seen:
                # Complete the quarantine: rewrite the live file without the
                # torn tail so the next reader sees a clean log.
                self._atomic_write("\n".join(lines) + ("\n" if lines else ""))
            return [json.loads(line) for line in lines]

    def purge_expired(self) -> int:
        """The retention job: delete records whose ``timestamp`` is older
        than :data:`EXCHANGE_LOG_RETENTION_DAYS` days at the injected
        clock's now; keep everything younger; return the count deleted.

        The rewrite is atomic (temp file + fsync + ``os.replace``), so a
        crash mid-write leaves the old file intact rather than destroying
        up to 90 days of records; a torn trailing line is quarantined,
        not raised, so retention keeps enforcing (finding #265)."""
        cutoff = self._clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS)
        with self._lock, exchange_file_lock(self.path):
            lines, torn_seen = self._read_valid_lines()
            if not lines and not torn_seen:
                return 0
            kept: list[str] = []
            removed = 0
            for line in lines:
                record = json.loads(line)
                if datetime.fromisoformat(record["timestamp"]) > cutoff:
                    kept.append(line)
                else:
                    removed += 1
            self._atomic_write("\n".join(kept) + ("\n" if kept else ""))
            return removed

    def record_feedback(self, exchange_id: str, verdict: str) -> bool:
        """Land one #56 thumbs verdict on the matched record's ``feedback``
        field, in place.

        RED-phase contract stub (issue #56); the failing suite in
        ``tests/unit/test_service_feedback.py`` pins the contract:

        - ``verdict`` outside :data:`FEEDBACK_VERDICTS` raises
          ``ValueError`` (defence in depth below the route's 422 — the
          closed vocabulary holds even for direct callers).
        - The record whose ``exchange_id`` matches gets ``feedback``
          set to EXACTLY ``{"verdict": verdict}`` — no timestamp, no
          client field, nothing else: the feedback references the
          exchange and carries the verdict, full stop (DESIGN §9: the
          feedback surface adds NO identifier). Returns True.
        - The write is a single-line rewrite under the log's lock:
          every OTHER line stays byte-identical, order is preserved,
          and the line count never changes — feedback never appends a
          record.
        - IDEMPOTENT PER EXCHANGE: a second verdict REPLACES the first
          (``feedback`` is one mapping, never a list; up-then-down
          leaves ``{"verdict": "down"}``) — a visitor changing their
          mind, or double-clicking, can never inflate the signal.
        - An ``exchange_id`` matching no retained record returns False
          and touches NOTHING (the file is byte-identical) — the route
          turns False into its uniform 404. A purged exchange and a
          never-existed exchange are indistinguishable here by design.
        - Purge interplay: because the verdict lives ON the record,
          ``purge_expired`` deletes it with its exchange — no second
          retention job, no orphaned feedback, no residue.
        """
        if verdict not in FEEDBACK_VERDICTS:
            raise ValueError(
                f"feedback verdict {verdict!r} is outside the closed vocabulary "
                f"{sorted(FEEDBACK_VERDICTS)}"
            )
        with self._lock, exchange_file_lock(self.path):
            lines, torn_seen = self._read_valid_lines()
            for index, line in enumerate(lines):
                record = json.loads(line)
                if record.get("exchange_id") != exchange_id:
                    continue
                # Single-line rewrite: only the matched record changes; every
                # other line is left byte-identical and in place, so the line
                # count and order never move (feedback never appends). The
                # write is atomic (temp + fsync + os.replace), so a crash
                # never clobbers the log and never claims success (#265).
                record["feedback"] = {"verdict": verdict}
                lines[index] = json.dumps(record, ensure_ascii=False)
                self._atomic_write("\n".join(lines) + "\n")
                return True
            # No match: touch nothing (the route turns this into its uniform
            # 404 — a purged id and a never-issued id are indistinguishable),
            # unless a torn tail was quarantined and needs healing out.
            if torn_seen:
                self._atomic_write("\n".join(lines) + ("\n" if lines else ""))
            return False


def harvest_candidates(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure: the eval-harvest review queue for a set of records.

    Excludes EVERY record with ``exclude_from_harvest`` truthy (unsafe
    and unsafe-suspected exchanges, DESIGN §3.1/§8 — the #10 flag
    consumed here) — exclusion is structural, before any human sees a
    queue, and holds even when the excluded exchange was thumbed down
    (a verdict never overrides the safety exclusion).

    Issue #56 triage ordering (pinned by the failing suite in
    ``tests/unit/test_service_feedback.py``): thumbs-DOWN exchanges
    (``feedback == {"verdict": "down"}``) surface FIRST — they are the
    reviewer's triage input (SotA rec 4: a visitor said the answer
    failed them; the reviewer sees those before anything else) — in
    their original order, followed by every other candidate (thumbs-up,
    unrated) in original order. The sort is stable both sides of the
    partition. Harvesting stays hand-review + irreversible detachment;
    the ordering changes WHAT surfaces first, never what may enter.
    """
    candidates = [dict(record) for record in records if not record.get("exclude_from_harvest")]
    # Stable partition: thumbs-down exchanges first (the reviewer's triage
    # input), then every other candidate — each side in original order.
    downvoted = [c for c in candidates if c.get("feedback") == {"verdict": FEEDBACK_DOWN}]
    others = [c for c in candidates if c.get("feedback") != {"verdict": FEEDBACK_DOWN}]
    return downvoted + others


def detach_for_harvest(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pure: the irreversible detachment step of the harvest flow.

    Returns ONLY the content fields (``question``,
    ``retrieved_chunk_ids``, ``answer_text``, ``citations``,
    ``validation``) — no ``timestamp``, no ``exchange_id``, no
    ``feedback``, no ``usage_records``: nothing that could re-join the
    promoted eval case to when it happened or to any other stored
    record. Raises ``ValueError`` on a record with
    ``exclude_from_harvest`` truthy — the exclusion cannot be bypassed
    by calling the detach step directly.
    """
    if record.get("exclude_from_harvest"):
        raise ValueError(
            "refusing to detach an excluded exchange for harvest — the "
            "unsafe/unsafe-suspected exclusion cannot be bypassed"
        )
    return {
        "question": record["question"],
        "retrieved_chunk_ids": list(record.get("retrieved_chunk_ids", [])),
        "answer_text": record["answer_text"],
        "citations": [dict(citation) for citation in record.get("citations", [])],
        "validation": dict(record.get("validation", {})),
    }
