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

import json
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "LOGGING_DISCLOSURE",
    "EXCHANGE_LOG_RETENTION_DAYS",
    "FORBIDDEN_IDENTIFIER_FIELDS",
    "FEEDBACK_UP",
    "FEEDBACK_DOWN",
    "FEEDBACK_VERDICTS",
    "build_exchange_record",
    "ExchangeLog",
    "harvest_candidates",
    "detach_for_harvest",
]

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
) -> dict[str, Any]:
    """Pure: one structured exchange record (the §9 logging schema).

    Returns a JSON-serialisable mapping with EXACTLY these top-level
    keys: ``exchange_id`` (the #56 feedback join key; identifies the
    exchange, never the person), ``timestamp`` (ISO 8601 of
    ``timestamp``), ``question``, ``route``, ``answer_text``,
    ``retrieved_chunk_ids``, ``citations``, ``validation`` (the #13
    ``exchange_log_record`` mapping, carried whole — including
    ``citation_support_rate``), ``usage_records`` (per-call model +
    usage + cost), ``exclude_from_harvest``, and ``feedback`` (None
    until a #56 verdict lands). No other keys; none of
    :data:`FORBIDDEN_IDENTIFIER_FIELDS` at any depth.

    ``exchange_id`` (issue #56): ``None`` — the pre-#56 behaviour —
    mints a fresh random UUID hex; a provided id is carried VERBATIM.
    The chat route mints the id once at stream start, rides it to the
    client in the ``meta`` event, and passes the SAME id here, so the
    id the visitor's thumbs click posts back is the id the logged
    record carries — one id, minted once, joining wire to record. The
    id is random either way: nothing about the content or client ever
    derives it.
    """
    if exchange_id is not None:
        raise NotImplementedError(
            "#56 red phase: explicit exchange_id carriage is pinned in "
            "tests/unit/test_service_feedback.py"
        )
    return {
        # A fresh random id per exchange: the #56 feedback join key. It
        # identifies the exchange, never the person — nothing about the
        # content or client derives it.
        "exchange_id": uuid.uuid4().hex,
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
    }


class ExchangeLog:
    """Append-only JSONL exchange log with a 90-day retention job."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record as a single JSON line."""
        line = json.dumps(dict(record), ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def records(self) -> list[dict[str, Any]]:
        """All records currently retained, in append order."""
        with self._lock:
            if not self.path.is_file():
                return []
            return [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def purge_expired(self) -> int:
        """The retention job: delete records whose ``timestamp`` is older
        than :data:`EXCHANGE_LOG_RETENTION_DAYS` days at the injected
        clock's now; keep everything younger; return the count deleted."""
        cutoff = self._clock() - timedelta(days=EXCHANGE_LOG_RETENTION_DAYS)
        with self._lock:
            if not self.path.is_file():
                return 0
            kept: list[str] = []
            removed = 0
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if datetime.fromisoformat(record["timestamp"]) > cutoff:
                    kept.append(line)
                else:
                    removed += 1
            self.path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
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
        raise NotImplementedError


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
    return [dict(record) for record in records if not record.get("exclude_from_harvest")]


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
