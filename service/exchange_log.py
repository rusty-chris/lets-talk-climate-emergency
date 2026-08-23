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
- **The #56 seam:** every record carries a random ``exchange_id`` (joins
  a later thumbs-up/down feedback event to the exchange — it identifies
  the exchange, never the person) and a ``feedback`` field, None until
  #56 writes it.
- **Disclosure:** :data:`LOGGING_DISCLOSURE` is the one-line notice the
  chat surface must carry (served to the UI in the chat stream's ``meta``
  event and on the /privacy page).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "LOGGING_DISCLOSURE",
    "EXCHANGE_LOG_RETENTION_DAYS",
    "FORBIDDEN_IDENTIFIER_FIELDS",
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
) -> dict[str, Any]:
    """Pure: one structured exchange record (the §9 logging schema).

    Returns a JSON-serialisable mapping with EXACTLY these top-level
    keys: ``exchange_id`` (fresh random UUID hex — the #56 feedback join
    key; identifies the exchange, never the person), ``timestamp`` (ISO
    8601 of ``timestamp``), ``question``, ``route``, ``answer_text``,
    ``retrieved_chunk_ids``, ``citations``, ``validation`` (the #13
    ``exchange_log_record`` mapping, carried whole — including
    ``citation_support_rate``), ``usage_records`` (per-call model +
    usage + cost), ``exclude_from_harvest``, and ``feedback`` (None
    until #56). No other keys; none of
    :data:`FORBIDDEN_IDENTIFIER_FIELDS` at any depth.
    """
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")


class ExchangeLog:
    """Append-only JSONL exchange log with a 90-day retention job."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        self.path = Path(path)
        self._clock = clock

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record as a single JSON line."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def records(self) -> list[dict[str, Any]]:
        """All records currently retained, in append order."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")

    def purge_expired(self) -> int:
        """The retention job: delete records whose ``timestamp`` is older
        than :data:`EXCHANGE_LOG_RETENTION_DAYS` days at the injected
        clock's now; keep everything younger; return the count deleted."""
        raise NotImplementedError("issue #22: red phase — implementer makes this pass")


def harvest_candidates(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure: the eval-harvest review queue for a set of records.

    Excludes EVERY record with ``exclude_from_harvest`` truthy (unsafe
    and unsafe-suspected exchanges, DESIGN §3.1/§8 — the #10 flag
    consumed here) — exclusion is structural, before any human sees a
    queue. Records flagged negatively by #56 feedback remain candidates
    (that is #56's triage input); harvesting stays hand-review + detach.
    """
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")


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
    raise NotImplementedError("issue #22: red phase — implementer makes this pass")
