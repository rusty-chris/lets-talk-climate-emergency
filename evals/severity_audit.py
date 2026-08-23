"""Owner severity-audit status guard (review finding #197).

DESIGN §6.1 promises *human-annotated* expected severity; the gold
set's 15 severity annotations were agent-authored, and the owner audit
is what converts them into the design's human-annotated artefact. The
audit lives in ``evals/gold/severity-audit-packet.md``: its header
carries ``owner_severity_audit: pending`` until the owner — Chris
McWilliams (Rusty Data — author & steward) — reviews the annotations
and flips the line to ``owner_severity_audit: complete <date>``.

The #21 release severity gate MUST call
:func:`assert_owner_severity_audit_complete` before scoring: while the
flag says pending the gate refuses to run, so releases can never be
blocked or passed against severity labels no human confirmed. A
missing packet or a malformed/unknown flag refuses loudly too — an
absent record is never treated as a completed audit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "evals" / "gold" / "severity-audit-packet.md"

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
_VALID_STATUSES = (STATUS_PENDING, STATUS_COMPLETE)

_FLAG_PATTERN = re.compile(r"^owner_severity_audit:\s*(\S+)", re.MULTILINE)


class SeverityAuditError(RuntimeError):
    """The audit packet is missing or its status flag is malformed —
    refused loudly, never treated as a completed audit."""


class SeverityAuditPendingError(SeverityAuditError):
    """The owner severity audit has not been completed yet: the release
    severity gate must not run (finding #197)."""


def severity_audit_status(packet_path: Path = PACKET_PATH) -> str:
    """The packet's recorded status: ``pending`` or ``complete``.

    Raises :class:`SeverityAuditError` when the packet is missing, the
    ``owner_severity_audit:`` line is absent, or the status value is
    not a known one.
    """
    if not packet_path.exists():
        raise SeverityAuditError(
            f"severity audit packet not found at {packet_path} — the owner "
            "audit record is required before the severity gate can run "
            "(finding #197)"
        )
    text = packet_path.read_text(encoding="utf-8")
    match = _FLAG_PATTERN.search(text)
    if match is None:
        raise SeverityAuditError(
            f"severity audit packet {packet_path} carries no "
            "'owner_severity_audit:' header line — a missing flag is a "
            "malformed audit record, never a completed one (finding #197)"
        )
    status = match.group(1)
    if status not in _VALID_STATUSES:
        raise SeverityAuditError(
            f"severity audit packet {packet_path} records "
            f"owner_severity_audit: {status!r}; expected one of "
            f"{_VALID_STATUSES} (finding #197)"
        )
    return status


def assert_owner_severity_audit_complete(packet_path: Path = PACKET_PATH) -> None:
    """Refuse (:class:`SeverityAuditPendingError`) while the packet
    header still says ``owner_severity_audit: pending``; return None
    once the owner has flipped it to complete. The #21 release severity
    gate calls this before scoring."""
    status = severity_audit_status(packet_path)
    if status == STATUS_PENDING:
        raise SeverityAuditPendingError(
            f"the owner severity audit is still pending ({packet_path} "
            "header: 'owner_severity_audit: pending') — the release "
            "severity gate refuses to run on unaudited labels; the owner "
            "reviews the packet and flips the flag to 'complete <date>' "
            "(finding #197, DESIGN §6.1 'human-annotated')"
        )
    return None
