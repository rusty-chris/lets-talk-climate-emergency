"""The owner severity-audit gate guard (review finding #197).

DESIGN §6.1 specifies *human-annotated* expected severity; issue #20's
acceptance criteria include an owner spot-audit of the severity
annotations. PR #179 discharged that only as prose — nothing tracked
completion, so #21 could build the severity release gate on labels the
owner never confirmed. The fix: a committed audit packet
(``evals/gold/severity-audit-packet.md``) whose header carries
``owner_severity_audit: pending`` until Chris McWilliams (Rusty Data —
author & steward) reviews the 15 annotations and flips it, and a guard
(`evals.severity_audit`) the #21 release severity gate MUST call —
refusing to run while the flag says pending.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals import severity_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = REPO_ROOT / "evals" / "gold" / "severity-audit-packet.md"
QA_PATH = REPO_ROOT / "evals" / "gold" / "climate_qa.yaml"


def _write_packet(tmp_path: Path, header_line: str) -> Path:
    packet = tmp_path / "severity-audit-packet.md"
    packet.write_text(
        f"{header_line}\n\n# Severity audit packet\n\ncontent\n",
        encoding="utf-8",
    )
    return packet


def test_release_severity_gate_refuses_while_owner_audit_pending(tmp_path):
    """The guard the #21 severity gate calls raises loudly while the
    packet header still says pending — unaudited labels never gate a
    release silently."""
    packet = _write_packet(tmp_path, "owner_severity_audit: pending")
    with pytest.raises(severity_audit.SeverityAuditPendingError) as excinfo:
        severity_audit.assert_owner_severity_audit_complete(packet)
    message = str(excinfo.value)
    assert "owner_severity_audit" in message
    assert "pending" in message
    assert str(packet) in message


def test_gate_runs_once_the_owner_flips_the_flag(tmp_path):
    packet = _write_packet(tmp_path, "owner_severity_audit: complete 2026-09-01")
    # No exception: the audit is recorded complete.
    severity_audit.assert_owner_severity_audit_complete(packet)
    assert severity_audit.severity_audit_status(packet) == "complete"


def test_missing_flag_refuses_rather_than_passing(tmp_path):
    """A packet without the flag (or a missing packet) is a malformed
    audit record — refused loudly, never treated as complete."""
    packet = _write_packet(tmp_path, "# no flag here")
    with pytest.raises(severity_audit.SeverityAuditError):
        severity_audit.assert_owner_severity_audit_complete(packet)
    with pytest.raises(severity_audit.SeverityAuditError):
        severity_audit.assert_owner_severity_audit_complete(tmp_path / "absent.md")


def test_unknown_status_refuses(tmp_path):
    packet = _write_packet(tmp_path, "owner_severity_audit: maybe")
    with pytest.raises(severity_audit.SeverityAuditError) as excinfo:
        severity_audit.severity_audit_status(packet)
    assert "maybe" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The committed packet itself
# ---------------------------------------------------------------------------


def test_committed_packet_exists_with_valid_status():
    status = severity_audit.severity_audit_status(PACKET_PATH)
    assert status in {"pending", "complete"}


def test_committed_packet_covers_every_severity_item():
    """The packet the owner reviews must present all 15 severity items —
    id, annotated level, source quote and rationale — so the audit is
    over the full set, not a sample the packet author chose."""
    packet_text = PACKET_PATH.read_text(encoding="utf-8")
    qa_items = yaml.safe_load(QA_PATH.read_text(encoding="utf-8"))["items"]
    severity_items = [i for i in qa_items if i["category"] == "severity"]
    assert len(severity_items) == 15
    for item in severity_items:
        assert item["id"] in packet_text, f"{item['id']} missing from the audit packet"
        annotation = item["severity"]
        section = packet_text.split(item["id"], 1)[1]
        assert annotation["expected_lead"] in section.split("## ", 1)[0], (
            f"{item['id']}: packet must state the annotated level"
        )
