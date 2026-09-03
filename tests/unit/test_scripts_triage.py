"""Review finding #266 RED — the #56-promised triage script exists.

Issue #56's scope: "Triage script: surfaces thumbed-down exchanges for
hand review; reviewer can promote an exchange into the gold set via the
existing eval-harvest flow." What merged is the pure ordering only
(``service.exchange_log.harvest_candidates`` / ``detach_for_harvest``);
no operator entrypoint was shipped, so a reviewer must write ad-hoc
Python against raw (pre-detachment, PII-plausible) exchange records —
exactly the ad-hoc handling the designed flow exists to prevent.

This suite pins the shipped script, ``scripts/run_harvest_triage.py``
(named per the fix-batch assignment; the finding's sketch said
``triage_feedback.py`` — flagged in the red-phase report), mirroring
``scripts/run_retention.py``'s shape and driven here as a real
subprocess (the operator's actual surface):

- ``run_harvest_triage.py <log_dir>`` prints the triage queue from
  ``<log_dir>/exchanges.jsonl``: thumbs-DOWN exchanges FIRST, safety
  exclusions (``exclude_from_harvest``) never shown — even when the
  excluded exchange was itself thumbed down — and each listed
  candidate's ``exchange_id`` visible (the reviewer needs it to
  promote).
- ``--detach <exchange_id>`` emits the promotion payload as ONE JSON
  object on stdout, exactly ``detach_for_harvest``'s shape: content
  fields only — no ``exchange_id``, ``timestamp``, ``feedback`` or
  ``usage_records`` ever ride into a gold set. Excluded and unknown
  ids are refused with a non-zero exit.
- A missing/empty log is a clean empty queue (exit 0); an unreadable
  log (mid-file garbage — NOT the #265 recoverable torn tail) is a
  clean REFUSAL: non-zero exit, no candidates emitted, no raw
  traceback at the operator.
- DEPLOYMENT.md mentions the script next to the retention runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from service.exchange_log import ExchangeLog, detach_for_harvest
from tests._service_fixtures import FrozenClock
from tests.unit.test_service_privacy_logging import make_record

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_harvest_triage.py"


def run_triage(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), (
        "scripts/run_harvest_triage.py is not shipped — issue #56's triage "
        "script has no operator entrypoint (finding #266)"
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def seed_log(log_dir: Path) -> dict[str, dict]:
    """A log holding one of each triage case, with the thumbs-down
    candidate deliberately NOT first in file order."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = ExchangeLog(log_dir / "exchanges.jsonl", clock=FrozenClock())
    records = {
        "up": make_record(question="An up-voted invented basin question?"),
        "unrated": make_record(question="An unrated invented basin question?"),
        "down": make_record(question="A thumbs-down invented basin question?"),
        "excluded": make_record(
            question="An excluded unsafe invented question?",
            exclude_from_harvest=True,
        ),
    }
    for record in records.values():
        log.append(record)
    log.record_feedback(records["up"]["exchange_id"], "up")
    log.record_feedback(records["down"]["exchange_id"], "down")
    # Thumbed down AND safety-excluded: the verdict never overrides the
    # exclusion — this record must not surface anywhere.
    log.record_feedback(records["excluded"]["exchange_id"], "down")
    return records


def test_triage_script_lists_negative_exchanges(tmp_path) -> None:
    """The original TDD-plan pin: thumbs-down exchanges print FIRST,
    identified by exchange_id; excluded exchanges never appear."""
    records = seed_log(tmp_path)
    result = run_triage(str(tmp_path))
    assert result.returncode == 0, result.stderr
    output = result.stdout

    down, up, unrated = records["down"], records["up"], records["unrated"]
    for record in (down, up, unrated):
        assert record["exchange_id"] in output, (
            f"candidate {record['exchange_id']} missing from the triage "
            "queue — the reviewer cannot promote what is not listed"
        )
    assert output.index(down["exchange_id"]) < output.index(up["exchange_id"])
    assert output.index(down["exchange_id"]) < output.index(unrated["exchange_id"]), (
        "thumbs-down exchanges must surface FIRST — they are the "
        "reviewer's triage input (issue #56 / finding #266)"
    )

    excluded = records["excluded"]
    assert excluded["exchange_id"] not in output
    assert excluded["question"] not in output, (
        "a safety-excluded exchange surfaced in the triage queue — the "
        "exclusion is structural and a thumbs-down never overrides it"
    )


def test_triage_script_survives_an_empty_or_missing_log(tmp_path) -> None:
    missing = run_triage(str(tmp_path / "no-such-dir"))
    assert missing.returncode == 0, missing.stderr
    assert "Traceback" not in missing.stderr + missing.stdout, (
        "a missing log must be a clean empty queue, not a crash"
    )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "exchanges.jsonl").write_text("", encoding="utf-8")
    empty = run_triage(str(empty_dir))
    assert empty.returncode == 0, empty.stderr
    assert "Traceback" not in empty.stderr + empty.stdout


def test_triage_script_refuses_an_unreadable_log(tmp_path) -> None:
    """Mid-file garbage (not a recoverable #265 torn TAIL) means the log
    cannot be trusted: refuse cleanly — non-zero exit, a message rather
    than a traceback, and no candidates emitted from the wreckage."""
    log_dir = tmp_path / "wrecked"
    log_dir.mkdir()
    valid = make_record(question="A valid invented question after garbage?")
    (log_dir / "exchanges.jsonl").write_text(
        "this line was never JSON\n" + json.dumps(valid, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = run_triage(str(log_dir))
    assert result.returncode != 0, (
        "the triage script emitted a queue from an unreadable log — it "
        "must refuse rather than guess (finding #266)"
    )
    assert "Traceback" not in result.stderr + result.stdout, (
        "the refusal must be an operator-facing message, not a raw traceback"
    )
    assert valid["exchange_id"] not in result.stdout


def test_detach_action_emits_only_the_detached_record(tmp_path) -> None:
    """--detach <exchange_id> promotes THROUGH detach_for_harvest: the
    emitted payload is exactly the detached shape — content fields only,
    nothing that could re-join the promoted case to a stored record."""
    records = seed_log(tmp_path)
    down = records["down"]
    result = run_triage(str(tmp_path), "--detach", down["exchange_id"])
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload == detach_for_harvest(down), (
        "the promotion payload must be exactly detach_for_harvest's "
        "output — no exchange_id, no timestamp, no feedback, no "
        "usage_records may ride into a published eval set (finding #266)"
    )
    for forbidden in ("exchange_id", "timestamp", "feedback", "usage_records"):
        assert forbidden not in payload


def test_detach_refuses_excluded_and_unknown_exchanges(tmp_path) -> None:
    records = seed_log(tmp_path)
    excluded = records["excluded"]

    refused = run_triage(str(tmp_path), "--detach", excluded["exchange_id"])
    assert refused.returncode != 0, (
        "--detach bypassed the safety exclusion — the unsafe/"
        "unsafe-suspected wall holds at every surface (finding #266)"
    )
    assert excluded["question"] not in refused.stdout

    unknown = run_triage(str(tmp_path), "--detach", "no-such-exchange-id")
    assert unknown.returncode != 0
    assert "Traceback" not in unknown.stderr + unknown.stdout


def test_deployment_runbook_references_the_triage_script() -> None:
    runbook = (REPO_ROOT / "service" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "run_harvest_triage" in runbook, (
        "DEPLOYMENT.md must point the reviewer at the shipped triage "
        "entrypoint next to the retention runner (finding #266)"
    )
