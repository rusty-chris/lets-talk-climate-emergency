"""The shipped eval-harvest triage runner (review finding #266).

Issue #56 promised a triage script — "surfaces thumbed-down exchanges for
hand review; reviewer can promote an exchange into the gold set via the
existing eval-harvest flow" — but only the pure library half shipped
(``service.exchange_log.harvest_candidates`` / ``detach_for_harvest``),
leaving a reviewer to write ad-hoc Python against raw, PII-plausible
exchange records. This is that operator entrypoint, mirroring
``scripts/run_retention.py``. Usage:

    # List the triage queue (thumbs-down first, safety exclusions hidden):
    uv run python scripts/run_harvest_triage.py <CLIMATE_CHAT_LOG_DIR>

    # Promote one exchange through the irreversible detachment step:
    uv run python scripts/run_harvest_triage.py <CLIMATE_CHAT_LOG_DIR> \\
        --detach <exchange_id>

The listing applies ``harvest_candidates`` (unsafe/unsafe-suspected
exchanges are never shown — a thumbs-down never overrides the exclusion —
and thumbed-down exchanges surface first). ``--detach`` emits exactly
``detach_for_harvest``'s payload as one JSON object on stdout: content
fields only, so no exchange_id, timestamp, feedback or usage_records can
ride into a published gold set. Excluded and unknown ids are refused with
a non-zero exit. A missing or empty log is a clean empty queue; a log with
mid-file corruption (not a recoverable #265 torn tail) is a clean refusal,
never a raw traceback at the operator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from service.exchange_log import (
    FEEDBACK_DOWN,
    ExchangeLog,
    detach_for_harvest,
    harvest_candidates,
)


def _load_records(log_dir: Path) -> list[dict]:
    """The retained records, or a clean operator-facing refusal on an
    unreadable (mid-file corrupt) log — never a raw traceback."""
    log = ExchangeLog(Path(log_dir) / "exchanges.jsonl", clock=lambda: datetime.now(UTC))
    try:
        return log.records()
    except (json.JSONDecodeError, ValueError, OSError) as error:
        print(
            f"refusing to triage {log.path}: the log is unreadable "
            f"({error}). This is not a recoverable torn trailing line; "
            "inspect the file before harvesting.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _list_queue(log_dir: Path) -> None:
    candidates = harvest_candidates(_load_records(log_dir))
    if not candidates:
        print("Harvest triage queue is empty (no promotable exchanges).")
        return
    print(f"Harvest triage queue: {len(candidates)} exchange(s), thumbs-down first.")
    for candidate in candidates:
        verdict = (candidate.get("feedback") or {}).get("verdict")
        marker = "DOWN" if verdict == FEEDBACK_DOWN else (verdict or "—")
        print(f"[{marker}] {candidate['exchange_id']}: {candidate.get('question', '')}")


def _detach(log_dir: Path, exchange_id: str) -> None:
    for record in _load_records(log_dir):
        if record.get("exchange_id") != exchange_id:
            continue
        try:
            payload = detach_for_harvest(record)
        except ValueError as error:
            print(f"refusing to detach {exchange_id}: {error}", file=sys.stderr)
            raise SystemExit(1) from None
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(
        f"no promotable exchange with id {exchange_id!r} in the log (unknown or already purged).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        type=Path,
        help="The service log directory (CLIMATE_CHAT_LOG_DIR) holding exchanges.jsonl",
    )
    parser.add_argument(
        "--detach",
        metavar="EXCHANGE_ID",
        help="Emit the detached promotion payload for one exchange id and exit",
    )
    args = parser.parse_args()
    if args.detach is not None:
        _detach(args.log_dir, args.detach)
    else:
        _list_queue(args.log_dir)


if __name__ == "__main__":
    main()
