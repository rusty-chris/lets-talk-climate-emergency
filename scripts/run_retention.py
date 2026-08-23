"""The shipped exchange-log retention runner (review finding #213).

DEPLOYMENT.md §7 tells the operator to run the 90-day exchange-log
retention job on a daily schedule; this is that script. Usage:

    uv run python scripts/run_retention.py <CLIMATE_CHAT_LOG_DIR>

It drives :func:`service.retention.purge_exchange_log` over the
file-backed ``exchanges.jsonl`` and prints the count removed. It
deliberately does NOT touch the rate-limit store: that store is
in-process memory, reachable only by the service's own scheduled purge
(``service.retention.run_retention_pass`` in the app lifespan) — an
external process constructing a fresh ``RateLimiter`` would purge an
empty list and prove nothing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from service.retention import purge_exchange_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        type=Path,
        help="The service log directory (CLIMATE_CHAT_LOG_DIR) holding exchanges.jsonl",
    )
    args = parser.parse_args()
    removed = purge_exchange_log(args.log_dir, clock=lambda: datetime.now(UTC))
    print(f"exchange-log retention: removed {removed} expired record(s)")


if __name__ == "__main__":
    main()
