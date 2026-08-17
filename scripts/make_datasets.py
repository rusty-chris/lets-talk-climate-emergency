"""`make datasets` implementation (issue #14, ADR-023).

Fetches every dataset in the chart-pack manifest from its origin URL,
verifies the bytes against the pinned sha256, parses with the committed
production parser, checks the manifest's `coverage` against the parsed
usable extent (review finding #52), and lands the verified raw file in a
gitignored directory. No dataset file is ever committed — the manifest
pins which bytes without hosting them (ADR-023).

Invoked via `make datasets` (see the repo-root Makefile); this module is
also runnable directly for debugging:

    uv run python scripts/make_datasets.py \\
        --manifest datasets/manifest.yaml --datasets-dir data/datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from charts.datasets import DatasetPackError, fetch_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifest.yaml",
        help="Path to the dataset manifest (default: datasets/manifest.yaml)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=REPO_ROOT / "data" / "datasets",
        help="Gitignored directory to land fetched files in (default: data/datasets)",
    )
    args = parser.parse_args(argv)

    try:
        landed = fetch_all(args.manifest, args.datasets_dir)
    except DatasetPackError as exc:
        print(f"make datasets: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except (OSError, yaml.YAMLError) as exc:
        print(f"make datasets: {exc}", file=sys.stderr)
        return 1

    for dataset_id, path in sorted(landed.items()):
        print(f"make datasets: {dataset_id} -> {path}")
    print(f"make datasets: {len(landed)} dataset(s) fetched, verified and landed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
