"""`make voices` implementation — ingest the voices layer (issue #8).

Drives the voices layer (``voices/voices.yaml``, DESIGN §2.5) through the
#7 production chunker into ``source_type: voices`` chunks + citation
blocks, alongside (never mixed into) the scientific corpus. ``make
ingest`` runs this after the corpus so a full ingest produces
voices-labelled chunks; ``make voices`` runs it alone.

Outputs (default ``data/ingest`` — gitignored, like the corpus ingest):

- ``<out-dir>/voices_chunks.jsonl`` — the voices ChunkRecords the indexer
  (#9) consumes; every one is labelled ``source_type: voices``.
- ``<out-dir>/voices_blocks.jsonl`` — one custom-content citation block
  per voices chunk, under the "About the movement" attribution.

Kept in separate files from ``chunks.jsonl`` deliberately: the corpus
ingest's output contract (one doc's chunks per run) is not disturbed, and
the voices/evidence separation stays visible at the file level too.

Exit codes mirror the corpus ingest (review #81):

    0  ingested cleanly
    1  voices.yaml schema refusal (missing field, undated snapshot fact …)
    2  environment failure (file/parse)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.pipeline import IngestResult  # noqa: E402
from voices.render import VoicesError, ingest_voices  # noqa: E402

EXIT_OK = 0
EXIT_INVARIANT = 1
EXIT_ENV = 2


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(voices_path: Path, out_dir: Path) -> IngestResult:
    result = ingest_voices(voices_path)
    _write_jsonl(out_dir / "voices_chunks.jsonl", [dataclasses.asdict(c) for c in result.chunks])
    _write_jsonl(out_dir / "voices_blocks.jsonl", [dict(b) for b in result.blocks])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voices",
        type=Path,
        default=REPO_ROOT / "voices" / "voices.yaml",
        help="Path to voices.yaml (default: voices/voices.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ingest",
        help="Directory for voices chunk/block payloads (default: data/ingest — gitignored)",
    )
    args = parser.parse_args(argv)

    if not args.voices.is_file():
        # No voices file is a clean no-op, not a failure: `make ingest`
        # must not break in a checkout that has no voices layer yet.
        print(f"make voices: no voices file at {args.voices} — nothing to ingest")
        return EXIT_OK

    try:
        result = run(args.voices, args.out_dir)
    except VoicesError as exc:
        print(f"make voices: voices.yaml schema refusal: {exc}", file=sys.stderr)
        return EXIT_INVARIANT
    except OSError as exc:
        print(f"make voices: environment failure: {exc}", file=sys.stderr)
        return EXIT_ENV

    labelled = sum(1 for c in result.chunks if c.source_type == "voices")
    print(
        f"make voices: {len(result.chunks)} chunks / {len(result.blocks)} blocks from "
        f"{len(result.documents)} entities — all {labelled} chunks labelled source_type: voices; "
        f"payloads at {args.out_dir / 'voices_chunks.jsonl'}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
