"""`make ingest` implementation (issue #7 acceptance criterion 1; review
finding #145).

Drives the full production ingestion pipeline over the corpus manifest:

    manifest gate -> verified fetch (fail-closed, #80/#144) -> parse
        (HTML direct; PDFs via Docling with the loud PyMuPDF fallback)
        -> structure-aware chunk -> citation metadata -> citation blocks

Outputs:

- ``<corpus-dir>/ingest_run.json`` — the written ingest manifest (#143):
  per-document backend, degraded/hand-review flags, warnings, skips.
- ``<out-dir>/chunks.jsonl`` + ``<out-dir>/blocks.jsonl`` — the chunk
  records and custom-content citation block payloads the indexer (#9)
  consumes. The default out-dir lives under ``data/`` (gitignored):
  chunk text of non-open documents must never sit in the repo tree
  (DESIGN §2.1), and open text is re-fetchable from its pins (ADR-023
  spirit).

The `make corpus` / `make ingest` split is deliberate (recorded here, in
the Makefile and corpus/README.md): `make corpus` remains the fast #5
licensing/invariant gate; `make ingest` is the #7 pipeline and pulls the
heavy Docling parse.

Exit-code taxonomy (same classes as make_corpus, review #81):

    0  ingested cleanly
    1  licensing/invariant or design refusal — must block, never retry
    2  fetch/environment failure — retryable, not a licensing verdict
    3  sha256 mismatch — upstream drift; review before re-pinning
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.fetch import FetchError, urllib_transport  # noqa: E402
from ingestion.manifest import ManifestError, Sha256MismatchError  # noqa: E402
from ingestion.pipeline import IngestError, IngestResult, ingest_corpus  # noqa: E402

EXIT_OK = 0
EXIT_INVARIANT = 1
EXIT_FETCH = 2
EXIT_HASH_MISMATCH = 3


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(manifest_path: Path, corpus_dir: Path, out_dir: Path) -> IngestResult:
    """Run the pipeline and persist chunks/blocks for the indexer."""
    result = ingest_corpus(
        manifest_path,
        corpus_dir,
        transport=urllib_transport,
    )
    _write_jsonl(out_dir / "chunks.jsonl", [dataclasses.asdict(c) for c in result.chunks])
    _write_jsonl(out_dir / "blocks.jsonl", [dict(b) for b in result.blocks])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "corpus" / "manifest.yaml",
        help="Path to the corpus manifest (default: corpus/manifest.yaml)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=REPO_ROOT / "corpus",
        help="Directory open-tier artefacts land in (default: corpus/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ingest",
        help="Directory for chunk/block payloads (default: data/ingest — gitignored)",
    )
    args = parser.parse_args(argv)

    try:
        result = run(args.manifest, args.corpus_dir, args.out_dir)
    except FetchError as exc:
        print(
            f"make ingest: fetch failure (retryable — not a licensing verdict): {exc}",
            file=sys.stderr,
        )
        return EXIT_FETCH
    except Sha256MismatchError as exc:
        print(
            f"make ingest: sha256 mismatch (upstream drift — review before re-pinning): {exc}",
            file=sys.stderr,
        )
        return EXIT_HASH_MISMATCH
    except IngestError as exc:
        print(f"make ingest: pipeline refusal: {exc}", file=sys.stderr)
        return EXIT_INVARIANT
    except ManifestError as exc:
        print(f"make ingest: licensing/invariant refusal: {exc}", file=sys.stderr)
        return EXIT_INVARIANT
    except (OSError, yaml.YAMLError) as exc:
        print(f"make ingest: environment failure: {exc}", file=sys.stderr)
        return EXIT_FETCH

    ingested = [r for r in result.documents.values() if not r.skipped]
    skipped = [r for r in result.documents.values() if r.skipped]
    flagged = [r for r in ingested if r.needs_hand_review]
    print(
        f"make ingest: {len(result.chunks)} chunks / {len(result.blocks)} blocks from "
        f"{len(ingested)} documents ({len(skipped)} recorded skips); "
        f"run record at {args.corpus_dir / 'ingest_run.json'}"
    )
    for record in flagged:
        print(
            f"make ingest: WARNING — {record.doc_id} parsed by the DEGRADED fallback; "
            "hand review required before indexing (DESIGN §2.4)",
            file=sys.stderr,
        )
    for record in result.documents.values():
        for warning in record.warnings:
            print(f"make ingest: warning: {warning}", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
