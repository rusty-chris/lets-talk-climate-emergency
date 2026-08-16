"""`make corpus` implementation (issue #5, ADR-023).

Re-fetches the manifest's `open`-tier document text into a corpus
directory, verifies every fetched file against its manifest-pinned
sha256, and runs the full set of `ingestion.manifest` licensing
invariants — failing loudly, naming the offending document, on any
violation. The repo pins which bytes without hosting them; the public
archives (or, in tests, `file://` fixtures) are the data store.

Invoked via `make corpus` (see the repo-root Makefile); this module is
also runnable directly for debugging:

    uv run python scripts/make_corpus.py --manifest corpus/manifest.yaml --corpus-dir corpus
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.manifest import (  # noqa: E402
    ManifestError,
    check_prepared_text_shipping,
    load_corpus_manifest,
    verify_fetched_sha256,
)


def _fetch(source_url: str) -> bytes:
    """Fetch bytes from a manifest `source_url` (file:// in tests, https:// live)."""
    with urllib.request.urlopen(source_url) as response:  # noqa: S310
        return response.read()


def run(manifest_path: Path, corpus_dir: Path) -> None:
    """Fetch open-tier text, verify hashes, and run every licensing invariant.

    Raises :class:`ingestion.manifest.ManifestError` naming the offending
    document/field on any violation. Parses the manifest exactly once,
    via `load_corpus_manifest` — the ship check runs over the same typed
    records the fetch loop used, never a second read of the file.
    """
    manifest = load_corpus_manifest(manifest_path)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for document in manifest.documents:
        if document.permitted_context != "open" or not document.path:
            continue
        destination = corpus_dir / document.path
        if document.source_url:
            # Fail closed (review #80): fetch to a temp path, verify, and
            # only then atomically rename into place — a sha256 mismatch
            # must leave nothing at the path an indexing step would read,
            # and the refused bytes are deleted, not quarantined.
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination.with_name(destination.name + ".part")
            temp_path.write_bytes(_fetch(document.source_url))
            try:
                verify_fetched_sha256(document.id, temp_path, document.sha256)
            except ManifestError:
                temp_path.unlink(missing_ok=True)
                raise
            temp_path.replace(destination)
        else:
            # Committed open text (path, no source_url — the in-repo shape
            # §2.1 permits): the pin says *which bytes* (ADR-023) however
            # the bytes arrived, so the committed file must verify too;
            # a missing or tampered file refuses naming the document.
            verify_fetched_sha256(document.id, destination, document.sha256)

    check_prepared_text_shipping(
        (
            {"id": doc.id, "path": doc.path, "permitted_context": doc.permitted_context}
            for doc in manifest.documents
        ),
        corpus_dir,
    )


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
        help="Directory to fetch open-tier text into (default: corpus/)",
    )
    args = parser.parse_args(argv)

    try:
        run(args.manifest, args.corpus_dir)
    except ManifestError as exc:
        print(f"make corpus: {exc}", file=sys.stderr)
        return 1
    except (OSError, yaml.YAMLError) as exc:
        print(f"make corpus: {exc}", file=sys.stderr)
        return 1

    print("make corpus: all invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
