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

Exit-code taxonomy (review finding #81 — GNU make reports any failed
recipe with its own exit 2, so run this script directly where the class
matters, e.g. in a CI retry wrapper):

    0  all invariants passed
    1  licensing/invariant refusal — must block, never retry
    2  fetch/environment failure — retryable, not a licensing verdict
    3  sha256 mismatch — upstream drift; review before re-pinning
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.fetch import FetchError, fetch_verified, urllib_transport  # noqa: E402
from ingestion.manifest import (  # noqa: E402
    ManifestError,
    Sha256MismatchError,
    check_prepared_text_shipping,
    load_corpus_manifest,
    verify_fetched_sha256,
)

EXIT_OK = 0
EXIT_INVARIANT = 1
EXIT_FETCH = 2
EXIT_HASH_MISMATCH = 3


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
            # and the refused bytes are deleted, not quarantined. The
            # mechanism is the SHARED helper (review #144) so this script
            # and ingestion.pipeline.ingest_corpus cannot drift apart.
            fetch_verified(
                document.id,
                document.source_url,
                destination,
                document.sha256,
                partial(urllib_transport, label=document.id),
            )
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
    except FetchError as exc:
        print(
            f"make corpus: fetch failure (retryable — not a licensing verdict): {exc}",
            file=sys.stderr,
        )
        return EXIT_FETCH
    except Sha256MismatchError as exc:
        print(
            f"make corpus: sha256 mismatch (upstream drift — review before re-pinning): {exc}",
            file=sys.stderr,
        )
        return EXIT_HASH_MISMATCH
    except ManifestError as exc:
        print(f"make corpus: licensing/invariant refusal: {exc}", file=sys.stderr)
        return EXIT_INVARIANT
    except (OSError, yaml.YAMLError) as exc:
        print(
            f"make corpus: environment failure reading {args.manifest}: {exc}",
            file=sys.stderr,
        )
        return EXIT_FETCH

    print("make corpus: all invariants passed.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
