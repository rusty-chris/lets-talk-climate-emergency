"""Fetching: the fail-closed verified fetch and the production transport
(IMPLEMENTATION.md §1 "Fetchers (§2.4)"; reviews #80/#144, issue #145).

One shared implementation of the #80 discipline — fetch to a temp path,
verify the manifest-pinned sha256, and only then atomically rename into
place, deleting refused bytes — used by BOTH `scripts/make_corpus.py`
(where #80 was originally fixed) and `ingestion.pipeline.ingest_corpus`
(which review #144 caught re-implementing the fetch without it). A
sha256 refusal must leave nothing at the path an indexing step would
read; extracting the helper is what keeps the two paths from drifting
again.

The transport seam (IMPLEMENTATION.md §1): tests inject ``file://``
fixture transports; :func:`urllib_transport` is the production
implementation (``https://`` live, ``file://`` for local runs).
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path

from ingestion.manifest import ManifestError, verify_fetched_sha256

__all__ = ["FetchError", "fetch_verified", "urllib_transport"]


class FetchError(RuntimeError):
    """A document fetch failed (network/filesystem). The message names the
    document id and the source URL (review #81: logs must identify the
    document and the failure class without a re-run). Retryable — never a
    licensing verdict.
    """


def urllib_transport(url: str) -> bytes:
    """The production transport: fetch ``url``'s bytes via urllib.

    ``file://`` in tests/local runs, ``https://`` live. OS-level errors
    re-raise as :class:`FetchError` (the retryable class); callers pass
    this — or any injected callable — as the ``transport`` seam.
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - manifest-pinned URLs
            return response.read()
    except OSError as exc:
        raise FetchError(f"fetch failed from {url}: {exc}") from exc


def fetch_verified(
    entry_id: str,
    source_url: str,
    destination: Path,
    expected_sha256: str,
    transport: Callable[[str], bytes],
) -> None:
    """Fetch ``source_url`` and land it at ``destination`` fail-closed
    (review #80, shared per review #144).

    The bytes are written to a ``.part`` temp path first, verified
    against the manifest-pinned sha256, and only then atomically renamed
    into place. On a verification refusal the temp file is DELETED and
    the error re-raised — a mismatch leaves nothing at the path an
    indexing step would read, and no partial artefact survives.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")
    temp_path.write_bytes(transport(source_url))
    try:
        verify_fetched_sha256(entry_id, temp_path, expected_sha256)
    except ManifestError:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(destination)
