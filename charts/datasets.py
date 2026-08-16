"""`make datasets` fetch → verify → parse → validate flow (issue #14, ADR-023).

The imperative shell around :mod:`charts.pack`: fetch each manifest
dataset from its origin URL, verify the bytes against the pinned sha256,
parse with the committed parser, check the manifest's coverage against
the parsed extent, and land the raw file in a **gitignored** data
directory. No dataset file is ever committed (ADR-023): the manifest pins
*which bytes* without hosting them, and provisionally-licensed datasets
(``permitted_context: open-provisional`` — Kaufman #23, Bereiter #45) are
origin-fetch only, never mirrored anywhere.

Reuses — never re-implements — the issue #5 invariants in
:mod:`ingestion.manifest`: :func:`ingestion.manifest.validate_dataset`
for the schema gate and :func:`ingestion.manifest.verify_fetched_sha256`
for the hash gate.

Failure taxonomy (each failure mode must be distinguishable by exception
class, and every message names the offending dataset id):

- :class:`DatasetFetchError` — the origin could not be reached / the
  transfer failed (network-shaped failures);
- :class:`DatasetHashMismatchError` — fetched bytes do not match the
  manifest-pinned sha256;
- :class:`DatasetParseError` — the committed parser refused the fetched
  file;
- :class:`DatasetSchemaError` — the manifest entry itself is refused
  (issue #5 schema violations, missing pack-level fields, or coverage
  metadata that disagrees with parser output — review finding #52).

RED phase: the functions below are contract stubs raising
:class:`NotImplementedError`; the exception classes are the committed
contract surface. Failing tests: ``tests/unit/test_dataset_pack_fetch.py``
and ``tests/integration/test_make_datasets.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

Transport = Callable[[str], bytes]


class DatasetPackError(Exception):
    """Base for every `make datasets` failure. Carries the offending

    ``dataset_id`` as an attribute and prefixes it to the message, so CI
    logs and callers identify the dataset without re-running.
    """

    def __init__(self, dataset_id: str, message: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"{dataset_id}: {message}")


class DatasetFetchError(DatasetPackError):
    """The origin archive could not be reached or the transfer failed."""


class DatasetHashMismatchError(DatasetPackError):
    """Fetched bytes do not match the manifest-pinned sha256 (ADR-023).

    Raised via :func:`ingestion.manifest.verify_fetched_sha256`; the
    message names the ``sha256`` field and both digests.
    """


class DatasetParseError(DatasetPackError):
    """The committed parser refused the fetched file (charts/pack.py

    contract: fail loudly, never silently drop malformed rows).
    """


class DatasetSchemaError(DatasetPackError):
    """The manifest entry is refused: an issue #5 schema violation

    (surfaced from :func:`ingestion.manifest.validate_dataset`, its
    field-naming message preserved), a missing pack-level field, or
    ``coverage`` metadata that disagrees with the parsed extent
    (review finding #52). Message names the violated field(s).
    """


def validate_pack_entry(entry: Mapping[str, Any]) -> None:
    """Pack-level schema gate for one dataset entry (id included).

    Runs :func:`ingestion.manifest.validate_dataset` (issue #5 — reused,
    not re-implemented) and additionally requires the fields the fetch
    flow itself consumes: ``parser`` (resolvable via
    :func:`charts.pack.resolve_parser`), ``time_axis`` (with a ``unit``),
    and ``coverage``. Raises :class:`DatasetSchemaError` naming the
    dataset id and every violated field; returns None when clean. Pure.
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_fetch.py")


def fetch_all(
    manifest_path: Path | str,
    dest_dir: Path | str,
    transport: Transport | None = None,
) -> dict[str, Path]:
    """The `make datasets` flow: fetch, verify, parse, validate, land.

    For the manifest at ``manifest_path`` (schema per issue #5 +
    :func:`validate_pack_entry`):

    1. **Validate every entry first** — any schema refusal raises
       :class:`DatasetSchemaError` *before a single byte is fetched*
       (asserted against an injected transport that records calls).
    2. Fetch each dataset's ``url`` via ``transport`` (default supports
       ``https://`` and ``file://`` — tests source from ``file://``
       synthetic fixtures, IMPLEMENTATION.md §3). Failures raise
       :class:`DatasetFetchError`.
    3. Verify the fetched bytes against the pinned ``sha256`` via
       :func:`ingestion.manifest.verify_fetched_sha256`; a mismatch
       raises :class:`DatasetHashMismatchError`.
    4. Parse with the committed parser the entry's ``parser`` field
       resolves to; a parser refusal raises :class:`DatasetParseError`.
    5. Check the entry's ``coverage`` equals
       :func:`charts.pack.dataset_coverage` of the parsed frame (review
       finding #52); disagreement raises :class:`DatasetSchemaError`
       naming ``coverage``.
    6. Land the verified raw bytes at ``dest_dir/<dataset_id><suffix>``
       (suffix taken from the URL path), creating ``dest_dir`` if
       needed. The default landing directory (``data/datasets/``) is
       gitignored — landed files must never become committable (ADR-023).

    Idempotent: a landed file that already verifies against its pin is
    left untouched (bytes *and* mtime), not re-fetched.

    Returns ``{dataset_id: landed_path}`` for every dataset in the
    manifest — including ``open-provisional`` ones, which are fetchable
    from origin like any other but excluded from every committed or
    mirrored artefact.
    """
    raise NotImplementedError("issue #14 red phase — see tests/unit/test_dataset_pack_fetch.py")
