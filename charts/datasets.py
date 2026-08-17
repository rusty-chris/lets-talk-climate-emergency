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
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from charts.pack import dataset_coverage, resolve_parser
from ingestion.manifest import ManifestError, validate_dataset, verify_fetched_sha256

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


def _strip_id_prefix(entry_id: str, message: str) -> str:
    """Undo a ManifestError's own ``"{entry_id}: "`` prefix before

    re-wrapping the message in a DatasetPackError subclass (which adds
    its own identical prefix) — avoids a doubled ``id: id: ...`` message.
    """
    prefix = f"{entry_id}: "
    return message[len(prefix) :] if message.startswith(prefix) else message


def validate_pack_entry(entry: Mapping[str, Any]) -> None:
    """Pack-level schema gate for one dataset entry (id included).

    Runs :func:`ingestion.manifest.validate_dataset` (issue #5 — reused,
    not re-implemented) and additionally requires the fields the fetch
    flow itself consumes: ``parser`` (resolvable via
    :func:`charts.pack.resolve_parser`), ``time_axis`` (with a ``unit``),
    and ``coverage``. Raises :class:`DatasetSchemaError` naming the
    dataset id and every violated field; returns None when clean. Pure.
    """
    entry_id = entry.get("id") or "<unknown>"
    violations: list[str] = []

    try:
        validate_dataset(entry)
    except ManifestError as exc:
        violations.append(_strip_id_prefix(entry_id, str(exc)))

    parser_ref = entry.get("parser")
    if not parser_ref:
        violations.append("missing required field 'parser'")
    else:
        try:
            resolve_parser(parser_ref)
        except ValueError as exc:
            violations.append(f"parser field {parser_ref!r} does not resolve: {exc}")

    time_axis = entry.get("time_axis")
    if not time_axis or not isinstance(time_axis, Mapping) or not time_axis.get("unit"):
        violations.append("missing required field 'time_axis' (with a 'unit')")

    if not entry.get("coverage"):
        violations.append("missing required field 'coverage'")

    if violations:
        raise DatasetSchemaError(entry_id, "; ".join(violations))


def _default_transport(url: str) -> bytes:
    """Default transport: ``https://``/``http://`` over the network,

    ``file://`` for local synthetic fixtures (tests, IMPLEMENTATION.md §3).
    """
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        return path.read_bytes()
    if parsed.scheme in ("http", "https"):
        request = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": "lets-talk-climate-emergency dataset fetcher"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read()
    raise ValueError(f"unsupported URL scheme {parsed.scheme!r} for {url!r}")


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
    manifest_path = Path(manifest_path)
    dest_dir = Path(dest_dir)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    datasets_raw: Mapping[str, Mapping[str, Any]] = raw.get("datasets") or {}

    # Pass 1: validate every entry before a single byte moves.
    entries: dict[str, dict[str, Any]] = {}
    for ds_id, entry in datasets_raw.items():
        entry_with_id = {**entry, "id": ds_id}
        validate_pack_entry(entry_with_id)
        entries[ds_id] = entry_with_id

    transport_fn: Transport = transport or _default_transport
    dest_dir.mkdir(parents=True, exist_ok=True)

    landed: dict[str, Path] = {}
    for ds_id, entry in entries.items():
        url = entry["url"]
        expected_sha256 = entry["sha256"]
        suffix = Path(urlparse(url).path).suffix
        landed_path = dest_dir / f"{ds_id}{suffix}"

        if landed_path.is_file():
            try:
                verify_fetched_sha256(ds_id, landed_path, expected_sha256)
            except ManifestError:
                pass  # stale/mismatched landed file — refetch below
            else:
                landed[ds_id] = landed_path
                continue

        try:
            content = transport_fn(url)
        except DatasetPackError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/transport failures are heterogeneous
            raise DatasetFetchError(ds_id, f"could not fetch {url}: {exc}") from exc

        fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=f".{ds_id}-", suffix=suffix or ".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)

            try:
                verify_fetched_sha256(ds_id, tmp_path, expected_sha256)
            except ManifestError as exc:
                raise DatasetHashMismatchError(ds_id, _strip_id_prefix(ds_id, str(exc))) from exc

            parser = resolve_parser(entry["parser"])
            try:
                frame = parser(tmp_path)
            except ValueError as exc:
                raise DatasetParseError(ds_id, str(exc)) from exc

            computed_coverage = dataset_coverage(frame, entry["time_axis"])
            if computed_coverage != entry["coverage"]:
                raise DatasetSchemaError(
                    ds_id,
                    f"coverage {entry['coverage']} disagrees with the parsed usable extent "
                    f"{computed_coverage} (review finding #52) — regenerate coverage from "
                    "parser output",
                )

            os.replace(tmp_path, landed_path)
            landed[ds_id] = landed_path
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return landed
