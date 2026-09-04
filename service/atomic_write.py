"""Single-source atomic file replacement (finding #302, durability per #265).

Every durable on-disk write in the service goes through here so the
temp-write + fsync + ``os.replace`` choreography — and its crash-survival
guarantee — is defined exactly once, not re-accreted per journal. Both the
#265 exchange-log rewrite and the #217 budget spend-journal use it, so the
journal that must survive a crash-loop can never again drift away from the
fsync the exchange log documents as load-bearing.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes, *, fsync: bool = True) -> None:
    """Replace ``path`` with ``data`` atomically.

    Write to a sibling ``<name>.tmp`` file in the same directory (so the
    rename is same-filesystem and therefore atomic), fsync it to disk when
    ``fsync`` (the default — a power loss must not install unflushed, e.g.
    empty, data behind the rename; finding #265), then ``os.replace`` it
    into place. A crash before the replace leaves any OLD file
    byte-identical; a failed replace deletes the temp file and re-raises.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(data)
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())
    try:
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = True,
) -> None:
    """Atomically replace ``path`` with ``text`` (see :func:`atomic_write_bytes`)."""
    atomic_write_bytes(path, text.encode(encoding), fsync=fsync)
