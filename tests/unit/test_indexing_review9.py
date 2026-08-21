"""Adversarial-review findings on the #9 embedding + indexing feature — RED.

One test (family) per review finding on PR #154, red before each fix:

- #159: an iterator passed as ``chunks`` must not bypass the pre-write
  refusals (the multi-pass checks silently see an exhausted stream).
- #167: the ``__meta`` companion-collection namespace is reserved; a
  chunk build into it must refuse before any write, and a foreign point
  without a ``chunk_id`` payload must fail with a named error.
- #158: ``source_type`` is a closed vocabulary end-to-end — unknown,
  miscased or null values refuse the build, and the query filter helper
  accepts only known values (it never normalises).
- #155: a metadata-only change (unchanged embedding text) must refresh
  the stored payload without re-encoding anything.
- #156: rebuilding under a different embedding model id (or dense_dim)
  must refuse unless the caller explicitly asks for a full reindex —
  never re-stamp the meta over stale vectors.
- #157: a crash between the first store mutation and the final meta
  write must leave the index loudly unqueryable, never confidently
  mislabeled with the previous corpus version.
- #163: the bge-m3 revision is pinned (full commit hash) and forwarded
  to the loader as the resolved snapshot path.

Unit tier: deterministic fakes, in-memory Qdrant, no weights, no network.
"""

from __future__ import annotations

import pytest

from rag.indexing import UnreviewedDegradedChunksError
from tests._indexing_fixtures import (
    COLLECTION,
    RecordingEmbeddingModel,
    build,
    degraded_records,
    fixture_corpus,
    fresh_client,
)


def test_build_materialises_input_before_refusal_checks() -> None:
    """Finding #159: `build_index` iterates its input several times; a
    generator/iterator argument exhausts on the first pass, so the
    needs-hand-review refusal sees an empty stream and unreviewed
    degraded chunks get INDEXED before the run dies late on an
    unrelated TypeError. The iterator case must behave identically to
    the list case: refuse loudly, write nothing."""
    chunks, records = fixture_corpus()
    flagged = degraded_records(records, "syn-idx-basin")
    client = fresh_client()
    model = RecordingEmbeddingModel()

    with pytest.raises(UnreviewedDegradedChunksError, match="syn-idx-basin"):
        build(client, iter(chunks), flagged, model=model)

    assert model.encoded_texts == [], "a refused build must not spend embedding work"
    assert client.collection_exists(COLLECTION) is False, (
        "a refused build must leave the store exactly as it found it — the iterator "
        "input bypassed the pre-write refusals"
    )
