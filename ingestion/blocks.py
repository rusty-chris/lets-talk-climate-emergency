"""Custom-content citation blocks (issue #7, DESIGN §2.4/§3.3).

RED phase: contract stub; the failing tests in
``tests/unit/test_ingestion_blocks.py`` define the behaviour.

Pure: chunks → custom-content document block payloads, one per citable
unit (IMPLEMENTATION.md §1). The payload shape is the one the spike-03
probe proved live against the API (``reviews/spike-03-probe-findings.md``)
and the generation call (#12/#13) consumes as-is::

    {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": <body>}]},
        "title": <display title / attribution>,
        "context": "chunk_id=<id>; header=<context header>; ...",
        "citations": {"enabled": True},
    }

Two lessons from the spike are contractual here:

- the block's citable text is the chunk **body only** — the context
  header travels in ``context`` (passed to the model, never cited), so
  ``cited_text`` quotes pure evidence;
- ``citations: {enabled: true}`` on every block (§3.4 all-or-none; the
  request builders in #12 enforce it again at the request level).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ingestion.pipeline import ChunkRecord

__all__ = ["build_citation_blocks"]


def build_citation_blocks(chunks: Sequence[ChunkRecord]) -> list[dict[str, Any]]:
    """One custom-content document block per chunk (one per citable unit).

    Contract: ``len(result) == len(chunks)``, order preserved; each
    block's content text is exactly ``chunk.body`` (no header); the
    ``context`` field carries the chunk id and context header for
    traceability; ``citations`` is enabled on every block; the block
    carries the chunk's citation metadata (attribution, canonical URL,
    licence, permitted_context, consensus_position, source_type) so the
    service layer never re-joins against the manifest at answer time.
    """
    raise NotImplementedError("issue #7: citation-block emission not implemented yet")
