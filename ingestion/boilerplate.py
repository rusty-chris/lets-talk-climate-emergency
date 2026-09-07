"""HTML boilerplate filtering before indexing (issue #331) — contract
stubs, RED phase.

The corpus-expansion packet (corpus/EXPANSION-SIGNOFF.md, owner-signed
2026-09-07) found that the first HTML sources through the pipeline chunk
cleanly but produce nav/footer/boilerplate chunks — breadcrumbs, menus,
"Follow NASA" social blocks, related-content teasers, the climate.gov
archive banner — roughly a third to half of chunks per page. Real agency
pages carry this furniture in plain ``<div>``/``<ul>`` markup, NOT in the
semantic ``<nav>``/``<header>``/``<footer>`` tags
:class:`ingestion.pipeline._HTMLBuilder` already drops, so it arrives as
ordinary TEXT/LIST_ITEM/HEADING blocks and would pollute retrieval if
indexed.

Seam decision (FLAGGED in the red report): the filter is a **pure
classification step over parsed blocks — post-parse, pre-chunk**. It
consumes and returns :class:`ingestion.parse.StructuredDoc`, so it works
identically for the stdlib ``parse_html`` path and any future
Docling-HTML path, needs no live DOM, and stays unit-testable without
any parser install. The production wiring point is
:func:`ingestion.pipeline.ingest_corpus` step 4, between the HTML parse
and ``chunk_document``. The filter must NOT be folded silently into
``parse_html``'s skip-list: a parse-time drop leaves no audit trail, and
the audit trail is a pinned contract (see below).

Contract points the #331 red suite pins:

- **Scope: HTML sources only.** The filter is self-scoped by
  ``doc.backend``: an ``"html"``-backend document is filtered; ANY other
  backend (``docling``, ``pymupdf``, fixture backends) passes through
  with its block list unchanged and an empty audit. PDF documents'
  pipeline behaviour is byte-identical with the filter in place — the
  scoping lives inside the filter so no caller can misapply it.
- **Drop-only, byte-identical survivors.** The filter only ever drops
  whole blocks. The kept blocks are an in-order subsequence of the input
  blocks with type/text/level/caption untouched — it never rewrites,
  trims or merges text (a citation quotes these bytes).
- **Rule-based classification** of the packet's boilerplate shapes:
  breadcrumb trails, site menu / footer-menu label runs, cookie/consent
  notices, social-follow blocks ("Follow NASA" and its platform-link
  items), related-content teaser lists (a bare "Related"-style heading
  and its teasers), and the verbatim climate.gov archive banner. A
  heading that OPENS a boilerplate section (e.g. "Follow NASA",
  "Related") is dropped with its items — it must not survive as an
  empty pseudo-section.
- **Fail-open honesty.** An ambiguous block is KEPT, never silently
  dropped — the hand-audit checklist catches residue; a lost substantive
  chunk is unrecoverable. In particular: substantive list items (claims
  with numbers) are not menus; a prose sentence merely *containing* a
  marker phrase ("… follow NASA-style calibration …") is not a social
  block; a genuine section heading merely containing a label word
  ("Related warming feedbacks") is not a teaser rail — the label must BE
  the heading (the #149 anchoring rule, applied here).
- **Audit trail, never silent.** Every dropped block is recorded in the
  returned :class:`BoilerplateAudit` — reason (from the closed
  :data:`BOILERPLATE_REASONS` vocabulary), block type, and the dropped
  text byte-identical. The pipeline surfaces the per-document count and
  reasons on :class:`ingestion.pipeline.DocumentIngestRecord`
  (``boilerplate_dropped`` / ``boilerplate_reasons``, entries formatted
  ``"<reason>: <dropped text>"``) and therefore in the written ingest
  report ``corpus/ingest_run.json``.
- **Determinism.** Identical input → identical output document and
  identical audit, across calls and across runs (the idempotent
  re-embedding hook must survive the filter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.parse import StructuredDoc

__all__ = [
    "BOILERPLATE_REASONS",
    "DroppedBlock",
    "BoilerplateAudit",
    "filter_html_boilerplate",
]

#: The closed reason vocabulary — every dropped block carries exactly one.
#:
#: - ``banner``     — site-status banners (the climate.gov archive banner);
#: - ``breadcrumb`` — breadcrumb-trail crumbs ("Home", "News & Features", …);
#: - ``cookie``     — cookie/consent notices;
#: - ``menu``       — site-menu / footer-menu label runs ("Privacy Policy", …);
#: - ``related``    — related-content teaser rails and their teasers;
#: - ``social``     — social-follow blocks ("Follow NASA" + platform items).
BOILERPLATE_REASONS = frozenset({"banner", "breadcrumb", "cookie", "menu", "related", "social"})


@dataclass(frozen=True)
class DroppedBlock:
    """One block the filter removed: the audit-trail unit.

    ``reason`` is a member of :data:`BOILERPLATE_REASONS`; ``block_type``
    is the :class:`~ingestion.parse.BlockType` value string; ``text`` is
    the dropped block's text byte-identical (the hand audit must be able
    to see exactly what was removed).
    """

    reason: str
    block_type: str
    text: str


@dataclass(frozen=True)
class BoilerplateAudit:
    """Per-document account of every drop — count + reasons, never silent.

    ``dropped`` holds one :class:`DroppedBlock` per removed block, in
    document order. ``dropped_count`` is its length;
    :meth:`report_lines` renders the ``"<reason>: <text>"`` lines the
    pipeline copies onto ``DocumentIngestRecord.boilerplate_reasons``
    (and thence into ``corpus/ingest_run.json``).
    """

    doc_id: str
    dropped: tuple[DroppedBlock, ...] = field(default_factory=tuple)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def report_lines(self) -> tuple[str, ...]:
        """One ``"<reason>: <text>"`` line per dropped block, in order."""
        return tuple(f"{item.reason}: {item.text}" for item in self.dropped)


def filter_html_boilerplate(doc: StructuredDoc) -> tuple[StructuredDoc, BoilerplateAudit]:
    """Drop nav/footer/social/related/banner boilerplate blocks from an
    HTML-parsed document (issue #331); return ``(filtered_doc, audit)``.

    See the module docstring for the full pinned contract: HTML-backend
    scoping (every other backend passes through unchanged with an empty
    audit), drop-only byte-identical survivors, rule-based classification
    of the packet's boilerplate shapes, fail-open keeps for ambiguous
    blocks, a complete per-drop audit trail, and determinism.
    """
    raise NotImplementedError(
        "issue #331 red phase: the HTML boilerplate filter is not implemented yet"
    )
