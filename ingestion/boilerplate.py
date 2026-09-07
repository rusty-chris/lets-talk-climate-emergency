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

import re
from dataclasses import dataclass, field

from ingestion.parse import HEADING_TYPES, BlockType, StructuredDoc

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


#: Structural classification (NOT a content-keyword filter): the
#: class/id/role tokens ``parse_html`` records on a block's
#: ``source_container`` name the furniture wrapper a real agency page carries
#: it in. Ordered specific-before-generic; the first match wins. Substantive
#: prose lives outside any classed container, so it never matches here.
_CONTAINER_REASON_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"cookie|consent"), "cookie"),
    (re.compile(r"breadcrumb"), "breadcrumb"),
    (re.compile(r"banner"), "banner"),
    (re.compile(r"social|share|follow"), "social"),
    (re.compile(r"related"), "related"),
    (re.compile(r"menu|navbar|nav\b|footer|masthead|utility|sidebar|toolbar"), "menu"),
)

#: A heading that IS a social-follow label ("Follow NASA", "Connect with
#: us") opens a social block whose following list items are platform links.
#: Anchored to the heading START and gated by a short word count (#149): a
#: substantive heading merely beginning with the word is not a rail.
_SOCIAL_HEADING_RE = re.compile(r"(?i)^(?:follow|connect|stay connected|find us)\b")

#: A heading that IS a bare related-content label opens a teaser rail. The
#: label must be the WHOLE heading (#149): "Related" is a rail, but "Related
#: warming feedbacks" is a genuine section and survives (fail-open).
_RELATED_HEADING_RE = re.compile(
    r"(?i)^(?:related(?:\s+(?:content|stories|articles|topics|links|resources|reading))?"
    r"|more\s+(?:from|on|stories)|see\s+also|you\s+may\s+also\s+like|explore\s+more)\s*$"
)


def _container_reason(hint: str | None) -> str | None:
    """The boilerplate reason a block's container hint names, or None."""
    if not hint:
        return None
    for pattern, reason in _CONTAINER_REASON_RULES:
        if pattern.search(hint):
            return reason
    return None


def _heading_reason(text: str) -> str | None:
    """The boilerplate reason a HEADING opens (social/related), or None.

    Fail-open: only a heading that IS the label — short and anchored — opens
    a section; a qualified heading merely containing a label word is kept.
    """
    if _RELATED_HEADING_RE.match(text):
        return "related"
    if _SOCIAL_HEADING_RE.match(text) and len(text.split()) <= 5:
        return "social"
    return None


def filter_html_boilerplate(doc: StructuredDoc) -> tuple[StructuredDoc, BoilerplateAudit]:
    """Drop nav/footer/social/related/banner boilerplate blocks from an
    HTML-parsed document (issue #331); return ``(filtered_doc, audit)``.

    See the module docstring for the full pinned contract: HTML-backend
    scoping (every other backend passes through unchanged with an empty
    audit), drop-only byte-identical survivors, rule-based classification
    of the packet's boilerplate shapes, fail-open keeps for ambiguous
    blocks, a complete per-drop audit trail, and determinism.

    Classification is structural, never a content-keyword scan: a block is
    dropped only when its markup container names the furniture (banner,
    cookie, menu, breadcrumb) or when it is a list item under a heading that
    IS a social-follow / related-content label. Everything else — including
    prose that merely mentions a marker phrase and genuine sections that
    contain a label word — is KEPT for the hand audit to judge.
    """
    if doc.backend != "html":
        return doc, BoilerplateAudit(doc_id=doc.doc_id)

    kept = []
    dropped: list[DroppedBlock] = []
    section_reason: str | None = None  # an open social/related teaser rail
    for block in doc.blocks:
        if block.type in HEADING_TYPES:
            section_reason = _heading_reason(block.text)
            if section_reason is not None:
                dropped.append(DroppedBlock(section_reason, block.type.value, block.text))
            else:
                kept.append(block)
            continue

        reason = _container_reason(block.source_container)
        if reason is None and section_reason is not None and block.type is BlockType.LIST_ITEM:
            # A platform-link / teaser item beneath an open boilerplate rail.
            reason = section_reason
        else:
            # A non-list-item, or a block the container reclassifies, closes
            # the rail so it can never swallow later substantive content.
            section_reason = None

        if reason is not None:
            dropped.append(DroppedBlock(reason, block.type.value, block.text))
        else:
            kept.append(block)

    filtered = StructuredDoc(doc_id=doc.doc_id, title=doc.title, blocks=kept, backend=doc.backend)
    return filtered, BoilerplateAudit(doc_id=doc.doc_id, dropped=tuple(dropped))
