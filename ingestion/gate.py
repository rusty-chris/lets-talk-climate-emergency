"""CC-BY licensing gate (issue #6, DESIGN.md §2.2).

RED-phase contract stubs. The failing tests in
tests/unit/test_licensing_gate.py and tests/integration/test_gate_cli.py
define the behaviour; every function below raises NotImplementedError
until the implementer makes them pass. Do not weaken the tests to get to
green (ORCHESTRATION.md).

The gate is the hardened three-step pipeline of DESIGN §2.2:

1. **Candidate filter (automated, pure).** Three licence lookups —
   OpenAlex, Crossref, Unpaywall — are *injected as already-fetched JSON
   mappings* (recorded fixtures in tests; live HTTP only inside the CLI's
   own fetch layer, never in pytest). A document is a candidate only if
   >=2 of the three sources agree on an accepted licence
   (:data:`ACCEPTED_LICENCES`). Automated lookups disagree ~37% of the
   time, so a single source is never authorisation. Disagreement rejects,
   and the rejection records *all three* verdicts so a human can see each
   source's claim.

   A source's CC claim counts **only when it asserts the licence applies
   to the article's published version** — journal-level or TDM-policy CC
   signals on a closed article never count (the hybrid-journal trap):

   - OpenAlex: article-level licence (``primary_location``/
     ``best_oa_location``) only when the work's ``open_access.is_oa`` is
     true.
   - Crossref: ``message.license`` entries whose ``content-version`` is
     ``vor`` or ``am`` — never ``tdm``.
   - Unpaywall: ``best_oa_location.license``.

   Free-to-read is not a licence: ``is_oa: true`` / ``oa_status: bronze``
   with no CC licence anywhere must reject (the Ripple-2019-shaped trap
   this gate exists for).

2. **Publisher-page confirmation.** The actual article page (HTML string
   injected; the CLI fetches it) yields the page's own licence/rights
   statement, captured **verbatim** with its URL into
   :class:`PageEvidence` and ultimately the manifest's
   ``licence_evidence``. A page whose statement contradicts the agreed
   lookup verdict (e.g. "All rights reserved" on a supposedly CC BY
   article) flags the document — it is never admitted automatically.

3. **Human sign-off (required, interactive).** A separate step reading
   from an injected ``input_fn`` writes the ``human_signoff``
   ``{who, date, note}`` record that ingestion.manifest (#5) requires.
   No manifest entry exists without it: :func:`build_manifest_entry`
   refuses (GateError) when the sign-off is missing or the report is not
   a confirmed candidate.

The CLI (``python -m ingestion.gate``) runs the pipeline for one DOI and
writes a single corpus-manifest document entry (YAML mapping) that
``ingestion.manifest.validate_document`` accepts unchanged. Recorded
mode (``--lookups-dir``/``--page-html``/``--page-url``) replays committed
fixtures with no network — the only mode any pytest tier uses.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Licences that qualify a document for candidacy (DESIGN §2.2), as
#: normalised tokens. Everything else — including every NC/ND variant and
#: "free to read" with no licence — is not ingestable via this gate.
ACCEPTED_LICENCES = frozenset({"cc-by", "cc-by-sa", "cc0"})

#: The three lookup sources, in canonical order.
LOOKUP_SOURCES = ("openalex", "crossref", "unpaywall")


class GateError(Exception):
    """A licensing-gate refusal (missing sign-off, non-candidate report,

    malformed inputs). Messages name what is missing/violated.
    """


@dataclass(frozen=True)
class LookupVerdict:
    """One source's licence claim for one DOI.

    ``licence`` is the normalised token (``cc-by``, ``cc-by-sa``, ``cc0``,
    ``cc-by-nc``, ``cc-by-nc-nd``, ...) or ``None`` when the source has no
    record or makes no article-level licence claim. Crossref licence URLs
    (``https://creativecommons.org/licenses/by/4.0/`` ...) normalise to the
    same tokens so verdicts are comparable across sources. ``claim`` is the
    human-readable statement of what the source said (shown in rejection
    output and kept for the evidence trail).
    """

    source: str
    licence: str | None
    claim: str


@dataclass(frozen=True)
class CandidateDecision:
    """The pure >=2-of-3 candidate-filter outcome.

    ``verdicts`` always carries all three sources (silent sources appear
    with ``licence=None``). ``reason`` is human-readable and, on
    rejection, includes every source's claim.
    """

    doi: str
    is_candidate: bool
    agreed_licence: str | None
    verdicts: tuple[LookupVerdict, ...]
    reason: str


@dataclass(frozen=True)
class PageEvidence:
    """The publisher page's licence/rights statement, verbatim, plus the

    URL it was captured from.
    """

    url: str
    statement: str


@dataclass(frozen=True)
class GateReport:
    """Full gate outcome for one DOI.

    ``status`` is one of:

    - ``"candidate"`` — >=2-of-3 agreement AND the publisher page
      confirms; eligible for human sign-off (still not admitted!).
    - ``"rejected"`` — the candidate filter failed.
    - ``"flagged"`` — lookups agreed but the publisher page contradicts
      (or yields no usable statement); needs human investigation, never
      admitted automatically.
    """

    doi: str
    status: str
    decision: CandidateDecision
    page_evidence: PageEvidence | None
    reason: str


def lookup_verdicts(
    doi: str,
    *,
    openalex: Mapping[str, Any] | None,
    crossref: Mapping[str, Any] | None,
    unpaywall: Mapping[str, Any] | None,
) -> tuple[LookupVerdict, LookupVerdict, LookupVerdict]:
    """Parse the three raw lookup responses into per-source verdicts.

    Pure over already-fetched JSON mappings (recorded fixtures in tests).
    ``None`` means the source was silent (no record / 404) and yields a
    ``licence=None`` verdict — always three verdicts, in
    :data:`LOOKUP_SOURCES` order. Article-level-only counting rules per
    the module docstring.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def evaluate_candidate(doi: str, verdicts: Sequence[LookupVerdict]) -> CandidateDecision:
    """The pure >=2-of-3 candidate filter (DESIGN §2.2 step 1).

    Candidate iff at least two verdicts carry the *same* licence token
    and that token is in :data:`ACCEPTED_LICENCES`. Anything else rejects
    with a ``reason`` that names every source and its claim.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def extract_licence_statement(html: str, url: str) -> PageEvidence:
    """Capture the publisher page's licence/rights statement verbatim.

    Pure over the fetched HTML text. Finds the licence or rights
    statement (Creative-Commons wording, "All rights reserved" notices,
    ...) and returns it verbatim together with ``url``. This is the
    evidence a human confirms at sign-off, so paraphrase or truncation is
    a defect.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def check_publisher_page(decision: CandidateDecision, evidence: PageEvidence) -> str:
    """DESIGN §2.2 step 2: does the page confirm the agreed licence?

    Returns ``"confirmed"`` when the captured statement is consistent
    with ``decision.agreed_licence``, ``"flagged"`` when it contradicts
    it (e.g. an all-rights-reserved notice) — the free-to-read trap is
    caught here at the latest.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def gate_document(
    doi: str,
    *,
    openalex: Mapping[str, Any] | None,
    crossref: Mapping[str, Any] | None,
    unpaywall: Mapping[str, Any] | None,
    page_html: str | None = None,
    page_url: str | None = None,
) -> GateReport:
    """Run steps 1-2 for one DOI, pure over injected responses/page.

    Lookup rejection short-circuits to ``status="rejected"``; agreement
    plus a contradicting (or missing/unusable) page yields ``"flagged"``;
    only agreement plus page confirmation yields ``"candidate"``.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def collect_signoff(
    input_fn: Callable[[str], str],
    *,
    today: datetime.date,
) -> dict[str, str]:
    """DESIGN §2.2 step 3: the interactive human sign-off record.

    Prompts (via the injected ``input_fn``, in this order) for **who**
    then **note**, re-prompting on blank answers — an empty sign-off
    would fail the #5 schema, so it is never accepted. Returns exactly
    ``{"who": ..., "date": <today as ISO>, "note": ...}``, the shape
    ``ingestion.manifest`` validates. The clock is injected (``today``),
    never read inside.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def build_manifest_entry(
    report: GateReport,
    signoff: Mapping[str, str] | None,
    *,
    doc_id: str,
    attribution_text: str,
    sha256: str,
    retrieved_at: datetime.date,
) -> dict[str, Any]:
    """Assemble the corpus-manifest document entry for an admitted paper.

    Refuses (:class:`GateError`, naming ``human_signoff`` when that is
    what is missing) unless ``report.status == "candidate"`` **and**
    ``signoff`` is a complete who/date/note record — no document reaches
    the manifest unsigned, and flagged/rejected reports never reach it
    at all.

    The returned mapping passes ``ingestion.manifest.validate_document``
    unchanged: ``licence`` from the agreed verdict (human-readable, e.g.
    "CC BY ..."), ``licence_evidence`` containing the verbatim publisher
    statement **and** the URL it came from, ``canonical_url`` of the form
    ``https://doi.org/<doi>``, ``permitted_context: "open"``,
    ``redistributable: True``, ``source_tier: "A"``, plus the provided
    id/attribution/sha256 (of the fetched artefact bytes)/retrieved_at
    and the sign-off.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the gate for one DOI and write a manifest document entry.

    Usage (recorded mode — the only mode tests use; omitting the
    recorded-input flags makes the CLI fetch live, which no pytest tier
    ever does)::

        python -m ingestion.gate <doi>
            --lookups-dir DIR       # DIR/{openalex,crossref,unpaywall}.json,
                                    #   a missing file = silent source
            --page-html FILE        # recorded publisher page
            --page-url URL          # the URL that page was captured from
            --doc-id ID --attribution TEXT
            --out FILE              # YAML mapping: one document entry

    Behaviour: prints all three lookup verdicts and the verbatim page
    statement (the evidence a human confirms), then runs the interactive
    sign-off on stdin — confirm ``y``/``n``, then who, then note.
    Only a confirmed candidate with a completed sign-off writes ``--out``
    (an entry ``ingestion.manifest.validate_document`` accepts, with
    ``sha256`` of the fetched page bytes and ``retrieved_at`` today);
    rejection, flagging, or a declined sign-off exits non-zero with the
    evidence shown and writes nothing.
    """
    raise NotImplementedError("issue #6 red phase — implementer makes this pass")


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    raise SystemExit(main())
