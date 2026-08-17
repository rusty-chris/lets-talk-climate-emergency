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

import argparse
import datetime
import hashlib
import html as html_lib
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


#: CC licence URL fragments -> normalised token, checked most-specific first
#: so e.g. a "by-nc-nd" URL is never mistaken for bare "by".
_CC_URL_TOKENS: tuple[tuple[str, str], ...] = (
    ("publicdomain/zero", "cc0"),
    ("by-nc-nd", "cc-by-nc-nd"),
    ("by-nc-sa", "cc-by-nc-sa"),
    ("by-sa", "cc-by-sa"),
    ("by-nc", "cc-by-nc"),
    ("by-nd", "cc-by-nd"),
)


def _normalise_cc_url(url: str | None) -> str | None:
    """Normalise a Creative-Commons licence URL to a short token, or None."""
    if not url:
        return None
    lowered = url.lower()
    for fragment, token in _CC_URL_TOKENS:
        if fragment in lowered:
            return token
    if "/by/" in lowered or lowered.rstrip("/").endswith("/by"):
        return "cc-by"
    return None


def _openalex_licence(raw: Mapping[str, Any] | None) -> tuple[str | None, str]:
    if not raw:
        return None, "no OpenAlex record"
    open_access = raw.get("open_access") or {}
    is_oa = bool(open_access.get("is_oa"))
    if not is_oa:
        return (
            None,
            f"OpenAlex: open_access.is_oa=False (oa_status={open_access.get('oa_status')!r})",
        )
    best = raw.get("best_oa_location") or {}
    primary = raw.get("primary_location") or {}
    licence = best.get("license") or primary.get("license")
    if licence:
        return licence, f"OpenAlex: open_access.is_oa=True, licence={licence!r}"
    return (
        None,
        "OpenAlex: open_access.is_oa=True but no article-level licence recorded "
        f"(oa_status={open_access.get('oa_status')!r})",
    )


def _crossref_licence(raw: Mapping[str, Any] | None) -> tuple[str | None, str]:
    if not raw:
        return None, "no Crossref record"
    message = raw.get("message") or {}
    entries = message.get("license") or []
    considered: list[tuple[str | None, str]] = []
    for entry in entries:
        version = entry.get("content-version")
        url = entry.get("URL", "")
        considered.append((version, url))
        if version in {"vor", "am"}:
            token = _normalise_cc_url(url)
            if token:
                return token, f"Crossref: license URL {url} (content-version={version!r})"
    if considered:
        details = "; ".join(f"content-version={v!r} URL={u}" for v, u in considered)
        return (
            None,
            "Crossref: license entries present but none apply to the published version "
            f"({details})",
        )
    return None, "Crossref: no license entries"


def _unpaywall_licence(raw: Mapping[str, Any] | None) -> tuple[str | None, str]:
    if not raw:
        return None, "no Unpaywall record"
    best = raw.get("best_oa_location") or {}
    licence = best.get("license")
    if licence:
        return licence, f"Unpaywall: best_oa_location.license={licence!r}"
    return (
        None,
        f"Unpaywall: is_oa={raw.get('is_oa')!r}, oa_status={raw.get('oa_status')!r}, "
        "no licence recorded",
    )


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
    oa_licence, oa_claim = _openalex_licence(openalex)
    cr_licence, cr_claim = _crossref_licence(crossref)
    up_licence, up_claim = _unpaywall_licence(unpaywall)
    return (
        LookupVerdict(source="openalex", licence=oa_licence, claim=oa_claim),
        LookupVerdict(source="crossref", licence=cr_licence, claim=cr_claim),
        LookupVerdict(source="unpaywall", licence=up_licence, claim=up_claim),
    )


def evaluate_candidate(doi: str, verdicts: Sequence[LookupVerdict]) -> CandidateDecision:
    """The pure >=2-of-3 candidate filter (DESIGN §2.2 step 1).

    Candidate iff at least two verdicts carry the *same* licence token
    and that token is in :data:`ACCEPTED_LICENCES`. Anything else rejects
    with a ``reason`` that names every source and its claim.
    """
    verdicts = tuple(verdicts)
    agreeing_sources: dict[str, list[str]] = {}
    for verdict in verdicts:
        if verdict.licence in ACCEPTED_LICENCES:
            agreeing_sources.setdefault(verdict.licence, []).append(verdict.source)

    agreed_licence = next(
        (licence for licence, sources in agreeing_sources.items() if len(sources) >= 2), None
    )
    is_candidate = agreed_licence is not None

    summary = "; ".join(f"{v.source}: {v.licence or 'no accepted licence claim'}" for v in verdicts)
    if is_candidate:
        agreeing = ", ".join(agreeing_sources[agreed_licence])
        reason = f"{agreed_licence} candidate — {agreeing} agree ({summary})"
    else:
        reason = f"no two sources agree on an accepted licence ({summary})"

    return CandidateDecision(
        doi=doi,
        is_candidate=is_candidate,
        agreed_licence=agreed_licence,
        verdicts=verdicts,
        reason=reason,
    )


#: Paragraph-level text likely to be (or contain) a licence/rights statement.
_STATEMENT_TRIGGER_PHRASES = (
    "creative commons",
    "rights reserved",
    "public domain",
    "cc0",
    "cc by",
    "licence",
    "license",
)

_TAG_RE = re.compile(r"<[^>]+>")
_PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def _clean_html_text(raw: str) -> str:
    text = _TAG_RE.sub("", raw)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_licence_statement(html: str, url: str) -> PageEvidence:
    """Capture the publisher page's licence/rights statement verbatim.

    Pure over the fetched HTML text. Finds the licence or rights
    statement (Creative-Commons wording, "All rights reserved" notices,
    ...) and returns it verbatim together with ``url``. This is the
    evidence a human confirms at sign-off, so paraphrase or truncation is
    a defect.
    """
    for match in _PARAGRAPH_RE.finditer(html):
        text = _clean_html_text(match.group(1))
        lowered = text.lower()
        if any(phrase in lowered for phrase in _STATEMENT_TRIGGER_PHRASES):
            return PageEvidence(url=url, statement=text)
    raise GateError(f"no licence/rights statement found on the publisher page at {url}")


#: Phrases that contradict any Creative-Commons claim outright.
_NEGATION_PHRASES = (
    "all rights reserved",
    "no creative commons",
    "no open licence",
    "no open license",
    "not licensed under",
    "not licenced under",
)

#: Phrases confirming a given agreed licence token, in the publisher-page text.
_LICENCE_CONFIRMATION_HINTS: dict[str, tuple[str, ...]] = {
    "cc-by-sa": (
        "creative commons attribution-sharealike",
        "creative commons attribution share alike",
        "attribution-sharealike",
        "attribution share alike",
    ),
    "cc-by": ("creative commons attribution",),
    "cc0": ("cc0", "public domain dedication", "no rights reserved"),
}


def check_publisher_page(decision: CandidateDecision, evidence: PageEvidence) -> str:
    """DESIGN §2.2 step 2: does the page confirm the agreed licence?

    Returns ``"confirmed"`` when the captured statement is consistent
    with ``decision.agreed_licence``, ``"flagged"`` when it contradicts
    it (e.g. an all-rights-reserved notice) — the free-to-read trap is
    caught here at the latest.
    """
    lowered = evidence.statement.lower()
    if any(phrase in lowered for phrase in _NEGATION_PHRASES):
        return "flagged"
    hints = _LICENCE_CONFIRMATION_HINTS.get(decision.agreed_licence or "", ())
    if hints and any(hint in lowered for hint in hints):
        return "confirmed"
    return "flagged"


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
    verdicts = lookup_verdicts(doi, openalex=openalex, crossref=crossref, unpaywall=unpaywall)
    decision = evaluate_candidate(doi, verdicts)

    if not decision.is_candidate:
        return GateReport(
            doi=doi,
            status="rejected",
            decision=decision,
            page_evidence=None,
            reason=decision.reason,
        )

    if not page_html or not page_url:
        return GateReport(
            doi=doi,
            status="flagged",
            decision=decision,
            page_evidence=None,
            reason="no publisher-page evidence provided; a candidate is never admitted unconfirmed",
        )

    try:
        evidence = extract_licence_statement(page_html, page_url)
    except GateError as exc:
        return GateReport(
            doi=doi, status="flagged", decision=decision, page_evidence=None, reason=str(exc)
        )

    if check_publisher_page(decision, evidence) == "confirmed":
        return GateReport(
            doi=doi,
            status="candidate",
            decision=decision,
            page_evidence=evidence,
            reason=f"{decision.reason}; publisher page confirms {decision.agreed_licence}",
        )

    return GateReport(
        doi=doi,
        status="flagged",
        decision=decision,
        page_evidence=evidence,
        reason=(
            f"publisher page contradicts the agreed licence {decision.agreed_licence!r}: "
            f"{evidence.statement}"
        ),
    )


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
    who = ""
    while not who:
        who = input_fn("Sign-off — who verified this licence? ").strip()

    note = ""
    while not note:
        note = input_fn("Sign-off — note (what was checked)? ").strip()

    return {"who": who, "date": today.isoformat(), "note": note}


#: Human-readable licence labels for the manifest's `licence` field.
_LICENCE_LABELS: dict[str, str] = {
    "cc-by": "CC BY 4.0",
    "cc-by-sa": "CC BY-SA 4.0",
    "cc0": "CC0 1.0",
}


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
    if report.status != "candidate":
        raise GateError(
            f"{doc_id}: cannot build a manifest entry for a {report.status!r} report "
            "(only a confirmed candidate may be admitted)"
        )

    if not signoff:
        raise GateError(
            f"{doc_id}: human_signoff is required before a candidate reaches the manifest"
        )

    missing = [field for field in ("who", "date", "note") if not signoff.get(field)]
    if missing:
        raise GateError(
            f"{doc_id}: human_signoff is missing required subfields: {', '.join(missing)}"
        )

    if report.page_evidence is None:
        # Should be unreachable for a "candidate" status, but guard anyway —
        # a candidate is only ever produced alongside confirming page evidence.
        raise GateError(f"{doc_id}: no publisher-page evidence recorded for a confirmed candidate")

    licence_label = _LICENCE_LABELS.get(
        report.decision.agreed_licence, report.decision.agreed_licence
    )
    licence_evidence = f"{report.page_evidence.statement} (source: {report.page_evidence.url})"

    return {
        "id": doc_id,
        "licence": licence_label,
        "licence_evidence": licence_evidence,
        "attribution_text": attribution_text,
        "canonical_url": f"https://doi.org/{report.doi}",
        "redistributable": True,
        "permitted_context": "open",
        "source_tier": "A",
        "sha256": sha256,
        "retrieved_at": retrieved_at,
        "human_signoff": dict(signoff),
    }


def _load_recorded_lookups(lookups_dir: Path) -> dict[str, dict[str, Any] | None]:
    """Recorded mode: DIR/{source}.json per :data:`LOOKUP_SOURCES`; a missing
    file means that source was silent — mirrors the fixture convention the
    unit tests use.
    """
    lookups: dict[str, dict[str, Any] | None] = {}
    for source in LOOKUP_SOURCES:
        path = lookups_dir / f"{source}.json"
        lookups[source] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return lookups


# ---------------------------------------------------------------------------
# Live HTTP fetch layer — the only part of this module that touches the
# network. No pytest tier ever calls these; the CLI test always passes
# --lookups-dir/--page-html so `main` never reaches them (module docstring,
# IMPLEMENTATION.md §1/§3).
# ---------------------------------------------------------------------------


def _http_get_json(url: str, *, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _http_get_text(url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def _fetch_openalex_live(doi: str, *, email: str | None) -> dict[str, Any] | None:
    headers = {"User-Agent": f"lets-talk-climate-emergency-gate (mailto:{email})" if email else ""}
    try:
        return _http_get_json(f"https://api.openalex.org/works/doi:{doi}", headers=headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _fetch_crossref_live(doi: str, *, email: str | None) -> dict[str, Any] | None:
    url = f"https://api.crossref.org/works/{doi}"
    if email:
        url += f"?mailto={email}"
    try:
        return _http_get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _fetch_unpaywall_live(doi: str, *, email: str | None) -> dict[str, Any] | None:
    if not email:
        raise GateError("live Unpaywall lookups require --email (Unpaywall's API mandates it)")
    try:
        return _http_get_json(f"https://api.unpaywall.org/v2/{doi}?email={email}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _fetch_lookups_live(doi: str, *, email: str | None) -> dict[str, dict[str, Any] | None]:
    return {
        "openalex": _fetch_openalex_live(doi, email=email),
        "crossref": _fetch_crossref_live(doi, email=email),
        "unpaywall": _fetch_unpaywall_live(doi, email=email),
    }


def _fetch_page_live(url: str) -> bytes:
    return _http_get_text(url, headers={"User-Agent": "lets-talk-climate-emergency-gate"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ingestion.gate", description=__doc__)
    parser.add_argument("doi", help="The paper's DOI, e.g. 10.1093/biosci/biz088")
    parser.add_argument(
        "--lookups-dir",
        type=Path,
        default=None,
        help="Directory of recorded lookup JSON (DIR/{openalex,crossref,unpaywall}.json); "
        "omit to fetch live",
    )
    parser.add_argument(
        "--page-html",
        type=Path,
        default=None,
        help="Recorded publisher page file; omit to fetch --page-url live",
    )
    parser.add_argument("--page-url", required=True, help="The publisher article page URL")
    parser.add_argument("--doc-id", required=True, dest="doc_id", help="The manifest document id")
    parser.add_argument(
        "--attribution",
        required=True,
        dest="attribution_text",
        help="The attribution_text to record",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Where to write the manifest entry YAML"
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Contact email for live lookups (required for live Unpaywall calls; "
        "ignored in recorded mode)",
    )
    return parser


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
    args = _build_arg_parser().parse_args(argv)

    try:
        if args.lookups_dir is not None:
            lookups = _load_recorded_lookups(args.lookups_dir)
        else:
            lookups = _fetch_lookups_live(
                args.doi, email=args.email
            )  # pragma: no cover - live path

        if args.page_html is not None:
            page_bytes = args.page_html.read_bytes()
            page_html = page_bytes.decode("utf-8")
        else:
            page_bytes = _fetch_page_live(args.page_url)  # pragma: no cover - live path
            page_html = page_bytes.decode("utf-8", errors="replace")
    except (OSError, GateError, urllib.error.URLError) as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return 1

    report = gate_document(args.doi, **lookups, page_html=page_html, page_url=args.page_url)

    print(f"DOI: {args.doi}")
    print("Lookup verdicts:")
    for verdict in report.decision.verdicts:
        print(f"  {verdict.source}: {verdict.claim}")
    print(
        "Candidate filter: "
        f"{'PASSED' if report.decision.is_candidate else 'FAILED'} — {report.decision.reason}"
    )
    if report.page_evidence is not None:
        print("Publisher-page statement (verbatim):")
        print(f"  {report.page_evidence.statement}")
        print(f"  (source: {report.page_evidence.url})")
    print(f"Gate status: {report.status}")

    if report.status != "candidate":
        print(f"gate: {report.reason}", file=sys.stderr)
        return 1

    confirmation = (
        input("Admit this document as a licensing-gate candidate, pending your sign-off? [y/n] ")
        .strip()
        .lower()
    )
    if confirmation != "y":
        print("gate: declined at confirmation; nothing written.", file=sys.stderr)
        return 1

    signoff = collect_signoff(input, today=datetime.date.today())

    try:
        entry = build_manifest_entry(
            report,
            signoff,
            doc_id=args.doc_id,
            attribution_text=args.attribution_text,
            sha256=hashlib.sha256(page_bytes).hexdigest(),
            retrieved_at=datetime.date.today(),
        )
    except GateError as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(entry, sort_keys=False), encoding="utf-8")
    print(f"Manifest entry written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    raise SystemExit(main())
