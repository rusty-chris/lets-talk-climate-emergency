"""CC-BY licensing-gate behaviour (issue #6, TDD plan items 1-8).

RED phase for `ingestion/gate.py` — the hardened DESIGN §2.2 gate:
>=2-of-3 lookup agreement as a *candidate filter* (never authorisation),
publisher-page evidence captured verbatim, and a mandatory interactive
human sign-off before anything reaches the manifest that #5 enforces.

Seams under test (IMPLEMENTATION.md §1): the three lookup sources
(OpenAlex/Crossref/Unpaywall) and the page fetch are injected — every
test here feeds recorded synthetic fixtures from
tests/fixtures/licensing_gate/ (invented DOIs on the 10.5555 test prefix,
.invalid-TLD URLs, fictional publishers; the SYNTHETIC FIXTURE discipline
of IMPLEMENTATION.md §5, made load-bearing by the meta-test at the
bottom). No test in any pytest tier makes a live lookup.

The regression the issue names — `test_ripple_2019_free_to_read_rejected`
— pins the failure shape of Ripple et al. 2019 (10.1093/biosci/biz088):
free-to-read (bronze OA, `is_oa: true` everywhere) under an
all-rights-reserved publisher model. The fixtures are a synthetic
stand-in modelled on that case, not the real OUP metadata or page text.

Gate output feeds the merged #5 schema: entries must pass
`ingestion.manifest.validate_document` unchanged — the gate reuses that
contract rather than duplicating it.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from ingestion.gate import (
    ACCEPTED_LICENCES,
    GateError,
    build_manifest_entry,
    collect_signoff,
    evaluate_candidate,
    extract_licence_statement,
    gate_document,
    lookup_verdicts,
)
from ingestion.manifest import validate_document

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "licensing_gate"

CLEAN_DOI = "10.5555/aurelian-syn.2024.0001"
DISAGREEMENT_DOI = "10.5555/aurelian-syn.2023.0407"
TRAP_DOI = "10.5555/synthetic-warning.2019.trap"
HYBRID_DOI = "10.5555/hybridjournal-syn.2025.0042"

CLEAN_PAGE_URL = "https://synthpress.example.invalid/articles/aurelian-syn-2024-0001"
TRAP_PAGE_URL = "https://synthpress.example.invalid/articles/synthetic-warning-2019"

#: The exact statement in pages/clean_cc_by.html — capture must be verbatim.
CLEAN_STATEMENT = (
    "This is an open access article distributed under the terms of the "
    "Creative Commons Attribution 4.0 licence (CC BY 4.0), which permits "
    "unrestricted reuse, distribution and reproduction in any medium, "
    "provided the original work is properly cited."
)

#: The exact statement in pages/free_to_read_trap.html.
TRAP_STATEMENT = (
    "© 2019 Synthetic University Press. All rights reserved. This article "
    "is made free to read on the publisher platform; no Creative Commons "
    "or other open licence applies. For permissions, contact "
    "permissions@synthpress.example.invalid."
)

SIGNOFF = {
    "who": "Test Signer",
    "date": "2026-08-16",
    "note": "Verified the publisher page statement matches CC BY.",
}

ENTRY_META = {
    "doc_id": "syn-gate-clean-review",
    "attribution_text": "Solari, A. & Okoye, B. (2024). Attribution of Aurelian Basin "
    "drying: a synthetic review. Synthetic Reviews of Climate (fictional). CC BY 4.0.",
    "sha256": "a" * 64,
    "retrieved_at": datetime.date(2026, 8, 16),
}


def _lookup(case: str, source: str) -> dict | None:
    """A recorded lookup response, or None when the source was silent."""
    path = GATE_FIXTURES / "lookups" / case / f"{source}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _lookups(case: str) -> dict[str, dict | None]:
    return {source: _lookup(case, source) for source in ("openalex", "crossref", "unpaywall")}


def _page(name: str) -> str:
    return (GATE_FIXTURES / "pages" / f"{name}.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1 — the >=2-of-3 candidate filter (TDD plan items 1-4)
# ---------------------------------------------------------------------------


def test_two_of_three_agreement_admits_candidate():
    """TDD plan item 1: OpenAlex + Crossref agreeing on CC BY with

    Unpaywall silent (no recorded response) clears the candidate filter.
    Silence is not a veto — but the silent source still appears in the
    verdicts, so the evidence trail always shows all three.
    """
    verdicts = lookup_verdicts(CLEAN_DOI, **_lookups("clean_cc_by"))
    decision = evaluate_candidate(CLEAN_DOI, verdicts)

    assert decision.is_candidate
    assert decision.agreed_licence == "cc-by"
    assert decision.agreed_licence in ACCEPTED_LICENCES
    assert len(decision.verdicts) == 3
    assert {v.source for v in decision.verdicts} == {"openalex", "crossref", "unpaywall"}
    silent = next(v for v in decision.verdicts if v.source == "unpaywall")
    assert silent.licence is None


def test_no_two_source_agreement_rejects_with_evidence():
    """TDD plan item 2: three different claims (cc-by-nc-nd / cc-by-nc /

    cc-by) mean no two sources agree on an accepted licence — rejected,
    and the decision's reason shows each source's claim so a human can
    see exactly why. Crossref's licence URL normalises to the same token
    vocabulary as the other sources; a single accepted-licence vote is
    never enough.
    """
    verdicts = lookup_verdicts(DISAGREEMENT_DOI, **_lookups("three_way_disagreement"))
    decision = evaluate_candidate(DISAGREEMENT_DOI, verdicts)

    assert not decision.is_candidate
    assert decision.agreed_licence is None
    by_source = {v.source: v for v in decision.verdicts}
    assert by_source["openalex"].licence == "cc-by-nc-nd"
    assert by_source["crossref"].licence == "cc-by-nc"
    assert by_source["unpaywall"].licence == "cc-by"
    for source, token in [
        ("openalex", "cc-by-nc-nd"),
        ("crossref", "cc-by-nc"),
        ("unpaywall", "cc-by"),
    ]:
        assert source in decision.reason
        assert token in decision.reason


def test_ripple_2019_free_to_read_rejected():
    """TDD plan item 3 — the regression this gate exists for (issue #6

    acceptance criterion). Synthetic stand-in for Ripple et al. 2019
    (10.1093/biosci/biz088): every lookup says free-to-read (bronze OA,
    is_oa true) but none asserts a CC licence, and the publisher page
    says all rights reserved. Free-to-read must never be conflated with
    an open licence: the gate rejects at the candidate stage or flags at
    the publisher-page stage — under no path does the document come out
    an admitted candidate.
    """
    report = gate_document(
        TRAP_DOI,
        **_lookups("free_to_read_trap"),
        page_html=_page("free_to_read_trap"),
        page_url=TRAP_PAGE_URL,
    )

    assert report.status in {"rejected", "flagged"}
    assert report.status != "candidate"
    # No source made an accepted-licence claim; OA-ness alone earned no vote.
    assert all(v.licence not in ACCEPTED_LICENCES for v in report.decision.verdicts)
    assert not report.decision.is_candidate or report.status == "flagged"


GREEN_OA_DOI = "10.5555/greenoa-syn.2025.0100"


def test_repository_preprint_licence_does_not_count_for_vor():
    """Review finding #97: for a green-OA article the best_oa_location in

    BOTH OpenAlex and Unpaywall is the repository preprint — one
    repository submittedVersion must never supply two agreeing votes for
    a closed version of record. A location's licence counts only when it
    asserts version == "publishedVersion"; the excluded repository claim
    still appears in the verdict's claim text so the human sees it.
    """
    verdicts = lookup_verdicts(GREEN_OA_DOI, **_lookups("green_oa_repository"))
    decision = evaluate_candidate(GREEN_OA_DOI, verdicts)

    assert not decision.is_candidate
    assert decision.agreed_licence is None
    by_source = {v.source: v for v in verdicts}
    assert by_source["openalex"].licence is None
    assert by_source["unpaywall"].licence is None
    # The excluded repository claim is recorded, not silently dropped.
    assert "submittedVersion" in by_source["openalex"].claim
    assert "cc-by" in by_source["openalex"].claim
    assert "submittedVersion" in by_source["unpaywall"].claim
    assert "cc-by" in by_source["unpaywall"].claim


def test_accepted_manuscript_version_does_not_count():
    """Review finding #97: an accepted manuscript (acceptedVersion) is not

    the version of record either — same exclusion as the submitted
    preprint.
    """
    lookups = _lookups("green_oa_repository")
    lookups["openalex"]["best_oa_location"]["version"] = "acceptedVersion"
    lookups["unpaywall"]["best_oa_location"]["version"] = "acceptedVersion"

    verdicts = lookup_verdicts(GREEN_OA_DOI, **lookups)
    decision = evaluate_candidate(GREEN_OA_DOI, verdicts)

    assert not decision.is_candidate
    by_source = {v.source: v for v in verdicts}
    assert by_source["openalex"].licence is None
    assert by_source["unpaywall"].licence is None


def test_crossref_am_licence_does_not_count():
    """Review finding #97: one counting rule for all three sources — only

    the published version. A Crossref licence entry with content-version
    "am" (accepted manuscript) earns no vote, mirroring the
    version-of-record-only rule applied to OpenAlex/Unpaywall locations.
    """
    crossref = {
        "message": {
            "DOI": GREEN_OA_DOI,
            "license": [
                {
                    "URL": "https://creativecommons.org/licenses/by/4.0/",
                    "content-version": "am",
                    "delay-in-days": 0,
                    "start": {"date-parts": [[2025, 2, 1]]},
                }
            ],
        }
    }
    verdicts = lookup_verdicts(GREEN_OA_DOI, openalex=None, crossref=crossref, unpaywall=None)
    by_source = {v.source: v for v in verdicts}
    assert by_source["crossref"].licence is None


def test_hybrid_journal_article_level_licence_wins():
    """TDD plan item 4: a subscription article in a hybrid journal whose

    metadata carries journal-level CC BY signals (stray OpenAlex licence
    on a closed work; Crossref licence entry with content-version "tdm")
    must be rejected — only claims about the article's published version
    count toward agreement.
    """
    verdicts = lookup_verdicts(HYBRID_DOI, **_lookups("hybrid_journal"))
    decision = evaluate_candidate(HYBRID_DOI, verdicts)

    assert not decision.is_candidate
    assert decision.agreed_licence is None


def test_retracted_work_flags_never_candidate():
    """Review finding #99: is_retracted is in the OpenAlex response the

    gate already parses — clean CC-BY agreement plus a confirming page
    must still flag when the work is retracted, with the reason naming
    the retraction, and build_manifest_entry must refuse even with a
    sign-off. A retracted paper can be perfectly CC-BY; the gate is the
    only per-DOI metadata checkpoint before ingestion.
    """
    lookups = _lookups("clean_cc_by")
    lookups["openalex"]["is_retracted"] = True

    report = gate_document(
        CLEAN_DOI,
        **lookups,
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )

    assert report.status == "flagged"
    assert report.status != "candidate"
    assert "retract" in report.reason.lower()
    with pytest.raises(GateError):
        build_manifest_entry(report, SIGNOFF, **ENTRY_META)


def test_crossref_retraction_update_flags():
    """Review finding #99: Crossref announces retractions as update-to

    relations — an update-to entry of type retraction/withdrawal flags
    the work even when OpenAlex has not caught up.
    """
    lookups = _lookups("clean_cc_by")
    lookups["crossref"]["message"]["update-to"] = [
        {
            "DOI": "10.5555/aurelian-syn.2026.retraction",
            "type": "retraction",
            "label": "Retraction",
            "updated": {"date-parts": [[2026, 5, 1]]},
        }
    ]

    report = gate_document(
        CLEAN_DOI,
        **lookups,
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )

    assert report.status == "flagged"
    assert "retract" in report.reason.lower()


# ---------------------------------------------------------------------------
# Step 2 — publisher-page evidence (TDD plan items 5-6)
# ---------------------------------------------------------------------------


def test_publisher_page_licence_statement_captured_verbatim():
    """TDD plan item 5: the page's licence statement is captured verbatim

    (no paraphrase, no truncation) with the URL it came from, and both
    land in the manifest entry's licence_evidence — the record the human
    confirms at sign-off and the audit trail afterwards.
    """
    evidence = extract_licence_statement(_page("clean_cc_by"), CLEAN_PAGE_URL)
    assert CLEAN_STATEMENT in evidence.statement
    assert evidence.url == CLEAN_PAGE_URL

    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"
    entry = build_manifest_entry(report, SIGNOFF, **ENTRY_META)
    assert CLEAN_STATEMENT in entry["licence_evidence"]
    assert CLEAN_PAGE_URL in entry["licence_evidence"]


def test_publisher_page_contradicting_lookups_flags():
    """TDD plan item 6: agreeing lookups are still only a candidate

    filter — a publisher page carrying an all-rights-reserved statement
    contradicts them, so the document is flagged (statement captured for
    the human) and can never be admitted, signed or not.
    """
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("free_to_read_trap"),
        page_url=TRAP_PAGE_URL,
    )

    assert report.status == "flagged"
    assert report.page_evidence is not None
    assert "All rights reserved" in report.page_evidence.statement
    with pytest.raises(GateError):
        build_manifest_entry(report, SIGNOFF, **ENTRY_META)


def _page_with_statement(statement: str) -> str:
    """A minimal synthetic publisher page wrapping one licence statement."""
    return (
        "<!-- SYNTHETIC FIXTURE — inline test page, fictional publisher -->\n"
        "<html><body>\n"
        "<section><h2>Copyright and licence</h2>\n"
        f'<p class="licence-statement">{statement}</p>\n'
        "</section></body></html>\n"
    )


@pytest.mark.parametrize(
    "rider_statement",
    [
        # Real NC/ND/SA publisher statements open with exactly the words
        # "Creative Commons Attribution", so substring confirmation of
        # cc-by is defeated by every rider variant (review #95).
        "This is an open access article distributed under the terms of the "
        "Creative Commons Attribution-NonCommercial 4.0 License, which "
        "permits non-commercial re-use provided the original work is "
        "properly cited.",
        "This is an open access article distributed under the terms of the "
        "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 "
        "License, which permits non-commercial reproduction in any medium "
        "provided no modifications are made.",
        "This is an open access article distributed under the terms of the "
        "Creative Commons Attribution-NoDerivatives 4.0 License.",
        "This is an open access article distributed under the terms of the "
        "Creative Commons Attribution-ShareAlike 4.0 License.",
        "This article is available under the CC BY-NC-ND 4.0 licence.",
    ],
)
def test_publisher_page_nc_rider_flags_despite_cc_by_agreement(rider_statement: str):
    """Review finding #95: a page statement carrying an NC/ND/SA rider is a

    *contradiction* of agreed cc-by, never a confirmation — the page's
    detected licence variant must equal the agreed token exactly. An NC
    document admitted as open/redistributable breaks the §2.1 invariants,
    so the flagged report must also refuse a manifest entry even when
    signed.
    """
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page_with_statement(rider_statement),
        page_url=CLEAN_PAGE_URL,
    )

    assert report.status == "flagged", rider_statement
    with pytest.raises(GateError):
        build_manifest_entry(report, SIGNOFF, **ENTRY_META)


def test_negation_anywhere_on_page_flags():
    """Review finding #96: the negation/rights scan must consider the whole

    page, not just the first trigger-matching paragraph. A per-figure CC
    credit before the rights notice previously shadowed an
    all-rights-reserved statement into a false confirmation.
    """
    page = (
        "<!-- SYNTHETIC FIXTURE — inline test page, fictional publisher -->\n"
        "<html><body>\n"
        "<p>Figure 2 is reproduced from Doe et al. (fictional) under a "
        "Creative Commons Attribution licence.</p>\n"
        "<p>Copyright 2024 Synth Press (fictional). All rights reserved.</p>\n"
        "</body></html>\n"
    )
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=page,
        page_url=CLEAN_PAGE_URL,
    )

    assert report.status == "flagged"
    assert report.status != "candidate"
    assert report.page_evidence is not None
    assert "All rights reserved" in report.page_evidence.statement
    with pytest.raises(GateError):
        build_manifest_entry(report, SIGNOFF, **ENTRY_META)


def test_evidence_is_the_rights_statement_not_first_trigger_match():
    """Review finding #96, evidence integrity: when a figure-credit CC

    paragraph precedes the page's own "Copyright and licence" section,
    the captured licence_evidence must be the rights-section statement —
    the record the human confirms — never the first trigger match.
    """
    evidence = extract_licence_statement(_page("figure_credit_rights_section"), CLEAN_PAGE_URL)
    assert CLEAN_STATEMENT in evidence.statement
    assert "Figure 2" not in evidence.statement

    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("figure_credit_rights_section"),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"
    assert report.page_evidence is not None
    assert CLEAN_STATEMENT in report.page_evidence.statement
    assert "Figure 2" not in report.page_evidence.statement


def test_conflicting_page_statements_flag_with_all_shown():
    """Review finding #96: two paragraphs asserting *different* CC variants

    (a CC BY figure credit vs a CC BY-NC rights statement) are a
    contradiction — flag, never confirm, and the human sees the conflict.
    """
    page = (
        "<!-- SYNTHETIC FIXTURE — inline test page, fictional publisher -->\n"
        "<html><body>\n"
        "<p>Figure 3 is reproduced from Roe et al. (fictional) under a "
        "Creative Commons Attribution licence.</p>\n"
        "<h2>Copyright and licence</h2>\n"
        "<p>This article is distributed under the terms of the Creative "
        "Commons Attribution-NonCommercial 4.0 License.</p>\n"
        "</body></html>\n"
    )
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=page,
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "flagged"


def test_publisher_page_exact_licence_still_confirms():
    """The stricter matching of review #95 must not break the clean case:

    a genuine CC BY statement still confirms agreed cc-by.
    """
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"


# ---------------------------------------------------------------------------
# Step 3 — human sign-off (TDD plan items 7-8)
# ---------------------------------------------------------------------------


def test_signoff_step_writes_human_signoff():
    """TDD plan item 7: the interactive step (input injected, clock

    injected) prompts for who then note, refuses blank answers by
    re-prompting, and returns exactly the {who, date, note} record the
    #5 schema requires, with the date as the injected today in ISO form.
    """
    answers = iter(
        [
            "",  # blank "who" — must be re-prompted, never accepted
            "Chris the Verifier",
            "Checked the publisher page; statement matches CC BY.",
        ]
    )
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    record = collect_signoff(input_fn, today=datetime.date(2026, 8, 16))

    assert record == {
        "who": "Chris the Verifier",
        "date": "2026-08-16",
        "note": "Checked the publisher page; statement matches CC BY.",
    }
    assert len(prompts) == 3  # who (blank), who again, note


def test_unsigned_candidate_never_reaches_manifest():
    """TDD plan item 8: the gate's own enforcement (with #5's validator

    as the backstop) — a confirmed candidate without a sign-off never
    becomes a manifest entry; with one, the produced entry passes
    ingestion.manifest.validate_document unchanged, and a rejected
    report never produces an entry even when signed.
    """
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"

    with pytest.raises(GateError, match="human_signoff"):
        build_manifest_entry(report, None, **ENTRY_META)

    entry = build_manifest_entry(report, SIGNOFF, **ENTRY_META)
    record = validate_document(entry)  # the #5 gate accepts it unchanged
    assert record.human_signoff.who == "Test Signer"
    assert entry["permitted_context"] == "open"
    assert entry["redistributable"] is True
    assert "CC BY" in entry["licence"]
    assert entry["canonical_url"] == f"https://doi.org/{CLEAN_DOI}"

    rejected = gate_document(
        TRAP_DOI,
        **_lookups("free_to_read_trap"),
        page_html=_page("free_to_read_trap"),
        page_url=TRAP_PAGE_URL,
    )
    with pytest.raises(GateError):
        build_manifest_entry(rejected, SIGNOFF, **ENTRY_META)


# ---------------------------------------------------------------------------
# Manifest-entry semantics (review #100)
# ---------------------------------------------------------------------------


def test_manifest_sha256_pins_the_ingest_artefact():
    """Review finding #100 (decision): the entry's `sha256` pins the bytes

    of the ingest artefact at `source_url` — what `make corpus`/#7 will
    re-fetch and verify under ADR-023 — never the dynamic landing-page
    HTML. The page-evidence hash is scoped inside licence_evidence
    (page-sha256: ...), keeping the audit trail without asserting an
    unverifiable pin.
    """
    report = gate_document(
        CLEAN_DOI,
        **_lookups("clean_cc_by"),
        page_html=_page("clean_cc_by"),
        page_url=CLEAN_PAGE_URL,
    )
    artefact_url = "https://synthpress.example.invalid/articles/aurelian-syn-2024-0001.pdf"
    entry = build_manifest_entry(
        report,
        SIGNOFF,
        **ENTRY_META,
        source_url=artefact_url,
        page_sha256="b" * 64,
    )

    assert entry["source_url"] == artefact_url
    assert entry["sha256"] == ENTRY_META["sha256"]  # the artefact hash the caller verified
    assert "page-sha256: " + "b" * 64 in entry["licence_evidence"]
    assert CLEAN_STATEMENT in entry["licence_evidence"]
    validate_document(entry)


def test_licence_label_derives_version_from_crossref_url():
    """Review finding #100: a CC-BY-3.0 paper must never be recorded as

    "CC BY 4.0" — the label's version comes from the Crossref licence
    URL when one is stated.
    """
    lookups = _lookups("clean_cc_by")
    lookups["crossref"]["message"]["license"][0]["URL"] = (
        "https://creativecommons.org/licenses/by/3.0/"
    )
    report = gate_document(
        CLEAN_DOI,
        **lookups,
        page_html=_page_with_statement(
            "This is an open access article distributed under the terms of "
            "the Creative Commons Attribution licence."
        ),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"
    entry = build_manifest_entry(
        report,
        SIGNOFF,
        **ENTRY_META,
        source_url="https://synthpress.example.invalid/articles/aurelian-syn-2024-0001.pdf",
    )
    assert "3.0" in entry["licence"]
    assert "4.0" not in entry["licence"]


def test_licence_label_carries_no_unverified_version():
    """Review finding #100: when no source states a licence version

    (OpenAlex/Unpaywall tokens carry none and Crossref is silent), the
    label is version-less "CC BY" — a version is never fabricated into
    the legal audit trail.
    """
    openalex = {
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "best_oa_location": {"is_oa": True, "license": "cc-by", "version": "publishedVersion"},
    }
    unpaywall = {
        "doi": CLEAN_DOI,
        "is_oa": True,
        "oa_status": "gold",
        "best_oa_location": {
            "host_type": "publisher",
            "license": "cc-by",
            "version": "publishedVersion",
        },
    }
    report = gate_document(
        CLEAN_DOI,
        openalex=openalex,
        crossref=None,
        unpaywall=unpaywall,
        page_html=_page_with_statement(
            "This is an open access article distributed under the terms of "
            "the Creative Commons Attribution licence."
        ),
        page_url=CLEAN_PAGE_URL,
    )
    assert report.status == "candidate"
    entry = build_manifest_entry(
        report,
        SIGNOFF,
        **ENTRY_META,
        source_url="https://synthpress.example.invalid/articles/aurelian-syn-2024-0001.pdf",
    )
    assert entry["licence"] == "CC BY"


# ---------------------------------------------------------------------------
# Fixture discipline (IMPLEMENTATION.md §5)
# ---------------------------------------------------------------------------


def test_gate_fixtures_carry_synthetic_marker():
    """Every licensing-gate fixture (lookup JSON and page HTML) declares

    itself synthetic on its first line — no real lookup metadata or
    publisher page text may ever be committed here, mirroring the corpus
    fixture guard in test_fixture_corpus.py.
    """
    fixture_files = sorted(
        path
        for path in GATE_FIXTURES.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert fixture_files, "licensing-gate fixtures are missing"
    for path in fixture_files:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert "SYNTHETIC FIXTURE" in first_line, f"{path} lacks the SYNTHETIC FIXTURE marker"
