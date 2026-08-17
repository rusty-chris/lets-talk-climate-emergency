"""Licensing-gate CLI end-to-end over recorded fixtures (issue #6, TDD

plan item 9).

`python -m ingestion.gate` in recorded mode (--lookups-dir / --page-html /
--page-url) is the whole DESIGN §2.2 pipeline as the operator runs it:
lookup verdicts -> >=2-of-3 candidate filter -> publisher-page evidence
shown -> interactive sign-off on stdin -> one corpus-manifest document
entry written that `ingestion.manifest.validate_document` accepts
unchanged. Recorded mode replays the committed synthetic fixtures under
tests/fixtures/licensing_gate/ — no live network in this tier
(IMPLEMENTATION.md §3; auto-marked `integration` by this directory's
conftest). The live run over the real candidate review papers is issue
task output, not a test.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

from ingestion.manifest import validate_document

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "licensing_gate"

CLEAN_DOI = "10.5555/aurelian-syn.2024.0001"
CLEAN_PAGE_URL = "https://synthpress.example.invalid/articles/aurelian-syn-2024-0001"

#: The exact statement in pages/clean_cc_by.html (kept in sync with
#: tests/unit/test_licensing_gate.py) — it must land verbatim in
#: licence_evidence.
CLEAN_STATEMENT = (
    "This is an open access article distributed under the terms of the "
    "Creative Commons Attribution 4.0 licence (CC BY 4.0), which permits "
    "unrestricted reuse, distribution and reproduction in any medium, "
    "provided the original work is properly cited."
)

ATTRIBUTION = (
    "Solari, A. & Okoye, B. (2024). Attribution of Aurelian Basin drying: "
    "a synthetic review. Synthetic Reviews of Climate (fictional). CC BY 4.0."
)


#: The artefact the entry's sha256 pins (review #100): the full-text
#: bytes `make corpus`/#7 will re-fetch from source_url and verify.
ARTEFACT_URL = "https://synthpress.example.invalid/articles/aurelian-syn-2024-0001.pdf"


def test_gate_cli_end_to_end_on_recorded_doi(tmp_path: Path) -> None:
    """TDD plan item 9 + review #100: the CLI over the recorded

    clean-CC-BY fixture set, with the sign-off answered on stdin,
    exits 0 and writes a manifest entry carrying the verbatim page
    evidence, the sha256 of the recorded *ingest artefact* (the bytes at
    source_url — what #7 re-fetch-verifies; the landing page's own hash
    is scoped into licence_evidence), and the typed-in sign-off — and
    the #5 validator accepts the entry unchanged.
    """
    page_file = GATE_FIXTURES / "pages" / "clean_cc_by.html"
    artefact_file = tmp_path / "artefact.pdf"
    artefact_file.write_bytes(b"%synthetic-artefact SYNTHETIC FIXTURE - invented full-text bytes\n")
    out_file = tmp_path / "gate_entry.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingestion.gate",
            CLEAN_DOI,
            "--lookups-dir",
            str(GATE_FIXTURES / "lookups" / "clean_cc_by"),
            "--page-html",
            str(page_file),
            "--page-url",
            CLEAN_PAGE_URL,
            "--artefact",
            str(artefact_file),
            "--artefact-url",
            ARTEFACT_URL,
            "--doc-id",
            "syn-gate-clean-review",
            "--attribution",
            ATTRIBUTION,
            "--out",
            str(out_file),
            # Review #103: piped stdin is refused unless this recorded-
            # fixture escape hatch is explicit; its use is printed loudly
            # and recorded in the sign-off note.
            "--signoff-tty-check=off",
        ],
        input=(
            "y\n"
            "Test Signer\n"
            "Verified the publisher page statement matches CC BY.\n"
            "admit syn-gate-clean-review\n"
        ),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    # The evidence the human confirmed was actually shown: all three
    # sources and the verbatim page statement appear in the CLI output.
    for source in ("openalex", "crossref", "unpaywall"):
        assert source in result.stdout.lower()
    assert CLEAN_STATEMENT in result.stdout

    assert out_file.exists(), "confirmed + signed candidate must write the manifest entry"
    entry = yaml.safe_load(out_file.read_text(encoding="utf-8"))
    assert isinstance(entry, dict)

    record = validate_document(entry)  # the merged #5 schema is the contract
    assert record.id == "syn-gate-clean-review"
    assert record.human_signoff.who == "Test Signer"
    # Review #103: the TTY-check override is recorded in the note.
    assert record.human_signoff.note.startswith(
        "Verified the publisher page statement matches CC BY."
    )
    assert "signoff-tty-check=off" in record.human_signoff.note
    # Review #103: the full sign-off record was echoed back before the
    # final typed confirmation, so the operator confirms what will be
    # written, not just that something will be.
    assert "Test Signer" in result.stdout
    assert "admit syn-gate-clean-review" in result.stdout
    assert CLEAN_STATEMENT in record.licence_evidence
    assert CLEAN_PAGE_URL in record.licence_evidence
    assert record.canonical_url == f"https://doi.org/{CLEAN_DOI}"
    assert record.permitted_context == "open"
    assert record.redistributable is True
    assert "CC BY" in record.licence
    assert record.attribution_text == ATTRIBUTION
    # Review #100: sha256 pins the ingest artefact at source_url — the
    # bytes make corpus/#7 re-fetch-verify — never the landing-page HTML.
    assert record.sha256 == hashlib.sha256(artefact_file.read_bytes()).hexdigest()
    assert record.source_url == ARTEFACT_URL
    # The landing page's own hash is evidence-scoped, not the ADR-023 pin.
    page_digest = hashlib.sha256(page_file.read_bytes()).hexdigest()
    assert f"page-sha256: {page_digest}" in record.licence_evidence
    assert record.retrieved_at is not None


def test_cli_requires_tty_or_explicit_batch_flag(tmp_path: Path) -> None:
    """Review finding #103: `yes y | python -m ingestion.gate ...` must not

    sail through — without a TTY on stdin and without the explicit
    override flag, the CLI exits non-zero naming the sign-off
    requirement and writes nothing.
    """
    artefact_file = tmp_path / "artefact.pdf"
    artefact_file.write_bytes(b"%synthetic-artefact SYNTHETIC FIXTURE - invented full-text bytes\n")
    out_file = tmp_path / "entry.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingestion.gate",
            CLEAN_DOI,
            "--lookups-dir",
            str(GATE_FIXTURES / "lookups" / "clean_cc_by"),
            "--page-html",
            str(GATE_FIXTURES / "pages" / "clean_cc_by.html"),
            "--page-url",
            CLEAN_PAGE_URL,
            "--artefact",
            str(artefact_file),
            "--artefact-url",
            ARTEFACT_URL,
            "--doc-id",
            "syn-gate-clean-review",
            "--attribution",
            ATTRIBUTION,
            "--out",
            str(out_file),
        ],
        input="y\ny\ny\nadmit syn-gate-clean-review\n",  # the yes-flood shape
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert result.returncode != 0
    assert "sign-off" in result.stderr.lower()
    assert not out_file.exists()


def test_cli_output_carries_no_markup_or_control_chars(tmp_path: Path) -> None:
    """Review finding #102: a hostile page's entity-encoded markup and

    ANSI escape sequences must never reach the operator's terminal — the
    printed statement is the sanitised statement, so the evidence the
    human sees cannot be repainted by the page being checked.
    """
    artefact_file = tmp_path / "artefact.pdf"
    artefact_file.write_bytes(b"%synthetic-artefact SYNTHETIC FIXTURE - invented full-text bytes\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingestion.gate",
            CLEAN_DOI,
            "--lookups-dir",
            str(GATE_FIXTURES / "lookups" / "clean_cc_by"),
            "--page-html",
            str(GATE_FIXTURES / "pages" / "hostile_markup.html"),
            "--page-url",
            CLEAN_PAGE_URL,
            "--artefact",
            str(artefact_file),
            "--artefact-url",
            ARTEFACT_URL,
            "--doc-id",
            "syn-gate-hostile-review",
            "--attribution",
            ATTRIBUTION,
            "--out",
            str(tmp_path / "entry.yaml"),
            "--signoff-tty-check=off",
        ],
        input="n\n",  # decline at confirmation — we only inspect the printed evidence
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert "Creative Commons Attribution 4.0" in result.stdout
    assert "\x1b" not in result.stdout
    assert "<script>" not in result.stdout
