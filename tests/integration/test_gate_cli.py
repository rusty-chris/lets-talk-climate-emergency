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


def test_gate_cli_end_to_end_on_recorded_doi(tmp_path: Path) -> None:
    """TDD plan item 9: the CLI over the recorded clean-CC-BY fixture set,

    with the sign-off answered on stdin (confirm / who / note), exits 0
    and writes a manifest entry carrying the verbatim page evidence, the
    sha256 of the recorded page bytes, and the typed-in sign-off — and
    the #5 validator accepts the entry unchanged.
    """
    page_file = GATE_FIXTURES / "pages" / "clean_cc_by.html"
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
            "--doc-id",
            "syn-gate-clean-review",
            "--attribution",
            ATTRIBUTION,
            "--out",
            str(out_file),
        ],
        input="y\nTest Signer\nVerified the publisher page statement matches CC BY.\n",
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
    assert record.human_signoff.note == "Verified the publisher page statement matches CC BY."
    assert CLEAN_STATEMENT in record.licence_evidence
    assert CLEAN_PAGE_URL in record.licence_evidence
    assert record.canonical_url == f"https://doi.org/{CLEAN_DOI}"
    assert record.permitted_context == "open"
    assert record.redistributable is True
    assert "CC BY" in record.licence
    assert record.attribution_text == ATTRIBUTION
    assert record.sha256 == hashlib.sha256(page_file.read_bytes()).hexdigest()
    assert record.retrieved_at is not None
