"""IPCC-style curated headline statements (issue #7 TDD plan 10; DESIGN
§2.1 Tier C / §2.4) — RED. Unit tier, synthetic statements only.

The curated set is behind a feature flag DEFAULTING OFF (a refusal from
the IPCC is then a config change, not a scramble); when enabled, one
statement = one chunk, and more than the ≤10-per-SPM cap fails the
ingest — the substantiality discipline is enforced by code, not habit.
"""

from __future__ import annotations

import pytest

from ingestion.pipeline import IngestConfig, IngestError, chunk_document
from tests._ingestion_fixtures import config, doc, heading, manifest_entry, text

STATEMENTS = [
    f"S.{i} Invented headline statement number {i} about the fictional basin's outlook."
    for i in range(1, 7)
]


def _headline_doc(statements=None):
    statements = STATEMENTS if statements is None else statements
    return doc(
        "syn-headlines",
        [heading("Statements")] + [text(s) for s in statements],
        title="MCA-3 headline statements (invented)",
    )


def _headline_entry():
    return manifest_entry("syn-headlines", ingest_profile="headline-statements")


def test_headline_feature_flag_defaults_off():
    """DESIGN §2.1: default config must not ingest the curated set — the
    flag is opt-in, enabled only after the legal check clears it."""
    assert IngestConfig().headline_statements_enabled is False
    chunks = chunk_document(_headline_doc(), _headline_entry(), config())
    assert chunks == [], (
        "headline-statements documents must produce no chunks while the feature flag is off"
    )


def test_headline_statement_is_single_chunk():
    """TDD plan 10: with the flag on, one statement = one chunk — natural
    units, never packed together or split."""
    cfg = config(headline_statements_enabled=True)
    chunks = chunk_document(_headline_doc(), _headline_entry(), cfg)
    assert len(chunks) == len(STATEMENTS)
    for statement, chunk in zip(STATEMENTS, chunks, strict=True):
        assert chunk.body == statement


def test_headline_statements_capped_at_10_per_document():
    """TDD plan 10: an 11-statement curated set violates the ≤10-per-SPM
    substantiality cap and must fail the ingest (IngestError), not be
    silently truncated."""
    eleven = [
        f"S.{i} Invented over-cap headline statement number {i} for the cap test."
        for i in range(1, 12)
    ]
    cfg = config(headline_statements_enabled=True)
    with pytest.raises(IngestError, match="(?i)headline|cap|10"):
        chunk_document(_headline_doc(eleven), _headline_entry(), cfg)


def test_exactly_ten_statements_pass_the_cap():
    """The boundary: ten statements are legal; the cap is ≤10, not <10."""
    ten = [
        f"S.{i} Invented boundary headline statement number {i} for the cap test."
        for i in range(1, 11)
    ]
    cfg = config(headline_statements_enabled=True)
    chunks = chunk_document(_headline_doc(ten), _headline_entry(), cfg)
    assert len(chunks) == 10


def test_unknown_ingest_profile_refuses_chunking():
    """Review finding #142: a one-character typo in ingest_profile
    ('headline_statements') previously ingested a Tier C curated set as
    ORDINARY EVIDENCE with the feature flag off — no cap, no
    one-statement-per-chunk, flag ignored. The chunker itself must fail
    closed on any non-enum value, naming the document and the valid
    values (mirror of the #79 consensus_position rule)."""
    typo_entry = manifest_entry("syn-headlines", ingest_profile="headline_statements")
    with pytest.raises(IngestError, match="(?i)ingest_profile.*headline-statements"):
        chunk_document(_headline_doc(), typo_entry, config())


def test_headline_statements_may_be_list_items():
    """Review finding #142: a curated set parsed as list items was
    silently ingested as ZERO chunks with the flag on, and list-item
    statements did not count toward the ≤10 cap. LIST_ITEM statements
    chunk one-per-statement and count against the cap."""
    from ingestion.parse import Block, BlockType

    li_statements = [
        f"S.{i} Invented list-item headline statement number {i}." for i in range(1, 5)
    ]
    li_doc = doc(
        "syn-headlines",
        [heading("Statements")] + [Block(BlockType.LIST_ITEM, s) for s in li_statements],
        title="MCA-3 headline statements (invented)",
    )
    cfg = config(headline_statements_enabled=True)
    chunks = chunk_document(li_doc, _headline_entry(), cfg)
    assert [c.body for c in chunks] == li_statements

    eleven_li = doc(
        "syn-headlines",
        [heading("Statements")]
        + [Block(BlockType.LIST_ITEM, f"S.{i} Invented statement {i}.") for i in range(1, 12)],
        title="MCA-3 headline statements (invented)",
    )
    with pytest.raises(IngestError, match="(?i)cap|10"):
        chunk_document(eleven_li, _headline_entry(), cfg)


def test_flag_on_zero_statement_headline_doc_refuses():
    """Review finding #142: a flag-on headline document that parses to no
    statements must refuse loudly, never silently ingest nothing."""
    empty_doc = doc(
        "syn-headlines",
        [heading("Statements")],
        title="MCA-3 headline statements (invented)",
    )
    cfg = config(headline_statements_enabled=True)
    with pytest.raises(IngestError, match="(?i)zero|no statements"):
        chunk_document(empty_doc, _headline_entry(), cfg)
