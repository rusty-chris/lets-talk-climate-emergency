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
