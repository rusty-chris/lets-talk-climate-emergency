"""Per-chunk citation metadata (issue #7 TDD plan 8–9; DESIGN §2.4 schema,
§2.3 severity-skew guardrail, §2.5 voices labelling) — RED. Unit tier.
"""

from __future__ import annotations

from ingestion.pipeline import chunk_document, extract_confidence_markers
from tests._ingestion_fixtures import config, doc, heading, manifest_entry, sentence, text

# --------------------------------------------------------------------------
# confidence_markers regex extraction
# --------------------------------------------------------------------------


def test_confidence_markers_extracted():
    """TDD plan 8: calibrated phrases are captured as they appear."""
    assert "very likely" in extract_confidence_markers(
        "Continued warming of at least a further 0.4 °C is considered very likely "
        "under every invented pathway."
    )
    assert "high confidence" in extract_confidence_markers(
        "Annual rainfall declined by 14 % over the invented period (high confidence)."
    )
    assert "virtually certain" in extract_confidence_markers(
        "It is virtually certain that the invented lowlands warmed faster than the shelf."
    )


def test_negated_qualifier_not_miscaptured():
    """TDD plan 8, the negation trap: 'not likely' must never be recorded
    as 'likely'."""
    markers = extract_confidence_markers(
        "It is not likely that this invented decline can be explained by internal "
        "variability alone."
    )
    assert "likely" not in markers, f"negated qualifier miscaptured: {markers}"
    assert "very likely" not in markers


def test_unlikely_not_miscaptured_as_likely():
    """The substring trap next door: 'unlikely' is its own marker, never
    the bare positive."""
    markers = extract_confidence_markers(
        "Recovery within a human lifetime is judged unlikely if invented extraction continues."
    )
    assert "unlikely" in markers
    assert "likely" not in markers, f"'unlikely' miscaptured as 'likely': {markers}"


def test_negation_within_short_window_not_captured():
    """Review finding #146: the negation guard only looked at the
    immediately preceding word, so realistic phrasing ('not considered
    likely') recorded the INVERTED marker. Negation up to a few words
    before the marker must suppress the positive-likelihood capture."""
    for phrase in (
        "It is not considered likely that the invented trend reverses.",
        "A reversal is not at all likely under the invented pathways.",
        "It does not seem likely that the invented shelf recovers.",
    ):
        markers = extract_confidence_markers(phrase)
        assert "likely" not in markers, f"{phrase!r}: inverted marker recorded: {markers}"


def test_cannot_negates_as_whole_word_not_substring():
    """Review finding #146, pinned policy: 'cannot' IS a negator — by
    design, matched as a whole word ('cannot likely resolve' asserts no
    likelihood). The substring accident is what must die: a word merely
    ending in 'not' (e.g. 'Huguenot') is NOT negation."""
    suppressed = extract_confidence_markers("The committee cannot likely resolve this question.")
    assert "likely" not in suppressed
    captured = extract_confidence_markers("The Huguenot likely settled along the invented coast.")
    assert "likely" in captured, (
        "whole-word matching: a word ending in 'not' must not suppress capture"
    )


def test_quoted_markers_not_captured_as_source_calibration():
    """Review finding #146, pinned policy: a qualifier inside quotation
    marks (a quoted sceptic's claim) is NOT the source's own calibration
    and is excluded; the same phrase unquoted still captures."""
    for quoted in (
        "Sceptics claim it is 'very unlikely' that humans cause the invented warming.",
        'Sceptics claim it is "very unlikely" that humans cause the invented warming.',
        "Sceptics claim it is “very unlikely” that humans cause the invented warming.",
    ):
        markers = extract_confidence_markers(quoted)
        assert "very unlikely" not in markers and "unlikely" not in markers, (
            f"{quoted!r}: quoted qualifier recorded as source calibration: {markers}"
        )
    unquoted = extract_confidence_markers(
        "It is very unlikely that the invented shelf regrows this century."
    )
    assert "very unlikely" in unquoted


def test_hyphen_compound_prefix_not_captured():
    """Review finding #146: 'ultra-high confidence' is not an IPCC
    calibrated phrase — a hyphen-glued prefix must not yield a
    mid-compound 'high confidence' capture."""
    markers = extract_confidence_markers("The ultra-high confidence interval is wide.")
    assert "high confidence" not in markers, f"mid-compound capture: {markers}"
    plain = extract_confidence_markers("The trend is robust (high confidence).")
    assert "high confidence" in plain


def test_chunk_carries_markers_from_its_own_body():
    """Markers land on the chunk whose body contains them."""
    calibrated = doc(
        "syn-calibrated",
        [
            heading("1 Findings"),
            text(
                "It is very likely that the invented terraces dried over the record, "
                "according to every invented station series this fixture maintains "
                "(high confidence)."
            ),
            heading("2 Unqualified"),
            text(sentence(25, "plain")),
        ],
    )
    chunks = chunk_document(calibrated, manifest_entry(), config())
    marked = [c for c in chunks if "very likely" in c.body]
    assert marked
    for chunk in marked:
        assert "very likely" in chunk.confidence_markers
        assert "high confidence" in chunk.confidence_markers
    unmarked = [c for c in chunks if "plain" in c.body]
    assert unmarked
    for chunk in unmarked:
        assert chunk.confidence_markers == ()


# --------------------------------------------------------------------------
# consensus_position propagation (the Hansen severity-skew guardrail)
# --------------------------------------------------------------------------


def _simple_doc(doc_id: str):
    return doc(
        doc_id,
        [
            heading("1 Argument"),
            text(sentence(25, "arga")),
            heading("2 Estimates"),
            text(sentence(25, "argb")),
        ],
    )


def test_consensus_position_propagates_to_every_chunk():
    """TDD plan 9: a beyond-assessed-range document flags EVERY chunk —
    the §2.3 guardrail keys on the chunk-level flag at answer time."""
    entry = manifest_entry("syn-beyond", consensus_position="beyond-assessed-range")
    chunks = chunk_document(_simple_doc("syn-beyond"), entry, config())
    assert chunks
    for chunk in chunks:
        assert chunk.consensus_position == "beyond-assessed-range", chunk.chunk_id


def test_consensus_position_defaults_to_assessed():
    chunks = chunk_document(_simple_doc("syn-assessed"), manifest_entry("syn-assessed"), config())
    assert chunks
    for chunk in chunks:
        assert chunk.consensus_position == "assessed"


# --------------------------------------------------------------------------
# voices labelling (DESIGN §2.5)
# --------------------------------------------------------------------------


def test_voices_source_type_labels_every_chunk():
    """A `source_type: voices` document labels every chunk so the
    structural voices filter (#11) and the 'About the movement' UI badge
    have a chunk-level key — voices text is never mixed into
    scientific-claim answers."""
    entry = manifest_entry("syn-voices", source_type="voices")
    chunks = chunk_document(_simple_doc("syn-voices"), entry, config())
    assert chunks
    for chunk in chunks:
        assert chunk.source_type == "voices", chunk.chunk_id


def test_evidence_source_type_is_the_default_label():
    chunks = chunk_document(_simple_doc("syn-evidence"), manifest_entry("syn-evidence"), config())
    assert chunks
    for chunk in chunks:
        assert chunk.source_type == "evidence"


# --------------------------------------------------------------------------
# degraded-parse hand-review flag carriage (review finding #143)
# --------------------------------------------------------------------------


def test_degraded_chunks_carry_the_hand_review_flag():
    """Review finding #143: DESIGN §2.4 (amended) requires a
    PyMuPDF-parsed document to be flagged for hand review BEFORE
    indexing, but the flag lived only on the in-memory run record —
    chunks were indistinguishable. Every chunk must carry
    ``parse_backend`` and ``needs_hand_review`` so an indexer can refuse
    or quarantine degraded chunks without a join against run state."""
    from ingestion.parse import StructuredDoc

    def build(backend: str):
        base = _simple_doc("syn-degraded")
        return StructuredDoc(
            doc_id=base.doc_id, title=base.title, blocks=base.blocks, backend=backend
        )

    degraded_chunks = chunk_document(build("pymupdf"), manifest_entry("syn-degraded"), config())
    assert degraded_chunks
    for chunk in degraded_chunks:
        assert chunk.parse_backend == "pymupdf", chunk.chunk_id
        assert chunk.needs_hand_review is True, (
            f"{chunk.chunk_id}: degraded-parse chunk not flagged for hand review"
        )

    clean_chunks = chunk_document(build("docling"), manifest_entry("syn-degraded"), config())
    assert clean_chunks
    for chunk in clean_chunks:
        assert chunk.parse_backend == "docling"
        assert chunk.needs_hand_review is False


# --------------------------------------------------------------------------
# §2.4 citation-metadata schema carriage
# --------------------------------------------------------------------------


def test_every_chunk_carries_citation_schema_fields():
    """Each chunk's citation_metadata carries the §2.4 fields from its
    manifest entry — the service layer must never re-join against the
    manifest at answer time."""
    entry = manifest_entry("syn-schema")
    chunks = chunk_document(_simple_doc("syn-schema"), entry, config())
    assert chunks
    for chunk in chunks:
        metadata = chunk.citation_metadata
        assert metadata["licence"] == entry["licence"]
        assert metadata["attribution_text"] == entry["attribution_text"]
        assert metadata["canonical_url"] == entry["canonical_url"]
        assert metadata["permitted_context"] == entry["permitted_context"]
        assert metadata["title"] == entry["title"]
        assert chunk.doc_id == "syn-schema"
