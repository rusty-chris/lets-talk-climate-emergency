"""Characterisation tests for the issue #2 chunker prototype (ingestion.chunk).

Spike posture (IMPLEMENTATION.md §2 item 6, issue #2 TDD plan): these are NOT
test-first design. They pin the *observed* behaviour of the prototype chunker on
a small synthetic fixture so that (a) the two invariants the hand-review checks —
"never cross a heading" and "section paths match the outline" — are executable,
and (b) issue #7's real red-phase suite and #24's fixture design have a concrete
seed. They run on the committed synthetic fixture only; the real NCA5 / ESD PDFs
are gitignored and their recorded chunk dumps live in data/spike/ (see
reviews/spike-02-parsing-findings.md), never in CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ingestion.chunk import ChunkConfig, block_text, chunk, group_sections, split_sentences

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "spike" / "synthetic_doc.py"


def _load_fixture():
    spec = importlib.util.spec_from_file_location("synthetic_doc", _FIXTURE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_FX = _load_fixture()


# --------------------------------------------------------------------------- #
# Fixture hygiene (IMPLEMENTATION.md §5)
# --------------------------------------------------------------------------- #
def test_synthetic_fixture_carries_marker() -> None:
    """The fixture's first line is the mandated SYNTHETIC FIXTURE marker."""
    first_line = _FIXTURE_PATH.read_text().splitlines()[0]
    assert "SYNTHETIC FIXTURE" in first_line


# --------------------------------------------------------------------------- #
# The two invariants the hand-review pins (issue #2 acceptance criteria)
# --------------------------------------------------------------------------- #
def test_spike_chunks_never_cross_headings() -> None:
    """No chunk mixes body text from two different sections.

    Reconstruct each section's rendered text, then assert every chunk's sentences
    belong to exactly the one section named by chunk.section_path.
    """
    doc = _FX.build_synthetic_doc()
    sections = group_sections(doc)
    section_text = {s.path: " ".join(block_text(b) for b in s.blocks) for s in sections}
    chunks = chunk(doc, ChunkConfig())
    assert chunks

    for c in chunks:
        owner = section_text[c.section_path]
        body = c.text.split("\n\n", 1)[1]  # strip the prepended context header
        for sentence in split_sentences(body) or [body]:
            core = sentence.strip()
            if not core:
                continue
            assert core in owner, (
                f"chunk {c.chunk_id} (path {c.section_path}) contains text not from "
                f"its own section: {core!r}"
            )


def test_spike_section_paths_match_document_outline() -> None:
    """Distinct chunk section paths, in order, reproduce the nested outline."""
    doc = _FX.build_synthetic_doc()
    chunks = chunk(doc, ChunkConfig())
    seen: list[tuple[str, ...]] = []
    for c in chunks:
        if not seen or seen[-1] != c.section_path:
            seen.append(c.section_path)
    assert seen == _FX.EXPECTED_OUTLINE


# --------------------------------------------------------------------------- #
# DESIGN §2.4 behaviours the spike must demonstrate
# --------------------------------------------------------------------------- #
def test_context_header_is_document_then_section_path() -> None:
    """Every chunk is prefixed with 'document title -> section path'."""
    doc = _FX.build_synthetic_doc()
    for c in chunk(doc, ChunkConfig()):
        assert c.text.startswith(c.context_header)
        assert c.context_header.startswith(doc.title)
        for level in c.section_path:
            assert level in c.context_header


def test_figure_and_table_retained_as_citable_text_with_captions() -> None:
    """Figure/table placeholders survive chunking carrying their captions."""
    doc = _FX.build_synthetic_doc()
    joined = "\n".join(c.text for c in chunk(doc, ChunkConfig()))
    assert "[TABLE] Table 1. Invented decadal temperature anomalies" in joined
    assert "[FIGURE] Figure 1. Fictional time series" in joined


def test_long_section_splits_with_one_sentence_overlap() -> None:
    """A >max_tokens section yields multiple chunks that overlap by one sentence."""
    doc = _FX.build_synthetic_doc()
    config = ChunkConfig(max_tokens=120, overlap_sentences=1)
    chunks = [c for c in chunk(doc, config) if c.section_path == ("Projections",)]
    assert len(chunks) >= 2, "long Projections section should split"

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        tail = split_sentences(earlier.text.split("\n\n", 1)[1])[-1]
        head = split_sentences(later.text.split("\n\n", 1)[1])[0]
        assert tail == head, "consecutive chunks should share one overlap sentence"


def test_no_chunk_greatly_exceeds_target_tokens() -> None:
    """Chunks respect the ~500-token target (single oversized units excepted)."""
    doc = _FX.build_synthetic_doc()
    config = ChunkConfig(max_tokens=120)
    for c in chunk(doc, config):
        # Allow one sentence of overshoot beyond the target; nothing runaway.
        assert c.token_estimate <= config.max_tokens + 40


def test_page_furniture_is_dropped() -> None:
    """Running headers/footers never leak into chunk text."""
    doc = _FX.build_synthetic_doc()
    joined = "\n".join(c.text for c in chunk(doc, ChunkConfig()))
    assert "page 1" not in joined
