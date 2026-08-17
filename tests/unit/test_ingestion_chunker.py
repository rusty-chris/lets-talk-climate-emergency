"""Production chunker behaviour (issue #7 TDD plan 1–7; DESIGN §2.4) — RED.

Unit tier: pure over synthetic StructuredDoc inputs (the parse seam,
IMPLEMENTATION.md §1); the token counter is injected so no model weights
are needed. Cap semantics follow the #41 amendment recorded in
reviews/spike-02-parsing-findings.md: the ≤max_tokens cap covers the
chunk's embedded text INCLUDING the context header; overlap seeding
re-checks the cap; oversized atomic units split at sentence boundaries;
overlap is intentionally skipped after an atomic-unit chunk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ingestion.pipeline import chunk_document
from tests._ingestion_fixtures import (
    ATOMIC_BLOCK_TYPES,
    config,
    doc,
    footnote,
    heading,
    manifest_entry,
    sentence,
    text,
    word_tokens,
)

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion" / "golden_chunks.yaml"


def _two_section_doc():
    return doc(
        "syn-two-sections",
        [
            heading("1 Observed changes"),
            text(sentence(12, "alphaobs")),
            text(sentence(12, "betaobs")),
            heading("2 Projected impacts"),
            text(sentence(12, "gammaproj")),
            text(sentence(12, "deltaproj")),
        ],
        title="Synthetic Basin Report",
    )


def test_chunks_never_cross_heading_boundaries():
    """TDD plan 1: no chunk mixes text from two sections — the boundary
    invariant is structural (DESIGN §2.4 'never across headings')."""
    chunks = chunk_document(_two_section_doc(), manifest_entry(), config(max_tokens=500))
    assert chunks, "the synthetic doc must produce chunks"
    for chunk in chunks:
        crosses = ("alphaobs" in chunk.body or "betaobs" in chunk.body) and (
            "gammaproj" in chunk.body or "deltaproj" in chunk.body
        )
        assert not crosses, f"{chunk.chunk_id} spans two sections: {chunk.section_path}"


def test_chunk_token_length_at_most_limit():
    """TDD plan 2 (amended by #41): every chunk's token_count — and an
    independent recount of its full embedded text, header included — is
    within the configured cap."""
    blocks = [heading("1 Long section")] + [text(sentence(12, f"s{i}")) for i in range(40)]
    cfg = config(max_tokens=100, min_tokens=1)
    chunks = chunk_document(doc("syn-long", blocks), manifest_entry(), cfg)
    assert len(chunks) > 1, "40 twelve-word sentences under a 100-token cap must split"
    for chunk in chunks:
        assert chunk.token_count <= 100, f"{chunk.chunk_id}: token_count {chunk.token_count} > cap"
        recount = word_tokens(chunk.embedding_text)
        assert recount <= 100, (
            f"{chunk.chunk_id}: embedded text (header included) recounts to {recount} > cap — "
            "the cap must cover header + body (#41)"
        )


def test_context_header_prepended():
    """TDD plan 3: each chunk carries a document → section-path context
    header; the embedded text opens with it; the citable body does NOT
    (spike-03: the header must never leak into cited_text)."""
    chunks = chunk_document(_two_section_doc(), manifest_entry(), config())
    for chunk in chunks:
        assert chunk.context_header.startswith("Synthetic Basin Report"), chunk.context_header
        for element in chunk.section_path:
            assert element in chunk.context_header
        assert chunk.embedding_text.startswith(chunk.context_header)
        assert not chunk.body.startswith(chunk.context_header), (
            f"{chunk.chunk_id}: the citable body must not embed the context header"
        )


def test_one_sentence_overlap_between_adjacent_chunks():
    """TDD plan 4: within a section, each chunk opens with the previous
    chunk's final sentence (the spike-verified behaviour, now a rule)."""
    sentences = [sentence(6, f"ov{i}") for i in range(30)]
    blocks = [heading("1 Overlap section")] + [text(s) for s in sentences]
    chunks = chunk_document(doc("syn-overlap", blocks), manifest_entry(), config(max_tokens=40))
    assert len(chunks) >= 3, "30 six-word sentences under a 40-token cap must yield several chunks"

    def indices(body: str) -> list[int]:
        return [i for i, s in enumerate(sentences) if s in body]

    for previous, current in zip(chunks, chunks[1:], strict=False):
        if previous.section_path != current.section_path:
            continue
        previous_idx, current_idx = indices(previous.body), indices(current.body)
        assert previous_idx and current_idx
        assert current_idx[0] == previous_idx[-1], (
            f"{current.chunk_id} must open with {previous.chunk_id}'s final sentence "
            f"(got sentence {current_idx[0]}, expected {previous_idx[-1]})"
        )


def test_cap_holds_when_overlap_seed_plus_next_unit_exceeds_limit():
    """#41 path (a): two near-cap sentences — the spike's overlap seeding
    emitted a 120-token chunk under a 100 cap here. The production cap is
    re-checked after seeding: no chunk may exceed it, and no sentence may
    be lost."""
    first, second = sentence(60, "capa"), sentence(60, "capb")
    blocks = [heading("1 Cap section"), text(first), text(second)]
    cfg = config(max_tokens=100, min_tokens=1)
    chunks = chunk_document(doc("syn-cap-overlap", blocks), manifest_entry(), cfg)
    for chunk in chunks:
        assert word_tokens(chunk.embedding_text) <= 100, (
            f"{chunk.chunk_id}: overlap seeding overshot the cap "
            f"({word_tokens(chunk.embedding_text)} tokens)"
        )
    joined = " ".join(chunk.body for chunk in chunks)
    assert first in joined and second in joined, "no sentence may be dropped to satisfy the cap"


def test_oversized_atomic_unit_split_at_sentence_boundaries():
    """#41 path (b), amended policy: an atomic unit longer than the cap
    (a 150-token footnote under a 70 cap) is split at sentence boundaries
    — never passed through as an oversized chunk, never dropped."""
    parts = [sentence(50, f"fn{i}") for i in range(3)]
    blocks = [heading("1 Notes"), footnote(" ".join(parts))]
    cfg = config(max_tokens=70, min_tokens=1)
    chunks = chunk_document(doc("syn-oversized-atomic", blocks), manifest_entry(), cfg)
    for chunk in chunks:
        assert word_tokens(chunk.embedding_text) <= 70, (
            f"{chunk.chunk_id}: oversized atomic unit leaked through the cap"
        )
    joined = " ".join(chunk.body for chunk in chunks)
    for part in parts:
        assert part in joined, "sentence-boundary splitting must preserve every sentence"


def test_cap_accounts_for_context_header():
    """#41 path (c): a long title + deep section path consume cap budget.
    token_count equals the injected counter over the full embedded text
    (header + body), and stays within the cap."""
    long_title = " ".join(f"title{i}" for i in range(20))
    blocks = [heading("1 " + " ".join(f"sec{i}" for i in range(10)))] + [
        text(sentence(10, f"hb{i}")) for i in range(12)
    ]
    cfg = config(max_tokens=60, min_tokens=1)
    budget_doc = doc("syn-header-budget", blocks, title=long_title)
    chunks = chunk_document(budget_doc, manifest_entry(), cfg)
    for chunk in chunks:
        assert chunk.token_count == word_tokens(chunk.embedding_text), (
            f"{chunk.chunk_id}: token_count must be the configured counter over header + body"
        )
        assert chunk.token_count <= 60, (
            f"{chunk.chunk_id}: header excluded from cap accounting "
            f"({chunk.token_count} tokens embedded)"
        )


def test_overlap_skipped_after_atomic_unit_is_intentional():
    """The #41-recorded boundary rule, now explicit: no overlap is carried
    out of a chunk that ends with an atomic unit — the atomic text appears
    exactly once across the section's chunks."""
    note = sentence(90, "atomicnote")
    follower = sentence(50, "after")
    blocks = [heading("1 Atomic boundary"), footnote(note), text(follower)]
    cfg = config(max_tokens=100, min_tokens=1)
    chunks = chunk_document(doc("syn-atomic-overlap", blocks), manifest_entry(), cfg)
    occurrences = sum(chunk.body.count(note) for chunk in chunks)
    assert occurrences == 1, (
        f"atomic unit text must appear exactly once (no overlap after atomic); got {occurrences}"
    )
    assert any(follower in chunk.body for chunk in chunks)


def test_no_chunk_below_minimum_token_floor_except_atomic():
    """Spike finding 8: tiny/degenerate chunks (a lone heading-follower
    fragment) are merged or suppressed — no emitted non-atomic chunk sits
    below the configured floor."""
    blocks = [
        heading("1 Fragment section"),
        text("45°N label."),
        heading("2 Real section"),
        text(sentence(30, "real")),
    ]
    cfg = config(max_tokens=100, min_tokens=10)
    chunks = chunk_document(doc("syn-tiny", blocks), manifest_entry(), cfg)
    for chunk in chunks:
        if set(chunk.block_types) <= ATOMIC_BLOCK_TYPES:
            continue  # deliberately atomic units are exempt
        assert chunk.token_count >= 10, (
            f"{chunk.chunk_id}: sub-floor chunk emitted "
            f"({chunk.token_count} tokens): {chunk.body!r}"
        )


def test_chunk_ids_hash_stable_across_reruns():
    """TDD plan 7: same input → identical ids and records (idempotent
    re-embedding hook, DESIGN §2.4)."""
    entry = manifest_entry("syn-stable")
    cfg = config(max_tokens=100)
    first = chunk_document(_two_section_doc(), entry, cfg)
    second = chunk_document(_two_section_doc(), entry, cfg)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first == second


def test_chunk_ids_are_content_hashes_not_positions():
    """Incremental re-embedding requires content-derived ids: editing one
    section changes only that section's chunk ids; untouched sections keep
    theirs (so their embeddings are reusable). A positional id scheme
    fails both arms."""
    entry = manifest_entry("syn-incremental")
    cfg = config(max_tokens=500)

    def build(first_sentence: str):
        return doc(
            "syn-incremental",
            [
                heading("1 Edited section"),
                text(first_sentence),
                heading("2 Untouched section"),
                text(sentence(12, "stable")),
            ],
        )

    original = chunk_document(build(sentence(12, "editedv1")), entry, cfg)
    edited = chunk_document(build(sentence(12, "editedv2")), entry, cfg)

    original_by_section = {c.section_path: c for c in original}
    edited_by_section = {c.section_path: c for c in edited}
    untouched = ("1 Edited section",), ("2 Untouched section",)
    edited_section, stable_section = untouched
    assert (
        edited_by_section[edited_section].chunk_id != original_by_section[edited_section].chunk_id
    ), "a changed chunk body must change the chunk id (content hash)"
    assert (
        edited_by_section[stable_section].chunk_id == original_by_section[stable_section].chunk_id
    ), "an untouched section's chunk ids must survive edits elsewhere in the document"


def _golden_docs():
    """Two deterministic synthetic docs the goldens pin (TDD plan 6)."""
    return [
        _two_section_doc(),
        doc(
            "syn-golden-notes",
            [
                heading("1 Findings"),
                text(sentence(15, "gold")),
                footnote(sentence(8, "goldnote")),
            ],
            title="Synthetic Golden Notes",
        ),
    ]


def test_golden_chunks_stable_for_fixture_docs():
    """TDD plan 6: full golden-output comparison — the chunking-experiment
    scope guard, mechanised. The implementer authors the golden file in
    the green commit; any later chunker change that alters it must ship
    the updated golden in the same commit with the diff explained."""
    if not GOLDEN_PATH.is_file():
        pytest.fail(
            f"golden chunk fixture missing at {GOLDEN_PATH} — the green commit must generate "
            "it from these synthetic docs (IMPLEMENTATION.md §5, golden chunk outputs; first "
            "line must carry the SYNTHETIC FIXTURE marker comment)"
        )
    golden = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    produced = []
    for fixture_doc in _golden_docs():
        for chunk in chunk_document(fixture_doc, manifest_entry(fixture_doc.doc_id), config()):
            produced.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_path": list(chunk.section_path),
                    "context_header": chunk.context_header,
                    "body": chunk.body,
                    "token_count": chunk.token_count,
                }
            )
    assert produced == golden["chunks"], (
        "chunk output diverged from the committed goldens — if the change is intended, update "
        "the golden file in the same commit and explain the diff (the scope guard)"
    )
