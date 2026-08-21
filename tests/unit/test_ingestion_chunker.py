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

from ingestion.parse import Block, BlockType
from ingestion.pipeline import chunk_document
from tests._ingestion_fixtures import (
    ATOMIC_BLOCK_TYPES,
    config,
    doc,
    figure,
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


def punct_tokens(text_value: str) -> int:
    """Deterministic punctuation-aware counter for the #139 tests — the
    same word-plus-punctuation rule as the production stand-in counter,
    still pure and weight-free (unit tier). A markdown separator row like
    ``|----|----|`` counts hundreds of tokens under it, exactly the shape
    that sailed through the whitespace-word accounting on the real ESD
    tables."""
    import re

    return len(re.findall(r"\w+|[^\w\s]", text_value))


def _wide_table_markdown(n_rows: int = 12, n_cols: int = 6) -> str:
    header = "| " + " | ".join(f"Column{i}" for i in range(n_cols)) + " |"
    separator = "|" + "|".join("-" * 30 for _ in range(n_cols)) + "|"
    rows = [
        "| " + " | ".join(f"r{r}c{c} value" for c in range(n_cols)) + " |" for r in range(n_rows)
    ]
    return "\n".join([header, separator, *rows])


def test_cap_holds_for_oversized_table_markdown():
    """Review finding #139: on the real spike documents, Docling table
    markdown passed through the #41 cap at up to 1742 tokens — the
    separator row is one whitespace 'word' that a punctuation-aware
    counter prices at hundreds of tokens. Every emitted chunk must
    respect the cap under the production counting rule."""
    table = Block(BlockType.TABLE, _wide_table_markdown(), caption=None)
    blocks = [heading("1 Data"), table]
    cfg = config(max_tokens=100, min_tokens=1, token_counter=punct_tokens)
    chunks = chunk_document(doc("syn-wide-table", blocks), manifest_entry(), cfg)
    assert chunks, "the table must still be chunked citable"
    for chunk in chunks:
        assert chunk.token_count <= 100, (
            f"{chunk.chunk_id}: table chunk of {chunk.token_count} tokens leaked through "
            f"the cap (#41/#139): {chunk.body[:80]!r}"
        )


def test_table_split_preserves_rows():
    """Review finding #139: a split table must break at ROW boundaries,
    never mid-cell (real trial chunks began mid-row, severing values from
    their row labels — the qualifier-from-claim separation §2.4 chunking
    exists to prevent). Each piece re-carries the header row so cells
    keep their labels; separator rows (no citable content) are dropped."""
    table = Block(BlockType.TABLE, _wide_table_markdown(), caption=None)
    blocks = [heading("1 Data"), table]
    cfg = config(max_tokens=100, min_tokens=1, token_counter=punct_tokens)
    chunks = chunk_document(doc("syn-row-split", blocks), manifest_entry(), cfg)
    table_chunks = [c for c in chunks if "table" in c.block_types]
    assert len(table_chunks) > 1, "the oversized table must split"
    header_row = "| " + " | ".join(f"Column{i}" for i in range(6)) + " |"
    for chunk in table_chunks:
        lines = chunk.body.splitlines()
        assert lines and lines[0] == header_row, (
            f"{chunk.chunk_id}: split table piece must open with the header row, got {lines[0]!r}"
        )
        for line in lines:
            assert line.startswith("|") and line.endswith("|"), (
                f"{chunk.chunk_id}: piece contains a severed/mid-row line: {line!r}"
            )
            stripped = set(line.replace(" ", ""))
            assert not (stripped <= set("|-:") and "-" in stripped), (
                f"{chunk.chunk_id}: separator row carried into a citable body: {line!r}"
            )
    joined = " ".join(c.body for c in table_chunks)
    for r in range(12):
        assert f"r{r}c0 value" in joined, f"row {r} lost in the split"


def test_single_row_over_cap_is_flagged_oversized_atomic():
    """Review finding #139 policy pin: a single table row that exceeds
    the cap even alone is NEVER split mid-cell — it becomes its own chunk
    carrying the header row, flagged ``oversized_atomic=True`` so the
    indexer can see (and quarantine/log) the deliberate cap exception.
    Silent pass-through and silent truncation are both forbidden."""
    header_row = "| Region | Reading |"
    separator = "|--------|---------|"
    giant_row = "| Region6 | " + " ".join(f"v{i}.{i}" for i in range(80)) + " |"
    small_row = "| Region1 | 5.5 |"
    table = Block(
        BlockType.TABLE, "\n".join([header_row, separator, small_row, giant_row]), caption=None
    )
    cfg = config(max_tokens=60, min_tokens=1, token_counter=punct_tokens)
    chunks = chunk_document(doc("syn-giant-row", [heading("1 Data"), table]), manifest_entry(), cfg)
    flagged = [c for c in chunks if c.oversized_atomic]
    assert len(flagged) == 1, (
        f"exactly the giant-row chunk must be flagged oversized_atomic, got {len(flagged)}"
    )
    assert "Region6" in flagged[0].body and "v79" in flagged[0].body, (
        "the giant row must be carried whole (never split mid-cell, never truncated)"
    )
    for chunk in chunks:
        if not chunk.oversized_atomic:
            assert chunk.token_count <= 60, (
                f"{chunk.chunk_id}: unflagged chunk over the cap ({chunk.token_count})"
            )


def test_single_token_dense_word_cannot_exceed_cap_silently():
    """Review finding #139, the same `not current` branch on prose: a
    single 'word' whose token count alone exceeds the cap (a long DOI/URL
    under a tight cap) must not sail through unmarked — it is emitted
    alone, flagged ``oversized_atomic=True``; every unflagged chunk
    respects the cap and no text is dropped."""
    dense = "https://doi.example.invalid/10.9999/" + ".".join(f"seg{i}" for i in range(40))
    prose = f"The invented record is archived at {dense} for reference purposes."
    cfg = config(max_tokens=30, min_tokens=1, token_counter=punct_tokens)
    chunks = chunk_document(
        doc("syn-dense-word", [heading("1 Archive"), text(prose)]), manifest_entry(), cfg
    )
    assert any(dense in c.body for c in chunks), "the dense word must not be dropped"
    for chunk in chunks:
        if dense in chunk.body and chunk.token_count > 30:
            assert chunk.oversized_atomic, (
                f"{chunk.chunk_id}: over-cap dense-word chunk emitted without the "
                "oversized_atomic flag (silent cap violation)"
            )
        if not chunk.oversized_atomic:
            assert chunk.token_count <= 30, (
                f"{chunk.chunk_id}: unflagged chunk over the cap ({chunk.token_count})"
            )


def test_chunk_ids_unique_within_a_run():
    """Review finding #138 (blocker): chunk ids key the whole #9
    incremental contract (Qdrant upsert by id), so two chunks in one run
    must NEVER share an id — even when a section repeats an identical
    body. On the real ESD review two bare-figure chunks collided on
    ``esd_tipping_review:53014e084096ffab``; this pins the general rule
    with (a) an identical repeated sentence and (b) two figures carrying
    identical captions."""
    repeated = sentence(30, "dup")
    same_caption = "Figure 9. Invented identical caption shared by two figures."
    blocks = [
        heading("1 Repetition section"),
        text(repeated),
        text(repeated),
        heading("2 Twin figures"),
        figure(caption=same_caption),
        figure(caption=same_caption),
    ]
    cfg = config(max_tokens=40, min_tokens=1)
    chunks = chunk_document(doc("syn-dup-bodies", blocks), manifest_entry(), cfg)
    twin_bodies = [c for c in chunks if same_caption in c.body]
    assert len(twin_bodies) == 2, "both identical-caption figures must chunk"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"duplicate chunk ids within one run: "
        f"{[i for i in ids if ids.count(i) > 1]} — an id collision makes the "
        "content-hash upsert semantics ill-defined (#9)"
    )


def test_duplicate_body_ids_stable_across_reruns_and_edits_elsewhere():
    """The #138 disambiguation must not cost id stability: identical input
    twice → identical ids; an edit in ANOTHER section leaves the
    duplicated bodies' ids untouched (their embeddings stay reusable)."""
    repeated = sentence(30, "stabledup")

    def build(other: str):
        return doc(
            "syn-dup-stable",
            [
                heading("1 Repetition section"),
                text(repeated),
                text(repeated),
                heading("2 Other section"),
                text(other),
            ],
        )

    cfg = config(max_tokens=40, min_tokens=1)
    entry = manifest_entry("syn-dup-stable")
    first = chunk_document(build(sentence(12, "otherv1")), entry, cfg)
    second = chunk_document(build(sentence(12, "otherv1")), entry, cfg)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    edited = chunk_document(build(sentence(12, "otherv2")), entry, cfg)
    dup_ids = lambda chunks: [c.chunk_id for c in chunks if repeated in c.body]  # noqa: E731
    assert dup_ids(first) == dup_ids(edited), (
        "editing another section must not move the duplicated bodies' ids"
    )


def test_bare_placeholder_chunks_not_emitted():
    """Review finding #138 (blocker), policy pin: a figure that has no
    caption (and no caption-bearing adjacent text) carries nothing
    citable — it must produce NO chunk, never a zero-content 1-token
    '[FIGURE]' citable unit. Same for a cell-less, caption-less table.
    A captioned figure still chunks."""
    blocks = [
        heading("1 Figures"),
        text(sentence(25, "prose")),
        figure(caption=None),
        Block(BlockType.TABLE, "[TABLE]", caption=None),
        figure(caption="Figure 4. Invented captioned figure that stays citable."),
    ]
    chunks = chunk_document(doc("syn-bare-figures", blocks), manifest_entry(), config())
    bodies = [c.body for c in chunks]
    assert "[FIGURE]" not in bodies and "[TABLE]" not in bodies, (
        f"bare placeholder emitted as a citable chunk: {bodies}"
    )
    assert any("Invented captioned figure" in body for body in bodies), (
        "a captioned figure must still be emitted citable"
    )


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
