"""Shared builders and embedding doubles for the issue-9 indexing red suite.

Everything here is synthetic — the fictional Aurelian-Basin universe of
the #24 fixture corpus, invented sentences, .invalid URLs. The chunks are
produced by the REAL production chunker (`ingestion.pipeline.chunk_document`)
over synthetic StructuredDocs, so the indexer is tested against the exact
upstream surface it will consume (issue #7's ChunkRecord), never a
hand-rolled imitation of it.

The embedding doubles implement the `rag.indexing.EmbeddingModel` seam:

- `HashEmbeddingModel` — deterministic, hash-based dense + sparse vectors
  computed from the text alone. NEVER touches the network, NEVER loads
  model weights (unit-tier rule, IMPLEMENTATION.md §3). Texts sharing
  words get similar dense vectors and overlapping sparse indices, so
  relevance-shaped assertions (the known-query test) are meaningful yet
  fully deterministic.
- `RecordingEmbeddingModel` — wraps any model and records every batch of
  texts encoded; the instrument the incremental-re-embedding tests use
  to assert "unchanged chunks are not re-encoded" as observed behaviour
  rather than trusting a report field.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ingestion.pipeline import ChunkRecord, DocumentIngestRecord, chunk_document
from rag.indexing import Embedding, build_index
from tests._ingestion_fixtures import config, doc, heading, manifest_entry, sentence, text

#: The corpus version the fixture index is built under; version-mismatch
#: tests query with a different string.
FIXTURE_CORPUS_VERSION = "fixture-corpus-v1"

#: Default collection name for the red suite (unit tier uses a fresh
#: in-memory client per test, so no cross-test collision is possible).
COLLECTION = "fixture-index"

#: The distinctive invented sentence the changed-chunk tests vary. It
#: lives in exactly one section of syn-idx-attribution.
DEFAULT_CLOSER = (
    "The review closes by noting that attribution confidence for the "
    "Aurelian drying signal strengthens with every added observation year."
)


class HashEmbeddingModel:
    """Deterministic fake bge-m3: hash-based dense + learned-sparse shapes.

    Dense: words are hashed into `dense_dim` buckets (term-frequency
    weighted, L2-normalised) — cosine similarity tracks word overlap.
    Sparse: one index per distinct word (a stable hash), weight = term
    frequency — a faithful *shape* stand-in for bge-m3's learned lexical
    weights. No weights are downloaded; no randomness anywhere.
    """

    model_id = "hash-fake-embedder-v1"
    dense_dim = 64

    _WORD_RE = re.compile(r"[a-z0-9°]+")

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text_value: str) -> Embedding:
        counts: dict[str, int] = {}
        for word in self._WORD_RE.findall(text_value.lower()):
            counts[word] = counts.get(word, 0) + 1
        dense = [0.0] * self.dense_dim
        sparse: dict[int, float] = {}
        for word, count in counts.items():
            digest = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:12], 16)
            dense[digest % self.dense_dim] += float(count)
            index = digest % 100_003
            sparse[index] = sparse.get(index, 0.0) + float(count)
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        return Embedding(dense=tuple(v / norm for v in dense), sparse=sparse)


class RecordingEmbeddingModel:
    """Wraps an EmbeddingModel and records every text it is asked to encode."""

    def __init__(self, inner: HashEmbeddingModel | None = None) -> None:
        self.inner = inner or HashEmbeddingModel()
        self.batches: list[tuple[str, ...]] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def dense_dim(self) -> int:
        return self.inner.dense_dim

    @property
    def encoded_texts(self) -> list[str]:
        return [text_value for batch in self.batches for text_value in batch]

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        self.batches.append(tuple(texts))
        return self.inner.encode(texts)


def _basin_doc():
    return doc(
        "syn-idx-basin",
        [
            heading("1 Observed warming", level=1),
            text(
                "Surface temperatures across the Aurelian Basin have very likely risen "
                "by one point nine degrees since the fictional baseline period, with the "
                "strongest warming recorded over the eastern escarpment stations."
            ),
            heading("2 Water resources", level=1),
            text(
                "Reservoir inflows in the basin declined in twenty one of the last "
                "twenty five invented water years, and managed aquifer recharge now "
                "supplies a growing share of dry season demand in the lowland districts."
            ),
            heading("3 Assessed ranges", level=1),
            text(
                "The assessed range for committed basin warming spans one point two to "
                "two point one degrees under the reference pathway, a span the fictional "
                "assessment reports with high confidence for planning purposes."
            ),
        ],
        title="Meridian Climate Assessment Cycle 3: The Aurelian Basin (invented)",
    )


def _attribution_doc(closer: str = DEFAULT_CLOSER):
    return doc(
        "syn-idx-attribution",
        [
            heading("1 Attribution methods", level=1),
            text(
                "Attribution studies of Aurelian Basin drying compare observed rainfall "
                "decline against invented counterfactual simulations, isolating the "
                "anthropogenic contribution to the drying trend from natural variability."
            ),
            heading("2 Rainfall decline findings", level=1),
            text(
                "The synthesis attributes most of the observed rainfall decline and "
                "soil moisture drying across the Aurelian Basin to circulation shifts, "
                "an attribution the invented authors rate as likely on current evidence."
            ),
            heading("3 Confidence and outlook", level=1),
            text(closer),
        ],
        title="Attribution of Aurelian Basin drying: a review (invented)",
    )


def _beyond_doc():
    return doc(
        "syn-idx-beyond",
        [
            heading("1 The provocation", level=1),
            text(
                "Pellier and Vance argue that committed warming for the basin may exceed "
                "the assessed range once slow feedback processes are counted, a claim the "
                "fictional assessment community has not adopted."
            ),
            heading("2 Reception", level=1),
            text(
                "Reviewers of the invented manuscript note that the above range estimate "
                "rests on a contested aerosol history, and that the mainstream assessed "
                "range remains the planning benchmark for the basin authorities."
            ),
        ],
        title="Committed warming may exceed assessed ranges (invented)",
    )


def _voices_doc():
    return doc(
        "syn-idx-voices",
        [
            heading("1 The Lanternwood Briefing", level=1),
            text(
                "The Lanternwood Briefing campaign gathered invented residents each "
                "month to read the assessment aloud in the market square, turning the "
                "fictional basin findings into a shared civic ritual."
            ),
            heading("2 What organisers learned", level=1),
            text(
                "Campaign organisers recount that listeners trusted the readings most "
                "when the invented speakers named their own villages, a lesson the "
                "voices layer of this fictional project preserves deliberately."
            ),
        ],
        title="Voices: the Lanternwood Briefing campaign (invented)",
    )


def _filler_doc(letter: str, section_count: int = 12):
    blocks = []
    for section_index in range(section_count):
        tag = f"filler{letter}{section_index}"
        blocks.append(heading(f"{section_index + 1} Filler section {tag}", level=1))
        blocks.append(text(sentence(24, tag)))
    return doc(
        f"syn-idx-filler-{letter}",
        blocks,
        title=f"Synthetic filler corpus volume {letter.upper()} (invented)",
    )


def _entry_overrides(doc_id: str) -> dict:
    if doc_id == "syn-idx-voices":
        return {"source_type": "voices"}
    if doc_id == "syn-idx-beyond":
        return {"consensus_position": "beyond-assessed-range"}
    return {}


def fixture_corpus(
    closer: str = DEFAULT_CLOSER,
) -> tuple[list[ChunkRecord], dict[str, DocumentIngestRecord]]:
    """The deterministic fixture corpus: >40 production chunks + records.

    Real `chunk_document` output over synthetic docs: evidence docs, one
    `source_type: voices` doc, one `consensus_position:
    beyond-assessed-range` doc, and filler volumes pushing the chunk
    count past the fused top-40. `closer` varies exactly one section of
    syn-idx-attribution so changed-chunk tests can flip a single chunk's
    content while every other chunk stays byte-identical.
    """
    docs = [
        _basin_doc(),
        _attribution_doc(closer),
        _beyond_doc(),
        _voices_doc(),
        _filler_doc("a"),
        _filler_doc("b"),
        _filler_doc("c"),
    ]
    chunks: list[ChunkRecord] = []
    records: dict[str, DocumentIngestRecord] = {}
    for sdoc in docs:
        entry = manifest_entry(sdoc.doc_id, title=sdoc.title, **_entry_overrides(sdoc.doc_id))
        chunks.extend(chunk_document(sdoc, entry, config()))
        records[sdoc.doc_id] = DocumentIngestRecord(doc_id=sdoc.doc_id, parse_backend="html")
    return chunks, records


def degraded_records(
    records: dict[str, DocumentIngestRecord], doc_id: str
) -> dict[str, DocumentIngestRecord]:
    """The same records with one document flagged as an unreviewed
    PyMuPDF-degraded parse (the #143 hand-review state)."""
    flagged = dict(records)
    flagged[doc_id] = DocumentIngestRecord(
        doc_id=doc_id,
        parse_backend="pymupdf",
        degraded_fallback=True,
        needs_hand_review=True,
        warnings=(f"{doc_id}: parsed with the PyMuPDF fallback — flagged for hand review",),
    )
    return flagged


def titled_doc_chunks(title: str) -> tuple[list[ChunkRecord], dict[str, DocumentIngestRecord]]:
    """One small doc whose title (hence every context header) is variable.

    `ChunkRecord.chunk_id` hashes doc id + section path + body — NOT the
    title — so two calls with different titles yield identical chunk ids
    but different `embedding_text`. This is the discriminating input for
    the content-hash-covers-the-header test.
    """
    sdoc = doc(
        "syn-idx-retitled",
        [
            heading("1 Stable section", level=1),
            text(
                "This invented body text never changes between builds, so any observed "
                "re-embedding must be caused by the context header alone and nothing else."
            ),
        ],
        title=title,
    )
    entry = manifest_entry(sdoc.doc_id, title=title)
    chunks = chunk_document(sdoc, entry, config())
    records = {sdoc.doc_id: DocumentIngestRecord(doc_id=sdoc.doc_id, parse_backend="html")}
    return chunks, records


def fresh_client():
    """An embedded (in-memory) Qdrant client — the same client interface
    the Docker server speaks, with no Docker, network or weights
    (verified against qdrant-client 1.19 local mode)."""
    from qdrant_client import QdrantClient

    return QdrantClient(":memory:")


def build(
    client,
    chunks,
    records,
    *,
    model,
    corpus_version=FIXTURE_CORPUS_VERSION,
    collection=COLLECTION,
):
    """build_index with the red suite's defaults filled in."""
    return build_index(
        client,
        collection,
        chunks,
        records,
        embedding_model=model,
        corpus_version=corpus_version,
    )
