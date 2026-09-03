# Repo-root build targets. See IMPLEMENTATION.md and ADR-023 (DECISIONS.md)
# for why `corpus`/`datasets` re-fetch rather than read committed data: no
# dataset or non-open corpus text files are ever committed to this repo.

CORPUS_MANIFEST ?= corpus/manifest.yaml
CORPUS_DIR ?= corpus

INGEST_MANIFEST ?= corpus/manifest.yaml
INGEST_CORPUS_DIR ?= corpus
INGEST_OUT_DIR ?= data/ingest

VOICES_FILE ?= voices/voices.yaml

DATASETS_MANIFEST ?= datasets/manifest.yaml
DATASETS_DIR ?= data/datasets

# Without .PHONY, `make corpus`/`make datasets` against the repo's real
# corpus/ or datasets/ directory reports "Nothing to be done" and exits
# zero (issue #5 note) — the targets must always run.
.PHONY: corpus ingest voices datasets

corpus:
	uv run python scripts/make_corpus.py --manifest $(CORPUS_MANIFEST) --corpus-dir $(CORPUS_DIR)

# The corpus/ingest split is deliberate (review #145): `make corpus` is
# the fast #5 licensing/invariant gate (fetch open text + verify pins +
# run every invariant); `make ingest` is the #7 production pipeline
# (manifest gate -> verified fetch -> parse -> chunk -> citation blocks)
# and pulls the heavy Docling parse. Chunk/block payloads land under
# data/ingest (gitignored); the run record at corpus/ingest_run.json.
# It also ingests the voices layer (#8) so a full ingest produces the
# source_type: voices chunks the indexer needs.
ingest:
	uv run python scripts/ingest_corpus.py --manifest $(INGEST_MANIFEST) --corpus-dir $(INGEST_CORPUS_DIR) --out-dir $(INGEST_OUT_DIR)
	uv run python scripts/ingest_voices.py --voices $(VOICES_FILE) --out-dir $(INGEST_OUT_DIR)

# `make voices` runs the voices ingest alone (#8, DESIGN §2.5): first-party
# voices.yaml -> #7 chunker -> source_type: voices chunks + citation
# blocks under the "About the movement" attribution. No network / no
# Docling — voices text is first-party and in-repo.
voices:
	uv run python scripts/ingest_voices.py --voices $(VOICES_FILE) --out-dir $(INGEST_OUT_DIR)

datasets:
	uv run python scripts/make_datasets.py --manifest $(DATASETS_MANIFEST) --datasets-dir $(DATASETS_DIR)
