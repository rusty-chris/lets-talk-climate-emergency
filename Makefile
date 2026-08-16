# Repo-root build targets. See IMPLEMENTATION.md and ADR-023 (DECISIONS.md)
# for why `corpus`/`datasets` re-fetch rather than read committed data: no
# dataset or non-open corpus text files are ever committed to this repo.

CORPUS_MANIFEST ?= corpus/manifest.yaml
CORPUS_DIR ?= corpus

DATASETS_MANIFEST ?= datasets/manifest.yaml
DATASETS_DIR ?= data/datasets

# Without .PHONY, `make corpus`/`make datasets` against the repo's real
# corpus/ or datasets/ directory reports "Nothing to be done" and exits
# zero (issue #5 note) — the targets must always run.
.PHONY: corpus datasets

corpus:
	uv run python scripts/make_corpus.py --manifest $(CORPUS_MANIFEST) --corpus-dir $(CORPUS_DIR)

datasets:
	uv run python scripts/make_datasets.py --manifest $(DATASETS_MANIFEST) --datasets-dir $(DATASETS_DIR)
