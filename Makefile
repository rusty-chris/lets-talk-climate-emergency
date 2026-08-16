# Repo-root build targets. See IMPLEMENTATION.md and ADR-023 (DECISIONS.md)
# for why `corpus` re-fetches rather than reads committed data: no dataset
# or non-open corpus text files are ever committed to this repo.

CORPUS_MANIFEST ?= corpus/manifest.yaml
CORPUS_DIR ?= corpus

# Without .PHONY, `make corpus` against the repo's real corpus/ directory
# reports "Nothing to be done" and exits zero (issue #5 note) — the target
# must always run.
.PHONY: corpus

corpus:
	uv run python scripts/make_corpus.py --manifest $(CORPUS_MANIFEST) --corpus-dir $(CORPUS_DIR)
