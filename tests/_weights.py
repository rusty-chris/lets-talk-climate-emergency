"""Shared local-model-weights guard for the integration tier.

Same shape as tests/_docker.py (review finding #32): on a dev machine
without the cached model weights the real-model checks SKIP; in CI
(IMPLEMENTATION.md §3 says integration runs with weights cached) their
absence is an infrastructure failure and must be RED — a skip would
silently turn the only real-model checks into a green no-op.

The probe looks for a downloaded snapshot in the Hugging Face hub cache
(honouring HF_HUB_CACHE / HF_HOME) without importing the model stack —
detection must stay as cheap as the tests it guards, and must NEVER
trigger a multi-GB download itself.

Covers both pinned local models: bge-m3 (issue #9, ADR-005 — pinned to
a specific hub revision, finding #163: any other cached revision is
different weights under the same model id and does not count) and
bge-reranker-v2-m3 (issue #11, ADR-006).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.indexing import BGE_M3_MODEL_ID, BGE_M3_REVISION
from rag.retrieval import BGE_RERANKER_MODEL_ID, BGE_RERANKER_REVISION


def _hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _weights_available(model_id: str, revision: str | None = None) -> bool:
    """True when a non-empty local snapshot of ``model_id`` exists.

    With ``revision`` (finding #163), only a snapshot of exactly that
    revision counts — a cache holding any other revision is different
    weights under the same model id.
    """
    snapshots = _hub_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if revision is not None:
        snapshot = snapshots / revision
        return snapshot.is_dir() and any(snapshot.iterdir())
    if not snapshots.is_dir():
        return False
    return any(entry.is_dir() and any(entry.iterdir()) for entry in snapshots.iterdir())


def _require_weights(model_id: str, revision: str | None = None) -> None:
    """Skip locally when the weights are missing; fail in CI.

    Call at the top of every real-model test instead of a skipif
    decorator (the #32 convention).
    """
    if _weights_available(model_id, revision):
        return
    pinned = f" at pinned revision {revision}" if revision is not None else ""
    if os.environ.get("CI"):
        pytest.fail(
            f"{model_id}{pinned} weights required in CI but not cached — the "
            "real-model check would otherwise silently no-op with a green job"
        )
    pytest.skip(f"local {model_id}{pinned} weights not available in this environment")


def bge_m3_weights_available() -> bool:
    """True when a non-empty local snapshot of the pinned model at the
    PINNED revision exists (finding #163)."""
    return _weights_available(BGE_M3_MODEL_ID, BGE_M3_REVISION)


def require_bge_m3_weights() -> None:
    """The #32 guard for the pinned embedding model (issue #9)."""
    _require_weights(BGE_M3_MODEL_ID, BGE_M3_REVISION)


def bge_reranker_weights_available() -> bool:
    """True when a non-empty local snapshot of the pinned reranker at the
    PINNED revision exists (findings #163/#178)."""
    return _weights_available(BGE_RERANKER_MODEL_ID, BGE_RERANKER_REVISION)


def require_bge_reranker_weights() -> None:
    """The #32 guard for the pinned cross-encoder reranker (issue #11),
    revision-pinned per finding #178."""
    _require_weights(BGE_RERANKER_MODEL_ID, BGE_RERANKER_REVISION)
