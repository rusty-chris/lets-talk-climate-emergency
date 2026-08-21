"""Shared local-model-weights guard for the integration tier.

Same shape as tests/_docker.py (review finding #32): on a dev machine
without the cached bge-m3 weights the real-model smoke SKIPS; in CI
(IMPLEMENTATION.md §3 says integration runs with weights cached) their
absence is an infrastructure failure and must be RED — a skip would
silently turn the only real-model check into a green no-op.

The probe looks for a downloaded snapshot in the Hugging Face hub cache
(honouring HF_HUB_CACHE / HF_HOME) without importing the model stack —
detection must stay as cheap as the tests it guards, and must NEVER
trigger a multi-GB download itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.indexing import BGE_M3_MODEL_ID, BGE_M3_REVISION


def _hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def bge_m3_weights_available() -> bool:
    """True when a non-empty local snapshot of the pinned model at the
    PINNED revision exists (finding #163: any other cached revision is
    different weights under the same model id and does not count)."""
    snapshot = (
        _hub_cache_dir()
        / f"models--{BGE_M3_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / BGE_M3_REVISION
    )
    return snapshot.is_dir() and any(snapshot.iterdir())


def require_bge_m3_weights() -> None:
    """Skip locally when the weights are missing; fail in CI.

    Call at the top of every real-model test instead of a skipif
    decorator (the #32 convention).
    """
    if bge_m3_weights_available():
        return
    if os.environ.get("CI"):
        pytest.fail(
            "bge-m3 weights required in CI but not cached — the real-model "
            "smoke would otherwise silently no-op with a green job"
        )
    pytest.skip(
        f"local {BGE_M3_MODEL_ID} weights at pinned revision "
        f"{BGE_M3_REVISION} not available in this environment"
    )
