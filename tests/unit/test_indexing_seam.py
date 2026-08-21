"""The embedding seam's import hygiene + pinned design constants (issue #9) — RED.

IMPLEMENTATION.md §1/§3: the real bge-m3 stack (torch, transformers,
FlagEmbedding) is heavyweight and lives behind the EmbeddingModel seam;
nothing in the unit tier may load it. The seam is only real if importing
the module doesn't drag the weights stack in.
"""

from __future__ import annotations

import subprocess
import sys

from rag import indexing


def test_importing_indexing_module_stays_weight_free() -> None:
    """`import rag.indexing` must not import the model stack.

    Checked in a fresh interpreter so this test cannot be fooled by
    modules other tests already imported. If this fails, the heavy
    imports have leaked to module level and every unit test (and every
    service cold start that never embeds) pays the multi-GB stack.
    """
    probe = (
        "import sys\n"
        "import rag.indexing\n"
        "heavy = {'torch', 'transformers', 'FlagEmbedding', 'docling', 'sentence_transformers'}\n"
        "loaded = sorted(heavy & set(sys.modules))\n"
        "assert not loaded, f'importing rag.indexing loaded the heavy stack: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_contract_constants_pin_design_values() -> None:
    """The design numbers are contract, not convention: the §3.2 fused
    top-40, the ADR-005 model pin and its 1024-dim dense space, and the
    two named-vector keys the collection schema uses."""
    assert indexing.DEFAULT_TOP_K == 40
    assert indexing.BGE_M3_MODEL_ID == "BAAI/bge-m3"
    assert indexing.BGE_M3_DENSE_DIM == 1024
    assert indexing.DENSE_VECTOR_NAME == "dense"
    assert indexing.SPARSE_VECTOR_NAME == "sparse"
