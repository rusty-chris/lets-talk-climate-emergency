"""Service import hygiene (issue #22, actioning issue #125's api-image half).

Process-topology DECISION pinned here (flagged for ratification in the
#22 red-phase notes): retrieval runs IN the api process — the api calls
qdrant plus the LOCAL bge-m3/bge-reranker models (DESIGN §3.2
"Embeddings: bge-m3, local"; ADR-016's three-service compose has no
retrieval worker, and a fourth torch-loaded process on the <£20/month VM
would cost memory, not save it). Therefore torch IS a runtime dependency
of the api image, and issue #125's premise ("api/ui … never parse PDFs"
⇒ no torch) is only half right: the INGESTION parse backends
(docling/pymupdf) genuinely never belong in the serving image and should
move to an optional dependency group (#125's own red-first test change),
but the embedding/reranker stack stays.

What the SERVICE must guarantee regardless of image contents — pinned
here: importing the service NEVER loads the heavy stack. Model loading
happens lazily behind the injected retrieval seam (the
``rag.retrieval.BgeRerankerV2M3`` convention), so process startup, unit
tests, health checks and the paused read-only state never pay a
multi-GB import for code paths that never infer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: Modules that must NOT be imported as a side effect of importing the
#: service: the torch stack (retrieval loads it lazily at first use) and
#: the ingestion parse backends (never a serving dependency — #125).
HEAVY_MODULES = ("torch", "transformers", "sentence_transformers", "docling", "fitz")

_PROBE = """
import json, sys
import service.app
import service.main
import service.config
import service.budget
import service.rate_limit
import service.exchange_log
import service.starter_cache
import service.chart_store
loaded = [name for name in {heavy!r} if name in sys.modules]
print(json.dumps(loaded))
"""


def test_service_import_does_not_load_the_heavy_stack() -> None:
    """`import service.*` in a fresh interpreter pulls no torch/docling."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY_MODULES)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"importing the service loaded heavy modules {loaded}: model/parse "
        "imports must stay lazy behind the retrieval seam (issue #125 / "
        "IMPLEMENTATION.md §1)"
    )


def test_service_module_has_no_ingestion_parse_dependency() -> None:
    """The service package never imports the ingestion parse backends,
    even lazily: parsing is an offline pipeline concern (#125)."""
    service_dir = Path(__file__).resolve().parents[2] / "service"
    offending: list[str] = []
    for path in sorted(service_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in ("import docling", "from docling", "import fitz", "from fitz"):
            if needle in text:
                offending.append(f"{path.name}: {needle}")
    assert offending == [], offending
