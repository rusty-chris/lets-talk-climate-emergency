"""Shared fixtures for the issue-19 transparency-page red suites.

Two kinds of input feed the renderers:

- the REAL manifests (``corpus/manifest.yaml``, ``datasets/manifest.yaml``)
  — the single sources of truth the /sources page is generated from; the
  tests load them with ``yaml.safe_load`` exactly as the build contract
  says, so every pin is manifest-driven and a new source cannot be
  forgotten;
- synthetic additions (Aurelian-Basin universe — clearly invented) used
  to pin that a manifest ADDITION appears on the page with zero code
  change, and that ``permitted_context: non-commercial-educational``
  entries carry their non-commercial note.

``FIXTURE_RESULTS_MD`` is a synthetic ``evals/RESULTS.md`` in the
issue-21 report shape (verdict line + gate table) with distinctive
numbers the /about pins look for.
"""

from __future__ import annotations

import copy
import html
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.yaml"
DATASETS_MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"


def load_real_corpus_manifest() -> dict[str, Any]:
    return yaml.safe_load(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))


def load_real_datasets_manifest() -> dict[str, Any]:
    return yaml.safe_load(DATASETS_MANIFEST_PATH.read_text(encoding="utf-8"))


def page_text(rendered_html: str) -> str:
    """HTML → comparable text: entity-unescaped, whitespace-collapsed.

    Attribution strings must appear manifest-VERBATIM on the page; the
    renderer may escape entities (``"`` → ``&quot;``) and wrap lines,
    neither of which changes the text — so the pins compare after
    unescaping and collapsing runs of whitespace.
    """
    return " ".join(html.unescape(rendered_html).split())


def contains_verbatim(rendered_html: str, needle: str) -> bool:
    return " ".join(needle.split()) in page_text(rendered_html)


def chars_between(text: str, first: str, second: str) -> int:
    """Smallest gap between an occurrence of ``first`` and one of ``second``.

    Asserts both are present; used for the ADR-018 pair-adjacency pin.
    """
    assert first in text, f"missing {first!r}"
    assert second in text, f"missing {second!r}"
    starts_first = [i for i in range(len(text)) if text.startswith(first, i)]
    starts_second = [i for i in range(len(text)) if text.startswith(second, i)]
    return min(
        abs(a - b) - (len(first) if a < b else len(second))
        for a in starts_first
        for b in starts_second
    )


# ---------------------------------------------------------------------------
# Synthetic manifest additions (invented — Aurelian Basin universe)
# ---------------------------------------------------------------------------

#: A brand-new open Tier-A document: pins "a manifest addition appears
#: on /sources with zero code change" and manifest-verbatim attribution.
SYNTHETIC_NEW_DOCUMENT: dict[str, Any] = {
    "id": "syn_meridian_assessment_c3",
    "title": "Meridian Climate Assessment, Cycle 3 (synthetic)",
    "licence": "CC BY 4.0",
    "licence_evidence": "Synthetic fixture: invented licensing evidence.",
    "attribution_text": (
        "Meridian Climate Council, 2099: Meridian Climate Assessment, "
        "Cycle 3. Aurelian Basin Institute (synthetic fixture)."
    ),
    "canonical_url": "https://example.invalid/meridian-assessment-c3",
    "source_url": "https://example.invalid/meridian-assessment-c3.pdf",
    "path": "syn_meridian_assessment_c3.pdf",
    "redistributable": True,
    "permitted_context": "open",
    "permission_evidence": None,
    "consensus_position": "assessed",
    "source_type": "evidence",
    "source_tier": "A",
    "sha256": "0" * 64,
    "retrieved_at": "2026-08-20",
    "human_signoff": {
        "who": "synthetic fixture",
        "date": "2026-08-20",
        "note": "invented for the transparency red suite",
    },
}

#: A Tier-B non-commercial-educational document: pins the NC note.
SYNTHETIC_NC_DOCUMENT: dict[str, Any] = {
    **SYNTHETIC_NEW_DOCUMENT,
    "id": "syn_basin_gap_report",
    "title": "Aurelian Basin Emissions Gap Report (synthetic)",
    "licence": "Synthetic educational/non-profit reproduction terms",
    "attribution_text": (
        "Aurelian Basin Environment Programme, 2099: Basin Emissions Gap "
        "Report (synthetic fixture)."
    ),
    "canonical_url": "https://example.invalid/basin-gap-report",
    "redistributable": False,
    "permitted_context": "non-commercial-educational",
    "source_tier": "B",
}


def corpus_manifest_with(*extra_documents: dict[str, Any]) -> dict[str, Any]:
    """The real corpus manifest plus ``extra_documents`` appended."""
    manifest = copy.deepcopy(load_real_corpus_manifest())
    manifest["documents"].extend(copy.deepcopy(doc) for doc in extra_documents)
    return manifest


#: A brand-new dataset entry: same zero-code-change pin for datasets.
SYNTHETIC_NEW_DATASET: dict[str, Any] = {
    "title": "Aurelian Basin annual anomaly (synthetic)",
    "provider": "Aurelian Basin Institute (synthetic)",
    "url": "https://example.invalid/basin-anomaly.csv",
    "citation": "Basin Institute 2099 (synthetic fixture)",
    "licence": "CC0 (synthetic fixture)",
    "attribution_text": "Aurelian Basin Institute, annual anomaly series (synthetic fixture)",
    "permitted_context": "open",
    "in_chart_pack": False,
    "sha256": "1" * 64,
    "retrieved_at": "2026-08-20",
    "human_signoff": {
        "who": "synthetic fixture",
        "date": "2026-08-20",
        "note": "invented for the transparency red suite",
    },
}


def datasets_manifest_with(**extra_datasets: dict[str, Any]) -> dict[str, Any]:
    """The real datasets manifest plus ``extra_datasets`` (id=entry)."""
    manifest = copy.deepcopy(load_real_datasets_manifest())
    manifest["datasets"].update(copy.deepcopy(extra_datasets))
    return manifest


# ---------------------------------------------------------------------------
# Eval results (the issue-21 RESULTS.md shape, synthetic numbers)
# ---------------------------------------------------------------------------

FIXTURE_RESULTS_VERDICT = "PASSED"
FIXTURE_RESULTS_SCORES = ("17/20", "94/95")

FIXTURE_RESULTS_MD = """# Release eval results (synthetic fixture)

Release verdict: PASSED
Selected model: claude-haiku-4-5

| gate | score | threshold |
| --- | --- | --- |
| retrieval_refusal | 17/20 | >0.90 |
| citation_support | 94/95 | >=0.90 |
"""


def write_fixture_results(tmp_path: Path) -> Path:
    path = tmp_path / "RESULTS.md"
    path.write_text(FIXTURE_RESULTS_MD, encoding="utf-8")
    return path


def write_fixture_manifests(tmp_path: Path) -> tuple[Path, Path]:
    """Copies of the REAL manifests under tmp_path (build-seam inputs)."""
    corpus = tmp_path / "corpus-manifest.yaml"
    corpus.write_text(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    datasets = tmp_path / "datasets-manifest.yaml"
    datasets.write_text(DATASETS_MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return corpus, datasets
