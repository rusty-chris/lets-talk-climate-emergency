"""Scheduled-CI live fetch of the real MVP pack (issue #14) — `live` tier.

Issue TDD plan item 9 and review finding #52's real-data check. These
tests hit the six origin archives (NASA GISS, Met Office, NOAA GML, NCEI
Paleo x2, OWID) — network, no LLM key — and therefore run in scheduled
CI / pre-release only, never per PR (IMPLEMENTATION.md §3/§6: the
integration and smoke jobs select `-m "<tier> and not live"`).

An expected failure shape here is upstream drift: providers append new
rows monthly, changing the bytes and the hash. That is the flow working
as designed — the remedy is a conscious re-pin commit (refreshed sha256 +
regenerated coverage via `make datasets`), never a weakened check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from charts import datasets, pack

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"


def test_fetch_all_datasets_verify_sha256(tmp_path):
    """TDD plan item 9: a real fetch of all six MVP datasets from their

    origin URLs verifies against the committed pins end-to-end (fetch ->
    sha256 -> parse -> coverage -> land).
    """
    result = datasets.fetch_all(MANIFEST_PATH, tmp_path / "landed")
    assert set(result) == pack.MVP_DATASET_IDS
    for ds_id, landed in result.items():
        assert landed.is_file(), f"{ds_id}: nothing landed at {landed}"


def test_coverage_reflects_usable_rows(tmp_path):
    """Review finding #52's named test, real-data side: for each dataset,

    the manifest coverage endpoints equal the first/last rows the
    committed parser actually yields from the fetched file.
    """
    landed = datasets.fetch_all(MANIFEST_PATH, tmp_path / "landed")
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))["datasets"]
    for ds_id in sorted(pack.MVP_DATASET_IDS):
        frame = pack.PARSERS[ds_id](landed[ds_id])
        computed = pack.dataset_coverage(frame, raw[ds_id]["time_axis"])
        assert computed == raw[ds_id]["coverage"], (
            f"{ds_id}: manifest coverage {raw[ds_id]['coverage']} disagrees with the parsed "
            f"usable extent {computed} — regenerate coverage from parser output (#52)"
        )
