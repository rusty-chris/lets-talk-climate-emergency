"""Canonical ChartSpec hashing for /chart/<hash> permalinks (issue #15) — RED.

Issue TDD plan item 10: semantically identical specs hash identically
(deep key order, provenance metadata and number representation must not
matter — permalinks are forever); any semantic change changes the hash.
Unit tier, pure over dicts; the only committed input is the flagship spec.

All inline specs are SYNTHETIC FIXTURES authored for these tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from charts import spec as chartspec

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAGSHIP_PATH = REPO_ROOT / "charts" / "spike" / "flagship_spec.json"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def base_spec() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: a small spec exercising nesting, arrays, unicode

    and mixed number kinds.
    """
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-hash",
        "chart_type": "line",
        "title": "Synthetic CO₂-style series",
        "time_range_ce": [1900, 2000],
        "series": [
            {
                "id": "w",
                "label": "Widgets (wd)",
                "unit": "wd",
                "dataset": "syn_abs_new",
                "scale_domain": [0, 120],
            }
        ],
    }


def _deep_reorder(obj: Any) -> Any:
    """Rebuild every dict with reversed key insertion order."""
    if isinstance(obj, dict):
        return {k: _deep_reorder(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_deep_reorder(v) for v in obj]
    return obj


def test_spec_hash_is_permalink_shaped():
    digest = chartspec.spec_hash(base_spec())
    assert _HASH_RE.match(digest), f"expected 64-char lowercase hex sha256, got {digest!r}"


def test_spec_hash_stable_under_key_order():
    spec = base_spec()
    assert chartspec.spec_hash(spec) == chartspec.spec_hash(_deep_reorder(spec))


def test_spec_hash_ignores_meta_block():
    """The top-level _meta provenance/sign-off block is not chart

    semantics: re-annotating provenance must not move a permalink.
    """
    spec = base_spec()
    annotated = base_spec()
    annotated["_meta"] = {"provenance": "synthetic", "content_signoff": {"who": "tests"}}
    assert chartspec.spec_hash(spec) == chartspec.spec_hash(annotated)


def test_spec_hash_number_representation_canonical():
    """Migration safety: a serialiser that turns 1900 into 1900.0 (or a

    YAML/JSON round-trip that floats an int) must not move every
    permalink — numerically equal values hash identically.
    """
    spec = base_spec()
    floated = base_spec()
    floated["time_range_ce"] = [1900.0, 2000.0]
    floated["series"][0]["scale_domain"] = [0.0, 120.0]
    assert chartspec.spec_hash(spec) == chartspec.spec_hash(floated)


def test_spec_hash_changes_on_semantic_change():
    """Any semantic change — title, range, domain, series, version —

    yields a new, distinct hash (a permalink pins exact chart semantics).
    """
    baseline = chartspec.spec_hash(base_spec())

    mutations: list[dict[str, Any]] = []

    changed = base_spec()
    changed["title"] = "A different title"
    mutations.append(changed)

    changed = base_spec()
    changed["time_range_ce"] = [1950, 2000]
    mutations.append(changed)

    changed = base_spec()
    changed["series"][0]["scale_domain"] = [0, 130]
    mutations.append(changed)

    changed = base_spec()
    changed["series"][0]["label"] = "Widgets (renamed)"
    mutations.append(changed)

    changed = base_spec()
    changed["spec_version"] = "1.0.1"
    mutations.append(changed)

    digests = [chartspec.spec_hash(m) for m in mutations]
    assert baseline not in digests
    assert len(set(digests)) == len(digests), "distinct changes must hash distinctly"


def test_flagship_spec_hashes_stably():
    """The committed flagship hashes deterministically across parses and

    key reorders — the permalink contract for the reference chart.
    """
    flagship = json.loads(FLAGSHIP_PATH.read_text(encoding="utf-8"))
    reparsed = json.loads(FLAGSHIP_PATH.read_text(encoding="utf-8"))
    digest = chartspec.spec_hash(flagship)
    assert _HASH_RE.match(digest)
    assert digest == chartspec.spec_hash(_deep_reorder(reparsed))
