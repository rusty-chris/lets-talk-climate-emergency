"""Permalink determinism of the pure render seam (issue #17) — RED.

Failing tests pinning the /chart/<spec-hash> re-render contract (DESIGN
§3.7): the pure function spec + pinned data -> Vega-Lite JSON is
byte-identical across calls and processes, the artefact bundle is
addressed by charts.spec.spec_hash, and the semantically inert _meta
block changes neither the hash nor the artefact. PNG determinism is
deliberately NOT pinned at the byte level (font rasterisation varies);
the loose contract lives in tests/integration/test_chart_render_exports.py.

All inputs are SYNTHETIC FIXTURES (tests/_chart_render_fixtures.py).
"""

from __future__ import annotations

import json

from charts import render
from charts.spec import spec_hash
from tests._chart_render_fixtures import (
    LINE_EXTENTS,
    PANEL_EXTENTS,
    SITE_URL,
    frames,
    line_spec,
    panel_spec,
    render_manifest,
    validated,
)


def _vl_text(spec_builder, extents) -> str:
    vl = render.build_vega_lite(
        validated(spec_builder(), extents), frames(), render_manifest(), SITE_URL
    )
    return render.vega_lite_json_text(vl)


def test_same_spec_and_data_produce_byte_identical_vl_json():
    """Two fully independent builds (fresh spec dicts, fresh frames)
    serialise to the identical canonical text — the re-render guarantee
    a stored spec + pinned dataset version relies on."""
    assert _vl_text(line_spec, LINE_EXTENTS) == _vl_text(line_spec, LINE_EXTENTS)
    assert _vl_text(panel_spec, PANEL_EXTENTS) == _vl_text(panel_spec, PANEL_EXTENTS)


def test_vega_lite_json_text_is_canonical_and_newline_terminated():
    """The byte-identity surface: valid JSON, newline-terminated (the
    repo's end-of-file convention), round-tripping to the same dict."""
    text = _vl_text(line_spec, LINE_EXTENTS)
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert render.vega_lite_json_text(parsed) == text


def test_artifact_is_addressed_by_spec_hash():
    """render_chart's bundle carries the canonical spec hash — the
    permalink identity — and its vega_lite_text equals the pure seam's
    output for the same inputs."""
    artifact = render.render_chart(line_spec(), frames(), render_manifest(), SITE_URL)
    assert artifact.spec_hash == spec_hash(line_spec())
    assert artifact.vega_lite_text == _vl_text(line_spec, LINE_EXTENTS)
    assert artifact.alt_text, "the bundle ships alt text"
    assert artifact.csv_text.startswith("#"), "the bundle ships the attributed CSV"


def test_meta_block_changes_neither_hash_nor_artifact():
    """_meta is semantically inert (issue #15): adding provenance notes
    must not move the permalink or change a single artefact byte."""
    with_meta = line_spec()
    with_meta["_meta"] = {"provenance": "synthetic determinism probe", "n": 1}
    plain = render.render_chart(line_spec(), frames(), render_manifest(), SITE_URL)
    noted = render.render_chart(with_meta, frames(), render_manifest(), SITE_URL)
    assert noted.spec_hash == plain.spec_hash
    assert noted.vega_lite_text == plain.vega_lite_text


def test_semantic_change_changes_hash_and_artifact():
    """The converse guard: a semantic edit (title) moves both the hash
    and the artefact text, so distinct charts can never collide at one
    permalink."""
    changed = line_spec()
    changed["title"] = "A different synthetic title"
    plain = render.render_chart(line_spec(), frames(), render_manifest(), SITE_URL)
    edited = render.render_chart(changed, frames(), render_manifest(), SITE_URL)
    assert edited.spec_hash != plain.spec_hash
    assert edited.vega_lite_text != plain.vega_lite_text
