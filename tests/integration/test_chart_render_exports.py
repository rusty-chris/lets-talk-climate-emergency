"""vl-convert exports + flagship snapshot + permalink re-render (issue #17) — RED.

Integration tier (IMPLEMENTATION.md §3): exercises the real vl-convert
edge, still no network and no LLM key. Assertions are structural — SVG
text/elements and Vega-Lite source identity — never PNG byte-equality
(font rasterisation varies across platforms); PNG determinism is pinned
loosely via image dimensions and the sha256 of the Vega-Lite source.

#117 discipline: the flagship renders here from SYNTHETIC frames invented
inline, against an in-memory pack-confirmed copy of the real manifest
(the committed manifest keeps in_chart_pack: false for the provisional
datasets until issue #23). No expected value in this file derives from
Kaufman/Bereiter or any real dataset; assertions are presence-of-element,
never data values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pandas as pd
import pytest
import yaml

from charts import render
from charts.spec import spec_hash

vl_convert = pytest.importorskip(
    "vl_convert", reason="vl-convert is the integration-tier export edge"
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAGSHIP_PATH = REPO_ROOT / "charts" / "spike" / "flagship_spec.json"
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"

SITE_URL = "https://climate-chat.example.test"


def flagship_spec() -> dict:
    return json.loads(FLAGSHIP_PATH.read_text(encoding="utf-8"))


def pack_confirmed_manifest() -> dict:
    """The real manifest as it will look once #23's confirmations land —
    in-memory only; the committed file must not change until then."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    confirmed = copy.deepcopy(manifest)
    for entry in confirmed["datasets"].values():
        entry["in_chart_pack"] = True
    return confirmed


def synthetic_flagship_frames() -> dict[str, pd.DataFrame]:
    """SYNTHETIC FIXTURE: invented frames in the four parsers' output
    shapes, spanning the flagship's ranges. All values are round invented
    ramps — none is a real dataset row (#117/ADR-023)."""
    bereiter_ages = [float(a) for a in range(10000, -1, -500)]  # 10000..0 BP
    gml_years = list(range(1959, 2025))
    gistemp_years = list(range(1880, 2025))
    kaufman_ages = [float(a) for a in range(10000, 99, -100)]  # 10000..100 BP
    return {
        "bereiter2015_co2": pd.DataFrame(
            {
                "age_bp": bereiter_ages,
                "co2_ppm": [210.0 + 0.008 * (10000 - a) for a in bereiter_ages],
                "co2_1s_ppm": [1.0] * len(bereiter_ages),
            }
        ),
        "noaa_gml_co2_mlo": pd.DataFrame(
            {
                "year_ce": gml_years,
                "co2_ppm": [300.0 + 1.5 * (y - 1959) for y in gml_years],
                "unc_ppm": [0.1] * len(gml_years),
            }
        ),
        "gistemp_v4": pd.DataFrame(
            {
                "year_ce": gistemp_years,
                "temp_anomaly_c": [-0.2 + 0.01 * (y - 1880) for y in gistemp_years],
            }
        ),
        "kaufman2020_temp12k": pd.DataFrame(
            {
                "age_bp": kaufman_ages,
                "temp_c": [-0.8 + 0.008 * (10000 - a) / 100 for a in kaufman_ages],
                "temp_c_5": [-1.1 + 0.008 * (10000 - a) / 100 for a in kaufman_ages],
                "temp_c_95": [-0.5 + 0.008 * (10000 - a) / 100 for a in kaufman_ages],
            }
        ),
    }


def _flagship_artifact() -> render.ChartArtifact:
    return render.render_chart(
        flagship_spec(), synthetic_flagship_frames(), pack_confirmed_manifest(), SITE_URL
    )


def _png_dimensions(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG stream"
    width, height = struct.unpack(">II", png[16:24])
    return width, height


# ---------------------------------------------------------------------------
# TDD plan item 14: the flagship SVG snapshot (structural, synthetic data)
# ---------------------------------------------------------------------------


def test_flagship_svg_snapshot():
    """Every §3.7 integrity element is present in the flagship SVG text:
    both colour-matched axis titles (the axis-drop regression, spike
    finding 10), splice labels + resolution notes, the #48 CO2 axis
    annotation, the labelled baseline, the #50 alignment disclosure, the
    #47 overlap disclosures, and the manifest-generated caption strip."""
    spec = flagship_spec()
    manifest = pack_confirmed_manifest()
    artifact = _flagship_artifact()
    svg = render.render_svg(artifact.vega_lite)
    assert svg.lstrip().startswith("<svg") or "<svg" in svg[:200]

    # Both series labels render as axis titles — the axis-drop regression.
    for series in spec["series"]:
        assert series["label"] in svg, f"axis title {series['label']!r} dropped from the SVG"
        assert series["color"] in svg, f"series colour {series['color']!r} missing"

    # Splice markers and resolution notes (mandatory, DESIGN §3.7).
    for series in spec["series"]:
        assert series["annotations"]["splice_point"]["label"] in svg
        assert series["annotations"]["resolution_note"] in svg

    # #48: the CO2 axis' non-zero-baseline annotation.
    assert spec["series"][0]["annotations"]["non_zero_baseline"]["label"] in svg

    # Labelled baseline with its display reference.
    assert spec["series"][1]["baseline"]["label"] in svg

    # #50: the alignment disclosure, verbatim from the manifest.
    pairs = {p["id"]: p for p in manifest["splice_pairs"]}
    assert pairs["temp_10k"]["rebaseline"]["disclosure"] in svg

    # #47: the overlap disclosures reach the artefact.
    for pair_id in ("co2_10k", "temp_10k"):
        note = pairs[pair_id]["overlap"]["note"].strip()
        assert note.split("\n")[0].strip()[:40] in svg, (
            f"overlap disclosure for {pair_id} missing from the artefact"
        )

    # Caption strip: manifest attribution, access date, site URL, hash.
    for dataset_id in ("bereiter2015_co2", "noaa_gml_co2_mlo", "gistemp_v4", "kaufman2020_temp12k"):
        attribution = manifest["datasets"][dataset_id]["attribution_text"]
        assert attribution in svg, f"attribution for {dataset_id} missing from the caption"
    assert str(manifest["access_date"]) in svg
    assert SITE_URL in svg
    assert spec_hash(spec)[:12] in svg

    # #137: no LLM-authored identifier strings in the artefact's caption.
    assert spec["chart_id"] not in svg
    assert f"v{spec['spec_version']}" not in svg


def test_flagship_svg_band_and_panels_present():
    """Both panels render (labels in the SVG), and the artefact the SVG
    was produced from carries the Kaufman-member uncertainty band (an
    area layer with a y2 channel and at least two band rows)."""
    spec = flagship_spec()
    artifact = _flagship_artifact()
    svg = render.render_svg(artifact.vega_lite)
    for panel in spec["panels"].values():
        assert panel["label"] in svg, f"panel label {panel['label']!r} missing"

    def _dicts(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from _dicts(value)
        elif isinstance(node, list):
            for value in node:
                yield from _dicts(value)

    band_rows = [
        row
        for node in _dicts(json.loads(artifact.vega_lite_text))
        if isinstance(node.get("encoding"), dict) and "y2" in node["encoding"]
        for row in ((node.get("data") or {}).get("values") or [])
    ]
    assert len(band_rows) >= 2, "the shipped uncertainty band is absent from the artefact"


# ---------------------------------------------------------------------------
# TDD plan item 15: permalink re-render reproducibility
# ---------------------------------------------------------------------------


def test_permalink_rerender_reproducible():
    """Store spec by hash, re-render against pinned data: two fully
    independent renders produce the identical Vega-Lite text and the
    identical SVG, addressed by the same spec hash."""
    first = _flagship_artifact()
    second = _flagship_artifact()
    assert first.spec_hash == second.spec_hash == spec_hash(flagship_spec())
    assert first.vega_lite_text == second.vega_lite_text
    assert render.render_svg(first.vega_lite) == render.render_svg(second.vega_lite)


def test_png_export_dimensions_and_source_hash_stable():
    """PNG determinism, loosely (never byte-equality): the export is a
    real PNG whose dimensions are stable across re-renders and scale
    with the requested factor, and the *source* Vega-Lite text hashes
    identically across renders."""
    first = _flagship_artifact()
    second = _flagship_artifact()
    assert (
        hashlib.sha256(first.vega_lite_text.encode()).hexdigest()
        == hashlib.sha256(second.vega_lite_text.encode()).hexdigest()
    )
    png_1x = render.render_png(first.vega_lite, scale=1.0)
    png_1x_again = render.render_png(second.vega_lite, scale=1.0)
    png_2x = render.render_png(first.vega_lite, scale=2.0)
    dims_1x = _png_dimensions(png_1x)
    assert dims_1x == _png_dimensions(png_1x_again), "re-render changed the PNG dimensions"
    width_2x, height_2x = _png_dimensions(png_2x)
    assert width_2x == pytest.approx(dims_1x[0] * 2, abs=2)
    assert height_2x == pytest.approx(dims_1x[1] * 2, abs=2)


def test_csv_export_matches_svg_attribution():
    """The CSV download carries the same attribution surface as the
    rendered caption: every dataset attribution present in the SVG's
    caption also heads the CSV comments."""
    manifest = pack_confirmed_manifest()
    artifact = _flagship_artifact()
    comment_block = "\n".join(
        line for line in artifact.csv_text.splitlines() if line.startswith("#")
    )
    for dataset_id in ("bereiter2015_co2", "noaa_gml_co2_mlo", "gistemp_v4", "kaufman2020_temp12k"):
        assert manifest["datasets"][dataset_id]["attribution_text"] in comment_block
    assert SITE_URL in comment_block
    assert spec_hash(flagship_spec())[:12] in comment_block
