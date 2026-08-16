"""Canary for #15/#17: the committed flagship ChartSpec still parses and
renders with its integrity annotations (issues/04.md TDD plan).

Runs the full spike pipeline — parsers -> transforms -> Vega-Lite ->
vl-convert SVG — over the synthetic fixtures (renamed to the raw filenames
the pipeline expects), then asserts on the SVG text: splice annotations,
resolution notes, labelled baseline, colour-matched dual-axis titles and the
caption strip are all present in the export. Not a correctness proof (the
hand spot-checks in reviews/spike-04-chart-findings.md are that); a canary
that schema/renderer changes cannot silently drop an integrity annotation.

Integration tier: exercises the real vl-convert renderer (IMPLEMENTATION 3).
"""

import shutil
from pathlib import Path

import pytest

from charts.spike.render import RAW_FILES, load_spec, render_flagship

FIXTURES = Path(__file__).parents[1] / "fixtures" / "spike"

FIXTURE_FOR_DATASET = {
    "bereiter2015_co2": "bereiter_synthetic.txt",
    "kaufman2020_temp12k": "kaufman_synthetic.csv",
    "noaa_gml_co2_mlo": "gml_synthetic.csv",
    "gistemp_v4": "gistemp_synthetic.csv",
}


@pytest.mark.integration
def test_spike_flagship_spec_parses_and_renders(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for dataset_id, (_, raw_name) in RAW_FILES.items():
        shutil.copy(FIXTURES / FIXTURE_FOR_DATASET[dataset_id], raw_dir / raw_name)

    out = render_flagship(raw_dir=raw_dir, out_dir=tmp_path / "out")
    svg = out["svg"].read_text(encoding="utf-8")
    spec = load_spec()

    # Splice-point + resolution annotations, rendered (mandatory, DESIGN 3.7).
    for series in spec["series"]:
        assert series["annotations"]["splice_point"]["label"] in svg
        assert series["annotations"]["resolution_note"].lower() in svg.lower()

    # Labelled baseline with its reference period.
    assert "0 °C = 1800–1900 average" in svg

    # Colour-matched dual-axis titles for both series.
    for series in spec["series"]:
        assert series["label"] in svg
        assert series["color"] in svg

    # Caption strip baked into the export: sources, access date, site URL.
    assert spec["caption"]["access_date"] in svg
    assert spec["caption"]["site_url"] in svg
    assert "Bereiter" in svg and "Kaufman" in svg and "GISTEMP" in svg

    # Both export formats were produced.
    assert out["png"].stat().st_size > 0
    assert out["vl"].stat().st_size > 0
