"""Caption strip, alt text and CSV export (issue #17) — RED.

Failing tests pinning the attribution surfaces of every artefact
(DESIGN §3.7: attribution is part of the artefact — every artefact):

- the caption strip is composed exclusively of manifest attribution +
  licence strings, the manifest access date, deployment config (site
  URL) and the spec hash — never spec free text (review finding #137,
  vocabulary amendment 9);
- the responsive rule: below charts.render.CAPTION_MIN_FULL_WIDTH_PX the
  strip drops to source names + URL (the 360 px embed case);
- CSV downloads carry the same attribution as leading # header comments;
- alt text is generated deterministically from the validated spec +
  frames (title, series, ranges, closed trend vocabulary) — never an
  LLM call.

All inputs are SYNTHETIC FIXTURES (tests/_chart_render_fixtures.py),
per review finding #117.
"""

from __future__ import annotations

import csv as csv_module
import io

import pytest

from charts import alt_text as alt_text_module
from charts import render
from charts.spec import spec_hash
from tests._chart_render_fixtures import (
    ACCESS_DATE,
    DUAL_EXTENTS,
    LINE_EXTENTS,
    SITE_URL,
    TEMP_ALIGNMENT_DISCLOSURE,
    dual_axis_spec,
    frames,
    line_spec,
    render_manifest,
    validated,
)

#: The attribution and licence strings of the datasets dual_axis_spec plots.
DUAL_ATTRIBUTIONS = (
    "Aurelian Ice-Core Consortium (fictional)",
    "Windvane Observatory (fictional)",
    "Basin Reconstruction Team (fictional)",
    "Meridian Climate Service (fictional)",
)
DUAL_LICENCES = (
    "Open with required credit (invented)",
    "Public domain (invented stand-in)",
)


def _caption(spec_builder, extents, **kwargs) -> list[str]:
    return render.caption_lines(
        validated(spec_builder(), extents), render_manifest(), SITE_URL, **kwargs
    )


# ---------------------------------------------------------------------------
# Caption content (issue TDD plan item 9; review finding #137)
# ---------------------------------------------------------------------------


def test_caption_strip_contains_sources_licences_date_url():
    """The full-width caption names every plotted dataset's manifest
    attribution and licence, the manifest access date, the deployment
    site URL and the spec hash (the permalink identity)."""
    joined = "\n".join(_caption(dual_axis_spec, DUAL_EXTENTS))
    for attribution in DUAL_ATTRIBUTIONS:
        assert attribution in joined, f"attribution {attribution!r} missing from caption"
    for licence in DUAL_LICENCES:
        assert licence in joined, f"licence {licence!r} missing from caption"
    assert ACCESS_DATE in joined
    assert SITE_URL in joined
    assert spec_hash(dual_axis_spec())[:12] in joined


def test_caption_strip_contains_no_spec_strings():
    """Review finding #137 / amendment 9: nothing LLM-authored reaches
    the caption. A spec with authoritative-looking free text renders a
    caption containing none of it — not the chart_id, not the title or
    subtitle, not a series label, not the spec_version."""
    spec = dual_axis_spec()
    spec["chart_id"] = "official-noaa-verified-chart"
    spec["title"] = "OFFICIAL VERIFIED CLIMATE RECORD"
    spec["subtitle"] = "certified by the spec author"
    spec["series"][0]["label"] = "Injected caption text (ppm)"
    lines = render.caption_lines(validated(spec, DUAL_EXTENTS), render_manifest(), SITE_URL)
    joined = "\n".join(lines)
    for spec_text in (
        spec["chart_id"],
        spec["title"],
        spec["subtitle"],
        spec["series"][0]["label"],
        f"v{spec['spec_version']}",
    ):
        assert spec_text not in joined, f"spec free text {spec_text!r} leaked into the caption"
    assert spec_hash(spec)[:12] in joined, (
        "the artefact is identified by its spec hash, not by chart_id/spec_version"
    )


def test_caption_includes_integrity_disclosures():
    """The #47 overlap and #50 alignment disclosures ride the caption
    surface (or equivalent artefact text) — here pinned on the caption
    generator itself so the strip is self-contained in exports."""
    joined = "\n".join(_caption(dual_axis_spec, DUAL_EXTENTS))
    assert TEMP_ALIGNMENT_DISCLOSURE in joined


# ---------------------------------------------------------------------------
# Responsive rule (issue acceptance criterion; TDD plan item 10)
# ---------------------------------------------------------------------------


def test_caption_min_width_constant_sane():
    """The responsive threshold sits between the 360 px embed case and
    the default render width, so both rule branches are reachable."""
    assert 360 < render.CAPTION_MIN_FULL_WIDTH_PX <= render.DEFAULT_WIDTH_PX


def test_caption_strip_below_min_width_drops_to_short_form():
    """At 360 px the strip drops to source names + site URL (DESIGN
    §3.7): attributions and the URL survive; the licence strings do not;
    the short form is strictly shorter than the full form."""
    full = _caption(dual_axis_spec, DUAL_EXTENTS, width_px=render.DEFAULT_WIDTH_PX)
    short = _caption(dual_axis_spec, DUAL_EXTENTS, width_px=360)
    joined_short = "\n".join(short)
    for attribution in DUAL_ATTRIBUTIONS:
        assert attribution in joined_short, f"short caption dropped source {attribution!r}"
    assert SITE_URL in joined_short
    for licence in DUAL_LICENCES:
        assert licence not in joined_short, (
            "the short-form caption must drop licence text to stay legible at 360 px"
        )
    assert len(joined_short) < len("\n".join(full))


# ---------------------------------------------------------------------------
# CSV export (TDD plan item 11)
# ---------------------------------------------------------------------------


def test_csv_export_carries_attribution_header_comments():
    """CSV downloads carry the caption's attribution as leading '#'
    comment lines, then the plotted data (which parses and matches the
    source at 1e-9)."""
    text = render.csv_export(
        validated(line_spec(), LINE_EXTENTS), frames(), render_manifest(), SITE_URL
    )
    lines = text.splitlines()
    comment_lines = []
    for line in lines:
        if not line.startswith("#"):
            break
        comment_lines.append(line)
    assert comment_lines, "no leading # attribution comments in the CSV export"
    joined_comments = "\n".join(comment_lines)
    assert "Meridian Climate Service (fictional)" in joined_comments
    assert ACCESS_DATE in joined_comments
    assert SITE_URL in joined_comments
    assert spec_hash(line_spec())[:12] in joined_comments

    data_part = "\n".join(lines[len(comment_lines) :])
    rows = list(csv_module.reader(io.StringIO(data_part)))
    header, data = rows[0], [r for r in rows[1:] if r]
    assert "year_ce" in header
    assert len(data) == 20, "every plotted row exports"
    year_index = header.index("year_ce")
    value_index = next(i for i in range(len(header)) if i != year_index)
    by_year = {float(r[year_index]): float(r[value_index]) for r in data}
    assert by_year[1990.0] == pytest.approx(-0.30, abs=1e-9)
    assert by_year[2009.0] == pytest.approx(0.65, abs=1e-9)


# ---------------------------------------------------------------------------
# Alt text (TDD plan item 12)
# ---------------------------------------------------------------------------


def test_alt_text_deterministic_from_spec():
    """Alt text is derived deterministically from the validated spec and
    frames: title, series labels, the plotted range's endpoint years and
    a closed-vocabulary trend word all present; two independent calls
    produce byte-identical text."""
    spec = line_spec()
    first = alt_text_module.alt_text(validated(spec, LINE_EXTENTS), frames())
    second = alt_text_module.alt_text(validated(line_spec(), LINE_EXTENTS), frames())
    assert first == second, "same spec + frames must produce identical alt text"
    assert spec["title"] in first
    assert spec["series"][0]["label"] in first
    assert "1990" in first and "2009" in first
    assert any(word in first for word in alt_text_module.TREND_WORDS)


def test_alt_text_trend_direction_tracks_the_data():
    """The trend word is computed from the plotted data: the rising
    synthetic ramp reads 'rising'; the same series negated reads
    'falling'."""
    rising = alt_text_module.alt_text(validated(line_spec(), LINE_EXTENTS), frames())
    assert "rising" in rising and "falling" not in rising

    falling_frames = frames()
    negated = falling_frames["syn_annual_anomaly"].copy()
    negated["anomaly_c"] = -negated["anomaly_c"]
    falling_frames["syn_annual_anomaly"] = negated
    falling_extents = {"anom": (-0.65, 0.30)}
    falling = alt_text_module.alt_text(validated(line_spec(), falling_extents), falling_frames)
    assert "falling" in falling and "rising" not in falling


def test_alt_text_module_makes_no_llm_or_network_calls():
    """Structural pin: charts/alt_text.py imports no adapter, network or
    LLM modules — accessibility for free, from data the renderer already
    has (DESIGN §3.7)."""
    import ast
    from pathlib import Path

    path = Path(alt_text_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"rag", "anthropic", "urllib", "requests", "httpx", "http", "socket"}
    assert not (imported & forbidden), sorted(imported & forbidden)
