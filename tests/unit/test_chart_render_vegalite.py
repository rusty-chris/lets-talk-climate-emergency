"""Renderer core: spec + data -> Vega-Lite JSON, structurally pinned (issue #17) — RED.

Failing tests for charts/render.py's pure seam (IMPLEMENTATION.md §1):
``build_vega_lite`` over a :class:`RenderValidatedSpec` and pre-landed
frames. Every §3.7 integrity rule is asserted **in the output** — on the
Vega-Lite JSON structure, never on pixels:

- entry seam: only RenderValidatedSpec renders; the artefact path refuses
  extent-only failures and missing frames (reviews #133/#48; ADR-023);
- marks per chart type; per-panel shared per-axis scales; the #4-spike
  axis-drop regression (identical axis objects across a scale group);
- labelled baselines, per-axis non-zero-baseline annotations (#48), bar
  zero-inclusion, splice rules + labels + resolution notes, overlap
  disclosure (#47), alignment disclosure on the artefact (#50),
  uncertainty bands, CVD-safe palette + direct labelling, BCE labelling,
  visible single-point segments (#49);
- plotted values equal the gold fixtures (1e-9 pass-through / 1e-6
  post-transform).

All inputs are SYNTHETIC FIXTURES (tests/_chart_render_fixtures.py) —
invented data only, per review finding #117.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from charts import render
from charts.spec import ChartSpecError
from tests._chart_render_fixtures import (
    BAR_EXTENTS,
    CO2_OVERLAP_NOTE,
    CO2_RESOLUTION_NOTE,
    DUAL_EXTENTS,
    LINE_EXTENTS,
    PANEL_EXTENTS,
    SITE_URL,
    TEMP_ALIGNMENT_DISCLOSURE,
    TEMP_OVERLAP_NOTE,
    TEMP_RESOLUTION_NOTE,
    TWO_SERIES_EXTENTS,
    bar_spec,
    data_values,
    dual_axis_spec,
    frames,
    iter_dicts,
    layers_with_y,
    line_spec,
    mark_types,
    panel_spec,
    read_gold_csv,
    render_manifest,
    two_series_line_spec,
    validated,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_PATH = REPO_ROOT / "charts" / "render.py"


def _blob(vl) -> str:
    return json.dumps(vl, ensure_ascii=False)


def _build(spec_builder, extents, **kwargs):
    spec = spec_builder()
    return render.build_vega_lite(
        validated(spec, extents), frames(), render_manifest(), SITE_URL, **kwargs
    )


# ---------------------------------------------------------------------------
# Entry seam (review finding #133): only RenderValidatedSpec renders
# ---------------------------------------------------------------------------


def test_build_vega_lite_refuses_a_bare_spec_mapping():
    """The pure renderer accepts ONLY the validate_spec_for_render proof
    token: a raw spec dict — even a perfectly valid one — refuses with
    TypeError naming RenderValidatedSpec, before any chart work."""
    with pytest.raises(TypeError, match="RenderValidatedSpec"):
        render.build_vega_lite(line_spec(), frames(), render_manifest(), SITE_URL)


def test_render_chart_refuses_spec_failing_only_the_extent_check():
    """#133's asked-for artefact-path test: a spec that passes every
    structural/manifest rule but whose scale_domain clips the plotted
    data must be refused by render_chart — proof the artefact path runs
    the render-mode (extents-aware) validation itself."""
    spec = line_spec()
    spec["series"][0]["scale_domain"] = [0.0, 0.5]  # data extent is (-0.30, 0.65)
    # Planner-time validation (no extents) accepts this spec…
    from charts.spec import validate_spec

    assert validate_spec(spec, render_manifest()) is None
    # …so only the render path's own extent computation can refuse it.
    with pytest.raises(ChartSpecError) as excinfo:
        render.render_chart(spec, frames(), render_manifest(), SITE_URL)
    assert any("scale_domain" in v.path for v in excinfo.value.violations)


def test_render_refuses_missing_frame_never_fetches():
    """ADR-023: the renderer consumes pre-landed frames only. A spec
    series whose dataset frame is absent refuses naming the dataset id —
    no silent fetch, no silent empty series."""
    incomplete = frames()
    del incomplete["syn_annual_anomaly"]
    with pytest.raises((render.ChartRenderError, KeyError), match="syn_annual_anomaly"):
        render.render_chart(line_spec(), incomplete, render_manifest(), SITE_URL)


def test_render_module_has_no_fetch_or_network_imports():
    """Structural ADR-023 pin: charts/render.py must not import network
    modules or the fetch shell (charts.datasets) — rendering can never
    grow a silent download path."""
    tree = ast.parse(RENDER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_roots = {"urllib", "requests", "httpx", "socket", "http"}
    assert not {m for m in imported if m.split(".")[0] in forbidden_roots}, (
        f"network imports in charts/render.py: {sorted(imported)}"
    )
    assert "charts.datasets" not in imported, (
        "charts/render.py must not import the fetch shell (ADR-023: render "
        "consumes pre-landed frames only)"
    )


def test_compute_data_extents_covers_every_series():
    """The extents mapping the render-mode validator requires: one
    (min, max) per series id, post-transform. Pinned against the
    hand-computed synthetic extents."""
    extents = render.compute_data_extents(dual_axis_spec(), frames(), render_manifest())
    assert set(extents) == {"co2", "temp"}
    for series_id, (expected_min, expected_max) in DUAL_EXTENTS.items():
        low, high = extents[series_id]
        assert low == pytest.approx(expected_min, abs=1e-6)
        assert high == pytest.approx(expected_max, abs=1e-6)


# ---------------------------------------------------------------------------
# Marks and per-type zero rules
# ---------------------------------------------------------------------------


def test_chart_type_maps_to_mark():
    """line/area/bar chart types produce their marks in the emitted VL."""
    assert "line" in mark_types(_build(line_spec, LINE_EXTENTS))
    from tests._chart_render_fixtures import area_spec

    assert "area" in mark_types(_build(area_spec, LINE_EXTENTS))
    assert "bar" in mark_types(_build(bar_spec, BAR_EXTENTS))


def test_bar_charts_include_zero():
    """§3.7 per-type rule: a bar chart's value scale includes zero even
    when the data floor is far above it (285+ ppm here) — either an
    explicit domain reaching zero or Vega-Lite zero:true on the scale."""
    vl = _build(bar_spec, BAR_EXTENTS)
    bar_scales = [
        node["encoding"]["y"].get("scale") or {}
        for node in layers_with_y(vl)
        if (node.get("mark") == "bar")
        or (isinstance(node.get("mark"), dict) and node["mark"].get("type") == "bar")
    ]
    assert bar_scales, "no bar layer with a y encoding found"
    for scale in bar_scales:
        domain = scale.get("domain")
        zero = scale.get("zero")
        includes_zero = (
            isinstance(domain, list) and len(domain) == 2 and domain[0] <= 0 <= domain[1]
        ) or zero is True
        assert includes_zero, f"bar y-scale must include zero, got scale {scale!r}"


def test_bar_chart_with_zero_excluding_domain_refuses():
    """Truncated bars are categorically misleading — not disclosable by
    annotation. A bar spec whose scale_domain excludes zero must refuse
    at render with a reason naming the rule."""
    spec = bar_spec(scale_domain=[280, 340])  # validates (#48 annotation present)…
    validated_spec = validated(spec, BAR_EXTENTS)
    with pytest.raises(render.ChartRenderError, match="(?i)zero"):
        render.build_vega_lite(validated_spec, frames(), render_manifest(), SITE_URL)


# ---------------------------------------------------------------------------
# Per-axis integrity on the dual axis (#48) and the axis-drop regression
# ---------------------------------------------------------------------------


def test_nonzero_baseline_annotated():
    """Review finding #48: the rule holds for EVERY axis. The CO2 axis
    (domain [250, 340], zero excluded) must render its annotation text
    in the output; the temperature axis (domain [-2, 2], zero included)
    must NOT carry a non-zero-baseline annotation."""
    vl = _build(dual_axis_spec, DUAL_EXTENTS)
    blob = _blob(vl)
    assert "CO2 axis starts at 250 ppm — no zero (invented)" in blob, (
        "the zero-excluding left axis must carry its non-zero-baseline annotation"
    )
    # No spurious annotation for the zero-including axis: the only
    # non-zero-baseline text in the artefact is the CO2 one.
    assert blob.count("no zero (invented)") == 1


def test_dual_axis_labels_colour_matched():
    """§3.7: dual-axis charts get explicit per-axis colour-matched
    labels — each y axis carries its series label as title and the
    series colour as titleColor, one left and one right."""
    vl = _build(dual_axis_spec, DUAL_EXTENTS)
    y_channels = [
        node["encoding"]["y"]
        for node in layers_with_y(vl)
        if isinstance(node["encoding"]["y"], dict)
    ]
    axes = [y["axis"] for y in y_channels if isinstance(y.get("axis"), dict)]
    by_title = {axis.get("title"): axis for axis in axes if axis.get("title")}
    spec = dual_axis_spec()
    for series in spec["series"]:
        assert series["label"] in by_title, f"no axis titled {series['label']!r}"
        axis = by_title[series["label"]]
        colour = series.get("color") or axis.get("titleColor")
        assert axis.get("titleColor") == axis.get("labelColor") == colour or (
            series.get("color") is None
        ), f"axis {series['label']!r} is not colour-matched to its series"
    orients = {axis.get("orient") for axis in by_title.values()}
    assert {"left", "right"} <= orients, "dual axis must render one left and one right axis"


def test_shared_scale_group_never_drops_an_axis():
    """The #4-discovered regression (findings note item 10): Vega-Lite
    silently drops an axis when any layer of a shared-scale group sets
    axis: null. Every layer in a y-scale group must carry the identical
    non-null axis object."""
    for builder, extents in ((dual_axis_spec, DUAL_EXTENTS), (panel_spec, PANEL_EXTENTS)):
        vl = _build(builder, extents)
        groups: dict[str, list] = {}
        for node in layers_with_y(vl):
            y = node["encoding"]["y"]
            if not isinstance(y, dict):
                continue
            scale = y.get("scale") or {}
            domain = scale.get("domain")
            if domain is None:
                continue
            groups.setdefault(json.dumps(domain), []).append(y.get("axis", "MISSING"))
        assert groups, f"{builder.__name__}: no y-scale groups found in the emitted VL"
        for domain, axes in groups.items():
            assert "MISSING" not in axes and None not in axes, (
                f"{builder.__name__}: a layer in scale group {domain} omits/nulls its "
                "axis — Vega-Lite will silently drop the whole axis (spike finding 10)"
            )
            serialised = {json.dumps(axis, sort_keys=True) for axis in axes}
            assert len(serialised) == 1, (
                f"{builder.__name__}: scale group {domain} carries differing axis "
                f"objects — the identical axis object must be set on every layer"
            )


# ---------------------------------------------------------------------------
# Panels: shared per-axis domains, honest ranges, BCE labelling
# ---------------------------------------------------------------------------


def test_panel_pair_shares_per_axis_domains():
    """Amendment 5: the context and recent panels apply each series'
    scale_domain identically — both panels contain y scales with the
    exact spec domains, and x scales equal to each panel's range."""
    vl = _build(panel_spec, PANEL_EXTENTS)
    blob = _blob(vl)
    spec = panel_spec()
    y_domains = [
        node["encoding"]["y"].get("scale", {}).get("domain")
        for node in layers_with_y(vl)
        if isinstance(node["encoding"]["y"], dict)
    ]
    for series in spec["series"]:
        expected = series["scale_domain"]
        assert y_domains.count(expected) >= 2, (
            f"series {series['id']!r} domain {expected} must be applied in both panels"
        )
    for panel in spec["panels"].values():
        assert json.dumps(panel["time_range_ce"]) in blob.replace(" ", ""), (
            f"panel x-domain {panel['time_range_ce']} missing from the emitted VL"
        )
        assert panel["label"] in blob


def test_bce_axis_labelling_for_negative_years():
    """Amendment 8: negative CE years label as BCE — the context panel's
    x axis carries a labelExpr mentioning BCE."""
    vl = _build(panel_spec, PANEL_EXTENTS)
    label_exprs = [
        node.get("labelExpr") for node in iter_dicts(vl) if isinstance(node.get("labelExpr"), str)
    ]
    assert any("BCE" in expr for expr in label_exprs), (
        "no x-axis labelExpr handles BCE years — negative CE years would render "
        "as bare negative numbers"
    )


def test_single_point_segment_renders_visible_mark():
    """Review finding #49: a series segment with exactly one row inside a
    panel's range must produce a visible mark. With recent = [1840,
    2009], the temp paleo segment contributes only the 1850 CE bin —
    some point/circle mark (or line with point overlay) must carry it."""
    spec = panel_spec(recent_range=(1840, 2009))
    vl = render.build_vega_lite(
        validated(spec, PANEL_EXTENTS), frames(), render_manifest(), SITE_URL
    )
    lone_value = 0.0  # temp_c of the 100 BP / 1850 CE bin (invented ramp end)

    def _carries_lone_point(row) -> bool:
        if not isinstance(row, dict) or row.get("year_ce") != 1850:
            return False
        return any(
            value == pytest.approx(lone_value)
            for key, value in row.items()
            if key != "year_ce" and isinstance(value, (int, float))
        )

    visible = False
    for node in iter_dicts(vl):
        mark = node.get("mark")
        mark_type = mark if isinstance(mark, str) else (mark or {}).get("type")
        has_point = isinstance(mark, dict) and bool(mark.get("point"))
        if mark_type in {"point", "circle", "square"} or (mark_type == "line" and has_point):
            rows = (node.get("data") or {}).get("values") or []
            if any(_carries_lone_point(row) for row in rows):
                visible = True
    assert visible, (
        "the lone 1850 paleo point in the recent panel renders no visible mark — "
        "a single-point line segment is invisible in Vega-Lite (review finding #49)"
    )


# ---------------------------------------------------------------------------
# Splice, overlap and alignment disclosures on the artefact (#47/#50)
# ---------------------------------------------------------------------------


def test_splice_annotation_present_in_vegalite_json():
    """DESIGN §3.7: the splice-point rule renders in every panel it
    intersects; the splice label and the manifest resolution note render
    in the artefact (once, context panel — amendment 7)."""
    vl = _build(panel_spec, PANEL_EXTENTS)
    blob = _blob(vl)
    spec = panel_spec()
    for series in spec["series"]:
        assert series["annotations"]["splice_point"]["label"] in blob
    assert CO2_RESOLUTION_NOTE in blob
    assert TEMP_RESOLUTION_NOTE in blob
    # The splice rules themselves: rule marks anchored at the manifest years.
    rule_years = set()
    for node in iter_dicts(vl):
        mark = node.get("mark")
        mark_type = mark if isinstance(mark, str) else (mark or {}).get("type")
        if mark_type == "rule":
            for row in (node.get("data") or {}).get("values") or []:
                if isinstance(row, dict):
                    rule_years.update(v for v in row.values() if isinstance(v, (int, float)))
    assert 1850 in rule_years, "no rule mark at the CO2 splice year 1850"
    assert 1880 in rule_years, "no rule mark at the temperature splice year 1880"


def test_splice_overlap_disclosed_in_caption_or_annotation():
    """Review finding #47: where the manifest records discarded overlap,
    the artefact carries the disclosure — the pair's overlap note must
    appear in the emitted VL (caption line or annotation), for both
    spliced series."""
    vl = _build(panel_spec, PANEL_EXTENTS)
    blob = _blob(vl)
    assert CO2_OVERLAP_NOTE in blob, "CO2 overlap disclosure missing from the artefact"
    assert TEMP_OVERLAP_NOTE in blob, "temperature overlap disclosure missing from the artefact"


def test_caption_discloses_rebaseline_alignment():
    """Review finding #50: the alignment method reaches the artefact
    itself — the manifest rebaseline.disclosure string appears verbatim
    in the emitted VL."""
    vl = _build(dual_axis_spec, DUAL_EXTENTS)
    assert TEMP_ALIGNMENT_DISCLOSURE in _blob(vl), (
        "the rebaseline alignment disclosure lives only in repo files — it must "
        "be rendered on the artefact (review finding #50)"
    )


def test_labelled_baseline_rendered_with_reference_period():
    """§3.7: baselines and reference periods always labelled — the zero
    baseline renders as a rule plus its display-reference label text."""
    vl = _build(dual_axis_spec, DUAL_EXTENTS)
    blob = _blob(vl)
    assert "0 = 1400-1500 average (invented)" in blob
    assert "rule" in mark_types(vl)


# ---------------------------------------------------------------------------
# Uncertainty bands (§3.7: rendered when the dataset ships them)
# ---------------------------------------------------------------------------


def test_uncertainty_band_rendered_when_dataset_ships_bounds():
    """The temp series' paleo member ships a 5-95% band; the emitted VL
    must contain an area mark spanning the band columns (y + y2), with
    the band's values present."""
    vl = _build(dual_axis_spec, DUAL_EXTENTS)
    band_layers = [
        node
        for node in iter_dicts(vl)
        if isinstance(node.get("encoding"), dict)
        and "y2" in node["encoding"]
        and (
            node.get("mark") == "area"
            or (isinstance(node.get("mark"), dict) and node["mark"].get("type") == "area")
        )
    ]
    assert band_layers, "no area band layer (y + y2) in the emitted VL"
    band_rows = [
        row
        for node in band_layers
        for row in ((node.get("data") or {}).get("values") or [])
        if isinstance(row, dict)
    ]
    assert any(row.get("temp_c_5") == pytest.approx(-1.1) for row in band_rows), (
        "band lower bound values missing (oldest invented bin: -0.8 - 0.3)"
    )


def test_no_uncertainty_band_when_not_shipped():
    """No invented certainty theatre: a spec without an uncertainty_band
    produces no y2 area layer."""
    vl = _build(line_spec, LINE_EXTENTS)
    assert not [
        node
        for node in iter_dicts(vl)
        if isinstance(node.get("encoding"), dict) and "y2" in node["encoding"]
    ], "a band rendered for a series that ships no bounds"


# ---------------------------------------------------------------------------
# Palette and direct labelling
# ---------------------------------------------------------------------------


def test_palette_constants_are_colour_blind_safe_pair_first():
    """The committed CVD-checked pair (spike findings) leads the palette;
    all entries are distinct six-digit hex colours."""
    assert render.PALETTE[0] == "#2a78d6"
    assert render.PALETTE[1] == "#e34948"
    assert len(set(render.PALETTE)) == len(render.PALETTE)
    for colour in render.PALETTE:
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", colour), colour


def test_series_colours_come_from_palette_and_are_directly_labelled():
    """Series without a spec colour draw distinct palette colours, and
    the chart uses direct labelling: every series label appears in the
    output and no Vega-Lite legend is configured."""
    vl = _build(two_series_line_spec, TWO_SERIES_EXTENTS)
    blob = _blob(vl)
    spec = two_series_line_spec()
    used_colours = set()
    for node in iter_dicts(vl):
        mark = node.get("mark")
        if isinstance(mark, dict) and isinstance(mark.get("color"), str):
            used_colours.add(mark["color"])
        encoding = node.get("encoding")
        if isinstance(encoding, dict):
            colour = encoding.get("color")
            if isinstance(colour, dict) and isinstance(colour.get("value"), str):
                used_colours.add(colour["value"])
            # Direct labelling: any colour encoding must disable the legend.
            if isinstance(colour, dict) and "field" in colour:
                assert colour.get("legend") is None, (
                    "series are identified by direct labels, never a legend"
                )
    series_colours = used_colours & set(render.PALETTE)
    assert len(series_colours) >= 2, (
        f"two unspecced series must draw two distinct palette colours, got {used_colours}"
    )
    for series in spec["series"]:
        assert series["label"] in blob, f"series {series['label']!r} is not directly labelled"


# ---------------------------------------------------------------------------
# Plotted values vs gold fixtures (tolerances per IMPLEMENTATION.md §5)
# ---------------------------------------------------------------------------


def test_rendered_values_match_gold_fixtures():
    """Pass-through: a plain line chart's inline data equals the source
    CSV at 1e-9. Post-transform: with a 5-year rolling mean, the inline
    data equals the independent gold fixture at 1e-6."""
    # Pass-through (1e-9): the committed synthetic ramp, untransformed.
    vl = _build(line_spec, LINE_EXTENTS)
    rows = {
        row["year_ce"]: row
        for row in data_values(vl)
        if isinstance(row.get("year_ce"), (int, float)) and "anomaly_c" in row
    }
    assert rows, "no plotted anomaly rows in the emitted VL"
    assert rows[1990]["anomaly_c"] == pytest.approx(-0.30, abs=1e-9)
    assert rows[2009]["anomaly_c"] == pytest.approx(0.65, abs=1e-9)
    assert len(rows) == 20

    # Post-transform (1e-6): rolling mean vs the independent gold values.
    spec = line_spec()
    spec["series"][0]["transforms"] = [{"op": "rolling_mean", "window_years": 5}]
    gold = read_gold_csv("gold_rolling_mean.csv")
    extents = {
        "anom": (float(gold["anomaly_c"].min()), float(gold["anomaly_c"].max())),
    }
    vl = render.build_vega_lite(validated(spec, extents), frames(), render_manifest(), SITE_URL)
    plotted = {
        row["year_ce"]: row["anomaly_c"]
        for row in data_values(vl)
        if isinstance(row.get("year_ce"), (int, float)) and "anomaly_c" in row
    }
    assert len(plotted) == len(gold)
    for _, gold_row in gold.iterrows():
        year = float(gold_row["year_ce"])
        assert year in plotted, f"gold year {year} missing from the plotted data"
        assert plotted[year] == pytest.approx(float(gold_row["anomaly_c"]), abs=1e-6), (
            f"plotted value at {year} deviates from the independent gold value"
        )
