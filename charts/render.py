"""Chart renderer: RenderValidatedSpec + pre-landed frames → Vega-Lite → exports (issue #17).

Productionises the #4 spike renderer (``charts/spike/render.py``) under
the split seam IMPLEMENTATION.md §1 pins:

- ``spec → Vega-Lite JSON`` is **pure** (:func:`build_vega_lite`) — every
  integrity rule, annotation, caption line and palette decision is
  assertable on the emitted JSON, and byte-identical for identical inputs
  (the permalink re-render guarantee, DESIGN §3.7);
- the ``vl-convert`` invocation is the thin side-effect edge
  (:func:`render_svg` / :func:`render_png`, integration tier only).

The artefact entry points consume only
:class:`charts.spec.RenderValidatedSpec` — the proof token minted by
:func:`charts.spec.validate_spec_for_render` — never a bare spec mapping
(review finding #133): a raw dict refuses with :class:`TypeError` before
any pixel work. :func:`render_chart` is the one convenience path that
accepts a raw spec, because it *runs* the render-mode validation itself
(extents computed from the frames), so a spec that fails only the
extents-aware checks (#48/#129) is refused here.

Data reaches the renderer exclusively as **pre-landed frames** keyed by
dataset id — the output of :func:`charts.pack.load_dataset_frame` over
origin-fetched, hash-verified files (ADR-023). The renderer performs no
fetching of any kind: a missing frame is a refusal naming the dataset,
never a download. The value column of each frame is the manifest entry's
``variable.name``; captions are composed exclusively from manifest
attribution/licence strings, the manifest access date, deployment config
(``site_url``) and the spec hash — never spec free text (review finding
#137, vocabulary amendment 9).

Chart-integrity rules owned here (DESIGN §3.7 as amended; pinned by tests)
--------------------------------------------------------------------------

- labelled zero baselines with their display reference period; per-axis
  non-zero-baseline annotations rendered for **every** axis whose domain
  excludes zero, including dual-axis charts (review finding #48);
- bar charts always include zero — a bar spec whose domain excludes zero
  refuses (:class:`ChartRenderError`), annotation or not: truncated bars
  are categorically misleading, not disclosable;
- splice-point rules render in every panel they intersect; splice labels
  and resolution notes render once (context panel, amendment 7); the
  manifest pair's overlap disclosure reaches the artefact (review #47)
  and the rebaseline alignment disclosure reaches the artefact (review
  #50);
- uncertainty bands render when the source frame ships the band columns;
- shared-scale layer groups carry the identical axis object on every
  layer — Vega-Lite silently drops an axis if any layer of a scale group
  sets ``axis: null`` (the #4 spike hit this; findings note item 10);
- single-row segments inside a panel render a visible point mark rather
  than an invisible one-point line (review finding #49);
- colour-blind-safe palette (:data:`PALETTE`) and direct labelling — no
  Vega-Lite legends; series are named by colour-matched axis titles
  (dual-axis) or direct text labels;
- BCE-aware x-axis labelling for negative CE years (amendment 8).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from charts import transforms
from charts.spec import (
    RenderValidatedSpec,
    spec_hash,
    validate_spec_for_render,
)
from charts.transforms import bp_to_ce

#: Colour-blind-safe categorical palette, in assignment order. The first
#: two entries are the CVD-checked blue/red pair the #4 spike validated
#: for the flagship (reviews/spike-04-chart-findings.md) and the committed
#: flagship spec pins; series without a spec ``color`` draw from this
#: tuple in order. Changing these values is a design change, not a
#: refactor.
PALETTE: tuple[str, ...] = ("#2a78d6", "#e34948", "#0f8a6f", "#8a5cd6", "#b8860b")

#: Default artefact width and the responsive caption threshold (DESIGN
#: §3.7): at render widths below :data:`CAPTION_MIN_FULL_WIDTH_PX` the
#: caption strip drops to source names + site URL so it stays legible in
#: a 360 px embed.
DEFAULT_WIDTH_PX = 860
CAPTION_MIN_FULL_WIDTH_PX = 480


class ChartRenderError(ValueError):
    """A refused render: integrity rule violation or missing pre-landed frame."""


@dataclass(frozen=True)
class ChartArtifact:
    """One rendered chart, addressed by its spec hash (the permalink identity).

    ``vega_lite`` is the pure-function output; ``vega_lite_text`` its
    canonical serialisation (byte-identical for identical spec + frames —
    the ``/chart/<spec_hash>`` re-render determinism contract). SVG/PNG
    bytes are produced separately at the vl-convert edge and are *not*
    part of this pure bundle.
    """

    spec_hash: str
    vega_lite: Mapping[str, Any]
    vega_lite_text: str
    alt_text: str
    csv_text: str


#: A neutral ink used for baselines, annotations and the caption strip.
_INK = "#52514e"
#: A dashed neutral used for the labelled zero baseline rule.
_BASELINE_INK = "#52514e"

_X_LABEL_EXPR = "datum.value < 0 ? format(-datum.value, 'd') + ' BCE' : format(datum.value, 'd')"


# ---------------------------------------------------------------------------
# Manifest access helpers (pure reads; no fetching — ADR-023)
# ---------------------------------------------------------------------------


def _datasets(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest.get("datasets") or {}


def _pairs(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {p.get("id"): p for p in (manifest.get("splice_pairs") or []) if isinstance(p, Mapping)}


def _series_dataset_ids(series: Mapping[str, Any]) -> list[str]:
    """The dataset ids a series plots, in caption/source order."""
    if series.get("splice_series"):
        return list(series["splice_series"])
    return [series["dataset"]]


def _clean(value: Any) -> Any:
    """A JSON-native scalar for an inline datum: strings pass through,
    integer-valued numbers collapse to ``int``, everything else to
    ``float`` (numpy scalars from Pandas would otherwise be unserialisable
    and non-deterministic)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    number = float(value)
    return int(number) if number.is_integer() else number


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Inline ``data.values`` rows, JSON-native and order-stable."""
    return [{key: _clean(val) for key, val in row.items()} for row in frame.to_dict("records")]


def _clip(frame: pd.DataFrame, x0: float, x1: float) -> pd.DataFrame:
    within = (frame["year_ce"] >= x0) & (frame["year_ce"] <= x1)
    return frame[within].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The per-series data pipeline (shared by extents and the VL builder)
# ---------------------------------------------------------------------------


def _prep_member(
    dataset_id: str, frames: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, str]:
    """One dataset's frame in year_ce shape plus its value column name.

    BP datasets are converted to CE here (the manifest ``time_axis.unit``
    decides). Raises :class:`ChartRenderError` naming a dataset whose
    pre-landed frame is absent — the renderer never fetches (ADR-023)."""
    if dataset_id not in frames:
        raise ChartRenderError(
            f"no pre-landed frame for dataset {dataset_id!r}: the renderer consumes "
            "origin-fetched, hash-verified frames only and never fetches (ADR-023)"
        )
    entry = _datasets(manifest).get(dataset_id) or {}
    value_col = entry.get("variable", {}).get("name")
    frame = frames[dataset_id].copy()
    if entry.get("time_axis", {}).get("unit") == "years_bp":
        frame = bp_to_ce(frame)
    return frame, value_col


def _apply_transform(
    frame: pd.DataFrame, value_col: str, from_unit: str, transform: Mapping[str, Any]
) -> pd.DataFrame:
    op = transform["op"]
    if op == "rolling_mean":
        return transforms.rolling_mean(frame, value_col, transform["window_years"])
    if op == "resample":
        return transforms.resample(frame, value_col, transform["window_years"])
    if op == "anomaly_vs_baseline":
        return transforms.anomaly_vs_baseline(frame, value_col)
    if op == "unit_conversion":
        return transforms.unit_conversion(frame, value_col, from_unit, transform["to_unit"])
    raise ChartRenderError(f"unknown transform op {op!r}")


def _series_frame(
    series: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    *,
    cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> tuple[pd.DataFrame, str]:
    """The post-transform plotted frame for one series, and its value
    column. Single-dataset series keep the dataset's own value-column
    name (so pass-through data is addressable by it); spliced series use a
    canonical ``value`` column carrying a ``segment`` label.

    ``cache`` (finding #297) memoises the result per series id within ONE
    ``render_chart`` — extents, the VL builder, alt text and CSV export all
    consume the SAME post-transform frame, so the per-series pipeline (a
    frame copy, BP→CE, and every transform incl. the O(rows²) rolling mean)
    runs once per artifact instead of four times. No consumer mutates the
    returned frame in place (each derives clipped copies), so sharing it is
    byte-identical. ``None`` (every direct caller/test) disables caching."""
    if cache is not None and (cached := cache.get(series["id"])) is not None:
        return cached
    result = _compute_series_frame(series, frames, manifest)
    if cache is not None:
        cache[series["id"]] = result
    return result


def _compute_series_frame(
    series: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    """The uncached per-series transform pipeline (see :func:`_series_frame`)."""
    if not series.get("splice_series"):
        dataset_id = series["dataset"]
        frame, value_col = _prep_member(dataset_id, frames, manifest)
        entry = _datasets(manifest).get(dataset_id) or {}
        from_unit = entry.get("variable", {}).get("unit")
        for transform in series.get("transforms") or []:
            frame = _apply_transform(frame, value_col, from_unit, transform)
        return frame.reset_index(drop=True), value_col

    pair = _pairs(manifest)[series["splice_pair_id"]]
    paleo_id, instrumental_id = series["splice_series"]
    paleo, paleo_col = _prep_member(paleo_id, frames, manifest)
    instrumental, instrumental_col = _prep_member(instrumental_id, frames, manifest)
    paleo = paleo.rename(columns={paleo_col: "value"})
    instrumental = instrumental.rename(columns={instrumental_col: "value"})

    rebaseline = pair.get("rebaseline")
    if rebaseline:
        period = tuple(rebaseline["alignment_period_ce"])
        target = rebaseline["apply_to"]
        if target == instrumental_id:
            instrumental = transforms.rebaseline(instrumental, "value", period)
        elif target == paleo_id:
            paleo = transforms.rebaseline(paleo, "value", period)
    spliced = transforms.splice_series(
        paleo, instrumental, pair["splice_year_ce"], series["overlap_policy"]
    )
    return spliced, "value"


def _band_frame(
    series: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> pd.DataFrame | None:
    """The uncertainty-band frame (lower/upper columns) for a series that
    ships one, restricted to the paleo member's pre-splice span; ``None``
    when the series ships no band (no invented certainty theatre)."""
    band = series.get("uncertainty_band")
    if not band:
        return None
    frame, _ = _prep_member(band["source"], frames, manifest)
    pair = _pairs(manifest).get(series.get("splice_pair_id"))
    if pair is not None:
        frame = frame[frame["year_ce"] < pair["splice_year_ce"]].reset_index(drop=True)
    return frame[["year_ce", band["lower"], band["upper"]]]


def _plot_range(spec: Mapping[str, Any]) -> tuple[float, float]:
    """The widest plotted x-range: the context panel for a panel pair, else
    the spec's ``time_range_ce``."""
    if spec.get("panels"):
        start, end = spec["panels"]["context"]["time_range_ce"]
        return start, end
    start, end = spec["time_range_ce"]
    return start, end


def compute_data_extents(
    spec: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    *,
    series_cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Post-transform (min, max) per series id, within the spec's plotted
    range — the mapping :func:`charts.spec.validate_spec_for_render`
    requires (review finding #133). Pure; raises
    :class:`ChartRenderError` naming any dataset whose frame is absent
    (pre-landed frames only, never a fetch — ADR-023). ``series_cache``
    (finding #297) shares the per-series pipeline across one render."""
    x0, x1 = _plot_range(spec)
    extents: dict[str, tuple[float, float]] = {}
    for series in spec["series"]:
        frame, value_col = _series_frame(series, frames, manifest, cache=series_cache)
        clipped = _clip(frame, x0, x1)
        extents[series["id"]] = (float(clipped[value_col].min()), float(clipped[value_col].max()))
    return extents


# ---------------------------------------------------------------------------
# Vega-Lite assembly (pure; every integrity rule assertable on the output)
# ---------------------------------------------------------------------------


def _series_colour(series: Mapping[str, Any], index: int) -> str:
    """The series' display colour: its explicit spec ``color`` or the next
    CVD-safe palette entry in assignment order."""
    return series.get("color") or PALETTE[index % len(PALETTE)]


def _domain_excludes_zero(domain: Any) -> bool:
    return isinstance(domain, list) and len(domain) == 2 and not (domain[0] <= 0 <= domain[1])


def _y_axis(series: Mapping[str, Any], colour: str) -> dict[str, Any]:
    """A colour-matched per-axis title object (the same dict content is
    reused on every layer sharing this series' scale — the #4 axis-drop
    regression needs identical axis objects across a shared-scale group)."""
    return {
        "title": series["label"],
        "titleColor": colour,
        "labelColor": colour,
        "orient": series.get("axis", "left"),
        "grid": False,
    }


def _x_encoding(x0: float, x1: float, label: str) -> dict[str, Any]:
    return {
        "field": "year_ce",
        "type": "quantitative",
        "scale": {"domain": [x0, x1], "nice": False},
        "axis": {"title": label, "labelExpr": _X_LABEL_EXPR, "grid": False},
    }


def _dual_panel(
    spec: Mapping[str, Any],
    prepared: list[dict[str, Any]],
    x0: float,
    x1: float,
    label: str,
    width: int,
    height: int,
    show_annotations: bool,
) -> dict[str, Any]:
    """One dual-axis panel: colour-matched per-series y axes, splice rules
    (every panel they intersect), and — once, where ``show_annotations`` —
    the splice labels, resolution notes, non-zero-baseline and labelled
    baseline annotations."""
    x_enc = _x_encoding(x0, x1, label)
    layer_groups: list[dict[str, Any]] = []
    annotation_layers: list[dict[str, Any]] = []

    for order, item in enumerate(prepared):
        series = item["series"]
        colour = item["colour"]
        value_col = item["value_col"]
        domain = series["scale_domain"]
        axis = _y_axis(series, colour)
        y_enc = {
            "field": value_col,
            "type": "quantitative",
            "scale": {"domain": domain, "nice": False},
            "axis": axis,
        }
        series_layers: list[dict[str, Any]] = [
            {
                "data": {"values": _records(_clip(item["frame"], x0, x1))},
                "mark": {"type": "line", "strokeWidth": 2, "color": colour, "point": True},
                "encoding": {"x": x_enc, "y": y_enc, "detail": {"field": "segment"}},
            }
        ]

        band = item["band"]
        if band is not None:
            clipped_band = _clip(band, x0, x1)
            if not clipped_band.empty:
                series_layers.append(
                    {
                        "data": {"values": _records(clipped_band)},
                        "mark": {"type": "area", "color": colour, "opacity": 0.15},
                        "encoding": {
                            "x": x_enc,
                            "y": {
                                "field": series["uncertainty_band"]["lower"],
                                "type": "quantitative",
                                "scale": {"domain": domain, "nice": False},
                                "axis": axis,
                            },
                            "y2": {"field": series["uncertainty_band"]["upper"]},
                        },
                    }
                )

        baseline = series.get("baseline")
        if baseline is not None:
            series_layers.append(
                {
                    "data": {"values": [{"y": _clean(baseline["value"])}]},
                    "mark": {
                        "type": "rule",
                        "strokeDash": [4, 3],
                        "color": _BASELINE_INK,
                        "opacity": 0.8,
                    },
                    "encoding": {
                        "y": {
                            "field": "y",
                            "type": "quantitative",
                            "scale": {"domain": domain, "nice": False},
                            "axis": axis,
                        }
                    },
                }
            )
            if show_annotations:
                annotation_layers.append(_text_datum(baseline["label"], 6, 10 + 30 * order, colour))

        annotations = series.get("annotations") or {}
        splice_point = annotations.get("splice_point")
        if splice_point is not None:
            splice_year = splice_point["year_ce"]
            if x0 <= splice_year <= x1:
                series_layers.append(
                    {
                        "data": {"values": [{"x": _clean(splice_year)}]},
                        "mark": {
                            "type": "rule",
                            "strokeDash": [2, 2],
                            "color": colour,
                            "opacity": 0.7,
                        },
                        "encoding": {"x": {**x_enc, "field": "x"}},
                    }
                )
            if show_annotations:
                note = annotations.get("resolution_note", "")
                annotation_layers.append(
                    _text_datum(
                        f"⇢ {splice_point['label']} — {note}",
                        6,
                        24 + 30 * order,
                        colour,
                    )
                )

        non_zero = annotations.get("non_zero_baseline")
        if show_annotations and non_zero is not None and _domain_excludes_zero(domain):
            annotation_layers.append(_text_datum(non_zero["label"], 6, 38 + 30 * order, colour))

        layer_groups.append({"layer": series_layers})

    return {
        "width": width,
        "height": height,
        "layer": layer_groups + annotation_layers,
        "resolve": {"scale": {"y": "independent"}},
    }


def _single_axis_panel(
    spec: Mapping[str, Any],
    prepared: list[dict[str, Any]],
    x0: float,
    x1: float,
    label: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """A single-y-axis chart (line / area / bar): direct-labelled series in
    the CVD-safe palette, per-type y-scale rules (bars include zero), and
    labelled baselines/annotations as needed."""
    chart_type = spec["chart_type"]
    mark_type = {"line": "line", "area": "area", "bar": "bar"}[chart_type]
    x_enc = _x_encoding(x0, x1, label)
    layers: list[dict[str, Any]] = []

    for order, item in enumerate(prepared):
        series = item["series"]
        colour = item["colour"]
        value_col = item["value_col"]
        domain = series.get("scale_domain")
        scale: dict[str, Any] = {"nice": False}
        if domain is not None:
            scale["domain"] = domain
        if chart_type == "bar":
            if domain is not None and _domain_excludes_zero(domain):
                raise ChartRenderError(
                    f"bar series {series['id']!r} has scale_domain {domain} that excludes "
                    "zero: truncated bars are categorically misleading and are refused, "
                    "annotation or not (DESIGN §3.7)"
                )
            scale["zero"] = True
        mark: dict[str, Any] = {"type": mark_type, "color": colour}
        if mark_type == "line":
            mark["strokeWidth"] = 2
        layers.append(
            {
                "data": {"values": _records(_clip(item["frame"], x0, x1))},
                "mark": mark,
                "encoding": {
                    "x": x_enc,
                    "y": {
                        "field": value_col,
                        "type": "quantitative",
                        "scale": scale,
                        "axis": {"title": series["label"], "grid": False},
                    },
                },
            }
        )
        # Direct labelling (no legend): the series label as on-chart text.
        layers.append(_text_datum(series["label"], 6, 12 + 14 * order, colour))

        baseline = series.get("baseline")
        if baseline is not None:
            layers.append(
                {
                    "data": {"values": [{"y": _clean(baseline["value"])}]},
                    "mark": {"type": "rule", "strokeDash": [4, 3], "color": _BASELINE_INK},
                    "encoding": {
                        "y": {"field": "y", "type": "quantitative", "scale": scale},
                    },
                }
            )
            layers.append(_text_datum(baseline["label"], 6, 26 + 14 * order, colour))

        annotations = series.get("annotations") or {}
        non_zero = annotations.get("non_zero_baseline")
        if non_zero is not None and _domain_excludes_zero(domain):
            layers.append(_text_datum(non_zero["label"], 6, 40 + 14 * order, colour))

    return {"width": width, "height": height, "layer": layers}


def _text_datum(text: str, x: float, y: float, colour: str) -> dict[str, Any]:
    """A pixel-positioned text mark carrying a single label — annotation
    text that must appear in the artefact without joining any y-scale
    group."""
    return {
        "data": {"values": [{"label": text}]},
        "mark": {
            "type": "text",
            "align": "left",
            "x": x,
            "y": y,
            "fontSize": 9.5,
            "color": colour,
        },
        "encoding": {"text": {"field": "label"}},
    }


def _caption_block(lines: list[str], width_px: int) -> dict[str, Any]:
    """The caption strip as a Vega-Lite text layer, baked into every
    export (DESIGN §3.7)."""
    return {
        "width": width_px,
        "height": 12 * len(lines) + 4,
        "data": {"values": [{"i": i, "line": line} for i, line in enumerate(lines)]},
        "mark": {"type": "text", "align": "left", "x": 0, "fontSize": 9.5, "color": _INK},
        "encoding": {
            "y": {"field": "i", "type": "ordinal", "axis": None, "scale": {"paddingInner": 0.4}},
            "text": {"field": "line"},
        },
    }


def build_vega_lite(
    validated: RenderValidatedSpec,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    site_url: str,
    width_px: int = DEFAULT_WIDTH_PX,
    *,
    series_cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> dict[str, Any]:
    """The pure renderer core: validated spec + frames → Vega-Lite JSON.

    Accepts only :class:`RenderValidatedSpec` (a bare mapping raises
    :class:`TypeError` — review finding #133). Deterministic: identical
    inputs produce an identical dict, and integrity annotations, caption
    strip and palette assignment are all assertable on the output.
    """
    if not isinstance(validated, RenderValidatedSpec):
        raise TypeError(
            "build_vega_lite renders only a RenderValidatedSpec (the "
            "validate_spec_for_render proof token), never a bare spec mapping "
            "(review finding #133)"
        )
    spec = validated.spec

    prepared: list[dict[str, Any]] = []
    for index, series in enumerate(spec["series"]):
        frame, value_col = _series_frame(series, frames, manifest, cache=series_cache)
        prepared.append(
            {
                "series": series,
                "colour": _series_colour(series, index),
                "value_col": value_col,
                "frame": frame,
                "band": _band_frame(series, frames, manifest),
            }
        )

    chart_type = spec["chart_type"]
    dual = chart_type in ("dual_axis_line", "context_recent_inset")

    if chart_type == "context_recent_inset":
        context = spec["panels"]["context"]
        recent = spec["panels"]["recent"]
        cx0, cx1 = context["time_range_ce"]
        rx0, rx1 = recent["time_range_ce"]
        chart: dict[str, Any] = {
            "hconcat": [
                _dual_panel(spec, prepared, cx0, cx1, context["label"], 560, 300, True),
                _dual_panel(spec, prepared, rx0, rx1, recent["label"], 230, 300, False),
            ]
        }
    elif dual:
        x0, x1 = spec["time_range_ce"]
        chart = _dual_panel(spec, prepared, x0, x1, spec["title"], 620, 320, True)
    else:
        x0, x1 = spec["time_range_ce"]
        chart = _single_axis_panel(spec, prepared, x0, x1, spec["title"], width_px, 320)

    lines = caption_lines(validated, manifest, site_url, width_px)
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "#fcfcfb",
        "title": {"text": spec["title"], "anchor": "start", "fontSize": 16, "offset": 12},
        "config": {
            "font": "sans-serif",
            "axis": {"titleFontWeight": 600, "domainColor": "#d5d4d0", "tickColor": "#d5d4d0"},
            "view": {"stroke": None},
            "concat": {"spacing": 30},
        },
        "vconcat": [chart, _caption_block(lines, width_px)],
    }


def caption_lines(
    validated: RenderValidatedSpec,
    manifest: Mapping[str, Any],
    site_url: str,
    width_px: int = DEFAULT_WIDTH_PX,
) -> list[str]:
    """The caption strip: manifest attribution + licence strings, manifest
    access date, deployment site URL and the spec hash — and nothing from
    the spec's free text (review finding #137). Below
    :data:`CAPTION_MIN_FULL_WIDTH_PX` the strip drops to source names +
    site URL (the responsive rule, DESIGN §3.7)."""
    spec = validated.spec
    datasets = _datasets(manifest)
    pairs = _pairs(manifest)

    ordered_ids: list[str] = []
    for series in spec["series"]:
        for dataset_id in _series_dataset_ids(series):
            if dataset_id not in ordered_ids:
                ordered_ids.append(dataset_id)

    short = width_px < CAPTION_MIN_FULL_WIDTH_PX
    lines: list[str] = []
    for dataset_id in ordered_ids:
        entry = datasets[dataset_id]
        attribution = str(entry["attribution_text"]).strip()
        if short:
            lines.append(f"Data: {attribution}")
        else:
            licence = str(entry.get("licence", "")).strip()
            lines.append(f"Data: {attribution} — {licence}")

    if not short:
        # Integrity disclosures ride the caption surface (reviews #47/#50):
        # overlap omissions and the rebaseline alignment method, from the
        # manifest pairs the spliced series reference.
        for series in spec["series"]:
            pair = pairs.get(series.get("splice_pair_id"))
            if pair is None:
                continue
            overlap = pair.get("overlap") or {}
            note = str(overlap.get("note", "")).strip()
            if note:
                lines.append(note)
            rebaseline = pair.get("rebaseline") or {}
            disclosure = str(rebaseline.get("disclosure", "")).strip()
            if disclosure:
                lines.append(disclosure)

    access_date = manifest.get("access_date")
    lines.append(f"Accessed {access_date} · {site_url} · ChartSpec {spec_hash(spec)[:12]}")
    return lines


def csv_export(
    validated: RenderValidatedSpec,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    site_url: str,
    *,
    series_cache: dict[str, tuple[pd.DataFrame, str]] | None = None,
) -> str:
    """CSV of the plotted data, with the caption strip's attribution as
    leading ``#`` header comment lines (DESIGN §3.7: attribution is part
    of every artefact)."""
    spec = validated.spec
    header_comments = [
        f"# {line}" for line in caption_lines(validated, manifest, site_url, DEFAULT_WIDTH_PX)
    ]

    x0, x1 = _plot_range(spec)
    wide: pd.DataFrame | None = None
    for series in spec["series"]:
        frame, value_col = _series_frame(series, frames, manifest, cache=series_cache)
        clipped = _clip(frame, x0, x1)[["year_ce", value_col]].rename(
            columns={value_col: series["id"]}
        )
        wide = clipped if wide is None else wide.merge(clipped, on="year_ce", how="outer")
    wide = (wide if wide is not None else pd.DataFrame({"year_ce": []})).sort_values("year_ce")

    return "\n".join(header_comments) + "\n" + wide.to_csv(index=False)


def vega_lite_json_text(vega_lite: Mapping[str, Any]) -> str:
    """Canonical, newline-terminated serialisation of a Vega-Lite dict —
    the byte-identity surface for permalink re-render determinism."""
    return json.dumps(vega_lite, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def render_chart(
    spec: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    site_url: str,
    width_px: int = DEFAULT_WIDTH_PX,
) -> ChartArtifact:
    """The artefact path: compute extents from the pre-landed frames, run
    :func:`charts.spec.validate_spec_for_render` (so a spec failing only
    the extents-aware checks refuses here — review finding #133), then
    build the pure artefact bundle. No fetching; a missing frame raises
    :class:`ChartRenderError` naming the dataset (ADR-023)."""
    from charts.alt_text import alt_text

    # One per-series pipeline cache shared across all four artifact passes
    # (finding #297): extents, VL build, alt text and CSV export otherwise
    # each re-run the same per-series transform pipeline (O(rows²) rolling
    # mean included) from scratch — 4×N executions to build one artifact.
    series_cache: dict[str, tuple[pd.DataFrame, str]] = {}
    extents = compute_data_extents(spec, frames, manifest, series_cache=series_cache)
    validated = validate_spec_for_render(spec, manifest, extents)
    vega_lite = build_vega_lite(
        validated, frames, manifest, site_url, width_px, series_cache=series_cache
    )
    return ChartArtifact(
        spec_hash=spec_hash(spec),
        vega_lite=vega_lite,
        vega_lite_text=vega_lite_json_text(vega_lite),
        alt_text=alt_text(validated, frames, manifest, series_cache=series_cache),
        csv_text=csv_export(validated, frames, manifest, site_url, series_cache=series_cache),
    )


def render_svg(vega_lite: Mapping[str, Any]) -> str:
    """vl-convert edge: Vega-Lite dict → SVG text (integration tier)."""
    import vl_convert

    return vl_convert.vegalite_to_svg(dict(vega_lite))


def render_png(vega_lite: Mapping[str, Any], scale: float = 2.0) -> bytes:
    """vl-convert edge: Vega-Lite dict → PNG bytes (integration tier)."""
    import vl_convert

    return vl_convert.vegalite_to_png(dict(vega_lite), scale=scale)
