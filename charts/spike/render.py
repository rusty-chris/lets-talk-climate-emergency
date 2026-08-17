"""Spike renderer: flagship ChartSpec JSON -> Vega-Lite -> SVG + PNG.

Pipeline (all pure until the vl-convert call at the edge, per
IMPLEMENTATION.md section 1):

    raw files --parsers--> tidy frames --transforms--> plot frames
        --build_vega_lite--> Vega-Lite dict --vl_convert--> SVG/PNG

Integrity rules exercised here and assertable on the emitted Vega-Lite/SVG
text (DESIGN section 3.7): rendered splice-point rules + labels, resolution
notes, a labelled zero baseline with the reference period, the Kaufman 5-95%
uncertainty band, per-axis colour-matched titles on the dual axis, and a
caption strip (sources, licences posture, access date, site URL) baked into
the export.

Run:  uv run python -m charts.spike.render [--raw-dir data/spike] [--out-dir reviews/spike-04]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from charts.spike import parsers, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("flagship_spec.json")
MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"

#: Deployment config, not spec content — the ChartSpec never carries a URL.
SITE_URL = "https://SITE-URL-PLACEHOLDER.example"

#: dataset id -> (parser, raw filename) — the spike's stand-in for charts/pack.py
RAW_FILES = {
    "bereiter2015_co2": (parsers.parse_bereiter_co2, "antarctica2015co2composite.txt"),
    "kaufman2020_temp12k": (parsers.parse_kaufman_temp12k, "temp12k_allmethods_percentiles.csv"),
    "noaa_gml_co2_mlo": (parsers.parse_gml_co2_annual, "co2_annmean_mlo.csv"),
    "gistemp_v4": (parsers.parse_gistemp_annual, "GLB.Ts+dSST.csv"),
}


def load_spec(path: Path | str = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def check_spec_against_manifest(spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    """The ADR-020 invariant: splice years and alignment periods come from the
    manifest (curation-time decisions), so a spec that disagrees is invalid."""
    pairs = {p["id"]: p for p in manifest["splice_pairs"]}
    for series in spec["series"]:
        pair = pairs[series["splice_pair_id"]]
        expected = [pair["paleo"], pair["instrumental"]]
        if series["splice_series"] != expected:
            raise ValueError(
                f"series {series['id']}: splice_series {series['splice_series']} "
                f"does not match manifest pair {pair['id']} {expected}"
            )
        if series["annotations"]["splice_point"]["year_ce"] != pair["splice_year_ce"]:
            raise ValueError(
                f"series {series['id']}: splice year differs from manifest pair {pair['id']}"
            )
        rb_spec, rb_manifest = series.get("rebaseline_to"), pair.get("rebaseline")
        if (rb_spec is None) != (rb_manifest is None):
            raise ValueError(f"series {series['id']}: rebaseline presence differs from manifest")
        if rb_spec is not None and list(rb_spec["alignment_period_ce"]) != list(
            rb_manifest["alignment_period_ce"]
        ):
            raise ValueError(
                f"series {series['id']}: alignment period {rb_spec['alignment_period_ce']} "
                f"is not the manifest-fixed period {rb_manifest['alignment_period_ce']}"
            )


def build_flagship_frames(
    spec: dict[str, Any], raw_dir: Path | str, manifest: dict[str, Any] | None = None
) -> dict[str, pd.DataFrame]:
    """Parse + transform the four sources into the frames the chart plots.

    Returns {'co2': spliced CO2 frame, 'temp': spliced temperature frame,
    'temp_band': Kaufman 5-95% band}, each restricted to the context panel's
    time range and carrying a `segment` column where spliced.
    """
    manifest = manifest or load_manifest()
    check_spec_against_manifest(spec, manifest)
    raw_dir = Path(raw_dir)
    parsed = {ds: parse(raw_dir / fname) for ds, (parse, fname) in RAW_FILES.items()}

    if spec["time_axis"] != {"calendar": "CE", "convert_bp": True}:
        raise ValueError("spike only implements time_axis {calendar: CE, convert_bp: true}")
    bereiter = transforms.bp_to_ce(parsed["bereiter2015_co2"])
    kaufman = transforms.bp_to_ce(parsed["kaufman2020_temp12k"])
    mlo = parsed["noaa_gml_co2_mlo"].astype({"year_ce": "float64"})
    gistemp = parsed["gistemp_v4"].astype({"year_ce": "float64"})

    series = {s["id"]: s for s in spec["series"]}
    start, end = spec["panels"]["context"]["time_range_ce"]

    co2_splice_year = series["co2"]["annotations"]["splice_point"]["year_ce"]
    co2 = transforms.splice(
        bereiter.rename(columns={"co2_ppm": "value"})[["year_ce", "value"]],
        mlo.rename(columns={"co2_ppm": "value"})[["year_ce", "value"]],
        co2_splice_year,
    )

    align = tuple(series["temp"]["rebaseline_to"]["alignment_period_ce"])
    gistemp_aligned = transforms.rebaseline(gistemp, "temp_anomaly_c", align)
    temp_splice_year = series["temp"]["annotations"]["splice_point"]["year_ce"]
    temp = transforms.splice(
        kaufman.rename(columns={"temp_c": "value"})[["year_ce", "value"]],
        gistemp_aligned.rename(columns={"temp_anomaly_c": "value"})[["year_ce", "value"]],
        temp_splice_year,
    )

    band = kaufman[kaufman["year_ce"] < temp_splice_year][["year_ce", "temp_c_5", "temp_c_95"]]

    in_range = lambda df: df[(df["year_ce"] >= start) & (df["year_ce"] <= end)].reset_index(  # noqa: E731
        drop=True
    )
    return {"co2": in_range(co2), "temp": in_range(temp), "temp_band": in_range(band)}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
        for r in df.to_dict("records")
    ]


def _panel(
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    panel_key: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """One panel of the context+recent-inset pair: layered dual-axis chart."""
    panel = spec["panels"][panel_key]
    x0, x1 = panel["time_range_ce"]
    co2_s = next(s for s in spec["series"] if s["id"] == "co2")
    temp_s = next(s for s in spec["series"] if s["id"] == "temp")
    show_annotations = panel_key == "context"  # inset repeats the axis, not the notes

    def clip(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["year_ce"] >= x0) & (df["year_ce"] <= x1)]

    x_enc = {
        "field": "year_ce",
        "type": "quantitative",
        "scale": {"domain": [x0, x1], "nice": False},
        "axis": {
            "title": panel["label"],
            "labelExpr": (
                "datum.value < 0 ? format(-datum.value, 'd') + ' BCE' : format(datum.value, 'd')"
            ),
            "grid": False,
            "tickCount": 6 if panel_key == "context" else 4,
        },
    }

    co2_layers: list[dict[str, Any]] = [
        {
            "data": {"values": _records(clip(frames["co2"]))},
            "mark": {"type": "line", "strokeWidth": 2, "color": co2_s["color"]},
            "encoding": {
                "x": x_enc,
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "scale": {"domain": co2_s["scale_domain"], "nice": False},
                    "axis": {
                        "title": co2_s["label"],
                        "titleColor": co2_s["color"],
                        "labelColor": co2_s["color"],
                        "orient": "left",
                        "grid": False,
                    },
                },
                "detail": {"field": "segment"},
            },
        }
    ]

    band = clip(frames["temp_band"])
    temp_axis = {
        "title": temp_s["label"],
        "titleColor": temp_s["color"],
        "labelColor": temp_s["color"],
        "orient": "right",
        "grid": False,
    }
    temp_layers: list[dict[str, Any]] = [
        {
            "data": {"values": _records(clip(frames["temp"]))},
            "mark": {"type": "line", "strokeWidth": 2, "color": temp_s["color"]},
            "encoding": {
                "x": x_enc,
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "scale": {"domain": temp_s["scale_domain"], "nice": False},
                    "axis": temp_axis,
                },
                "detail": {"field": "segment"},
            },
        }
    ]
    if not band.empty:
        temp_layers.append(
            {
                "data": {"values": _records(band)},
                "mark": {"type": "area", "color": temp_s["color"], "opacity": 0.15},
                "encoding": {
                    "x": x_enc,
                    "y": {
                        "field": "temp_c_5",
                        "type": "quantitative",
                        "scale": {"domain": temp_s["scale_domain"], "nice": False},
                        "axis": temp_axis,
                    },
                    "y2": {"field": "temp_c_95"},
                },
            }
        )
    # Labelled zero baseline (integrity default: reference period named on-chart).
    temp_layers.append(
        {
            "data": {
                "values": [{"y": temp_s["baseline"]["value"], "label": temp_s["baseline"]["label"]}]
            },
            "mark": {"type": "rule", "strokeDash": [4, 3], "color": "#52514e", "opacity": 0.8},
            "encoding": {
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "scale": {"domain": temp_s["scale_domain"]},
                    "axis": temp_axis,
                }
            },
        }
    )
    if show_annotations:
        temp_layers.append(
            {
                "data": {
                    "values": [
                        {"y": temp_s["baseline"]["value"], "label": temp_s["baseline"]["label"]}
                    ]
                },
                "mark": {
                    "type": "text",
                    "align": "left",
                    "baseline": "bottom",
                    "dx": 4,
                    "dy": -3,
                    "x": 0,
                    "fontSize": 10,
                    "color": "#52514e",
                },
                "encoding": {
                    "y": {
                        "field": "y",
                        "type": "quantitative",
                        "scale": {"domain": temp_s["scale_domain"]},
                        "axis": temp_axis,
                    },
                    "text": {"field": "label"},
                },
            }
        )

    # Splice-point rules + labels + resolution notes (mandatory, DESIGN 3.7).
    annotation_layers: list[dict[str, Any]] = []
    for i, s in enumerate((co2_s, temp_s)):
        splice_year = s["annotations"]["splice_point"]["year_ce"]
        if not (x0 <= splice_year <= x1):
            continue
        label = s["annotations"]["splice_point"]["label"]
        note = s["annotations"]["resolution_note"]
        annotation_layers.append(
            {
                "data": {"values": [{"x": splice_year}]},
                "mark": {"type": "rule", "strokeDash": [2, 2], "color": s["color"], "opacity": 0.7},
                "encoding": {"x": {**x_enc, "field": "x"}},
            }
        )
        if show_annotations:
            annotation_layers.append(
                {
                    "data": {"values": [{"label": f"⇢ {label} — {note.lower()}"}]},
                    "mark": {
                        "type": "text",
                        "align": "left",
                        "x": 6,
                        "y": 10 + 14 * i,
                        "fontSize": 9.5,
                        "color": s["color"],
                    },
                    "encoding": {"text": {"field": "label"}},
                }
            )

    return {
        "width": width,
        "height": height,
        "layer": [{"layer": co2_layers}, {"layer": temp_layers + annotation_layers}],
        "resolve": {"scale": {"y": "independent"}},
    }


def caption_lines_from_manifest(spec: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Caption sources are generated from the manifest attribution strings,

    never free text in the spec (issue #15 vocabulary amendment 9 — the
    frozen ChartSpec schema has no caption block, so captions cannot drift
    from the licensing record, and open-provisional datasets render a
    distinguishable posture).
    """
    used: list[str] = []
    for series in spec["series"]:
        used.extend(series.get("splice_series") or [series["dataset"]])
    lines = []
    for ds_id in dict.fromkeys(used):
        entry = manifest["datasets"][ds_id]
        posture = (
            " [licence confirmation pending — open-provisional]"
            if entry.get("permitted_context") == "open-provisional"
            else ""
        )
        lines.append(f"Data: {entry['attribution_text']}{posture}")
    lines.append(
        f"Accessed {manifest['access_date']} · {SITE_URL} · "
        f"Rendered from ChartSpec {spec['chart_id']} v{spec['spec_version']}"
    )
    return lines


def build_vega_lite(
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    caption_lines = caption_lines_from_manifest(spec, manifest or load_manifest())
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "#fcfcfb",
        "title": {
            "text": spec["title"],
            "subtitle": spec["subtitle"],
            "anchor": "start",
            "fontSize": 17,
            "subtitleFontSize": 12,
            "subtitleColor": "#52514e",
            "offset": 14,
        },
        "config": {
            "font": "sans-serif",
            "axis": {
                "labelColor": "#52514e",
                "titleFontWeight": 600,
                "domainColor": "#d5d4d0",
                "tickColor": "#d5d4d0",
            },
            "view": {"stroke": None},
            "concat": {"spacing": 34},
        },
        "vconcat": [
            {
                "hconcat": [
                    _panel(spec, frames, "context", width=560, height=300),
                    _panel(spec, frames, "recent", width=230, height=300),
                ]
            },
            {
                # Caption strip, baked into every export (DESIGN 3.7).
                "width": 860,
                "height": 12 * len(caption_lines) + 4,
                "data": {"values": [{"i": i, "line": ln} for i, ln in enumerate(caption_lines)]},
                "mark": {
                    "type": "text",
                    "align": "left",
                    "x": 0,
                    "fontSize": 9.5,
                    "color": "#52514e",
                },
                "encoding": {
                    "y": {
                        "field": "i",
                        "type": "ordinal",
                        "axis": None,
                        "scale": {"paddingInner": 0.4},
                    },
                    "text": {"field": "line"},
                },
            },
        ],
    }


def render_flagship(
    spec_path: Path | str = SPEC_PATH,
    raw_dir: Path | str = REPO_ROOT / "data" / "spike",
    out_dir: Path | str = REPO_ROOT / "reviews" / "spike-04",
) -> dict[str, Path]:
    """The side-effect edge: build everything, then write VL JSON, SVG, PNG."""
    import vl_convert

    spec = load_spec(spec_path)
    frames = build_flagship_frames(spec, raw_dir)
    vl = build_vega_lite(spec, frames)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "vl": out_dir / "flagship.vl.json",
        "svg": out_dir / "flagship.svg",
        "png": out_dir / "flagship.png",
    }
    # Newline-terminated so committed artefacts satisfy the end-of-file hook.
    paths["vl"].write_text(json.dumps(vl, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["svg"].write_text(vl_convert.vegalite_to_svg(vl).rstrip("\n") + "\n", encoding="utf-8")
    # scale 1.5 keeps the committed PNG under the pre-commit large-file limit
    paths["png"].write_bytes(vl_convert.vegalite_to_png(vl, scale=1.5))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "spike"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reviews" / "spike-04"))
    args = ap.parse_args()
    written = render_flagship(raw_dir=args.raw_dir, out_dir=args.out_dir)
    for kind, p in written.items():
        print(f"{kind}: {p}")
