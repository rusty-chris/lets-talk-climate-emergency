"""Single-entity rule for multi-entity pack datasets (issue #425) — RED.

`owid_co2` is planner-reachable (`permitted_context: open` +
`in_chart_pack: true`, datasets/manifest.yaml) and its parser
(`charts.pack.parse_owid_co2`) returns a `country` column in which
aggregates like `World` appear alongside ~250 countries. The ChartSpec
vocabulary cannot express an entity selection and the renderer does no
country filtering, so a perfectly valid `{dataset: "owid_co2"}` spec plots
EVERY entity as one series — and the alt text derives its trend word from
an arbitrary last-sorted row. The verified reproduction renders the false
public sentence "CO₂ emissions is falling over this range".

The pinned fix (issue #425, orchestrator comment 2026-10-02) is a
**manifest-anchored default entity** — `default_entity: "World"` on the
dataset entry — applied at frame-load/render time, deliberately NOT a
planner-visible vocabulary field:

- exactly one entity is plotted (the manifest default), code-enforced in
  the render pipeline the way every other curation decision is (ADR-020,
  the #164 discipline);
- a multi-entity frame whose dataset has NO `default_entity` is refused,
  fail-closed, so the next multi-entity dataset cannot repeat this
  silently;
- the caption and the CSV header disclose which entity is shown
  ("World"), because a chart silently showing one entity under an
  all-entities title merely swaps one falsehood for another;
- the alt-text trend word derives from the plotted entity's series;
- `charts.planner.build_dataset_catalogue` keeps its FIXED per-dataset
  field list (`title`, `variable`, `time_axis`, `coverage`) so the new
  manifest field never reaches the planner prompt: the canonical request
  hash stays stable and the two recorded replay fixtures that reference
  owid_co2 (tests/fixtures/replay/31c5f7e0….json, 4b1dbd09….json) are
  not invalidated — re-recording is a paid session, so fixture-hash
  stability is a hard requirement.

All frames here are SYNTHETIC FIXTURES in the exact `parse_owid_co2`
output shape (`[country, iso_code, year_ce, co2_mt]`, sorted by country
then year) — invented entities and values, no real dataset is fetched or
read (ADR-023 / review finding #117).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from charts import planner, render
from tests._chart_render_fixtures import ACCESS_DATE, SITE_URL, validated

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST = REPO_ROOT / "datasets" / "manifest.yaml"

#: The fixed per-dataset field list build_dataset_catalogue exposes to the
#: planner prompt. This is the fixture-hash safety pin: `default_entity`
#: (or any new manifest field) joining this set changes the planner
#: request payload, hence the canonical request hash, hence invalidates
#: the recorded replay fixtures (a paid re-record).
CATALOGUE_DATASET_FIELDS = frozenset({"title", "variable", "time_axis", "coverage"})

#: The configured entity's plotted values, by year — what the chart must
#: show once exactly one entity renders. Rising, so the honest trend word
#: is "rising"; the invented country below falls to near zero so that the
#: CURRENT bug (all entities in one series, trend from an arbitrary
#: last-sorted row) deterministically reads "falling" — the same false
#: sentence shape the review probe produced on the real dataset.
WORLD_CO2_BY_YEAR = {1950: 5000.0, 1960: 10000.0, 1970: 15000.0, 1980: 20000.0, 1990: 25000.0}

#: The non-default entity. Note it extends PAST the default entity's last
#: year (to 2000): today the plotted range's final row is therefore
#: guaranteed to be this entity's, making the mixed-series trend word
#: deterministic ("falling") regardless of pandas sort stability.
ZUBROWKA_CO2_BY_YEAR = {
    1950: 2000.0,
    1960: 1500.0,
    1970: 1000.0,
    1980: 600.0,
    1990: 300.0,
    2000: 10.0,
}


def entity_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: a renderer-facing manifest with two multi-entity
    datasets in the owid_co2 shape — one carrying the manifest-anchored
    `default_entity`, one (the fail-closed case) without it. None of the
    caption-facing strings contain "World", so the disclosure assertions
    below can only be satisfied by the entity name itself (the real
    manifest's attribution says "Our World in Data" — a fixture reusing
    that text would vacuously pass the disclosure tests)."""
    entry_common: dict[str, Any] = {
        "permitted_context": "open",
        "in_chart_pack": True,
        "time_axis": {"unit": "year_ce"},
        "variable": {
            "name": "co2_mt",
            "unit": "Mt CO2/yr",
            "scope": "annual emissions by invented entity",
        },
        "coverage": {"first_year_ce": 1950, "last_year_ce": 2000},
        "attribution_text": "Cartographers' Guild (fictional)",
        "licence": "Open with required credit (invented)",
        "retrieved_at": ACCESS_DATE,
    }
    return {
        "version": "0.0.1-test",
        "access_date": ACCESS_DATE,
        "datasets": {
            "syn_entities_co2": {**entry_common, "default_entity": "World"},
            # The exact gap #425 fail-closes against: a parser that yields
            # multiple entities, and no default-entity configuration.
            "syn_entities_unconfigured": dict(entry_common),
        },
        "splice_pairs": [],
    }


def entity_frame() -> pd.DataFrame:
    """SYNTHETIC FIXTURE: a pre-landed frame in the exact parse_owid_co2
    output shape — `[country (object), iso_code (object), year_ce (int64),
    co2_mt (float64)]`, sorted by country then year — carrying the `World`
    aggregate alongside one invented country."""
    rows = [("World", "OWID_WRL", year, value) for year, value in sorted(WORLD_CO2_BY_YEAR.items())]
    rows += [
        ("Zubrowka", "ZUB", year, value) for year, value in sorted(ZUBROWKA_CO2_BY_YEAR.items())
    ]
    frame = pd.DataFrame(rows, columns=["country", "iso_code", "year_ce", "co2_mt"])
    frame["year_ce"] = frame["year_ce"].astype("int64")
    frame["co2_mt"] = frame["co2_mt"].astype("float64")
    return frame.sort_values(["country", "year_ce"]).reset_index(drop=True)


def entity_frames() -> dict[str, pd.DataFrame]:
    """Fresh pre-landed frames for both synthetic dataset ids."""
    return {
        "syn_entities_co2": entity_frame(),
        "syn_entities_unconfigured": entity_frame(),
    }


def entity_line_spec(dataset_id: str = "syn_entities_co2") -> dict[str, Any]:
    """A minimal, fully spec-legal line chart over a multi-entity dataset —
    the exact shape the planner can emit today, because the ChartSpec
    vocabulary has no entity field to require or refuse."""
    return {
        "spec_version": "1.0.0",
        "chart_id": "syn-entities",
        "chart_type": "line",
        "title": "Synthetic emissions over time",
        "time_range_ce": [1950, 2000],
        "series": [
            {
                "id": "co2",
                "label": "CO2 emissions (invented)",
                "unit": "Mt CO2/yr",
                "dataset": dataset_id,
            }
        ],
    }


#: Extents used only to mint the RenderValidatedSpec token for the caption
#: and CSV surfaces (no scale_domain in the spec, so no extent-aware check
#: consumes them): the configured entity's post-fix extent.
ENTITY_EXTENTS = {"co2": (5000.0, 25000.0)}


def _artifact() -> render.ChartArtifact:
    """One rendered artefact over the configured multi-entity dataset —
    the full render_chart path (extents → render-mode validation → VL /
    alt text / CSV), exactly how the service builds artefacts."""
    return render.render_chart(entity_line_spec(), entity_frames(), entity_manifest(), SITE_URL)


# ---------------------------------------------------------------------------
# (A) Exactly one entity is plotted (acceptance criterion 1)
# ---------------------------------------------------------------------------


def test_owid_shaped_series_plots_single_entity():
    """A spec over a multi-entity dataset with `default_entity: "World"`
    plots exactly the World series: one value per plotted year, and each
    year's value is World's — never a zigzag through every entity's rows
    rendered as one line (the #425 defect)."""
    artifact = _artifact()
    plotted = [
        row for row in _inline_rows(artifact.vega_lite) if "year_ce" in row and "co2_mt" in row
    ]
    assert plotted, "no plotted co2_mt rows found in the Vega-Lite output"
    counts = Counter(int(row["year_ce"]) for row in plotted)
    duplicated = {year: n for year, n in counts.items() if n > 1}
    assert not duplicated, (
        f"years plotted more than once {duplicated!r}: every entity's rows are "
        "being drawn as ONE series — the manifest default_entity must select "
        "exactly one entity at frame-prep time (issue #425)"
    )
    by_year = {int(row["year_ce"]): float(row["co2_mt"]) for row in plotted}
    assert by_year == WORLD_CO2_BY_YEAR, (
        f"the plotted values must be the manifest default entity's (World) — got {by_year!r}"
    )


def _inline_rows(vega_lite: Any) -> list[dict[str, Any]]:
    """Every inline datum row anywhere in the Vega-Lite structure."""
    rows: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            data = node.get("data")
            if isinstance(data, dict) and isinstance(data.get("values"), list):
                rows.extend(r for r in data["values"] if isinstance(r, dict))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(vega_lite)
    return rows


# ---------------------------------------------------------------------------
# (B) Fail-closed: an unconfigured multi-entity dataset refuses
#     (acceptance criterion 2)
# ---------------------------------------------------------------------------


def test_unconfigured_multi_entity_dataset_refuses_fail_closed():
    """A dataset whose frame carries multiple entities and whose manifest
    entry has NO `default_entity` must refuse at render-time frame
    preparation — a ChartRenderError naming the dataset and the missing
    field — never silently plot the many-entities-as-one-series falsehood.
    Fail-closed is the point: the NEXT multi-entity dataset added without
    configuration must be unable to repeat #425 silently."""
    spec = entity_line_spec("syn_entities_unconfigured")
    with pytest.raises(render.ChartRenderError, match="default_entity") as excinfo:
        render.render_chart(spec, entity_frames(), entity_manifest(), SITE_URL)
    assert "syn_entities_unconfigured" in str(excinfo.value), (
        "the refusal must name the offending dataset so a log line alone is actionable"
    )


# ---------------------------------------------------------------------------
# (C) Honest disclosure: caption and CSV header name the plotted entity
#     (acceptance criterion 3)
# ---------------------------------------------------------------------------


def test_caption_discloses_plotted_entity():
    """The rendered caption names the entity actually shown ("World") —
    manifest-anchored text, like every other caption line (amendment 9):
    a chart titled as emissions data that silently shows one entity is a
    differently incomplete falsehood. No fixture string other than the
    entity name contains "World", so only the disclosure satisfies this."""
    lines = render.caption_lines(
        validated(entity_line_spec(), ENTITY_EXTENTS, entity_manifest()),
        entity_manifest(),
        SITE_URL,
    )
    joined = "\n".join(lines)
    assert "World" in joined, f"caption does not disclose the plotted entity 'World': {joined!r}"


def test_csv_header_discloses_plotted_entity():
    """The CSV export's leading '#' header comments name the plotted
    entity — the download must carry the same disclosure as the caption
    (DESIGN §3.7: attribution/disclosure is part of every artefact)."""
    text = render.csv_export(
        validated(entity_line_spec(), ENTITY_EXTENTS, entity_manifest()),
        entity_frames(),
        entity_manifest(),
        SITE_URL,
    )
    comment_lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        comment_lines.append(line)
    assert comment_lines, "no leading # header comments in the CSV export"
    joined = "\n".join(comment_lines)
    assert "World" in joined, (
        f"CSV header comments do not disclose the plotted entity 'World': {joined!r}"
    )


# ---------------------------------------------------------------------------
# (D) Alt text derives its trend from the plotted entity
#     (acceptance criterion 4)
# ---------------------------------------------------------------------------


def test_alt_text_trend_derives_from_plotted_entity():
    """The alt-text trend word describes the plotted (World) series —
    rising — never an arbitrary last-sorted row. Today the mixed frame's
    final plotted row is the invented falling country's near-zero value,
    so the artefact asserts "falling": the exact false-public-sentence
    failure the review probe reproduced on real global emissions."""
    artifact = _artifact()
    assert "rising" in artifact.alt_text, (
        f"alt text must state the plotted entity's (World) rising trend: {artifact.alt_text!r}"
    )
    assert "falling" not in artifact.alt_text, (
        "alt text derives its trend word from an arbitrary row of the "
        f"multi-entity frame, not the plotted entity: {artifact.alt_text!r}"
    )


# ---------------------------------------------------------------------------
# (E) Fixture-hash safety pin: the planner catalogue's per-dataset field
#     list is fixed, so default_entity never reaches the prompt
# ---------------------------------------------------------------------------


def test_catalogue_field_list_is_unchanged_by_default_entity():
    """`build_dataset_catalogue` exposes ONLY the fixed per-dataset fields
    (`title`, `variable`, `time_axis`, `coverage`) — a manifest entry
    carrying `default_entity` must not leak it into the catalogue. The
    catalogue rides inside the planner request, which is keyed by its
    canonical hash: a new field would change the hash and invalidate the
    two recorded replay fixtures referencing owid_co2 (a paid re-record).
    GREEN today by construction — this is the regression pin that makes
    the #425 fix provably hash-safe."""
    catalogue = planner.build_dataset_catalogue(entity_manifest())
    for ds_id, entry in catalogue["datasets"].items():
        extras = set(entry) - CATALOGUE_DATASET_FIELDS
        assert not extras, f"catalogue entry {ds_id!r} leaks fields {sorted(extras)!r}"
    assert "default_entity" not in json.dumps(catalogue)

    # The same pin over the real manifest: once the implementer adds
    # default_entity to owid_co2, the live catalogue payload must still
    # serialise without it.
    real_catalogue = planner.build_dataset_catalogue(REAL_MANIFEST)
    for ds_id, entry in real_catalogue["datasets"].items():
        extras = set(entry) - CATALOGUE_DATASET_FIELDS
        assert not extras, f"catalogue entry {ds_id!r} leaks fields {sorted(extras)!r}"
    assert "default_entity" not in json.dumps(real_catalogue)


# ---------------------------------------------------------------------------
# The real manifest anchors the fix: owid_co2 names its default entity
# ---------------------------------------------------------------------------


def test_owid_co2_manifest_entry_carries_default_entity_world():
    """`datasets/manifest.yaml` must record `default_entity: "World"` on
    owid_co2 — the curation-time decision the render-time filter applies
    (ADR-020: curation decisions live in the manifest, code enforces
    them). "World" is OWID's own aggregate-entity name in the `country`
    column, verified against the parser's documented output."""
    raw = yaml.safe_load(REAL_MANIFEST.read_text(encoding="utf-8"))
    entry = raw["datasets"]["owid_co2"]
    assert entry.get("default_entity") == "World", (
        "owid_co2 is a multi-entity dataset reachable by the planner; it must "
        "carry the manifest-anchored default_entity 'World' (issue #425)"
    )
