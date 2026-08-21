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

RED phase: the public functions below are contract stubs raising
:class:`NotImplementedError`. The failing tests in
``tests/unit/test_chart_render_vegalite.py``,
``tests/unit/test_chart_caption_alt_csv.py``,
``tests/unit/test_chart_render_determinism.py`` and
``tests/integration/test_chart_render_exports.py`` pin the contracts; the
implementer makes them pass without weakening them (ORCHESTRATION.md) and
then deletes this paragraph.

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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from charts.spec import RenderValidatedSpec

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


def compute_data_extents(
    spec: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Post-transform (min, max) per series id, within the spec's plotted
    range — the mapping :func:`charts.spec.validate_spec_for_render`
    requires (review finding #133). Pure; raises
    :class:`ChartRenderError` naming any dataset whose frame is absent
    (pre-landed frames only, never a fetch — ADR-023)."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def build_vega_lite(
    validated: RenderValidatedSpec,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    site_url: str,
    width_px: int = DEFAULT_WIDTH_PX,
) -> dict[str, Any]:
    """The pure renderer core: validated spec + frames → Vega-Lite JSON.

    Accepts only :class:`RenderValidatedSpec` (a bare mapping raises
    :class:`TypeError` — review finding #133). Deterministic: identical
    inputs produce an identical dict, and integrity annotations, caption
    strip and palette assignment are all assertable on the output.
    """
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


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
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def csv_export(
    validated: RenderValidatedSpec,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    site_url: str,
) -> str:
    """CSV of the plotted data, with the caption strip's attribution as
    leading ``#`` header comment lines (DESIGN §3.7: attribution is part
    of every artefact)."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def vega_lite_json_text(vega_lite: Mapping[str, Any]) -> str:
    """Canonical, newline-terminated serialisation of a Vega-Lite dict —
    the byte-identity surface for permalink re-render determinism."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


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
    raise NotImplementedError("issue #17 red phase — implement to pass tests/unit")


def render_svg(vega_lite: Mapping[str, Any]) -> str:
    """vl-convert edge: Vega-Lite dict → SVG text (integration tier)."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/integration")


def render_png(vega_lite: Mapping[str, Any], scale: float = 2.0) -> bytes:
    """vl-convert edge: Vega-Lite dict → PNG bytes (integration tier)."""
    raise NotImplementedError("issue #17 red phase — implement to pass tests/integration")
