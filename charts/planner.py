"""Chart planner (issue #16): chart_request → validated ChartSpec or honest refusal.

Contract stubs for the RED phase — every function below documents its
contract and raises :class:`NotImplementedError`; the failing tests in
``tests/unit/test_chart_planner.py`` pin the behaviour the implementer
must satisfy (ORCHESTRATION.md loop steps 2–3).

Design anchors
--------------

- **DESIGN §3.7 / ADR-020**: the LLM writes a ChartSpec, never code. The
  planner is one *structured-output* call (``claude-haiku-4-5``) through
  the provider seam — never the citations call (§3.4) — whose request
  carries the dataset catalogue and the ChartSpec schema, and whose
  output is validated by :func:`charts.spec.validate_spec` in planner
  mode (``data_extents=None``) before anything downstream sees it.
- **Seam choice (resolved for ratification)**: the planner calls
  ``ProviderAdapter.structured`` via the pure builder
  :func:`build_planner_request`, not the legacy
  ``ProviderAdapter.plan_chart`` method. Reasons: the mandate requires
  the request to carry the ChartSpec schema and the prompt scaffold
  (``plan_chart``'s ``(request, catalog)`` payload carries neither), and
  requires usage accounting via ``StructuredResult.usage`` for the
  #21/#22 spend ledger (``plan_chart`` returns a bare dict with no usage
  channel). ``plan_chart`` remains on the protocol untouched; its
  retirement is an orchestrator decision, not this issue's.
- **Review finding #117**: the catalogue is manifest-derived and contains
  chart-pack datasets ONLY (``in_chart_pack: true`` via
  :func:`charts.pack.chart_pack_dataset_ids`) and renderable splice
  pairs only (:func:`charts.pack.blocked_splice_pairs` empty for the
  pair). Provisional datasets and the pairs that depend on them are
  excluded from what the model can even see — and a model that
  hallucinates them anyway is stopped by the validator cross-check.
- **ADR-021**: a request needing data outside the pack gets an honest
  refusal naming the nearest available datasets (pure lexical match over
  the catalogue — no model call, no network), and the gap is written as
  one structured log record for pack curation. No web fetch, ever, in
  MVP.
- **Retry discipline (IMPLEMENTATION §4.3, mirroring #10)**: exactly one
  retry across the whole call, then a typed error. Output that fails the
  planner *output schema* (not parseable as a spec/unavailable outcome)
  retries with the SAME request, like the #10 classifier. A
  schema-valid spec REFUSED by the validator does not retry blindly:
  the retry request carries the validator's violations
  (``build_planner_request(..., violations=...)``) so the model can fix
  what was actually wrong. Never a third call.
- **Cherry-pick resistance**: the prompt scaffold instructs the model to
  default every chart to the FULL available range of its datasets
  (DESIGN §3.7 "no cherry-picked default ranges"); the catalogue carries
  each dataset's coverage so the model knows what "full" is. The
  validator refuses zero-excluding axis windows without disclosure and
  out-of-coverage ranges regardless of what the model does.

Planner output shape (the structured call's value)
--------------------------------------------------

The model returns exactly one of::

    {"outcome": "spec", "spec": { ...ChartSpec (charts/spec.py schema)... }}
    {"outcome": "unavailable", "requested_data": "<the data the request needs>"}

``planner_output_schema()`` steers the constrained decoder to this shape;
``plan_chart_request`` enforces it on whatever comes back (a schema is
steering, not validation — the #10 convention).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag.provider import ProviderAdapter

#: The planner model (DESIGN §3.7: a separate structured-output Haiku
#: call; §9 cost model ~$0.002/chart). Model id is config, not code.
PLANNER_MODEL = "claude-haiku-4-5"

#: Output budget for one planner call: a ChartSpec is ~1 KB of JSON; the
#: cap keeps a runaway response from breaking the §9 cost model.
PLANNER_MAX_TOKENS = 2048

#: The dedicated logger for ADR-021 curation-gap records. The service
#: layer (#22) subscribes to this name; tests capture it with caplog.
CURATION_GAP_LOGGER_NAME = "charts.planner.curation_gaps"


@dataclass(frozen=True)
class CurationGap:
    """One ADR-021 curation-gap record: a chart request the pack cannot serve.

    Written (via :func:`log_curation_gap`) whenever the planner refuses
    for unavailable data, so the pack grows demand-driven with human
    review in the loop. Pure data — no timestamp (the logging layer owns
    clocks), no network, and never a fetch attempt.
    """

    #: The rewritten chart request (``QueryDecision.chart_request``).
    chart_request: str
    #: The model's description of the data the request needs
    #: (the ``requested_data`` field of the unavailable outcome).
    requested_data: str
    #: Nearest available pack datasets, best match first
    #: (:func:`nearest_available_datasets` over the same catalogue the
    #: model saw).
    nearest_datasets: tuple[str, ...]


@dataclass(frozen=True)
class PlannedChart:
    """A successful plan: a spec that passed ``validate_spec`` in planner mode.

    ``spec`` is the model's ChartSpec exactly as returned (no silent
    normalisation — the render path re-validates with data extents).
    ``usage`` is the summed ``StructuredResult.usage`` of every adapter
    call made (both calls when the retry fired, finding #92), or None
    when the adapter reported none.
    """

    spec: dict[str, Any]
    usage: Mapping[str, int] | None = None


@dataclass(frozen=True)
class ChartRefusal:
    """The honest unavailable-data refusal (DESIGN §3.7 / ADR-021).

    ``message`` is user-facing: it names the requested data and the
    nearest available datasets (by id or catalogue title). ``gap`` is the
    curation record that was logged. ``usage`` as on
    :class:`PlannedChart`.
    """

    message: str
    gap: CurationGap
    usage: Mapping[str, int] | None = None


class PlannerSpecError(Exception):
    """The planner could not produce a validated ChartSpec after its retry.

    The typed honest-failure signal (issue #16 TDD plan step 3; mirrors
    #10's ``MalformedClassifierOutputError``): raised after exactly two
    adapter calls when both outputs were either malformed against the
    planner output schema or refused by ``charts.spec.validate_spec``.
    ``violations`` carries the final attempt's failure detail — schema
    complaints or ``SpecViolation`` ``path: reason`` strings — so a log
    line alone is actionable. Never a bare KeyError/ValueError crash.
    """

    def __init__(self, message: str, violations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.violations: tuple[str, ...] = tuple(violations)


def build_dataset_catalogue(manifest: Any) -> dict[str, Any]:
    """Pure: the planner-facing dataset catalogue from the dataset manifest.

    ``manifest`` is anything :func:`charts.pack.chart_pack_dataset_ids`
    accepts (path, raw mapping, or loaded object). Returns a plain
    JSON-serialisable mapping::

        {
          "datasets": {<id>: {"title"?, "variable", "unit"?, "time_axis",
                              "coverage"}, ...},
          "splice_pairs": [{"id", "paleo", "instrumental",
                            "splice_year_ce", ...}, ...],
        }

    Contract (review finding #117 — pinned by tests):

    - ``datasets`` keys are EXACTLY ``chart_pack_dataset_ids(manifest)``
      — ``in_chart_pack: true`` members only. ``open-provisional``
      entries (Kaufman, Bereiter) never appear, in any field, so the
      model cannot even see them.
    - ``splice_pairs`` contains only pairs whose members are all in the
      chart pack (``blocked_splice_pairs`` says nothing about them);
      blocked pair ids never appear.
    - every dataset entry carries the manifest's ``variable``,
      ``time_axis`` and ``coverage`` blocks — coverage is what lets the
      model honour the full-available-range default.
    """
    raise NotImplementedError("issue #16: implement after the RED tests")


def planner_output_schema() -> dict[str, Any]:
    """Pure: the JSON Schema steering the planner's structured output.

    Admits exactly the two outcomes documented in the module docstring:
    ``outcome: "spec"`` carrying a ChartSpec constrained by
    :func:`charts.spec.chartspec_schema` (so the frozen vocabulary —
    every chart type, transform op and overlap policy — reaches the
    constrained decoder), or ``outcome: "unavailable"`` carrying
    ``requested_data``. Closed (``additionalProperties: false``).
    """
    raise NotImplementedError("issue #16: implement after the RED tests")


def build_planner_request(
    chart_request: str,
    catalogue: Mapping[str, Any],
    violations: Sequence[str] = (),
) -> dict[str, Any]:
    """Pure builder: the ``ProviderAdapter.structured`` payload for one
    planner attempt.

    Returns ``{"messages", "system", "schema", "config"}``:

    - ``system`` (the dedicated top-level channel, finding #91) carries
      the planner instructions: emit a ChartSpec over the catalogue's
      datasets only; default to the FULL available range of the plotted
      datasets unless the user explicitly asked for a narrower window
      (the anti-cherry-pick scaffold, DESIGN §3.7); use the
      ``unavailable`` outcome — never an invented dataset — when the
      catalogue cannot serve the request. The catalogue rides in the
      request verbatim (JSON), so what the model saw is auditable.
    - ``messages`` ends with the user's ``chart_request``; never a
      ``role: "system"`` entry, never a ``documents`` key.
    - ``schema`` is :func:`planner_output_schema`.
    - ``config`` is ``{"model": PLANNER_MODEL, "max_tokens":
      PLANNER_MAX_TOKENS}`` — NEVER a ``citations`` key
      (§3.4/IMPLEMENTATION §4.3).
    - ``violations`` non-empty builds the single retry's request: the
      validator's ``path: reason`` strings are included in the prompt so
      the model can repair the refused spec (never a blind retry).

    Deterministic and canonicalisable: identical inputs produce an
    identical payload with a stable
    ``rag.provider.canonical_request_hash`` — the property replay
    fixtures key on.
    """
    raise NotImplementedError("issue #16: implement after the RED tests")


def nearest_available_datasets(
    requested: str,
    catalogue: Mapping[str, Any],
    limit: int = 3,
) -> tuple[str, ...]:
    """Pure: the nearest available pack datasets for an unservable request.

    Deterministic lexical match of ``requested`` (the unavailable
    outcome's ``requested_data``, or the raw request) against the
    catalogue's dataset text (title/variable), best match first, at most
    ``limit`` ids, every id a catalogue (= chart pack) member. Never
    empty while the catalogue has datasets — the ADR-021 refusal must
    always have something honest to offer. Case-insensitive. No model
    call, no network.
    """
    raise NotImplementedError("issue #16: implement after the RED tests")


def log_curation_gap(gap: CurationGap) -> None:
    """Write one structured ADR-021 curation-gap log record.

    Emits a single record on :data:`CURATION_GAP_LOGGER_NAME` whose
    ``extra`` fields carry ``chart_request``, ``requested_data`` and
    ``nearest_datasets`` verbatim (structured, greppable — not prose
    only). Logging is the ONLY side effect: no fetch, no file, no
    network (allowlisted live-fetch is Phase 2, ADR-021).
    """
    raise NotImplementedError("issue #16: implement after the RED tests")


def plan_chart_request(
    adapter: ProviderAdapter,
    chart_request: str,
    manifest: Any,
) -> PlannedChart | ChartRefusal:
    """The chart-planner entry point: one structured call, validate, honest exit.

    Consumes ``QueryDecision.chart_request`` (the #10 CHART route). Flow:

    1. ``catalogue = build_dataset_catalogue(manifest)``;
       ``adapter.structured(**build_planner_request(chart_request,
       catalogue))`` — the ONLY adapter method ever called is
       ``structured``; never ``generate`` (§3.4), never a fetch.
    2. ``outcome == "unavailable"`` → compute
       :func:`nearest_available_datasets`, :func:`log_curation_gap`, and
       return a :class:`ChartRefusal` naming the requested data and the
       nearest datasets. No retry — an honest refusal is a success path.
    3. ``outcome == "spec"`` → ``charts.spec.validate_spec(spec,
       manifest)`` in planner mode (``data_extents=None``). Valid →
       :class:`PlannedChart`.
    4. Failure handling, single shared retry budget (never a third
       call): output malformed against :func:`planner_output_schema` →
       retry once with the SAME request (the #10 convention); spec
       refused by the validator → retry once with the violations fed
       back (``build_planner_request(..., violations=...)``). A second
       failure of either kind raises :class:`PlannerSpecError` carrying
       the final violations.

    Usage accounting (finding #92): the returned object's ``usage`` sums
    ``StructuredResult.usage`` across every call made, so the #21/#22
    ledger never under-reports a retried plan.
    """
    raise NotImplementedError("issue #16: implement after the RED tests")
