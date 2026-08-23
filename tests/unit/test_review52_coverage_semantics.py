"""Review finding #52 — RED: the coverage contract fails CLOSED at the
range validator.

The #52 semantics are already ratified-in-practice across the tree:
``coverage`` in datasets/manifest.yaml means the *parsed usable extent*
(what the pack load surface — committed parser + partial-current-year
policy — actually yields from the pinned bytes), the ``make datasets``
flow refuses any manifest coverage that disagrees with parser output
(charts/datasets.py, pinned in test_dataset_pack_fetch.py), and the real
manifest's endpoints are pinned below the access year
(test_dataset_pack_manifest.py). The GISTEMP 2026 endpoint the finding
names is already corrected to 2025.

What is NOT yet honest is the range validator's failure mode when the
coverage contract cannot be evaluated at all: ``charts.spec.validate_spec``
silently SKIPS time-range containment when a referenced chart-pack
dataset's coverage block is missing, malformed, or unconvertible
(``_coverage_ce``/``_series_coverage_ce`` return None as a guard). A
spec over such an entry is accepted with NO range check — fail-open —
which is exactly the "validator accepts ranges the renderer cannot fill"
failure #52 exists to prevent, reachable whenever validate_spec is
handed a manifest mapping the write-time validator did not bless.

DECISION PINNED HERE (flagged for orchestrator ratification): a series
whose dataset coverage cannot be resolved to a CE extent must REFUSE the
spec, naming the dataset and its coverage — coverage is a load-bearing
input to range validation, so an unevaluable coverage is a refusal, not
a skipped check. NOTE the interaction with the #161 CAVEAT: the
``DatasetManifest`` form (whose DatasetRecord carries no coverage) would
refuse every explicitly-ranged spec under this contract; the implementer
must either carry coverage on DatasetRecord or bar that form from
ranged-spec validation — a ratification point, recorded in the red-phase
report.

Unit tier: pure validate_spec calls over synthetic inline manifests
(ADR-023: no real data files, no fetching, no network, no API key).
"""

from __future__ import annotations

from typing import Any

import pytest

from charts import spec as chartspec


def _syn_manifest() -> dict[str, Any]:
    """SYNTHETIC FIXTURE: minimal raw-mapping pack manifest (invented
    "widgets" datasets — no real values anywhere)."""
    return {
        "datasets": {
            "syn_widgets": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "year_ce"},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"first_year_ce": 1900, "last_year_ce": 2000},
            },
            "syn_widgets_old": {
                "permitted_context": "open",
                "in_chart_pack": True,
                "time_axis": {"unit": "years_bp", "present_ce": 1950},
                "variable": {"name": "widgets", "unit": "wd"},
                "coverage": {"oldest_bp": 12000, "youngest_bp": -40},
            },
        },
        "splice_pairs": [],
    }


def _line_spec(dataset: str = "syn_widgets", time_range=None) -> dict[str, Any]:
    """SYNTHETIC FIXTURE: minimal valid single-dataset line chart."""
    spec: dict[str, Any] = {
        "spec_version": "1.0.0",
        "chart_id": "syn-line",
        "chart_type": "line",
        "title": "Synthetic widgets over time",
        "time_range_ce": list(time_range or [1900, 2000]),
        "series": [
            {
                "id": "w",
                "label": "Widgets (wd)",
                "unit": "wd",
                "dataset": dataset,
                "scale_domain": [0, 120],
            }
        ],
    }
    return spec


def _coverage_refusals(spec: dict[str, Any], manifest: dict[str, Any]) -> list[Any]:
    with pytest.raises(chartspec.ChartSpecError) as excinfo:
        chartspec.validate_spec(spec, manifest)
    return [
        violation
        for violation in excinfo.value.violations
        if "coverage" in violation.reason.lower()
    ]


class TestUnevaluableCoverageFailsClosed:
    def test_control_spec_is_valid_with_well_formed_coverage(self):
        """The mutations below must be the only reason for a refusal."""
        assert chartspec.validate_spec(_line_spec(), _syn_manifest()) is None
        bp_spec = _line_spec(dataset="syn_widgets_old", time_range=[1900, 1990])
        bp_spec["time_axis"] = {"calendar": "CE", "convert_bp": True}
        assert chartspec.validate_spec(bp_spec, _syn_manifest()) is None

    def test_missing_coverage_block_refuses_a_ranged_spec(self):
        """A chart-pack dataset entry with NO coverage block cannot
        support range validation — an explicitly ranged spec over it must
        refuse (naming the dataset and coverage), never validate with the
        containment check silently skipped (#52 fail-closed)."""
        manifest = _syn_manifest()
        del manifest["datasets"]["syn_widgets"]["coverage"]
        refusals = _coverage_refusals(_line_spec(), manifest)
        assert refusals, (
            "a spec with time_range_ce over a dataset with no coverage block "
            "validated silently — range validation must fail closed, with a "
            "violation naming the dataset's missing/unusable coverage (#52)"
        )
        assert any("syn_widgets" in violation.reason for violation in refusals)

    def test_malformed_coverage_block_refuses_a_ranged_spec(self):
        """Coverage that cannot be read as numeric endpoints is the same
        unevaluable state as a missing block: refuse, do not skip."""
        manifest = _syn_manifest()
        manifest["datasets"]["syn_widgets"]["coverage"] = {
            "first_year_ce": "nineteen hundred",
            "last_year_ce": 2000,
        }
        refusals = _coverage_refusals(_line_spec(), manifest)
        assert refusals, (
            "a spec with time_range_ce over a dataset whose coverage endpoints "
            "are non-numeric validated silently — range validation must fail "
            "closed (#52)"
        )
        assert any("syn_widgets" in violation.reason for violation in refusals)

    def test_bp_coverage_without_present_ce_refuses_a_ranged_spec(self):
        """A years_bp coverage with no time_axis.present_ce cannot convert
        to CE, so a CE-ranged spec over it (even with convert_bp: true)
        has an unevaluable range check — refuse, do not skip."""
        manifest = _syn_manifest()
        del manifest["datasets"]["syn_widgets_old"]["time_axis"]["present_ce"]
        spec = _line_spec(dataset="syn_widgets_old", time_range=[1900, 1990])
        spec["time_axis"] = {"calendar": "CE", "convert_bp": True}
        refusals = _coverage_refusals(spec, manifest)
        assert refusals, (
            "a CE-ranged spec over a years_bp dataset whose coverage cannot "
            "convert (no present_ce) validated silently — range validation "
            "must fail closed (#52)"
        )
        assert any("syn_widgets_old" in violation.reason for violation in refusals)
