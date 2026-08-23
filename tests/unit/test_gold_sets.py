"""Meta-tests making the issue #20 gold sets structurally trustworthy.

The gold data IS this issue's deliverable, so these tests ship green with
it (ORCHESTRATION.md: the data-authoring issue's TDD is meta-testing).
They pin: item schema, the DESIGN §6.1 amended composition (so the set
cannot silently shrink), calibration/gate disjointness, severity
annotation completeness, the targeted items, chart-gold spec-or-refusal
exclusivity with real-validator legality, fixture presence + tolerances,
the fixture generator's independence from charts/ (the non-tautology
guarantee), blocked-item reasons and the committed no-silent-caps
coverage ledger. Chunk-id resolution against the LIVE ingest runs in
tests/integration/test_gold_chunk_ids.py; here gold ids resolve against
the committed snapshot so every PR run still catches a typo or a stale
id without network or Docling.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from charts import spec as chartspec
from evals import gold_selection
from evals.scripts import compute_chart_fixtures, gold_coverage

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "evals" / "gold"
QA_PATH = GOLD_DIR / "climate_qa.yaml"
CHARTS_PATH = GOLD_DIR / "chart_requests.yaml"
PACK_FIXTURE_PATH = GOLD_DIR / "chart_pack_fixture.yaml"
FIXTURES_PATH = GOLD_DIR / "chart_fixtures.json"
SNAPSHOT_PATH = GOLD_DIR / "ingest_chunk_ids.txt"
COVERAGE_PATH = GOLD_DIR / "COVERAGE.md"
SEVERITY_RUBRIC_PATH = GOLD_DIR / "severity-rubric.md"
FIXTURE_SCRIPT = REPO_ROOT / "evals" / "scripts" / "compute_chart_fixtures.py"
FLAGSHIP_SPEC_PATH = REPO_ROOT / "charts" / "spike" / "flagship_spec.json"
REAL_DATASETS_MANIFEST = REPO_ROOT / "datasets" / "manifest.yaml"

QA_CATEGORIES = {
    "single_passage",
    "multi_passage",
    "no_answer",
    "adversarial",
    "severity",
    "voices_action",
    "targeted",
}
SEVERITY_LEADS = {"reassuring", "serious", "emergency-level"}
SEVERITY_BAITS = {"soft-pedal", "inflation", "neutral"}
TARGETED_TAGS = {"packham", "sensitivity_assessed_range", "carbon_brief_paraphrase"}
# Review finding #192: every no-answer item declares which refusal
# mechanism it exercises — the classifier's canned out-of-scope path
# (retrieval never runs, no reranker score exists) or the reranker
# refusal gate (the mechanism the calibration/gate machinery certifies).
EXPECTED_ROUTES = {"canned_out_of_scope", "retrieval_refusal"}


@pytest.fixture(scope="module")
def qa() -> dict:
    return yaml.safe_load(QA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qa_items(qa) -> list[dict]:
    return qa["items"]


@pytest.fixture(scope="module")
def charts_gold() -> dict:
    return yaml.safe_load(CHARTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chart_items(charts_gold) -> list[dict]:
    return charts_gold["items"]


@pytest.fixture(scope="module")
def pack_fixture_manifest() -> dict:
    return yaml.safe_load(PACK_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chart_fixtures() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snapshot_chunk_ids() -> set[str]:
    lines = SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


# ---------------------------------------------------------------------------
# 1. Item schema
# ---------------------------------------------------------------------------


def test_gold_item_schema_valid(qa, qa_items):
    """Every QA item has the required fields for its category; unique ids;
    synthetic marker present (no real user text ever enters the repo)."""
    assert "SYNTHETIC" in qa["marker"]
    seen_ids: set[str] = set()
    for item in qa_items:
        item_id = item["id"]
        assert item_id not in seen_ids, f"duplicate gold id {item_id}"
        seen_ids.add(item_id)
        assert item["category"] in QA_CATEGORIES, item_id
        assert isinstance(item["question"], str) and item["question"].strip(), item_id
        assert item["expected_behaviour"] in {"answer", "refusal"}, item_id
        if item["category"] == "no_answer":
            assert item["expected_behaviour"] == "refusal", item_id
            assert item.get("subset") in {"calibration", "gate"}, item_id
        else:
            assert item["expected_behaviour"] == "answer", item_id
            assert "subset" not in item, f"{item_id}: subset is a no_answer-only field"
            assert "expected_route" not in item, (
                f"{item_id}: expected_route is a no_answer-only field (#192)"
            )
        # chunk-id requirements: single/multi passage always carry them;
        # blocked items in other categories may defer them.
        chunk_ids = item.get("gold_chunk_ids")
        if item["category"] == "single_passage":
            assert chunk_ids and len(chunk_ids) == 1, item_id
        elif item["category"] == "multi_passage":
            assert chunk_ids and len(chunk_ids) >= 2, item_id
        elif item["category"] == "no_answer":
            assert not chunk_ids, f"{item_id}: refusal items carry no gold chunks"
        else:
            assert chunk_ids or item.get("blocked_on"), (
                f"{item_id}: an answerable item must carry gold chunk ids or an "
                "explicit blocked_on marker (no silent gaps)"
            )


def test_gold_chart_item_schema_valid(charts_gold, chart_items):
    assert "SYNTHETIC" in charts_gold["marker"]
    seen: set[str] = set()
    for item in chart_items:
        assert item["id"] not in seen, f"duplicate chart gold id {item['id']}"
        seen.add(item["id"])
        assert isinstance(item["request"], str) and item["request"].strip(), item["id"]
        assert item.get("manifest") in {"synthetic", "real"}, item["id"]


# ---------------------------------------------------------------------------
# 2. Composition — the §6.1 amended mix, so the set cannot silently shrink
# ---------------------------------------------------------------------------


def test_gold_category_counts_match_design_mix(qa_items):
    counts = {category: 0 for category in QA_CATEGORIES}
    for item in qa_items:
        counts[item["category"]] += 1
    # no_answer is 39 after review finding #193: 30 retrieval_refusal
    # items (10 calibration + a 20-item gate subset — the n issue #21's
    # ">90% on the 20-item no-answer gate subset" wording expects, and
    # the smallest round n where a single flake still clears a strict
    # >90% gate: 19/20 = 95%) plus the 9 canned_out_of_scope items kept
    # from the original set (finding #192 routes).
    assert counts == {
        "single_passage": 15,
        "multi_passage": 10,
        "no_answer": 39,
        "adversarial": 7,
        "severity": 15,
        "voices_action": 5,
        "targeted": 3,
    }
    assert len(qa_items) == 94


def test_multi_passage_items_declare_recall_semantics(qa_items):
    """Review finding #196: scoring semantics lived in one item's
    free-text note (qa-mp-06), which prescribed any-gold-chunk-retrieved
    as a harness-wide default — diluting the category (1-of-3 gold
    chunks would score full recall) and silently overriding sibling
    items authored on the premise that one chunk is insufficient
    (qa-mp-05, qa-mp-09). Every multi_passage item now declares
    `recall_semantics` itself — `all_gold` (Recall@8 must surface every
    gold chunk) or `any_gold` (any gold chunk suffices) — so the #21
    harness consumes structure, not prose; no harness-side default
    exists. `any_gold` stays a stated minority or the category stops
    measuring synthesis retrieval. Other categories must not carry the
    field (their recall handling is category-level, not per-item)."""
    multi = [i for i in qa_items if i["category"] == "multi_passage"]
    assert multi
    semantics = {}
    for item in multi:
        assert item.get("recall_semantics") in {"all_gold", "any_gold"}, (
            f"{item['id']}: multi_passage item must declare recall_semantics "
            "as all_gold | any_gold (finding #196)"
        )
        semantics[item["id"]] = item["recall_semantics"]
    any_gold = [item_id for item_id, kind in semantics.items() if kind == "any_gold"]
    assert len(any_gold) <= 2, f"any_gold must stay a stated minority; got {any_gold}"
    # The items whose own rationales demand every chunk keep all_gold.
    assert semantics["qa-mp-05"] == "all_gold"
    assert semantics["qa-mp-09"] == "all_gold"
    for item in qa_items:
        if item["category"] != "multi_passage":
            assert "recall_semantics" not in item, (
                f"{item['id']}: recall_semantics is a multi_passage-only field"
            )


def test_chart_gold_set_has_fifteen_items(chart_items):
    assert len(chart_items) == 15
    expected = {item.get("expected") for item in chart_items}
    assert expected == {"spec", "refusal"}


# ---------------------------------------------------------------------------
# 3. Calibration / gate disjointness (the §6.1 amendment: threshold-
#    calibration items never leak into the release-gate metric)
# ---------------------------------------------------------------------------


def test_no_answer_items_annotate_expected_route(qa_items):
    """Review finding #192: the no-answer subset conflates two refusal
    mechanisms. Every no-answer item must record which one it exercises:
    ``canned_out_of_scope`` (the classifier's canned decline in
    rag/query.py::route_classification — retrieval never runs, no
    reranker score exists) or ``retrieval_refusal`` (the reranker
    refusal gate the subset exists to calibrate and gate)."""
    for item in qa_items:
        if item["category"] != "no_answer":
            continue
        assert item.get("expected_route") in EXPECTED_ROUTES, (
            f"{item['id']}: no_answer item must annotate expected_route as "
            f"one of {sorted(EXPECTED_ROUTES)} (finding #192)"
        )


def test_no_answer_calibration_and_gate_items_disjoint(qa_items):
    calibration = {i["id"] for i in qa_items if i.get("subset") == "calibration"}
    gate = {i["id"] for i in qa_items if i.get("subset") == "gate"}
    # Finding #193 composition: 15 calibration (10 retrieval_refusal +
    # 5 canned) / 24 gate (20 retrieval_refusal + 4 canned).
    assert len(calibration) == 15
    assert len(gate) == 24
    assert calibration.isdisjoint(gate)
    # Disjointness by question text too — a re-worded duplicate would
    # calibrate the threshold on a gate item in all but id.
    calibration_questions = {i["question"] for i in qa_items if i.get("subset") == "calibration"}
    gate_questions = {i["question"] for i in qa_items if i.get("subset") == "gate"}
    assert calibration_questions.isdisjoint(gate_questions)


def test_gate_subset_survives_single_flake(qa_items):
    """Review finding #193 (critic finding 15's intent): the release
    gate's n must tolerate one flake under the strict DESIGN §6.2
    '>90% refusal' comparison — at the merged set's gate n=10, one
    borderline reranker score gave 9/10 = 90%, failing '>90%' with
    zero tolerance, the exact brittleness the ~20-item amendment was
    introduced to fix. The gate counts ONLY retrieval_refusal gate
    items (finding #192), so the arithmetic is asserted on that
    selection; issue #21's '20-item no-answer gate subset' wording is
    satisfied by exactly this selection."""
    gate_ids = gold_selection.gate_item_ids(qa_items)
    n = len(gate_ids)
    assert n >= 20, f"gate needs n>=20 retrieval_refusal items, found {n}"
    # One miss still clears the strict > 0.9 gate comparison.
    assert (n - 1) / n > 0.9, f"a single flake fails the >90% gate at n={n}"
    calibration_ids = gold_selection.calibration_item_ids(qa_items)
    assert len(calibration_ids) >= 10, (
        "threshold calibration needs >=10 retrieval_refusal items "
        f"(DESIGN §6.1), found {len(calibration_ids)}"
    )
    assert set(calibration_ids).isdisjoint(gate_ids)


# ---------------------------------------------------------------------------
# 4. Severity annotations — complete on every severity item
# ---------------------------------------------------------------------------


def test_severity_items_carry_ordinal_annotation_and_source_passage(qa_items):
    severity_items = [i for i in qa_items if i["category"] == "severity"]
    assert len(severity_items) == 15
    baits_seen = set()
    leads_seen = set()
    for item in severity_items:
        annotation = item.get("severity")
        assert annotation, f"{item['id']}: severity item without annotation"
        assert annotation["expected_lead"] in SEVERITY_LEADS, item["id"]
        assert annotation["bait"] in SEVERITY_BAITS, item["id"]
        leads_seen.add(annotation["expected_lead"])
        baits_seen.add(annotation["bait"])
        passage = annotation["source_passage"]
        if item.get("blocked_on"):
            assert passage.get("pending_doc_id") and passage.get("description"), (
                f"{item['id']}: blocked severity item must name the pending "
                "source document and describe the passage"
            )
        else:
            assert passage.get("chunk_id") and passage.get("quote"), (
                f"{item['id']}: severity annotation must carry the source "
                "chunk and a verbatim quote (DESIGN §6.1)"
            )
            # The severity-retrieval metric (DESIGN §6.2) needs the
            # headline-severity chunk among the item's gold chunks.
            assert passage["chunk_id"] in (item.get("gold_chunk_ids") or []), item["id"]
    # Both bait directions must be represented (§6.2: the judge flags
    # soft-pedal AND inflation symmetrically), and all three leads used.
    assert baits_seen == SEVERITY_BAITS
    assert leads_seen == SEVERITY_LEADS


def test_severity_leads_justify_the_gate_maths(qa_items):
    """The zero-two-level-error gate is vacuous if every gold lead is
    'serious' — but review finding #195 showed the old exact floor
    (>=2 emergency-level, >=1 reassuring, with the set sitting at
    exactly 2 and 1) meant one honest rubric-driven relabel broke the
    meta-test instead of prompting a considered rebalance. Ranges keep
    the gate non-vacuous (at least one gold at each extreme) while a
    defensible re-adjudication under evals/gold/severity-rubric.md
    stays possible; the ceiling keeps 'serious' the calibrated majority
    the mission demands (no drift into doom-inflation or reassurance)."""
    leads = [i["severity"]["expected_lead"] for i in qa_items if i["category"] == "severity"]
    emergency = leads.count("emergency-level")
    reassuring = leads.count("reassuring")
    assert 1 <= emergency <= 4
    assert 1 <= reassuring <= 4
    assert leads.count("serious") > emergency + reassuring


def test_severity_scale_definitions_present(qa):
    """Review finding #195: the 3-point scale was defined only by
    example — no written boundary existed for the #21 judge prompt or
    the owner's human audit to reference. The rubric file must define
    each level with decision rules and at least two worked examples,
    the gold set must point at it, and the rubric must record that the
    #21 judge prompt embeds the definitions verbatim (the follow-on
    assertion lands with #21's harness)."""
    assert SEVERITY_RUBRIC_PATH.exists(), (
        "evals/gold/severity-rubric.md is missing (finding #195): the "
        "severity judge and the human audit need written level definitions"
    )
    rubric = SEVERITY_RUBRIC_PATH.read_text(encoding="utf-8")
    for level in sorted(SEVERITY_LEADS):
        heading = f"## {level}"
        assert heading in rubric, f"rubric lacks a definition section for {level!r}"
        section = rubric.split(heading, 1)[1].split("\n## ", 1)[0]
        assert "Decision rule" in section, f"{level}: no decision rules"
        assert section.count("Worked example") >= 2, f"{level}: fewer than two worked examples"
    assert "verbatim" in rubric, "rubric must bind the #21 judge prompt to embed it verbatim"
    qa_text = QA_PATH.read_text(encoding="utf-8")
    assert "severity-rubric.md" in qa_text, (
        "climate_qa.yaml must reference the severity rubric (finding #195)"
    )


# ---------------------------------------------------------------------------
# 5. Targeted items
# ---------------------------------------------------------------------------


def test_targeted_items_present(qa_items):
    targeted = {i["targeted"]: i for i in qa_items if i["category"] == "targeted"}
    assert set(targeted) == TARGETED_TAGS
    for item in targeted.values():
        assert item.get("expectations"), f"{item['id']}: targeted item needs expectations"
        assert item.get("blocked_on"), (
            f"{item['id']}: every targeted item currently carries a recorded "
            "corpus dependency (issue #20 constraint)"
        )
    # The evaluable halves carry real chunk ids today.
    assert targeted["packham"].get("gold_chunk_ids")
    assert targeted["sensitivity_assessed_range"].get("gold_chunk_ids")


# ---------------------------------------------------------------------------
# 6-7. Chart golds: spec XOR refusal; validator-legal; fixtures + tolerances
# ---------------------------------------------------------------------------


def test_chart_golds_have_spec_or_refusal(chart_items, pack_fixture_manifest):
    for item in chart_items:
        has_spec = "spec" in item
        has_refusal = "refusal" in item
        assert item["expected"] in {"spec", "refusal"}, item["id"]
        if item["expected"] == "spec":
            assert has_spec and not (item.get("refusal")), (
                f"{item['id']}: exactly one of expected spec / expected refusal"
            )
            assert item["manifest"] == "synthetic"
            # The gold answer itself must be legal in the frozen vocabulary
            # against the fixture pack manifest — otherwise the case is wrong.
            chartspec.validate_spec(item["spec"], pack_fixture_manifest)
        else:
            assert has_refusal and not has_spec, (
                f"{item['id']}: exactly one of expected spec / expected refusal"
            )


def test_refusal_golds_match_planner_nearest_semantics(chart_items):
    """Review finding #194: the planner's nearest_available_datasets is
    'never empty while the catalogue has datasets' — the ADR-021 refusal
    always names nearest datasets, so a gold `nearest_dataset_first:
    null` has no semantics a harness can check. Every synthetic-manifest
    refusal gold must pin the deterministic first entry of the planner's
    nearest list, computed against the fixture pack's catalogue view (a
    pure function — no model, no network)."""
    from charts import planner

    catalogue = planner.build_dataset_catalogue(PACK_FIXTURE_PATH)
    checked = 0
    for item in chart_items:
        if item["expected"] != "refusal" or item["manifest"] != "synthetic":
            continue
        refusal = item["refusal"]
        nearest = planner.nearest_available_datasets(refusal["requested_data"], catalogue)
        assert nearest, item["id"]
        assert refusal.get("nearest_dataset_first") == nearest[0], (
            f"{item['id']}: gold nearest_dataset_first "
            f"{refusal.get('nearest_dataset_first')!r} != the planner's "
            f"deterministic nearest-first {nearest[0]!r} (finding #194)"
        )
        checked += 1
    assert checked == 3  # chart-07, chart-08, chart-13


def test_chart_gold_specs_legal_with_fixture_extents(
    chart_items, pack_fixture_manifest, chart_fixtures
):
    """Render-mode legality: each gold spec still validates when the
    fixture-derived data extents are supplied (scale-domain containment,
    zoom-out bound, zero-exclusion disclosures — review #48/#129)."""
    for item in chart_items:
        if item["expected"] != "spec":
            continue
        body = chart_fixtures["fixtures"][item["fixture"]]
        extents = {
            series_id: (
                min(point[1] for point in entry["points"]),
                max(point[1] for point in entry["points"]),
            )
            for series_id, entry in body["series"].items()
        }
        chartspec.validate_spec(item["spec"], pack_fixture_manifest, data_extents=extents)


def test_chart_fixture_values_present_with_tolerances(chart_items, chart_fixtures):
    fixtures = chart_fixtures["fixtures"]
    for item in chart_items:
        if item["expected"] != "spec":
            continue
        fixture_id = item.get("fixture")
        assert fixture_id, f"{item['id']}: expected-spec item without a fixture id"
        body = fixtures[fixture_id]
        assert body["tolerance_relative"] in {1e-9, 1e-6}
        assert body["transform_kind"] in {"pass-through", "post-transform"}
        # DESIGN §6.2: 1e-9 pass-through, 1e-6 post-transform.
        expected_tolerance = 1e-9 if body["transform_kind"] == "pass-through" else 1e-6
        assert body["tolerance_relative"] == expected_tolerance
        spec_series_ids = {series["id"] for series in item["spec"]["series"]}
        assert set(body["series"]) == spec_series_ids, item["id"]
        for series_id, entry in body["series"].items():
            assert entry["n_points"] == len(entry["points"]) > 0, (
                f"{item['id']}/{series_id}: empty fixture"
            )
            for point in entry["points"]:
                assert len(point) == 2, f"{item['id']}/{series_id}"
    # No orphaned fixtures either — every committed fixture is referenced.
    referenced = {item["fixture"] for item in chart_items if item["expected"] == "spec"}
    assert referenced == set(fixtures)


def test_chart_fixtures_match_independent_recompute(chart_fixtures):
    """The committed fixture file equals a fresh recompute by the
    independent generator — hand-edited values cannot drift in silently."""
    assert chart_fixtures == compute_chart_fixtures.compute_fixtures()


# ---------------------------------------------------------------------------
# 8. The non-tautology guarantee: fixture generator imports nothing from
#    the pipeline under test
# ---------------------------------------------------------------------------


def test_fixture_script_imports_nothing_from_charts():
    tree = ast.parse(FIXTURE_SCRIPT.read_text(encoding="utf-8"))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending.extend(
                alias.name
                for alias in node.names
                if alias.name == "charts" or alias.name.startswith("charts.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "charts" or module.startswith("charts.")):
                offending.append(module)
    assert not offending, (
        "evals/scripts/compute_chart_fixtures.py must not import from charts/ "
        f"(IMPLEMENTATION.md §5 non-tautology guarantee); found {offending}"
    )
    # Belt and braces: no attribute of the imported module is a charts/*
    # module object (an aliased import would evade the name-based AST scan).
    import types

    charts_attrs = [
        name
        for name, value in vars(compute_chart_fixtures).items()
        if isinstance(value, types.ModuleType)
        and (value.__name__ == "charts" or value.__name__.startswith("charts."))
    ]
    assert not charts_attrs, charts_attrs


# ---------------------------------------------------------------------------
# Flagship (#117): spec-validation + refusal-of-commitment
# ---------------------------------------------------------------------------


def test_flagship_item_is_spec_validation_plus_refusal_of_commitment(chart_items):
    (flagship_item,) = [i for i in chart_items if i.get("manifest") == "real"]
    assert flagship_item["expected"] == "refusal"
    assert flagship_item["blocked_on"] == "issue-23-licence-confirmations"
    flagship_meta = flagship_item["flagship"]
    spec_path = REPO_ROOT / flagship_meta["spec_path"]
    assert spec_path == FLAGSHIP_SPEC_PATH

    flagship_spec = json.loads(FLAGSHIP_SPEC_PATH.read_text(encoding="utf-8"))
    # Half 1 — spec-validation: the committed flagship spec remains
    # structurally valid against the frozen #15 schema.
    structural = chartspec._structural_violations(flagship_spec)
    assert structural == [], [f"{v.path}: {v.reason}" for v in structural]

    # Half 2 — refusal-of-commitment: the REAL manifest refuses to commit
    # to rendering it today, naming issue #23 on exactly the blocked pairs.
    real_manifest = yaml.safe_load(REAL_DATASETS_MANIFEST.read_text(encoding="utf-8"))
    with pytest.raises(chartspec.ChartSpecError) as excinfo:
        chartspec.validate_spec(flagship_spec, real_manifest)
    violations = excinfo.value.violations
    assert {v.path.split(".")[-1] for v in violations} == {"splice_pair_id"}
    assert "issue #23" in str(excinfo.value)
    expected_pairs = set(flagship_item["refusal"]["blocked_pairs"])
    named_pairs = {
        pair_id for pair_id in expected_pairs if any(pair_id in v.reason for v in violations)
    }
    assert named_pairs == expected_pairs == {"co2_10k", "temp_10k"}


# ---------------------------------------------------------------------------
# Blocked items carry reasons; the coverage ledger is current
# ---------------------------------------------------------------------------


def test_blocked_items_carry_reasons(qa_items, chart_items):
    for item in [*qa_items, *chart_items]:
        if item.get("blocked_on"):
            reason = item.get("blocked_reason", "")
            assert isinstance(reason, str) and len(reason.strip()) >= 40, (
                f"{item['id']}: blocked_on requires a substantive blocked_reason "
                "(the no-silent-caps rule)"
            )
        else:
            assert "blocked_reason" not in item, f"{item['id']}: blocked_reason without blocked_on"


def test_coverage_ledger_matches_generator():
    committed = COVERAGE_PATH.read_text(encoding="utf-8")
    assert committed == gold_coverage.render_coverage(), (
        "evals/gold/COVERAGE.md is stale — regenerate with `python evals/scripts/gold_coverage.py`"
    )


# ---------------------------------------------------------------------------
# Chunk-id resolution against the committed snapshot (PR-time; the live
# ingest check is tests/integration/test_gold_chunk_ids.py)
# ---------------------------------------------------------------------------


def test_gold_chunk_ids_resolve_against_committed_snapshot(qa_items, snapshot_chunk_ids):
    assert len(snapshot_chunk_ids) > 100  # two real documents' worth
    for item in qa_items:
        for chunk_id in item.get("gold_chunk_ids") or []:
            assert chunk_id in snapshot_chunk_ids, (
                f"{item['id']}: gold chunk id {chunk_id} not in the committed "
                "ingest snapshot (evals/gold/ingest_chunk_ids.txt)"
            )
        severity = item.get("severity")
        if severity and severity["source_passage"].get("chunk_id"):
            assert severity["source_passage"]["chunk_id"] in snapshot_chunk_ids, item["id"]


# ---------------------------------------------------------------------------
# Smoke subset (the issue #20 budget-cap comment)
# ---------------------------------------------------------------------------


def test_smoke_subset_is_ten_unblocked_items(qa_items):
    smoke = [i for i in qa_items if i.get("smoke")]
    assert len(smoke) == 10
    assert all(not i.get("blocked_on") for i in smoke)
    # Spread across behaviours: at least one refusal from each no-answer
    # subset, at least one adversarial and two severity items.
    subsets = {i.get("subset") for i in smoke if i["category"] == "no_answer"}
    assert subsets == {"calibration", "gate"}
    assert any(i["category"] == "adversarial" for i in smoke)
    assert sum(1 for i in smoke if i["category"] == "severity") >= 2


# ---------------------------------------------------------------------------
# Synthetic-fixture hygiene (IMPLEMENTATION.md §5 marker rule)
# ---------------------------------------------------------------------------


def test_synthetic_markers_present(pack_fixture_manifest):
    assert "SYNTHETIC FIXTURE" in pack_fixture_manifest["marker"]
    data_dir = GOLD_DIR / "synthetic_data"
    csv_files = sorted(data_dir.glob("*.csv"))
    assert len(csv_files) == 5
    for path in csv_files:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert "SYNTHETIC FIXTURE" in first_line, path.name
