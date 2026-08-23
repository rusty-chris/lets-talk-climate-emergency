"""Gold-set coverage ledger generator (issue #20 — the no-silent-caps rule).

Renders ``evals/gold/COVERAGE.md`` deterministically from the two gold-set
YAMLs: composition counts, every ``blocked_on`` item with its reason, and
the chart-side fixture gap. The committed file must always equal a fresh
render (enforced by ``tests/unit/test_gold_sets.py``), so no coverage cap —
an item authored but not yet evaluable against today's corpus — can ever
be silent: blocking, unblocking or re-scoping an item forces a visible
diff in a committed, reviewed markdown file.

Also owns ``--write-snapshot``: refresh ``evals/gold/ingest_chunk_ids.txt``
from ``data/ingest/chunks.jsonl`` after a corpus/chunker change (run
``make ingest`` first).

Usage:
    python evals/scripts/gold_coverage.py                  # write COVERAGE.md
    python evals/scripts/gold_coverage.py --check          # verify committed
    python evals/scripts/gold_coverage.py --write-snapshot # refresh chunk ids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from evals import gold_selection

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "evals" / "gold"
QA_PATH = GOLD_DIR / "climate_qa.yaml"
CHARTS_PATH = GOLD_DIR / "chart_requests.yaml"
COVERAGE_PATH = GOLD_DIR / "COVERAGE.md"
SNAPSHOT_PATH = GOLD_DIR / "ingest_chunk_ids.txt"
CHUNKS_JSONL = REPO_ROOT / "data" / "ingest" / "chunks.jsonl"

QA_CATEGORY_ORDER = [
    "single_passage",
    "multi_passage",
    "no_answer",
    "adversarial",
    "severity",
    "voices_action",
    "targeted",
]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render_coverage() -> str:
    qa = _load(QA_PATH)
    charts = _load(CHARTS_PATH)
    qa_items = qa["items"]
    chart_items = charts["items"]

    lines: list[str] = []
    add = lines.append
    add("# Gold-set coverage ledger (issue #20)")
    add("")
    add("> GENERATED — do not edit by hand. Regenerate with")
    add("> `python evals/scripts/gold_coverage.py`; the meta-test")
    add("> `test_coverage_ledger_matches_generator` fails on any drift.")
    add("> This file is the no-silent-caps record: every gold item that")
    add("> cannot be fully evaluated against today's corpus/pack is listed")
    add("> below with its reason. Shrinking coverage silently is impossible")
    add("> without a diff here.")
    add("")
    add("## Climate-QA composition (DESIGN §6.1 as amended)")
    add("")
    add("| category | items | blocked | with gold chunk ids |")
    add("|---|---|---|---|")
    for category in QA_CATEGORY_ORDER:
        members = [item for item in qa_items if item["category"] == category]
        blocked = [item for item in members if item.get("blocked_on")]
        with_ids = [item for item in members if item.get("gold_chunk_ids")]
        add(f"| {category} | {len(members)} | {len(blocked)} | {len(with_ids)} |")
    total_blocked = [item for item in qa_items if item.get("blocked_on")]
    add(
        f"| **total** | **{len(qa_items)}** | **{len(total_blocked)}** | "
        f"**{len([i for i in qa_items if i.get('gold_chunk_ids')])}** |"
    )
    add("")
    smoke = [item["id"] for item in qa_items if item.get("smoke")]
    add(f"Smoke subset ({len(smoke)} items, the dev-iteration budget set): " + ", ".join(smoke))
    add("")
    add("## No-answer gate arithmetic (review findings #192/#193)")
    add("")
    no_answer = [item for item in qa_items if item["category"] == "no_answer"]
    calibration_rr = gold_selection.calibration_item_ids(qa_items)
    gate_rr = gold_selection.gate_item_ids(qa_items)
    canned = [i for i in no_answer if i["expected_route"] == "canned_out_of_scope"]
    n_gate = len(gate_rr)
    one_flake_pct = 100.0 * (n_gate - 1) / n_gate if n_gate else 0.0
    add(
        "Every no-answer item annotates `expected_route`; the reranker "
        "threshold calibration and the DESIGN §6.2 refusal release gate "
        "consume ONLY `retrieval_refusal` items (selection seam: "
        "`evals/gold_selection.py`). `canned_out_of_scope` items exercise "
        "the classifier's canned decline and are gated by the classifier's "
        "labelled query set, never by the reranker gate."
    )
    add("")
    add(
        f"- no_answer items: {len(no_answer)} "
        f"({len(no_answer) - len(canned)} retrieval_refusal + "
        f"{len(canned)} canned_out_of_scope)"
    )
    add(
        f"- release-gate subset (`gate` ∩ `retrieval_refusal`): {n_gate} items — "
        f"the '20-item no-answer gate subset' issue #21 expects. One flake is "
        f"{n_gate - 1}/{n_gate} = {one_flake_pct:.0f}%, so the strict '>90%' "
        "gate survives a single flake (critic finding 15's intent); two "
        "flakes fail it."
    )
    add(
        f"- calibration subset (`calibration` ∩ `retrieval_refusal`): "
        f"{len(calibration_rr)} items, disjoint from the gate subset — "
        "threshold-calibration items never grade the threshold they tuned."
    )
    add(
        "- cost note: the growth from 75 to "
        f"{len(qa_items)} climate-QA items adds only refusal-path items "
        "(~$0.001 each, no generation call), leaving the dev-cost-plan "
        "full-run estimate materially unchanged."
    )
    add("")
    add("## Blocked climate-QA items")
    add("")
    add("Corpus today: two ingested documents (nca5_ch2, esd_tipping_review).")
    add("Each item below is authored and schema-complete but waits on corpus")
    add("expansion; unblocking = assigning gold chunk ids (and, for severity,")
    add("the pending source passage) after the named ingest, then rerunning")
    add("this generator.")
    add("")
    for item in qa_items:
        if not item.get("blocked_on"):
            continue
        reason = " ".join(str(item["blocked_reason"]).split())
        add(f"- **{item['id']}** ({item['category']}; blocked on {item['blocked_on']}): {reason}")
    add("")
    add("## Chart gold set")
    add("")
    spec_items = [item for item in chart_items if item.get("expected") == "spec"]
    refusal_items = [item for item in chart_items if item.get("expected") == "refusal"]
    with_fixture = [item for item in spec_items if item.get("fixture")]
    add(
        f"- items: {len(chart_items)} (expected-spec {len(spec_items)}, "
        f"expected-refusal {len(refusal_items)})"
    )
    add(
        f"- expected-spec items with committed rendered-value fixtures: "
        f"{len(with_fixture)}/{len(spec_items)} "
        "(independent generator: evals/scripts/compute_chart_fixtures.py; "
        "synthetic data only)"
    )
    add("")
    add("### Chart-side gaps")
    add("")
    for item in chart_items:
        if not item.get("blocked_on"):
            continue
        reason = " ".join(str(item["blocked_reason"]).split())
        add(f"- **{item['id']}** (blocked on {item['blocked_on']}): {reason}")
    add("")
    add("## Standing caps (recorded, not silent)")
    add("")
    add(
        "- Gold chunk ids reference the CURRENT two-document ingest "
        "(snapshot: evals/gold/ingest_chunk_ids.txt). Chunk ids are "
        "content-hash based: any corpus or chunker change invalidates them "
        "loudly via the snapshot tests, never silently."
    )
    add(
        "- Real-pack chart fixtures (including the flagship) are excluded "
        "until issue #23's licence confirmations (review finding #117); the "
        "transform arithmetic is fixture-covered with synthetic data "
        "meanwhile."
    )
    add(
        "- Severity spot-audit by the project owner and second-pass peer "
        "review of item quality are process criteria recorded on the "
        "issue-20 PR, not encoded in these files."
    )
    add("")
    return "\n".join(lines)


def write_snapshot() -> None:
    ids = [
        json.loads(line)["chunk_id"]
        for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    header = (
        # First line carries the ADR-023 exemption marker
        # (ingestion.manifest.PROJECT_OPERATIONAL_DATA_MARKER): chunk ids
        # are a first-party record of this project's own ingest run
        # (content-hash identifiers, no source text), not licensed data.
        "# PROJECT-OPERATIONAL DATA — first-party record, no external licence\n"
        "# Snapshot of every chunk_id produced by `make ingest` over corpus/manifest.yaml\n"
        "# (documents: nca5_ch2, esd_tipping_review; ingested 2026-08-21).\n"
        "# Regenerate after any corpus or chunker change:\n"
        "#   make ingest && python evals/scripts/gold_coverage.py --write-snapshot\n"
        "# Unit tests resolve gold chunk_ids against this file; the integration\n"
        "# test asserts this file matches the live ingest output exactly.\n"
    )
    SNAPSHOT_PATH.write_text(header + "\n".join(ids) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)} ({len(ids)} chunk ids)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed COVERAGE.md")
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="refresh evals/gold/ingest_chunk_ids.txt from data/ingest/chunks.jsonl",
    )
    args = parser.parse_args(argv)
    if args.write_snapshot:
        write_snapshot()
        return 0
    rendered = render_coverage()
    if args.check:
        if COVERAGE_PATH.read_text(encoding="utf-8") != rendered:
            print("COVERAGE.md does NOT match a fresh render", file=sys.stderr)
            return 1
        print("COVERAGE.md matches a fresh render")
        return 0
    COVERAGE_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {COVERAGE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
