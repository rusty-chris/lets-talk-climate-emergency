#!/usr/bin/env python3
"""Publish the issues/ directory to a GitHub repo as labels, milestones and issues.

Usage:
    python3 scripts/publish_issues.py <owner>/<repo> [--dry-run]

Requirements: `gh` CLI authenticated with repo scope. Run against a FRESH repo
(no existing issues): the files' cross-references (#1..#23) assume issue numbers
are assigned sequentially from 1 in file order.
"""

import re
import subprocess
import sys
from pathlib import Path

LABELS = {
    "phase-0": "d4c5f9",
    "phase-1": "0e8a16",
    "phase-1.5": "fbca04",
    "spike": "bfdadc",
    "ingestion": "1d76db",
    "licensing": "b60205",
    "rag": "5319e7",
    "charts": "006b75",
    "ui": "e99695",
    "evals": "0052cc",
    "infra": "c2e0c6",
    "voices": "f9d0c4",
}

MILESTONES = [
    "Phase 0 — Spike & de-risk",
    "Phase 1 — MVP",
    "Phase 1.5 — Permissions",
]


def sh(args, dry, capture=False):
    print("+", " ".join(args))
    if dry:
        return ""
    result = subprocess.run(args, check=True, capture_output=capture, text=True)
    return result.stdout if capture else ""


def parse(path):
    text = path.read_text()
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        sys.exit(f"{path}: missing front matter")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, match.group(2).strip()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    dry = "--dry-run" in sys.argv

    for name, color in LABELS.items():
        sh(["gh", "label", "create", name, "--repo", repo, "--color", color,
            "--force"], dry)

    milestone_numbers = {}
    for idx, title in enumerate(MILESTONES, start=1):
        sh(["gh", "api", f"repos/{repo}/milestones", "-f", f"title={title}"],
           dry)
        milestone_numbers[title] = idx

    files = sorted(Path(__file__).resolve().parent.parent.glob("issues/*.md"))
    for path in files:
        meta, body = parse(path)
        args = ["gh", "issue", "create", "--repo", repo,
                "--title", meta["title"], "--body", body]
        for label in meta.get("labels", "").split(","):
            if label.strip():
                args += ["--label", label.strip()]
        if meta.get("milestone"):
            args += ["--milestone", meta["milestone"]]
        sh(args, dry)


if __name__ == "__main__":
    main()
