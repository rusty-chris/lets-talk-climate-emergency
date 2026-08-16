"""Link-integrity test for the README "Techniques" section (issue #55).

Every intra-repo link in the Techniques section (including its "What we
deliberately don't do" subsection) must resolve: the path part to a file
that exists in the repo, and the ``#fragment`` part to a heading in the
target file. Anchors are checked with the same slug algorithm GitHub uses
when it generates heading anchors: render inline markdown to plain text,
lowercase, strip everything that is not a word character, space, or
hyphen, replace each space with a hyphen, and suffix ``-1``, ``-2``, ...
onto repeated slugs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_INLINE_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _github_slug(heading: str) -> str:
    """Slug a single heading the way GitHub's anchor generator does."""
    # Inline markdown renders away before slugging: code ticks and
    # emphasis markers vanish, links contribute only their text.
    text = _INLINE_LINK_RE.sub(r"\1", heading)
    text = text.replace("`", "").replace("*", "")
    text = text.lower()
    # Keep word characters (letters, digits, underscore), spaces, and
    # hyphens; drop all other punctuation and symbols.
    text = re.sub(r"[^\w\- ]", "", text)
    # Every space becomes a hyphen -- GitHub does not collapse runs.
    return text.replace(" ", "-")


def _github_anchors(markdown: str) -> set[str]:
    """All heading anchors GitHub would generate for a markdown document."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    in_code_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        slug = _github_slug(match.group(2))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _techniques_section() -> str:
    """The README's Techniques section, from its H2 to the next H2."""
    text = README.read_text()
    match = re.search(r"^## Techniques\s*$", text, flags=re.MULTILINE)
    assert match is not None, "README.md has no '## Techniques' section (issue #55)"
    rest = text[match.end() :]
    next_h2 = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest if next_h2 is None else rest[: next_h2.start()]


def test_readme_technique_links_resolve() -> None:
    """Every intra-repo link in the Techniques section resolves."""
    section = _techniques_section()
    targets = _LINK_RE.findall(section)
    assert targets, "the Techniques section contains no links"
    broken: list[str] = []
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, fragment = target.partition("#")
        target_file = README if not path_part else REPO_ROOT / path_part
        if not target_file.is_file():
            broken.append(f"{target} (file missing)")
            continue
        if fragment and fragment not in _github_anchors(target_file.read_text()):
            broken.append(f"{target} (no such anchor)")
    assert not broken, "unresolvable links in the Techniques section:\n" + "\n".join(broken)
