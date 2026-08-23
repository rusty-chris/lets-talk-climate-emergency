"""Docker build-context hygiene (review finding #30).

The Dockerfile ends with `COPY . .`, so anything not excluded by
`.dockerignore` is baked into the api/ui image layers. Local secrets
(`.env`, `.streamlit/secrets.toml`), agent worktrees (`.claude/`) and
private prose (`letters/`) must never reach an image: images outlive
containers, get cached, saved and eventually pushed to a registry (#22).

Pure text tests, unit tier: they parse `.dockerignore` (and cross-check
`.gitignore`) without needing Docker.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dockerignore_patterns() -> set[str]:
    lines = (REPO_ROOT / ".dockerignore").read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_dockerignore_covers_secret_and_private_paths() -> None:
    """Every secret-bearing or private path is excluded from the build context.

    - `.env` / `.env.*`: local env files (the natural home of
      ANTHROPIC_API_KEY once #14 lands) — a key in an image layer is a
      leak waiting for its distribution channel.
    - `.streamlit/`: covers `.streamlit/secrets.toml`.
    - `.claude/`: the main checkout carries full agent worktree copies of
      the repo under `.claude/worktrees/` (the declared parallel-agent
      workflow) — N repo copies in every build context.
    - `letters/`: draft permission letters under the owner's name
      (ORCHESTRATION.md stop-and-ask item) — private prose with no
      business inside a service image.
    """
    patterns = _dockerignore_patterns()
    missing = {".env", ".env.*", ".streamlit/", ".claude/", "letters/"} - patterns
    assert not missing, f".dockerignore is missing exclusions for: {sorted(missing)}"


def test_dockerignore_covers_gitignored_secret_patterns() -> None:
    """The secret patterns `.gitignore` anticipates are also dockerignored.

    `.gitignore` and `.dockerignore` must not drift apart: a path
    sensitive enough to keep out of git history is sensitive enough to
    keep out of image layers. Guards the specific secret-bearing entries
    in `.gitignore` (env files and Streamlit secrets).
    """
    gitignore = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith(("#", "!"))
    }
    secret_gitignore_patterns = {
        p for p in gitignore if p.startswith(".env") or "secrets" in p.lower()
    }
    assert secret_gitignore_patterns, ".gitignore lost its secret-file entries entirely?"

    docker_patterns = _dockerignore_patterns()

    def covered(pattern: str) -> bool:
        if pattern in docker_patterns:
            return True
        # A directory exclusion covers everything beneath it.
        return any(d.endswith("/") and pattern.startswith(d) for d in docker_patterns)

    uncovered = {p for p in secret_gitignore_patterns if not covered(p)}
    assert not uncovered, (
        f"secret patterns in .gitignore lack a covering .dockerignore entry: {sorted(uncovered)}"
    )
