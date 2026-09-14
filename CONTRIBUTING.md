# Contributing / version-control protocol

> **Status:** defaults adopted 2026-08-16 during an autonomous design session; confirm or amend with Chris and update this file.

- **Trunk:** `main` is the main development line. Design-phase work commits directly to `main`; once implementation starts, use short-lived feature branches.
- **Branching:** one topic branch per GitHub issue (`issue-<n>-<slug>`), branched off `main`.
- **Commit cadence:** one atomic (single logical) change per commit, committed frequently, with a verbose why-focused message (concise imperative subject + body explaining what changed and why). No squashing of granular history unless asked.
- **Merging:** feature branches merge to `main` via PR when the issue's acceptance criteria pass. No protected-branch approvals needed yet (solo project); revisit when collaborators join. **On merge, delete the head branch** — locally (`git branch -d <branch>`) and on the remote (GitHub's *Automatically delete head branches* setting does this for origin). Any worktree for a merged branch is removed (`git worktree remove --force`), never left behind. `scripts/git-cleanup.sh` sweeps up anything missed; run it at the start and end of each session.
- **Remote:** GitHub under the `rusty-chris` account; push `main` after each work session at minimum.
- **Protected:** none yet.
