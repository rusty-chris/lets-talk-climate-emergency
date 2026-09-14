#!/usr/bin/env bash
#
# git-cleanup.sh — idempotent, safe teardown of merged agent worktrees & branches.
#
# The autonomous build (ORCHESTRATION.md §Parallelism) spins up one isolated
# worktree + `issue-<n>` branch per agent but historically had no teardown step,
# so hundreds of merged workspaces piled up. This script reclaims that cruft
# *safely*: it only ever touches work that is already merged into the trunk.
# It uses `git branch -d` (never -D), `--merged`, and `git merge-base
# --is-ancestor` as gates, so it can NEVER drop unmerged work. Anything not
# merged is left in place and reported for a human to look at.
#
# Run it at session start and end. It is a near-no-op on a clean repo.
#
# Env knobs:
#   MAIN_BRANCH=main   trunk to measure "merged" against (default: main)
#   PRUNE_REMOTE=1     also delete merged remote branches on origin (default: off)
#   DRY_RUN=1          print what would be done, change nothing
#
set -euo pipefail

MAIN_BRANCH="${MAIN_BRANCH:-main}"
PRUNE_REMOTE="${PRUNE_REMOTE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Resolve repo layout. --git-common-dir points at the *shared* .git even when
# invoked from inside a linked worktree, so we always find the real metadata.
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
GIT_COMMON_DIR="$(cd "$GIT_COMMON_DIR" && pwd)"
MAIN_CHECKOUT="$(git rev-parse --show-toplevel)"

log()  { printf '  %s\n' "$*"; }
step() { printf '\n== %s ==\n' "$*"; }
run()  {
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN: $*"
  else
    "$@"
  fi
}

# Is the given pid a live process? (kill -0 succeeds for a running pid we can see.)
pid_alive() { kill -0 "$1" 2>/dev/null; }

# Is <ref> fully merged into $MAIN_BRANCH? (empty/unknown ref => not merged.)
is_merged() {
  local ref="$1"
  [ -n "$ref" ] || return 1
  git merge-base --is-ancestor "$ref" "$MAIN_BRANCH" 2>/dev/null
}

if ! git rev-parse --verify --quiet "$MAIN_BRANCH" >/dev/null; then
  echo "ERROR: trunk branch '$MAIN_BRANCH' not found; set MAIN_BRANCH=..." >&2
  exit 1
fi

echo "git-cleanup: trunk=$MAIN_BRANCH  repo=$MAIN_CHECKOUT  dry_run=$DRY_RUN  prune_remote=$PRUNE_REMOTE"

# ---------------------------------------------------------------------------
step "Fetch + prune remote-tracking refs"
run git fetch --prune

# ---------------------------------------------------------------------------
step "Release stale worktree locks held by DEAD agent pids"
# A worktree locked by a killed agent stays pinned forever. We release a lock
# only when the reason string names a pid AND every named pid is dead. Locks
# with a live pid, or with no parseable pid, are left untouched (conservative).
shopt -s nullglob
for lockfile in "$GIT_COMMON_DIR"/worktrees/*/locked; do
  wt_meta_dir="$(dirname "$lockfile")"
  wt_name="$(basename "$wt_meta_dir")"
  reason="$(tr -d '\000' <"$lockfile" 2>/dev/null || true)"
  # Extract candidate pids (integer tokens) from the lock reason.
  pids="$(printf '%s' "$reason" | grep -oE '[0-9]+' || true)"
  if [ -z "$pids" ]; then
    log "skip '$wt_name': lock reason has no pid to check (${reason:-<empty>})"
    continue
  fi
  any_alive=0
  for p in $pids; do
    if pid_alive "$p"; then any_alive=1; break; fi
  done
  if [ "$any_alive" = "1" ]; then
    log "skip '$wt_name': held by a live pid"
    continue
  fi
  # Recover the worktree's path from gitdir to unlock by path.
  wt_path=""
  if [ -f "$wt_meta_dir/gitdir" ]; then
    wt_path="$(dirname "$(cat "$wt_meta_dir/gitdir")")"
  fi
  log "releasing dead-pid lock on '$wt_name' (pids: $(echo $pids | tr '\n' ' '))"
  if [ -n "$wt_path" ]; then
    run git worktree unlock "$wt_path" || run rm -f "$lockfile"
  else
    run rm -f "$lockfile"
  fi
done
shopt -u nullglob

# ---------------------------------------------------------------------------
step "Remove agent worktrees whose branch is merged into $MAIN_BRANCH"
# Walk `git worktree list --porcelain`, keyed by the '.claude/worktrees/' path
# prefix. Never touch the main checkout. Only remove when the branch is merged.
removed_any=0
current_path=""
current_branch=""
process_wt() {
  local path="$1" branch="$2"
  [ -n "$path" ] || return 0
  # Guard: never the main checkout.
  [ "$path" = "$MAIN_CHECKOUT" ] && return 0
  # Only manage agent worktrees under .claude/worktrees/.
  case "$path" in
    */.claude/worktrees/*) ;;
    *) return 0 ;;
  esac
  if [ -z "$branch" ]; then
    log "skip '$path': detached HEAD (no branch to check merged); left for human"
    return 0
  fi
  local short="${branch#refs/heads/}"
  if is_merged "$branch"; then
    log "removing merged worktree '$path' (branch $short)"
    run git worktree remove --force "$path"
    removed_any=1
  else
    log "keep '$path': branch $short NOT merged; left for human"
  fi
}
while IFS= read -r line; do
  case "$line" in
    worktree\ *) current_path="${line#worktree }" ; current_branch="" ;;
    branch\ *)   current_branch="${line#branch }" ;;
    "")          process_wt "$current_path" "$current_branch" ; current_path="" ; current_branch="" ;;
  esac
done < <(git worktree list --porcelain; printf '\n')

# ---------------------------------------------------------------------------
step "Prune stale worktree administrative entries"
run git worktree prune -v

# ---------------------------------------------------------------------------
step "Delete local branches merged into $MAIN_BRANCH (safe -d)"
# `git branch --merged` lists only merged branches; -d refuses to delete an
# unmerged branch or one checked out in a worktree, so this is doubly safe.
this_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
deleted_any=0
while IFS= read -r br; do
  br="${br#"${br%%[![:space:]]*}"}"   # ltrim
  br="${br%% *}"                        # drop any trailing marker
  [ -n "$br" ] || continue
  case "$br" in
    "$MAIN_BRANCH"|"$this_branch"|"*"|"("*) continue ;;
  esac
  log "deleting merged local branch '$br'"
  run git branch -d "$br" || log "  (kept '$br': git refused — likely checked out or not fully merged)"
  deleted_any=1
done < <(git branch --merged "$MAIN_BRANCH" --format='%(refname:short)')
[ "$deleted_any" = "1" ] || log "nothing to delete — no merged local branches besides $MAIN_BRANCH"

# ---------------------------------------------------------------------------
if [ "$PRUNE_REMOTE" = "1" ]; then
  step "Delete merged REMOTE branches on origin (PRUNE_REMOTE=1)"
  rdeleted_any=0
  while IFS= read -r rref; do
    rref="${rref#"${rref%%[![:space:]]*}"}"
    [ -n "$rref" ] || continue
    case "$rref" in
      origin/HEAD*|origin/"$MAIN_BRANCH") continue ;;
      origin/*) : ;;
      *) continue ;;
    esac
    name="${rref#origin/}"
    log "deleting merged remote branch 'origin/$name'"
    run git push origin --delete "$name" || log "  (failed to delete origin/$name)"
    rdeleted_any=1
  done < <(git branch -r --merged "$MAIN_BRANCH" --format='%(refname:short)')
  [ "$rdeleted_any" = "1" ] || log "nothing to delete — no merged remote branches besides origin/$MAIN_BRANCH"
else
  step "Remote branch pruning skipped (set PRUNE_REMOTE=1 to enable)"
fi

# ---------------------------------------------------------------------------
step "Remaining worktrees"
git worktree list

step "Unmerged local branches (strays — review by hand)"
strays="$(git branch --no-merged "$MAIN_BRANCH" --format='%(refname:short)' || true)"
if [ -n "$strays" ]; then
  printf '%s\n' "$strays" | sed 's/^/  /'
else
  log "none — every local branch is merged into $MAIN_BRANCH"
fi

echo
echo "git-cleanup: done."
