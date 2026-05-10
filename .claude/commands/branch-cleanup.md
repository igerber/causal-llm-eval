---
description: Delete local branches whose PRs have been merged on GitHub
argument-hint: "[--dry-run] [--yes]"
---

# Branch Cleanup

Identify and delete local git branches whose pull requests have been merged. Arguments: $ARGUMENTS

## Arguments

`$ARGUMENTS` may contain:
- `--dry-run` (optional): Show what would be deleted without deleting anything.
- `--yes` / `-y` (optional): Skip the confirmation prompt and delete immediately.

Parse by splitting `$ARGUMENTS` on whitespace. Recognise `--dry-run`, `--yes`, and `-y`. If any other token is present, emit a warning (e.g., `Warning: unrecognised argument '<token>', ignoring`) but continue.

## Instructions

### 1. Verify Prerequisites

1. **Confirm git repo**:
   ```bash
   git rev-parse --is-inside-work-tree
   ```
   If this fails, abort:
   ```
   Error: Not inside a git repository.
   ```

2. **Confirm `origin` remote exists**:
   ```bash
   git remote get-url origin
   ```
   If this fails, abort:
   ```
   Error: No 'origin' remote configured.
   ```
   Then show `git remote` output (remote names only, no URLs — URLs may contain embedded credentials) so the user can see what remotes are configured.

3. **Check `gh` CLI availability**:
   ```bash
   gh --version 2>/dev/null
   ```
   If not available, set a flag `GH_AVAILABLE=false` and warn:
   ```
   Warning: GitHub CLI (gh) not found. Falling back to git-only detection.
   Squash-merged or rebase-merged branches may not be detected automatically.
   ```

4. **Record current branch**:
   ```bash
   git branch --show-current
   ```

5. **Detect worktree-checked-out branches**:
   ```bash
   git worktree list --porcelain
   ```
   Parse all `branch refs/heads/<name>` lines to build a set of branches checked out in worktrees.

### 2. Sync Remote State

```bash
git fetch --prune origin
```

- If fetch fails and `--dry-run` is **not** set, abort:
  ```
  Error: Failed to fetch from origin. Check your network connection.
  ```
- If fetch fails and `--dry-run` **is** set, warn and continue with stale data:
  ```
  Warning: Fetch failed. Showing results based on last fetch.
  ```

### 3. Enumerate Local Branches

```bash
git branch --format='%(refname:short)'
```

Filter out **protected branches** that must never be deleted:
- `main`
- `master`
- The current branch (from step 1.4)
- Any branch checked out in a worktree (from step 1.5)

### 4. Detect Merged-PR Candidates

For each remaining branch:

#### 4a. GitHub CLI detection (if `GH_AVAILABLE`)

```bash
gh pr list --state merged --head "<branch>" --json number,title,headRefOid --limit 1
```

- **If the command succeeds** (exit code 0) and returns valid JSON:
  - If the JSON array is non-empty, mark the branch as **gh-confirmed merged** and record the PR number, title, and `headRefOid` (the commit SHA at the branch tip when the PR was merged).
  - If the array is empty, the branch has no merged PR via this method — continue to 4b only if running the git-only fallback in parallel, otherwise skip.
- **If the command fails** (non-zero exit code or invalid JSON output):
  - If `gh` has failed for **every** branch checked so far (suggesting a systemic issue like auth failure), emit a single summary warning and disable `gh` for remaining branches:
    ```
    Warning: GitHub CLI authentication failed. Falling back to git-only detection for all branches.
    ```
    Set `GH_AVAILABLE=false` and fall back to step 4b for all remaining branches.
  - Otherwise (isolated failure for one branch), emit a per-branch warning and fall back to step 4b for that branch:
    ```
    Warning: gh pr list failed for branch '<branch>' — falling back to git-only detection.
    ```

#### 4b. Git gone-tracking detection (fallback, or supplement when gh unavailable)

Parse the output of:
```bash
git branch -vv
```

A branch whose tracking info shows `[origin/...: gone]` is a **gone-tracking** candidate. This means the remote branch was deleted (typically after PR merge).

#### Deduplication

If both methods identify the same branch, keep the **gh-confirmed** status (it enables safer `-D` deletion).

### 5. Enrich Candidates

For each candidate branch, collect:

- **Last commit subject**:
  ```bash
  git log -1 --format="%s" "refs/heads/<branch>"
  ```

- **Relative age**:
  ```bash
  git log -1 --format="%ar" "refs/heads/<branch>"
  ```

- **PR number and title** (from step 4a, if available)

### 6. Display Candidates

Present a table:

```
Merged branches found:

  Branch              Last Commit                     PR        Age
  ────────────────    ──────────────────────────────  ────────  ──────────
  fix/typo-readme     Fix typo in README              #42       3 days ago
  feature/bacon       Add Bacon decomposition         #38       2 weeks ago
  old-experiment      Refactor linalg backend         (gone)    1 month ago
```

- For gh-confirmed branches, show `#<number>` in the PR column.
- For gone-tracking-only branches, show `(gone)` in the PR column.
- If the current branch was a candidate, note it was skipped:
  ```
  Note: Current branch '<name>' also has a merged PR but cannot be deleted while checked out.
  Switch to another branch first if you want to clean it up.
  ```
- If a worktree branch was a candidate, note it was skipped:
  ```
  Note: Branch '<name>' skipped — checked out in worktree at <path>.
  ```

### 7. Handle No Candidates

If no candidates were found, report and exit:

```
No merged branches found. Your local branches are clean.
```

### 8. Confirm Deletion

- If `--dry-run`: print `Dry run — nothing was deleted.` and exit.
- If `--yes` / `-y`: skip confirmation and proceed to step 9.
- Otherwise, use **AskUserQuestion** with:
  - Option 1: "Delete all N branches"
  - Option 2: "Cancel"

If the user chooses "Cancel", exit without deleting.

### 9. Delete Branches

For each candidate, use a two-pass approach:

1. **Try safe delete first** (`-d`) for ALL candidates regardless of detection method:
   ```bash
   git branch -d -- "<branch>"
   ```

2. **If `-d` succeeds**: the branch is fully merged into HEAD — no data loss possible. Record as deleted.

3. **If `-d` fails AND the branch is gh-confirmed merged**: compare the local branch tip to the stored `headRefOid` before force-deleting:
   ```bash
   git rev-parse "<branch>"
   ```
   - **If the local tip matches `headRefOid`**: the branch has not changed since the PR was merged. Force-delete is safe:
     ```bash
     git branch -D -- "<branch>"
     ```
   - **If the local tip does NOT match `headRefOid`**: the branch has new commits since PR #N was merged. Do **not** force-delete — record as a failure with a specific message:
     ```
     Branch '<branch>' has new commits since PR #<N> was merged. Skipping force-delete to protect unmerged work.
     ```

4. **If `-d` fails AND the branch is gone-tracking only** (no gh confirmation): record as a failure. Do not force-delete — there is no external confirmation that the work is preserved.

### 10. Report Results

Print a summary:

```
Branch cleanup complete.

  Deleted: N branches
  Failed:  M branches
  Skipped: K branches (protected/checked-out)
```

If any branches failed deletion, group by failure reason:

For **gone-tracking branches** that failed `-d` (likely squash-merged, no gh confirmation):
```
The following branches could not be safely deleted (likely squash-merged):

  <branch-name>

To delete manually after verifying on GitHub:
  git branch -D -- "<branch-name>"
```

For **gh-confirmed branches** that diverged from their merged PR tip:
```
The following branches have new local commits since their PRs were merged:

  <branch-name>  (PR #<N> merged, but branch has new commits)

Review the new commits before deleting:
  git log <headRefOid>..refs/heads/<branch-name>
  git branch -D -- "<branch-name>"
```

## Error Handling

### Not a git repository
```
Error: Not inside a git repository.
```

### No origin remote
```
Error: No 'origin' remote configured.
Run 'git remote' to see configured remote names.
```

### Network failure during fetch
- Normal mode: abort with error and recovery tip.
- `--dry-run`: warn and continue with stale data.

### gh CLI not installed
Warn and fall back to git-only detection. Note that squash-merged branches won't be auto-detected.

### Branch checked out in worktree
Skip with message naming the worktree path.

## Examples

```bash
# Preview which branches would be cleaned up
/branch-cleanup --dry-run

# Interactive mode — shows candidates, asks for confirmation
/branch-cleanup

# Non-interactive — delete without prompting
/branch-cleanup --yes
/branch-cleanup -y
```

## Notes

- Uses `--` before branch names in `git branch` deletion commands to prevent branch names starting with `-` from being interpreted as flags. Uses `refs/heads/<branch>` in `git log` commands to avoid both flag interpretation (branch names starting with `-`) and pathspec ambiguity (where `--` would cause git to treat the branch name as a file path).
- GitHub CLI detection handles squash-merged and rebase-merged PRs correctly (GitHub knows the PR was merged regardless of how commit hashes changed).
- Git gone-tracking detection is less reliable for squash/rebase merges since `git branch -d` requires the exact commits to be reachable from HEAD.
- Protected branches (`main`, `master`, current branch, worktree branches) are never deleted regardless of merge status.
