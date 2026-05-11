---
description: Commit changes to a new branch, push to GitHub, and open a PR with project template
argument-hint: "[title] [--branch <name>] [--base <branch>] [--draft]"
---

# Submit Pull Request

Commit work, push to a new branch, and open a pull request with the project-specific PR template.

## Arguments

`$ARGUMENTS` may contain:
- **title** (optional): PR title. If omitted, auto-generate from changes/commits.
- `--branch <name>` (optional): Branch name. If omitted, auto-generate from title.
- `--base <branch>` (optional): Base branch for PR. Default: `main`.
- `--draft` (optional): Create as draft PR.

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract title, `--branch`, `--base` (default `main`), `--draft`.

### 2. Detect Remote Configuration

Determine if this is a fork-based workflow:

```bash
git remote get-url upstream 2>/dev/null
```

- If `upstream` exists -> **fork workflow**: `<base-remote>=upstream`, `<push-remote>=origin`. Extract `<upstream-owner>/<upstream-repo>` from upstream URL; extract `<fork-owner>` from origin URL.
- Else -> **direct workflow**: `<base-remote>=origin`, `<push-remote>=origin`. Extract `<owner>/<repo>` from origin URL.

Then:
```bash
git fetch <base-remote>
```

### 3. Sync with Remote

```bash
git rev-list --count HEAD..<base-remote>/<base-branch>
```

If count > 0, branch is behind. Use AskUserQuestion to offer rebase or continue.

### 4. Check for Changes

```bash
git status --porcelain
```

If non-empty, proceed to step 5. If empty, check for unpushed commits:
```bash
git rev-list --count <base-remote>/<base-branch>..HEAD
```

If 0, exit: "No changes detected." If > 0, skip to step 7.

### 5. Resolve Branch Name (BEFORE any commits)

**IMPORTANT**: Always resolve branch name before staging or committing to avoid commits on the base branch.

```bash
git branch --show-current
```

If on base branch (e.g., `main`):
- Use `--branch` if provided, otherwise generate from title or change analysis
- Sanitize: lowercase, hyphens for spaces, strip invalid git ref chars (`:?*[]^~\\@{..`), collapse consecutive separators, trim, truncate to 50 chars
- Prefix by change type: `feature/`, `fix/`, `refactor/`, `docs/`, `infra/`
- Validate: `git check-ref-format --branch "<branch-name>"`
- Create: `git checkout -b <branch-name>`

If already on a feature branch, use the current branch name.

### 5b. Stage and Quick Pattern Check

```bash
git add -A
```

If harness/grader/analysis/prompt/rubric/Makefile files are staged, run pattern checks per `/pre-merge-check` Sections 2.1 (cold-start integrity), 2.2 (prompt and rubric versioning), and 2.5 (reproducibility schema):

```bash
git diff --cached --name-only | grep -E "^(harness|graders|analysis)/.*\.py$" >/dev/null && echo "harness/grader/analysis changes detected"
git diff --cached --name-only | grep -E "^prompts/" >/dev/null && echo "prompt changes detected"
git diff --cached --name-only | grep -E "^rubrics/" >/dev/null && echo "rubric changes detected"
git diff --cached --name-only | grep -E "^Makefile$" >/dev/null && echo "Makefile changes detected (reproducibility-target surface)"
```

A rubric-only change must also bump the rubric registry version (no in-place edit of recorded rubrics; new rubric = `vN+1.yaml`). The pre-merge-check pattern guidance covers both prompts and rubrics under the same versioning convention. Makefile changes that touch `case-study-v1`, `preflight`, `calibration`, or `smoke` targets are reproducibility-relevant - run Section 2.5.

Run all relevant pattern checks (A through E) on the staged files. For matches, display file:line and offer:
```
Pre-commit pattern check found N potential issues:
<list warnings>

Options:
1. Fix issues before committing (recommended)
2. Continue anyway
```

### 6. Commit Changes

**Secret scanning** using canonical patterns from `/pre-merge-check` Section 2.6:
```bash
secret_files=$(git diff --cached -G "<content pattern>" --name-only 2>/dev/null || true)
sensitive_files=$(git diff --cached --name-only | grep -iE "<filename pattern>" || true)
```

If patterns detected, unstage and warn:
```bash
git reset HEAD
```

Use AskUserQuestion to confirm. If user continues, re-stage with `git add -A`.

**Generate commit message**:
- Run `git diff --cached --stat`
- Analyze changes; generate descriptive commit message in imperative mood
- Use HEREDOC:
  ```bash
  git commit -m "$(cat <<'EOF'
  <generated commit message>

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### 7. Push Branch to Remote

```bash
git branch --show-current
```

**Guard**: if current branch equals `<base-branch>`, abort with error or create a new branch.

```bash
git push -u <push-remote> <branch-name>
```

### 8. Extract Commit Information for PR Body

```bash
git log <base-remote>/<base-branch>..HEAD --oneline
git diff <base-remote>/<base-branch>..HEAD --stat
```

Categorize changes for the template:
- **Harness/grader changes**: files in `harness/`, `graders/`, `analysis/`
- **Prompt/rubric changes**: files in `prompts/`, `rubrics/`
- **Dataset/DGP changes**: files in `datasets/`
- **Test changes**: files in `tests/`
- **Documentation**: files in `docs/`, `*.md`, `*.rst`
- **Infrastructure**: files in `.github/`, `.claude/`, `pyproject.toml`, `Makefile`

### 9. Generate PR Body

```markdown
## Summary
- <bullet point for each commit>

## Eval validity references (required if changes touch `harness/`, `graders/`, `analysis/`, `prompts/`, `rubrics/`, or reproducibility-relevant `Makefile` targets like `case-study-v1` / `smoke` / `preflight` / `calibration`)
- Affected validity guarantee(s): <cold-start | telemetry | prompt-contamination | reproducibility | comparator-fairness | "N/A">
- Plan-file section / decision: <link or section name in plan, or "N/A">
- Any intentional deviations from the plan (and why): <if applicable, or "None">

## Validation
- Tests added/updated: <list test files or "No test changes">
- `make smoke` status: <pass/fail/not-run>
- Cold-start probe verification: <pass/fail/not-applicable>

## Security / privacy
- Confirm no secrets/PII in this PR: Yes

---
Generated with Claude Code
```

**Template logic**:
- **Eval validity**: Mark "N/A" only if NO files changed in `harness/`, `graders/`, `analysis/`, `prompts/`, `rubrics/`, or any reproducibility-relevant `Makefile` targets (`case-study-v1`, `preflight`, `calibration`, `smoke`). If any changed, identify which validity guarantee is affected.
- **Validation**: List `test_*.py` files changed; report `make smoke` status.
- **Security**: Default "Yes"; warn if `.env`, credentials, or API key patterns detected.

### 10. Create Pull Request

Use the MCP GitHub tool (or `gh pr create` if MCP unavailable):

```
mcp__github__create_pull_request:
  - owner: <target-owner>
  - repo: <target-repo>
  - title: <PR title>
  - head: <head-ref>           # direct: <branch-name>; fork: <fork-owner>:<branch-name>
  - base: <base-branch>
  - body: <generated PR body>
  - draft: <true if --draft>
```

**Per memory `feedback_no_label_ready_for_ci_on_open`**: do NOT add `ready-for-ci` label immediately on PR open. Leave unlabeled; user adds the label when ready (the label freezes the branch).

### 10b. Ensure PR ref exists

```bash
git push <push-remote> <branch-name>
git ls-remote <push-remote> refs/pull/<number>/head
```

If still missing, push an empty commit:
```bash
git commit --allow-empty -m "Trigger PR ref creation"
git push <push-remote> <branch-name>
```

### 11. Report Results

```
Pull request created successfully!

Branch: <branch-name>
PR: #<number> - <title>
URL: https://github.com/<target-owner>/<target-repo>/pull/<number>

Changes included:
<list of changed files>

Next steps:
- Review the PR at the URL above
- AI code review runs automatically on PR open
- When AI review is green and you're ready for CI tests, add the `ready-for-ci` label
- For follow-up updates, use /push-pr-update
```

## Error Handling

### No Changes to Commit
```
No changes detected. Your working directory is clean. Nothing to submit.
```

### Branch Already Exists
```
Branch '<name>' already exists.
Options:
1. Provide different name: /submit-pr "title" --branch <new-name>
2. Delete existing: git branch -D <name>
```

### Push/PR Creation Failed
Show the error and provide manual fallback commands.

## Examples

```bash
# Auto-generate everything
/submit-pr

# With custom title
/submit-pr "Add cold-start verification probe"

# With custom branch
/submit-pr "Fix telemetry parity" --branch fix/telemetry-parity

# Draft PR against different base
/submit-pr "Bootstrap harness skeleton" --base develop --draft
```

## Notes

- Always stages ALL changes (`git add -A`). Stage manually first for partial commits.
- Branch names auto-prefixed: `feature/`, `fix/`, `refactor/`, `docs/`, `infra/`.
- Uses MCP GitHub server for PR creation (requires PAT with repo access). Falls back to `gh pr create`.
- Git push uses SSH or HTTPS based on remote URL configuration.
- **Fork workflows supported**: If `upstream` remote exists, PRs target upstream with `<fork-owner>:<branch>` head reference.
