---
description: Run AI code review locally using OpenAI API before opening a PR
argument-hint: "[--model <model>] [--timeout <seconds>] [--base <ref>]"
---

# Local AI Code Review

Run a structured code review using the OpenAI Responses API. Reviews changes against the same criteria used by the CI reviewer (`.github/codex/prompts/pr_review.md`), framed for local pre-PR use.

This is the minimal Phase 0 version. Re-review delta-diff, finding tracking, full-file secret scanning, and cost estimation will be added incrementally.

## Arguments

`$ARGUMENTS` may contain optional flags:
- `--model <name>`: Override the OpenAI model (default: `gpt-5.5`).
- `--timeout <seconds>`: HTTP request timeout. Defaults to 900s for reasoning models (`gpt-5.4`, `gpt-5.5`, `*-pro`, `o1/o3/o4`) and 300s otherwise.
- `--base <ref>`: Comparison ref (default: auto-resolve repo's default branch).

## Constraints

This skill does not modify source code files. It may:
- Create a commit if there are uncommitted changes (Step 3)
- Write a review markdown file to `.claude/reviews/` (gitignored)
- Write temporary files to `/tmp/` (cleaned up at end)

Step 5 makes a single external API call to OpenAI. Step 3b runs a secret scan before any data is sent externally.

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` for the optional flags listed above. All flags are optional - the default behavior (auto-resolve base, gpt-5.5, default timeout) requires no arguments.

### Step 2: Validate Prerequisites

```bash
test -n "$OPENAI_API_KEY" && echo "API key: set" || echo "API key: MISSING"
test -f .claude/scripts/openai_review.py && echo "Script: found" || echo "Script: MISSING"
test -f .github/codex/prompts/pr_review.md && echo "Prompt: found" || echo "Prompt: MISSING"
mkdir -p .claude/reviews
```

If `OPENAI_API_KEY` is missing:
```
Error: OPENAI_API_KEY is not set.

To set it up:
1. Get a key from https://platform.openai.com/api-keys
2. Add to your shell: echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshrc
3. Reload: source ~/.zshrc
```

If the script or prompt file is missing, surface the specific path and stop.

### Step 3: Resolve Base Ref and Commit Changes

If `--base` was passed, use it. Otherwise auto-resolve:

```bash
default_branch=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name' 2>/dev/null || echo "main")

if git rev-parse --verify "$default_branch" >/dev/null 2>&1; then
    comparison_ref="$default_branch"
elif git rev-parse --verify "origin/$default_branch" >/dev/null 2>&1; then
    comparison_ref="origin/$default_branch"
else
    git fetch origin "$default_branch" --depth=1 2>/dev/null || true
    comparison_ref="origin/$default_branch"
fi
```

Check for uncommitted changes:
```bash
git status --porcelain
```

If non-empty, commit them before proceeding:
1. Show user what will be committed
2. `git add -A` and create a descriptive commit (follow recent `git log --oneline` conventions)
3. Report the commit message and short SHA

### Step 3b: Secret Scan

Run the canonical secret scan from `/pre-merge-check` Section 2.6 against the diff range:

```bash
secret_files=$(git diff -G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])" --name-only "${comparison_ref}...HEAD" 2>/dev/null || true)

sensitive_files=$(git diff --name-only "${comparison_ref}...HEAD" | grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$" || true)
```

If either is non-empty, use AskUserQuestion to confirm transmission to OpenAI before continuing.

### Step 4: Run the Review Script

```bash
output_path=".claude/reviews/local-review-latest.md"
python3 .claude/scripts/openai_review.py \
    --review-criteria .github/codex/prompts/pr_review.md \
    --output "$output_path" \
    --base "$comparison_ref" \
    --repo-root "$(pwd)" \
    [--model <model>] \
    [--timeout <seconds>]
```

Reasoning model handling: if `--model` resolves to a reasoning model (contains `-pro`, starts with `o1/o3/o4`, or starts with `gpt-5.4/gpt-5.5`), the request can take 10-15 minutes. Run the Bash command with `run_in_background: true` to bypass the 600s Bash timeout cap, then continue once it completes.

If the script exits non-zero, display the stderr output and stop.

### Step 5: Display the Review

Read and display `.claude/reviews/local-review-latest.md` in full.

### Step 6: Summarize Findings and Offer Next Steps

Parse the review for findings (Severity P0/P1/P2/P3, Section, one-line summary, file:line). Present a summary grouped by severity, then use AskUserQuestion:

**No findings (clean review)**:
```
Suggested next step:
- /submit-pr - commit and open a pull request
```

**P0/P1 findings (Blocker / Needs changes)**:
```
Options:
1. Enter plan mode to address findings (Recommended)
2. Skip - I'll address these manually
```

**P2/P3 only (Looks good)**:
```
Options:
1. Address findings before submitting
2. Skip - proceed to /submit-pr
```

If the user chooses to address findings: parse them from the review output (already in conversation context). For P0/P1 use `EnterPlanMode`; for P2/P3 fix directly.

After fixes are committed, the user re-runs `/ai-review-local` for a fresh review (delta-diff and finding-tracking are not yet implemented; each run is a full review).

### Step 7: Cleanup

The minimal script writes only to the `--output` path; no temp files to clean.

## Error Handling

| Scenario | Response |
|---|---|
| `OPENAI_API_KEY` not set | Error with setup instructions (see Step 2) |
| Script file missing | Error suggesting it should be checked in |
| No diff vs base ref | Clean exit with message |
| Script exits non-zero | Display stderr from script |
| Network error | Display urlerror reason |

## Notes

- This skill does NOT modify source files - it generates a review markdown in `.claude/reviews/` (gitignored). It may create a commit if there are uncommitted changes (Step 3).
- The review criteria are loaded from `.github/codex/prompts/pr_review.md` - the same file the CI reviewer uses.
- The CI review (Codex action with full repo access) remains the authoritative final check; local review is a fast first pass.
- **Data transmission**: in non-dry-run mode, this skill transmits the unified diff and changed-file metadata to OpenAI via the Responses API. The minimal script does NOT include full source file contents.
- Pairs with the iterative workflow: `/ai-review-local` -> address findings -> `/ai-review-local` -> `/submit-pr`.
