---
description: Run pre-merge checks before submitting a PR
argument-hint: ""
---

# Pre-Merge Check

Run automated checks and display the pre-merge checklist before submitting a PR.

## Instructions

### 1. Identify Changed Files

```bash
git diff --name-only HEAD
git diff --cached --name-only
git ls-files --others --exclude-standard
```

Categorize:
- **Harness/grader code**: `harness/**/*.py`, `graders/**/*.py`, `analysis/**/*.py`
- **Prompts/rubrics**: `prompts/**/*.txt`, `rubrics/**/*.yaml`
- **Test files**: `tests/**/*.py`
- **Documentation**: `*.md`, `*.rst`, `docs/**`
- **Datasets/DGP**: `datasets/**/*.py` (generators), `datasets/**/*.parquet`, `datasets/**/*.json`
- **Infrastructure**: `.github/**`, `.claude/**`, `pyproject.toml`, `Makefile`

### 2. Run Automated Pattern Checks

#### 2.1 Cold-Start Integrity Patterns (for harness changes)

> **Canonical definitions** - referenced by `/submit-pr`, `/push-pr-update`, `/ai-review-local`. Single source of truth for harness validity checks.

If any harness files changed:

**Check A - Subprocess spawn missing `--bare`**:
```bash
grep -rn 'subprocess.*claude' harness/ graders/ | grep -v '\-\-bare' | grep -v '^\s*#'
```
Flag: "Cold-start spawn must include `claude --bare` (locked architectural decision). Missing in <file:line>."

**Check B - Subprocess spawn missing `--setting-sources ""`**:
```bash
grep -rn '"claude".*"--bare"' harness/ graders/ | grep -v 'setting-sources' | grep -v '^\s*#'
```
Flag: "Cold-start spawn must include `--setting-sources ""` to suppress global settings."

**Check C - In-process telemetry shim parity**:
For changes touching the diff-diff arm shim, check the statsmodels arm shim was updated:
```bash
git diff --name-only HEAD -- harness/sitecustomize_template.py harness/sitecustomize_*.py | head
```
Flag if only one arm's shim changed without documented rationale.

#### 2.2 Prompt Versioning Patterns (for prompt changes)

If `prompts/**/*.txt` changed:
```bash
git diff --name-only HEAD -- prompts/
```

**Check D - Prompt edited in place vs new version**:
Versioned prompts (e.g., `prompts/case_study/v1.txt`) should NOT be edited after first run. New prompt = new version file.
```bash
# Flag in-place edits to existing versioned prompts
git diff --name-only HEAD -- prompts/ | xargs -I {} sh -c 'echo "Edited: {}"; git log --oneline {} | head -3'
```
Flag: "If this prompt has been used in a recorded run, create `vN+1.txt` instead of editing `vN.txt` in place."

**Check E - Prompt contamination patterns**:
```bash
grep -rn -E '(get_llm_guide|llms\.txt|llms-practitioner|practitioner workflow|CallawaySantAnna|SunAbraham|dCDH|de Chaisemartin|HonestDiD|BaconDecomposition|pre-trends|sensitivity analysis|placebo)' prompts/case_study/ 2>/dev/null
```
Flag: "Case-study prompts must NOT contain guidance hints (estimator names, methodology terms, library-specific surfaces). See pr_review.md anti-pattern #3."

#### 2.3 Test Existence Check

For each changed source file (harness, grader, analysis), check for corresponding test:

| Source File | Expected Test |
|---|---|
| `harness/runner.py` | `tests/test_runner.py` |
| `harness/telemetry.py` | `tests/test_telemetry.py` |
| `harness/extractor.py` | `tests/test_extractor.py` |
| `harness/venv_pool.py` | `tests/test_venv_pool.py` |
| `harness/scheduler.py` | `tests/test_scheduler.py` |
| `graders/ai_judge.py` | `tests/test_ai_judge.py` |
| `analysis/cell_summary.py` | `tests/test_cell_summary.py` |
| `analysis/variability_report.py` | `tests/test_variability_report.py` |
| `analysis/reproducibility_check.py` | `tests/test_reproducibility_check.py` |

Report any source files without corresponding test changes.

#### 2.4 Docstring Check (heuristic)

```bash
grep -n "^def [^_]" <changed-py-files> | head -10
grep -n "^    def [^_]" <changed-py-files> | head -10
```

Verify docstrings exist for new public functions.

#### 2.5 Reproducibility Schema Check

If changes affect what's recorded per run (per-run record fields, schema validators, `make case-study-v1` targets):
```bash
git diff --name-only HEAD -- harness/telemetry.py harness/extractor.py analysis/reproducibility_check.py Makefile
```

Flag: "Per-run schema or reproducibility check changed. Verify the schema validator AND `make case-study-v1`'s reproducibility check are updated together."

#### 2.6 Secret Scanning Patterns (Canonical Definitions)

> Referenced by `/submit-pr`, `/push-pr-update`, `/ai-review-local`.

**Content pattern** (use with `-G`, `--name-only` to avoid leaking secrets):
```bash
-G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])"
```

**Sensitive filename pattern**:
```bash
grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$"
```

**Usage**: Apply content pattern to `--cached` for staged changes, or `<ref>..HEAD` for already-committed changes. Always use `--name-only` and `|| true`.

### 3. Display Context-Specific Checklist

#### Always Show (Core Checklist)
```
## Pre-Merge Checklist

Based on your changes to: <list of changed files>

### Behavioral Completeness
- [ ] Happy path tested
- [ ] Edge cases tested (empty data, missing telemetry, malformed transcripts)
- [ ] Error/warning paths tested with behavioral assertions
```

#### If Harness Files Changed
```
### Cold-Start Integrity
- [ ] Subprocess spawn includes `claude --bare --setting-sources "" --strict-mcp-config --disable-slash-commands`
- [ ] `make smoke` still passes the inheritance probe (agent reports no skills/memory/CLAUDE.md)
- [ ] No new code path reads from operator's env (`$HOME/.claude`, keychain) and passes into the spawned agent

### Telemetry Completeness
- [ ] All three layers captured for new tracked surfaces (stream-JSON + in-process shim + stderr)
- [ ] Per-run record fields match the schema validator
- [ ] In-process shim updated symmetrically across both arms (or one-sided documented)

### Reproducibility
- [ ] Version pins captured per run (library version, Claude Code version, model id)
- [ ] No new randomness introduced without seed capture
```

#### If Prompts Changed
```
### Prompt Versioning
- [ ] In-place edit of recorded prompts caught and corrected to vN+1.txt naming
- [ ] No guidance hints leaked (estimator names, methodology terms, library-specific surfaces)
- [ ] Arm 1 and arm 2 prompts identical word-for-word except library name (or amendment documented)
```

#### If Documentation Files Changed
```
### Documentation Sync
- [ ] CLAUDE.md updated if conventions changed
- [ ] README.md updated only for landing-page relevant changes
- [ ] Docstrings updated for changed function signatures
```

#### If This Appears to Be a Bug Fix
```
### Pattern Consistency (Bug Fix)
- [ ] Grepped for similar patterns across codebase before fixing
- [ ] Fixed ALL occurrences, not just the one reported
- [ ] Verified fix doesn't break other code paths
```

### 4. Ask About Running Tests

Use AskUserQuestion:
```
Would you like to run tests now?

Options:
1. Yes - run full test suite (pytest)
2. Yes - run only tests for changed files
3. No - skip tests for now
```

If option 1: `pytest`
If option 2: run targeted tests based on changed files (e.g., `pytest tests/test_runner.py tests/test_telemetry.py`).

### 5. Report Summary

```
## Pre-Merge Check Complete

### Automated Checks
- Pattern checks: [PASS/WARN - N potential issues found]
- Test coverage: [PASS/WARN - N source files without test changes]
- Cold-start integrity: [PASS/WARN]
- Prompt contamination: [PASS/WARN]

### Manual Checklist
Review the checklist items above before running /submit-pr.

### Findings to Address
<list any warnings>

### Next Steps
- Address any warnings above
- Complete manual checklist items
- When ready: /submit-pr "Your PR title"
```

## Notes

- This skill is read-only - it analyzes and reports, doesn't modify files.
- Run BEFORE `/submit-pr` to catch issues early.
- Pattern checks are heuristics - review flagged items manually.
- If pattern checks find issues, fix before submitting.
