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

The locked subprocess invocation is multiline (a Python list literal split across lines), so single-line `grep` will miss it. Use multiline-aware tooling: `rg -U` (or a small Python AST scanner) to inspect the full subprocess call block, not just one line at a time.

**Check A - Subprocess `claude` spawn missing required cold-start flags**:
```bash
# Multiline rg: find each subprocess.run/Popen call that mentions "claude"
# and verify it contains all four locked flags.
python3 - <<'PY'
import ast, pathlib, sys
required = {"--bare", "--setting-sources", "--strict-mcp-config", "--disable-slash-commands"}
problems = []
for path in pathlib.Path(".").rglob("*.py"):
    if any(part in {".venv", ".venv-pool", "__pycache__"} for part in path.parts):
        continue
    if not any(str(path).startswith(p) for p in ("harness/", "graders/", "tests/")):
        continue
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not any(name in func_str for name in ("subprocess.run", "subprocess.Popen", "subprocess.check_call", "subprocess.check_output")):
            continue
        # Look for the literal "claude" in any positional or keyword arg
        try:
            args_str = " ".join(ast.unparse(a) for a in node.args)
        except Exception:
            continue
        if '"claude"' not in args_str and "'claude'" not in args_str:
            continue
        missing = sorted(f for f in required if f not in args_str)
        if missing:
            problems.append((path, node.lineno, missing))
for p, ln, missing in problems:
    print(f"{p}:{ln}: missing cold-start flags: {missing}")
sys.exit(1 if problems else 0)
PY
```
Flag: "Cold-start spawn must include `--bare`, `--setting-sources \"\"`, `--strict-mcp-config`, `--disable-slash-commands` (locked architectural decision). Missing flags in <file:line>."

The Python AST scan is robust to multiline subprocess calls; the previous single-line grep heuristic missed them. If `python3` is unavailable, fall back to:
```bash
rg -U --multiline -n 'subprocess\.(run|Popen|check_call|check_output)\(\s*\[.*?"claude".*?\]' harness/ graders/ tests/ 2>/dev/null
```
and manually verify each match contains all four flags.

**Check B - In-process telemetry shim parity**:
For changes touching the diff-diff arm shim, check the statsmodels arm shim was updated:
```bash
git diff --name-only HEAD -- harness/sitecustomize_template.py harness/sitecustomize_*.py | head
```
Flag if only one arm's shim changed without documented rationale.

#### 2.2 Prompt / Rubric Versioning Patterns (for prompt or rubric changes)

If `prompts/**/*.txt` OR `rubrics/**/*.yaml` changed:
```bash
git diff --name-only HEAD -- prompts/ rubrics/
```

**Check C - Versioned artifact edited in place vs new version**:
Versioned prompts (e.g., `prompts/case_study/v1.txt`) and rubrics (e.g., `rubrics/case_study_v1.yaml`) should NOT be edited after first use in a recorded run. New artifact = new version file (`v2.txt`, `case_study_v2.yaml`).
```bash
# Flag in-place edits to existing versioned prompts/rubrics
git diff --name-only HEAD -- prompts/ rubrics/ | xargs -I {} sh -c 'echo "Edited: {}"; git log --oneline {} | head -3'
```
Flag: "If this prompt/rubric has been used in a recorded run, create `vN+1` instead of editing `vN` in place. Per-run records reference the version string; mutating a recorded artifact breaks reproducibility."

**Check D - Prompt contamination patterns**:
```bash
grep -rn -E '(get_llm_guide|llms\.txt|llms-practitioner|practitioner workflow|CallawaySantAnna|SunAbraham|dCDH|de Chaisemartin|HonestDiD|BaconDecomposition|pre-trends|sensitivity analysis|placebo)' prompts/case_study/ 2>/dev/null
```
Flag: "Case-study prompts must NOT contain guidance hints (estimator names, methodology terms, library-specific surfaces). See pr_review.md anti-pattern #3."

**Check E - Rubric schema sanity** (rubric files only):
```bash
# Verify YAML parses
python3 -c "import yaml; yaml.safe_load(open('rubrics/<changed-file>.yaml'))"
```
Flag: "Rubric YAML must parse cleanly; downstream judge code consumes the structured fields."

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

#### If Rubrics Changed
```
### Rubric Versioning
- [ ] In-place edit of recorded rubrics caught and corrected to vN+1.yaml naming (per-run records pin the rubric version; mutating a recorded rubric breaks reproducibility)
- [ ] YAML parses cleanly (downstream judge code consumes the structured fields)
- [ ] If a new dimension was added: extractor and judge updated to populate it; analysis code updated to consume it
- [ ] If a dimension was renamed: spot-check old recorded grades for backward-compat or document the schema break
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
