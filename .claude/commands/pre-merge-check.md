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

**Check A - Subprocess `claude` spawn missing required cold-start flags or hygiene**:
```bash
# Multiline AST scan: find each subprocess.run/Popen call that mentions "claude"
# and verify it contains ALL seven locked flags AND the cwd/env hygiene keywords.
# This is the canonical cold-start surface check; it must match the full locked
# invocation in CLAUDE.md "Cold-start agent runner" and harness/COLD_START_VERIFICATION.md
# "The locked invocation" + "Subprocess hygiene".
#
# Argv resolution: handles three call shapes:
#   subprocess.run(["claude", ...], cwd=..., env=...)              # inline positional
#   subprocess.run(args=["claude", ...], cwd=..., env=...)         # inline kwarg
#   cmd = ["claude", ...]; subprocess.run(cmd, cwd=..., env=...)   # simple variable
# For dynamic argv (function call result, conditional, list comprehension, etc.)
# the scanner FAILS CLOSED with a "manual review required" warning rather than
# silently passing. This prevents future spawn sites from bypassing the check
# by hiding the argv behind a non-resolvable expression.
python3 - <<'PY'
import ast, pathlib, sys

# All seven locked CLI flags (per CLAUDE.md "Cold-start agent runner" + COLD_START_VERIFICATION.md).
required_flags = {
    "--bare",
    "--setting-sources",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--print",
    "--output-format",
    "--add-dir",
}
# Required subprocess kwargs (per COLD_START_VERIFICATION.md "Subprocess hygiene").
required_kwargs = {"cwd", "env"}
SUBPROC_FUNCS = ("subprocess.run", "subprocess.Popen", "subprocess.check_call", "subprocess.check_output")


def find_argv_node(call, scope_body):
    """Resolve a subprocess.* call's argv to an ast.List node, or return a sentinel.

    Returns (list_node, kind):
      (ast.List, "inline-positional") - subprocess.run([...])
      (ast.List, "inline-kwarg")      - subprocess.run(args=[...])
      (ast.List, "resolved-variable") - cmd = [...]; subprocess.run(cmd) (or args=cmd)
      (None,     "dynamic")           - argv is a non-list expression we can't resolve; FAIL CLOSED
      (None,     "no-argv")           - call has no positional and no args= keyword
    """
    candidate = None  # the ast.Name or ast.List we'll try to resolve
    if call.args and isinstance(call.args[0], (ast.List, ast.Name)):
        candidate = call.args[0]
        kind_inline = "inline-positional"
    else:
        for kw in call.keywords:
            if kw.arg == "args" and isinstance(kw.value, (ast.List, ast.Name)):
                candidate = kw.value
                kind_inline = "inline-kwarg"
                break
        else:
            # Positional 0 exists but isn't a List/Name (e.g., tuple, function call, comprehension)
            if call.args:
                return (None, "dynamic")
            return (None, "no-argv")

    if isinstance(candidate, ast.List):
        return (candidate, kind_inline)

    # ast.Name: walk the surrounding scope_body for the most recent simple
    # `<name> = [...]` assignment that precedes the call.
    name = candidate.id
    resolved = None
    for stmt in ast.walk(scope_body):
        if not isinstance(stmt, ast.Assign):
            continue
        if getattr(stmt, "lineno", 1 << 30) >= call.lineno:
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name and isinstance(stmt.value, ast.List):
                # Keep the latest one before the call site
                if resolved is None or stmt.lineno > resolved.lineno:
                    resolved = stmt
    if resolved is not None:
        return (resolved.value, "resolved-variable")
    # Variable can't be resolved to a simple List literal - fail closed
    return (None, "dynamic")


def list_contains_claude(list_node):
    for el in list_node.elts:
        if isinstance(el, ast.Constant) and isinstance(el.value, str) and el.value == "claude":
            return True
    return False


def list_to_string(list_node):
    parts = []
    for el in list_node.elts:
        if isinstance(el, ast.Constant) and isinstance(el.value, str):
            parts.append(el.value)
        else:
            parts.append("<dynamic>")
    return " ".join(parts)


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
        if not any(fn in func_str for fn in SUBPROC_FUNCS):
            continue

        # Resolve argv. If the call is in an obvious enclosing function, use that
        # scope; else use the whole module.
        scope_body = tree
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(child is node for child in ast.walk(parent)):
                    scope_body = parent
                    break

        argv_node, kind = find_argv_node(node, scope_body)

        if kind == "no-argv":
            continue  # not a relevant subprocess call

        if kind == "dynamic":
            # Fail closed: we can't statically verify a non-list argv. If the
            # call doesn't reference "claude" anywhere in its source, skip; else
            # warn for manual review.
            try:
                src = ast.unparse(node)
            except Exception:
                src = ""
            if '"claude"' in src or "'claude'" in src:
                problems.append((path, node.lineno, [], [], ["argv is dynamic - manual cold-start review required"]))
            continue

        if not list_contains_claude(argv_node):
            continue

        args_str = list_to_string(argv_node)
        missing_flags = sorted(f for f in required_flags if f not in args_str)
        provided_kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
        missing_kwargs = sorted(required_kwargs - provided_kwargs)

        # Verify locked value pairings via pairwise scan of the resolved argv list.
        bad_pairings = []
        elements = argv_node.elts
        for i, el in enumerate(elements):
            if not isinstance(el, ast.Constant) or not isinstance(el.value, str):
                continue
            next_el = elements[i + 1] if i + 1 < len(elements) else None
            next_val = next_el.value if isinstance(next_el, ast.Constant) and isinstance(next_el.value, str) else None
            if el.value == "--setting-sources" and next_val != "":
                bad_pairings.append("--setting-sources must be paired with empty string \"\"")
            if el.value == "--output-format" and next_val != "stream-json":
                bad_pairings.append("--output-format must be paired with \"stream-json\"")
            if el.value == "--add-dir" and next_val is None:
                bad_pairings.append("--add-dir requires a tmpdir value as next list element")

        if missing_flags or missing_kwargs or bad_pairings:
            problems.append((path, node.lineno, missing_flags, missing_kwargs, bad_pairings))

for p, ln, mf, mk, bp in problems:
    parts = []
    if mf:
        parts.append(f"missing flags: {mf}")
    if mk:
        parts.append(f"missing kwargs: {mk}")
    parts.extend(bp)
    print(f"{p}:{ln}: " + "; ".join(parts))
sys.exit(1 if problems else 0)
PY
```
Flag: "Cold-start spawn must include all seven locked flags (`--bare`, `--setting-sources`, `--strict-mcp-config`, `--disable-slash-commands`, `--print`, `--output-format stream-json`, `--add-dir`) AND the `cwd=` and `env=` keyword arguments (per `harness/COLD_START_VERIFICATION.md` 'Subprocess hygiene' section). Missing items in <file:line>."

The Python AST scan is robust to multiline subprocess calls; the previous single-line grep heuristic missed them. If `python3` is unavailable, fall back to:
```bash
rg -U --multiline -n 'subprocess\.(run|Popen|check_call|check_output)\(\s*\[.*?"claude".*?\]' harness/ graders/ tests/ 2>/dev/null
```
and manually verify each match contains all seven flags plus `cwd=` and `env=`.

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
- [ ] Subprocess spawn includes ALL seven locked flags: `--bare`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--print`, `--output-format stream-json`, `--add-dir <tmpdir>`
- [ ] Subprocess hygiene present: `cwd=<run tmpdir>` and `env=clean_env` (allowlist) keyword arguments on the subprocess.* call
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
