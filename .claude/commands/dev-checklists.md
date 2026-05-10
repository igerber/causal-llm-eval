---
description: Development checklists for code changes (params, eval-validity, warnings, reviews, bugs)
argument-hint: "[checklist-name]"
---

# Development Checklists

## Adding a New Configuration Parameter

When adding a new field to `RunConfig`, `RunMetadata`, or another harness/grader/analysis dataclass:

1. **Implementation**:
   - [ ] Add to dataclass with type annotation; default value if optional
   - [ ] If it influences per-run behavior, also add to `RunMetadata` so it's pinned in the per-run record
   - [ ] If it's tri-state with arm-specific contracts (e.g., `opened_llms_*`), update `__post_init__` validation

2. **Consistency** - apply to all relevant call sites:
   - [ ] Runner spawn site (`harness/runner.py`)
   - [ ] Telemetry merge (`harness/telemetry.py:merge_layers`)
   - [ ] Grader (`graders/ai_judge.py`) if the field affects rubric scoring
   - [ ] Reproducibility check (`analysis/reproducibility_check.py`) if the field is part of the schema

3. **Testing**:
   - [ ] Smoke test asserts the field exists and has the right type
   - [ ] Behavioral test asserts the field affects downstream behavior
   - [ ] Test with non-default value

4. **Downstream tracing**:
   - [ ] Before implementing: `grep -rn "<field_name>" harness/ graders/ analysis/` to find ALL downstream paths
   - [ ] Field handled in all per-arm code paths (diff-diff arm AND statsmodels arm if relevant)

5. **Documentation**:
   - [ ] Update CLAUDE.md if it's a key design pattern
   - [ ] Update `harness/COLD_START_VERIFICATION.md` if it touches the cold-start contract

## Touching Eval-Validity Code

When implementing or modifying code that affects eval validity (cold-start integrity, telemetry completeness, prompt versioning, comparator fairness, reproducibility):

1. **Before coding - consult the locked decisions**:
   - [ ] Read the latest plan in `~/.claude/plans/` (the active wave's plan)
   - [ ] Read `harness/COLD_START_VERIFICATION.md` for the cold-start contract
   - [ ] Read `pre-merge-check.md` Sections 2.1-2.5 for the canonical patterns the AI reviewer applies

2. **During implementation**:
   - [ ] Cold-start spawn includes ALL locked flags: `--bare`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--print`, `--output-format stream-json`, `--add-dir <tmpdir>`
   - [ ] Subprocess hygiene: `cwd=<tmpdir>`, `env=clean_env` (allowlist), no operator-state inheritance
   - [ ] Three-layer telemetry maintained when adding a new tracked surface (stream-JSON + in-process shim + stderr)
   - [ ] In-process shim instrumentation symmetrical across both arms (or one-sided change documented with rationale)
   - [ ] Sentinel semantics enforced via `__post_init__` (arm-specific tri-state for `opened_llms_*` etc.)
   - [ ] Reproducibility metadata pinned per run (RunMetadata fields)

3. **When deviating from the locked plan**:
   - [ ] Add a `**Note:**` or `**Deviation from plan:**` label in the code section, PR description, or plan file (see CLAUDE.md "Documenting Deviations")
   - [ ] Include rationale (why the locked decision doesn't fit this case)
   - [ ] If deferring related P2/P3 work: add a row to `TODO.md` table under "Tech Debt from Code Reviews"

4. **Testing eval-validity**:
   - [ ] Cold-start probe assertion (when runner is implemented): agent reports no inheritance
   - [ ] Per-run record schema check: required fields present, types correct
   - [ ] Sentinel violations rejected at construction (test both arms with invalid combos)
   - [ ] Stderr/in-process warning capture verified end-to-end

## Adding Warning/Error/Fallback Handling

When adding code that emits warnings or handles errors in the harness/grader/analysis paths:

1. **Fail closed by default**:
   - [ ] Telemetry write failures raise (don't silently no-op); the runner catches and marks the run failed
   - [ ] Cold-start probe failures abort the run before agent execution
   - [ ] Per-run schema violations reject at construction time, not at consumption

2. **Verify behavior matches message**:
   - [ ] Manually trace the code path after warning/error
   - [ ] Confirm the stated behavior actually occurs

3. **Write behavioral tests**:
   - [ ] Don't just test "no exception raised"
   - [ ] Assert the expected outcome occurred (run marked failed, exception type, error message substring)
   - [ ] For fallback paths: verify fallback was applied AND that the fallback didn't silently corrupt the per-run record

4. **Cross-layer warning capture**:
   - [ ] If a Python warning is emitted, both layer 2 (in-process shim) AND layer 3 (subprocess stderr) should observe it
   - [ ] If they disagree, that's a telemetry bug worth flagging

## Reviewing New Features or Code Paths

When reviewing PRs that add new harness/grader/analysis features or new tracked telemetry surfaces:

1. **Edge Case Coverage**:
   - [ ] Empty/missing per-run records (no transcript, no in-process events, malformed JSON)
   - [ ] Sentinel violations across both arms (arm 1 with None guide field, arm 2 with bool guide field)
   - [ ] Concurrent run interleaving (when scheduler is implemented; per-run venv isolation must hold)
   - [ ] Comparator fairness: any new instrumentation symmetrical across arms or documented one-sided

2. **Documentation Completeness**:
   - [ ] All new fields/parameters have docstrings with type, default, contract description
   - [ ] If touching the cold-start contract: `harness/COLD_START_VERIFICATION.md` updated
   - [ ] If touching telemetry layers: TelemetryRecord docstring updated
   - [ ] If touching prompts/rubrics: registry version bumped (no in-place edit of recorded prompts)

3. **Logic Audit for New Code Paths**:
   - [ ] When adding new arms or new surfaces, trace ALL downstream effects (runner -> telemetry -> extractor -> judge -> analysis)
   - [ ] Check the in-process shim is updated for the new surface (NOT just stream-JSON layer)
   - [ ] Explicitly test arm-1 vs arm-2 behavior in new code paths

4. **Pattern Consistency**:
   - [ ] Search for similar patterns in codebase (e.g., subprocess spawn sites, telemetry merge call sites)
   - [ ] Ensure new code follows established patterns or updates ALL instances
   - [ ] If fixing a pattern, grep for ALL occurrences first:
     ```bash
     grep -rn '<pattern>' harness/ graders/ analysis/ tests/
     ```

## Fixing Bugs Across Multiple Locations

When a bug fix involves a pattern that appears in multiple places:

1. **Find All Instances First**:
   - [ ] Use grep/search to find ALL occurrences of the pattern before fixing
   - [ ] Document the locations found (file:line)
   - [ ] Example: a missing cold-start flag in a subprocess spawn might appear in runner + scheduler + venv_pool

2. **Fix Comprehensively in One Round**:
   - [ ] Fix ALL instances in the same PR/commit
   - [ ] Add a regression test that covers the pattern (parametrized over locations if helpful)
   - [ ] Don't fix incrementally across multiple review rounds

3. **Regression Test the Fix**:
   - [ ] Verify fix doesn't break other code paths
   - [ ] For sentinel-violation fixes: ensure both armed-positive and armed-negative cases tested

4. **Common Patterns to Watch For**:
   - Subprocess spawn missing one of the locked cold-start flags -> cold-start leak
   - In-process shim added a hook for arm 1 but not arm 2 -> comparator-fairness asymmetry
   - New telemetry field added to TelemetryRecord but not to merge_layers or post_init -> silent contract drift
   - Prompt edited in place rather than versioned -> reproducibility break

## Pre-Merge Review Checklist

Final checklist before approving a PR:

1. **Behavioral Completeness**:
   - [ ] Happy path tested (when implementation lands)
   - [ ] Edge cases tested (missing telemetry, sentinel violations, malformed transcripts)
   - [ ] Error/warning paths tested with behavioral assertions

2. **Cold-Start + Telemetry Consistency**:
   - [ ] Any new subprocess spawn includes all locked cold-start flags
   - [ ] Any new tracked surface has in-process shim parity across arms (or documented one-sided)
   - [ ] Per-run record schema unchanged OR schema validator + reproducibility check updated together

3. **Documentation Sync**:
   - [ ] Docstrings updated for all changed signatures
   - [ ] CLAUDE.md updated if conventions changed
   - [ ] `prompts/` registry bumped if a recorded prompt changed (vN+1.txt, never in-place edit)
   - [ ] `rubrics/` registry bumped if a recorded rubric changed
   - [ ] `harness/COLD_START_VERIFICATION.md` updated if the cold-start contract changed
   - [ ] `README.md` updated ONLY for landing-page-relevant changes (status, hero/tagline, top-level capability summary)
   - [ ] `TODO.md` updated if new tech debt is introduced or addressed
   - [ ] `ROADMAP.md` updated if a planned-feature item is shipped or rescoped

## Quick Reference: Common Patterns to Check

Before submitting harness/telemetry changes, verify these patterns:

```bash
# Subprocess spawn sites (must include --bare and other locked flags)
grep -rn 'subprocess\.\(run\|Popen\|check_call\|check_output\)' harness/ graders/ tests/

# Telemetry record construction sites (must pass arm explicitly)
grep -rn 'TelemetryRecord(' harness/ graders/ analysis/

# Prompt registry version refs (catch in-place edits to recorded prompts)
git log --oneline -- prompts/

# Cold-start verification doc references
grep -rn 'COLD_START_VERIFICATION' harness/ graders/ tests/ docs/ 2>/dev/null
```
