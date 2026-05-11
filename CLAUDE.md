# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`causal-llm-eval` is a black-box evaluation framework that measures how LLM agents make methodology choices in causal inference tasks, and whether library design (specifically LLM-targeted guidance surfaces like `llms.txt`, fit-time warnings, native diagnostics, and pedagogical docstrings) measurably affects those choices.

The framework spawns truly cold-start Claude Code agents on causal inference tasks, captures three-layer telemetry (stream-JSON event log, in-process Python instrumentation, subprocess stderr), and grades the resulting choices against a pre-defined rubric. The eval's central scientific claim depends on the integrity of the agent runs and the completeness of the telemetry.

The first study (Phase 1 case study) compares diff-diff vs statsmodels on a synthetic staggered-adoption DGP, with N=15 cold-start agents per arm. The single sharp claim: library design measurably shifts which estimator LLM agents pick. The pre-defined fallback claim: library design measurably shifts which diagnostics agents run.

## Repo layout

```
harness/        # cold-start agent runner, telemetry, venv pool, extractor
graders/        # AI judge that applies the rubric to transcripts
prompts/        # versioned task prompts (prompts/case_study/v1.txt etc.)
rubrics/        # versioned grading rubrics (rubrics/case_study_v1.yaml etc.)
datasets/       # synthetic DGPs + ground-truth metadata sidecars + calibration logs
runs/           # per-run records (mostly gitignored)
analysis/       # cell summaries, variability reports, reproducibility checks
writeups/       # case-study writeup drafts
```

## Common Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Format code
black harness graders analysis tests

# Lint code
ruff check harness graders analysis tests

# Type checking
mypy harness graders analysis

# Smoke test (verifies cold-start invocation, runs the inheritance probe)
make smoke

# Run the Phase 1 case study end-to-end
make case-study-v1

# Calibrate the synthetic DGP
make calibration

# Run the comparator-asymmetry pre-flight
make preflight
```

## Key Design Patterns

1. **Cold-start agent runner**: Spawn `claude --bare --setting-sources "" --strict-mcp-config --disable-slash-commands --print --output-format stream-json --add-dir <tmpdir>` in a fresh tmpdir with a per-run venv. The `--bare` flag is load-bearing; without it the spawned agent inherits operator state (`$HOME/.claude/CLAUDE.md`, auto-memory, plugins, keychain). The runner ALSO pins `cwd=<run tmpdir>` and `env=clean_env` (an explicit allowlist of variables, not a denylist) so operator state cannot leak via `$HOME`, `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, AWS/MCP/GitHub env, etc. See `harness/COLD_START_VERIFICATION.md` for the full env contract. Verified by `make smoke`'s inheritance probe.

2. **Three-layer telemetry**: Every run captures (a) stream-JSON event log from Claude Code (transcript + tool calls + file reads), (b) in-process Python instrumentation via `sitecustomize.py` (logs `import diff_diff`, `get_llm_guide(variant)`, fit-time `warnings.warn`, diagnostic method calls, estimator instantiations), and (c) subprocess stderr capture. Stream-JSON alone misses Python-internal access; the in-process layer is the discoverability ground truth.

3. **Per-run venvs**: Each run gets a fresh venv with one library installed (diff-diff XOR statsmodels for Phase 1). For Phase 2's larger sample sizes, pre-built per-arm venv templates are cloned per run; never mutated post-clone.

4. **Versioned prompts**: Once a prompt is recorded against runs, it is immutable. New prompt = new version file (e.g., `v2.txt`), not in-place edit. Captured per run via `prompt_version` metadata field.

5. **Pre-defined rubrics**: Grading rubrics are written before runs, not tuned after seeing transcripts. Same rubric across arms.

6. **Two-stage extraction**: Deterministic (from in-process instrumentation log) for ground-truth signals like estimator class instantiated; AI judge (separate Claude API call) for transcript-derived signals like estimator-choice classification and reasoning. Disagreements flagged for spot-check.

7. **Reproducibility schema**: Per-run records pin library version (PyPI), claude binary version, model id, dataset SHA, prompt version, harness git SHA, random seed. `make case-study-v1` validates re-runs fall within documented tolerances.

## Documenting Deviations (AI Review Compatibility)

The AI PR reviewer recognizes deviations as documented (and downgrades them to P3) ONLY when they use specific label patterns. Using different wording will cause a P1 finding ("undocumented architectural deviation").

**Recognized labels** - use one of these in the relevant code section, PR description, or plan file:

| Label | When to use | Example |
|-------|------------|---------|
| `**Note:** <text>` | Defensive enhancements, implementation choices | `**Note:** Per-run venvs use shallow clone for speed; equivalent to per-run install for our purposes.` |
| `**Deviation from plan:** <text>` | Intentional differences from the locked plan | `**Deviation from plan:** Telemetry layer 2 written via decorator instead of sitecustomize because the venv pool clones don't carry sitecustomize.py reliably.` |

## Testing Conventions

- **Cold-start probe**: Tests that invoke the harness must include a probe to verify no inheritance from operator state. The probe asks the spawned agent "what skills/memory/CLAUDE.md do you have access to?" and asserts the response reports none.
- **Behavioral assertions**: Always assert expected outcomes, not just no-exception. Bad: `result = func(bad_input)`. Good: `result = func(bad_input); assert result.estimator_class is None`.
- **`@pytest.mark.live`**: Tests that hit the live Claude API or spawn real `claude` subprocesses are slow/expensive. Mark with `@pytest.mark.live` and exclude by default. Run with `pytest -m live` to include.
- **`@pytest.mark.slow`**: Tests that run in > 30 seconds; excluded by default.

## Key Reference Files

| File | Contains |
|------|----------|
| The latest plan file in `~/.claude/plans/` | Phase scope, locked architectural decisions, treatment design, telemetry layers, reproducibility schema. **Consult before harness changes.** |
| `.claude/commands/dev-checklists.md` | Checklists for configuration parameters, eval-validity code, warnings, reviews, bug fixes (run `/dev-checklists`) |
| `harness/COLD_START_VERIFICATION.md` | How to verify the cold-start probe works end-to-end |
| `prompts/case_study/v1.txt` | The Phase 1 case-study prompt (versioned, immutable once recorded) |
| `rubrics/case_study_v1.yaml` | The Phase 1 grading rubric (versioned, pre-defined before runs) |

## Workflow

- CI tests are gated behind the `ready-for-ci` label. The `CI Gate` required status check enforces this - PRs cannot merge until the label is added. **Phase 0 status**: only the label gate is implemented; the actual test workflow (`pytest`, `make smoke`) lands in a follow-up PR alongside the harness implementation. Until then, the gate enforces the convention but does not run tests.
- For non-trivial tasks, use `EnterPlanMode`. Consult the latest plan in `~/.claude/plans/` for locked decisions.
- For bug fixes, grep for the pattern across all files before fixing.
- Follow the relevant development checklists (run `/dev-checklists`).
- Before submitting: run `/pre-merge-check`, then `/ai-review-local` for pre-PR AI review.
- Submit with `/submit-pr`.

## Plan Review Before Approval

When writing a new plan file (via EnterPlanMode), update the sentinel:
```bash
echo "<plan-file-path>" > ~/.claude/plans/.last-reviewed
```

Before calling `ExitPlanMode`, offer the user an independent plan review via `AskUserQuestion`:
- "Run review agent for independent feedback" (Recommended)
- "Present plan for approval as-is"

**If review requested**: Spawn review agent (Task tool, `subagent_type: "general-purpose"`) to read `.claude/commands/review-plan.md` and follow Steps 2-5. Display output in conversation. Save to `~/.claude/plans/<plan-basename>.review.md` with YAML frontmatter (plan path, timestamp, assessment, issue counts). Update sentinel. Collect feedback and revise if needed. Touch review file after revision to avoid staleness check failure.

**If skipped**: Write a minimal review marker to `~/.claude/plans/<plan-basename>.review.md`:
```yaml
---
plan: <plan-file-path>
reviewed_at: <ISO 8601 timestamp>
assessment: "Skipped"
critical_count: 0
medium_count: 0
low_count: 0
flags: []
---
Review skipped by user.
```
Update sentinel. The `check-plan-review.sh` hook enforces this workflow.
