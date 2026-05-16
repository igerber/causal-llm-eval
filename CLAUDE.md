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

1. **Cold-start agent runner**: Spawn `claude --bare --setting-sources "" --strict-mcp-config --disable-slash-commands --print --output-format stream-json --verbose --permission-mode bypassPermissions --add-dir <tmpdir> --model <model> <prompt>` in a fresh tmpdir with a per-run venv. PR #6 added `--verbose` (required by claude CLI 2.1.143+ with `--print --output-format=stream-json`), `--permission-mode bypassPermissions` (no TTY in `--print` mode to approve Bash; agent runs in sandboxed tmpdir so bypass doesn't leak operator state), `--model <model>` (pin so CLI defaults can't drift), and the `--add-dir <tmpdir>` BEFORE `--model` ordering (variadic `--add-dir` would otherwise consume the prompt). See `harness/COLD_START_VERIFICATION.md` for the full rationale. The `--bare` flag is load-bearing; without it the spawned agent inherits operator state (`$HOME/.claude/CLAUDE.md`, auto-memory, plugins, keychain). The runner ALSO pins `cwd=<run tmpdir>` and `env=clean_env` (an explicit allowlist of variables, not a denylist) so operator state cannot leak via `$HOME`, `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, AWS/MCP/GitHub env, etc. The in-process shim's event log path is communicated via `_PYRUNTIME_EVENT_LOG` (underscore prefix + Python-runtime framing — deliberate low-reactivity naming so the agent is less likely to flag the env var as eval-related if they enumerate `os.environ`). See `harness/COLD_START_VERIFICATION.md` for the full env contract. Verified by `make smoke`'s inheritance probe.

2. **Three-layer telemetry**: Every run captures (a) stream-JSON event log from Claude Code (transcript + tool calls + file reads), (b) in-process Python instrumentation via `_pyruntime_shim.py` + `_pyruntime_shim.pth` installed into the venv site-packages (the `.pth`-based load survives Homebrew Python's stdlib-level `sitecustomize.py`; PR #6 fix). The shim logs `import diff_diff` / `import statsmodels`, `get_llm_guide(variant)`, fit-time `warnings.warn`, diagnostic method calls, estimator instantiations, and (for statsmodels) post-fit results-method calls. Every library-surface event (estimator init/fit, diagnostic call, results-method call, fit-time warning, guide-file read) carries a `library` attribution field (`"diff_diff"` or `"statsmodels"`); the merger's schema validator REJECTS records on these event types that omit `library` or carry an unrecognized arm, so cross-arm bleed-through cannot inflate either arm's counts. Structural events (`session_start`, `session_end`, `module_import`) do NOT carry `library`. (c) subprocess stderr capture. Stream-JSON alone misses Python-internal access; the in-process layer is the discoverability ground truth.

3. **Per-run venvs**: Each run gets a fresh venv with one library installed (diff-diff XOR statsmodels for Phase 1). PR #5 implements `harness.venv_pool.build_arm_template`: per-run fresh venv at `tmpdir/venv` with the arm library pip-installed at the pinned PyPI version, the layer-2 in-process shim installed as `_pyruntime_shim.py` + `_pyruntime_shim.pth` in site-packages (PR #6 fix; the `.pth` load survives Homebrew Python's stdlib-level `sitecustomize.py`), and a layer-1.5 `python` wrapper installed at `bin/python*` (real interpreter hidden at `${venv}/.pyruntime-real/.actual-python` behind a strip-S shim). The runner prepends `${venv}/bin/` to PATH so any `python` invocation routes through the wrapper. For Phase 2's larger sample sizes, `clone_for_run` will replace fresh-build with per-run clones of a pre-built template (PR #6+; currently a `NotImplementedError` stub). Both arms are instrumented as of PR #7 (`diff-diff==3.3.2` and `statsmodels==0.14.6`).

4. **Versioned prompts**: Once a prompt is recorded against runs, it is immutable. New prompt = new version file (e.g., `v2.txt`), not in-place edit. Captured per run via `prompt_version` metadata field.

5. **Pre-defined rubrics**: Grading rubrics are written before runs, not tuned after seeing transcripts. Same rubric across arms.

6. **Two-stage extraction**: Deterministic (from in-process instrumentation log) for ground-truth signals like estimator class instantiated; AI judge (separate Claude API call) for transcript-derived signals like estimator-choice classification and reasoning. Disagreements flagged for spot-check. The Phase 1 grading rubric is committed at `rubrics/case_study_v2.yaml` (PR #7); the judge implementation is deferred to PR #9.

7. **Reproducibility schema**: Per-run records pin library version (PyPI), claude binary version, model id, dataset SHA, prompt version, rubric version, harness git SHA, random seed, run_id, and arm. PR #6 wired emission via `harness.runner.run_one()` writing `output_dir/metadata.json` ON CLEAN EXIT ONLY. `RunMetadata.__post_init__` validates field formats so a malformed record can't be silently constructed. `make case-study-v1` (future) will validate re-runs against documented tolerances using the now-emitted `metadata.json`.

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
| `prompts/case_study/v2.txt` | The Phase 1 case-study prompt (active; PR #7). `v1.txt` is a reserved stub per PR #6 — do not edit in place. |
| `rubrics/case_study_v2.yaml` | The Phase 1 grading rubric (active; PR #7). `case_study_v1.yaml` is a reserved stub — do not edit in place. |

## Workflow

- CI tests are gated behind the `ready-for-ci` label. The `CI Gate` required status check enforces this - PRs cannot merge until the label is added. The `.github/workflows/tests.yml` workflow runs `pytest` (default excludes `slow` and `live`) on labeled PRs and on push to main. `make smoke` (the live inheritance probe) is a developer-only command, not a CI hook (it costs ~$0.05 per invocation).
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
