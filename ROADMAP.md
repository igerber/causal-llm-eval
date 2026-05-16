# Roadmap

Planned features and deliverables for `causal-llm-eval`.

For tech debt and deferred items, see [TODO.md](TODO.md).

## Current state

Cold-start infrastructure is fully wired: the runner (`harness.runner.run_one`) spawns `claude --bare` into a per-run venv with the arm library + layer-2 in-process shim (installed as `_pyruntime_shim.py` + `_pyruntime_shim.pth` in site-packages; the `.pth`-based load survives Homebrew Python's stdlib-level `sitecustomize.py`) + layer-1.5 `python` wrapper installed, captures three independent telemetry layers, and the merger (`harness.telemetry.merge_layers`) cross-validates them. A build-time sentinel proves wrapper + shim wiring before each agent runs. PR #6 added the synthetic DGP generator and the per-run reproducibility-anchored `metadata.json` sidecar; the runner now copies `RunConfig.dataset_path` into the per-run tmpdir under strict-reject regular-file validation.

The next substantive deliverable is the **statsmodels arm instrumentation + case-study prompt** (PR #7), which together with the existing DGP + per-run-record pipeline make `make case-study-v1` runnable.

## Recently shipped

- **2026-05-10** | PR #1 (`b44f51f`) | Bootstrap Phase 0: CI/AI-review infrastructure, harness skeleton, locked architectural decisions in CLAUDE.md
- **2026-05-10** | PR #2 (`002ad44`) | Phase 0.5 cleanup: adapt three ported `.claude/commands/` files for eval-research domain; introduce TODO.md (tech debt) and ROADMAP.md (planned features)
- **2026-05-12** | PR #3 (`443dbfd`) | Cold-start agent runner + inheritance probe + live `make smoke`. Implements `harness.runner.run_one()` and `harness.probe.run_probe()` with full subprocess hygiene (clean_env, cwd, killpg-on-timeout), tmpdir-local event log, exit-code sentinels, and the fail-closed probe contract.
- **2026-05-14** | PR #4 (`766d3f2`) | Three-layer telemetry capture: bashlex AST layer-1 parser (`harness/shell_parser.py`), in-process sitecustomize shim (`harness/sitecustomize_template.py`), and merger (`harness/telemetry.py::merge_layers`). 36 CI review rounds; closes the wrapper-attribution enumeration class structurally.
- **2026-05-14** | PR #5 | Per-arm venv pool + sitecustomize install + layer-1.5 exec wrapper. `harness.venv_pool.build_arm_template` builds a fresh venv per run with the arm library, shim, and `python_wrapper.sh` installed; the wrapper emits `exec_python` events to the layer-2 log; the merger cross-checks layer-1 ↔ layer-1.5 ↔ layer-2; a build-time sentinel proves wiring before agent spawn. Adds `tests/test_telemetry_live.py` (first end-to-end live attestation test).
- **2026-05-15** | PR #6 | Synthetic DGP generator + per-run dataset copy + `metadata.json` sidecar emission. `harness/dgp.py::generate_case_study_v1` wraps `diff_diff.generate_staggered_data` with locked Phase 1 starter parameters (uncalibrated; calibration loop is next) and persists `datasets/case_study_v1/{data.parquet,dgp_truth.json,README.md}`. Bit-identical parquet writes via a shared `_deterministic_write_parquet` helper (validated empirically at pyarrow 24.0.0). `run_one()` now validates `dataset_path` strictly (regular files only; symlinks rejected via `lstat`), copies it into `tmpdir/data.parquet`, and emits `metadata.json` only on clean exit (absence is the failed-run signal). `RunConfig.rubric_version` is now required; `RunMetadata.random_seed` is `int | None`; `RunMetadata.__post_init__` validates field formats. Closes both High-priority "PR #6+" TODO rows.

## Shipping next

The items below are sequenced in rough build order, but several can run in parallel once the runner lands. Each becomes a focused PR.

- **Statsmodels arm instrumentation** - wires the in-process shim's hook surface for `statsmodels.regression.linear_model.OLS`, `statsmodels.regression.linear_model.OLSResults`, and any other classes the case-study statsmodels prompt exercises. Couples to the statsmodels case-study prompt PR. Until this lands, `arm="statsmodels"` records carry all-False non-guide flags by construction.
- **Phase 2 venv template + clone-per-run** - `clone_for_run` becomes real (currently a deferred stub). Phase 1 (PR #5) builds a fresh venv per run (~10-30s overhead). Phase 2 builds the template once per pytest session / per cell, then clone-per-run cuts the per-run cost to <1s. Lands when eval volume justifies the optimization.
- **Layer-1 transcript parsing in merger** - extends `telemetry.merge_layers()` to parse the stream-JSON transcript for Read-tool calls on guide files and Bash-level estimator invocations that the in-process shim missed (e.g., agent reads `llms.txt` via Claude's Read tool without running Python that touches `importlib.resources`). Refines the tri-state discoverability flags. Pairs with the judge-prompt PR so we know which transcript evidence the rubric weighs.
- **Deterministic result extractor** - parses the per-run in-process event log into structured signals (estimator class, diagnostics invoked, warnings observed) for the two-stage extraction pipeline.
- **AI judge against the rubric** - separate Claude API call (not Claude Code) that applies the versioned rubric to a transcript + final code, returns structured JSON, supports the spot-check protocol for human-vs-judge agreement.
- **DGP calibration loop (`make calibration`)** - tunes the locked Phase 1 starter parameters in `harness/dgp.py::_DGP_CALL_KWARGS`; bumps `CASE_STUDY_V1_DGP_VERSION` and regenerates the committed `datasets/case_study_v1/` artifact. Synthetic DGP generator landed in PR #6; calibration is the next step before locking the case study.
- **Comparator-asymmetry pre-flight (`make preflight`)** - runs 3-5 sample agents per arm; checks the statsmodels arm spends < 60% of tokens on construction; documents the prompt-amendment escape hatch if asymmetry persists.
- **Case-study v1 end-to-end (`make case-study-v1`)** - runs all 30 cells (15 per arm), produces structured records, cell summaries, primary + fallback claim outcomes. Includes the reproducibility-schema check for re-runs.
- **Case-study v1 writeup** - ~1500-word draft leading on framework + methodology, reporting both claim outcomes regardless of direction. Single sharable artifact (blog post or LinkedIn or repo README).

## Under consideration

Items the framework might add depending on Phase 1 findings:

- **"diff-diff without `llms.txt`" arm** - Phase 2 candidate to isolate library-design vs guide-content. Requires a controlled mechanism to bundle-strip or env-disable the guides without other library changes.
- **Cross-prompt-framing variation** - multiple prompt versions per dataset to measure prompt-sensitivity within an arm.
- **Additional comparator libraries** - econml, dowhy, fixest (via rpy2 or reticulate) as Phase 2 study subjects to extend the comparator ladder.
- **Doc-impact dependency map** - the analog of diff-diff's `docs/doc-deps.yaml` for this repo, wired into `/pre-merge-check` and `/push-pr-update`.

## Long-term research directions

Phase 2 (arxiv-stage) and Phase 3 (journal-stage) capabilities, framed as research directions rather than committed deliverables:

- **Pre-registered confirmatory study** - comparison ladder of 4-5 libraries, multiple datasets (synthetic + classic + newer with lower training-data contamination risk), N=20-30 per cell, multi-rater grading with reported inter-rater reliability (Cohen's kappa), full reproducibility kit. Pre-registered on OSF before runs.
- **Cross-model + cross-vendor sensitivity** - extends the framework to Claude Sonnet and Haiku (cheap), and possibly across vendors (GPT, Gemini) at higher cost. Tests whether library-design effects generalize beyond Claude Opus.
- **Human-expert control arm** - how do LLM agents compare to applied econometricians on the same task? Requires recruiting and instrumenting a small cohort; out of scope for the early phases but flagged as the version of this study that gets a much wider audience.
