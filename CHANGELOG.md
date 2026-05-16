# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (PR #7)
- Statsmodels arm instrumentation in `harness/sitecustomize_template.py`:
  hooks for `OLS`, `WLS`, `GLS`, `GLSAR`, `RLM`, `GLM`, `MixedLM`, `Logit`,
  `Probit` (estimator `__init__` + `fit`); statsmodels-relevant diagnostic
  functions (`het_breuschpagan`, `het_white`, `linear_reset`,
  `acorr_breusch_godfrey`, `acorr_ljungbox`, `durbin_watson`); and post-fit
  results methods (`RegressionResults.summary`,
  `RegressionResults.get_robustcov_results`). Result-method wrapper records
  `type(self).__name__` at call time, so OLSResults / RLMResults /
  MixedLMResults instances surface as the correct class without enumerating
  every subclass. `_attach_statsmodels_hooks` walks each submodule and
  patches the target attribute; `_StatsmodelsPostImportHook` (mirrored
  architecture of `_DiffDiffPostImportHook`) triggers attachment when
  `statsmodels` is imported.
- `library: str` attribution field on every event the shim emits
  (`"diff_diff"` or `"statsmodels"`). The merger's record-builders filter
  events by `library` rather than by filename substring, eliminating
  cross-arm bleed-through. Backward compatible: events lacking `library`
  default to `"diff_diff"` so PR #5/#6 records on disk remain parseable.
- New event type `estimator_diagnostic_method` (distinct from
  `estimator_init` / `estimator_fit` / `diagnostic_call`) for post-fit
  results-method calls.
- `prompts/case_study/v2.txt`: Phase 1 case-study task prompt. Task-only
  by design — names no library and no estimator-class so both arms see
  identical text. Properties asserted by `tests/test_case_study_prompt.py`.
- `prompts/case_study/README.md`: documents v1 reserved-stub state, v2
  active prompt, and the new-version-on-edit policy.
- `rubrics/case_study_v2.yaml`: Phase 1 grading rubric (6 pre-defined
  criteria covering estimator classification, evidence, diagnostics,
  reasoning quality, confidence intervals, and final ATT estimate). The
  estimator-classification enum covers both arms in one rubric so a
  TWFE choice resolves correctly regardless of arm.
- `rubrics/README.md`: documents the v2 schema shape.
- ~15 new tests in `tests/test_sitecustomize.py` covering statsmodels
  hooks (OLS/OLSResults/het_breuschpagan), warning attribution via stack
  walk for both libraries, library-field presence on diff_diff events,
  result-method runtime-class recording, idempotency, and bidirectional
  constants regression.
- 7 new tests in `tests/test_telemetry_merger.py` covering
  `_build_statsmodels_record` field population (estimator classes,
  diagnostics from both event types, fit-time warnings), cross-arm
  event filtering, backward compatibility (events without `library`),
  and the sentinel-fields-stay-None invariant.
- 3 new `@pytest.mark.slow` tests in `tests/test_venv_pool.py` covering
  the statsmodels-arm template build end-to-end (correct library version
  installed, `_pyruntime_shim.py` + `_pyruntime_shim.pth` present,
  `module_import` event fires on `import statsmodels`).
- New test files `tests/test_case_study_prompt.py` (5 tests) and
  `tests/test_case_study_rubric.py` (8 tests).

### Changed (PR #7)
- `harness/venv_pool.py::_ARM_TO_PIP_PACKAGE` now includes
  `"statsmodels": "statsmodels"`; `build_arm_template` no longer raises
  `NotImplementedError` for `arm="statsmodels"`.
- `harness/telemetry.py::_build_statsmodels_record` now populates
  `estimator_classes_instantiated`, `diagnostic_methods_invoked`, and
  `saw_fit_time_warning` from real statsmodels-attributed events (was:
  hard-coded `False`/`()` since PR #4). `diagnostic_methods_invoked`
  aggregates both module-level `diagnostic_call` events and post-fit
  `estimator_diagnostic_method` events (the latter prefixed
  `<ClassName>.<method>` to disambiguate them from module-level functions).
  Filename-substring filter for warnings replaced by structural
  `library == "statsmodels"` check.
- `harness/telemetry.py::_build_diff_diff_record` now defensively filters
  events by `library in ("diff_diff", missing)` so a cross-arm bleed-through
  doesn't pollute the record. Backward-compat default preserves PR #5/#6
  record readability.
- `_caller_is_from_diff_diff` renamed and generalized to
  `_caller_is_from_library(start_frame, prefixes)`; returns the matched
  prefix so the warning hook can stamp the `library` field without a
  second stack walk. Called for both diff_diff and statsmodels frames.
- `_wrap_estimator_init` / `_wrap_estimator_fit` / `_wrap_diagnostic`
  decorators now take a keyword-only `library: str = "diff_diff"` argument;
  the diff_diff attach call sites pass it explicitly. The default
  preserves backward compatibility with any external caller.

### Added (PR #6)
- `harness/dgp.py`: synthetic data-generating process for the Phase 1
  case study. Wraps `diff_diff.generate_staggered_data` with locked
  starter parameters (uncalibrated; calibration loop is the next ROADMAP
  deliverable). Exposes `generate_case_study_v1(out_dir, seed=42)` plus
  a CLI entry point (`python -m harness.dgp <out_dir> [--seed N]`). Uses
  a shared `_deterministic_write_parquet` helper that pins
  `pyarrow.parquet.write_table` options + strips the pandas/pyarrow
  metadata blob so two invocations with the same seed produce
  bit-identical bytes (validated empirically at pyarrow 24.0.0). The
  helper is also reused by the probe so placeholder bytes don't narrate
  the harness's pandas/pyarrow versions.
- `datasets/case_study_v1/{data.parquet, dgp_truth.json, README.md}`:
  the committed Phase 1 case-study dataset (seed=42). The DGP's
  `true_effect` ground-truth column is intentionally DROPPED from
  `data.parquet` to preserve eval validity; ground truth lives only in
  `dgp_truth.json`. `dgp_truth.json` pins a 2-level shape for
  `true_effects_per_event_time_per_cohort` (cohort_onset → event_time →
  tau) so downstream judges/extractors can target a stable schema.
- `rubrics/case_study_v1.yaml` + `rubrics/README.md` +
  `prompts/case_study/v1.txt`: stub registry files (with `# STUB ONLY -
  DO NOT FILL IN PLACE` headers) so `RunConfig.rubric_version` and
  `RunConfig.prompt_version` resolve to real on-disk files. The v1 IDs
  are RESERVED for the stub state; PR #7+ should write `*_v2` files
  rather than editing v1 in place (would silently invalidate any
  per-run records that reference v1).
- `RunConfig.rubric_version: str` (REQUIRED — no default). Mirrors
  `JudgeResult.rubric_version` semantically. Cascaded through
  `harness/probe.py`, `tests/test_runner.py::_config`, and the live
  test fixtures.
- `_harness_version()` helper in `harness/runner.py`: returns the harness
  git SHA, with a `-dirty` suffix when `git status --porcelain`
  (untracked included — a stray `.bak` could shadow imports) returns
  anything. Walks upward from `__file__` to find `.git` so editable
  installs work. 30s timeout for `git status` on slow disks.
- `_claude_version()` helper in `harness/runner.py`: returns the last
  non-empty line of `claude --version` (strips deprecation banners and
  node-engine warnings that may precede the version string in future CLI
  releases). Pinned `env={"PATH": os.environ.get("PATH", "")}` so the
  operator's `_PYRUNTIME_EVENT_LOG` cannot leak into the claude CLI
  subprocess and write stray events.
- `_validate_dataset_path()` helper: strict-reject regular-file
  validation. Catches symlinks via `lstat()` (NOT `stat()`, which
  follows symlinks). Rejects symlink, directory, device, FIFO, socket,
  NUL byte in path, and missing files. Runs at the TOP of `run_one()`,
  BEFORE `tempfile.mkdtemp()`, so failures don't leak tmpdirs.
- `_copy_dataset_into_tmpdir()` helper: copies the validated dataset
  into `tmpdir/data.parquet` via `shutil.copy2` (cross-device-safe) and
  returns the sha256 of the COPIED bytes (the artifact-of-record). The
  agent reads from `tmpdir/data.parquet` (top-level for relative-path
  simplicity).
- `harness.runner.run_one()` now writes `output_dir/metadata.json`
  ONLY on clean exit (`exit_code == 0` AND `not descendants_live` AND
  no `telemetry_missing` sentinel). Absence is the signal that the run
  did not complete cleanly. Fields: harness_version, library_version,
  claude_code_version, model_version, dataset_sha, prompt_version,
  rubric_version, random_seed (JSON null when None), run_id, arm. Keys
  are sort_keys=True + indent=2 for byte-stability across runs.
- `RunMetadata.__post_init__` validation: `harness_version` matches
  `[0-9a-f]{40}(-dirty)?`, `dataset_sha` is 64-hex sha256,
  `claude_code_version` is non-empty, `arm` is `diff_diff` or
  `statsmodels`. Catches malformed metadata at construction time.
- `RunResult.dataset_in_tmpdir_path` and `RunResult.metadata_json_path`
  fields. `metadata_json_path` is `None` on failure paths (matches the
  on-disk absence).
- `make dgp` target: `python -m harness.dgp datasets/case_study_v1
  --seed 42` (regenerates the committed artifact in-place; idempotent
  given a stable pyarrow version).
- ~25 new tests in `tests/test_dgp.py` and `tests/test_runner.py`
  covering: parquet bit-identity (single-platform), seed-changes-bytes
  sanity, dgp_truth schema shape, committed-vs-regenerated dataset
  match, dtype pinning, `RunConfig.rubric_version` requiredness,
  `RunMetadata` post-init validation rules, dataset rejection family
  (symlink / directory / FIFO / NUL byte / missing), pre-tmpdir
  validation ordering, metadata.json round-trip, metadata.json key
  order stability, metadata.json suppression on each failure path
  (timeout / descendants_live / telemetry_missing), `_claude_version`
  multi-line stripping, `_claude_version` env hygiene (no
  `_PYRUNTIME_EVENT_LOG` leak), `_harness_version` `.git` walk-up,
  `_harness_version` dirty-suffix, `_harness_version` no-repo
  fail-closed.

### Fixed (PR #6)
- Locked invocation now passes `--permission-mode bypassPermissions`. In
  `--print` mode there is no TTY for the operator to approve Bash
  invocations, so any tool requiring approval blocks at "This command
  requires approval" and the agent reports the call was not executed
  (which broke the probe's structural python command). The agent runs
  in a sandboxed tmpdir with HOME=tmpdir + clean_env, so bypassing
  permissions does not leak operator state — it just lets the agent
  actually use its tools.
- Probe blacklist no longer includes `project_` (false-fires on
  `CLAUDE_PROJECT_DIR`, which is the env-var name the probe prompt
  itself recites in the structural python command). `feedback_` and
  `user_role` remain as auto-memory-file-name signatures.
- Locked invocation argv reordered: `--add-dir <tmpdir>` is now followed
  by `--model <model>` rather than the prompt. Claude CLI's `--add-dir
  <dirs...>` is variadic; with the prompt immediately after `--add-dir
  <tmpdir>`, claude was consuming the prompt as a second directory,
  leaving `--print` with no prompt arg and blocking on stdin for ~30s
  before exiting 0 with a 0-byte transcript. Reordering forces the
  variadic to terminate at the next flag. Defense-in-depth:
  `subprocess.Popen(..., stdin=subprocess.DEVNULL)` so any future
  variadic-args regression causes claude to bail immediately rather
  than hang.
- Locked invocation now passes `--verbose` alongside `--print
  --output-format stream-json`. Claude CLI 2.1.143+ requires this flag
  combination; without it the CLI produces a 0-byte transcript and
  exits 0 (silent on `--bare`; explicit "requires --verbose" error on
  the non-bare path), which the runner cannot distinguish from "agent
  emitted nothing." `--verbose` does not affect cold-start isolation —
  it just makes the streaming output actually stream. Verified by
  `make smoke` against claude 2.1.143.
- Shim auto-load now uses a `.pth` file (`_pyruntime_shim.pth` containing
  `import _pyruntime_shim`) instead of `sitecustomize.py`. Homebrew's
  `python@3.13` (Feb 2026+) ships a stdlib-level `sitecustomize.py` that
  takes precedence over any `sitecustomize.py` installed in venv
  site-packages because stdlib comes before site-packages in `sys.path`,
  silently breaking the layer-2 sitecustomize shim's load. The `.pth`
  file is processed by Python's site machinery during initialization
  (BEFORE `execsitecustomize` runs) regardless of whether stdlib has its
  own sitecustomize, restoring the load. `harness.venv_pool` installs
  the template as `_pyruntime_shim.py` + `_pyruntime_shim.pth` (no longer
  as `sitecustomize.py`); the `__name__` gate in
  `harness/sitecustomize_template.py` matches `_pyruntime_shim`.
  Verified by `make smoke` against a Homebrew-affected machine.

### Changed (PR #6)
- `harness.probe.run_probe()` now materializes a 1-row placeholder
  parquet inside its own output_dir (via the shared
  `_deterministic_write_parquet` helper from `harness.dgp`) and passes
  that path as `RunConfig.dataset_path`, replacing the prior
  `Path("/dev/null")` placeholder which the new strict-reject runner
  validation would reject (character device).
- `RunMetadata.random_seed` annotation: `int` → `int | None`. JSON
  `null` represents "no harness-side seed configured for this run."
  Replaces a planned `-1` sentinel that would have collided with
  realistic future calibrated seeds.
- `harness.runner.run_one()` overwrite-refusal guard now includes
  `metadata.json` alongside `transcript.jsonl`,
  `in_process_events.jsonl`, and `cli_stderr.log`.

### Added
- `harness/venv_pool.py::build_arm_template` (was a Phase 0 stub):
  builds a fresh per-run venv at `tmpdir/venv`, pip-installs the arm
  library at the pinned version, copies `harness/sitecustomize_template.py`
  into the venv's `site-packages` as `sitecustomize.py`, and installs the
  layer-1.5 `python` wrapper at every interpreter alias (`python`,
  `python3`, `python3.X`) with the original interpreter preserved at
  `python-real`. Closes the PR #4-deferred "no per-arm venv pool" gap.
- `harness/python_wrapper.sh`: layer-1.5 exec-time wrapper around the
  per-arm venv's `python`. POSIX shell (no bashisms; tested with
  dash / busybox sh / macOS bash 3.2 in POSIX mode). Appends a single
  JSONL `exec_python` event (pid, ppid, ts, executable, argv) to
  `_PYRUNTIME_EVENT_LOG` then `exec`s the real interpreter at
  `${VENV}/.pyruntime-real/python-real`. Fail-closed (exit 2) on write failure.
- `exec_python` event type in the layer-2 event log schema.
  `harness/telemetry.py` validates required fields (pid, ppid, ts,
  executable, argv) and, when `merge_layers` is called with `venv_path`,
  enforces a venv-root-anchored allowlist on `executable` so an agent
  cannot forge events pointing at arbitrary binaries.
- Three-layer cross-check (`_validate_three_layer_consistency`) in the
  merger: when `merge_layers` is called with `runner_pid`, every layer-1
  AST-extracted python invocation must have at least one matching agent
  `exec_python` event AND at least one matching `session_start` event by
  `argv[1:]`. Cardinality is N-to-many on the runtime side (xargs / find /
  parallel can spawn N python children for one layer-1 invocation).
  argv[0] is excluded from the match key (wrapper records basename
  `python`; sitecustomize records `${venv}/.pyruntime-real/python-real` after exec).
- Build-time python sentinel in `harness.runner.run_one`: a direct
  `subprocess.run([${venv}/bin/python, "-c", "pass"])` runs BEFORE the
  `claude --bare` subprocess, with `_PYRUNTIME_EVENT_LOG` set, so the
  wrapper + shim always produce at least one `exec_python` +
  `session_start` event in the log. The merger demands at least one
  sentinel event (`ppid == runner_pid`) and uses ppid to partition
  sentinel-vs-agent events. Closes the shell-only-agent gap (a run with
  no agent-invoked python is still attestable via the sentinel).
- `if __name__ == "sitecustomize"` gate in
  `harness/sitecustomize_template.py`. Production load (Python's site
  machinery loads the file as `sitecustomize` from the venv's
  site-packages) fires the existing top-level side effects unchanged;
  test / docs imports as `harness.sitecustomize_template` skip the
  gate. Closes the PR #4-deferred "import side-effects" TODO row.
- `tests/test_telemetry_live.py`: first end-to-end live test of the
  three-layer attestation chain. Spawns a real `claude --bare`
  subprocess, asserts the build-time sentinel + a `session_start` event
  both fire, and asserts `merge_layers` validates the run successfully
  with `runner_pid` + `venv_path`. Cost: roughly one short Claude
  invocation worth of API spend per CI cycle (gated by
  `@pytest.mark.live`).
- `tests/test_venv_pool.py`: 13 tests for `build_arm_template` using a
  session-scoped venv fixture (one ~30s build shared across 11 tests; 2
  tests need no venv). Marked `slow` at file level.
- 12 new three-layer cross-check tests in `tests/test_telemetry_merger.py`
  covering: consistent attribution, shell-only-agent + sentinel,
  missing-sentinel failure mode, forged-executable rejection,
  layer-1.5/layer-2-missing detection, argv[1:] match strictness,
  N-to-many xargs cardinality, malformed-event schema rejection,
  legacy-mode (no runner_pid) backward compatibility.
- 3 new gate tests in `tests/test_sitecustomize.py` covering the
  `__name__` gate behavior (no side effects on plain importlib import,
  side effects via `runpy.run_path(run_name="sitecustomize")`,
  defensive other-name skip).

### Changed
- `harness.runner.run_one` now builds a per-arm venv at `tmpdir/venv`
  on every run, prepends the venv's `bin/` to the spawned process's
  PATH so the agent's `python` resolves to the wrapper, and records
  `venv_path` + `runner_pid` on `RunResult` for the merger.
- `harness.runner.RunResult` gained two optional fields: `venv_path`
  (passed to `merge_layers` for the venv-root-anchored `executable`
  check) and `runner_pid` (passed for the sentinel-vs-agent ppid
  partition).
- `harness.telemetry.merge_layers` accepts two new keyword-only
  arguments: `runner_pid` (enables the three-layer cross-check when
  provided) and `venv_path` (enables the venv-root-anchored `executable`
  schema check). Backward-compatible: existing 138 merger fixtures
  continue to pass without supplying either.
- `tests/test_sitecustomize.py::_import_shim_fresh()` now uses a
  dual-path approach: `runpy.run_path(template, run_name="sitecustomize")`
  fires the gated side effects, then `importlib.import_module(...)`
  returns the module object for attribute access. Gate-populated state
  (`_EVENT_LOG_FD`, `_EVENT_LOG_PATH`, `_initial_path`) is bridged onto
  the module via setattr so the existing 18 attribute-accessing tests
  continue to work unchanged.
- `harness.probe.run_probe` now passes `library_version="3.3.2"` to
  `RunConfig` so `build_arm_template` has a real PyPI version to
  install. Previously `library_version="n/a"` was a placeholder valid
  only because no per-arm venv was built.
- `tests/test_runner_live.py::test_run_one_spawns_real_agent_with_trivial_prompt`
  similarly bumped to `library_version="3.3.2"` and `timeout_seconds=300`
  (venv build adds ~10-30s to the cold-start path).

### Removed
- TODO row "Command-delegating exec forms (`find -exec python`,
  `xargs python`, `parallel python`)" - layer-1.5 exec wrapper closes
  this class structurally (any python invocation hits the wrapper
  regardless of the parent shell form).
- TODO row "`harness/sitecustomize_template.py` runs top-level side
  effects on import" - the `if __name__ == "sitecustomize"` gate
  closes this.
- TODO row "End-to-end live test of the shim firing inside a
  `claude --bare` subprocess" - `tests/test_telemetry_live.py` closes
  this now that per-arm venv install is wired.

### Added (PR #4)
- `harness/shell_parser.py`: bashlex-AST-based Bash command parser for
  layer-1 attestation. Public surface: `parse_python_invocations`,
  `find_python_bypass_invocations`, `argv_contains_bypass_flag`.
  Replaces ~600 lines of regex-based extractors and wrapper-unwrappers
  that grew shape-by-shape across PR #4's 26 review rounds. The AST
  walker visits every CommandNode reachable from the parsed tree
  (including bodies of if/while/for, contents of `eval` /
  `bash -c "..."` payloads via recursive re-parse, and command
  substitutions in `$(...)` / backticks) so wrapper-attribution forms
  surface from language structure rather than from a hand-curated
  list of regex patterns.
- `RunValidityError` exception (neutral parent class).
  `TelemetryMergeError` is now a subclass. New parser-side subclasses
  `ShellCommandIndeterminate` (variable expansion / command-
  substitution / parameter expansion in a Python command-word or
  argv-word) and `ShellCommandParseError` (bashlex cannot model the
  command; e.g. `case` patterns) both inherit from `RunValidityError`,
  so callers catching the parent unify fail-closed handling across
  layer-1 and layer-2/3 failures.
- `bashlex==0.18` as a new runtime dependency (pinned exactly; upstream
  is dormant and AST-shape drift would silently break parsing).

### Changed
- Layer-1 Python-invocation extraction in `harness/telemetry.py` now
  delegates to `harness.shell_parser`. The previous regex extractors
  (`_extract_python_invocations_from_command`,
  `_unwrap_command_for_inspection`, `_strip_heredoc_bodies`,
  `_strip_command_modifier_prefix`, `_strip_shell_control_prefix`,
  `_is_in_command_position`, `_find_unquoted_separator`, plus ~20
  regex constants) are removed. The wrapper-attribution enumeration
  class (R20 command substitution, R23 P0#2 relative-slash paths, R25
  modifiers-after-separators, R26 quoted command words /
  path-qualified wrappers / sudo) closes as a category - no longer a
  list of patterns to maintain.
- Indeterminate command-words (variable expansion, command
  substitution, parameter expansion) and bashlex parse failures
  now raise `RunValidityError` subclasses. The merger fails closed
  rather than silently treating these as no-Python.

### Removed
- ~600 lines of regex-based shell-parsing helpers and their
  constants from `harness/telemetry.py`. Functionality is preserved
  in `harness/shell_parser.py` via AST walking. No external
  consumers existed (verified by audit).


- Cold-start agent runner (`harness.runner.run_one`) implementing the locked
  `claude --bare ...` invocation with `cwd=tmpdir`, `env=clean_env`, and
  pre-spawn writability check on the per-run event log path. `_PYRUNTIME_EVENT_LOG`
  is set to an absolute, resolved path so the in-process shim and the runner
  reference the same file regardless of how `output_dir` was passed.
- Cold-start inheritance probe (`harness.probe.run_probe`) with a separated
  output directory under `runs/probe/`. The probe never feeds eval data. Two
  assessment layers: self-report (blacklist + affirmative-no) and structural
  (verifies `cwd`, `home`, and env-key allowlist via a `python -c` one-liner
  the agent runs). Structural layer catches leaky cold-starts where the
  agent's self-report would miss the leak.
- `make smoke` target: live cold-start probe with `ANTHROPIC_API_KEY` fail-fast
  precheck. Costs ~$0.05 per invocation.
- `.github/workflows/tests.yml`: GitHub Actions test workflow running `pytest`
  (default excludes `slow` and `live`) on labeled PRs and push to main.
- `cli_stderr_log_path` on `RunResult` so downstream telemetry merging does
  not need to rely on filename convention.
- `--model` CLI flag passed to `claude --bare`, sourced from `RunConfig.model`,
  so default-model drift cannot affect runs.
- 35 unit tests for `harness.runner` and `harness.probe`, plus 2 live tests
  gated by `@pytest.mark.live`.
- In-process telemetry layer-2 hooks (`harness/sitecustomize_template.py`):
  guide-file reads via `get_llm_guide` (wrapped at both
  `diff_diff.get_llm_guide` and `diff_diff._guides_api.get_llm_guide` to close
  the submodule-direct-import bypass), guide-file reads via `builtins.open`
  and `io.open` (catches the `importlib.resources.files(...).read_text()` path
  that pathlib routes through `io.open`), fit-time `warnings.warn` filtered
  to `diff_diff.*`, estimator class instantiations and fits (21 classes
  including HonestDiD, PreTrendsPower, and LinearRegression), diagnostic
  function calls (5 functions), `session_start` event written at shim load.
  All hooks use `functools.wraps` and `_pyruntime_wrapped` idempotency
  markers. The shim opens the event-log fd once at startup
  (POSIX-resilient to chmod / rename / unlink of the path) and hard-exits
  the Python interpreter with `os._exit(2)` on event-write failure; the
  merger fails closed on shim markers in stderr, on `[pyruntime]` markers
  in Bash tool_result content, and on Bash `is_error=True` for any
  command containing a Python invocation. This closes the chain "shim
  fails -> merger detects" even when the agent suppresses stderr via
  `2>/dev/null`.
- `harness.telemetry.merge_layers()` real implementation: parses
  `in_process_events.jsonl` plus a layer-1 (`transcript.jsonl`)
  python-invocation count cross-check; raises new `TelemetryMergeError` on
  missing event log, malformed JSONL, the runner-written `telemetry_missing`
  sentinel, or python-invoked-but-shim-never-loaded. Returns a populated
  `TelemetryRecord` honoring arm-aware sentinel semantics for both
  `arm="diff_diff"` (bool flags) and `arm="statsmodels"` (None sentinels on
  guide fields; bool/tuple zeros on the rest until the statsmodels arm
  instrumentation lands).
- 16 unit tests in `tests/test_sitecustomize.py` covering hook builders,
  meta_path finder, `_guides_api` bypass closure, warning filter, fail-open
  contract, idempotency, bidirectional regression guards against
  `diff_diff.__all__`, and the 500-char message cap.
- 17 unit tests in `tests/test_telemetry_merger.py` covering merger happy
  paths (diff_diff + statsmodels), fail-closed inputs (missing file,
  malformed JSONL, sentinel, python-invoked-without-session_start), 0-byte
  events.jsonl, variant deduplication, open-vs-get_llm_guide flag
  attribution, python-invocation word-boundary regex including compound
  shell commands, and `_VARIANT_TO_FILENAME` dict-equality against
  `diff_diff._guides_api._VARIANT_TO_FILE`.
- `diff-diff>=3.3.2` added to `dev` extras in `pyproject.toml` so the
  bidirectional regression tests actually run in CI rather than skipping.

### Changed
- Per-invocation attribution now matches each transcript-visible python
  invocation against a `session_start` by `argv` (interpreter basename +
  args), not by `sys_executable` pool. Closes the `pip --version && python
  script.py` masking case where pip's `session_start` could satisfy the
  visible `python` token under unused-slot pooling, hiding `script.py`'s
  missing instrumentation. Surplus session_starts that have no visible
  counterpart (pip console-scripts, child processes) are still allowed.
- `merge_layers()` now rejects stream-JSON transcripts that lack a terminal
  successful `result` entry. A capture truncated mid-run (after a Read
  tool_use but before its tool_result, or after the last visible Bash
  command but before the final result frame) cannot be treated as
  complete: per-run evidence past the cut-off is silently missing.
- `merge_layers()` now fails closed when a guide-file Read tool_use has
  no matching tool_result in the transcript. Same fail-closed posture as
  the empty-transcript and missing-terminal-result checks: incomplete
  transcript means we cannot emit a definitive `opened_llms_*=False`
  without risking silent layer-1 loss.
- Renamed in-process telemetry env var from `CAUSAL_LLM_EVAL_EVENT_LOG` to
  `_PYRUNTIME_EVENT_LOG`. The underscore prefix + Python-runtime framing is a
  deliberate low-reactivity choice so an agent enumerating `os.environ` is less
  likely to flag the name as eval-related.
- Dropped the Phase 0 status caveat from `CLAUDE.md` and
  `harness/COLD_START_VERIFICATION.md`: the CI test workflow now exists.
- Default probe output directory now uses microsecond-resolution timestamps
  plus a short UUID suffix to prevent collision between same-second runs.
- Probe structural env check upgraded from a small denylist to a fail-
  closed schema + required-keys + denylist (explicit + substring + prefix)
  + narrow allowlist:
  - Schema: env_keys must be a non-empty list of strings; missing, empty,
    or malformed entries are findings.
  - Required: PATH, HOME, and `_PYRUNTIME_EVENT_LOG` must be present.
  - Explicit denylist: `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, `AWS_*`,
    `OPENAI_API_KEY`, `CODEX_HOME`, `ANTHROPIC_PROJECT_*`,
    `ANTHROPIC_AUTH_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`.
  - Deny substrings: `KEY`, `TOKEN`, `SECRET`, `OAUTH`, `PASSWORD`,
    `PASSWD` (overridden only by exact allowlist entries like
    `ANTHROPIC_API_KEY`).
  - Deny prefixes: `AWS_`, `CODEX_`, `MCP_`/`MCP`, `ANTHROPIC_PROJECT_`,
    `ANTHROPIC_OAUTH`, `CLAUDE_OAUTH`, `CLAUDE_MCP`, `CLAUDE_CONFIG`,
    `GITHUB_`, `GH_`.
  - Narrowed allow prefixes: `CLAUDE_CODE_`, `CLAUDECODE_` (the prior
    broad `CLAUDE_*` / `ANTHROPIC_*` allowance let `CLAUDE_OAUTH_TOKEN`,
    `ANTHROPIC_PROJECT_NAME`, etc. pass; the prior blanket `PYTHON*`
    allowance let `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP` pass —
    those alter import resolution or run startup code and are now in
    the explicit denylist).
  - Probe `run_probe()` now marks the assessment as failed and prepends
    a `cli_nonzero_exit` finding whenever `RunResult.exit_code != 0`,
    so `make smoke` cannot green-light a probe whose CLI failed.
  - Runner overwrite guard now refuses to overwrite `cli_stderr.log`
    in addition to `transcript.jsonl` and `in_process_events.jsonl`.
    All three layer-1/2/3 telemetry sinks are protected.
- Runner timeout now kills the full subprocess tree via
  `start_new_session=True` + `os.killpg(proc.pid, SIGKILL)` instead of only
  the parent. Stray Bash/Python children no longer linger after
  `RunResult(exit_code=-1)` is returned.
- In-process event log path no longer leaks the harness repo location to
  the agent. The shim writes to `tmpdir/.pyruntime/events.jsonl` during
  execution (so `_PYRUNTIME_EVENT_LOG` resolves inside the agent's
  tmpdir); after the subprocess exits the file is moved to
  `output_dir/in_process_events.jsonl` for forensics. `RunResult`
  exposes the post-move location.
- Probe payload now includes `env_path_values` for `_PYRUNTIME_EVENT_LOG`,
  `PWD`, and `CLAUDE_PROJECT_DIR`. The structural check verifies each
  reported path resolves under the per-run tmpdir; off-tmpdir paths
  trigger `env_path_outside_tmpdir` findings.
- Probe per-key check now applies denylist BEFORE allowlist exact-match
  so a future overlap cannot silently suppress a denylist hit. The two
  sets remain disjoint (asserted via test).
- Missing layer-2 telemetry post-exec is now fail-closed. If
  `tmpdir/.pyruntime/events.jsonl` does not exist after the subprocess
  exits, the runner writes a `{"event":"telemetry_missing","fatal":true}`
  sentinel to `output_dir/in_process_events.jsonl`, marks
  `cli_stderr.log`, and downgrades a clean `exit_code=0` to a new
  `EXIT_CODE_TELEMETRY_MISSING=-2` sentinel. Downstream extractors can
  branch on the sentinel rather than mistaking an empty log for "agent
  discovered nothing".
- Probe path-value verification is now fail-closed. `env_path_values`
  must be a dict; if `_PYRUNTIME_EVENT_LOG`, `PWD`, or `CLAUDE_PROJECT_DIR`
  appear in `env_keys`, their values must be present and non-empty in
  `env_path_values`. Missing values produce `missing_env_path_value`,
  `empty_env_path_value`, or `missing_env_path_values` findings.
- Dropped the deprecated `License :: OSI Approved :: MIT License` classifier
  from `pyproject.toml` (PEP 639 conflict with the modern `license = "MIT"`
  expression; previously blocked editable install).
- `harness/telemetry.py` layer-3 docstring no longer claims to capture
  Python warnings (layer 2 does that via the `warnings.warn` wrapper); layer
  3 is CLI-level stderr capture only.
- `harness/sitecustomize_template.py` is no longer a skeleton: implements
  the full diff_diff hook surface per the contract in its module docstring.
