# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  markers; wrappers fail-open on transient `OSError` from the event log so
  the agent's run is not aborted by telemetry hiccups.
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
