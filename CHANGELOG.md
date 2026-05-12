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

### Changed
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
- Dropped the deprecated `License :: OSI Approved :: MIT License` classifier
  from `pyproject.toml` (PEP 639 conflict with the modern `license = "MIT"`
  expression; previously blocked editable install).
