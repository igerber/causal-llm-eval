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
- Probe structural env check upgraded from a small denylist to a hybrid
  denylist + allowlist. Denylist covers known auth/config leaks
  (`XDG_CONFIG_HOME`, `AWS_*`, `OPENAI_API_KEY`, `CODEX_HOME`,
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_PROJECT_ID`, `GITHUB_TOKEN`, `GH_TOKEN`,
  `CLAUDE_CONFIG_DIR`). Allowlist recognizes expected keys + `CLAUDE_*` /
  `CLAUDECODE_*` / `ANTHROPIC_*` / `PYTHON*` prefixes; everything else is
  flagged for review.
- Dropped the deprecated `License :: OSI Approved :: MIT License` classifier
  from `pyproject.toml` (PEP 639 conflict with the modern `license = "MIT"`
  expression; previously blocked editable install).
