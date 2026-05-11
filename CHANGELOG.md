# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cold-start agent runner (`harness.runner.run_one`) implementing the locked
  `claude --bare ...` invocation with `cwd=tmpdir`, `env=clean_env`, and
  pre-spawn writability check on the per-run event log path.
- Cold-start inheritance probe (`harness.probe.run_probe`) with a separated
  output directory under `runs/probe/`. The probe never feeds eval data.
- `make smoke` target: live cold-start probe with `ANTHROPIC_API_KEY` fail-fast
  precheck. Costs ~$0.05 per invocation.
- `.github/workflows/tests.yml`: GitHub Actions test workflow running `pytest`
  (default excludes `slow` and `live`) on labeled PRs and push to main.
- 19 unit tests for `harness.runner` (13) and `harness.probe` (6), plus 2 live
  tests gated by `@pytest.mark.live`.

### Changed
- Renamed in-process telemetry env var from `CAUSAL_LLM_EVAL_EVENT_LOG` to
  `_PYRUNTIME_EVENT_LOG`. The underscore prefix + Python-runtime framing is a
  deliberate low-reactivity choice so an agent enumerating `os.environ` is less
  likely to flag the name as eval-related.
- Dropped the Phase 0 status caveat from `CLAUDE.md` and
  `harness/COLD_START_VERIFICATION.md`: the CI test workflow now exists.
