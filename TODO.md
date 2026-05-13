# TODO

Tech debt and deferred items - things we owe, not things we plan to build.

For planned features and deliverables, see [ROADMAP.md](ROADMAP.md).

## Known Limitations

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Codex CI re-review on push uses `pull_request: opened` only; subsequent push-on-PR re-reviews require explicit `/ai-review` comment. By design post-merge to limit cost. | `.github/workflows/ai_pr_review.yml` | #1 | informational |

## Tech Debt from Code Reviews

### Methodology / Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Probe leakage assessment uses substring blacklist with hardcoded operator-token list; could miss novel leakage forms or false-fire on unanticipated negation contexts. Revisit with negation-aware matching or AI-judge if real-world probe runs reveal heuristic gaps. | `harness/probe.py` | #3 | Low |
| Statsmodels arm: shim has no hooks yet; `arm="statsmodels"` TelemetryRecord defaults to sentinels + all-False bools. Wires alongside the case-study statsmodels prompt (ROADMAP: "Statsmodels arm instrumentation"). | `harness/sitecustomize_template.py` | #4 → PR #7 prerequisite | Medium |
| Layer-1 transcript parsing partially implemented in PR #4: Read-tool guide-file accesses ARE parsed into `opened_llms_*` flags; per-invocation Python attribution and bypass detection are also live. Still deferred: Bash-level estimator/diagnostic usage parsing (e.g. `cat llms.txt` or `python -c "from diff_diff import ..."` outside our existing hooks). Lands when the judge prompt names the evidence it needs. | `harness/telemetry.py` | #4 → ROADMAP "Layer-1 transcript parsing" | Low |
| Hook reactivity: monkey-patches detectable via `inspect.getsource`, `sys.modules['sitecustomize']`, `warnings.warn is custom`, `builtins.open` / `io.open` is custom. The `_ESTIMATOR_CLASS_NAMES` and `_DIAGNOSTIC_FUNCTION_NAMES` constants also leak the tracked-methodology surface. Accepted per PR #4 low-reactivity-via-documentation posture. Revisit if any early case-study agent probes the shim. | `harness/sitecustomize_template.py` | #4 → future | Low |
| End-to-end live test of the shim firing inside a `claude --bare` subprocess deferred. Requires installing `sitecustomize.py` somewhere on `sys.path`; the only safe location is a per-arm venv. Lands alongside per-arm-venv pool. | `tests/test_telemetry_live.py` (not created in PR #4) | #4 → PR #5 | Medium |
| `RunConfig.dataset_path` is plumbed through but the runner does NOT copy the dataset into the per-run tmpdir. Dataset copy + symlink guard + reject-non-file-paths land in PR #6+ alongside the synthetic DGP generator. PR #3's runner is intended for the probe + smoke tests only; real eval runs require this step. | `harness/runner.py:run_one` | #3 | High |
| `RunMetadata` schema is locked but `run_one()` does NOT emit a populated `metadata.json` sidecar. Population (harness git SHA, claude binary version, dataset SHA, library version, prompt/rubric registry ids, seed, run_id, arm, model) lands in PR #6+ alongside the case-study runner. The schema is pinned HERE so subsequent PRs cannot quietly omit fields. | `harness/runner.py:run_one` | #3 | High |

### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| _(none yet)_ | | | |

### Testing / Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| _(none yet)_ | | | |

## Adapting Diff-Diff-Ported Files

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| `test-check-plan-review.sh` referenced in hook script comment but not ported from diff-diff. Either port the test file or update the comment to remove the reference. | `.claude/hooks/check-plan-review.sh:23` | #1 | Low |
| Module docstring in `openai_review.py` mentions "REGISTRY.md section extraction" as a feature dropped in the minimal port. Edit for clarity once the script is otherwise touched. | `.claude/scripts/openai_review.py:26-28` | #1 | Low |
