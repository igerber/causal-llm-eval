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
| Sitecustomize instrumentation hooks not yet wired (only env-var contract + empty event-log file as integration point). Layer 2 of the three-layer telemetry contract is unfilled until the hooks land. | `harness/sitecustomize_template.py` | #3 | Medium |
| Probe leakage assessment uses substring blacklist with hardcoded operator-token list; could miss novel leakage forms or false-fire on unanticipated negation contexts. Revisit with negation-aware matching or AI-judge if real-world probe runs reveal heuristic gaps. | `harness/probe.py` | #3 | Low |
| `RunConfig.dataset_path` is plumbed through but the runner does NOT copy the dataset into the per-run tmpdir. Dataset copy + symlink guard + reject-non-file-paths land in PR #6+ alongside the synthetic DGP generator. PR #3's runner is intended for the probe + smoke tests only; real eval runs require this step. | `harness/runner.py:run_one` | #3 | High |
| `RunMetadata` schema is locked but `run_one()` does NOT emit a populated `metadata.json` sidecar. Population (harness git SHA, claude binary version, dataset SHA, library version, prompt/rubric registry ids, seed, run_id, arm, model) lands in PR #6+ alongside the case-study runner. The schema is pinned HERE so subsequent PRs cannot quietly omit fields. | `harness/runner.py:run_one` | #3 | High |

### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| _(none yet)_ | | | |

### Testing / Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Pyright "unused" warning on `_write_event` (intentional helper for not-yet-written hooks). Either silence with a `noqa` comment, restructure, or accept until the hook callers land. | `harness/sitecustomize_template.py:_write_event` | #1 | Low |
| `telemetry.py` layer-3 docstring says "captures Python warnings from agent's Python subprocesses"; in practice the CLI captures those into stream-JSON and layer 3 only catches CLI-level errors. Refine wording when the telemetry merger lands. | `harness/telemetry.py:14-16` | #3 | Low |

## Adapting Diff-Diff-Ported Files

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| `test-check-plan-review.sh` referenced in hook script comment but not ported from diff-diff. Either port the test file or update the comment to remove the reference. | `.claude/hooks/check-plan-review.sh:23` | #1 | Low |
| Module docstring in `openai_review.py` mentions "REGISTRY.md section extraction" as a feature dropped in the minimal port. Edit for clarity once the script is otherwise touched. | `.claude/scripts/openai_review.py:26-28` | #1 | Low |
