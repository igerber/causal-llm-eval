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
| Fail-early on unwritable event-log path: shim raises `OSError` on first write attempt (mid-agent-execution); add upfront `os.access`/touch-test in the runner before agent spawn so misconfiguration aborts the run before the agent starts. | `harness/runner.py` (new check) + `harness/sitecustomize_template.py` (current behavior) | #1 | Low |

### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| _(none yet)_ | | | |

### Testing / Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Pyright "unused" warning on `_write_event` (intentional helper for not-yet-written hooks). Either silence with a `noqa` comment, restructure, or accept until the hook callers land. | `harness/sitecustomize_template.py:_write_event` | #1 | Low |
| Smoke test does not yet execute the inheritance probe (deferred until runner is implemented). `make smoke` is a placeholder that intentionally fails. | `tests/test_harness_smoke.py` + `Makefile:smoke` | #1 | Medium |

## Adapting Diff-Diff-Ported Files

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| `test-check-plan-review.sh` referenced in hook script comment but not ported from diff-diff. Either port the test file or update the comment to remove the reference. | `.claude/hooks/check-plan-review.sh:23` | #1 | Low |
| Module docstring in `openai_review.py` mentions "REGISTRY.md section extraction" as a feature dropped in the minimal port. Edit for clarity once the script is otherwise touched. | `.claude/scripts/openai_review.py:26-28` | #1 | Low |
