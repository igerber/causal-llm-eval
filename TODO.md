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
| Hook reactivity: monkey-patches detectable via `inspect.getsource`, `sys.modules['_pyruntime_shim']` (post-PR-#6; previously `sys.modules['sitecustomize']`), `warnings.warn is custom`, `builtins.open` / `io.open` is custom. The `_ESTIMATOR_CLASS_NAMES` and `_DIAGNOSTIC_FUNCTION_NAMES` constants also leak the tracked-methodology surface. Accepted per PR #4 low-reactivity-via-documentation posture. Revisit if any early case-study agent probes the shim. | `harness/sitecustomize_template.py` | #4 → future | Low |
| `tests/test_attestation_default.py::_detect_shadowing_sitecustomize` skip is now over-cautious post-PR-#6: the `.pth`-based load survives Homebrew sitecustomize shadow, so the layer-2 tests would actually fire on systems the skip currently auto-skips. Revisit by removing the skip and asserting layer-2 fires unconditionally. | `tests/test_attestation_default.py` | #6 → future | Low |
| `make smoke` is occasionally flaky on `no_structural_block` (~10% failure rate). The agent sometimes emits the Part 1 prose answer + the Part 2 heading and stops without invoking Bash for the structural python command. Confirmed via retry; not a cold-start integrity defect. Mitigations to consider: prompt-side ("END YOUR ANSWER ONLY AFTER YOU HAVE INVOKED BASH"), runner-side (auto-retry once on `no_structural_block`), or assessment-side (explicit `--max-turns N` to give the agent more room). | `harness/probe.py::PROBE_PROMPT` | #6 → future | Low |
| Low-level guide-read coverage: shim hooks `builtins.open` / `io.open` (catches direct calls, pathlib, `importlib.resources`) but not `os.open` + `os.read`, `pkgutil.get_data`, `mmap`, or C-extension reads. The merger emits `opened_llms_*=False` for guide reads through these vectors, which is misleading. Uncommon in agent flows; closing requires fd-tracking shim architecture. | `harness/sitecustomize_template.py` | #4 → PR #5 | Medium |
| Phase 1 DGP parameters in `harness/dgp.py::_DGP_CALL_KWARGS` are uncalibrated starter values; calibration loop (separate future PR) tunes them before locking the case study. Bumping requires bumping `CASE_STUDY_V1_DGP_VERSION` and regenerating committed `datasets/case_study_v1/` artifacts. | `harness/dgp.py` | #6 | Low |
| `analysis/reproducibility_check.py` does not yet exist; will need to consume `metadata.json` (now emitted by `run_one()`) and verify re-runs match documented tolerances when implemented (PR #8+). | `analysis/` | #6 | Low |

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
