# Case-study prompts

Versioned task prompts the cold-start agent reads via the `prompt` argument
to `harness.runner.run_one`. The `RunConfig.prompt_version` registry id
(e.g. `"case_study/v2"`) is recorded in `metadata.json` and binds the
per-run record to the exact prompt the agent saw.

## Versions

| File | Status | Notes |
|---|---|---|
| `v1.txt` | **Reserved stub** | Header-only file reserved for the `case_study/v1` registry id. PR #6 reserved this slot to lock the id; never edit in place. |
| `v2.txt` | **Active** (PR #7) | The Phase 1 case-study task prompt. Task-only by design — names no library and no estimator, so both arms see identical text and the central scientific claim is not invalidated by prompt-side methodology hinting. Asserted by `tests/test_case_study_prompt.py`. |

## Versioning policy

Per `CLAUDE.md` item 4: **once a prompt is recorded against any run, it is
IMMUTABLE.** New prompt content = new version file (`v3.txt`, `v4.txt`, …),
NOT in-place edit. The current active version is `v2.txt`; the
`case_study/v2` registry id will be locked once the first case-study run
records it in its `metadata.json`.

If you need to evolve the prompt, copy `v2.txt` → `v3.txt`, edit there,
and update the case-study runner's `prompt_version` to `"case_study/v3"`.
