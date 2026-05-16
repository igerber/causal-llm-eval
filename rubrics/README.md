# Rubrics

Versioned grading rubrics consumed by `graders/ai_judge.py`. Once a
rubric is recorded against runs (via `RunConfig.rubric_version` →
`metadata.json`), it is IMMUTABLE; new rubric = new version file. See
`CLAUDE.md` "Pre-defined rubrics" for the contract.

## Versions

| File | Status | Notes |
|---|---|---|
| `case_study_v1.yaml` | **Reserved stub** | Header-only file reserved for the `case_study_v1` registry id. PR #6 reserved this slot; never edit in place. |
| `case_study_v2.yaml` | **Active** (PR #7) | Phase 1 case-study grading rubric (6 criteria, pre-defined before runs). |

## Schema (v2)

Each YAML file has top-level keys `version`, `task`, `notes`, and a
`criteria` list. Each criterion has `id`, `prompt` (the question the
judge asks), and `output_type` (one of `enum`, `string`, `bool`,
`float_or_null`, `list_of_strings`); `enum` types also carry an
`enum_values` list. `graders.ai_judge.JudgeResult.fields: dict` is
keyed by `id`; downstream extractors read against the same schema.

Schema enforcement is currently lightweight (presence + type checks in
`tests/test_case_study_rubric.py`); strict YAML validation lands when
the AI judge implementation does (PR #9).
