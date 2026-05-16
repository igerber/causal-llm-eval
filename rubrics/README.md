# Rubrics

Versioned grading rubrics consumed by `graders/ai_judge.py`. Once a
rubric is recorded against runs (via `RunConfig.rubric_version` →
`metadata.json`), it is IMMUTABLE; new rubric = new version file. See
`CLAUDE.md` "Pre-defined rubrics" for the contract.
