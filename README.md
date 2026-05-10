# causal-llm-eval

A black-box evaluation framework for measuring how LLM agents make methodology choices in causal inference tasks, and whether library design (specifically LLM-targeted guidance surfaces like `llms.txt`, fit-time warnings, native diagnostics, and pedagogical docstrings) measurably affects those choices.

## Status

Early development. Phase 1 case study in progress: comparing diff-diff vs statsmodels on a staggered-adoption synthetic DGP, with N=15 cold-start agents per arm.

## Repo layout (planned)

```
harness/        # cold-start agent runner, telemetry capture, venv management
graders/        # AI judge applying the rubric to transcripts
prompts/        # versioned task prompts
rubrics/        # versioned grading rubrics
datasets/       # synthetic DGPs and metadata sidecars
runs/           # per-run records (mostly gitignored)
analysis/       # cell summaries, variability reports, reproducibility checks
writeups/       # case-study writeup drafts
```

## Why a separate repo?

Eval lives independently of the libraries it evaluates. Independence supports the framework's generalizability, isolates dependency footprints, and keeps reproducibility kits self-contained.

## License

MIT
