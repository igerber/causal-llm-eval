# Roadmap

Planned features and deliverables for `causal-llm-eval`.

For tech debt and deferred items, see [TODO.md](TODO.md).

## Current state

The bootstrap infrastructure is in place: CI gate (`ready-for-ci` label enforcement), AI PR review workflow (Codex with two-tier secret scanning, untrusted-diff fencing, prompt loaded from base ref, split read-only/write-only jobs), `.claude/` skills for the PR workflow, plan-review hook with sentinel + staleness check, and the package skeleton for `harness/`, `graders/`, `analysis/` with `NotImplementedError` contracts behind documented dataclass schemas. CLAUDE.md captures the locked architectural decisions; `harness/COLD_START_VERIFICATION.md` captures the cold-start contract.

No agent has yet been spawned. No per-run record has been written. The runner implementation is the next substantive deliverable.

## Recently shipped

- **2026-05-10** | PR #1 (`b44f51f`) | Bootstrap Phase 0: CI/AI-review infrastructure, harness skeleton, locked architectural decisions in CLAUDE.md
- **2026-05-10** | this PR | Phase 0.5 cleanup: adapt three ported `.claude/commands/` files for eval-research domain; introduce TODO.md (tech debt) and ROADMAP.md (planned features)

## Shipping next

The items below are sequenced in rough build order, but several can run in parallel once the runner lands. Each becomes a focused PR.

- **Cold-start agent runner with `make smoke` inheritance probe** - the load-bearing piece per the plan; without it nothing else can be measured. Implements the locked `claude --bare ...` invocation, the `cwd=<tmpdir>` + `env=clean_env` hygiene, the inheritance probe that asserts the spawned agent reports no skills/memory/CLAUDE.md, and wires the probe into `make smoke` as a CI gate.
- **Three-layer telemetry capture** - `sitecustomize.py` shim hooks (guide-file reads, `get_llm_guide`, fit-time warnings, diagnostic methods, estimator instantiation) for both arms; merger that assembles per-run records with arm-aware sentinel validation; subprocess stderr capture cross-checked against the in-process layer.
- **Per-arm venv pool** - `build_arm_template` + `clone_for_run` for the diff-diff and statsmodels arms at pinned PyPI versions. Phase 1 uses fresh venvs; Phase 2 uses pre-built templates cloned per run.
- **Deterministic result extractor** - parses the per-run in-process event log into structured signals (estimator class, diagnostics invoked, warnings observed) for the two-stage extraction pipeline.
- **AI judge against the rubric** - separate Claude API call (not Claude Code) that applies the versioned rubric to a transcript + final code, returns structured JSON, supports the spot-check protocol for human-vs-judge agreement.
- **Synthetic staggered-adoption DGP generator** - uses `generate_staggered_data` from the diff-diff package; ships ground-truth metadata sidecar (`dgp_truth.json`) and calibration log.
- **Comparator-asymmetry pre-flight (`make preflight`)** - runs 3-5 sample agents per arm; checks the statsmodels arm spends < 60% of tokens on construction; documents the prompt-amendment escape hatch if asymmetry persists.
- **DGP calibration loop (`make calibration`)** - runs the statsmodels arm 3-5x against a candidate DGP; accept-criteria gate (>=2/5 pick TWFE AND >=1/5 pick something correct); retunes if saturated; documents locked DGP parameters.
- **Case-study v1 end-to-end (`make case-study-v1`)** - runs all 30 cells (15 per arm), produces structured records, cell summaries, primary + fallback claim outcomes. Includes the reproducibility-schema check for re-runs.
- **Case-study v1 writeup** - ~1500-word draft leading on framework + methodology, reporting both claim outcomes regardless of direction. Single sharable artifact (blog post or LinkedIn or repo README).

## Under consideration

Items the framework might add depending on Phase 1 findings:

- **"diff-diff without `llms.txt`" arm** - Phase 2 candidate to isolate library-design vs guide-content. Requires a controlled mechanism to bundle-strip or env-disable the guides without other library changes.
- **Cross-prompt-framing variation** - multiple prompt versions per dataset to measure prompt-sensitivity within an arm.
- **Additional comparator libraries** - econml, dowhy, fixest (via rpy2 or reticulate) as Phase 2 study subjects to extend the comparator ladder.
- **Doc-impact dependency map** - the analog of diff-diff's `docs/doc-deps.yaml` for this repo, wired into `/pre-merge-check` and `/push-pr-update`.

## Long-term research directions

Phase 2 (arxiv-stage) and Phase 3 (journal-stage) capabilities, framed as research directions rather than committed deliverables:

- **Pre-registered confirmatory study** - comparison ladder of 4-5 libraries, multiple datasets (synthetic + classic + newer with lower training-data contamination risk), N=20-30 per cell, multi-rater grading with reported inter-rater reliability (Cohen's kappa), full reproducibility kit. Pre-registered on OSF before runs.
- **Cross-model + cross-vendor sensitivity** - extends the framework to Claude Sonnet and Haiku (cheap), and possibly across vendors (GPT, Gemini) at higher cost. Tests whether library-design effects generalize beyond Claude Opus.
- **Human-expert control arm** - how do LLM agents compare to applied econometricians on the same task? Requires recruiting and instrumenting a small cohort; out of scope for the early phases but flagged as the version of this study that gets a much wider audience.
