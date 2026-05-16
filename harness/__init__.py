"""causal-llm-eval harness: cold-start agent runner, telemetry, and infrastructure.

The harness spawns truly cold-start Claude Code agents on causal-inference tasks
with one library installed per arm, captures three-layer telemetry (stream-JSON
event log, in-process Python instrumentation, subprocess stderr), and produces
structured per-run records suitable for grading by `graders/ai_judge.py`.

Modules:
    runner       - cold-start agent spawner with the locked --bare invocation
    telemetry    - three-layer capture
    sitecustomize_template - in-process instrumentation shim. Installed per
                             venv as ``_pyruntime_shim.py`` + ``_pyruntime_shim.pth``
                             (PR #6 fix — the .pth-based load survives Homebrew
                             Python's stdlib-level sitecustomize.py shadow).
    venv_pool    - per-arm venv management
    dgp          - synthetic data-generating process for the case study
    scheduler    - parallelism + budget tracking (used Phase 2+)
    extractor    - deterministic result extraction from in-process event log
    probe        - cold-start inheritance probe (live ``make smoke``)

See the latest plan in ~/.claude/plans/ for locked architectural decisions.
"""

__version__ = "0.0.1"
