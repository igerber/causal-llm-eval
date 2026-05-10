"""causal-llm-eval harness: cold-start agent runner, telemetry, and infrastructure.

The harness spawns truly cold-start Claude Code agents on causal-inference tasks
with one library installed per arm, captures three-layer telemetry (stream-JSON
event log, in-process Python instrumentation, subprocess stderr), and produces
structured per-run records suitable for grading by `graders/ai_judge.py`.

Modules:
    runner       - cold-start agent spawner with the locked --bare invocation
    telemetry    - three-layer capture
    sitecustomize_template - in-process instrumentation shim, copied per venv
    venv_pool    - per-arm venv management
    scheduler    - parallelism + budget tracking (used Phase 2+)
    extractor    - deterministic result extraction from in-process event log

See the latest plan in ~/.claude/plans/ for locked architectural decisions.
"""

__version__ = "0.0.1"
