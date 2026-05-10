"""Cold-start Claude Code agent runner.

Spawns standalone `claude --bare` processes in isolated tmpdirs with a per-run
venv carrying one library at a time (diff-diff XOR statsmodels for Phase 1).

The locked invocation:

    claude --bare \\
           --setting-sources "" \\
           --strict-mcp-config \\
           --disable-slash-commands \\
           --print \\
           --output-format stream-json \\
           --add-dir <tmpdir> \\
           <prompt>

The `--bare` flag is load-bearing: it suppresses CLAUDE.md auto-discovery,
auto-memory, plugin sync, attribution, and keychain reads. Without it the
spawned agent inherits operator state and the eval is invalid.

Verified by `make smoke` running the inheritance probe.

Skeleton only. Phase 0 deliverable is the contract; implementation lands in
subsequent PRs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    """Configuration for a single agent run."""

    arm: str  # "diff_diff" or "statsmodels"
    library_version: str  # PyPI version pinned for the arm
    dataset_path: Path
    prompt_path: Path
    prompt_version: str
    model: str = "claude-opus-4-7"
    timeout_seconds: int = 1800
    random_seed: int | None = None


@dataclass
class RunResult:
    """Outcome of a single agent run."""

    run_id: str
    arm: str
    tmpdir: Path
    transcript_jsonl_path: Path
    in_process_events_path: Path
    record_parquet_path: Path
    final_code_path: Path | None
    wall_clock_seconds: float
    exit_code: int


@dataclass
class RunMetadata:
    """Reproducibility metadata pinned per run.

    Every per-run record MUST carry these fields. Missing any one of them
    invalidates the reproducibility schema check in `make case-study-v1`
    (see plan section "Reproducibility schema"). Defining the contract here
    in Phase 0 prevents subsequent PRs from satisfying the surface tests
    while quietly omitting reproducibility metadata.
    """

    # Versioning: every layer that influences the per-run record's bytes
    harness_version: str  # git SHA of causal-llm-eval at run time
    library_version: str  # PyPI version (arm 1: diff-diff; arm 2: statsmodels)
    claude_code_version: str  # output of `claude --version`
    model_version: str  # the string passed to claude's --print (e.g. "claude-opus-4-7")
    # Inputs
    dataset_sha: str  # sha256 of the dataset parquet
    prompt_version: str  # registry id (e.g. "case_study/v1")
    rubric_version: str  # registry id (e.g. "case_study_v1")
    # Stochasticity
    random_seed: int  # captured per cell for any harness-side randomness
    # Identity
    run_id: str  # ULID or equivalent unique id; primary key for the record
    arm: str  # "diff_diff" or "statsmodels"


def run_one(config: RunConfig, output_dir: Path) -> RunResult:
    """Spawn a single cold-start agent and return its run record.

    Implementation pending. See plan section "Cold-start agent runner" for the
    locked invocation and three-layer telemetry capture.
    """
    del config, output_dir
    raise NotImplementedError("runner.run_one is not yet implemented")
