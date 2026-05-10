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


def run_one(config: RunConfig, output_dir: Path) -> RunResult:
    """Spawn a single cold-start agent and return its run record.

    Implementation pending. See plan section "Cold-start agent runner" for the
    locked invocation and three-layer telemetry capture.
    """
    del config, output_dir
    raise NotImplementedError("runner.run_one is not yet implemented")
