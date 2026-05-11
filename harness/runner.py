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

**Note:** PR #3 runs the spawned agent in whichever venv is active for the
harness process (no per-arm venv pool yet). The per-arm venv pool — fresh
venv per run with one library installed (diff-diff XOR statsmodels), with
PATH prepended to the venv bin — lands in PR #5 (`harness.venv_pool`).
Until then, `clean_env()` passes operator `PATH` verbatim. The cold-start
isolation contract for tmpdir, HOME, env allowlist, and CLI flags is
fully enforced in this PR; only the per-arm venv tier is deferred.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_ALLOWLISTED_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "PATH",
    "LANG",
    # Explicit LC_ enumeration (POSIX-defined keys). No `LC_*` wildcard so an
    # unrelated key like `LC_RPATH` cannot leak through.
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "ANTHROPIC_API_KEY",
)

_TIMEOUT_MARKER_FMT = "=== TIMEOUT after {timeout}s; process killed ==="


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
    cli_stderr_log_path: Path
    record_parquet_path: Path | None
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


def clean_env(tmpdir: Path, event_log_path: Path) -> dict[str, str]:
    """Construct the allowlisted environment for the spawned process.

    Two-tier construction:
        - Passthrough keys (PATH, LANG, LC_*, ANTHROPIC_API_KEY): copied
          verbatim from os.environ if present. Dropped if absent.
        - Runner-set keys (HOME, _PYRUNTIME_EVENT_LOG): ALWAYS set from
          runner-computed values; any inherited operator value is IGNORED.

    No prefix scan, no wildcard. Anything not in either tier (XDG_CONFIG_HOME,
    CLAUDE_CONFIG_DIR, AWS/MCP/GitHub env, CODEX_*, etc.) is dropped.

    Args:
        tmpdir: per-run temporary directory. Becomes the spawned process's $HOME
            (NOT the operator's homedir) so any `~` lookup lands in the sandbox.
        event_log_path: path to the per-run in-process event log. Becomes
            _PYRUNTIME_EVENT_LOG so the in-process shim writes there.

    Returns:
        Dict suitable for subprocess.Popen's env= argument.
    """
    env: dict[str, str] = {}
    for key in _ALLOWLISTED_PASSTHROUGH_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env["HOME"] = str(tmpdir)
    env["_PYRUNTIME_EVENT_LOG"] = str(event_log_path)
    return env


def _build_command(prompt: str, tmpdir: Path, model: str) -> list[str]:
    """Construct the exact locked argv for `claude --bare ...`.

    The seven cold-start isolation flags are: --bare, --setting-sources "",
    --strict-mcp-config, --disable-slash-commands, --print, --output-format
    stream-json, --add-dir <tmpdir>. COLD_START_VERIFICATION.md specifies
    the contract; the pre-merge-check skill verifies it via AST scan.

    `--model <id>` is additional: it pins the model so CLI defaults can't
    drift across runs. The value is `RunConfig.model`.
    """
    return [
        "claude",
        "--bare",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--print",
        "--output-format",
        "stream-json",
        "--model",
        model,
        "--add-dir",
        str(tmpdir),
        prompt,
    ]


def run_one(config: RunConfig, prompt: str, output_dir: Path) -> RunResult:
    """Spawn a single cold-start agent and return its run record.

    Args:
        config: RunConfig for the run (arm, library_version, timeout, etc.).
            The prompt is passed separately; config.prompt_path is metadata
            for reproducibility tracking, not the source of truth for the
            actual prompt text.
        prompt: the prompt string passed as the final argv element to
            `claude --bare`.
        output_dir: directory where transcript.jsonl, in_process_events.jsonl,
            and cli_stderr.log are written. Created if missing. If
            transcript.jsonl or in_process_events.jsonl already exist there,
            FileExistsError is raised to preserve prior runs for forensics.

    Returns:
        RunResult with run_id, paths, wall-clock seconds, and exit code.
        On TimeoutExpired the runner kills the subprocess, writes a marker
        line to cli_stderr.log, and returns RunResult(exit_code=-1) without
        raising.

    See harness/COLD_START_VERIFICATION.md for the cold-start invocation
    contract.
    """
    run_id = uuid.uuid4().hex[:16]
    # Resolve to an absolute path so the subprocess (running with cwd=tmpdir)
    # cannot misinterpret a relative `output_dir` against its own cwd. Without
    # this, the shim's `_PYRUNTIME_EVENT_LOG` lookup would resolve against
    # the per-run tmpdir, writing to a different file than the one we touched.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_jsonl_path = output_dir / "transcript.jsonl"
    event_log_path = output_dir / "in_process_events.jsonl"
    cli_stderr_log_path = output_dir / "cli_stderr.log"

    for preexisting in (transcript_jsonl_path, event_log_path):
        if preexisting.exists():
            raise FileExistsError(
                f"{preexisting} already exists. The runner refuses to overwrite "
                f"to preserve prior runs for forensics. Clear the output_dir "
                f"or use a fresh path."
            )

    tmpdir = Path(tempfile.mkdtemp(prefix="causal_run_"))

    event_log_path.touch()

    env = clean_env(tmpdir, event_log_path)
    cmd = _build_command(prompt, tmpdir, config.model)

    start = time.monotonic()
    timed_out = False
    with (
        open(transcript_jsonl_path, "w") as stdout_file,
        open(cli_stderr_log_path, "w") as stderr_file,
    ):
        proc = subprocess.Popen(
            cmd,
            cwd=str(tmpdir),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        try:
            exit_code = proc.wait(timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = -1
            timed_out = True
    wall_clock_seconds = time.monotonic() - start

    if timed_out:
        with open(cli_stderr_log_path, "a") as f:
            f.write(_TIMEOUT_MARKER_FMT.format(timeout=config.timeout_seconds) + "\n")

    return RunResult(
        run_id=run_id,
        arm=config.arm,
        tmpdir=tmpdir,
        transcript_jsonl_path=transcript_jsonl_path,
        in_process_events_path=event_log_path,
        cli_stderr_log_path=cli_stderr_log_path,
        record_parquet_path=None,
        final_code_path=None,
        wall_clock_seconds=wall_clock_seconds,
        exit_code=exit_code,
    )
