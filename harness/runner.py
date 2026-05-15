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

PR #5 wires the per-arm venv pool: ``run_one()`` builds a fresh venv at
``tmpdir/venv`` (Phase 1), installs the layer-1.5 ``python`` wrapper +
sitecustomize shim into it, runs a build-time sentinel to attest wrapper
+ shim wiring, then prepends the venv's ``bin/`` to ``PATH`` so the
agent's ``python`` resolves to the wrapper. Phase 2 (template-and-clone
per run) lands in PR #6+ when eval volume justifies the optimization.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness.shell_parser import RunValidityError
from harness.venv_pool import build_arm_template

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
_TELEMETRY_MISSING_MARKER = (
    "=== TELEMETRY MISSING: agent_event_log_path did not exist post-exec ==="
)
# Exit-code sentinels for fatal harness conditions. Distinct from any real
# CLI exit code so downstream extractors can branch on them.
EXIT_CODE_TIMEOUT = -1
EXIT_CODE_TELEMETRY_MISSING = -2


@dataclass
class RunConfig:
    """Configuration for a single agent run.

    **Note:** `dataset_path` is plumbed through for metadata tracking but the
    runner does NOT yet copy the dataset into the per-run tmpdir. Dataset
    copy + symlink guard + reject-non-file-paths logic land in PR #6+
    alongside the synthetic DGP generator (`harness/dgp.py`). Until then,
    probe/test runs pass `Path("/dev/null")` as a placeholder. Real eval
    runs that reach `run_one()` before PR #6 lands would not exercise the
    isolation guarantee documented in `COLD_START_VERIFICATION.md`; PR #3's
    runner is intended for the probe + smoke tests only.
    """

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
    """Outcome of a single agent run.

    PR #5 added ``venv_path`` and ``runner_pid`` so the merger can validate
    layer-1.5 events against the run's actual venv root and partition
    sentinel-vs-agent events by ppid.
    """

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
    venv_path: Path | None = None
    runner_pid: int | None = None


@dataclass
class RunMetadata:
    """Reproducibility metadata pinned per run.

    **Note:** PR #3 locks this schema but `run_one()` does NOT yet emit a
    populated `RunMetadata` sidecar. Population + `metadata.json` emission
    land in PR #6+ alongside the case-study runner, which knows the dataset
    SHA, prompt registry id, rubric registry id, and other fields PR #3
    cannot wire (no DGP, no prompt registry, no rubric registry yet).
    The schema is locked HERE so subsequent PRs cannot quietly omit fields.

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
    final_event_log_path = output_dir / "in_process_events.jsonl"
    cli_stderr_log_path = output_dir / "cli_stderr.log"

    for preexisting in (transcript_jsonl_path, final_event_log_path, cli_stderr_log_path):
        if preexisting.exists():
            raise FileExistsError(
                f"{preexisting} already exists. The runner refuses to overwrite "
                f"to preserve prior runs for forensics. Clear the output_dir "
                f"or use a fresh path."
            )

    tmpdir = Path(tempfile.mkdtemp(prefix="causal_run_"))

    # R4 P1 fix: the in-process shim writes to a path INSIDE the agent tmpdir
    # during execution. After the subprocess exits we move the file to
    # output_dir/in_process_events.jsonl for forensics. The agent's view of
    # _PYRUNTIME_EVENT_LOG no longer leaks the harness repo path.
    agent_event_log_path = tmpdir / ".pyruntime" / "events.jsonl"
    agent_event_log_path.parent.mkdir(parents=True, exist_ok=True)
    agent_event_log_path.touch()

    # PR #5: build a per-arm venv with the layer-1.5 wrapper + sitecustomize
    # shim installed. This is the per-run cost of Phase 1 (~10-30s); Phase 2
    # (PR #6+) replaces this with a clone-from-template.
    venv_dir = tmpdir / "venv"
    build_arm_template(config.arm, config.library_version, venv_dir)

    # Build-time python sentinel: prove the wrapper + shim are wired before
    # the agent ever runs. Closes the implicit-trigger zero-python gap (a
    # shell-only agent produces zero exec_python events; without this
    # sentinel the merger cannot distinguish "agent did no python" from
    # "agent neutered the wrapper"). The sentinel runs WITH
    # _PYRUNTIME_EVENT_LOG set so the wrapper emits exactly one exec_python
    # event and sitecustomize emits one session_start + session_end into
    # the runner's event log, BEFORE the agent subprocess starts. The
    # merger keys on RunResult.runner_pid to distinguish the sentinel
    # (ppid == runner_pid) from agent-spawned events (ppid != runner_pid).
    # PATH for the sentinel must include a system bin dir so the wrapper's
    # internal calls to ``awk`` / ``date`` / ``basename`` / ``printf`` /
    # ``sed`` resolve. (The wrapper itself uses ``#!/bin/sh`` absolute
    # shebang and pins its internal PATH to ``/usr/bin:/bin``, but the
    # sentinel subprocess's PATH still needs system bins for the wrapper
    # to find them.) The venv bin is prepended so ``python`` resolves
    # to the wrapper before reaching system Python.
    sentinel_path = f"{venv_dir / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin"
    sentinel_result = subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "pass"],
        env={
            "PATH": sentinel_path,
            "_PYRUNTIME_EVENT_LOG": str(agent_event_log_path),
        },
        capture_output=True,
        timeout=30,
    )
    if sentinel_result.returncode != 0:
        raise RunValidityError(
            f"build-time sentinel failed (exit={sentinel_result.returncode}): "
            f"wrapper or shim not wired correctly; stderr={sentinel_result.stderr!r}"
        )

    env = clean_env(tmpdir, agent_event_log_path)
    # Prepend the venv's bin/ to PATH so the agent's `python` resolves to
    # the wrapper, not the operator's interpreter.
    env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    cmd = _build_command(prompt, tmpdir, config.model)

    start = time.monotonic()
    timed_out = False
    with (
        open(transcript_jsonl_path, "w") as stdout_file,
        open(cli_stderr_log_path, "w") as stderr_file,
    ):
        # start_new_session=True puts the spawned process (and any children
        # it spawns) in a new session/process group. On timeout we kill the
        # whole group via os.killpg(proc.pid, SIGKILL), preventing leftover
        # Bash/Python children from continuing to run or mutate per-run
        # files after RunResult is returned.
        proc = subprocess.Popen(
            cmd,
            cwd=str(tmpdir),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                # Process already exited between timeout and killpg; harmless.
                pass
            proc.wait()
            exit_code = EXIT_CODE_TIMEOUT
            timed_out = True
    wall_clock_seconds = time.monotonic() - start

    if timed_out:
        with open(cli_stderr_log_path, "a") as f:
            f.write(_TIMEOUT_MARKER_FMT.format(timeout=config.timeout_seconds) + "\n")

    # Promote the in-tmpdir event log into output_dir for forensics. A
    # missing file post-exec is fail-closed (R5 P0): the runner touched it
    # pre-spawn, so absence implies either the agent removed it or the
    # tmpdir was disturbed. Treat as fatal telemetry loss: write a sentinel
    # event + stderr marker + downgrade exit_code if the CLI itself was
    # clean. Downstream extractors can branch on the sentinel rather than
    # mistaking an empty file for "agent discovered nothing".
    if agent_event_log_path.exists():
        shutil.move(str(agent_event_log_path), str(final_event_log_path))
    else:
        with open(final_event_log_path, "w") as f:
            json.dump(
                {
                    "event": "telemetry_missing",
                    "fatal": True,
                    "note": "agent_event_log_path did not exist post-exec",
                },
                f,
            )
            f.write("\n")
        with open(cli_stderr_log_path, "a") as f:
            f.write(_TELEMETRY_MISSING_MARKER + "\n")
        if exit_code == 0:
            exit_code = EXIT_CODE_TELEMETRY_MISSING

    return RunResult(
        run_id=run_id,
        arm=config.arm,
        tmpdir=tmpdir,
        transcript_jsonl_path=transcript_jsonl_path,
        in_process_events_path=final_event_log_path,
        cli_stderr_log_path=cli_stderr_log_path,
        record_parquet_path=None,
        final_code_path=None,
        wall_clock_seconds=wall_clock_seconds,
        exit_code=exit_code,
        venv_path=venv_dir,
        runner_pid=os.getpid(),
    )
