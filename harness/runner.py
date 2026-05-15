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
           --verbose \\
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

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness.shell_parser import RunValidityError
from harness.venv_pool import build_arm_template

_ALLOWLISTED_PASSTHROUGH_KEYS: tuple[str, ...] = (
    # PR #5 R12 P0: ``PATH`` is intentionally NOT passed through. The
    # operator's PATH may contain operator-local directories (e.g.,
    # ``/opt/homebrew/bin``, ``~/.cargo/bin``, custom shims) that would
    # leak operator state into a "cold-start" agent. The runner instead
    # constructs a sanitized agent PATH explicitly: the per-arm venv's
    # bin/ followed by a small system-bin allowlist (see _agent_path).
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

# PR #5 R12 P0 / R15 P1 (EV-2): sanitized agent PATH. The runner
# prepends the per-run venv bin (so the agent's ``python`` resolves
# to the wrapper) and follows with a minimal system-bin allowlist
# (so utilities like ``ls``, ``cat``, ``find`` resolve).
#
# Operator-local directories are intentionally excluded:
#
#   * ``/opt/homebrew/bin``, ``~/.cargo/bin``, ``~/.local/bin``:
#     classic operator-installed tool dirs.
#   * ``/usr/local/bin``: dropped at R15. On macOS Intel Homebrew
#     (and many developer machines) ``/usr/local/bin`` is the
#     Homebrew prefix and contains operator-installed Python
#     shims, CLIs, and project tools. Even on systems where
#     ``/usr/local/bin`` is conventionally admin-managed, it can
#     contain locally-built tools the agent should not be able to
#     resolve. ``ls`` / ``cat`` / ``find`` / ``mkdir`` etc. all
#     live in ``/usr/bin`` and ``/bin``, which are sufficient for
#     the cold-start contract.
#
# An agent that needs an external tool the venv doesn't provide
# must fail rather than reach into operator state.
_AGENT_SYSTEM_PATH_DIRS: tuple[str, ...] = (
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)

_TIMEOUT_MARKER_FMT = "=== TIMEOUT after {timeout}s; process killed ==="
_TELEMETRY_MISSING_MARKER = (
    "=== TELEMETRY MISSING: agent_event_log_path did not exist post-exec ==="
)
_DESCENDANTS_LIVE_MARKER = (
    "=== DESCENDANTS LIVE: agent process group had surviving children "
    "after main process exited; killed via SIGKILL. Telemetry may be "
    "incomplete (children could have written events between proc.wait() "
    "and killpg). Run marked invalid. ==="
)
# Exit-code sentinels for fatal harness conditions. Distinct from any real
# CLI exit code so downstream extractors can branch on them.
EXIT_CODE_TIMEOUT = -1
EXIT_CODE_TELEMETRY_MISSING = -2
EXIT_CODE_DESCENDANTS_LIVE = -3


@dataclass
class RunConfig:
    """Configuration for a single agent run.

    ``dataset_path`` MUST be a regular file (not a symlink, directory,
    device, FIFO, or socket). The runner validates this strictly at the
    top of :func:`run_one` and copies the file into ``tmpdir/data.parquet``
    before spawning the agent. Symlinks are rejected via ``lstat()`` to
    preserve the cold-start integrity claim (an operator-home symlink
    would otherwise leak operator state into a "cold-start" agent).

    ``rubric_version`` matches ``graders.ai_judge.JudgeResult.rubric_version``
    semantically: both carry the registry id of the rubric the run is
    graded against.
    """

    arm: str  # "diff_diff" or "statsmodels"
    library_version: str  # PyPI version pinned for the arm
    dataset_path: Path
    prompt_path: Path
    prompt_version: str
    rubric_version: str  # registry id (e.g. "case_study_v1"); see graders/ai_judge.py
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
    # PR #6: the dataset copied into the per-run tmpdir at start; populated
    # by ``run_one`` from ``config.dataset_path``. Always Path for runs that
    # go through ``run_one`` (which now requires a regular-file dataset).
    dataset_in_tmpdir_path: Path | None = None
    # PR #6: ``output_dir/metadata.json``. Populated only on CLEAN exit
    # (exit_code == 0 AND not descendants_live AND no telemetry_missing
    # sentinel). Absence is the signal: "this run did not complete cleanly
    # enough to be a reproducibility-anchored record."
    metadata_json_path: Path | None = None


@dataclass
class RunMetadata:
    """Reproducibility metadata pinned per run.

    Serialized to ``output_dir/metadata.json`` by :func:`run_one` ONLY on
    clean exit (``exit_code == 0`` AND no descendants_live AND no
    telemetry_missing sentinel). Absence of ``metadata.json`` is itself
    the signal that the run did not complete cleanly enough to be a
    reproducibility-anchored record.

    Every per-run record carries these fields; ``__post_init__`` enforces
    format invariants (sha shapes, non-empty version strings, recognized
    arm). The schema is checked in :mod:`tests.test_harness_smoke` so
    additions or removals are caught at PR-time.
    """

    # Versioning: every layer that influences the per-run record's bytes
    harness_version: str  # git SHA of causal-llm-eval at run time, optionally suffixed "-dirty"
    library_version: str  # PyPI version (arm 1: diff-diff; arm 2: statsmodels)
    claude_code_version: str  # output of `claude --version`
    model_version: str  # the string passed to claude's --print (e.g. "claude-opus-4-7")
    # Inputs
    dataset_sha: str  # sha256 of the dataset parquet (64 hex chars)
    prompt_version: str  # registry id (e.g. "case_study/v1")
    rubric_version: str  # registry id (e.g. "case_study_v1"); see graders/ai_judge.py::JudgeResult.rubric_version
    # Stochasticity
    random_seed: (
        int | None
    )  # None = no harness-side seed configured for this run; serialized as JSON null
    # Identity
    run_id: str  # ULID or equivalent unique id; primary key for the record
    arm: str  # "diff_diff" or "statsmodels"

    def __post_init__(self) -> None:
        """Lightweight format validation so a malformed record can't be silently constructed."""
        import re

        if not re.fullmatch(r"[0-9a-f]{40}(-dirty)?", self.harness_version):
            raise ValueError(
                f"harness_version must be 40-hex SHA + optional -dirty: {self.harness_version!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_sha):
            raise ValueError(f"dataset_sha must be 64-hex sha256: {self.dataset_sha!r}")
        if not self.claude_code_version.strip():
            raise ValueError("claude_code_version must be non-empty")
        if self.arm not in ("diff_diff", "statsmodels"):
            raise ValueError(f"arm must be 'diff_diff' or 'statsmodels': {self.arm!r}")


def clean_env(
    tmpdir: Path,
    event_log_path: Path,
    *,
    venv_bin_dir: Path | None = None,
) -> dict[str, str]:
    """Construct the allowlisted environment for the spawned process.

    Two-tier construction:
        - Passthrough keys (LANG, LC_*, ANTHROPIC_API_KEY): copied
          verbatim from os.environ if present. Dropped if absent.
        - Runner-set keys (HOME, _PYRUNTIME_EVENT_LOG, PATH): ALWAYS
          set from runner-computed values; any inherited operator
          value is IGNORED.

    PR #5 R12 P0 / R15 P1 (EV-2): ``PATH`` is RUNNER-SET, not
    passthrough. The agent receives
    ``${venv_bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin`` (when
    ``venv_bin_dir`` is supplied) or just the system bin allowlist
    (when it's None - probe / test paths). Operator-local directories
    (``/opt/homebrew/bin``, ``/usr/local/bin``, ``~/.cargo/bin``,
    custom shims) are intentionally excluded; see
    ``_AGENT_SYSTEM_PATH_DIRS`` for the rationale on each.

    No prefix scan, no wildcard. Anything not in either tier (XDG_CONFIG_HOME,
    CLAUDE_CONFIG_DIR, AWS/MCP/GitHub env, CODEX_*, etc.) is dropped.

    Args:
        tmpdir: per-run temporary directory. Becomes the spawned process's $HOME
            (NOT the operator's homedir) so any `~` lookup lands in the sandbox.
        event_log_path: path to the per-run in-process event log. Becomes
            _PYRUNTIME_EVENT_LOG so the in-process shim writes there.
        venv_bin_dir: per-arm venv's bin/ directory. Prepended to the
            sanitized agent PATH so the agent's ``python`` resolves to
            the layer-1.5 wrapper.

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
    if venv_bin_dir is not None:
        env["PATH"] = os.pathsep.join([str(venv_bin_dir), *_AGENT_SYSTEM_PATH_DIRS])
    else:
        env["PATH"] = os.pathsep.join(_AGENT_SYSTEM_PATH_DIRS)
    return env


def _resolve_claude_executable() -> str:
    """Resolve ``claude`` to an absolute path using the operator's PATH.

    PR #5 R12 P0: the spawned agent's PATH is sanitized (operator-local
    directories are excluded), so the runner must invoke ``claude`` by
    absolute path - the agent's PATH no longer reaches into operator
    state to find it. Resolution happens once at runner construction
    time using the operator's PATH (which DOES contain the directory
    where claude is installed). Raises ``FileNotFoundError`` if claude
    is not on the operator's PATH.
    """
    claude = shutil.which("claude")
    if claude is None:
        raise FileNotFoundError("`claude` not found on operator PATH; install Claude Code first")
    return claude


def _resolve_claude_invocation_prefix(claude_path: str) -> list[str]:
    """Return the argv prefix to launch Claude under a sanitized PATH.

    PR #5 R19 CQ-1 (P2): when ``claude`` is an npm-installed script
    starting with ``#!/usr/bin/env node`` (the common Claude Code
    install shape), the spawned process inherits the runner's
    sanitized agent PATH (which excludes the operator's
    ``/opt/homebrew/bin`` / ``/usr/local/bin`` etc.). Kernel shebang
    resolution then runs ``/usr/bin/env``, which searches the
    SUBPROCESS PATH for ``node``, finds nothing, and the launch
    fails with ``env: 'node': No such file or directory``.

    Fix: read the script's shebang line at runner-build time (using
    the operator PATH for interpreter resolution), then return the
    explicit interpreter prefix so the kernel does not have to
    re-resolve it under the sanitized agent PATH:

      * Native binary (no shebang or unresolvable shebang): returns
        ``[claude_path]`` - kernel exec the binary directly.
      * Shebang ``#!/usr/local/bin/node``: returns
        ``["/usr/local/bin/node", claude_path]`` - absolute path,
        no PATH search needed.
      * Shebang ``#!/usr/bin/env node``: resolves ``node`` via
        the operator PATH (``shutil.which("node")``) and returns
        ``[node_abs_path, claude_path]``. Falls back to
        ``[claude_path]`` if resolution fails (the launch will
        then surface the original env-resolution error visibly).
      * Shebang ``#!/usr/bin/env -S node --some-flag`` or
        ``#!/usr/bin/node --some-flag``: returns
        ``[interp_path, "--some-flag", claude_path]`` so script
        author's interpreter args are preserved.

    The interpreter is recorded as an ABSOLUTE path so its directory
    does not need to leak into the agent's PATH allowlist.
    """
    try:
        with open(claude_path, "rb") as fh:
            first_line = fh.readline(1024).decode("utf-8", errors="replace")
    except OSError:
        return [claude_path]
    if not first_line.startswith("#!"):
        return [claude_path]
    # Strip the "#!" + trailing whitespace/newline; tokenize on whitespace.
    shebang = first_line[2:].strip()
    if not shebang:
        return [claude_path]
    parts = shebang.split()
    interp_spec = parts[0]
    interp_args = parts[1:]
    # Handle "/usr/bin/env [-S] <interp> [args...]" by resolving <interp>
    # via the operator PATH; otherwise treat parts[0] as the absolute
    # interpreter path.
    if os.path.basename(interp_spec) == "env":
        # Skip optional ``-S`` (env -S supports multi-arg shebangs on Linux).
        if interp_args and interp_args[0] == "-S":
            interp_args = interp_args[1:]
        if not interp_args:
            return [claude_path]
        interp_name = interp_args[0]
        interp_args = interp_args[1:]
        interp_abs = shutil.which(interp_name)
        if interp_abs is None:
            # Cannot resolve interpreter via operator PATH; fall back to
            # the original direct-launch behavior (which will surface the
            # env-resolution error visibly rather than silently failing).
            return [claude_path]
    elif os.path.isabs(interp_spec):
        interp_abs = interp_spec
    else:
        # Relative interpreter path is unusual; resolve via operator PATH.
        interp_abs = shutil.which(interp_spec)
        if interp_abs is None:
            return [claude_path]
    return [interp_abs, *interp_args, claude_path]


def _harness_version() -> str:
    """Return the harness git SHA, plus '-dirty' suffix if working tree dirty.

    Walks upward from this file looking for ``.git`` so editable installs
    (where ``__file__`` may live in site-packages) still resolve correctly.
    Untracked files COUNT as dirty: a stray ``harness/dgp.py.bak`` could
    shadow imports and the ``-dirty`` flag is the operator's signal.
    """
    candidate = Path(__file__).resolve().parent
    repo_root = None
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise RunValidityError(
            f".git not found walking up from {Path(__file__).resolve()}; "
            f"cannot pin harness_version (running from a tarball or "
            f"non-git checkout?)"
        )
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if sha_result.returncode != 0:
        raise RunValidityError(f"git rev-parse HEAD failed: {sha_result.stderr!r}")
    # 30s timeout for ``git status``: on a repo with a large untracked tree
    # (e.g. accumulated runs/ artifacts), porcelain status walks the working
    # tree and can be slow on cold caches.
    dirty_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if dirty_result.returncode != 0:
        raise RunValidityError(f"git status --porcelain failed: {dirty_result.stderr!r}")
    suffix = "-dirty" if dirty_result.stdout.strip() else ""
    return sha_result.stdout.strip() + suffix


def _claude_version() -> str:
    """Return ``claude --version`` output, last non-empty line only.

    Scrubbed env so the operator's ``_PYRUNTIME_EVENT_LOG`` cannot leak
    into the claude CLI subprocess and write stray events to a stale path.
    PATH is the only operator var passed (claude needs it to resolve node
    in the npm-installed-script case). Resolution uses operator PATH (same
    as :func:`_resolve_claude_executable`) since the agent's sanitized PATH
    is irrelevant for harness-side metadata capture.

    Returns the last non-empty line of stdout — strips deprecation banners
    and node-engine warnings that may precede the version string in future
    CLI releases.
    """
    result = subprocess.run(
        [_resolve_claude_executable(), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0:
        raise RunValidityError(
            f"`claude --version` exited {result.returncode}; stderr={result.stderr!r}"
        )
    output = result.stdout.strip()
    if not output:
        raise RunValidityError("`claude --version` produced empty stdout")
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return lines[-1].strip()


def _validate_dataset_path(src: Path) -> None:
    """Strict-reject validation: regular file only.

    Catches symlinks via ``lstat()`` (NOT ``stat()``, which follows
    symlinks). Runs BEFORE any tmpdir is created so failure cleanup is
    automatic.
    """
    if "\x00" in str(src):
        raise RunValidityError(f"dataset_path contains NUL byte: {src!r}")
    if not src.exists():
        raise RunValidityError(f"dataset_path does not exist: {src!r}")
    if src.is_symlink():
        raise RunValidityError(f"dataset_path is a symlink (regular file required): {src!r}")
    mode = src.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise RunValidityError(
            f"dataset_path is not a regular file (got mode={stat.filemode(mode)}): {src!r}"
        )


def _copy_dataset_into_tmpdir(src: Path, tmpdir: Path) -> tuple[Path, str]:
    """Copy dataset into ``tmpdir/data.parquet``; return (dst_path, sha256_hex).

    ``sha256`` is computed from the COPIED bytes (the artifact-of-record).
    If a kernel bug ever corrupts bytes during copy, the in-tmpdir file is
    what the agent saw; that's what we record.
    """
    dst = tmpdir / "data.parquet"
    try:
        shutil.copy2(src, dst)
    except OSError as e:
        raise RunValidityError(f"failed to copy dataset_path into tmpdir: {e}") from e
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    return dst, sha


def _build_command(
    claude_invocation: list[str], prompt: str, tmpdir: Path, model: str
) -> list[str]:
    """Construct the exact locked argv for `claude --bare ...`.

    The seven cold-start isolation flags are: --bare, --setting-sources "",
    --strict-mcp-config, --disable-slash-commands, --print, --output-format
    stream-json, --add-dir <tmpdir>. COLD_START_VERIFICATION.md specifies
    the contract; the pre-merge-check skill verifies it via AST scan.

    Plus ``--verbose`` (CLI 2.1.143+ requirement when --print is combined
    with --output-format=stream-json; without it the CLI produces no output).

    `--model <id>` is additional: it pins the model so CLI defaults can't
    drift across runs. The value is `RunConfig.model`.

    PR #5 R12 P0 / R19 CQ-1: ``claude_invocation`` is the resolved
    invocation prefix from ``_resolve_claude_invocation_prefix``. For
    native binaries it's ``[abs_claude_path]``; for shebang-script
    Claude installs (npm `#!/usr/bin/env node`) it's
    ``[abs_interpreter_path, claude_path]`` (or with extra interpreter
    args if the shebang specified them). The agent's PATH is sanitized
    and may not include the interpreter's or claude's directory, so
    both are absolute by construction.
    """
    return [
        *claude_invocation,
        "--bare",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--print",
        "--output-format",
        "stream-json",
        # Claude CLI 2.1.143+ requires --verbose when --print is combined
        # with --output-format=stream-json; without it, the CLI emits no
        # transcript output (silent on --bare; "Error: ... requires
        # --verbose" on the non-bare path). Without this flag the runner
        # would still exit 0 but produce a 0-byte transcript, which the
        # merger has no way to distinguish from "agent emitted nothing".
        "--verbose",
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
    # PR #6: pre-tmpdir validation — fail fast before any compute is spent.
    # ``_harness_version`` and ``_claude_version`` are captured here (not
    # later) so a transient git/claude failure cannot invalidate an
    # otherwise-clean agent run that's already produced telemetry.
    # ``_validate_dataset_path`` runs before ``tempfile.mkdtemp`` so a
    # dataset failure does NOT leak a tmpdir.
    harness_version = _harness_version()
    claude_code_version = _claude_version()
    _validate_dataset_path(config.dataset_path)

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
    metadata_json_path = output_dir / "metadata.json"

    for preexisting in (
        transcript_jsonl_path,
        final_event_log_path,
        cli_stderr_log_path,
        metadata_json_path,
    ):
        if preexisting.exists():
            raise FileExistsError(
                f"{preexisting} already exists. The runner refuses to overwrite "
                f"to preserve prior runs for forensics. Clear the output_dir "
                f"or use a fresh path."
            )

    tmpdir = Path(tempfile.mkdtemp(prefix="causal_run_"))

    # PR #6: copy the dataset into the per-run tmpdir at top-level
    # ``data.parquet``. Symlink guard + regular-file check already passed
    # via ``_validate_dataset_path`` above. ``dataset_sha`` is captured
    # from the COPIED bytes (the artifact-of-record) and recorded in
    # metadata.json on clean exit (PR #6 step 6).
    dataset_in_tmpdir_path, dataset_sha = _copy_dataset_into_tmpdir(config.dataset_path, tmpdir)

    # R4 P1 fix: the in-process shim writes to a path INSIDE the agent tmpdir
    # during execution. After the subprocess exits we move the file to
    # output_dir/in_process_events.jsonl for forensics. The agent's view of
    # _PYRUNTIME_EVENT_LOG no longer leaks the harness repo path.
    agent_event_log_path = tmpdir / ".pyruntime" / "events.jsonl"
    agent_event_log_path.parent.mkdir(parents=True, exist_ok=True)
    agent_event_log_path.touch()

    # PR #5: build a per-arm venv with the layer-1.5 wrapper + sitecustomize
    # shim installed. This is the per-run cost of Phase 1 (~10-30s); Phase 2
    # replaces this with a clone-from-template (deferred; ROADMAP).
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

    # PR #5 R14 P2 (EV-1): parse the events log AFTER the sentinel
    # subprocess returns and before spawning Claude. Exit 0 alone proves
    # the python interpreter ran, not that the wrapper + shim actually
    # emitted their layer-1.5 + layer-2 events. Without this check, a
    # broken shim install (e.g. site machinery skipped sitecustomize)
    # would still let the sentinel exit 0 and the agent would launch
    # against a half-wired attestation chain. The merger's
    # ``_validate_three_layer_consistency`` would catch it post-hoc, but
    # the cold-start doc claims a pre-agent guarantee. Enforce it here.
    runner_pid = os.getpid()
    sentinel_lines = agent_event_log_path.read_text().splitlines()
    sentinel_events = [json.loads(line) for line in sentinel_lines if line.strip()]
    sentinel_exec = next(
        (
            e
            for e in sentinel_events
            if e.get("event") == "exec_python" and e.get("ppid") == runner_pid
        ),
        None,
    )
    if sentinel_exec is None:
        raise RunValidityError(
            f"build-time sentinel did not emit an exec_python event with "
            f"ppid={runner_pid} (the runner pid); the layer-1.5 wrapper "
            f"is not wired. Events seen: {sentinel_events!r}"
        )
    sentinel_pid = sentinel_exec.get("pid")
    if not any(
        e.get("event") == "session_start" and e.get("pid") == sentinel_pid for e in sentinel_events
    ):
        raise RunValidityError(
            f"build-time sentinel emitted exec_python (pid={sentinel_pid}) "
            f"but no matching session_start; the layer-2 sitecustomize "
            f"shim is not loaded. Events seen: {sentinel_events!r}"
        )
    if not any(
        e.get("event") == "session_end" and e.get("pid") == sentinel_pid for e in sentinel_events
    ):
        raise RunValidityError(
            f"build-time sentinel emitted exec_python + session_start "
            f"(pid={sentinel_pid}) but no matching session_end; the "
            f"layer-2 atexit hook is not registered. "
            f"Events seen: {sentinel_events!r}"
        )

    # PR #5 R12 P0: clean_env now sets a sanitized PATH containing only
    # the per-arm venv bin + system bin allowlist. Operator-local
    # directories (~/.cargo/bin, /opt/homebrew/bin, etc.) do NOT leak
    # into the agent's PATH. ``claude`` is invoked by absolute path
    # below since the agent's PATH may not contain the directory where
    # it lives.
    env = clean_env(tmpdir, agent_event_log_path, venv_bin_dir=venv_dir / "bin")
    cmd = _build_command(
        _resolve_claude_invocation_prefix(_resolve_claude_executable()),
        prompt,
        tmpdir,
        config.model,
    )

    start = time.monotonic()
    timed_out = False
    descendants_live = False
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

    # PR #5 R16 P1 (EV-1): quiesce the spawned process group on the
    # NORMAL exit path too. A non-timeout proc.wait() returns when
    # claude itself exits, but background descendants (e.g. an agent
    # ``Bash`` invocation that did ``nohup python script.py &``) can
    # outlive claude and continue writing layer-1.5 / layer-2 events
    # AFTER ``run_one`` moves the event log into ``output_dir``. That
    # is a real telemetry race: the merger could see a clean record
    # at finalization time that gets mutated post-hoc by a surviving
    # child, or the moved log could miss late writes entirely.
    #
    # Try ``killpg(SIGKILL)``: if it succeeds, the PG still had
    # members (descendants existed), the run is INVALID. Mark
    # ``descendants_live`` and downgrade ``exit_code`` to
    # ``EXIT_CODE_DESCENDANTS_LIVE`` (only if not already a
    # sentinel - timeout takes precedence). If it raises
    # ``ProcessLookupError``, the kernel has already reaped the PG;
    # clean exit, no action.
    if not timed_out:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            descendants_live = True
        except ProcessLookupError:
            # PG already empty - claude exited cleanly with no surviving
            # descendants. No action.
            pass

    if timed_out:
        with open(cli_stderr_log_path, "a") as f:
            f.write(_TIMEOUT_MARKER_FMT.format(timeout=config.timeout_seconds) + "\n")
    elif descendants_live:
        with open(cli_stderr_log_path, "a") as f:
            f.write(_DESCENDANTS_LIVE_MARKER + "\n")
        exit_code = EXIT_CODE_DESCENDANTS_LIVE
        # PR #5 R17 P1 (EV-1): also write a fail-closed sentinel
        # event into the agent event log itself. Without this, a
        # downstream caller that invokes ``merge_layers`` directly on
        # the artifact set (without inspecting RunResult.exit_code or
        # cli_stderr.log) could still produce a clean
        # ``TelemetryRecord`` for a run the runner explicitly marked
        # invalid. The merger's existing ``run_invalid`` schema
        # rejects this event class; this is the same pattern as the
        # ``telemetry_missing`` sentinel below.
        if agent_event_log_path.exists():
            with open(agent_event_log_path, "a") as f:
                json.dump(
                    {
                        "event": "run_invalid",
                        "fatal": True,
                        "reason": "descendants_live",
                        "note": (
                            "agent process group had surviving children "
                            "after main process exited; runner killed via "
                            "SIGKILL but late writes to the event log "
                            "between proc.wait() and killpg may have "
                            "produced an incomplete or post-finalization "
                            "telemetry record"
                        ),
                    },
                    f,
                )
                f.write("\n")

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

    # PR #6: emit metadata.json ONLY on clean exit. Absence is the signal
    # that the run did not complete cleanly enough to be a
    # reproducibility-anchored record. The three failure-path sentinels
    # (timeout / descendants_live / telemetry_missing) all suppress the
    # metadata write; downstream extractors branch on file presence.
    metadata_emitted_path: Path | None = None
    clean_exit = (
        exit_code == 0 and not descendants_live and exit_code != EXIT_CODE_TELEMETRY_MISSING
    )
    if clean_exit:
        metadata = RunMetadata(
            harness_version=harness_version,
            library_version=config.library_version,
            claude_code_version=claude_code_version,
            model_version=config.model,
            dataset_sha=dataset_sha,
            prompt_version=config.prompt_version,
            rubric_version=config.rubric_version,
            random_seed=config.random_seed,
            run_id=run_id,
            arm=config.arm,
        )
        with open(metadata_json_path, "w") as f:
            json.dump(
                dataclasses.asdict(metadata),
                f,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            f.write("\n")
        metadata_emitted_path = metadata_json_path

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
        dataset_in_tmpdir_path=dataset_in_tmpdir_path,
        metadata_json_path=metadata_emitted_path,
    )
