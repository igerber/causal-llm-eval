"""Three-layer telemetry capture for agent runs.

Layer 1 - Stream-JSON event log from Claude Code:
    Parsed from `claude --print --output-format stream-json` output. Contains
    every user/assistant turn, every tool call (Bash, Read, Edit, Write, Grep)
    with arguments and results, and file reads with paths.

Layer 2 - In-process Python instrumentation (the discoverability ground truth):
    A `sitecustomize.py` installed in the per-run venv hooks the target library
    and writes per-event JSON records. Catches access that stream-JSON misses
    (e.g., `python -c "from diff_diff import get_llm_guide"` reads the file via
    Python, not Claude's Read tool). See `harness/sitecustomize_template.py`.

Layer 3 - Subprocess stderr capture:
    Captures CLI-level errors emitted by `claude --bare` and any other stderr
    the agent's process writes. Python-level warnings from the agent's
    diff_diff calls are captured by layer 2 via the `warnings.warn` wrapper
    (stack-inspecting, stacklevel-safe), not here.

The merger (`merge_layers`) parses layers 1+2 and emits a per-run
`TelemetryRecord` with arm-aware sentinel semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TelemetryRecord:
    """Merged per-run record assembled from three layers.

    Discoverability fields use a three-state encoding:
        - ``True``  - the agent accessed this surface
        - ``False`` - the agent did NOT access this surface (and could have)
        - ``None``  - this surface is "not applicable" to the arm; absence is
          structural, not behavioral. Used for arm-2 (statsmodels) on the
          `opened_llms_*` and `called_get_llm_guide` fields, since statsmodels
          ships no LLM-targeted guides.

    Comparator-fairness analysis distinguishes "could-have-but-didn't" (False)
    from "couldn't-have-because-no-such-feature" (None). Collapsing these
    biases the comparison in favor of the arm with no guidance surface.

    The ``arm`` field is required so ``merge_layers()`` and downstream graders
    can validate that the sentinel pattern matches the arm's contract (e.g.,
    ``opened_llms_txt is None`` iff ``arm == "statsmodels"``).
    """

    arm: str  # "diff_diff" or "statsmodels"; drives sentinel semantics below
    stream_json_path: Path
    in_process_events_path: Path
    stderr_path: Path
    # Discoverability flags - tri-state per docstring above
    opened_llms_txt: bool | None = None
    opened_llms_practitioner: bool | None = None
    opened_llms_autonomous: bool | None = None
    opened_llms_full: bool | None = None
    called_get_llm_guide: bool | None = None
    get_llm_guide_variants: tuple[str, ...] = ()
    # The remaining flags ARE applicable to both arms (warnings, diagnostics,
    # estimator instantiation), so they remain plain bool / tuple.
    saw_fit_time_warning: bool = False
    diagnostic_methods_invoked: tuple[str, ...] = ()
    estimator_classes_instantiated: tuple[str, ...] = ()

    # Set of fields whose tri-state encoding depends on arm.
    _ARM_SENTINEL_FIELDS = (
        "opened_llms_txt",
        "opened_llms_practitioner",
        "opened_llms_autonomous",
        "opened_llms_full",
        "called_get_llm_guide",
    )
    _VALID_ARMS = ("diff_diff", "statsmodels")

    def __post_init__(self) -> None:
        """Enforce arm-specific contracts on construction.

        - Reject unknown arms.
        - For arm == "diff_diff", every sentinel-bearing field MUST be a bool
          (not None). The diff-diff arm has guide surfaces; the merger is
          obligated to fill them in.
        - For arm == "statsmodels", every sentinel-bearing field MUST be None
          (not bool). The statsmodels arm has no guide surfaces; encoding
          False would conflate "not applicable" with "not discovered".

        Catching this at construction prevents downstream graders/analysis from
        silently consuming corrupted records.
        """
        if self.arm not in self._VALID_ARMS:
            raise ValueError(f"TelemetryRecord.arm={self.arm!r} is not one of {self._VALID_ARMS}")
        for field_name in self._ARM_SENTINEL_FIELDS:
            value = getattr(self, field_name)
            if self.arm == "diff_diff":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"TelemetryRecord(arm='diff_diff').{field_name} must be "
                        f"bool (True/False), got {value!r}. The diff-diff arm "
                        f"has guide surfaces; the merger must record discovery "
                        f"outcome, not leave the field as None."
                    )
            elif self.arm == "statsmodels":
                if value is not None:
                    raise ValueError(
                        f"TelemetryRecord(arm='statsmodels').{field_name} must "
                        f"be None (not applicable), got {value!r}. statsmodels "
                        f"has no guide surfaces; encoding True/False would "
                        f"conflate 'not applicable' with 'not discovered'."
                    )


class TelemetryMergeError(RuntimeError):
    """Raised when ``merge_layers`` cannot produce a valid TelemetryRecord.

    Distinct cases:
    - In-process event log file missing entirely.
    - Malformed line in the event log (non-JSON).
    - Runner-written ``telemetry_missing`` sentinel present (post-exec the
      agent's event log disappeared; the runner already wrote the sentinel
      and downgraded exit_code to -2).
    - Cross-layer inconsistency: agent's transcript shows python invocations
      but the in-process layer has no ``session_start`` event (shim never
      loaded; cold-start eval is invalid).

    All four cases are fail-closed; the merger does not silently downgrade
    to a "no agent activity" record.
    """


# diff_diff bundled guide files mapped to the variant string `get_llm_guide`
# accepts. Verified against `diff_diff/_guides_api.py:7-12`. Key ordering
# follows the source for diff-friendliness.
_VARIANT_TO_FILENAME: dict[str, str] = {
    "concise": "llms.txt",
    "full": "llms-full.txt",
    "practitioner": "llms-practitioner.txt",
    "autonomous": "llms-autonomous.txt",
}


# Word-boundary regex for python-invocation detection in the layer-1
# cross-check. Matches `python`, `python3`, `python3.11` at a token boundary.
# Boundary set includes `/` so absolute interpreter paths
# (`/usr/bin/python3 script.py`, `/opt/venv/bin/python -c "..."`) are caught,
# and `;`/`&`/`|`/`(` so compound shell commands
# (`pip install foo && python script.py`) are caught. The trailing `\s|$`
# requirement guards against false positives like `/opt/python/` (the `/`
# after `python` is neither whitespace nor end-of-string).
_PYTHON_INVOCATION_RE = re.compile(r"(?:^|[\s;&|()/])python(?:3(?:\.\d+)?)?(?:[\s<>]|$)")


# Shell patterns that mutate or override the resolved interpreter location.
# Any of these in a command containing a python invocation means the
# resolved interpreter may NOT be the per-arm-venv python — fail-closed.
#
# Forms detected:
# - Inline / pre-invocation / exported PATH mutation
# - venv activation: `source X`, `. X`, `conda activate`, `pyenv shell`
# - env-driven resolution: `env python`, `env -u VAR python`
# - Local binary: `./python`
#
# Detection is conservative: ANY of these patterns in a command that ALSO
# contains a python invocation triggers fail-closed. This over-catches
# benign cases (e.g. `echo PATH=/foo && python`), but agents rarely emit
# such patterns for non-bypass reasons; failing-closed is correct.
_PATH_ASSIGNMENT_RE = re.compile(r"(?:^|[\s;&|])(?:export\s+)?PATH=\S+")
# PYTHONPATH / PYTHONHOME / PYTHONSTARTUP / PYTHONUSERBASE all alter
# Python's import resolution or run startup code in the agent's process;
# if either is set, the interpreter resolves modules differently from a
# clean per-arm-venv start, and the shim's discoverability claims are
# at risk.
_PYTHON_ENV_VAR_RE = re.compile(
    r"(?:^|[\s;&|])(?:export\s+)?PYTHON(?:PATH|HOME|STARTUP|USERBASE)=\S+"
)
_SHELL_ACTIVATION_RE = re.compile(
    r"(?:^|[\s;&|])(?:source\s+\S+|\.\s+\S+|conda\s+activate|pyenv\s+shell)"
)
# Wrapper commands that launch Python through a tool-managed environment.
# The launched Python may not have sitecustomize installed (uv/poetry use
# their own venv; conda run / pyenv exec use the named env).
_PYTHON_WRAPPER_RE = re.compile(r"(?:^|[\s;&|])(?:uv\s+run|poetry\s+run|conda\s+run|pyenv\s+exec)")
_ENV_BEFORE_PYTHON_RE = re.compile(r"(?:^|[\s;&|])env\s+(?:-\S+\s+)*python")
_DOT_PYTHON_RE = re.compile(r"(?:^|[\s;&|])\./python")


# Python interpreter flag that bypasses `sitecustomize.py`: `-S` disables
# `site.py` (which is what imports sitecustomize). Note `-I` (isolated mode)
# implies `-E`, `-P`, and `-s` (lowercase) — NOT `-S`. Isolated mode does
# not skip sitecustomize when the shim is installed in the venv's
# site-packages, so it is not in the bypass list.
#
# Compact short-flag forms are matched too: `python -Sc 'code'` is
# equivalent to `python -S -c 'code'` (the `-S` is the bypass; `c` is the
# `-c` short flag for inline code). Lowercase `-s` (skip user site) does
# NOT bypass sitecustomize and is correctly ignored (regex requires
# uppercase `S` specifically).
_PYTHON_BYPASS_FLAG_RE = re.compile(r"(?:^|\s)-[A-Za-z]*S[A-Za-z]*(?:[\s=]|$)")


def _parse_jsonl_strict(path: Path, label: str) -> list[dict]:
    """Parse a JSONL file strictly; raise TelemetryMergeError on malformed lines.

    Shared parser used by both layer-2 event-log reads and layer-1 transcript
    reads. Empty lines are skipped; non-JSON content raises; non-object
    (scalar/list) entries also raise — the runner's contract is that every
    line is a JSON OBJECT. The ``label`` string identifies the layer/file
    in the error message.

    An empty (0-byte) file is permitted and returns ``[]``. The caller is
    responsible for whether emptiness is acceptable in that layer's
    semantics.
    """
    events: list[dict] = []
    with open(path) as f:
        for line_num, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TelemetryMergeError(
                    f"malformed JSON in {label} {path} at line {line_num}: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise TelemetryMergeError(
                    f"non-object JSON in {label} {path} at line {line_num}: "
                    f"expected a JSON object, got {type(parsed).__name__}"
                )
            events.append(parsed)
    return events


# Substring marker the shim's `_write_event` prints to stderr when a write
# fails mid-run. Used by the merger to fail-closed when telemetry events were
# dropped after the hook deliberately continued executing the wrapped call.
_SHIM_WRITE_FAILURE_MARKER = "[pyruntime] cannot write event"


def _scan_stderr_for_shim_failures(stderr_path: Path) -> bool:
    """Return True if stderr contains the shim's event-write failure marker.

    The shim's hook wrappers catch transient OSError on event writes so the
    agent's diff_diff call still completes (avoids aborting on telemetry
    hiccups). When that happens, `_write_event` first prints
    `[pyruntime] cannot write event to <path>: <err>` to stderr. The merger
    looks for that marker post-hoc: presence means at least one event was
    dropped, so the per-run record may be silently incomplete.
    """
    try:
        content = stderr_path.read_text(errors="replace")
    except OSError:
        # Existence was already validated; a read error here is itself a
        # capture problem worth surfacing.
        return True
    return _SHIM_WRITE_FAILURE_MARKER in content


def _validate_layer_artifacts(stream_json_path: Path, stderr_path: Path) -> list[dict]:
    """Fail-closed preflight: layer-1 and layer-3 capture files must exist,
    and the transcript must be a non-empty sequence of JSON objects.
    Returns the parsed transcript entries so the caller doesn't reparse.

    The three-layer telemetry contract requires the stream-JSON transcript
    (layer 1) and the CLI stderr capture (layer 3) to be present for every
    run.

    - Missing files mean the runner did not finish capturing or the output
      directory was tampered with.
    - An EMPTY stream-JSON transcript means stdout capture failed or was
      truncated; without at least one entry, the merger cannot distinguish
      a Read-tool guide access from no agent activity, so emitting
      definitive ``opened_llms_*=False`` would be silent layer-1 loss.
    - A non-object JSON line (scalar / list) violates the runner's contract
      that every line is a single JSON object.

    Empty stderr capture remains valid (a trivial agent response can produce
    an empty stderr log; layer-3 only carries CLI-level errors).
    """
    if not stream_json_path.exists():
        raise TelemetryMergeError(
            f"stream-JSON transcript missing at {stream_json_path}; "
            f"layer-1 capture incomplete, per-run record cannot be validated"
        )
    if not stderr_path.exists():
        raise TelemetryMergeError(
            f"CLI stderr capture missing at {stderr_path}; "
            f"layer-3 capture incomplete, per-run record cannot be validated"
        )
    transcript_entries = _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")
    if not transcript_entries:
        raise TelemetryMergeError(
            f"stream-JSON transcript at {stream_json_path} is empty; "
            f"mergeable runs require at least one transcript entry "
            f"(empty transcript likely indicates stdout-capture failure "
            f"and would silently zero out layer-1 evidence)"
        )
    # Truncation check: a complete Claude stream-json transcript ends with a
    # successful `result` entry. A transcript whose final entry is anything
    # else (assistant message, tool_use, tool_result, or a `result` carrying
    # `is_error=true`) indicates capture was cut short before the run
    # finished, and per-run evidence (Bash invocations, Read tool_results,
    # later guide accesses) may be silently missing.
    last = transcript_entries[-1]
    if not (isinstance(last, dict) and last.get("type") == "result" and not last.get("is_error")):
        raise TelemetryMergeError(
            f"stream-JSON transcript at {stream_json_path} does not end "
            f"with a successful `type=result` entry; capture is truncated "
            f"or the run did not complete cleanly, and per-run telemetry "
            f"cannot be treated as complete"
        )
    return transcript_entries


def _read_events(path: Path) -> list[dict]:
    """Parse the in-process event log into a list of dicts.

    Raises ``TelemetryMergeError`` if the file is missing or contains a
    non-JSON line. An empty (0-byte) file is permitted and returns ``[]``;
    the validity check is the caller's responsibility.
    """
    if not path.exists():
        raise TelemetryMergeError(
            f"in-process event log not found at {path}; "
            f"the runner should have written a telemetry_missing sentinel"
        )
    return _parse_jsonl_strict(path, "event log")


def _iter_tool_result_blocks(entry):
    """Yield tool_result blocks from a stream-JSON entry.

    tool_result entries live on user messages (the role="user" turn that
    carries tool outputs back to the assistant). The block contains
    `tool_use_id` matching the originating Read/Bash request, plus
    `is_error` and `content` indicating success/failure.
    """
    if not isinstance(entry, dict):
        return
    msg = entry.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    yield block
    if entry.get("type") == "tool_result":
        yield entry


def _scan_read_tool_guide_accesses_in_entries(
    entries: list[dict],
) -> dict[str, bool]:
    """Return ``{filename: True}`` for each bundled guide file SUCCESSFULLY
    accessed via Claude's Read tool in the transcript.

    Catches the case where an agent reads `llms.txt` (or any other bundled
    guide) via the Read tool without invoking Python; the in-process shim
    sees nothing in that path because no `open()` or `get_llm_guide` call
    runs in the agent's subprocess. Without this layer-1 check the merger
    would emit a definitive `opened_llms_txt=False` for what was actually
    a guide discovery.

    Path-matching uses adjacent path parts (``diff_diff`` / ``guides`` /
    ``<filename>``). Substring match would accept paths like
    ``/tmp/notdiff_diff/guides/llms.txt`` or
    ``/some/path/diff_diff/guides_extra/llms.txt``; the part-adjacency
    check requires the read to be in the real bundled location.

    A Read tool_use is only counted as evidence if the matching
    tool_result is non-error (``is_error`` falsy or absent). A failed,
    denied, or hallucinated Read does not flip `opened_llms_*`. If a
    guide-file Read request has no matching tool_result, the merger
    fails closed: an incomplete transcript that would silently emit
    `opened_llms_*=False` for a Read that may have succeeded is treated
    the same as the terminal-result truncation check - the per-run
    record cannot be trusted.

    Only Read-tool evidence is recognized here. Bash-level guide reads
    (e.g. ``cat llms.txt``) are still part of the broader layer-1 parsing
    deferred to a future PR.
    """
    known_filenames = set(_VARIANT_TO_FILENAME.values())

    # First pass: collect candidate Read requests that target a guide path.
    pending: dict[str, str] = {}  # tool_use_id -> filename
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Read":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            file_path = tool_input.get("file_path", "")
            if not isinstance(file_path, str) or not file_path:
                continue
            p = Path(file_path)
            if not (
                p.name in known_filenames
                and p.parent.name == "guides"
                and p.parent.parent.name == "diff_diff"
            ):
                continue
            tool_use_id = block.get("id")
            if isinstance(tool_use_id, str):
                pending[tool_use_id] = p.name

    if not pending:
        return {}

    # Second pass: find matching tool_results.
    opened: dict[str, bool] = {}
    matched_ids: set[str] = set()
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            filename = pending.get(tool_use_id)
            if filename is None:
                continue
            matched_ids.add(tool_use_id)
            if not result.get("is_error", False):
                opened[filename] = True

    unmatched_ids = set(pending) - matched_ids
    if unmatched_ids:
        unmatched_filenames = sorted({pending[i] for i in unmatched_ids})
        raise TelemetryMergeError(
            f"Read tool_use for bundled guide file(s) {unmatched_filenames!r} "
            f"has no matching tool_result in the transcript (tool_use_ids: "
            f"{sorted(unmatched_ids)!r}). Transcript is incomplete; the "
            f"per-run record cannot be trusted to reflect guide-discovery state."
        )
    return opened


def _iter_tool_use_blocks(entry):
    """Yield tool_use blocks from a stream-JSON entry, handling both
    nested (`message.content[*]`) and shallow (top-level type=tool_use) shapes.
    """
    if not isinstance(entry, dict):
        return
    msg = entry.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block
    if entry.get("type") == "tool_use":
        yield entry


def _extract_python_invocations_from_command(command: str) -> list[list[str]]:
    """Return a list of argv-shaped token lists for each python invocation.

    Each entry is the full argv slice: `[interpreter, *args]`. Args run up
    to the next shell separator (``;``, ``&``, ``|``, end-of-string), with
    quoting handled by `shlex.split`. The interpreter token preserves any
    absolute path prefix (e.g. `/usr/bin/python3`) so the merger can later
    match by basename or full path against `session_start.argv`.
    """
    invocations: list[list[str]] = []
    for m in _PYTHON_INVOCATION_RE.finditer(command):
        # Walk left to recover any absolute-path prefix on the interpreter
        # token (e.g. `/usr/bin/python3`).
        match_start = m.start()
        i = match_start
        while i > 0 and command[i - 1] not in (" ", "\t", ";", "&", "|", "(", ")"):
            i -= 1
        # The trailing-boundary char (space, <, etc.) is part of m.group(0);
        # the interpreter token ends one char before m.end().
        interp_end = m.end() - 1
        interpreter = command[i:interp_end]
        if interpreter and interpreter[0] in "; & | ( )":
            interpreter = interpreter[1:]
        # Slice the args region: from end of match to next shell separator.
        rest_start = m.end() - 1  # include the boundary char in rest
        rest = command[rest_start:]
        sep_match = re.search(r"[;&|]", rest)
        args_region = rest[: sep_match.start()] if sep_match else rest
        # Tokenize args with shlex; fall back to whitespace split if shlex
        # raises (unbalanced quotes, etc.).
        try:
            import shlex

            args_tokens = shlex.split(args_region, comments=False, posix=True)
        except ValueError:
            args_tokens = args_region.split()
        invocations.append([interpreter, *args_tokens])
    return invocations


def _session_argv_matches_invocation(session_argv: list[str], visible_argv: list[str]) -> bool:
    """Return True iff a session_start's argv matches a transcript-visible
    python invocation's argv.

    Args after the interpreter must match exactly (shell-tokenized argv from
    the visible Bash command compared against ``sys.orig_argv[1:]``). The
    interpreter argv[0] matches either exactly or by basename (e.g. visible
    ``python`` matches session ``python``, ``/usr/bin/python``, or
    ``python3.11``); Python's argv[0] is whatever the shell passed to
    execvp, but path-form variation occurs across PATH lookups, alias
    expansion, and absolute-path invocations.
    """
    if not session_argv or not visible_argv:
        return False
    if session_argv[1:] != visible_argv[1:]:
        return False
    if session_argv[0] == visible_argv[0]:
        return True
    return Path(session_argv[0]).name == Path(visible_argv[0]).name


def _attribute_python_invocations(transcript_entries: list[dict], events: list[dict]) -> None:
    """Per-invocation cross-check by argv matching.

    Every transcript-visible python invocation must match an unused
    ``session_start`` by argv (interpreter basename + args). Surplus
    session_starts that have no visible counterpart are allowed; pip
    console-scripts and child processes spawned by an attributed run
    record session_start without surfacing as a Bash ``python`` argv[0]
    in the transcript.

    Argv matching closes the masking class where the transcript shows
    e.g. ``pip --version && python script.py`` while the event log
    contains a session_start for pip (matching the visible ``python``
    token by name) but none for the actual script. With basename-only
    pooling, pip's session would be popped and the script's missing
    session would go undetected. Argv matching binds pip's session to
    pip's argv shape (e.g. ``[".../bin/pip", "--version"]``), leaving the
    visible ``[python, script.py]`` invocation unattributed and forcing
    the merger to fail closed.
    """
    sessions: list[dict] = [e for e in events if e.get("event") == "session_start"]
    available_idx: list[int] = list(range(len(sessions)))

    visible_invocations: list[tuple[str, list[str]]] = []
    for entry in transcript_entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if not isinstance(command, str):
                continue
            for argv in _extract_python_invocations_from_command(command):
                if argv:
                    visible_invocations.append((command, argv))

    for command, visible_argv in visible_invocations:
        matched_idx: int | None = None
        for idx in available_idx:
            session_argv = sessions[idx].get("argv")
            if not isinstance(session_argv, list):
                continue
            if _session_argv_matches_invocation(session_argv, visible_argv):
                matched_idx = idx
                break
        if matched_idx is None:
            remaining = [sessions[i].get("argv") for i in available_idx]
            raise TelemetryMergeError(
                f"python invocation argv={visible_argv!r} in command "
                f"{command[:160]!r} has no matching session_start "
                f"(remaining session argvs: {remaining!r}). The interpreter "
                f"either ran without sitecustomize, or its session was "
                f"masked by an unrelated entry point (e.g. a pip "
                f"console-script). Cold-start eval is invalid."
            )
        available_idx.remove(matched_idx)


def _find_python_bypass_invocations_in_entries(entries: list[dict]) -> list[str]:
    """Return Bash commands containing a `-S` (or compact `-Sc`) flag,
    inline `PATH=...` interpreter override, `env python`, or `./python`.

    All of these forms can cause a Python interpreter to run without the
    shim loading: `-S` disables `site.py`; `PATH=/usr/bin python3` resolves
    to a non-per-arm-venv interpreter; `env python` resolves via PATH;
    `./python` invokes a local binary that almost certainly isn't the per-
    arm-venv interpreter.

    Per-invocation attribution cannot recover from these because the
    bypassed interpreter never writes `session_start`. Fail closed.

    Note: `-I` (isolated mode) is NOT a bypass; it implies `-E`, `-P`, and
    lowercase `-s` — none of which disable sitecustomize.
    """
    bypass_commands: list[str] = []
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if not isinstance(command, str):
                continue
            # PATH mutation / activation script / env / ./python — any of
            # these in a command containing a python invocation triggers
            # fail-closed because the resolved interpreter may not be the
            # per-arm-venv python.
            has_python = bool(_PYTHON_INVOCATION_RE.search(command))
            has_path_mutation = bool(_PATH_ASSIGNMENT_RE.search(command))
            has_python_env_var = bool(_PYTHON_ENV_VAR_RE.search(command))
            has_activation = bool(_SHELL_ACTIVATION_RE.search(command))
            has_wrapper = bool(_PYTHON_WRAPPER_RE.search(command))
            if (
                (has_path_mutation and has_python)
                or (has_python_env_var and has_python)
                or (has_activation and has_python)
                or has_wrapper
                or _ENV_BEFORE_PYTHON_RE.search(command)
                or _DOT_PYTHON_RE.search(command)
            ):
                bypass_commands.append(command)
                continue
            # -S / -Sc / etc. as flag to a python invocation
            for m in _PYTHON_INVOCATION_RE.finditer(command):
                rest = command[m.end() :]
                sep_match = re.search(r"[;&|]", rest)
                args_segment = rest[: sep_match.start()] if sep_match else rest
                if _PYTHON_BYPASS_FLAG_RE.search(args_segment):
                    bypass_commands.append(command)
                    break  # one bypass per command is enough to record
    return bypass_commands


def _count_python_invocations(stream_json_path: Path) -> int:
    """Return the total number of python interpreter invocations across all
    Bash tool commands in the transcript.

    Counts EVERY occurrence within each command, not just one per Bash call,
    so compound commands like
    ``python a.py && python -S b.py`` register as 2 invocations. The
    cross-check in ``_validate_shim_loaded`` requires this count to match
    the number of ``session_start`` events; an undercount here would let
    partial instrumentation silently pass.

    Parses each line of the stream-JSON transcript strictly via the shared
    ``_parse_jsonl_strict`` helper. Existence of the file is the caller's
    responsibility (``_validate_layer_artifacts`` runs first in
    ``merge_layers``).

    The word-boundary regex in ``_PYTHON_INVOCATION_RE`` and the structural
    tool_use/Bash/command match guard against false positives like
    ``/opt/python/`` (directory) or ``pythonic`` (substring).
    """
    entries = _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")
    count = 0
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if isinstance(command, str):
                count += len(_PYTHON_INVOCATION_RE.findall(command))
    return count


def _validate_shim_loaded(events: list[dict], transcript_entries: list[dict]) -> None:
    """Cross-layer fail-closed check before building the record.

    Cases:
    1. ``telemetry_missing`` sentinel present (written by the runner when
       the event log disappeared post-exec): raise.
    2. ANY visible python invocation uses ``-S`` (sitecustomize bypass) or
       a compact form like ``-Sc`` / ``-Sm``: raise. Per-invocation
       attribution cannot help here because the bypassed process WOULD
       have a sys.executable but never fires `session_start`.
    3. Per-invocation attribution by argv: every visible python invocation
       in the transcript MUST claim a `session_start` whose ``argv``
       matches the shell-tokenized invocation (interpreter token by
       basename, args[1:] exact). Unmatchable invocations raise. This
       replaces the previous sys_executable-only pooling, which could be
       masked by unrelated instrumented Python processes (e.g. `pip
       --version` supplying a session_start that satisfied a later
       relative `python` token without actually corresponding to it).
    4. Else accept.
    """
    for event in events:
        if event.get("event") == "telemetry_missing":
            raise TelemetryMergeError(
                "telemetry_missing sentinel present in event log; "
                "the agent's event log disappeared post-exec and the run "
                "is invalid for evaluation"
            )
    # A genuine shim-produced event log always writes `session_start`
    # before any hook events. Nonempty event log + zero session_starts
    # indicates truncation, deletion-then-recreation, or fabrication —
    # fail closed.
    has_session_start = any(e.get("event") == "session_start" for e in events)
    has_hook_events = any(
        e.get("event")
        in {
            "module_import",
            "guide_file_read",
            "estimator_init",
            "estimator_fit",
            "diagnostic_call",
            "warning_emitted",
        }
        for e in events
    )
    if has_hook_events and not has_session_start:
        raise TelemetryMergeError(
            "in-process event log has hook events but no session_start; "
            "log was truncated, deleted-and-recreated, or fabricated. "
            "Cold-start eval is invalid."
        )
    bypass_commands = _find_python_bypass_invocations_in_entries(transcript_entries)
    if bypass_commands:
        first = bypass_commands[0]
        if len(first) > 200:
            first = first[:200] + "..."
        raise TelemetryMergeError(
            f"python interpreter bypass flag (-S, including compact forms "
            f"like -Sc) detected in {len(bypass_commands)} transcript "
            f"command(s); first: {first!r}. This flag skips sitecustomize.py "
            f"and bypasses the in-process shim, leaving layer-2 silently "
            f"incomplete."
        )
    _attribute_python_invocations(transcript_entries, events)


def _build_diff_diff_record(
    events: list[dict],
    transcript_entries: list[dict],
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Construct a TelemetryRecord for arm='diff_diff' from parsed events.

    Discoverability fields default to ``False`` (the diff-diff arm requires
    explicit bool encoding per `TelemetryRecord.__post_init__`). Any event
    flips its field to ``True``.
    """
    guide_reads = [e for e in events if e.get("event") == "guide_file_read"]
    warnings_emitted = [e for e in events if e.get("event") == "warning_emitted"]
    estimator_inits = [e for e in events if e.get("event") == "estimator_init"]
    estimator_fits = [e for e in events if e.get("event") == "estimator_fit"]
    diagnostic_calls = [e for e in events if e.get("event") == "diagnostic_call"]

    opened = {filename: False for filename in _VARIANT_TO_FILENAME.values()}
    variants_seen: set[str] = set()
    for r in guide_reads:
        via = r.get("via")
        if via == "get_llm_guide":
            variant = r.get("variant", "")
            if variant in _VARIANT_TO_FILENAME:
                variants_seen.add(variant)
                opened[_VARIANT_TO_FILENAME[variant]] = True
        elif via == "open":
            # The shim writes basename-only filenames after its own
            # `_path_is_diff_diff_guide` check confirmed the path was under
            # `diff_diff/guides/`. Match exactly here; defense in depth.
            filename = r.get("filename", "")
            if filename in opened:
                opened[filename] = True

    # Layer-1 evidence: Claude's Read tool accessing a bundled guide file
    # without invoking Python. Layer-2 hooks see nothing in that path, so
    # without this OR-merge the merger would silently emit
    # opened_llms_*=False for an actual discovery.
    read_tool_opens = _scan_read_tool_guide_accesses_in_entries(transcript_entries)
    for filename, was_opened in read_tool_opens.items():
        if was_opened:
            opened[filename] = True

    return TelemetryRecord(
        arm="diff_diff",
        stream_json_path=stream_json_path,
        in_process_events_path=in_process_events_path,
        stderr_path=stderr_path,
        opened_llms_txt=opened["llms.txt"],
        opened_llms_practitioner=opened["llms-practitioner.txt"],
        opened_llms_autonomous=opened["llms-autonomous.txt"],
        opened_llms_full=opened["llms-full.txt"],
        called_get_llm_guide=any(r.get("via") == "get_llm_guide" for r in guide_reads),
        get_llm_guide_variants=tuple(sorted(variants_seen)),
        saw_fit_time_warning=bool(warnings_emitted),
        diagnostic_methods_invoked=tuple(
            sorted({e["name"] for e in diagnostic_calls if "name" in e})
        ),
        estimator_classes_instantiated=tuple(
            sorted({e["class"] for e in (estimator_inits + estimator_fits) if "class" in e})
        ),
    )


def _build_statsmodels_record(
    events: list[dict],
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Construct a TelemetryRecord for arm='statsmodels' from parsed events.

    Guide-related fields are ``None`` (sentinel: not applicable). Bool/tuple
    fields are populated from events targeting `statsmodels` — the shim
    in PR #4 has no statsmodels-specific hooks, so these will be False/()
    until the statsmodels arm instrumentation lands.
    """
    statsmodels_warnings = [
        e
        for e in events
        if e.get("event") == "warning_emitted" and "statsmodels" in str(e.get("filename", ""))
    ]
    return TelemetryRecord(
        arm="statsmodels",
        stream_json_path=stream_json_path,
        in_process_events_path=in_process_events_path,
        stderr_path=stderr_path,
        opened_llms_txt=None,
        opened_llms_practitioner=None,
        opened_llms_autonomous=None,
        opened_llms_full=None,
        called_get_llm_guide=None,
        get_llm_guide_variants=(),
        saw_fit_time_warning=bool(statsmodels_warnings),
        diagnostic_methods_invoked=(),
        estimator_classes_instantiated=(),
    )


def merge_layers(
    arm: str,
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Merge the three telemetry layers into a single record.

    `arm` drives the sentinel semantics on guide-discovery fields (see
    TelemetryRecord docstring): for arm == "statsmodels", `opened_llms_*`
    and `called_get_llm_guide` are encoded as None ("not applicable").

    Raises ``TelemetryMergeError`` on fail-closed conditions:
    - In-process event log missing or malformed.
    - Runner-written ``telemetry_missing`` sentinel present.
    - Empty or non-object transcript.
    - Shim event-write failure marker in stderr.
    - Python bypass flag (`-S`) in any transcript command.
    - Stream-JSON transcript missing a terminal successful `result` entry
      (truncated capture).
    - Any visible python invocation that cannot be attributed to a
      session_start by ``argv`` (interpreter basename + args).
    - A guide-file Read tool_use with no matching tool_result
      (incomplete transcript / cannot determine success).

    The transcript is parsed exactly once (per-invocation attribution,
    guide-read scan, and bypass detection all consume the parsed entries).
    """
    transcript_entries = _validate_layer_artifacts(stream_json_path, stderr_path)
    if _scan_stderr_for_shim_failures(stderr_path):
        raise TelemetryMergeError(
            f"shim event-write failure marker present in {stderr_path}; "
            f"at least one layer-2 event was dropped mid-run and the "
            f"per-run record may be silently incomplete"
        )
    events = _read_events(in_process_events_path)
    _validate_shim_loaded(events, transcript_entries)
    if arm == "diff_diff":
        return _build_diff_diff_record(
            events,
            transcript_entries,
            stream_json_path,
            in_process_events_path,
            stderr_path,
        )
    elif arm == "statsmodels":
        return _build_statsmodels_record(
            events, stream_json_path, in_process_events_path, stderr_path
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")
