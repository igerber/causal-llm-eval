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
import os.path
import re
from dataclasses import dataclass
from pathlib import Path

from harness.shell_parser import (
    RunValidityError,
    ShellCommandIndeterminate,
    ShellCommandParseError,
    argv_contains_bypass_flag,
    find_python_bypass_invocations,
    parse_python_invocations,
)


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


class TelemetryMergeError(RunValidityError):
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

    Subclass of ``RunValidityError`` so callers catching either class
    handle this case. ``RunValidityError`` is the neutral parent that
    layer-1 parser issues (``ShellCommandIndeterminate``,
    ``ShellCommandParseError``) also derive from.
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
    """Return True if outer Claude CLI stderr contains the shim's event-write
    failure marker.

    The shim's hook wrappers catch transient OSError on event writes so the
    agent's diff_diff call still completes (avoids aborting on telemetry
    hiccups). When that happens, `_write_event` first prints
    `[pyruntime] cannot write event to <path>: <err>` to stderr. The merger
    looks for that marker post-hoc: presence means at least one event was
    dropped, so the per-run record may be silently incomplete.

    NOTE: this only covers stderr from the outer ``claude --bare`` process.
    When the shim runs inside an agent-invoked python subprocess (the
    common case), its stderr is captured into the corresponding Bash
    ``tool_result`` block in the stream-JSON transcript, not into
    ``cli_stderr.log``. Use ``_scan_tool_results_for_shim_failures`` for
    that path; both are checked in ``merge_layers``.
    """
    try:
        content = stderr_path.read_text(errors="replace")
    except OSError:
        # Existence was already validated; a read error here is itself a
        # capture problem worth surfacing.
        return True
    return _SHIM_WRITE_FAILURE_MARKER in content


def _scan_tool_results_for_shim_failures(transcript_entries: list[dict]) -> bool:
    """Return True if any Bash ``tool_result`` content contains the shim's
    event-write failure marker.

    Most shim event-write failures occur inside an agent-spawned python
    subprocess (Claude's Bash tool runs the python child whose stderr is
    captured by Claude into the matching ``tool_result`` block). The
    outer ``cli_stderr.log`` only carries stderr from the ``claude
    --bare`` process itself, which rarely sees inner subprocess output,
    so a marker emitted by the shim is invisible to the layer-3 scan.

    Iterating ``tool_result`` blocks closes that gap. Stringified content
    (Claude sometimes returns a single string) and list-shaped content
    (one or more text blocks) are both supported. A match is enough to
    fail closed; per-block granularity is not needed because the
    per-run record cannot be trusted once a single event is known to
    have been dropped.
    """
    for entry in transcript_entries:
        for block in _iter_tool_result_blocks(entry):
            content = block.get("content")
            if isinstance(content, str):
                if _SHIM_WRITE_FAILURE_MARKER in content:
                    return True
            elif isinstance(content, list):
                for sub in content:
                    if isinstance(sub, dict):
                        text = sub.get("text") or sub.get("content") or ""
                        if isinstance(text, str) and _SHIM_WRITE_FAILURE_MARKER in text:
                            return True
                    elif isinstance(sub, str) and _SHIM_WRITE_FAILURE_MARKER in sub:
                        return True
    return False


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
    # Require the explicit `subtype == "success"` shape rather than just
    # "type == result and no is_error". Claude's stream-JSON can emit
    # result frames with subtypes like ``error_max_turns``,
    # ``error_during_execution``, etc. that do NOT set ``is_error=true``
    # but also do not represent a clean run completion.
    if not (
        isinstance(last, dict)
        and last.get("type") == "result"
        and last.get("subtype") == "success"
        and not last.get("is_error")
    ):
        raise TelemetryMergeError(
            f"stream-JSON transcript at {stream_json_path} does not end "
            f"with a `type=result subtype=success` entry; capture is "
            f"truncated, the run failed, or the result subtype is unknown, "
            f"and per-run telemetry cannot be treated as complete"
        )
    return transcript_entries


def _read_events(path: Path, *, venv_path: Path | None = None) -> list[dict]:
    """Parse the in-process event log into a list of dicts.

    Raises ``TelemetryMergeError`` if the file is missing or contains a
    non-JSON line. An empty (0-byte) file is permitted and returns ``[]``;
    the validity check is the caller's responsibility.

    PR #5: ``venv_path`` is forwarded to ``_validate_event_schemas`` so
    that ``exec_python`` events can be checked against the run's
    venv-root-anchored ``executable`` allowlist.
    """
    if not path.exists():
        raise TelemetryMergeError(
            f"in-process event log not found at {path}; "
            f"the runner should have written a telemetry_missing sentinel"
        )
    events = _parse_jsonl_strict(path, "event log")
    _validate_event_schemas(events, path, venv_path=venv_path)
    return events


# Required field schemas for known event types. The shim writes these under
# strict producer-side control (see ``harness/sitecustomize_template.py``);
# missing required fields means the event is malformed and would silently
# zero out telemetry fields if accepted as-is. Reject at parse time.
_EVENT_SCHEMA: dict[str, tuple[str, ...]] = {
    "session_start": ("argv", "pid"),
    "session_end": ("pid",),
    "module_import": ("module",),
    "guide_file_read": (
        "via",
    ),  # via=='get_llm_guide' needs variant; via=='open' needs filename - checked below
    "estimator_init": ("class",),
    "estimator_fit": ("class",),
    "diagnostic_call": ("name",),
    "warning_emitted": ("filename",),
    # PR #5: layer-1.5 wrapper-emitted exec_python event. Required fields per
    # the wrapper script + venv-root-anchored executable check applied when
    # ``venv_path`` is supplied to ``merge_layers``.
    "exec_python": ("pid", "ppid", "ts", "executable", "argv"),
    # telemetry_missing sentinel has no required fields; the merger raises on
    # its presence regardless of payload.
    "telemetry_missing": (),
}


def _expected_exec_python_executable(venv_path: Path) -> str:
    """The canonical ``executable`` value for an ``exec_python`` event
    given a known venv root, in ``os.path.normpath`` form.

    PR #5 R1: the real interpreter lives at
    ``${venv}/.pyruntime-real/python-real`` so it stays off the agent's
    PATH. The wrapper records the path as
    ``$(dirname "$0")/../.pyruntime-real/python-real`` (with an embedded
    ``/..`` segment); the merger normalizes both sides before comparing,
    so the wrapper's literal form and the resolved form both compare
    equal to this canonical value.
    """
    return os.path.normpath(str(venv_path / ".pyruntime-real" / "python-real"))


def _validate_event_schemas(
    events: list[dict],
    path: Path,
    *,
    venv_path: Path | None = None,
) -> None:
    """Reject malformed known-event records.

    Validates two layers per known event type:

    1. **Required field presence**: missing required fields would silently
       zero out telemetry under downstream ``event.get("field", "")``
       reads.
    2. **Required field type and enum**: a present-but-wrong-typed value
       (e.g. ``argv="string"`` instead of ``list``) or an unknown enum
       member (e.g. ``variant="not-real"``, ``filename="not-a-guide.txt"``)
       would also silently zero out telemetry because the unknown value
       doesn't match any bundled-guide check.

    Unknown event types are ignored (forward-compatibility: a future shim
    may emit new event types, and we don't want old mergers to reject
    them outright). Only KNOWN event types are schema-checked.

    PR #5: when ``venv_path`` is supplied, ``exec_python`` events are
    additionally validated against the run's venv-root-anchored allowlist
    (mirrors the rename set produced by ``harness.venv_pool._install_python_wrapper``).
    Forged executables outside the venv root fail validation. When
    ``venv_path`` is None (legacy fixture mode), only the type checks
    apply.
    """
    valid_filenames = set(_VARIANT_TO_FILENAME.values())
    valid_variants = set(_VARIANT_TO_FILENAME.keys())
    for i, event in enumerate(events):
        event_type = event.get("event")
        if not isinstance(event_type, str):
            continue  # not a typed event; ignore
        required = _EVENT_SCHEMA.get(event_type)
        if required is None:
            continue

        def _bail(msg: str) -> None:
            raise TelemetryMergeError(f"event log {path} entry {i} (event={event_type!r}): {msg}")

        for field in required:
            if field not in event:
                _bail(
                    f"missing required field {field!r}; the shim wrote a "
                    f"malformed record and downstream telemetry would "
                    f"silently omit this evidence"
                )

        # Per-event-type type and enum checks.
        if event_type == "session_start":
            argv = event["argv"]
            if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                _bail(f"argv must be list[str], got {type(argv).__name__}={argv!r}")
            pid = event["pid"]
            # bool is a subclass of int; reject explicitly so True/False
            # don't slip through as valid pids.
            if not isinstance(pid, int) or isinstance(pid, bool):
                _bail(f"pid must be int, got {type(pid).__name__}={pid!r}")
        elif event_type == "session_end":
            pid = event["pid"]
            if not isinstance(pid, int) or isinstance(pid, bool):
                _bail(f"pid must be int, got {type(pid).__name__}={pid!r}")
        elif event_type == "module_import":
            module = event["module"]
            if not isinstance(module, str):
                _bail(f"module must be str, got {type(module).__name__}={module!r}")
        elif event_type == "guide_file_read":
            via = event.get("via")
            if via not in ("get_llm_guide", "open"):
                _bail(f"unknown via={via!r}; recognized values: " f"'get_llm_guide', 'open'")
            if via == "get_llm_guide":
                if "variant" not in event:
                    _bail("missing required 'variant' field for via='get_llm_guide'")
                variant = event["variant"]
                if not isinstance(variant, str):
                    _bail(f"variant must be str, got " f"{type(variant).__name__}={variant!r}")
                if variant not in valid_variants:
                    _bail(
                        f"unknown variant={variant!r}; recognized values: "
                        f"{sorted(valid_variants)!r}"
                    )
            elif via == "open":
                if "filename" not in event:
                    _bail("missing required 'filename' field for via='open'")
                filename = event["filename"]
                if not isinstance(filename, str):
                    _bail(f"filename must be str, got " f"{type(filename).__name__}={filename!r}")
                if filename not in valid_filenames:
                    _bail(
                        f"unknown filename={filename!r}; recognized values: "
                        f"{sorted(valid_filenames)!r}"
                    )
        elif event_type in ("estimator_init", "estimator_fit"):
            class_name = event["class"]
            if not isinstance(class_name, str):
                _bail(f"class must be str, got " f"{type(class_name).__name__}={class_name!r}")
        elif event_type == "diagnostic_call":
            name = event["name"]
            if not isinstance(name, str):
                _bail(f"name must be str, got {type(name).__name__}={name!r}")
        elif event_type == "warning_emitted":
            filename = event["filename"]
            if not isinstance(filename, str):
                _bail(f"filename must be str, got " f"{type(filename).__name__}={filename!r}")
        elif event_type == "exec_python":
            pid = event["pid"]
            if not isinstance(pid, int) or isinstance(pid, bool):
                _bail(f"pid must be int, got {type(pid).__name__}={pid!r}")
            ppid = event["ppid"]
            if not isinstance(ppid, int) or isinstance(ppid, bool):
                _bail(f"ppid must be int, got {type(ppid).__name__}={ppid!r}")
            ts = event["ts"]
            if not isinstance(ts, str):
                _bail(f"ts must be str, got {type(ts).__name__}={ts!r}")
            executable = event["executable"]
            if not isinstance(executable, str):
                _bail(f"executable must be str, got " f"{type(executable).__name__}={executable!r}")
            argv = event["argv"]
            if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                _bail(f"argv must be list[str], got {type(argv).__name__}={argv!r}")
            # Venv-root-anchored exact match (PR #5): structurally rejects
            # agent-forged exec_python events whose `executable` points
            # outside the run's venv root. Both sides normalized through
            # os.path.normpath so the wrapper's literal
            # ``${venv}/bin/../.pyruntime-real/python-real`` and the
            # resolved form ``${venv}/.pyruntime-real/python-real``
            # compare equal. Skipped when venv_path is None (legacy
            # fixture mode).
            if venv_path is not None:
                expected = _expected_exec_python_executable(venv_path)
                if os.path.normpath(executable) != expected:
                    _bail(
                        f"executable {executable!r} is not the run's venv "
                        f"real interpreter; expected {expected!r}. This "
                        f"guards against agent-forged exec_python events."
                    )


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


def _validate_bash_tool_results_complete(entries: list[dict]) -> None:
    """Raise ``TelemetryMergeError`` if any Bash ``tool_use`` in the
    transcript lacks a matching ``tool_result``.

    Claude's tool-call flow guarantees every ``tool_use`` is followed by a
    ``tool_result`` in the same conversation (success or error). A
    missing match indicates the transcript was truncated between the
    request and the response, which silently drops the subprocess stderr
    for that command. If that stderr would have contained the shim's
    ``[pyruntime] cannot write event`` marker, a dropped layer-2 event
    becomes silently invisible and the merger emits an all-False
    telemetry record. Reciprocal of the Read tool_result fail-closed
    check (``_scan_read_tool_guide_accesses_in_entries``); same
    invariant, different tool surface.

    Tool uses without an ``id`` field are also rejected; without an id
    the match cannot be established and the result may be missing
    silently.
    """
    bash_uses: dict[str, dict] = {}
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise TelemetryMergeError(
                    "Bash tool_use is missing or has empty 'id' field; the "
                    "transcript cannot match its tool_result, so a silently "
                    "dropped stderr (including a possible shim event-write "
                    "failure marker) cannot be ruled out"
                )
            bash_uses[tool_use_id] = block

    if not bash_uses:
        return

    matched_ids: set[str] = set()
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id in bash_uses:
                matched_ids.add(tool_use_id)

    unmatched = set(bash_uses) - matched_ids
    if unmatched:
        raise TelemetryMergeError(
            f"Bash tool_use(s) {sorted(unmatched)!r} have no matching "
            f"tool_result in the transcript. The transcript is truncated "
            f"between request and response, so the subprocess stderr "
            f"(which may contain the shim event-write failure marker) "
            f"is silently missing. Per-run record cannot be trusted."
        )


def _validate_python_bash_results_non_error(entries: list[dict]) -> None:
    """Raise ``TelemetryMergeError`` if any Bash ``tool_result`` for a
    command containing a Python invocation has ``is_error=True``.

    Completes the failure-marker chain established by R13 and R22:

    - R13 catches the shim's ``[pyruntime] cannot write event`` marker
      when it reaches stderr (``_scan_stderr_for_shim_failures``) or a
      Bash tool_result's content (``_scan_tool_results_for_shim_failures``).
    - R22 hardens the shim to ``os._exit(2)`` on event-write failure
      rather than silently continuing.
    - This validator closes the remaining hole: if the agent invokes
      ``python script.py 2>/dev/null`` and the shim hard-exits, the
      stderr marker never reaches a scannable surface, but the
      subprocess exit code is non-zero, which surfaces as
      ``tool_result.is_error=True`` on the Bash result. Any non-zero
      exit for a Python invocation is therefore treated as a
      telemetry-completeness failure: the subprocess died with the
      content of its run undetermined, and a downstream guide-read /
      estimator / diagnostic event might have been dropped.

    Non-Python Bash commands (``ls``, ``cat``, etc.) are NOT subject to
    this check; their failure has no telemetry-completeness implication.
    Bash commands without a recognizable Python invocation by
    ``parse_python_invocations`` (in ``harness.shell_parser``) are also
    exempt - only commands the merger considers tracked Python launches.
    """
    bash_results: dict[str, dict] = {}
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id:
                bash_results[tool_use_id] = result

    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                # Missing id is caught by _validate_bash_tool_results_complete;
                # skip here.
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if not isinstance(command, str) or not command:
                continue
            python_invocations = parse_python_invocations(command)
            if not python_invocations:
                continue
            result = bash_results.get(tool_use_id)
            if result is None:
                # Missing tool_result is caught by
                # _validate_bash_tool_results_complete; skip here.
                continue
            if result.get("is_error", False):
                raise TelemetryMergeError(
                    f"Bash tool_result for tool_use_id {tool_use_id!r} "
                    f"(Python invocation: {python_invocations[0]!r}) has "
                    f"is_error=True. The subprocess exited with a "
                    f"non-zero status, which may indicate the shim's "
                    f"hard-exit on an event-write failure - particularly "
                    f"when stderr is suppressed (e.g. '2>/dev/null') so "
                    f"the [pyruntime] failure marker is invisible to the "
                    f"merger. Telemetry completeness for this run cannot "
                    f"be verified; per-run record cannot be trusted."
                )


def _validate_tool_use_ids_unique(entries: list[dict]) -> None:
    """Raise ``TelemetryMergeError`` if any two ``tool_use`` blocks share
    a tool_use ``id`` across the transcript.

    Claude's tool-call flow assigns unique IDs to every tool_use. A
    duplicate id - whether two Bash uses, two Read uses, or a Bash and a
    Read sharing the same id - silently cross-matches in the id-keyed
    dicts that ``_validate_bash_tool_results_complete`` and
    ``_scan_read_tool_guide_accesses_in_entries`` build, so a missing
    tool_result on one block could be falsely "covered" by another
    block's tool_result. Fail closed at merge time.

    Blocks without a string id are skipped here; the per-surface
    validators (Bash and guide-Read) enforce id presence on the surfaces
    that need them.
    """
    seen: set[str] = set()
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                continue
            if tool_use_id in seen:
                raise TelemetryMergeError(
                    f"tool_use_id {tool_use_id!r} appears on two tool_use "
                    f"blocks in the transcript; tool-use IDs must be unique "
                    f"to avoid silent cross-tool result overwrites"
                )
            seen.add(tool_use_id)


def _validate_tool_result_ids_unique(entries: list[dict]) -> None:
    """Raise ``TelemetryMergeError`` if any two ``tool_result`` blocks share
    a ``tool_use_id`` value across the transcript.

    Mirror of ``_validate_tool_use_ids_unique`` on the result side.
    Claude's tool-call flow assigns exactly one ``tool_result`` per
    ``tool_use``. A duplicate ``tool_use_id`` across results silently
    overwrites a prior result in the id-keyed dicts built by
    ``_validate_python_bash_results_non_error`` and
    ``_scan_read_tool_guide_accesses_in_entries``: a first result with
    ``is_error=True`` can be hidden by a second result with
    ``is_error=False``, reopening the hard-exit observability hole that
    R23 closed.

    Results without a string ``tool_use_id`` are skipped; matching them
    to a tool_use is already impossible regardless of uniqueness.
    """
    seen: set[str] = set()
    for entry in entries:
        for block in _iter_tool_result_blocks(entry):
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                continue
            if tool_use_id in seen:
                raise TelemetryMergeError(
                    f"tool_use_id {tool_use_id!r} appears on two tool_result "
                    f"blocks in the transcript; results must be unique per "
                    f"tool_use to avoid masking is_error=True with a later "
                    f"is_error=False (or vice versa for layer-1 guide reads)"
                )
            seen.add(tool_use_id)


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
            # Lex-normalize before adjacent-part matching so paths with
            # ``.`` / ``..`` / duplicate separators resolve to their
            # canonical form. ``Path(file_path).parent`` does not
            # normalize, so a successful Read of
            # ``/install/diff_diff/guides/../guides/llms.txt`` would
            # otherwise have ``p.parent.parent.name == '..'`` and miss
            # the check (R24 P1#2). ``os.path.normpath`` is purely
            # lexical (no filesystem access); we cannot use
            # ``Path.resolve()`` since the transcript path is from the
            # agent's environment, not ours. Mirror of
            # ``_path_is_diff_diff_guide`` in sitecustomize_template
            # which uses ``Path.resolve()`` (the resolution there IS
            # filesystem-backed because the shim runs in the agent's
            # process).
            p = Path(os.path.normpath(file_path))
            if not (
                p.name in known_filenames
                and p.parent.name == "guides"
                and p.parent.parent.name == "diff_diff"
            ):
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise TelemetryMergeError(
                    f"Read tool_use targeting guide path {file_path!r} is "
                    f"missing or has empty 'id' field; the transcript "
                    f"cannot match its tool_result, so a silent drop of "
                    f"layer-1 guide-discovery evidence cannot be ruled out"
                )
            if tool_use_id in pending:
                raise TelemetryMergeError(
                    f"Read tool_use for guide path {file_path!r} reuses "
                    f"tool_use_id {tool_use_id!r}; transcript tool-use IDs "
                    f"must be unique to avoid silent result overwrites"
                )
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


def _scan_grep_tool_guide_accesses_in_entries(
    entries: list[dict],
) -> dict[str, bool]:
    """Return ``{filename: True}`` for each bundled guide file SUCCESSFULLY
    accessed via Claude's Grep tool in the transcript.

    Mirror of ``_scan_read_tool_guide_accesses_in_entries`` for the
    Grep tool surface (R32 P0; R33 P0 extended with glob handling).
    Grep reads file content while searching; a successful Grep that
    can be attributed to one or more bundled guide files exposes that
    content to the agent the same way Read does.

    Grep input shape (Claude's Grep tool):
    - ``path``: file or directory to search.
    - ``glob``: file-pattern filter applied to files under ``path``.
    - ``pattern``: search regex (content; not relevant to guide attribution).

    Attribution rules:
    - ``path`` is a specific bundled guide file: flag that one file.
    - ``glob`` resolves (under ``path``) to specific bundled guide
      file(s): flag each matching file by ``fnmatch``. Glob expansion
      is lex-only (we can't list the agent's filesystem); we match the
      glob against the known bundled-guide basenames.
    - Ambiguous searches (``path`` is the guides directory and ``glob``
      does NOT narrow to specific files): fail closed. We cannot
      attribute which guide files the agent saw without parsing
      ``tool_result.content``, which would be fragile across Grep
      output modes. Refusing to attest is the safe default.

    Same id-presence and tool_result-matching contract as the Read scanner."""
    known_filenames = set(_VARIANT_TO_FILENAME.values())

    pending: dict[str, frozenset[str]] = {}
    ambiguous: dict[str, str] = {}  # tool_use_id -> source path-or-glob for error msg
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Grep":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            grep_path = tool_input.get("path", "")
            grep_glob = tool_input.get("glob", "")
            if not isinstance(grep_path, str):
                grep_path = ""
            if not isinstance(grep_glob, str):
                grep_glob = ""
            if not grep_path and not grep_glob:
                continue
            matched_files, is_ambiguous = _classify_grep_target(
                grep_path, grep_glob, known_filenames
            )
            if not matched_files and not is_ambiguous:
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise TelemetryMergeError(
                    f"Grep tool_use targeting guide-related path "
                    f"{grep_path!r} (glob={grep_glob!r}) is missing or "
                    f"has empty 'id' field; the transcript cannot match "
                    f"its tool_result, so a silent drop of layer-1 "
                    f"guide-discovery evidence cannot be ruled out"
                )
            if tool_use_id in pending or tool_use_id in ambiguous:
                raise TelemetryMergeError(
                    f"Grep tool_use for guide-related path {grep_path!r} "
                    f"reuses tool_use_id {tool_use_id!r}; transcript "
                    f"tool-use IDs must be unique to avoid silent result "
                    f"overwrites"
                )
            if is_ambiguous:
                ambiguous[tool_use_id] = f"path={grep_path!r} glob={grep_glob!r}"
            else:
                pending[tool_use_id] = matched_files

    if not pending and not ambiguous:
        return {}

    # Match tool_results to both attributed (pending) and ambiguous
    # Greps. An ambiguous Grep with a NON-ERROR result raises fail-
    # closed: we cannot attest a record where the agent searched
    # guide-related territory and we don't know which files matched.
    # An ambiguous Grep with is_error=True is benign (the search
    # failed; no content exposed).
    opened: dict[str, bool] = {}
    matched_ids: set[str] = set()
    ambiguous_succeeded: list[str] = []
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            is_err = result.get("is_error", False)
            if tool_use_id in pending:
                matched_ids.add(tool_use_id)
                if not is_err:
                    for fname in pending[tool_use_id]:
                        opened[fname] = True
            elif tool_use_id in ambiguous:
                matched_ids.add(tool_use_id)
                if not is_err:
                    ambiguous_succeeded.append(ambiguous[tool_use_id])

    if ambiguous_succeeded:
        raise TelemetryMergeError(
            f"Grep tool_use over guide-related territory cannot be "
            f"attributed to specific guide files; the search succeeded "
            f"but we don't know which guides the agent saw. Sources: "
            f"{ambiguous_succeeded!r}. Either narrow the search with a "
            f"file-specific glob or wait for PR #5's exec-time wrapper "
            f"to provide independent attestation."
        )

    unmatched_ids = (set(pending) | set(ambiguous)) - matched_ids
    if unmatched_ids:
        sources = [
            f"path={p!r}"
            for tid in unmatched_ids
            for p in [next(iter(pending.get(tid, frozenset())), ambiguous.get(tid, "?"))]
        ]
        raise TelemetryMergeError(
            f"Grep tool_use for guide-related target(s) {sources!r} "
            f"has no matching tool_result in the transcript (tool_use_ids: "
            f"{sorted(unmatched_ids)!r}). Transcript is incomplete; the "
            f"per-run record cannot be trusted to reflect guide-discovery state."
        )
    return opened


def _resolve_glob_to_guide_basename(
    glob: str,
    *,
    path_at_guides_dir: bool,
) -> str | None:
    """Resolve a Grep ``glob`` parameter to a basename pattern that can
    be fnmatched against bundled guide filenames, assuming the search
    path is already scoped under ``diff_diff`` or ``diff_diff/guides``.

    Returns the basename pattern (e.g. ``llms.txt``, ``llms*.txt``), or
    ``None`` if the glob does not reduce to a guide-file pattern.

    Recognized glob shapes (after path scoping by the caller):
      - When ``path`` IS the guides directory: glob is a basename
        pattern directly, OR a recursive form like ``**/<basename>``
        (which still resolves to the basename under guides).
      - When ``path`` is the diff_diff package root: glob must point
        AT guides, so accept ``guides/<basename>``, ``./guides/<basename>``,
        or recursive ``**/guides/<basename>``.

    Bash ``**`` is glob-recursive and POSIX-bash needs ``globstar``
    enabled, but Claude's Grep glob treats both ``**`` and ``*`` as
    recursive-ish wildcards on its own; we accept both to mirror agent
    usage patterns.
    """
    if path_at_guides_dir:
        # path is already at guides; glob is a basename or
        # ``**/<basename>`` / ``*/<basename>`` recursive form.
        if "/" in glob:
            # Strip recursive prefix and treat the last segment as the
            # basename pattern.
            last = glob.rsplit("/", 1)[-1]
            if "/" in last:
                return None
            return last
        return glob

    # path is at diff_diff (one level up from guides). Glob must
    # explicitly target the guides subdirectory.
    for prefix in ("guides/", "./guides/", "**/guides/", "*/guides/"):
        if glob.startswith(prefix):
            rest = glob[len(prefix) :]
            if "/" in rest:
                # Further nesting like guides/sub/llms.txt - bundled
                # layout has no subdirs under guides.
                return None
            return rest
    return None


def _classify_grep_target(
    grep_path: str,
    grep_glob: str,
    known_filenames: set,
) -> tuple[frozenset[str], bool]:
    """Return (attributed_files, is_ambiguous) for a Grep input.

    - attributed_files: the bundled guide filenames the Grep can be
      proved to have searched. Empty if the Grep is not guide-related.
    - is_ambiguous: True if the Grep IS guide-related but cannot be
      attributed to specific guide files. The caller fails closed when
      this is True (so the merger refuses to attest a clean record for
      a search whose target file set we can't enumerate).
    """
    import fnmatch

    p = Path(os.path.normpath(grep_path)) if grep_path else None

    # Direct guide-file path.
    if (
        p is not None
        and p.name in known_filenames
        and p.parent.name == "guides"
        and p.parent.parent.name == "diff_diff"
    ):
        return frozenset({p.name}), False

    # path under or at the guides directory.
    path_at_guides_dir = p is not None and p.name == "guides" and p.parent.name == "diff_diff"
    path_at_diff_diff = p is not None and p.name == "diff_diff"

    if grep_glob:
        # Glob attribution requires the path to be SCOPED under the
        # bundled diff_diff package. Otherwise a glob like
        # ``guides/llms.txt`` paired with ``path=/tmp`` would match
        # files outside our bundled location and falsely flag a guide
        # as opened (R34 P1).
        if not (path_at_guides_dir or path_at_diff_diff):
            return frozenset(), False

        glob_pattern = _resolve_glob_to_guide_basename(
            grep_glob, path_at_guides_dir=path_at_guides_dir
        )
        if glob_pattern is None:
            # Glob doesn't resolve to a guide-file basename pattern
            # under the scoped path. Not a guide search.
            return frozenset(), False
        matches: set[str] = set()
        for fname in known_filenames:
            if fnmatch.fnmatchcase(fname, glob_pattern):
                matches.add(fname)
        # Empty matches set: glob is shaped like a guide pattern but
        # doesn't match any bundled file (e.g. ``llms-future.txt``).
        # Treat as non-guide rather than ambiguous.
        return frozenset(matches), False

    # No glob; path-only.
    if path_at_guides_dir:
        # Ambiguous: agent grep'd the whole guides directory. We
        # cannot statically know which files matched without parsing
        # tool_result.content. Fail closed.
        return frozenset(), True
    if path_at_diff_diff:
        # Grep on the package root with no glob. The agent may or may
        # not have seen guides depending on Grep's recursion. Fail
        # closed.
        return frozenset(), True

    return frozenset(), False


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


def _session_argv_matches_invocation(
    session_argv: list[str],
    visible_argv: list[str],
    *,
    venv_path: Path | None = None,
) -> bool:
    """Return True iff a session_start's argv matches a transcript-visible
    python invocation's argv.

    Args after the interpreter must match exactly (shell-tokenized argv from
    the visible Bash command compared against ``sys.orig_argv[1:]``).

    Interpreter argv[0] matching:

    - If the VISIBLE argv[0] contains any path separator (``/``), require
      exact equality with the session's argv[0]. EXCEPTION (PR #5 R3 P1):
      when ``venv_path`` is supplied AND both the session's argv[0] and
      the visible argv[0] resolve to python-family executables under the
      same venv root, treat them as equivalent. This bridges valid
      ``${venv}/bin/python script.py`` invocations to their session
      ``${venv}/.pyruntime-real/python-real script.py`` while still
      rejecting off-venv absolute paths like ``/usr/bin/python``.
    - If the VISIBLE argv[0] is a bare token with no path separator
      (``python`` / ``python3`` / ``python3.11``), allow path-vs-basename
      fallback. The shell PATH-resolves the bare name to an absolute
      interpreter, so ``sys.orig_argv[0]`` carries the resolved path even
      though the visible argv[0] is just the bare token.
    """
    if not session_argv or not visible_argv:
        return False
    if session_argv[1:] != visible_argv[1:]:
        return False
    if session_argv[0] == visible_argv[0]:
        return True
    if "/" in visible_argv[0]:
        # PR #5 R3 P1: bridge venv-internal path-qualified invocations.
        # ``${venv}/bin/python script.py`` is the wrapper; the session's
        # argv[0] is ``${venv}/.pyruntime-real/python-real script.py``.
        # Both resolve under the same venv root; both are python-family.
        if venv_path is not None and _both_under_same_venv_python(
            session_argv[0], visible_argv[0], venv_path
        ):
            return True
        return False
    session_basename = Path(session_argv[0]).name
    visible_basename = Path(visible_argv[0]).name
    # PR #5 R2: after the wrapper's ``exec "$real" "$@"`` the real python
    # records ``sys.orig_argv[0]`` as the path to ``python-real``,
    # regardless of which alias (``python`` / ``python3`` / ``python3.X``)
    # the agent originally typed. The merger uses pid-keyed matching as
    # the canonical bridge (see _validate_three_layer_consistency), but
    # the legacy attribution still runs first. Treat any python-family
    # basename (``python``, ``python3``, ``python3.X``, plus the
    # ``-real`` variants) as equivalent for argv[0] matching: a visible
    # ``python3 script.py`` should attribute to the session
    # ``${venv}/.pyruntime-real/python-real script.py`` because the
    # wrapper routed the invocation.
    if _is_python_family_basename(session_basename) and _is_python_family_basename(
        visible_basename
    ):
        return True
    return session_basename == visible_basename


_PYTHON_FAMILY_BASENAME_RE = re.compile(r"^(?:python(?:3(?:\.\d+)?)?(?:-real)?|\.actual-python)$")


def _is_python_family_basename(name: str) -> bool:
    """Return True if ``name`` is any python interpreter alias the
    venv may produce: ``python``, ``python3``, ``python3.11``,
    ``python-real``, ``python3-real``, ``python3.11-real``, or
    ``.actual-python`` (the R3 two-stage hidden binary).
    """
    return bool(_PYTHON_FAMILY_BASENAME_RE.match(name))


def _both_under_same_venv_python(session_argv0: str, visible_argv0: str, venv_path: Path) -> bool:
    """Return True iff both paths resolve to a python-family executable
    under ``venv_path``.

    Used by ``_session_argv_matches_invocation`` to bridge
    ``${venv}/bin/python script.py`` (visible) to the session's
    ``${venv}/.pyruntime-real/python-real script.py``. Both paths must
    normalize to a location inside ``venv_path`` AND have a
    python-family basename.
    """
    venv_root = os.path.normpath(str(venv_path))
    sess_norm = os.path.normpath(session_argv0)
    vis_norm = os.path.normpath(visible_argv0)
    # Both paths must be under the venv root.
    if not sess_norm.startswith(venv_root + os.sep):
        return False
    # Visible may be relative; if so, resolve it against venv_root's parent
    # only when it starts with the venv directory name. Conservative: require
    # absolute or already-normalized path prefix.
    if not vis_norm.startswith(venv_root + os.sep):
        return False
    return _is_python_family_basename(Path(sess_norm).name) and _is_python_family_basename(
        Path(vis_norm).name
    )


def _attribute_python_invocations(
    transcript_entries: list[dict],
    events: list[dict],
    *,
    venv_path: Path | None = None,
) -> None:
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
            for argv in parse_python_invocations(command):
                if argv:
                    visible_invocations.append((command, argv))

    for command, visible_argv in visible_invocations:
        matched_idx: int | None = None
        for idx in available_idx:
            session_argv = sessions[idx].get("argv")
            if not isinstance(session_argv, list):
                continue
            if _session_argv_matches_invocation(session_argv, visible_argv, venv_path=venv_path):
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
        # session_end pairing is enforced globally in _validate_shim_loaded
        # for EVERY session_start (attributed or surplus). By the time we
        # reach attribution here, all session_starts are guaranteed to
        # have matching session_ends - no per-invocation check needed.
        available_idx.remove(matched_idx)


def _partition_exec_python_events(
    events: list[dict], runner_pid: int
) -> tuple[list[dict], list[dict]]:
    """Split ``exec_python`` events into ``(sentinel, agent)`` lists by ppid.

    The wrapper records ``ppid`` as the parent of the wrapper invocation:
        - The build-time sentinel runs as a direct child of the runner
          process (``subprocess.run`` from ``run_one``), so its ppid equals
          ``runner_pid``.
        - Agent-spawned python invocations run under the ``claude --bare``
          subprocess tree (``ppid != runner_pid``).

    Binary partition by ``ppid == runner_pid`` - no process-tree
    introspection needed.
    """
    sentinels: list[dict] = []
    agents: list[dict] = []
    for event in events:
        if event.get("event") != "exec_python":
            continue
        if event.get("ppid") == runner_pid:
            sentinels.append(event)
        else:
            agents.append(event)
    return sentinels, agents


def _validate_three_layer_consistency(
    transcript_entries: list[dict],
    events: list[dict],
    runner_pid: int,
) -> None:
    """Three-layer cross-check: AST ↔ wrapper ↔ sitecustomize.

    Rules (PR #5):

    1. **Build-time sentinel demand**: the merged log MUST contain ≥1
       sentinel ``exec_python`` event (``ppid == runner_pid``). Zero
       sentinel events → ``RunValidityError``. Closes the shell-only-agent
       gap: even a run with no agent-invoked python passes attestation if
       the sentinel proves the wrapper was correctly wired before agent
       spawn.

    2. **Cardinality**: for each layer-1 invocation extracted by the AST
       parser, require AT LEAST ONE matching agent ``exec_python`` event
       AND AT LEAST ONE matching ``session_start`` event. (Agent events
       only; sentinel events are excluded from cardinality match.) Surplus
       runtime events are allowed (xargs/find/parallel spawn N children;
       ``subprocess.Popen`` from a parent python spawns surplus).

    3. **Match key**: ``argv[1:]`` equality. argv[0] differs by construction
       (wrapper records ``argv[0] = "python"`` resolved by PATH; layer-2's
       ``sys.orig_argv[0]`` is ``${venv}/bin/python-real`` after the
       wrapper's ``exec``). The load-bearing tokens for attribution are
       the script path + flags, not the interpreter.

    4. **Failure messages** quote the failing layer name so audits can
       grep by class.

    5. **Bypass priority**: callers run bypass detection BEFORE this check
       (see ``merge_layers`` orchestration). If layer-1 flags a bypass
       invocation, the merger fails-closed on the bypass first; this
       function does not get a chance to emit a "missing layer-1.5"
       duplicate error.
    """
    sentinels, agent_exec_events = _partition_exec_python_events(events, runner_pid)
    if not sentinels:
        raise RunValidityError(
            f"build-time sentinel missing: zero exec_python events with "
            f"ppid={runner_pid} found in event log. Either the wrapper was "
            f"not installed in the venv, the shim was not installed, or the "
            f"sentinel subprocess failed silently. The cold-start eval is "
            f"invalid."
        )

    sessions: list[dict] = [e for e in events if e.get("event") == "session_start"]
    sessions_by_pid: dict[int, list[dict]] = {}
    for s in sessions:
        pid = s.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            sessions_by_pid.setdefault(pid, []).append(s)

    all_exec_pids: set[int] = set()
    for evt in sentinels + agent_exec_events:
        pid = evt.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            all_exec_pids.add(pid)

    # Reciprocal check #1 (R0 P0 #2): every exec_python event - sentinel
    # and agent alike - MUST have a matching session_start by pid. The
    # wrapper exec's into the real python, which loads sitecustomize and
    # emits session_start with the SAME pid. A missing session_start
    # means sitecustomize failed to load (e.g., the agent invoked the
    # real python with ``-S`` which skips ``site.py`` import).
    for evt in sentinels + agent_exec_events:
        pid = evt.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        if pid not in sessions_by_pid:
            label = "sentinel" if evt in sentinels else "agent"
            raise RunValidityError(
                f"layer-2 attestation missing for {label} exec_python "
                f"pid={pid}: no session_start event with matching pid. "
                f"The wrapper recorded a python invocation but sitecustomize "
                f"did not load (likely a ``-S`` flag or otherwise neutered "
                f"site.py import). Cold-start eval is invalid."
            )

    # Reciprocal check #2 (R1 P0 #1): every session_start MUST have a
    # matching exec_python by pid. The wrapper fires BEFORE sitecustomize
    # (wrapper writes exec_python then exec's the real python which loads
    # sitecustomize and emits session_start), so a session_start without
    # a matching exec_python proves sitecustomize loaded on an interpreter
    # the wrapper did NOT see. The only way that happens is the agent
    # invoked the real interpreter directly (e.g.,
    # ``${venv}/.pyruntime-real/python-real script.py`` via filesystem
    # discovery, or a ``find -exec`` delegated form that found the real
    # binary). The merger fails closed.
    for s in sessions:
        pid = s.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        if pid not in all_exec_pids:
            raise RunValidityError(
                f"layer-1.5 attestation missing for session_start pid={pid}: "
                f"no exec_python event with matching pid. Sitecustomize "
                f"loaded but the layer-1.5 wrapper did NOT fire for this "
                f"interpreter invocation. Likely a direct exec of the real "
                f"interpreter that bypassed the wrapper. Cold-start eval "
                f"is invalid."
            )

    # Reciprocal check: any exec_python.argv containing ``-S`` (or compact
    # ``-Sc`` / ``-IS``) fails closed regardless of whether layer-1 saw the
    # invocation. Catches ``find -exec python -S script.py \\;`` and other
    # delegated forms the AST parser cannot model.
    for evt in agent_exec_events:
        argv = evt.get("argv")
        if isinstance(argv, list) and argv_contains_bypass_flag(argv[1:]):
            raise RunValidityError(
                f"layer-1.5 bypass detected: agent exec_python argv={argv!r} "
                f"contains ``-S`` (sitecustomize-disabling flag). The wrapper "
                f"recorded the invocation but sitecustomize would not have "
                f"loaded. Cold-start eval is invalid."
            )

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
            try:
                argvs = parse_python_invocations(command)
            except (ShellCommandIndeterminate, ShellCommandParseError):
                # Indeterminate or parse-failure commands are caught at the
                # bypass / shell-parser layer; not our problem here.
                continue
            for argv in argvs:
                if argv:
                    visible_invocations.append((command, argv))

    for command, visible_argv in visible_invocations:
        # argv[1:] match against agent exec_python events.
        wanted_argv_tail = visible_argv[1:]
        layer15_match = any(
            isinstance(e.get("argv"), list) and e["argv"][1:] == wanted_argv_tail
            for e in agent_exec_events
        )
        if not layer15_match:
            raise RunValidityError(
                f"layer-1.5 attestation missing for layer-1 invocation: "
                f"argv[1:]={wanted_argv_tail!r} in command {command[:160]!r}. "
                f"The AST parser saw a python invocation but no agent "
                f"exec_python event matched argv[1:]. Either the wrapper "
                f"was bypassed (e.g., direct exec of python-real) or the "
                f"event log was truncated."
            )
        # argv[1:] match against session_start events.
        layer2_match = any(
            isinstance(s.get("argv"), list) and s["argv"][1:] == wanted_argv_tail for s in sessions
        )
        if not layer2_match:
            raise RunValidityError(
                f"layer-2 attestation missing for layer-1 invocation: "
                f"argv[1:]={wanted_argv_tail!r} in command {command[:160]!r}. "
                f"The AST parser saw a python invocation but no session_start "
                f"event matched argv[1:]. Sitecustomize did not load for this "
                f"invocation."
            )


def _find_python_bypass_invocations_in_entries(entries: list[dict]) -> list[str]:
    """Return Bash commands that contain a Python invocation paired with
    any bypass primitive.

    Bypass primitives:
    - ``-S`` (or compact ``-Sc``) flag on the Python interpreter; disables
      site.py and skips sitecustomize.
    - ``PATH=...`` env-prefix on the Python CommandNode; redirects which
      interpreter resolves.
    - ``PYTHON*`` env-prefix (PYTHONHOME, PYTHONPATH); changes site /
      install root.
    - ``.`` / ``source`` activation script preceding a Python CommandNode
      in the same outer command; shell activation mutates env (often PATH).

    Per-invocation attribution alone cannot recover from these because
    the bypassed interpreter never writes ``session_start``; fail closed.

    Layer-1 analysis defers to ``harness.shell_parser`` which walks the
    bashlex AST. Indeterminate command-words (variable expansion, command
    substitution) propagate as ``ShellCommandIndeterminate``; bashlex
    parse failures propagate as ``ShellCommandParseError``. Both are
    fail-closed by design.

    Note: ``-I`` (isolated mode) is NOT a bypass; it implies ``-E``,
    ``-P``, and lowercase ``-s`` - none of which disable sitecustomize.
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
            if not isinstance(command, str) or not command:
                continue
            hits = find_python_bypass_invocations(command)
            if hits:
                bypass_commands.extend(hits)
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

    Counting uses ``parse_python_invocations`` (in
    ``harness.shell_parser``); the bashlex AST walker guards against
    false positives like ``/opt/python/`` (directory) or ``pythonic``
    (substring) at the language level rather than via regex word
    boundaries.
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
            if not isinstance(command, str) or not command:
                continue
            try:
                count += len(parse_python_invocations(command))
            except (ShellCommandIndeterminate, ShellCommandParseError):
                # _count is a coarse cross-check used by _validate_shim_loaded;
                # the actual fail-closed decisions on indeterminate / parse-
                # failed commands live in _attribute_python_invocations and
                # _validate_python_bash_results_non_error, which propagate
                # the exception. Here we just skip - we know the command
                # had SOMETHING undecidable; the validators above raise on
                # it independently.
                pass
    return count


def _validate_shim_loaded(
    events: list[dict],
    transcript_entries: list[dict],
    *,
    venv_path: Path | None = None,
) -> None:
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
    # R30 P0: every session_start must have a matching session_end -
    # including SURPLUS sessions (child Python processes invisible in
    # the transcript). pid is required by schema validation on both
    # session_start and session_end (R31 P0), so we can iterate
    # unconditionally without an optional-pid skip path that would
    # bypass the check.
    session_starts = [e for e in events if e.get("event") == "session_start"]
    session_end_pids = {e["pid"] for e in events if e.get("event") == "session_end"}
    for s in session_starts:
        pid = s["pid"]
        if pid not in session_end_pids:
            argv = s.get("argv", "<unknown>")
            raise TelemetryMergeError(
                f"session_start pid={pid!r} (argv={argv!r}) has no "
                f"matching session_end. The Python interpreter exited "
                f"via os._exit (typically a shim hard-exit on event-write "
                f"failure) or was killed by SIGKILL; atexit did not fire. "
                f"Layer-2 events from that process may be silently "
                f"incomplete."
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
    _attribute_python_invocations(transcript_entries, events, venv_path=venv_path)


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

    # Layer-1 evidence: Claude's Read/Grep tools accessing a bundled
    # guide file without invoking Python. Layer-2 hooks see nothing in
    # that path, so without this OR-merge the merger would silently
    # emit opened_llms_*=False for an actual discovery.
    read_tool_opens = _scan_read_tool_guide_accesses_in_entries(transcript_entries)
    for filename, was_opened in read_tool_opens.items():
        if was_opened:
            opened[filename] = True
    grep_tool_opens = _scan_grep_tool_guide_accesses_in_entries(transcript_entries)
    for filename, was_opened in grep_tool_opens.items():
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
    *,
    runner_pid: int | None = None,
    venv_path: Path | None = None,
) -> TelemetryRecord:
    """Merge the three telemetry layers into a single record.

    `arm` drives the sentinel semantics on guide-discovery fields (see
    TelemetryRecord docstring): for arm == "statsmodels", `opened_llms_*`
    and `called_get_llm_guide` are encoded as None ("not applicable").

    Raises ``TelemetryMergeError`` on any fail-closed condition. The full
    invariant matrix:

    **Cold-start integrity** (shim-side, asserted indirectly via merger):
    - The shim hard-exits if ``_PYRUNTIME_EVENT_LOG`` is unset or the
      event-log path is unopenable (visible as a Bash tool_result with
      ``is_error=True`` plus the ``[pyruntime]`` marker on stderr or in
      tool_result content; see the failure-marker class below).

    **Transcript completeness**:
    - Stream-JSON transcript is non-empty and ends with a terminal
      successful ``result`` entry (no mid-run truncation).
    - Every Bash ``tool_use`` has a non-empty string ``id`` and a
      matching ``tool_result`` (no truncation between request and
      response; stderr / subprocess content is recoverable).
    - Every Bash ``tool_result`` for a command containing a Python
      invocation has ``is_error=False`` (a non-zero exit can mask a
      stderr-suppressed shim hard-exit; see R23 P0#1).
    - Every guide-targeting Read ``tool_use`` has a non-empty string
      ``id`` and a matching ``tool_result`` (mirror of Bash check).
    - All ``tool_use`` IDs are unique across surfaces (would otherwise
      silently cross-match a tool_result between a Bash use and a Read
      use).
    - All ``tool_result.tool_use_id`` values are unique (mirror on the
      result side; otherwise a later ``is_error=False`` could mask an
      earlier ``is_error=True`` in dict-keyed validators; R24 P1#1).

    **Failure-marker propagation**:
    - No ``[pyruntime] cannot write event`` marker in
      ``cli_stderr.log`` (caught when stderr is not redirected).
    - No ``[pyruntime]`` marker in any Bash tool_result content (caught
      when subprocess stderr is captured into the tool result rather
      than redirected to /dev/null).
    - No Bash ``is_error=True`` for any Python invocation (the
      remaining defense when stderr is suppressed and the hard-exit
      can only be observed via subprocess exit code).

    **Argv attribution**:
    - Every transcript-visible Python invocation has a matching
      ``session_start`` event by exact argv (interpreter argv[0] +
      script args). Basename fallback on argv[0] is allowed ONLY for
      bare tokens (``python`` / ``python3`` / ``python3.x``); any
      path separator in argv[0] - absolute, leading-dot relative, or
      embedded - requires exact match (no per-arm-venv masking of
      off-venv launches; see R23 P0#2).

    **Bypass detection**:
    - No transcript-visible Python invocation uses the ``-S`` short
      flag (would skip ``site.py`` and prevent sitecustomize import).
    - No ``PATH=`` / ``env -u`` / ``./python`` style primitive bypass
      forms in the outer command.

    **Schema validation** (events.jsonl):
    - Every known event has the required fields with the correct
      types and (where applicable) valid enum values.

    **Data layer**:
    - In-process event log file is readable, well-formed JSONL, and
      contains no ``telemetry_missing`` runner sentinel.

    The transcript is parsed exactly once (per-invocation attribution,
    guide-read scan, completeness check, and bypass detection all
    consume the parsed entries).
    """
    transcript_entries = _validate_layer_artifacts(stream_json_path, stderr_path)
    _validate_bash_tool_results_complete(transcript_entries)
    _validate_tool_use_ids_unique(transcript_entries)
    _validate_tool_result_ids_unique(transcript_entries)
    _validate_python_bash_results_non_error(transcript_entries)
    if _scan_stderr_for_shim_failures(stderr_path):
        raise TelemetryMergeError(
            f"shim event-write failure marker present in {stderr_path}; "
            f"at least one layer-2 event was dropped mid-run and the "
            f"per-run record may be silently incomplete"
        )
    if _scan_tool_results_for_shim_failures(transcript_entries):
        raise TelemetryMergeError(
            "shim event-write failure marker present in a Bash tool_result "
            "(agent python subprocess stderr); at least one layer-2 event "
            "was dropped mid-run and the per-run record may be silently "
            "incomplete"
        )
    events = _read_events(in_process_events_path, venv_path=venv_path)
    _validate_shim_loaded(events, transcript_entries, venv_path=venv_path)
    # PR #5: three-layer cross-check fires only when the runner has
    # supplied its pid (production path). Direct-call test fixtures that
    # don't supply runner_pid stay in legacy mode (layer-1 + layer-2 only).
    if runner_pid is not None:
        _validate_three_layer_consistency(transcript_entries, events, runner_pid)
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
