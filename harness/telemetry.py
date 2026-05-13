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
    diff_diff calls are captured by layer 2 via the `showwarning` override,
    not here.

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
# cross-check. Matches `python`, `python3`, `python3.11`, etc. at a token
# boundary so compound shell commands (`pip install foo && python script.py`)
# are caught and false-positives like `/opt/python/` are not.
_PYTHON_INVOCATION_RE = re.compile(r"(?:^|[\s;&|()])python(?:3(?:\.\d+)?)?(?:\s|$)")


def _parse_jsonl_strict(path: Path, label: str) -> list[dict]:
    """Parse a JSONL file strictly; raise TelemetryMergeError on malformed lines.

    Shared parser used by both layer-2 event-log reads and layer-1 transcript
    reads. Empty lines are skipped; non-JSON content raises. The ``label``
    string identifies the layer/file in the error message.

    An empty (0-byte) file is permitted and returns ``[]``.
    """
    events: list[dict] = []
    with open(path) as f:
        for line_num, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise TelemetryMergeError(
                    f"malformed JSON in {label} {path} at line {line_num}: {e}"
                ) from e
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


def _validate_layer_artifacts(stream_json_path: Path, stderr_path: Path) -> None:
    """Fail-closed preflight: layer-1 and layer-3 capture files must exist
    AND the transcript must be well-formed JSONL.

    The three-layer telemetry contract requires the stream-JSON transcript
    (layer 1) and the CLI stderr capture (layer 3) to be present for every
    run; missing files mean the runner did not finish capturing or the
    output directory was tampered with. Either way, the per-run record is
    not safe to consume.

    Empty (0-byte) files are accepted — a trivial agent response can produce
    an empty stderr log, and a vacuous transcript is rare but legitimate.
    Non-JSON content in the transcript raises (the runner's contract is that
    every line of `transcript.jsonl` is a single JSON object).
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
    # Validate transcript is parseable JSONL. Discard the parsed entries
    # here; `_count_python_invocations` re-reads later (small redundant
    # cost; clearer than threading parsed state through the call chain).
    _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")


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


def _scan_read_tool_guide_accesses(stream_json_path: Path) -> dict[str, bool]:
    """Return ``{filename: True}`` for each bundled guide file accessed via
    Claude's Read tool in the transcript.

    Catches the case where an agent reads `llms.txt` (or any other bundled
    guide) via the Read tool without invoking Python; the in-process shim
    sees nothing in that path because no `open()` or `get_llm_guide` call
    runs in the agent's subprocess. Without this layer-1 check the merger
    would emit a definitive `opened_llms_txt=False` for what was actually
    a guide discovery — a silent telemetry-validity bug per the cold-start
    contract.

    Existence of the transcript file is the caller's responsibility
    (`_validate_layer_artifacts` runs first in `merge_layers`). Parsing
    uses the shared strict JSONL helper.

    Only Read-tool evidence is recognized here. Bash-level guide reads
    (e.g. ``cat llms.txt``) are still part of the broader layer-1 parsing
    deferred to a future PR.
    """
    entries = _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")
    opened: dict[str, bool] = {}
    known_filenames = set(_VARIANT_TO_FILENAME.values())
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Read":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            file_path = tool_input.get("file_path", "")
            if not isinstance(file_path, str):
                continue
            for known in known_filenames:
                if file_path.endswith(known):
                    opened[known] = True
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


def _count_python_invocations(stream_json_path: Path) -> int:
    """Return the number of Bash tool calls in the transcript whose command
    contains a python invocation.

    Parses each line of the stream-JSON transcript strictly: malformed JSONL
    raises ``TelemetryMergeError`` via the shared `_parse_jsonl_strict`
    helper, consistent with the layer-2 event-log parser. Existence of the
    file is the caller's responsibility (`_validate_layer_artifacts` runs
    first in `merge_layers`).

    The cross-check the result feeds is intentionally minimal — false
    negatives (we miss some invocations) are tolerable, but false positives
    (counting non-invocations) must not cause spurious shim-never-loaded
    errors. The word-boundary regex in `_PYTHON_INVOCATION_RE` and the
    structural `tool_use`/Bash/command match guard against both.
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
            if isinstance(command, str) and _PYTHON_INVOCATION_RE.search(command):
                count += 1
    return count


def _validate_shim_loaded(events: list[dict], stream_json_path: Path) -> None:
    """Cross-layer fail-closed check before building the record.

    Three cases:
    1. ``telemetry_missing`` sentinel present (written by the runner when
       the event log disappeared post-exec): raise.
    2. Zero ``session_start`` events AND transcript shows python
       invocations: shim never loaded inside the agent's Python subprocess
       — eval invalid; raise.
    3. Else accept (legitimate states: shim loaded + agent activity, OR
       shim loaded + no Python at all, OR no Python + no session_start).
    """
    for event in events:
        if event.get("event") == "telemetry_missing":
            raise TelemetryMergeError(
                "telemetry_missing sentinel present in event log; "
                "the agent's event log disappeared post-exec and the run "
                "is invalid for evaluation"
            )
    session_start_count = sum(1 for e in events if e.get("event") == "session_start")
    if session_start_count == 0:
        python_count = _count_python_invocations(stream_json_path)
        if python_count > 0:
            raise TelemetryMergeError(
                f"agent transcript shows {python_count} python invocation(s) "
                f"but in-process event log has no session_start event; "
                f"sitecustomize shim never loaded in the agent's subprocess. "
                f"Cold-start eval is invalid."
            )


def _build_diff_diff_record(
    events: list[dict],
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
            filename = r.get("filename", "")
            for known in opened:
                if filename.endswith(known):
                    opened[known] = True

    # Layer-1 evidence: Claude's Read tool accessing a bundled guide file
    # without invoking Python. Layer-2 hooks see nothing in that path, so
    # without this OR-merge the merger would silently emit
    # opened_llms_*=False for an actual discovery.
    read_tool_opens = _scan_read_tool_guide_accesses(stream_json_path)
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
    - Cross-layer inconsistency (python invoked but shim never loaded).
    """
    _validate_layer_artifacts(stream_json_path, stderr_path)
    if _scan_stderr_for_shim_failures(stderr_path):
        raise TelemetryMergeError(
            f"shim event-write failure marker present in {stderr_path}; "
            f"at least one layer-2 event was dropped mid-run and the "
            f"per-run record may be silently incomplete"
        )
    events = _read_events(in_process_events_path)
    _validate_shim_loaded(events, stream_json_path)
    if arm == "diff_diff":
        return _build_diff_diff_record(
            events, stream_json_path, in_process_events_path, stderr_path
        )
    elif arm == "statsmodels":
        return _build_statsmodels_record(
            events, stream_json_path, in_process_events_path, stderr_path
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")
