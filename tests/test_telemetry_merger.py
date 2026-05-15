"""Unit tests for `harness/telemetry.py::merge_layers` and helpers."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from harness.telemetry import (
    _VARIANT_TO_FILENAME,
    RunValidityError,
    TelemetryMergeError,
    TelemetryRecord,
    _count_python_invocations,
    merge_layers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events_jsonl(
    path: Path, events: list[dict], *, skip_auto_session_end: bool = False
) -> None:
    """Write a list of event dicts as one JSON object per line.

    Unless ``skip_auto_session_end=True``, auto-emits a matching
    ``session_end`` for every ``session_start.pid`` that doesn't already
    have one in the provided list. The merger requires session_end
    pairing for every session_start with a pid; most tests don't care
    about that check directly, so the helper keeps existing test setups
    working without per-test changes. Tests exercising the
    session_end-missing fail-closed path pass ``skip_auto_session_end=True``
    so the omission survives."""
    if skip_auto_session_end:
        augmented = events
    else:
        existing_end_pids = {
            e["pid"] for e in events if e.get("event") == "session_end" and "pid" in e
        }
        needs_end_pids = {
            e["pid"]
            for e in events
            if e.get("event") == "session_start"
            and "pid" in e
            and e["pid"] not in existing_end_pids
        }
        augmented = list(events)
        for pid in sorted(needs_end_pids):
            augmented.append(_session_end_event(pid=pid))
    with open(path, "w") as f:
        for event in augmented:
            f.write(json.dumps(event) + "\n")


def _write_transcript(path: Path, bash_commands: list[str]) -> None:
    """Write a minimal stream-JSON transcript with the given Bash commands.

    Each command becomes a tool_use block inside an assistant message
    entry, paired with a matching tool_result on a user-role entry. A
    terminal `result` entry is appended so the merger's completeness
    checks pass; tests that want to exercise truncation should overwrite
    the file without the result entry.
    """
    with open(path, "w") as f:
        for i, cmd in enumerate(bash_commands):
            tool_use_id = f"bash_{i}"
            f.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "id": tool_use_id,
                                    "input": {"command": cmd},
                                }
                            ],
                        },
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "is_error": False,
                                    "content": "",
                                }
                            ],
                        },
                    }
                )
                + "\n"
            )
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")


def _make_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (events_jsonl, transcript, stderr_log) paths under tmp_path.

    transcript is pre-populated with a minimal result entry so the merger's
    non-empty-transcript precondition is satisfied. Tests that want to
    exercise an empty transcript should overwrite the file explicitly.
    """
    events = tmp_path / "in_process_events.jsonl"
    transcript = tmp_path / "transcript.jsonl"
    stderr_log = tmp_path / "cli_stderr.log"
    transcript.write_text('{"type": "result", "subtype": "success"}\n')
    stderr_log.touch()
    return events, transcript, stderr_log


_DEFAULT_PID = 11111


def _session_start_event(
    sys_executable: str | None = None,
    argv: list | None = None,
    pid: int | None = None,
) -> dict:
    """Build a session_start event. ``argv`` defaults to a placeholder so
    schema validation passes; tests that need attribution against a
    specific python invocation should pass an explicit ``argv``.

    ``pid`` defaults to ``_DEFAULT_PID`` so the merger's session_end
    pairing check has a key to match against (the session_end helper
    uses the same default). Tests exercising session_end pairing
    failures should pass an explicit pid and omit the matching
    session_end."""
    event: dict = {
        "event": "session_start",
        "ts": "2026-05-12T00:00:00.000000+00:00",
        "argv": argv if argv is not None else ["python", "-c", "pass"],
        "pid": pid if pid is not None else _DEFAULT_PID,
    }
    if sys_executable is not None:
        event["sys_executable"] = sys_executable
    return event


def _session_end_event(pid: int | None = None) -> dict:
    """Build a session_end event paired with a session_start by pid."""
    return {
        "event": "session_end",
        "ts": "2026-05-12T00:00:01.000000+00:00",
        "pid": pid if pid is not None else _DEFAULT_PID,
    }


# PR #5: layer-1.5 exec_python event fixtures.
#
# ``merge_layers`` now accepts optional ``runner_pid`` + ``venv_path``
# kwargs. When BOTH are provided (production runner path) the merger
# enforces the three-layer cross-check: every layer-1 invocation must
# match at least one agent exec_python event by ``argv[1:]`` AND at least
# one ``session_start`` event by ``argv[1:]``; the log must also contain
# at least one sentinel exec_python event (``ppid == runner_pid``). When
# ``runner_pid`` is None (legacy fixture mode) the three-layer check is
# skipped; the existing 138 fixtures stay valid unchanged.

_DEFAULT_RUNNER_PID = 99999


def _exec_python_event(
    pid: int,
    argv: list[str],
    *,
    ppid: int = _DEFAULT_RUNNER_PID,
    ts: str = "2026-05-12T00:00:00Z",
    executable: str | None = None,
) -> dict:
    """Build a layer-1.5 ``exec_python`` event.

    Default ``ppid`` matches the default sentinel ppid so a hand-built
    event without an explicit ppid reads as a sentinel. Tests that need
    an *agent* exec_python event pass an explicit ``ppid`` distinct from
    the runner pid they pass to ``merge_layers``.
    """
    return {
        "event": "exec_python",
        "pid": pid,
        "ppid": ppid,
        "ts": ts,
        "executable": (
            executable if executable is not None else "/tmp/venv/.pyruntime-real/python-real"
        ),
        "argv": argv,
    }


_SENTINEL_PID = _DEFAULT_RUNNER_PID + 1


def _sentinel_exec_python_event(
    runner_pid: int = _DEFAULT_RUNNER_PID,
    *,
    executable: str | None = None,
) -> dict:
    """Build a sentinel exec_python event (``ppid == runner_pid``).

    Mirrors what ``run_one``'s build-time sentinel produces. Tests that
    pass ``runner_pid`` to ``merge_layers`` need at least one of these in
    the event log to pass the sentinel-demand rule.
    """
    return _exec_python_event(
        pid=_SENTINEL_PID,
        argv=["python", "-c", "pass"],
        ppid=runner_pid,
        executable=executable,
    )


def _sentinel_events(
    runner_pid: int = _DEFAULT_RUNNER_PID,
    *,
    executable: str | None = None,
) -> list[dict]:
    """Build the full sentinel event triple (exec_python + session_start
    + session_end) with matching pid.

    PR #5 reciprocal check: every exec_python event (including the
    sentinel) must have a matching session_start by pid. Tests that
    construct a sentinel need all three to pass merger validation.
    """
    return [
        _sentinel_exec_python_event(runner_pid=runner_pid, executable=executable),
        _session_start_event(argv=["python", "-c", "pass"], pid=_SENTINEL_PID),
    ]


def _write_transcript_entries(path: Path, entries: list[dict]) -> None:
    """Write a list of stream-JSON entries to `path`, appending a terminal
    `result` entry so the merger's completeness check passes. Tests
    exercising truncation should use a direct open() instead."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")


# ---------------------------------------------------------------------------
# diff_diff arm: happy paths and structure
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_empty_events_with_no_python_invocations(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False
    assert record.opened_llms_autonomous is False
    assert record.opened_llms_full is False
    assert record.called_get_llm_guide is False
    assert record.get_llm_guide_variants == ()
    assert record.saw_fit_time_warning is False
    assert record.diagnostic_methods_invoked == ()
    assert record.estimator_classes_instantiated == ()


def test_merge_layers_diff_diff_zero_byte_events_jsonl(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    events_path.touch()  # 0-byte file; no events, no session_start
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    # No python invocations in transcript + zero events = vacuous agent;
    # merger does NOT raise.
    assert record.arm == "diff_diff"
    assert record.opened_llms_txt is False
    assert record.called_get_llm_guide is False


def test_merge_layers_diff_diff_full_events(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # Constants hardcoded here to avoid importing harness.sitecustomize_template
    # (which mutates global state at top-level). The bidirectional regression
    # test in test_sitecustomize.py is the source of truth for these lists
    # against diff_diff exports.
    _ESTIMATOR_CLASS_NAMES = (
        "DifferenceInDifferences",
        "TwoWayFixedEffects",
        "MultiPeriodDiD",
        "SyntheticDiD",
        "CallawaySantAnna",
        "ChaisemartinDHaultfoeuille",
        "ContinuousDiD",
        "SunAbraham",
        "ImputationDiD",
        "TwoStageDiD",
        "TripleDifference",
        "TROP",
        "StackedDiD",
        "StaggeredTripleDifference",
        "EfficientDiD",
        "WooldridgeDiD",
        "BaconDecomposition",
        "HeterogeneousAdoptionDiD",
        "HonestDiD",
        "PreTrendsPower",
        "LinearRegression",
    )
    _DIAGNOSTIC_FUNCTION_NAMES = (
        "compute_pretrends_power",
        "compute_honest_did",
        "bacon_decompose",
        "run_placebo_test",
        "compute_power",
    )

    events: list[dict] = [_session_start_event()]
    for cls in _ESTIMATOR_CLASS_NAMES:
        events.append({"event": "estimator_init", "class": cls, "ts": "x"})
        events.append({"event": "estimator_fit", "class": cls, "ts": "x"})
    for func in _DIAGNOSTIC_FUNCTION_NAMES:
        events.append({"event": "diagnostic_call", "name": func, "ts": "x"})
    for variant in _VARIANT_TO_FILENAME:
        events.append(
            {
                "event": "guide_file_read",
                "via": "get_llm_guide",
                "variant": variant,
                "ts": "x",
            }
        )
    events.append(
        {
            "event": "warning_emitted",
            "category": "UserWarning",
            "filename": "/path/to/diff_diff/estimators.py",
            "lineno": 1,
            "message": "test warning",
            "ts": "x",
        }
    )
    _write_events_jsonl(events_path, events)
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    # Successful construction implicitly verifies __post_init__ accepts
    # the record (arm="diff_diff" requires every sentinel field to be bool).
    assert record.opened_llms_txt is True
    assert record.opened_llms_practitioner is True
    assert record.opened_llms_autonomous is True
    assert record.opened_llms_full is True
    assert record.called_get_llm_guide is True
    assert record.get_llm_guide_variants == tuple(sorted(_VARIANT_TO_FILENAME))
    assert record.saw_fit_time_warning is True
    assert record.diagnostic_methods_invoked == tuple(sorted(_DIAGNOSTIC_FUNCTION_NAMES))
    assert record.estimator_classes_instantiated == tuple(sorted(_ESTIMATOR_CLASS_NAMES))


# ---------------------------------------------------------------------------
# diff_diff arm: fail-closed cross-checks
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_python_invocation_without_session_start(
    tmp_path,
):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # NO session_start
    _write_transcript(transcript, ["python -c 'print(1)'"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_telemetry_missing_sentinel(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            {
                "event": "telemetry_missing",
                "fatal": True,
                "note": "agent_event_log_path did not exist post-exec",
            }
        ],
    )
    with pytest.raises(TelemetryMergeError, match="telemetry_missing sentinel"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_raises_on_run_invalid_descendants_live_sentinel(tmp_path):
    """R17 P1 (EV-1): runner-written ``run_invalid`` sentinel (with
    reason=``descendants_live`` from the R16 P1 fix) must make the run
    unmergeable. Without this, a downstream caller invoking
    ``merge_layers`` directly on the artifact set (without inspecting
    RunResult.exit_code or cli_stderr.log) could produce a clean
    ``TelemetryRecord`` for a run the runner explicitly marked invalid.
    Mirror of the telemetry_missing rejection above.
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            # A normal session_start to demonstrate the run_invalid
            # rejection fires REGARDLESS of other telemetry presence.
            _session_start_event(),
            {
                "event": "run_invalid",
                "fatal": True,
                "reason": "descendants_live",
                "note": "agent process group had surviving children after main process exited",
            },
        ],
    )
    with pytest.raises(TelemetryMergeError, match="run_invalid sentinel.*descendants_live"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_raises_on_run_invalid_with_arbitrary_reason(tmp_path):
    """R17 P1 (EV-1): the run_invalid rejection is reason-agnostic;
    any future invariant the runner adds (e.g., a new post-spawn
    quiescence check) gets rejection for free as long as it writes
    the run_invalid sentinel.
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            {
                "event": "run_invalid",
                "fatal": True,
                "reason": "future_post_spawn_invariant_X",
                "note": "hypothetical future invariant violation",
            },
        ],
    )
    with pytest.raises(
        TelemetryMergeError,
        match="run_invalid sentinel.*future_post_spawn_invariant_X",
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_malformed_jsonl(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    events_path.write_text("{not valid json\n")
    with pytest.raises(TelemetryMergeError, match="malformed JSON"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_missing_event_log_file(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # Don't create events_path — _make_paths sets the path but only touches
    # transcript and stderr.
    assert not events_path.exists()
    with pytest.raises(TelemetryMergeError, match="not found"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_missing_transcript(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.unlink()  # remove layer-1 capture
    with pytest.raises(TelemetryMergeError, match="stream-JSON transcript missing"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_malformed_transcript(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text("{not valid json\n")
    with pytest.raises(TelemetryMergeError, match="malformed JSON.*stream-JSON"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_missing_stderr(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    stderr_log.unlink()  # remove layer-3 capture
    with pytest.raises(TelemetryMergeError, match="stderr capture missing"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_empty_transcript(tmp_path):
    """R3 P0: empty stream-JSON transcripts indicate stdout-capture failure
    and would silently zero out layer-1 evidence; must fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text("")  # explicitly empty
    with pytest.raises(TelemetryMergeError, match="transcript.*is empty"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_non_object_transcript(tmp_path):
    """R3 P0: every transcript line must be a JSON object; scalars/lists
    indicate a corrupted capture."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text('"just a string"\n')
    with pytest.raises(TelemetryMergeError, match="non-object JSON"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_accepts_empty_stderr_only(tmp_path):
    """Empty stderr is fine (no CLI-level errors). The transcript must be
    non-empty though, so provide a minimal entry."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text('{"type": "result", "subtype": "success"}\n')
    # stderr_log is 0-byte from _make_paths; that's fine.
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_raises_on_shim_write_failure_marker(tmp_path):
    """Layer-3 cross-check: stderr containing the shim's event-write failure
    marker means at least one layer-2 event was dropped mid-run; the per-run
    record cannot be treated as complete."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    stderr_log.write_text(
        "[pyruntime] cannot write event to /tmp/somewhere/events.jsonl: "
        "[Errno 28] No space left on device\n"
    )
    with pytest.raises(TelemetryMergeError, match="shim event-write failure marker"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def _read_tool_request(tool_use_id: str, file_path: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "Read",
                    "input": {"file_path": file_path},
                }
            ],
        },
    }


def _read_tool_result(tool_use_id: str, *, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": is_error,
                    "content": "file contents" if not is_error else "error",
                }
            ],
        },
    }


def test_merge_layers_diff_diff_read_tool_guide_access_flips_opened_flag(tmp_path):
    """Layer-1 evidence: Claude's Read tool on a bundled guide file (with
    a non-error tool_result) populates `opened_llms_*`."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("read_1", "/some/install/diff_diff/guides/llms.txt"),
            _read_tool_result("read_1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    # Read-tool-only access should NOT flip called_get_llm_guide.
    assert record.called_get_llm_guide is False
    assert record.get_llm_guide_variants == ()


def test_merge_layers_diff_diff_read_tool_recognizes_all_guide_filenames(tmp_path):
    """Verify all four bundled guides are detected when each has a
    successful tool_result."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript_entries = []
    for i, name in enumerate(
        ("llms.txt", "llms-practitioner.txt", "llms-autonomous.txt", "llms-full.txt"),
        start=1,
    ):
        tool_use_id = f"read_{i}"
        transcript_entries.append(
            _read_tool_request(tool_use_id, f"/install/diff_diff/guides/{name}")
        )
        transcript_entries.append(_read_tool_result(tool_use_id))
    _write_transcript_entries(transcript, transcript_entries)
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    assert record.opened_llms_practitioner is True
    assert record.opened_llms_autonomous is True
    assert record.opened_llms_full is True


def test_merge_layers_diff_diff_read_tool_failed_read_does_not_flip_flag(tmp_path):
    """R8 P1: a Read with is_error=True must NOT flip opened_llms_*."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("r1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("r1", is_error=True),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False


def test_merge_layers_diff_diff_read_tool_missing_result_raises(tmp_path):
    """R10 P0: a guide-file Read tool_use with no matching tool_result
    fails closed. Transcript may be truncated; emitting definitive False
    would be silent layer-1 loss, identical in spirit to the empty-
    transcript / missing-terminal-result check."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("r1", "/install/diff_diff/guides/llms.txt"),
            # no tool_result entry
        ],
    )
    with pytest.raises(TelemetryMergeError, match="no matching tool_result"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_read_tool_guide_missing_id_raises(tmp_path):
    """R22 P0: a guide-file Read tool_use without an ``id`` field fails
    closed. Without an id the merger cannot match a tool_result; the read
    might have succeeded but no evidence connects to a Read evidence flag.
    Reciprocal of R21's Bash-tool_use missing-id check; same invariant,
    different surface."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            # no "id" field
                            "input": {"file_path": "/install/diff_diff/guides/llms.txt"},
                        }
                    ],
                },
            }
        ],
    )
    with pytest.raises(TelemetryMergeError, match="missing or has empty 'id'"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_read_tool_guide_empty_id_raises(tmp_path):
    """R22 P0: empty-string id is treated the same as missing id."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "id": "",
                            "input": {"file_path": "/install/diff_diff/guides/llms.txt"},
                        }
                    ],
                },
            }
        ],
    )
    with pytest.raises(TelemetryMergeError, match="missing or has empty 'id'"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_duplicate_tool_use_id_across_bash_read_raises(tmp_path):
    """R22 P0: a tool_use_id reused between a Bash tool_use and a Read
    tool_use silently cross-matches in the per-surface dicts, so a missing
    tool_result on one block could be falsely "covered" by another
    block's tool_result. Fail closed at merge time."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "id": "dup_1",
                            "input": {"command": "echo hello"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "dup_1",
                            "is_error": False,
                            "content": "hello",
                        }
                    ],
                },
            },
            _read_tool_request("dup_1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("dup_1"),
        ],
    )
    with pytest.raises(TelemetryMergeError, match="appears on two tool_use blocks"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_missing_session_end_raises(tmp_path):
    """R29 P0: a session_start with a pid that has no matching session_end
    in the event log signals that the shim hard-exited (os._exit) and
    skipped its atexit handler. The agent can mask the Bash exit code
    with ``2>/dev/null || true``, so neither stderr markers nor
    is_error catch the failure; session_end is the unmaskable signal.
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # session_start with pid, but NO matching session_end - simulates a
    # shim hard-exit that skipped atexit.
    session_start = _session_start_event(argv=["python", "script.py"], pid=12345)
    _write_events_jsonl(events_path, [session_start], skip_auto_session_end=True)
    _write_transcript(transcript, ["python script.py 2>/dev/null || true"])
    with pytest.raises(TelemetryMergeError, match="no matching session_end"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def _grep_tool_request(tool_use_id: str, path: str, pattern: str = "foo") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "Grep",
                    "input": {"path": path, "pattern": pattern},
                }
            ],
        },
    }


def test_merge_layers_diff_diff_grep_tool_guide_access_flips_flag(tmp_path):
    """R32 P0: a successful Grep against a bundled guide file flips
    ``opened_llms_*`` the same way Read does. Pre-R32 the merger only
    scanned Read tool_uses; a Grep on llms.txt left the flag False
    despite the agent seeing guide content."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_tool_request("g1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True


def test_merge_layers_diff_diff_grep_tool_guides_dir_no_glob_fails_closed(tmp_path):
    """R33 P0: a Grep against the guides directory itself with NO glob
    is ambiguous - we cannot statically attribute which guide files
    were searched. Fail closed rather than flag-all-four (the previous
    R32 behavior was unsafe because the agent may have used the
    output_mode=files_with_matches without seeing any content)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_tool_request("g1", "/install/diff_diff/guides"),
            _read_tool_result("g1"),
        ],
    )
    with pytest.raises(
        TelemetryMergeError,
        match="cannot be attributed to specific guide files",
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def _grep_request_with_glob(tool_use_id: str, path: str, glob: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "Grep",
                    "input": {"path": path, "pattern": "foo", "glob": glob},
                }
            ],
        },
    }


def test_merge_layers_diff_diff_grep_tool_glob_specific_file_flips_one(tmp_path):
    """R33 P0: Grep with ``path=diff_diff, glob=guides/llms.txt`` is
    attributable to llms.txt only."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/install/diff_diff", "guides/llms.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    assert record.opened_llms_practitioner is False
    assert record.opened_llms_autonomous is False
    assert record.opened_llms_full is False


def test_merge_layers_diff_diff_grep_tool_glob_pattern_flips_matching(tmp_path):
    """R33 P0: Grep with ``glob=guides/llms*.txt`` matches all four
    bundled guides by fnmatch; each is flagged."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/install/diff_diff", "guides/llms*.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    assert record.opened_llms_practitioner is True
    assert record.opened_llms_autonomous is True
    assert record.opened_llms_full is True


def test_merge_layers_diff_diff_grep_tool_unscoped_path_no_attribution(tmp_path):
    """R34 P1: glob attribution requires the path to be scoped under
    bundled diff_diff. ``path=/tmp + glob=guides/llms.txt`` must NOT
    flag llms.txt - the search is at /tmp, not under bundled guides."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/tmp", "guides/llms.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False


def test_merge_layers_diff_diff_grep_tool_recursive_glob_scoped_attributes(tmp_path):
    """R34 P1: recursive glob forms under scoped path are attributed.
    ``path=/install/diff_diff + glob=**/guides/llms.txt`` resolves to
    llms.txt under the bundled guides location."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/install/diff_diff", "**/guides/llms.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True


def test_merge_layers_diff_diff_grep_tool_recursive_at_guides_dir(tmp_path):
    """R34 P1: when path IS the guides directory, glob can be a basename
    pattern directly or a recursive ``**/<basename>`` form."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/install/diff_diff/guides", "**/llms-practitioner.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_practitioner is True


def test_merge_layers_diff_diff_grep_tool_glob_non_guide_not_flipped(tmp_path):
    """R33 P0 negative: a glob that's guide-shaped but doesn't match
    any bundled file leaves all flags False."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_request_with_glob("g1", "/install/diff_diff", "guides/llms-future.txt"),
            _read_tool_result("g1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False


def test_merge_layers_diff_diff_grep_tool_failed_does_not_flip(tmp_path):
    """R32 P0: a failed Grep (is_error=True) does NOT flip the flag.
    Mirrors the Read scanner's failure semantics."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _grep_tool_request("g1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("g1", is_error=True),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False


def test_merge_layers_diff_diff_grep_tool_missing_result_raises(tmp_path):
    """R32 P0: a Grep tool_use for a bundled guide path with no matching
    tool_result fails closed (same as Read)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [_grep_tool_request("g1", "/install/diff_diff/guides/llms.txt")],
    )
    with pytest.raises(TelemetryMergeError, match="no matching tool_result"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_pidless_session_start_raises(tmp_path):
    """R31 P0: a session_start record missing the required ``pid`` field
    is rejected at schema validation, closing the pid-less bypass that
    would otherwise skip the session_end pairing check."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    pidless_start = {
        "event": "session_start",
        "ts": "2026-05-12T00:00:00.000000+00:00",
        "argv": ["python", "script.py"],
    }
    _write_events_jsonl(events_path, [pidless_start], skip_auto_session_end=True)
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(TelemetryMergeError, match="missing required field 'pid'"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_pidless_surplus_session_start_raises(tmp_path):
    """R31 P0: rejection applies to SURPLUS pid-less session_start
    events too (not transcript-visible). Schema validation runs before
    attribution, so visible/surplus distinction doesn't matter."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    parent_start = _session_start_event(argv=["python", "parent.py"], pid=11111)
    parent_end = _session_end_event(pid=11111)
    pidless_child = {
        "event": "session_start",
        "ts": "2026-05-12T00:00:00.000000+00:00",
        "argv": ["python", "child.py"],
    }
    _write_events_jsonl(
        events_path,
        [parent_start, parent_end, pidless_child],
        skip_auto_session_end=True,
    )
    _write_transcript(transcript, ["python parent.py"])
    with pytest.raises(TelemetryMergeError, match="missing required field 'pid'"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_non_int_session_pid_raises(tmp_path):
    """R31 P0: pid type-check. A string pid (or any non-int) is rejected.
    bool is rejected explicitly even though it subclasses int."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    bad_start = {
        "event": "session_start",
        "ts": "2026-05-12T00:00:00.000000+00:00",
        "argv": ["python", "script.py"],
        "pid": "12345",
    }
    _write_events_jsonl(events_path, [bad_start], skip_auto_session_end=True)
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(TelemetryMergeError, match="pid must be int"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_surplus_child_missing_session_end_raises(tmp_path):
    """R30 P0 #1: a child Python process invisible in the transcript can
    emit session_start, drop a layer-2 event via shim hard-exit, and
    skip session_end. Pre-R30 the merger's session_end check fired only
    on attributed (transcript-visible) sessions; surplus sessions
    bypassed the check.

    Now every session_start with a pid must have a matching session_end,
    including child processes the parent script spawned but the
    transcript never sees as a Bash command."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # Parent session: visible, properly paired with session_end.
    parent_start = _session_start_event(argv=["python", "parent.py"], pid=11111)
    parent_end = _session_end_event(pid=11111)
    # Child session: NOT visible in transcript, NO session_end -
    # simulates a child Python hard-exit.
    child_start = _session_start_event(argv=["python", "child.py"], pid=22222)
    _write_events_jsonl(
        events_path,
        [parent_start, parent_end, child_start],
        skip_auto_session_end=True,
    )
    _write_transcript(transcript, ["python parent.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_end"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_session_end_pairing_attributes(tmp_path):
    """R29 positive case: when session_start.pid has a matching
    session_end, the run merges cleanly. Ensures the new check doesn't
    over-reject."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    session_start = _session_start_event(argv=["python", "script.py"])
    session_start["pid"] = 12345
    session_end = {
        "event": "session_end",
        "ts": "2026-05-12T00:00:01.000000+00:00",
        "pid": 12345,
    }
    _write_events_jsonl(events_path, [session_start, session_end])
    _write_transcript(transcript, ["python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_python_bash_is_error_raises(tmp_path):
    """R23 P0#1: a Bash tool_result with ``is_error=True`` for a command
    containing a Python invocation is rejected at merge time.

    Catches the remaining hard-exit observability hole from R22: the shim
    hard-exits with ``os._exit(2)`` on an event-write failure, but if the
    agent runs ``python script.py 2>/dev/null`` the stderr marker is
    suppressed and never reaches ``cli_stderr.log`` or the Bash
    tool_result content. The subprocess exit code still propagates as
    ``tool_result.is_error=True``; this validator closes the chain by
    treating any non-zero exit on a tracked Python invocation as
    telemetry-completeness failure."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    # Write a transcript where the Python Bash tool_result has is_error=True.
    with open(transcript, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "bash_1",
                                "input": {"command": "python script.py 2>/dev/null"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bash_1",
                                "is_error": True,
                                "content": "",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")
    with pytest.raises(TelemetryMergeError, match="is_error=True"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_non_python_bash_is_error_passes(tmp_path):
    """R23: ``is_error=True`` on a non-Python Bash command (``ls``,
    ``cat``, etc.) is NOT a telemetry-completeness failure - the agent
    can run anything and have it fail; only Python invocations carry the
    hard-exit-observability invariant.

    Verifies the validator scopes correctly to Python-invocation
    commands; a failing ``ls /nonexistent`` doesn't break the merge."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    with open(transcript, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "bash_1",
                                "input": {"command": "ls /nonexistent"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bash_1",
                                "is_error": True,
                                "content": "ls: cannot access '/nonexistent'",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")
    # Should NOT raise: ls failure has no telemetry-completeness implication.
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_duplicate_tool_result_id_masks_is_error_raises(tmp_path):
    """R24 P1#1: a transcript with two tool_result blocks sharing the
    same tool_use_id (first is_error=True, second is_error=False) must
    fail closed. Otherwise the dict-keyed lookup in
    _validate_python_bash_results_non_error sees only the later
    (False) result and the hard-exit observability is_error chain
    closes silently.

    Reciprocal of R22's _validate_tool_use_ids_unique check; same
    invariant, applied to the tool_result side."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event(argv=["python", "script.py"])])
    with open(transcript, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "bash_1",
                                "input": {"command": "python script.py 2>/dev/null"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # First result: error (would normally trigger fail-closed).
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bash_1",
                                "is_error": True,
                                "content": "",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Second result with same id: masks the first in any dict-keyed
        # validator. The merger must reject the duplicate id rather than
        # silently overwrite.
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bash_1",
                                "is_error": False,
                                "content": "fake recovery",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")
    with pytest.raises(TelemetryMergeError, match="two tool_result blocks"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_duplicate_tool_result_id_for_guide_read_raises(tmp_path):
    """R24 P1#1: same uniqueness check applies to Read tool_results.
    Two results for the same guide-Read tool_use_id (first is_error=True
    masking layer-1 evidence, second False inserting fake evidence)
    fails closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("r1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("r1", is_error=True),
            _read_tool_result("r1", is_error=False),
        ],
    )
    with pytest.raises(TelemetryMergeError, match="two tool_result blocks"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


@pytest.mark.parametrize(
    "guide_path",
    [
        "/install/diff_diff/guides/../guides/llms.txt",
        "/install/diff_diff/guides/./llms.txt",
        "/install/./diff_diff/guides/llms.txt",
        "/install//diff_diff/guides/llms.txt",
        "/install/diff_diff/foo/../guides/llms.txt",
    ],
)
def test_merge_layers_diff_diff_read_tool_normalizes_guide_path(tmp_path, guide_path):
    """R24 P1#2: Read tool paths with ``.`` / ``..`` / duplicate separators
    that lex-normalize to a bundled guide location must be counted as
    successful guide discoveries. Pre-R24 the raw
    ``Path(file_path).parent.name`` segment check missed these forms.

    All five variants normalize to ``/install/diff_diff/guides/llms.txt``;
    the merger flips ``opened_llms_txt=True`` after lex-normalizing via
    ``os.path.normpath``. ``Path.resolve()`` would require filesystem
    access (the path is from the agent's environment, not ours), so a
    purely lexical normalization is used instead."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("r1", guide_path),
            _read_tool_result("r1"),
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True


def test_merge_layers_diff_diff_python_bash_compound_command_is_error_raises(tmp_path):
    """R23: a compound command like ``pip install foo && python script.py``
    with ``is_error=True`` triggers the validator. The bash command
    contains a Python invocation; we cannot tell whether pip or python
    failed, so the per-run record cannot be trusted."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event(argv=["python", "script.py"])])
    with open(transcript, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "bash_1",
                                "input": {"command": "pip install foo && python script.py"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bash_1",
                                "is_error": True,
                                "content": "pip install failed",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")
    with pytest.raises(TelemetryMergeError, match="is_error=True"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_duplicate_tool_use_id_within_read_raises(tmp_path):
    """R22 P0: two guide-Read tool_uses sharing the same id are rejected
    by the per-surface uniqueness check (raises before the cross-surface
    check)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            _read_tool_request("r1", "/install/diff_diff/guides/llms.txt"),
            _read_tool_result("r1"),
            _read_tool_request("r1", "/install/diff_diff/guides/llms-practitioner.txt"),
            _read_tool_result("r1"),
        ],
    )
    # Either the per-surface "reuses tool_use_id" or the cross-surface
    # "appears on two tool_use blocks" check may catch this depending on
    # ordering; both are valid fail-closed outcomes.
    with pytest.raises(
        TelemetryMergeError,
        match="(reuses tool_use_id|appears on two tool_use blocks)",
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_read_tool_ignores_non_guide_files(tmp_path):
    """Read tool on a non-guide path should not flip any flag."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/some/path/data.csv"},
                        }
                    ],
                },
            }
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False
    assert record.opened_llms_autonomous is False
    assert record.opened_llms_full is False


# ---------------------------------------------------------------------------
# statsmodels arm
# ---------------------------------------------------------------------------


def test_merge_layers_statsmodels_returns_sentinel_record(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    record = merge_layers("statsmodels", transcript, events_path, stderr_log)
    assert record.arm == "statsmodels"
    assert record.opened_llms_txt is None
    assert record.opened_llms_practitioner is None
    assert record.opened_llms_autonomous is None
    assert record.opened_llms_full is None
    assert record.called_get_llm_guide is None
    assert record.get_llm_guide_variants == ()
    assert record.saw_fit_time_warning is False
    assert record.diagnostic_methods_invoked == ()
    assert record.estimator_classes_instantiated == ()


def test_merge_layers_statsmodels_empty_events_passes_validation(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    events_path.touch()  # 0-byte
    # No python invocations + no session_start = legitimate vacuous state
    record = merge_layers("statsmodels", transcript, events_path, stderr_log)
    assert record.arm == "statsmodels"


# ---------------------------------------------------------------------------
# Discoverability field rules
# ---------------------------------------------------------------------------


def test_merge_layers_get_llm_guide_variants_deduplicated_and_sorted(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(),
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": "concise"},
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": "concise"},
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": "concise"},
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": "practitioner"},
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.get_llm_guide_variants == ("concise", "practitioner")


def test_merge_layers_open_guide_read_sets_correct_flag(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(),
            {
                "event": "guide_file_read",
                "via": "open",
                "filename": "llms-practitioner.txt",
            },
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_practitioner is True
    assert record.opened_llms_txt is False
    # `called_get_llm_guide` reflects ONLY get_llm_guide-via events; open-via
    # reads don't flip it.
    assert record.called_get_llm_guide is False


def test_merge_layers_estimator_init_or_fit_both_contribute(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(),
            {"event": "estimator_init", "class": "TwoWayFixedEffects"},
            {"event": "estimator_fit", "class": "SyntheticDiD"},
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.estimator_classes_instantiated == (
        "SyntheticDiD",
        "TwoWayFixedEffects",
    )


# ---------------------------------------------------------------------------
# Cross-package contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_variant_to_filename_mapping_matches_diff_diff_variant_to_file():
    from diff_diff._guides_api import _VARIANT_TO_FILE

    assert _VARIANT_TO_FILENAME == _VARIANT_TO_FILE


# ---------------------------------------------------------------------------
# Python-invocation detection (cross-layer heuristic)
# ---------------------------------------------------------------------------


def test_python_invocation_detection_recognizes_python_python3(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        ["python script.py", "python3 -c 'import diff_diff'", "python3.11 script.py"],
    )
    assert _count_python_invocations(transcript) == 3


def test_python_invocation_detection_ignores_paths_containing_python(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, ["ls /opt/python/", "echo 'pythonic'"])
    # Neither command actually invokes the python binary; should not trigger.
    assert _count_python_invocations(transcript) == 0


def test_python_invocation_detection_recognizes_compound_commands(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            "pip install foo && python script.py",
            "cd /tmp && python -c 'pass'",
            "python script.py | grep result",
        ],
    )
    assert _count_python_invocations(transcript) == 3


def test_python_invocation_detection_recognizes_absolute_paths(tmp_path):
    """R1 P0: `/usr/bin/python3 script.py` must trigger detection."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            "/usr/bin/python3 script.py",
            "/opt/venv/bin/python -c 'import diff_diff'",
            "/Users/me/.venv/bin/python3.11 script.py",
        ],
    )
    assert _count_python_invocations(transcript) == 3


def test_python_invocation_detection_ignores_python_in_directory_names(tmp_path):
    """`/opt/python/` is a directory, not an invocation; must not trigger."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            "ls /opt/python/",
            "cat /home/python_project/README.md",
            "cd /Users/me/python_tools && ls",
        ],
    )
    assert _count_python_invocations(transcript) == 0


def test_merge_layers_diff_diff_raises_on_partial_instrumentation(tmp_path):
    """R1 P0: an uninstrumented invocation (whether by `python -S` or by an
    absolute-path interpreter outside the per-arm venv) must fail closed.
    Updated post-R3: the bypass-flag detector catches `-S` directly with a
    different error message; the partial-instrumentation count check
    remains for absolute-path-without-shim cases."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["python -c 'import diff_diff'", "python -S uninstrumented.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_absolute_python_without_session_start(
    tmp_path,
):
    """Absolute-path invocation with no matching session_start sys.executable
    must fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no session_start
    _write_transcript(transcript, ["/usr/bin/python3 script.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_python_invocation_detection_counts_each_occurrence_in_compound_command(
    tmp_path,
):
    """R2 P0: a single Bash command containing two python invocations
    (`python a.py && python -S b.py`) must count as 2, not 1."""
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        ["python -c 'import diff_diff' && python -S uninstrumented.py"],
    )
    assert _count_python_invocations(transcript) == 2


def test_python_invocation_detection_counts_semicolon_separated_invocations(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        ["python a.py; /usr/bin/python3 b.py"],
    )
    assert _count_python_invocations(transcript) == 2


def test_python_invocation_detection_counts_pipe_separated_invocations(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        ["python -c 'print(1)' | python -c 'import sys; print(sys.stdin.read())'"],
    )
    assert _count_python_invocations(transcript) == 2


def test_merge_layers_diff_diff_raises_on_python_dash_S_flag(tmp_path):
    """R3 P0: `python -S` skips sitecustomize even if aggregate counts
    balance via an unrelated instrumented invocation. Detect the flag
    directly."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])  # 1 session
    _write_transcript(
        transcript,
        ["pip --version && python -S uninstrumented.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_accepts_python_dash_I_flag(tmp_path):
    """`-I` (isolated mode) does NOT skip sitecustomize when the shim is in
    the per-arm venv's site-packages. Isolated mode implies `-E`, `-P`,
    and lowercase `-s` — not `-S`. R4 code-quality correction: do not
    falsely reject `-I` runs."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-I", "script.py"])],
    )
    _write_transcript(transcript, ["python -I script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_raises_on_inline_path_override(tmp_path):
    """R5 P0: `pip --version && PATH=/usr/bin python3 ...` — the PATH=
    override forces resolution outside the per-arm-venv, so even though
    `python3` is relative the resolved interpreter does not have the shim."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["pip --version && PATH=/usr/bin python3 uninstrumented.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_export_path_mutation_before_python(
    tmp_path,
):
    """R6 P0: shell-level `export PATH=/usr/bin:$PATH && python3 ...`
    mutates PATH for subsequent commands; python3 resolves to a non-shim
    interpreter even though it's relative."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["pip --version && export PATH=/usr/bin:$PATH && python3 script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_standalone_path_then_python_amp(
    tmp_path,
):
    """R6 P0: `PATH=/usr/bin:$PATH && python3 ...` — standalone assignment
    propagated to the python3 invocation via `&&`."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["pip --version && PATH=/usr/bin:$PATH && python3 script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_source_activate_then_python(tmp_path):
    """R7 P0: `source X/bin/activate` mutates PATH for the shell;
    a subsequent `python` runs against the activated env (not the per-arm
    venv)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["pip --version && source /tmp/no-shim/bin/activate && python script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_dot_activate_then_python(tmp_path):
    """R7 P0: POSIX `. X/bin/activate && python` form."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        [". /tmp/no-shim/bin/activate && python script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_conda_activate_then_python(tmp_path):
    """R7 P0: `conda activate envname && python ...`"""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["conda activate other && python script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_heredoc_python_invocation(tmp_path):
    """R7 P0: `PATH=/usr/bin python3<<EOF\\n...` — heredoc syntax has `<`
    immediately after `python3`, no whitespace; the previous regex
    `python3(?:[\\s]|$)` missed it. New regex accepts `<` as a trailing
    boundary."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["PATH=/usr/bin python3<<'PY'\nimport diff_diff\nPY"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_standalone_path_then_python_semicolon(
    tmp_path,
):
    """R6 P0: same pattern with `;` separator instead of `&&`."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["pip --version && PATH=/usr/bin:$PATH ; python3 script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_env_python(tmp_path):
    """R5 P0: ``env python`` resolves the interpreter via PATH; in the
    per-arm-venv design the resolved interpreter may differ from the
    session_start's argv (which records the absolute resolved path
    via sys.orig_argv). With the AST parser, ``env`` is stripped as a
    wrapper and ``python script.py`` is the extracted invocation;
    attribution fails because the session_start argv won't match.
    Fail-closed via 'no matching session_start' instead of 'bypass'."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["env python script.py"])
    with pytest.raises(RunValidityError, match="bypass flag|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_dot_python(tmp_path):
    """R5 P0: ``./python`` invokes a local binary - one whose path
    contains a separator. The AST parser extracts ``[./python,
    script.py]``; exact-match attribution (no basename fallback for
    slash-containing paths) finds no matching session and fails closed
    via 'no matching session_start'."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["./python script.py"])
    with pytest.raises(RunValidityError, match="bypass flag|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_python_dash_Sc_compact_flag(tmp_path):
    """R4 P0: compact `-Sc 'code'` form (S combined with -c) also bypasses
    sitecustomize. The regex must catch it."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["pip --version && python -Sc 'import diff_diff'"])
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_does_not_raise_on_lowercase_dash_s(tmp_path):
    """Lowercase `-s` (skip user site) does NOT bypass sitecustomize; the
    bypass detector must distinguish it from uppercase `-S`."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-s", "script.py"])],
    )
    _write_transcript(transcript, ["python -s script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_raises_on_combined_bypass_flags(tmp_path):
    """Combined short flag `-IS` or `-SI` is also a bypass."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["python -IS script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_compound_partial_instrumentation(tmp_path):
    """R2 P0: one Bash call running both an instrumented `python` and an
    uninstrumented `python -S` must fail closed. Post-R3 the bypass-flag
    detector catches the `-S` directly."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])  # only 1
    _write_transcript(
        transcript,
        ["python -c 'import diff_diff' && python -S uninstrumented.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_accepts_balanced_invocation_session_counts(tmp_path):
    """3 python invocations + 3 session_starts (each matching by argv) =
    fully instrumented. Absolute-path and relative invocations both use
    argv matching post-R10: the visible argv must equal a session's
    ``argv`` (interpreter basename + args)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(argv=["python", "a.py"]),
            _session_start_event(
                sys_executable="/usr/bin/python3",
                argv=["/usr/bin/python3", "b.py"],
            ),
            _session_start_event(argv=["python", "-c", "pass"]),
        ],
    )
    _write_transcript(
        transcript,
        ["python a.py", "/usr/bin/python3 b.py", "python -c 'pass'"],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_raises_on_masked_absolute_python(tmp_path):
    """R4 P0: `pip --version && /usr/bin/python3 ...`. pip's session_start
    used to mask the uninstrumented /usr/bin/python3 under sys_executable
    pooling; under R10 argv attribution, the absolute visible path must
    match a session whose argv[0] equals it exactly. An unrelated session
    (from pip's per-arm-venv python, argv[0]=/per-arm-venv/bin/pip)
    cannot supply that match."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(
                sys_executable="/per-arm-venv/bin/python",
                argv=["/per-arm-venv/bin/pip", "--version"],
            )
        ],
    )
    _write_transcript(
        transcript,
        ["pip --version && /usr/bin/python3 script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_read_tool_requires_diff_diff_guides_path_segment(
    tmp_path,
):
    """R2 P2: a Read on `/tmp/llms.txt` (basename matches but no
    `diff_diff/guides/` segment) must NOT flip opened_llms_txt."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/tmp/llms.txt"},
                        },
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/Users/me/notes/llms-practitioner.txt"},
                        },
                    ],
                },
            }
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False


def test_merge_layers_diff_diff_read_tool_exact_basename_does_not_overmatch(tmp_path):
    """R1 P2: a path ending in `-llms.txt` (e.g. `my-llms.txt`) must not be
    treated as a bundled diff_diff guide."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/some/tmpdir/my-llms.txt"},
                        },
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/tmp/llms.txt.bak"},
                        },
                    ],
                },
            }
        ],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False
    assert record.opened_llms_autonomous is False
    assert record.opened_llms_full is False


# ---------------------------------------------------------------------------
# R10 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_pip_masking_relative_python(tmp_path):
    """R10 P0: `pip --version && python script.py` masking case.

    Pre-R10: pip's session_start (sys_executable = some per-arm-venv path,
    argv = [pip-script, --version]) would satisfy the visible `python
    script.py` relative invocation under unused-slot pooling, leaving
    `script.py`'s missing instrumentation undetected.

    Post-R10 argv attribution: the visible argv [`python`, `script.py`]
    must match a session's ``argv``; pip's argv does not match, and no
    other session is present, so the merger fails closed.
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(
                sys_executable="/per-arm-venv/bin/python",
                argv=["/per-arm-venv/bin/pip", "--version"],
            )
        ],
    )
    _write_transcript(transcript, ["pip --version && python script.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_truncated_transcript_no_terminal_result(
    tmp_path,
):
    """R10 P0: a stream-JSON transcript missing the terminal successful
    `result` entry indicates capture was cut short, and per-run evidence
    after the cut-off (later Bash invocations, tool_results) may be
    silently lost. Fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    with open(transcript, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"file_path": "/some/path.txt"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
    with pytest.raises(
        TelemetryMergeError, match="does not end with a `type=result subtype=success`"
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_error_result_entry(tmp_path):
    """R10 P0: a terminal `result` entry with ``is_error=true`` indicates
    the run did not complete cleanly. Fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text(
        json.dumps({"type": "result", "is_error": True, "subtype": "error_during_execution"}) + "\n"
    )
    with pytest.raises(
        TelemetryMergeError, match="does not end with a `type=result subtype=success`"
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_argv_matches_by_basename(tmp_path):
    """Argv-matching should accept path-vs-basename variation on the
    interpreter argv[0]. Visible ``python`` (bare name from a PATH lookup
    in the shell) matches session ``argv[0] = /per-arm-venv/bin/python``
    (sys.orig_argv may carry the resolved absolute path)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["/per-arm-venv/bin/python", "script.py"])],
    )
    _write_transcript(transcript, ["python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_args_must_match_exactly(tmp_path):
    """Argv args[1:] must match exactly; same interpreter + different
    args does not satisfy attribution."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "other.py"])],
    )
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_absolute_path_does_not_basename_match(tmp_path):
    """R11 P0: when the visible argv[0] is absolute, basename fallback is
    forbidden. `/usr/bin/python3 script.py` must NOT match a session whose
    argv[0] is `/per-arm-venv/bin/python3 script.py`, even though the
    basenames agree and the args match. The visible absolute path is
    unambiguous and must identify the exact interpreter; otherwise an
    off-venv `/usr/bin/python3` could silently inherit attribution from a
    different instrumented session and skip layer-2 telemetry capture."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(
                sys_executable="/per-arm-venv/bin/python3",
                argv=["/per-arm-venv/bin/python3", "script.py"],
            )
        ],
    )
    _write_transcript(transcript, ["/usr/bin/python3 script.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_absolute_path_exact_match_attributes(tmp_path):
    """R11: absolute-path argv[0] DOES match the session_start with the
    identical argv[0] (positive test). Pairs with the negative test above."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["/usr/bin/python3", "script.py"])],
    )
    _write_transcript(transcript, ["/usr/bin/python3 script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


@pytest.mark.parametrize(
    "visible_argv0",
    ["./venv/bin/python", "../venv/bin/python", "venv/bin/python"],
)
def test_merge_layers_diff_diff_relative_slash_path_does_not_basename_match(
    tmp_path, visible_argv0
):
    """R23 P0#2: visible argv[0] containing ANY path separator (absolute or
    relative) must require exact match. ``./venv/bin/python script.py``,
    ``../venv/bin/python script.py``, and ``venv/bin/python script.py`` are
    all explicit path launches; they may point at a project-local venv that
    is NOT the per-arm instrumented venv, and silently basename-matching to
    a ``/per-arm-venv/bin/python script.py`` session would mask an
    off-instrumentation invocation.

    Pre-R23 only absolute paths (``startswith('/')``) blocked the basename
    fallback; relative slash-containing paths slipped through and inherited
    the wrong session. R23 widens the gate to ``'/' in visible_argv[0]``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(
                sys_executable="/per-arm-venv/bin/python",
                argv=["/per-arm-venv/bin/python", "script.py"],
            )
        ],
    )
    _write_transcript(transcript, [f"{visible_argv0} script.py"])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_relative_slash_path_exact_match_attributes(tmp_path):
    """R23: relative-slash argv[0] DOES match when the session's argv[0] is
    identical (positive case). The exact-match rule applies the same way
    to relative-with-slash as to absolute paths."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["./venv/bin/python", "script.py"])],
    )
    _write_transcript(transcript, ["./venv/bin/python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_bare_token_still_basename_fallback(tmp_path):
    """R23: bare interpreter tokens (no path separator) still get basename
    fallback. ``python script.py`` matches a session whose argv[0] is the
    PATH-resolved ``/per-arm-venv/bin/python``; this is the legitimate
    case where the shell PATH-resolved the bare name and the absolute
    resolved path lives in ``sys.orig_argv[0]``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(
                sys_executable="/per-arm-venv/bin/python",
                argv=["/per-arm-venv/bin/python", "script.py"],
            )
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_strips_stdout_redirection(tmp_path):
    """R11 P2: shell redirection like `python script.py > out.txt` is NOT
    part of `sys.orig_argv`; the merger must strip it from the visible
    argv before comparing, or instrumented heredoc-style runs will be
    rejected as unattributed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(transcript, ["python script.py > out.txt"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_strips_heredoc(tmp_path):
    """R11 P2: heredoc redirection `python - <<'PY' ... PY` puts the
    heredoc body on stdin; the python program's argv is just `[python,
    -]`. Visible argv after redirection-stripping must match."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-"])],
    )
    _write_transcript(transcript, ["python - <<'PY'\nprint(1)\nPY"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_strips_fd_redirection(tmp_path):
    """R11 P2: fd-redirection `python script.py 2>&1` is not in argv."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(transcript, ["python script.py 2>&1"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_handles_quoted_semicolon(tmp_path):
    """R12 P2: `python -c 'import os; print(1)'` must NOT truncate at
    the in-quotes ``;``. Pre-R12 the raw-regex separator scan caught it
    and emitted argv like ``['python', '-c', "'import", 'os']``, which
    would never match a real session_start argv. Quote-aware scanning
    keeps the quoted region intact so shlex tokenization recovers the
    correct argv."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-c", "import os; print(1)"])],
    )
    _write_transcript(transcript, ["python -c 'import os; print(1)'"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_handles_quoted_pipe(tmp_path):
    """R12 P2: quoted ``|`` in `python -c 'a | b'` must not split args."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-c", "a | b"])],
    )
    _write_transcript(transcript, ["python -c 'a | b'"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_handles_double_quoted_separator(tmp_path):
    """R12 P2: double-quoted separators are also quote-aware. Backslash
    escapes inside double quotes are honored (so an escaped quote inside
    a double-quoted region does not prematurely close the quote)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-c", "print('x;y')"])],
    )
    _write_transcript(transcript, ["python -c \"print('x;y')\""])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_argv_still_splits_unquoted_semicolon(tmp_path):
    """R12 P2 regression-of-regression: the quote-aware scan must still
    truncate at an UNQUOTED ``;`` (otherwise we'd accept `python a.py;
    rm -rf /` as `python a.py rm -rf /`)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "a.py"])],
    )
    _write_transcript(transcript, ["python a.py; echo done"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# R13 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_tool_result_shim_failure_marker(tmp_path):
    """R13 P0: the shim's ``[pyruntime] cannot write event`` marker is
    emitted from the agent-spawned python subprocess and ends up inside
    the matching Bash ``tool_result.content``, NOT in outer
    ``cli_stderr.log``. Pre-R13 the merger only scanned cli_stderr, so a
    layer-2 event-write failure during agent execution was silently
    invisible and the per-run record could emit false-False guide flags.
    Post-R13 the merger scans both."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    # Build a transcript whose Bash tool_result contains the marker.
    transcript_entries = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "bash_1",
                        "input": {"command": "echo hi"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "bash_1",
                        "is_error": False,
                        "content": (
                            "stdout: hi\n"
                            "stderr: [pyruntime] cannot write event to "
                            "/tmp/run123/events.jsonl: [Errno 28] No space left on device\n"
                        ),
                    }
                ],
            },
        },
    ]
    _write_transcript_entries(transcript, transcript_entries)
    with pytest.raises(TelemetryMergeError, match="shim event-write failure marker.*tool_result"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_tool_result_marker_in_list_content(tmp_path):
    """R13 P0 (variant): tool_result content can be a list of text blocks
    instead of a single string. The scan must walk into list-shaped
    content too."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript_entries = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "bash_1",
                        "is_error": False,
                        "content": [
                            {"type": "text", "text": "some stdout"},
                            {
                                "type": "text",
                                "text": "[pyruntime] cannot write event to events.jsonl: ...",
                            },
                        ],
                    }
                ],
            },
        },
    ]
    _write_transcript_entries(transcript, transcript_entries)
    with pytest.raises(TelemetryMergeError, match="shim event-write failure marker.*tool_result"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_heredoc_body_not_treated_as_invocation(tmp_path):
    """R13 P2: a heredoc body like ``cat <<'EOF'\\npython x.py\\nEOF`` is
    cat-creates-file data, not a python invocation. Pre-R13 the regex
    scanned the body and treated the inner ``python x.py`` as a
    transcript-visible python invocation, requiring (and failing to
    find) a session_start. Post-R13 the body is stripped before
    scanning."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no python invocations expected
    _write_transcript(
        transcript,
        ["cat <<'EOF' > script.py\npython x.py\nimport diff_diff\nEOF"],
    )
    # No python invocations should be detected from the heredoc body; merger
    # accepts the run with all-False discoverability.
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_quoted_dash_S_in_python_c_not_bypass(tmp_path):
    """R13 P2: ``python -c "print(' -S ')"`` contains the substring
    ``-S`` only inside the quoted code argument. That is not an
    interpreter flag and must not be flagged as a sitecustomize-bypass.
    Pre-R13 the raw-text bypass regex caught it. Post-R13 the bypass
    detection walks shlex-tokenized argv and stops at the first non-flag
    token (the quoted code string)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-c", "print(' -S ')"])],
    )
    _write_transcript(transcript, ["python -c \"print(' -S ')\""])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_bypass_S_after_script_is_not_bypass(tmp_path):
    """R13 P2 (variant): ``python script.py -S`` does NOT bypass
    sitecustomize: once the python interpreter sees the script argument,
    everything after is ``sys.argv`` for the script. The argv-walking
    bypass detector stops at the first non-flag token, so the trailing
    ``-S`` is correctly ignored."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py", "-S"])],
    )
    _write_transcript(transcript, ["python script.py -S"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_inline_env_assignment_attributes(tmp_path):
    """R13 P2 (variant): ``MPLBACKEND=Agg python script.py`` should
    attribute normally. The env-var prefix is shell-level and not part of
    ``sys.orig_argv``, so the visible argv is just ``[python, script.py]``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(transcript, ["MPLBACKEND=Agg python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# R14 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_bash_dash_c_python(tmp_path):
    """R14 P0: ``bash -c "python -S script.py"`` puts the python token
    inside a quoted shell payload that the regex extractor does not visit.
    Pre-R14 the merger could accept this with an empty event log because no
    visible python invocation was extracted. Post-R14 the bypass detector
    flags the shell-wrapper + ``python`` literal combination."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ['bash -c "python -S script.py"'])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_bash_dash_lc_python(tmp_path):
    """R14 P0 variant: ``bash -lc`` (login shell) is a common form. Post-R18
    this no longer raises with 'bypass' (no -S primitive in payload), but
    unwrap+attribution catches the inner python invocation and fails closed
    with no-matching-session-start."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["bash -lc 'python script.py'"])
    with pytest.raises(TelemetryMergeError, match="bypass|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_eval_python(tmp_path):
    """R14 P0 variant: ``eval 'python script.py'`` similarly hides the
    python token from the regex extractor. Post-R18 unwrap+attribution
    catches it via the eval payload extractor."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["eval 'python script.py'"])
    with pytest.raises(TelemetryMergeError, match="bypass|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_sh_dash_c_python(tmp_path):
    """R14 P0 variant: ``sh -c`` works the same way."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ['sh -c "PATH=/usr/bin python script.py"'])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_grep_python_not_treated_as_invocation(tmp_path):
    """R14 P2: ``grep python script.py`` searches for the literal string
    ``python`` in script.py; ``python`` is an argument to grep, NOT an
    interpreter invocation. Pre-R14 the regex extractor matched the
    ``python`` token by leading-space boundary and required a session_start
    for it. Post-R14 the command-position check rejects matches whose
    preceding non-whitespace char is not a segment separator."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no python invocations expected
    _write_transcript(transcript, ["grep python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_echo_python_not_treated_as_invocation(tmp_path):
    """R14 P2 variant: ``echo python`` should not flag."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["echo python"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_subshell_python_attributes(tmp_path):
    """R14 P2 variant: ``(python script.py)`` runs python in a subshell.
    The trailing ``)`` must NOT attach to the script.py argv token, and
    the python token IS in command position (immediately follows ``(``)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(transcript, ["(python script.py)"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# R15 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_command_modifier_python_dash_S(tmp_path):
    """R15 P0: ``command python -S script.py`` invokes python through the
    POSIX ``command`` builtin so the python token is not in command
    position; pre-R15 the extractor skipped it and the ``-S`` ran without
    sitecustomize. Post-R15 the bypass detector flags
    modifier+python+``-S`` literal."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["command python -S script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_time_python_dash_S(tmp_path):
    """R15 P0 variant: ``time python -S ...``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["time python -S script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_nohup_python_dash_S(tmp_path):
    """R15 P0 variant: ``nohup python -S ...``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["nohup python -S script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_timeout_python_dash_S(tmp_path):
    """R15 P0 variant: ``timeout 30 python -S ...`` (modifier with arg)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["timeout 30 python -S script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_bash_long_option_dash_c(tmp_path):
    """R15 P0: ``bash --noprofile -c "python -S ..."`` has options before
    the ``-c`` token. Pre-R15 the wrapper regex required ``-c`` to follow
    the shell name immediately (no intervening options); post-R15 it
    allows option tokens between."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ['bash --noprofile -c "python -S script.py"'])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_bash_dash_o_pipefail_dash_c(tmp_path):
    """R15 P0 variant: ``bash -o pipefail -c "..."`` (multi-token option).
    Post-R18 the wrapper without -S/PATH= primitive falls through to
    unwrap+attribution, which fails closed on missing session_start."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ['bash -o pipefail -c "python script.py"'])
    with pytest.raises(TelemetryMergeError, match="bypass|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


# ---------------------------------------------------------------------------
# R16 regressions: wrapper-aware unwrap closes the architectural class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # Absolute path under modifiers (R16 finding 1, examples 1-2)
        "command /usr/bin/python -S script.py",
        "time /usr/bin/python -S script.py",
        # Modifier with options (R16 finding 1, examples 3-5)
        "nice -n 10 python -S script.py",
        "timeout --signal=KILL 30 python -S script.py",
        "xargs -I{} python -S script.py",
        # env modifier (R16 finding 1, example 6)
        "env /usr/bin/python -S script.py",
        # Shell control structures (R16 finding 1, examples 7-9)
        "{ python -S script.py; }",
        "if python -S script.py; then echo ok; fi",
        "! python -S script.py",
        # Backtick command substitution (R16 finding 1, example 10)
        "echo `python -S script.py`",
    ],
)
def test_merge_layers_diff_diff_unwrap_catches_hidden_dash_S(command, tmp_path):
    """R16 P0: shell wrappers / modifiers / control structures /
    substitutions can hide the ``-S`` bypass primitive from outer
    scanning. The unwrapper recursively strips known forms and the
    bypass detector inspects each variant for python+``-S``.
    Pre-R16 the merger could accept these with an empty event log
    because no visible python invocation was extracted."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, [command])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_unwrap_does_not_overflag_safe_wrappers(tmp_path):
    """R16 negative test: a benign wrapper around python (no ``-S``) must
    not be flagged as bypass. ``time python script.py`` should attribute
    normally given a matching session_start."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(transcript, ["time python script.py"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


@pytest.mark.parametrize(
    "command",
    [
        # Reviewer-named cases from R25 P0.
        "cd /tmp && time python script.py",
        "cd /tmp && time /usr/bin/python script.py",
        "cd /tmp && command python script.py",
        "cd /tmp && xargs python script.py",
        "cd /tmp && exec python script.py",
        # Other modifier families on the same compound shape.
        "cd /tmp && nice python script.py",
        "cd /tmp && nice -n 10 python script.py",
        "cd /tmp && timeout 30 python script.py",
        "cd /tmp && timeout --signal=KILL 30 python script.py",
        "cd /tmp && stdbuf -oL python script.py",
        "cd /tmp && ionice python script.py",
        "cd /tmp && chrt -f 1 python script.py",
        "cd /tmp && nohup python script.py",
        # `;` and newline separators behave like `&&` for segment boundaries.
        "cd /tmp; time python script.py",
        "cd /tmp\ntime python script.py",
        # Env-prefix in front of the modifier inside a segment.
        "cd /tmp && VAR=1 time python script.py",
        "cd /tmp && VAR=1 OTHER=2 nice -n 10 python script.py",
        # Env-prefix at start of command, then modifier (no compound).
        "VAR=1 nice -n 10 python script.py",
        "MPLBACKEND=Agg time python script.py",
    ],
)
def test_merge_layers_diff_diff_modifier_after_separator_fails_closed(tmp_path, command):
    """R25 P0: command modifiers (``time``, ``nice``, ``command``, etc.)
    after a shell separator or env-prefix must still be unwrapped so the
    python invocation is extracted and attribution required.

    Pre-R25 ``_strip_command_modifier_prefix`` only stripped modifiers at
    start of the whole Bash command. ``cd /tmp && time python script.py``
    produced NO extracted invocation; an uninstrumented python could run
    with no required session_start, and the merger would emit a clean
    all-False record - silent layer-2 telemetry loss.

    The fix splits the command on unquoted shell separators FIRST, then
    applies modifier-strip per segment. Each parametrized command above
    contains a python invocation that must be visible to attribution;
    with an empty event log and no matching session_start, the merger
    raises ``no matching session_start``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, [command])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


@pytest.mark.parametrize(
    "command,session_argv",
    [
        ("cd /tmp && time python script.py", ["python", "script.py"]),
        (
            "cd /tmp && time /usr/bin/python script.py",
            ["/usr/bin/python", "script.py"],
        ),
        (
            "cd /tmp && nice -n 10 python script.py",
            ["python", "script.py"],
        ),
        (
            "VAR=1 nice -n 10 python script.py",
            ["python", "script.py"],
        ),
    ],
)
def test_merge_layers_diff_diff_modifier_after_separator_with_session_attributes(
    tmp_path, command, session_argv
):
    """R25: matching session_start (interpreter + args) for the same
    compound-modifier forms attributes correctly. Pairs with the
    negative test above; verifies the fix doesn't over-extract or
    misattribute when the session is present."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event(argv=session_argv)])
    _write_transcript(transcript, [command])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# R17 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_post_heredoc_python_extracted(tmp_path):
    """R17 P0: ``cat > script.py <<'PY'\\n...\\nPY\\npython script.py`` is
    a common agent pattern (heredoc-create a script, then run it). Pre-R17
    the heredoc strip consumed the trailing newline and glued the post-
    heredoc command onto the opener (``...PY'python script.py``), hiding
    the python invocation from extraction. Post-R17 a synthetic
    separator preserves the post-heredoc command for attribution."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(
        transcript,
        ["cat > script.py <<'PY'\nimport os\nprint('hi')\nPY\npython script.py"],
    )
    # No session_start matches `python script.py`, so attribution fails closed.
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_post_heredoc_python_attributes_with_session(tmp_path):
    """R17 positive: heredoc-create then run, with a matching session_start,
    attributes normally."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"])],
    )
    _write_transcript(
        transcript,
        ["cat > script.py <<'PY'\nimport os\nPY\npython script.py"],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_post_heredoc_absolute_python(tmp_path):
    """R17 P0 variant: ``...heredoc...\\n/usr/bin/python script.py``,
    absolute interpreter after heredoc. Attribution requires an
    absolute-path session_start (no basename fallback for absolute);
    none is present, so fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(
        transcript,
        ["cat > script.py <<'PY'\nx = 1\nPY\n/usr/bin/python script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_post_heredoc_python_dash_S(tmp_path):
    """R17 P0 variant: heredoc + ``python -S script.py`` after. Either
    the bypass detector or attribution catches the post-heredoc python
    -S; both are fail-closed and either error is correct."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(
        transcript,
        ["cat > script.py <<'PY'\nx = 1\nPY\npython -S script.py"],
    )
    with pytest.raises(TelemetryMergeError, match="bypass|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


# ---------------------------------------------------------------------------
# R18 regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # for-loop body (R18 P0 #1, example 1)
        'for f in script.py; do python "$f"; done',
        # while-loop body (R18 P0 #1, example 2)
        "while true; do python script.py; break; done",
        # if-then body (R18 P0 #1, example 3) - the body, not the test
        "if true; then python script.py; fi",
        # case arm body (R18 P0 #1, example 4)
        "case x in x) python script.py ;; esac",
    ],
)
def test_merge_layers_diff_diff_shell_control_body_requires_session(command, tmp_path):
    """R18 P0 #1: python in a one-line shell-control body (for/while/if-then/
    case-arm) must be extracted and attributed. The AST walker recurses
    into body subtrees so all variants surface; ``for`` bodies with
    non-literal argv (``$f``) raise ShellCommandIndeterminate, ``case``
    raises ShellCommandParseError (bashlex doesn't model patterns), and
    the rest produce 'no matching session_start' on an empty event log."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no session
    _write_transcript(transcript, [command])
    with pytest.raises(
        RunValidityError,
        match=("bypass|no matching session_start|non-literal expansion|" "unsupported Bash form"),
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_shell_control_body_with_dash_S(tmp_path):
    """R18 P0 #1 variant: ``if true; then python -S script.py; fi`` puts a
    bypass primitive inside a then-body. Unwrap surfaces it for bypass."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["if true; then python -S script.py; fi"])
    with pytest.raises(TelemetryMergeError, match="bypass|no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


@pytest.mark.parametrize(
    "bad_event",
    [
        # session_start without argv (the shim always writes argv).
        {"event": "session_start", "ts": "2026-05-12T00:00:00Z"},
        # guide_file_read with no via.
        {"event": "guide_file_read"},
        # guide_file_read via=get_llm_guide without variant.
        {"event": "guide_file_read", "via": "get_llm_guide"},
        # guide_file_read via=open without filename.
        {"event": "guide_file_read", "via": "open"},
        # guide_file_read with unknown via.
        {"event": "guide_file_read", "via": "unknown_method"},
        # estimator_init without class.
        {"event": "estimator_init"},
        # estimator_fit without class.
        {"event": "estimator_fit"},
        # diagnostic_call without name.
        {"event": "diagnostic_call"},
        # warning_emitted without filename.
        {"event": "warning_emitted", "category": "UserWarning"},
        # module_import without module.
        {"event": "module_import"},
    ],
)
def test_merge_layers_diff_diff_raises_on_malformed_known_event(bad_event, tmp_path):
    """R18 P0 #2: a known event type with missing required fields is silently
    lossy under the previous merger (downstream code reads
    ``event.get("filename", "")`` and the empty default doesn't match any
    bundled guide, so the event APPEARS in the log but contributes
    nothing). Schema validation rejects it at parse time."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # Pair the malformed event with a valid session_start so the
    # validate-shim-loaded check doesn't fire first.
    _write_events_jsonl(events_path, [_session_start_event(), bad_event])
    with pytest.raises(TelemetryMergeError, match="missing required|unknown via"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


@pytest.mark.parametrize(
    "bad_event,match",
    [
        # session_start argv must be list[str].
        (
            {"event": "session_start", "ts": "x", "argv": "python script.py", "pid": 1},
            "argv must be list",
        ),
        (
            {"event": "session_start", "ts": "x", "argv": ["python", 123], "pid": 1},
            "argv must be list",
        ),
        # guide_file_read with unknown variant.
        (
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": "not-real"},
            "unknown variant",
        ),
        # guide_file_read with non-string variant.
        (
            {"event": "guide_file_read", "via": "get_llm_guide", "variant": 42},
            "variant must be str",
        ),
        # guide_file_read with unknown filename.
        (
            {"event": "guide_file_read", "via": "open", "filename": "not-a-guide.txt"},
            "unknown filename",
        ),
        # guide_file_read with non-string filename.
        (
            {"event": "guide_file_read", "via": "open", "filename": ["llms.txt"]},
            "filename must be str",
        ),
        # estimator_init with non-string class.
        ({"event": "estimator_init", "class": 123}, "class must be str"),
        # estimator_fit with non-string class.
        ({"event": "estimator_fit", "class": None}, "class must be str"),
        # diagnostic_call with non-string name.
        ({"event": "diagnostic_call", "name": None}, "name must be str"),
        # warning_emitted with non-string filename.
        (
            {"event": "warning_emitted", "filename": ["x"]},
            "filename must be str",
        ),
        # module_import with non-string module.
        ({"event": "module_import", "module": 99}, "module must be str"),
    ],
)
def test_merge_layers_diff_diff_raises_on_malformed_value(bad_event, match, tmp_path):
    """R19 P0: schema validation must check field TYPES and ENUM values,
    not just presence. A present-but-malformed value (wrong type, unknown
    enum member) silently zeros out telemetry under downstream
    ``event.get(field, "")`` reads. Reject at parse time."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event(), bad_event])
    with pytest.raises(TelemetryMergeError, match=match):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


# ---------------------------------------------------------------------------
# R21 regressions
# ---------------------------------------------------------------------------


def test_merge_layers_diff_diff_raises_on_bash_tool_use_without_result(tmp_path):
    """R21 P0: a Bash tool_use without a matching tool_result must fail
    closed. The transcript is truncated between request and response; the
    subprocess stderr (potentially containing the shim event-write
    failure marker) is silently missing."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "-c", "print(1)"])],
    )
    # Write a Bash tool_use with NO matching tool_result.
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "id": "bash_1",
                            "input": {"command": "python -c 'print(1)'"},
                        }
                    ],
                },
            },
        ],
    )
    with pytest.raises(TelemetryMergeError, match="no matching tool_result"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_bash_tool_use_without_id(tmp_path):
    """R21 P0 variant: a Bash tool_use without an ``id`` field cannot be
    matched to its tool_result, so the merger fails closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript_entries(
        transcript,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "echo hi"},
                            # No "id" field
                        }
                    ],
                },
            },
        ],
    )
    with pytest.raises(TelemetryMergeError, match="missing or has empty 'id'"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_no_arg_python_attribution(tmp_path):
    """R21 P2: bare ``python`` invocation with no args (REPL) attributes
    against a session_start with ``argv=["python"]``. Pre-R21 the
    interp_end calculation truncated the trailing char to ``pytho``
    when the regex matched via ``$``."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python"])],
    )
    _write_transcript(transcript, ["python"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_no_arg_absolute_python(tmp_path):
    """R21 P2 variant: absolute path with no args. ``/usr/bin/python3``
    with no trailing space must extract correctly."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["/usr/bin/python3"])],
    )
    _write_transcript(transcript, ["/usr/bin/python3"])
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# R20 regressions: absolute env + command-substitution python launches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # Absolute env path (R20 finding 1, examples 1-2)
        "/usr/bin/env python -S script.py",
        "/bin/env python -S script.py",
        "/usr/local/bin/env python script.py",
        # env -S form (R20 finding 1, example 3)
        "/usr/bin/env -S python -S script.py",
        # Command substitution forms (R20 finding 1, examples 4-5)
        "$(which python) -S script.py",
        "`which python` -S script.py",
        "$(command -v python) script.py",
        "`type -p python3` script.py",
    ],
)
def test_merge_layers_diff_diff_env_wrapper_and_command_substitution(command, tmp_path):
    """R20 P0: absolute env wrappers (/usr/bin/env, /bin/env,
    /usr/local/bin/env), env -S forms, and command-substitution python
    launches (``$(which python)``, ``\\`which python\\```, ``$(command -v
    python)``) all fail closed. With the AST parser, command-
    substitution argv[0] forms raise ShellCommandIndeterminate (their
    resolved value cannot be statically known); env wrappers walk to the
    inner python and fail attribution."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, [command])
    with pytest.raises(
        RunValidityError,
        match=("bypass|no matching session_start|non-literal expansion|" "unsupported Bash form"),
    ):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_unknown_event_type_ignored(tmp_path):
    """R18 P0 #2 negative: unknown event TYPES are ignored (forward
    compatibility - a future shim version may emit new event types and we
    don't want old mergers to reject them)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(), {"event": "future_event_type", "data": "anything"}],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


@pytest.mark.parametrize(
    "command",
    [
        "time python script.py",
        "command python script.py",
        "nice -n 10 python script.py",
        "timeout 30 python script.py",
        "{ python script.py; }",
        "if python script.py; then echo ok; fi",
        "! python script.py",
        "echo `python script.py`",
        "time /usr/bin/python script.py",
    ],
)
def test_merge_layers_diff_diff_wrapped_python_requires_session(command, tmp_path):
    """R17 P0: even WITHOUT ``-S``, a python invocation hidden inside any
    recognized shell wrapper must be extracted and attributed. Pre-R17
    the unwrapper only checked variants for python+``-S``; benign-shape
    wrapped invocations were not extracted, so an absent session_start
    didn't fail closed and the merger could emit all-false discoverability."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no session
    _write_transcript(transcript, [command])
    with pytest.raises(TelemetryMergeError, match="no matching session_start"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


# ---------------------------------------------------------------------------
# Arm validation
# ---------------------------------------------------------------------------


def test_merge_layers_raises_on_unknown_arm(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    with pytest.raises(ValueError, match="unknown arm"):
        merge_layers("not_an_arm", transcript, events_path, stderr_log)


# ---------------------------------------------------------------------------
# TelemetryRecord smoke (post-init validation still works through merger)
# ---------------------------------------------------------------------------


def test_telemetry_record_is_dataclass_with_arm_field():
    # Quick sanity: TelemetryRecord is importable and arm-aware.
    rec = TelemetryRecord(
        arm="statsmodels",
        stream_json_path=Path("/dev/null"),
        in_process_events_path=Path("/dev/null"),
        stderr_path=Path("/dev/null"),
    )
    assert rec.arm == "statsmodels"
    assert rec.opened_llms_txt is None


# ---------------------------------------------------------------------------
# PR #5: three-layer cross-check (layer-1 AST ↔ layer-1.5 wrapper ↔ layer-2 shim)
# ---------------------------------------------------------------------------


def _make_paths_with_venv(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Variant of ``_make_paths`` that also returns a venv-shaped directory.

    The venv directory has the expected ``bin/python-real`` structure so the
    venv-root-anchored ``executable`` schema check accepts it. Used by PR #5
    three-layer tests that pass ``venv_path`` to ``merge_layers``.
    """
    events, transcript, stderr_log = _make_paths(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / ".pyruntime-real").mkdir(parents=True, exist_ok=True)
    (venv / ".pyruntime-real" / "python-real").touch()
    return events, transcript, stderr_log, venv


def test_merge_layers_three_layer_consistent_attribution(tmp_path):
    """layer-1 + layer-1.5 + layer-2 all present with matching argv[1:] and
    matching pid → clean TelemetryRecord.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "script.py"], pid=12345),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_zero_agent_exec_python_with_sentinel_passes(tmp_path):
    """Shell-only agent (no python invocations); only sentinel exec_python
    present → clean. The sentinel proves wiring, the AST sees no python
    invocations, the three-layer check has nothing to enforce."""
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        _sentinel_events(executable=exe),
    )
    _write_transcript(transcript, ["ls -la"])  # no python in transcript
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_zero_exec_python_events_fails_sentinel_demand(tmp_path):
    """No sentinel, no agent exec_python events → RunValidityError on
    sentinel demand. Uses skip_auto_session_end=False (default) so the
    session_start has a paired session_end, isolating the sentinel
    failure mode."""
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, [])
    with pytest.raises(RunValidityError, match="build-time sentinel missing"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_forged_executable_outside_venv_root_raises(tmp_path):
    """An exec_python event whose ``executable`` points outside the run's
    venv root fails schema validation (the structural defense against
    agent-forged events).
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _exec_python_event(
                pid=12345,
                argv=["-c", "pass"],
                ppid=_DEFAULT_RUNNER_PID,
                executable="/tmp/forged/python-real",
            ),
        ],
    )
    _write_transcript(transcript, [])
    with pytest.raises(TelemetryMergeError, match="not the run's venv real interpreter"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_layer1_invocation_with_no_layer15_event_raises(tmp_path):
    """An AST-visible python invocation with no matching agent exec_python
    event raises ``RunValidityError("layer-1.5 attestation missing")``.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "script.py"], pid=12345),
            # No agent exec_python event.
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(RunValidityError, match="layer-1.5 attestation missing"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_layer1_invocation_with_no_session_start_raises(tmp_path):
    """An AST-visible python invocation with an agent exec_python event but
    no matching session_start raises ``RunValidityError("layer-2
    attestation missing")``.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
            # No session_start with matching argv.
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    # The merger reaches argv attribution before three-layer-check; this
    # produces a "no matching session_start" failure from the existing
    # _attribute_python_invocations check, NOT the new layer-2-missing
    # message. Either failure mode is fine - the run is invalid; the test
    # asserts on the RunValidityError parent class.
    with pytest.raises(RunValidityError):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_exec_python_argv1plus_must_match(tmp_path):
    """``argv[1:]`` mismatch between layer-1.5 and layer-1 → fail-closed.
    argv[0] divergence (wrapper="python", sitecustomize=".../python-real")
    is allowed; argv[1:] mismatch is not.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "script.py"], pid=12345),
            _exec_python_event(
                pid=12345,
                argv=["python", "different_script.py"],  # argv[1:] mismatch
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(RunValidityError, match="layer-1.5 attestation missing"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_xargs_one_layer1_invocation_matches_multiple_exec_python_events(
    tmp_path,
):
    """N-to-many cardinality: one layer-1 invocation can match multiple
    agent exec_python events (xargs/find/parallel spawn N children).
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    # Build 3 agent exec_python events with the same argv[1:] (simulating
    # xargs spawning 3 python children).
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "worker.py"], pid=12345),
            _session_start_event(argv=["python", "worker.py"], pid=12346),
            _session_start_event(argv=["python", "worker.py"], pid=12347),
            _exec_python_event(pid=12345, argv=["python", "worker.py"], ppid=22222, executable=exe),
            _exec_python_event(pid=12346, argv=["python", "worker.py"], ppid=22222, executable=exe),
            _exec_python_event(pid=12347, argv=["python", "worker.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["python worker.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_schema_rejects_malformed_exec_python_event(tmp_path):
    """Schema validation rejects an exec_python event missing required
    fields (e.g., argv).
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    malformed = {
        "event": "exec_python",
        "pid": 12345,
        "ppid": _DEFAULT_RUNNER_PID,
        "ts": "2026-05-12T00:00:00Z",
        "executable": str(venv / ".pyruntime-real" / "python-real"),
        # missing "argv"
    }
    with open(events_path, "w") as f:
        f.write(json.dumps(malformed) + "\n")
    _write_transcript(transcript, [])
    with pytest.raises(TelemetryMergeError, match="missing required field 'argv'"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_schema_rejects_exec_python_executable_not_in_venv(tmp_path):
    """Specifically: executable with the right name but wrong venv root
    fails schema validation. /usr/bin/python-real is structurally different
    from ${venv}/bin/python-real.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _exec_python_event(
                pid=12345,
                argv=["-c", "pass"],
                ppid=_DEFAULT_RUNNER_PID,
                executable="/usr/bin/python-real",  # right name, wrong root
            ),
        ],
    )
    _write_transcript(transcript, [])
    with pytest.raises(TelemetryMergeError, match="not the run's venv real interpreter"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_legacy_mode_skips_three_layer_check(tmp_path):
    """When ``runner_pid`` is None (legacy fixture mode), the three-layer
    check is skipped entirely. The existing 138 merger fixtures pass
    unchanged.
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # No exec_python events at all; no sentinel; transcript has a python
    # invocation that would fail the three-layer check if it fired.
    _write_events_jsonl(
        events_path,
        [_session_start_event(argv=["python", "script.py"], pid=12345)],
    )
    _write_transcript(transcript, ["python script.py"])
    # No runner_pid passed → three-layer check skipped → merger uses only
    # the existing layer-1↔layer-2 attribution which succeeds here.
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


# ---------------------------------------------------------------------------
# PR #5 R0 regressions: reciprocal exec_python ↔ session_start, python-real
# bypass, -S flag in delegated exec.
# ---------------------------------------------------------------------------


def test_merge_layers_sentinel_exec_python_without_matching_session_start_raises(tmp_path):
    """R0 P0 #2: every exec_python (including the sentinel) MUST have a
    matching session_start by pid. If sitecustomize fails to load for the
    sentinel invocation, no session_start is emitted -> fail closed.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            # Sentinel exec_python without matching session_start.
            _sentinel_exec_python_event(executable=exe),
        ],
    )
    _write_transcript(transcript, [])
    with pytest.raises(RunValidityError, match="sentinel exec_python pid"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_agent_exec_python_without_matching_session_start_raises(tmp_path):
    """R0 P0 #2: an agent exec_python without a matching session_start
    by pid is a sitecustomize-failure signal (e.g., -S flag bypass). Fail
    closed even when layer-1 didn't extract the invocation.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            # Agent exec_python WITHOUT matching session_start.
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, [])  # AST sees nothing.
    with pytest.raises(RunValidityError, match="agent exec_python pid"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_agent_exec_python_with_dash_S_flag_raises(tmp_path):
    """R0 P0 #2: an exec_python event whose argv contains -S (or compact
    -Sc / -IS) fails closed even when sitecustomize somehow emitted a
    matching session_start. -S skips site.py import; treating any -S
    invocation as bypass closes the find-exec / xargs delegated-form
    bypass class.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "-S", "script.py"], pid=12345),
            _exec_python_event(
                pid=12345,
                argv=["python", "-S", "script.py"],
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, [])  # AST doesn't see the invocation (delegated form).
    with pytest.raises(RunValidityError, match="layer-1.5 bypass detected"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_python_real_direct_invocation_flagged_as_bypass(tmp_path):
    """R0 P0 #1: a transcript-visible direct invocation of python-real
    (which bypasses the layer-1.5 wrapper) is flagged by the shell_parser
    bypass detector. The merger fails closed before reaching three-layer
    checks.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    # Provide events that would otherwise validate; the bypass check
    # should fire FIRST.
    _write_events_jsonl(events_path, _sentinel_events(executable=exe))
    _write_transcript(transcript, [f"{venv}/bin/python-real script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_session_start_without_matching_exec_python_raises(tmp_path):
    """R1 P0 #1 reciprocal: every session_start MUST have a matching
    exec_python by pid. A session_start with no exec_python proves
    sitecustomize loaded on an interpreter the wrapper did NOT see
    (i.e., the agent invoked the real interpreter directly, bypassing
    the wrapper).
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            # Bypass session: agent invoked python-real directly. Sitecustomize
            # loaded (so session_start fires) but no exec_python event because
            # the wrapper was skipped.
            _session_start_event(argv=["python-real", "script.py"], pid=12345),
        ],
    )
    _write_transcript(transcript, [])
    with pytest.raises(RunValidityError, match="session_start pid"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_python3_visible_invocation_matches_python_real_session(tmp_path):
    """R2 P1 #2: `python3 script.py` is invoked via the wrapper which
    execs python-real. The wrapper records argv[0]="python3" but the
    sitecustomize records argv[0]=".../python-real". The merger's
    python-family basename matcher accepts any python alias for argv[0]
    so this attributes correctly.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            # session_start records argv[0] as the path to python-real.
            _session_start_event(
                argv=[exe, "script.py"],
                pid=12345,
            ),
            # exec_python records argv[0] as basename of $0 = "python3".
            _exec_python_event(
                pid=12345,
                argv=["python3", "script.py"],
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, ["python3 script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_python3X_visible_invocation_matches_python_real_session(tmp_path):
    """Same as above but for ``python3.11 script.py``."""
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(
                pid=12345,
                argv=["python3.11", "script.py"],
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, ["python3.11 script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_path_qualified_venv_python_matches_session(tmp_path):
    """R3 P1: ``${venv}/bin/python script.py`` (path-qualified visible)
    must attribute to the session whose argv[0] is the venv's
    ``.pyruntime-real/python-real``. Both paths resolve under the same
    venv root and are python-family executables; the legacy matcher's
    venv-aware bridge accepts the match.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    visible_python = str(venv / "bin" / "python")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(
                pid=12345,
                argv=["python", "script.py"],
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, [f"{visible_python} script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_path_qualified_off_venv_python_does_not_match(tmp_path):
    """Off-venv path-qualified invocations like ``/usr/bin/python script.py``
    MUST NOT silently attribute to a same-args venv session. The
    venv-aware bridge accepts only paths under the SAME venv root.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["/usr/bin/python script.py"])
    with pytest.raises((TelemetryMergeError, RunValidityError)):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_actual_python_substring_flagged_as_bypass(tmp_path):
    """R3 P0 follow-on: any visible reference to ``.actual-python``
    (the hidden CPython binary one layer beneath python-real) is also
    flagged as a bypass primitive.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(events_path, _sentinel_events(executable=exe))
    _write_transcript(transcript, [f"{venv}/.pyruntime-real/.actual-python -S script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            venv_path=venv,
        )


def test_merge_layers_session_argv_with_actual_python_basename_matches(tmp_path):
    """R4 P1 #1: after the strip-S shim execs .actual-python, sitecustomize
    records sys.orig_argv[0] = ``${venv}/.pyruntime-real/.actual-python``.
    The merger's python-family basename matcher must accept .actual-python
    so that visible ``python script.py`` still attributes correctly.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    actual = str(venv / ".pyruntime-real" / ".actual-python")
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            # session_start records argv[0] as the .actual-python path
            # (kernel-set by the strip-S shim's exec call).
            _session_start_event(argv=[actual, "script.py"], pid=12345),
            _exec_python_event(
                pid=12345,
                argv=["python", "script.py"],
                ppid=22222,
                executable=exe,
            ),
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_relative_venv_bin_python_matches(tmp_path, monkeypatch):
    """R5 P1: ``venv/bin/python script.py`` (visible relative path
    against the run cwd) must attribute to the session whose argv[0] is
    ``${venv}/.pyruntime-real/python-real``. The runner sets
    cwd=tmpdir and the venv lives at tmpdir/venv, so the relative
    visible path resolves under venv_path.parent + relative.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["venv/bin/python script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_relative_dot_slash_venv_bin_python_matches(tmp_path):
    """``./venv/bin/python script.py`` form same as above."""
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["./venv/bin/python script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_relative_venv_bin_python3X_matches(tmp_path):
    """Same with the python3.X alias."""
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=[exe, "script.py"], pid=12345),
            _exec_python_event(
                pid=12345, argv=["python3.11", "script.py"], ppid=22222, executable=exe
            ),
        ],
    )
    _write_transcript(transcript, ["venv/bin/python3.11 script.py"])
    record = merge_layers(
        "diff_diff",
        transcript,
        events_path,
        stderr_log,
        runner_pid=_DEFAULT_RUNNER_PID,
        venv_path=venv,
    )
    assert record.arm == "diff_diff"


def test_merge_layers_requires_runner_pid_and_venv_path_together(tmp_path):
    """R6 P2 CQ-1: production mode requires both runner_pid AND venv_path.
    Supplying only one is a validity footgun (skips the missing check
    silently).
    """
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, [])
    with pytest.raises(ValueError, match="runner_pid and venv_path together"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            runner_pid=_DEFAULT_RUNNER_PID,
            # venv_path missing
        )
    with pytest.raises(ValueError, match="runner_pid and venv_path together"):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            venv_path=tmp_path / "venv",
            # runner_pid missing
        )


def test_merge_layers_legacy_mode_rejects_log_with_exec_python_events(tmp_path):
    """R13 P1 (EV-1): legacy mode (runner_pid/venv_path both None) is
    only valid for fixture-style logs that contain ZERO exec_python
    events. A real PR #5 production log always contains at least one
    sentinel exec_python event; calling ``merge_layers`` with the old
    pre-PR-#5 signature on such a log would silently skip the venv-root
    allowlist + three-layer cross-check. Fail-closed instead.
    """
    events_path, transcript, stderr_log, venv = _make_paths_with_venv(tmp_path)
    exe = str(venv / ".pyruntime-real" / "python-real")
    # Production-shaped log: sentinel + agent exec_python + matching
    # session_start. Calling merge_layers WITHOUT runner_pid/venv_path
    # must reject the log rather than silently produce a clean record.
    _write_events_jsonl(
        events_path,
        [
            *_sentinel_events(executable=exe),
            _session_start_event(argv=["python", "script.py"], pid=12345),
            _exec_python_event(pid=12345, argv=["python", "script.py"], ppid=22222, executable=exe),
        ],
    )
    _write_transcript(transcript, ["python script.py"])
    with pytest.raises(
        TelemetryMergeError, match="exec_python events but runner_pid/venv_path were not supplied"
    ):
        merge_layers(
            "diff_diff",
            transcript,
            events_path,
            stderr_log,
            # runner_pid + venv_path both omitted -> legacy fixture mode,
            # but the log contains exec_python events -> fail closed.
        )
