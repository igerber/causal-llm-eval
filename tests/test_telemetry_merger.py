"""Unit tests for `harness/telemetry.py::merge_layers` and helpers."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from harness.telemetry import (
    _VARIANT_TO_FILENAME,
    TelemetryMergeError,
    TelemetryRecord,
    _count_python_invocations,
    merge_layers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events_jsonl(path: Path, events: list[dict]) -> None:
    """Write a list of event dicts as one JSON object per line."""
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_transcript(path: Path, bash_commands: list[str]) -> None:
    """Write a minimal stream-JSON transcript with the given Bash commands.

    Each command becomes a tool_use block inside an assistant message entry.
    A terminal `result` entry is appended so the merger's completeness
    check passes; tests that want to exercise truncation should overwrite
    the file without the result entry.
    """
    with open(path, "w") as f:
        for cmd in bash_commands:
            entry = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": cmd},
                        }
                    ],
                },
            }
            f.write(json.dumps(entry) + "\n")
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
    transcript.write_text('{"type": "result", "status": "ok"}\n')
    stderr_log.touch()
    return events, transcript, stderr_log


def _session_start_event(sys_executable: str | None = None, argv: list | None = None) -> dict:
    """Build a session_start event. Pass `argv` to match the corresponding
    transcript-visible python invocation under the new per-invocation
    attribution check."""
    event: dict = {"event": "session_start", "ts": "2026-05-12T00:00:00.000000+00:00"}
    if sys_executable is not None:
        event["sys_executable"] = sys_executable
    if argv is not None:
        event["argv"] = argv
    return event


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
    transcript.write_text('{"type": "result", "status": "ok"}\n')
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
    """R5 P0: `env python` resolves via PATH; can pick up a non-shim
    interpreter."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["env python script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_dot_python(tmp_path):
    """R5 P0: `./python` invokes a local binary."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(transcript, ["./python script.py"])
    with pytest.raises(TelemetryMergeError, match="bypass flag"):
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
    with pytest.raises(TelemetryMergeError, match="does not end with a successful"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_error_result_entry(tmp_path):
    """R10 P0: a terminal `result` entry with ``is_error=true`` indicates
    the run did not complete cleanly. Fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript.write_text(
        json.dumps({"type": "result", "is_error": True, "subtype": "error_during_execution"}) + "\n"
    )
    with pytest.raises(TelemetryMergeError, match="does not end with a successful"):
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
    """R14 P0 variant: ``bash -lc`` (login shell) is a common form."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["bash -lc 'python script.py'"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_eval_python(tmp_path):
    """R14 P0 variant: ``eval 'python script.py'`` similarly hides the
    python token from the regex extractor."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ["eval 'python script.py'"])
    with pytest.raises(TelemetryMergeError, match="bypass"):
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
    """R15 P0 variant: ``bash -o pipefail -c "..."`` (multi-token option)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])
    _write_transcript(transcript, ['bash -o pipefail -c "python script.py"'])
    with pytest.raises(TelemetryMergeError, match="bypass"):
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
