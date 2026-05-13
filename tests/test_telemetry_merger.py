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


def _make_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (events_jsonl, transcript, stderr_log) paths under tmp_path."""
    events = tmp_path / "in_process_events.jsonl"
    transcript = tmp_path / "transcript.jsonl"
    stderr_log = tmp_path / "cli_stderr.log"
    transcript.touch()
    stderr_log.touch()
    return events, transcript, stderr_log


def _session_start_event() -> dict:
    return {"event": "session_start", "ts": "2026-05-12T00:00:00.000000+00:00"}


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


def test_merge_layers_diff_diff_full_events(tmp_path, monkeypatch):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    # Set the env var so importing the shim module's top-level code doesn't raise.
    # (Shim top-level writes a session_start event at import time.)
    shim_event_log = tmp_path / "shim_session_start.jsonl"
    shim_event_log.touch()
    monkeypatch.setenv("_PYRUNTIME_EVENT_LOG", str(shim_event_log))
    import sys

    sys.modules.pop("harness.sitecustomize_template", None)
    from harness.sitecustomize_template import (
        _DIAGNOSTIC_FUNCTION_NAMES,
        _ESTIMATOR_CLASS_NAMES,
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
    with pytest.raises(TelemetryMergeError, match="partial instrumentation"):
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


def test_merge_layers_diff_diff_accepts_empty_transcript_and_stderr(tmp_path):
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    # transcript and stderr_log are already 0-byte (touched by _make_paths).
    # Zero-byte capture is acceptable (trivial agent responses produce these).
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


def test_merge_layers_diff_diff_read_tool_guide_access_flips_opened_flag(tmp_path):
    """Layer-1 evidence: Claude's Read tool on a bundled guide file must
    populate `opened_llms_*` even when the in-process shim sees nothing
    (agent read the file without invoking Python)."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    # Transcript: a Read tool call on llms.txt; no Python invocation.
    transcript_entries = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/some/install/diff_diff/guides/llms.txt"},
                    }
                ],
            },
        }
    ]
    with open(transcript, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    # Read-tool-only access should NOT flip called_get_llm_guide.
    assert record.called_get_llm_guide is False
    assert record.get_llm_guide_variants == ()


def test_merge_layers_diff_diff_read_tool_recognizes_all_guide_filenames(tmp_path):
    """Verify all four bundled guides are detected via Read-tool evidence."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript_entries = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": f"/install/diff_diff/guides/{name}"},
                    }
                    for name in (
                        "llms.txt",
                        "llms-practitioner.txt",
                        "llms-autonomous.txt",
                        "llms-full.txt",
                    )
                ],
            },
        }
    ]
    with open(transcript, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is True
    assert record.opened_llms_practitioner is True
    assert record.opened_llms_autonomous is True
    assert record.opened_llms_full is True


def test_merge_layers_diff_diff_read_tool_ignores_non_guide_files(tmp_path):
    """Read tool on a non-guide path should not flip any flag."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript_entries = [
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
    ]
    with open(transcript, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")
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
    """R1 P0: 2 python invocations + 1 session_start = partial instrumentation
    (e.g. `python -S` bypassing sitecustomize). Must fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    _write_transcript(
        transcript,
        ["python -c 'import diff_diff'", "python -S uninstrumented.py"],
    )
    with pytest.raises(TelemetryMergeError, match="partial instrumentation"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_raises_on_absolute_python_without_session_start(
    tmp_path,
):
    """Absolute-path invocation with no session_start must fail closed."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [])  # no session_start
    _write_transcript(transcript, ["/usr/bin/python3 script.py"])
    with pytest.raises(TelemetryMergeError, match="partial instrumentation"):
        merge_layers("diff_diff", transcript, events_path, stderr_log)


def test_merge_layers_diff_diff_accepts_balanced_invocation_session_counts(tmp_path):
    """3 python invocations + 3 session_starts = fully instrumented; OK."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(), _session_start_event(), _session_start_event()],
    )
    _write_transcript(
        transcript,
        ["python a.py", "/usr/bin/python3 b.py", "python -c 'pass'"],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_read_tool_exact_basename_does_not_overmatch(tmp_path):
    """R1 P2: a path ending in `-llms.txt` (e.g. `my-llms.txt`) must not be
    treated as a bundled diff_diff guide."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(events_path, [_session_start_event()])
    transcript_entries = [
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
    ]
    with open(transcript, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False
    assert record.opened_llms_autonomous is False
    assert record.opened_llms_full is False


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
