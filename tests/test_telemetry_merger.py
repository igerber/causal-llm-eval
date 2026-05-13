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


def _session_start_event(sys_executable: str | None = None) -> dict:
    """Build a session_start event. `sys_executable` defaults to None which
    matches relative-form python invocations in the per-invocation
    attribution check; pass an explicit absolute path to match an
    absolute-path invocation."""
    event = {"event": "session_start", "ts": "2026-05-12T00:00:00.000000+00:00"}
    if sys_executable is not None:
        event["sys_executable"] = sys_executable
    return event


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
    _write_events_jsonl(events_path, [_session_start_event()])
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
    _write_events_jsonl(events_path, [_session_start_event()])
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
    """3 python invocations + 3 session_starts = fully instrumented. The
    absolute-path invocation `/usr/bin/python3 b.py` must have a matching
    session_start whose `sys_executable` equals that path; the relative
    invocations claim any remaining session_start."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [
            _session_start_event(sys_executable="/usr/bin/python3"),
            _session_start_event(),
            _session_start_event(),
        ],
    )
    _write_transcript(
        transcript,
        ["python a.py", "/usr/bin/python3 b.py", "python -c 'pass'"],
    )
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.arm == "diff_diff"


def test_merge_layers_diff_diff_raises_on_masked_absolute_python(tmp_path):
    """R4 P0: `pip --version && /usr/bin/python3 ...` — pip's session_start
    used to mask the uninstrumented /usr/bin/python3. With per-invocation
    sys.executable matching, the absolute path must find a matching
    session_start; an unrelated session_start (from pip's per-arm-venv
    python, sys.executable = /per-arm-venv/bin/python) cannot supply it."""
    events_path, transcript, stderr_log = _make_paths(tmp_path)
    _write_events_jsonl(
        events_path,
        [_session_start_event(sys_executable="/per-arm-venv/bin/python")],
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
    transcript_entries = [
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
    ]
    with open(transcript, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")
    record = merge_layers("diff_diff", transcript, events_path, stderr_log)
    assert record.opened_llms_txt is False
    assert record.opened_llms_practitioner is False


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
