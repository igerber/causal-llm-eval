"""Live three-layer attestation test.

@pytest.mark.live - actually spawns ``claude --bare``, builds a per-arm
venv, and pays for one short agent invocation. Excluded by default (see
pyproject.toml addopts). Run explicitly via ``pytest -m live`` or as part
of ``make smoke``.

PR #5: this is the FIRST end-to-end test of the three-layer attestation
chain (layer-1 AST + layer-1.5 wrapper + layer-2 sitecustomize). Asserts:

    1. The build-time sentinel exec_python event is in the log.
    2. The shim's session_start event is in the log.
    3. ``merge_layers`` validates the run successfully when given the
       runner's ``runner_pid`` and ``venv_path`` (i.e., the three-layer
       cross-check passes end-to-end).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.runner import RunConfig, run_one
from harness.telemetry import merge_layers


@pytest.mark.live
def test_run_one_attests_across_all_three_layers(tmp_path):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    output_dir = tmp_path / "live_attestation"
    config = RunConfig(
        arm="diff_diff",
        library_version="3.3.2",
        dataset_path=Path("/dev/null"),
        prompt_path=Path("/dev/null"),
        prompt_version="test_telemetry_live/v1",
        timeout_seconds=300,
    )
    result = run_one(
        config,
        "Respond with just the word OK and nothing else.",
        output_dir,
    )

    assert result.exit_code == 0, (
        f"Spawn failed (exit {result.exit_code}); " f"see {output_dir / 'cli_stderr.log'}"
    )
    assert result.venv_path is not None
    assert result.runner_pid is not None

    events = [
        json.loads(line)
        for line in result.in_process_events_path.read_text().splitlines()
        if line.strip()
    ]
    # Layer-1.5 fired at least once (the build-time sentinel).
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert len(exec_events) >= 1, f"no exec_python events in event log: {events}"
    sentinel_events = [e for e in exec_events if e.get("ppid") == result.runner_pid]
    assert (
        len(sentinel_events) >= 1
    ), f"no sentinel exec_python events (ppid={result.runner_pid}): {exec_events}"

    # Layer-2 fired (sitecustomize loaded for the sentinel invocation).
    session_starts = [e for e in events if e.get("event") == "session_start"]
    assert len(session_starts) >= 1, f"no session_start events in event log: {events}"

    # Merger validates end-to-end with runner_pid + venv_path.
    record = merge_layers(
        config.arm,
        result.transcript_jsonl_path,
        result.in_process_events_path,
        result.cli_stderr_log_path,
        runner_pid=result.runner_pid,
        venv_path=result.venv_path,
    )
    assert record.arm == "diff_diff"
