"""Live cold-start runner test.

@pytest.mark.live - actually spawns `claude --bare` and pays for one short
agent invocation. Excluded by default (see pyproject.toml addopts). Run
explicitly via `pytest -m live` or as part of `make smoke`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.probe import _extract_final_assistant_text, _materialize_placeholder_dataset
from harness.runner import RunConfig, run_one


@pytest.mark.live
def test_run_one_spawns_real_agent_with_trivial_prompt(tmp_path):
    """Spawn a real cold-start agent with a one-word task; verify the pipeline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    output_dir = tmp_path / "live_run"
    # PR #6 R1 P2: dataset_path must be a regular file (the runner now
    # rejects /dev/null as a character device). Reuse the probe's
    # placeholder-parquet helper for a deterministic 1-row fixture.
    dataset_path = _materialize_placeholder_dataset(tmp_path / "fixture")
    config = RunConfig(
        arm="diff_diff",
        library_version="3.3.2",
        dataset_path=dataset_path,
        prompt_path=Path("/dev/null"),
        prompt_version="test_live/v1",
        rubric_version="test_live/v1",
        timeout_seconds=300,
    )

    result = run_one(config, "Respond with just the word OK and nothing else.", output_dir)

    assert result.exit_code == 0, (
        f"Spawn failed (exit {result.exit_code}); " f"see {output_dir / 'cli_stderr.log'}"
    )
    transcript = output_dir / "transcript.jsonl"
    assert transcript.exists()
    text = transcript.read_text()
    assert text, "transcript.jsonl is empty"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        json.loads(stripped)

    final = _extract_final_assistant_text(transcript)
    assert "OK" in final.upper(), f"Final assistant text missing 'OK': {final!r}"
