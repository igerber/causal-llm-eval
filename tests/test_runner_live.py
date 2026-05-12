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

from harness.probe import _extract_final_assistant_text
from harness.runner import RunConfig, run_one


@pytest.mark.live
def test_run_one_spawns_real_agent_with_trivial_prompt(tmp_path):
    """Spawn a real cold-start agent with a one-word task; verify the pipeline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    output_dir = tmp_path / "live_run"
    config = RunConfig(
        arm="diff_diff",
        library_version="n/a",
        dataset_path=Path("/dev/null"),
        prompt_path=Path("/dev/null"),
        prompt_version="test_live/v1",
        timeout_seconds=180,
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
