"""Live cold-start probe test.

@pytest.mark.live - actually spawns `claude --bare` and pays for one short
agent invocation. Excluded by default (see pyproject.toml addopts). Run
explicitly via `pytest -m live` or as part of `make smoke`.
"""

from __future__ import annotations

import os

import pytest

from harness.probe import run_probe


@pytest.mark.live
def test_probe_returns_pass_on_cold_start(tmp_path):
    """The live probe must PASS on a correctly-isolated cold-start agent."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    output_dir = tmp_path / "live_probe"
    result = run_probe(output_dir=output_dir, timeout_seconds=180)

    assert result.run_result.exit_code == 0, (
        f"Probe spawn failed (exit {result.run_result.exit_code}); "
        f"see {output_dir / 'cli_stderr.log'}"
    )
    assert result.assessment.passed, (
        f"Probe failed with findings: {result.assessment.findings}\n"
        f"Response: {result.assessment.agent_response!r}"
    )
    assert result.assessment.findings == []
