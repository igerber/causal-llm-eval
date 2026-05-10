"""Smoke tests for the harness skeleton.

Verifies the package imports and the public contracts are present. The actual
behavioral implementations are added in subsequent PRs; these tests ensure the
module surface stays stable.
"""

from __future__ import annotations


def test_harness_imports():
    """Harness package imports without error and exposes expected modules."""
    import harness  # noqa: F401
    from harness import (  # noqa: F401
        extractor,
        runner,
        scheduler,
        sitecustomize_template,
        telemetry,
        venv_pool,
    )

    assert harness.__version__


def test_graders_imports():
    """Graders package imports without error."""
    import graders  # noqa: F401
    from graders import ai_judge  # noqa: F401

    assert hasattr(ai_judge, "judge_transcript")


def test_runner_contract():
    """RunConfig and RunResult dataclass shapes are present."""
    import typing

    from harness.runner import RunConfig, RunResult, run_one

    # Use get_type_hints to resolve `from __future__ import annotations` strings
    assert typing.get_type_hints(RunConfig)["arm"] is str
    assert "transcript_jsonl_path" in typing.get_type_hints(RunResult)
    assert callable(run_one)


def test_telemetry_contract():
    """TelemetryRecord exposes the discoverability flags the rubric expects."""
    import typing

    from harness.telemetry import TelemetryRecord

    flags = {
        "opened_llms_txt",
        "opened_llms_practitioner",
        "opened_llms_autonomous",
        "opened_llms_full",
        "called_get_llm_guide",
        "saw_fit_time_warning",
    }
    assert flags <= set(typing.get_type_hints(TelemetryRecord))
