"""Smoke tests for the harness skeleton.

Verifies the package imports and the public contracts are present. The actual
behavioral implementations are added in subsequent PRs; these tests ensure the
module surface stays stable.
"""

from __future__ import annotations


def test_harness_imports(tmp_path, monkeypatch):
    """Harness package imports without error and exposes expected modules.

    `harness.sitecustomize_template` writes a `session_start` event at module
    load (fail-closed contract: missing `_PYRUNTIME_EVENT_LOG` raises). The
    test sets a tmp event-log path so the import succeeds.
    """
    event_log = tmp_path / "events.jsonl"
    event_log.touch()
    monkeypatch.setenv("_PYRUNTIME_EVENT_LOG", str(event_log))

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


def test_run_metadata_schema_has_all_reproducibility_fields():
    """RunMetadata locks the reproducibility schema from the plan.

    Plan section "Reproducibility schema" requires every per-run record to
    pin these fields. Locking them in the dataclass at Phase 0 prevents
    subsequent PRs from satisfying surface tests while quietly omitting
    reproducibility metadata. Each field name corresponds to a documented
    requirement.
    """
    import typing

    from harness.runner import RunMetadata

    required = {
        "harness_version",
        "library_version",
        "claude_code_version",
        "model_version",
        "dataset_sha",
        "prompt_version",
        "rubric_version",
        "random_seed",
        "run_id",
        "arm",
    }
    hints = typing.get_type_hints(RunMetadata)
    missing = required - set(hints)
    assert not missing, f"RunMetadata missing required reproducibility fields: {missing}"


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
    hints = typing.get_type_hints(TelemetryRecord)
    assert flags <= set(hints)
    # arm field is required so downstream graders can validate the sentinel
    # pattern (None for not-applicable surfaces) against the arm's contract.
    assert "arm" in hints and hints["arm"] is str


def test_telemetry_sentinel_semantics():
    """Guide-discovery flags are tri-state to distinguish "no access" from "no surface".

    For arm 1 (diff-diff): True = agent accessed; False = agent did not access.
    For arm 2 (statsmodels): None = surface does not exist (not applicable).

    Collapsing None into False would bias comparator-fairness analysis in
    favor of statsmodels because absence-of-feature would look like
    absence-of-discovery.
    """
    from pathlib import Path

    from harness.telemetry import TelemetryRecord

    # Default construction is statsmodels-shaped: guide fields are None.
    record_statsmodels = TelemetryRecord(
        arm="statsmodels",
        stream_json_path=Path("/tmp/x"),
        in_process_events_path=Path("/tmp/y"),
        stderr_path=Path("/tmp/z"),
    )
    assert record_statsmodels.opened_llms_txt is None
    assert record_statsmodels.opened_llms_practitioner is None
    assert record_statsmodels.opened_llms_autonomous is None
    assert record_statsmodels.opened_llms_full is None
    assert record_statsmodels.called_get_llm_guide is None
    # Always-applicable fields default to False, not None.
    assert record_statsmodels.saw_fit_time_warning is False

    # diff-diff arm: caller must explicitly set guide fields to True/False.
    record_diff_diff = TelemetryRecord(
        arm="diff_diff",
        stream_json_path=Path("/tmp/x"),
        in_process_events_path=Path("/tmp/y"),
        stderr_path=Path("/tmp/z"),
        opened_llms_txt=False,
        opened_llms_practitioner=True,
        opened_llms_autonomous=False,
        opened_llms_full=False,
        called_get_llm_guide=True,
    )
    assert record_diff_diff.opened_llms_txt is False
    assert record_diff_diff.opened_llms_practitioner is True
    assert record_diff_diff.called_get_llm_guide is True


def test_telemetry_post_init_rejects_invalid_arm():
    """Unknown arm strings are rejected at construction time."""
    from pathlib import Path

    import pytest

    from harness.telemetry import TelemetryRecord

    with pytest.raises(ValueError, match="not one of"):
        TelemetryRecord(
            arm="unknown_lib",
            stream_json_path=Path("/tmp/x"),
            in_process_events_path=Path("/tmp/y"),
            stderr_path=Path("/tmp/z"),
        )


def test_telemetry_post_init_rejects_diff_diff_with_none_guide_fields():
    """diff-diff arm requires bool guide fields; None violates the contract."""
    from pathlib import Path

    import pytest

    from harness.telemetry import TelemetryRecord

    # Constructing with arm="diff_diff" and guide fields left at default None
    # is invalid - the merger must record discovery outcome, not leave None.
    with pytest.raises(ValueError, match="must be bool"):
        TelemetryRecord(
            arm="diff_diff",
            stream_json_path=Path("/tmp/x"),
            in_process_events_path=Path("/tmp/y"),
            stderr_path=Path("/tmp/z"),
            # opened_llms_txt left as default None - should raise
            opened_llms_practitioner=False,
            opened_llms_autonomous=False,
            opened_llms_full=False,
            called_get_llm_guide=False,
        )


def test_telemetry_post_init_rejects_statsmodels_with_bool_guide_fields():
    """statsmodels arm requires None guide fields; bool violates the contract."""
    from pathlib import Path

    import pytest

    from harness.telemetry import TelemetryRecord

    # Constructing arm="statsmodels" with a bool guide field is invalid -
    # statsmodels has no guide surfaces, so True/False conflates "not
    # applicable" with "not discovered".
    with pytest.raises(ValueError, match="must be None"):
        TelemetryRecord(
            arm="statsmodels",
            stream_json_path=Path("/tmp/x"),
            in_process_events_path=Path("/tmp/y"),
            stderr_path=Path("/tmp/z"),
            opened_llms_txt=False,  # invalid for statsmodels
        )
