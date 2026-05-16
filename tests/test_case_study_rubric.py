"""Unit tests for ``rubrics/case_study_v2.yaml``.

The Phase 1 grading rubric is consumed by ``graders/ai_judge.py`` and
must be PRE-DEFINED before any case-study runs (CLAUDE.md item 5).
Tests assert:

  1. YAML parses cleanly.
  2. Top-level schema shape (``version``, ``task``, ``criteria``).
  3. Each criterion has ``id``, ``prompt``, ``output_type``.
  4. The required six criteria are present (no missing, no extras —
     extras would silently change graded outputs; new criteria = new
     rubric version).
  5. ``estimator_classification`` enum covers the 11 locked buckets
     spanning both arms.
  6. ``reasoning_quality`` enum covers the 4 locked buckets.
  7. The PR #6 stub at ``rubrics/case_study_v1.yaml`` remains untouched.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"
_RUBRIC_V2 = _RUBRICS_DIR / "case_study_v2.yaml"
_RUBRIC_V1 = _RUBRICS_DIR / "case_study_v1.yaml"

_REQUIRED_CRITERIA_IDS: frozenset[str] = frozenset(
    {
        "estimator_classification",
        "estimator_classification_evidence",
        "diagnostics_invoked",
        "reasoning_quality",
        "confidence_intervals_reported",
        "final_att_estimate",
    }
)

_REQUIRED_ESTIMATOR_BUCKETS: frozenset[str] = frozenset(
    {
        "twoway_fixed_effects",
        "callaway_santanna",
        "sun_abraham",
        "did_basic_2x2",
        "chaisemartin_dhaultfoeuille",
        "ols_naive",
        "ols_with_explicit_fe",
        "mixedlm_panel",
        "other_statsmodels_regressor",
        "other_diff_diff_estimator",
        "no_estimator_instantiated",
    }
)

_REQUIRED_REASONING_BUCKETS: frozenset[str] = frozenset(
    {
        "rigorous_with_alternatives",
        "rigorous_no_alternatives",
        "minimal_justification",
        "incorrect_or_invalid",
    }
)

_VALID_OUTPUT_TYPES: frozenset[str] = frozenset(
    {
        "enum",
        "string",
        "bool",
        "float_or_null",
        "list_of_strings",
    }
)


def _load_v2() -> dict:
    return yaml.safe_load(_RUBRIC_V2.read_text())


def test_rubric_v2_parses() -> None:
    assert _RUBRIC_V2.exists(), f"{_RUBRIC_V2} missing"
    loaded = _load_v2()
    assert isinstance(loaded, dict)


def test_rubric_v2_top_level_schema() -> None:
    loaded = _load_v2()
    for key in ("version", "task", "criteria"):
        assert key in loaded, f"top-level key {key!r} missing from rubric v2"
    assert isinstance(loaded["criteria"], list), "rubric criteria must be a list"
    assert loaded["criteria"], "rubric criteria list is empty"


def test_rubric_v2_version_string() -> None:
    assert _load_v2()["version"] == "v2.0"


def test_rubric_v2_task_targets_case_study_v1_dataset() -> None:
    """The rubric grades runs against the case_study_v1 dataset+DGP combo."""
    assert _load_v2()["task"] == "case_study_v1"


def test_rubric_v2_each_criterion_has_required_fields() -> None:
    loaded = _load_v2()
    for i, criterion in enumerate(loaded["criteria"]):
        assert isinstance(criterion, dict), f"criterion[{i}] not a dict"
        for field in ("id", "prompt", "output_type"):
            assert field in criterion, f"criterion[{i}] missing {field!r}: {criterion!r}"
        assert criterion["output_type"] in _VALID_OUTPUT_TYPES, (
            f"criterion[{i}] {criterion['id']!r}: output_type "
            f"{criterion['output_type']!r} not in {sorted(_VALID_OUTPUT_TYPES)}"
        )
        if criterion["output_type"] == "enum":
            assert "enum_values" in criterion, (
                f"criterion[{i}] {criterion['id']!r}: output_type=enum requires " f"enum_values"
            )
            assert isinstance(criterion["enum_values"], list)
            assert criterion["enum_values"], "enum_values list is empty"


def test_rubric_v2_includes_exactly_the_required_criteria() -> None:
    """No missing AND no extras. Extras would silently change graded
    outputs; new criteria = new rubric version per the immutability policy."""
    loaded = _load_v2()
    ids = {c["id"] for c in loaded["criteria"]}
    missing = _REQUIRED_CRITERIA_IDS - ids
    extras = ids - _REQUIRED_CRITERIA_IDS
    assert not missing, f"rubric v2 missing criteria: {sorted(missing)}"
    assert not extras, (
        f"rubric v2 has unexpected criteria: {sorted(extras)}. New "
        f"criteria = new rubric version (case_study_v3.yaml), not in-place add."
    )


def test_rubric_v2_estimator_classification_enum_covers_required_buckets() -> None:
    loaded = _load_v2()
    estimator_criterion = next(
        c for c in loaded["criteria"] if c["id"] == "estimator_classification"
    )
    assert estimator_criterion["output_type"] == "enum"
    enum_values = set(estimator_criterion["enum_values"])
    missing = _REQUIRED_ESTIMATOR_BUCKETS - enum_values
    extras = enum_values - _REQUIRED_ESTIMATOR_BUCKETS
    assert not missing, f"estimator_classification missing buckets: {sorted(missing)}"
    assert not extras, (
        f"estimator_classification has unexpected buckets: {sorted(extras)}. "
        f"New bucket = new rubric version."
    )


def test_rubric_v2_reasoning_quality_enum_covers_required_buckets() -> None:
    loaded = _load_v2()
    reasoning_criterion = next(c for c in loaded["criteria"] if c["id"] == "reasoning_quality")
    assert reasoning_criterion["output_type"] == "enum"
    enum_values = set(reasoning_criterion["enum_values"])
    missing = _REQUIRED_REASONING_BUCKETS - enum_values
    extras = enum_values - _REQUIRED_REASONING_BUCKETS
    assert not missing, f"reasoning_quality missing buckets: {sorted(missing)}"
    assert not extras, f"reasoning_quality has unexpected buckets: {sorted(extras)}"


def test_rubric_v1_still_reserved_stub() -> None:
    """PR #6's reserved stub at ``case_study_v1.yaml`` must remain a
    placeholder. The v1 id is reserved per the PR #6 convention; in-place
    fill would silently invalidate any records referencing the v1 id."""
    assert _RUBRIC_V1.exists()
    loaded = yaml.safe_load(_RUBRIC_V1.read_text())
    assert loaded.get("criteria") == [], (
        "case_study_v1.yaml no longer has criteria == []; PR #6 reserved "
        "this stub. New criteria = case_study_v2.yaml (which now exists), "
        "not in-place edit of v1."
    )
