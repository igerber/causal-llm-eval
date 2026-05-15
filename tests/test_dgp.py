"""Tests for harness.dgp.

Headline contract: ``generate_case_study_v1`` produces bit-identical
``data.parquet`` bytes given the same seed + same harness commit + same
pyarrow version. The committed-vs-regenerated test guards against silent
parameter drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from harness.dgp import (
    _DGP_CALL_KWARGS,
    _PERSISTED_COLUMNS,
    _PERSISTED_DTYPES,
    CASE_STUDY_V1_DGP_VERSION,
    generate_case_study_v1,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMMITTED_DATASET = _REPO_ROOT / "datasets" / "case_study_v1" / "data.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_case_study_v1_deterministic(tmp_path: Path) -> None:
    """Two invocations with the same seed produce bit-identical bytes.

    This is the headline determinism contract. Justifies emitting a stable
    dataset_sha into per-run RunMetadata.
    """
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    parquet_a = generate_case_study_v1(out_a, seed=42)
    parquet_b = generate_case_study_v1(out_b, seed=42)
    assert _sha256(parquet_a) == _sha256(parquet_b)


def test_seed_changes_bytes(tmp_path: Path) -> None:
    """Different seed produces different bytes — verifies seed is wired."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    parquet_a = generate_case_study_v1(out_a, seed=42)
    parquet_b = generate_case_study_v1(out_b, seed=43)
    assert _sha256(parquet_a) != _sha256(parquet_b)


def test_dgp_truth_json_schema(tmp_path: Path) -> None:
    """Sidecar has the locked schema and the 2-level effects shape."""
    out_dir = tmp_path / "out"
    generate_case_study_v1(out_dir, seed=42)
    payload = json.loads((out_dir / "dgp_truth.json").read_text())

    required_top_level = {
        "dgp_version",
        "generator_function",
        "diff_diff_version",
        "seed",
        "parameters",
        "ground_truth",
        "schema",
        "uncalibrated",
        "notes",
    }
    assert required_top_level.issubset(payload.keys())

    assert payload["dgp_version"] == CASE_STUDY_V1_DGP_VERSION
    assert payload["seed"] == 42
    assert payload["uncalibrated"] is True
    assert payload["parameters"] == _DGP_CALL_KWARGS

    gt = payload["ground_truth"]
    assert set(gt.keys()) == {
        "n_units_per_cohort",
        "true_effects_per_event_time_per_cohort",
        "overall_att_unweighted",
    }
    # 2-level dict shape per plan §1
    effects = gt["true_effects_per_event_time_per_cohort"]
    assert isinstance(effects, dict)
    for cohort_key, inner in effects.items():
        assert isinstance(cohort_key, str)
        assert isinstance(inner, dict)
        for et_key, tau in inner.items():
            assert isinstance(et_key, str)
            assert isinstance(tau, (int, float))
    # never_treated key is reserved for the always-untreated cohort
    assert "never_treated" in gt["n_units_per_cohort"]

    schema = payload["schema"]
    assert schema["columns"] == list(_PERSISTED_COLUMNS)
    assert schema["dtypes"] == dict(_PERSISTED_DTYPES)
    assert isinstance(schema["n_rows"], int)


def test_committed_dataset_matches_regeneration(tmp_path: Path) -> None:
    """Regenerating with seed=42 reproduces the committed bytes.

    Catches silent parameter drift: if someone edits _DGP_CALL_KWARGS
    without bumping CASE_STUDY_V1_DGP_VERSION and regenerating the
    committed artifact, this test fails loudly.

    Recovery if pyarrow drift breaks bit-identity (not parameter drift):
    re-run ``python -m harness.dgp datasets/case_study_v1 --seed 42``,
    commit the new bytes, note in CHANGELOG.
    """
    if not _COMMITTED_DATASET.exists():
        pytest.skip("Committed dataset not present yet (PR #6 step 2 produces it).")
    parquet = generate_case_study_v1(tmp_path, seed=42)
    assert _sha256(parquet) == _sha256(_COMMITTED_DATASET), (
        "Regenerated parquet does not match committed bytes. "
        "If parameter drift: bump CASE_STUDY_V1_DGP_VERSION + regenerate. "
        "If pyarrow drift: regenerate + recommit + note in CHANGELOG."
    )


def test_round_trip_dataframe_equals(tmp_path: Path) -> None:
    """Reading the parquet back yields the same DataFrame we wrote."""
    parquet = generate_case_study_v1(tmp_path, seed=42)
    df_read = pd.read_parquet(parquet)
    assert list(df_read.columns) == list(_PERSISTED_COLUMNS)
    # Re-generate the in-memory equivalent and compare column-by-column.
    import diff_diff

    df_full = diff_diff.generate_staggered_data(seed=42, **_DGP_CALL_KWARGS)
    df_expected = df_full[list(_PERSISTED_COLUMNS)].copy()
    for col, dtype in _PERSISTED_DTYPES.items():
        df_expected[col] = df_expected[col].astype(dtype)
    pd.testing.assert_frame_equal(
        df_read.reset_index(drop=True), df_expected.reset_index(drop=True)
    )


def test_dgp_dtypes_pinned(tmp_path: Path) -> None:
    """Parquet columns have the exact pinned dtypes (cross-platform stability).

    Without explicit casts, pandas defaults can vary (e.g., Windows int32
    vs linux/macOS int64), breaking the byte-identity contract. This test
    encodes the contract that ``_coerce_dgp_dtypes`` enforces.
    """
    parquet = generate_case_study_v1(tmp_path, seed=42)
    table = pq.read_table(parquet)
    actual_arrow_dtypes = {field.name: str(field.type) for field in table.schema}
    expected_arrow_dtypes = {
        "unit": "int64",
        "period": "int64",
        "outcome": "double",  # pyarrow renders float64 as 'double'
        "first_treat": "int64",
        "treated": "int64",
    }
    assert actual_arrow_dtypes == expected_arrow_dtypes

    # Also confirm via pandas that the dtypes survive the round-trip.
    df = pd.read_parquet(parquet)
    assert df["unit"].dtype == np.int64
    assert df["period"].dtype == np.int64
    assert df["outcome"].dtype == np.float64
    assert df["first_treat"].dtype == np.int64
    assert df["treated"].dtype == np.int64


def test_persisted_parquet_excludes_true_effect(tmp_path: Path) -> None:
    """The DGP's ground-truth `true_effect` column MUST NOT leak to the agent.

    Eval validity: the agent's task is to ESTIMATE the effect; if true_effect
    is in the parquet, the agent can read the answer (df["true_effect"].mean()).
    """
    parquet = generate_case_study_v1(tmp_path, seed=42)
    df = pd.read_parquet(parquet)
    assert "true_effect" not in df.columns
    assert "treat" not in df.columns  # also dropped (derivable from first_treat)
