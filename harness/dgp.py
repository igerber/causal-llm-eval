"""Synthetic data-generating process for the Phase 1 case study.

Wraps :func:`diff_diff.generate_staggered_data` with locked starter
parameters and persists a deterministic-bytes parquet artifact plus a
``dgp_truth.json`` ground-truth sidecar.

The persisted parquet INTENTIONALLY DROPS the DGP's ``true_effect`` column
so the agent cannot read the answer from the dataset. The ground truth
lives only in ``dgp_truth.json`` (consumed by judges/extractors, never by
the agent's per-run venv).

Determinism contract: same ``seed`` + same harness commit + same pyarrow
+ same pandas → bit-identical ``data.parquet`` bytes on any platform.
Empirically verified at pyarrow 24.0.0 + pandas 3.0.3 + diff_diff 3.3.2.
Cross pyarrow major versions, bytes MAY change; the regeneration test
(``tests/test_dgp.py::test_committed_dataset_matches_regeneration``) is
the tripwire. Recovery is "regenerate, recommit, note in CHANGELOG".

Phase 1 starter parameters are uncalibrated. The calibration loop (a
future PR) tunes them; bumping any parameter requires bumping
:data:`CASE_STUDY_V1_DGP_VERSION` and regenerating the committed
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import diff_diff
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CASE_STUDY_V1_DGP_VERSION = "v1.0"

# Locked Phase 1 starter parameters. UNCALIBRATED — calibration loop tunes
# these before locking the case study. Bumping any value requires bumping
# CASE_STUDY_V1_DGP_VERSION and regenerating committed artifacts.
_DGP_CALL_KWARGS: dict = {
    "n_units": 200,
    "n_periods": 10,
    "cohort_periods": [4, 6, 8],
    "never_treated_frac": 0.25,
    "treatment_effect": 2.0,
    "dynamic_effects": True,
    "effect_growth": 0.5,
    "unit_fe_sd": 2.0,
    "time_trend": 0.1,
    "noise_sd": 0.5,
    "panel": True,
}

# Columns persisted to data.parquet. EXCLUDES `true_effect` (the DGP's
# ground-truth column) to preserve eval validity. EXCLUDES `treat`
# (derivable from `first_treat > 0`) to keep the agent's input minimal
# and standard.
_PERSISTED_COLUMNS: tuple[str, ...] = ("unit", "period", "outcome", "first_treat", "treated")

# Pinned dtypes for cross-platform parquet byte identity. Without these
# casts, pandas defaults can drift (Windows int32 vs linux/macOS int64),
# breaking the byte-identity contract.
_PERSISTED_DTYPES: dict[str, str] = {
    "unit": "int64",
    "period": "int64",
    "outcome": "float64",
    "first_treat": "int64",
    "treated": "int64",
}


def _deterministic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write df to parquet with bit-identical bytes across runs.

    Strips pandas/pyarrow version-stamped metadata via ``replace_schema_metadata({})``
    and pins all parquet writer options that have non-deterministic defaults
    (timestamps in metadata, dictionary encoding, statistics blocks, etc.).
    """
    table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata({})
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=False,
        store_schema=False,
        data_page_version="2.0",
        version="2.6",
        write_page_index=False,
        write_page_checksum=False,
    )


def _coerce_dgp_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pin column dtypes per :data:`_PERSISTED_DTYPES` for byte identity."""
    for col, dtype in _PERSISTED_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def _compute_ground_truth(df: pd.DataFrame) -> dict:
    """Extract the ground-truth structure from a generated DataFrame.

    The DGP records per-row `true_effect` in the unfiltered DataFrame; this
    helper aggregates it into the locked sidecar shape:

        n_units_per_cohort: {"<onset>": count, ..., "never_treated": count}
        true_effects_per_event_time_per_cohort: {"<onset>": {"<event_time>": tau}}
        overall_att_unweighted: mean of true_effect over treated observations

    Cohorts use stringified onset periods so JSON keys are stable. Counts are
    derived from the DGP's actual stochastic cohort assignment (the DGP
    samples cohort membership; counts vary slightly per seed).
    """
    per_unit = df.drop_duplicates("unit")[["unit", "first_treat"]]
    cohort_counts: dict[str, int] = {}
    for onset, count in per_unit["first_treat"].value_counts().items():
        key = "never_treated" if int(onset) == 0 else str(int(onset))
        cohort_counts[key] = int(count)

    treated_obs = df[df["treated"] == 1].copy()
    treated_obs["event_time"] = treated_obs["period"] - treated_obs["first_treat"]
    effects: dict[str, dict[str, float]] = {}
    for (onset, et), tau in treated_obs.groupby(["first_treat", "event_time"])["true_effect"].first().items():
        effects.setdefault(str(int(onset)), {})[str(int(et))] = float(tau)

    overall_att = float(treated_obs["true_effect"].mean()) if len(treated_obs) else 0.0
    return {
        "n_units_per_cohort": cohort_counts,
        "true_effects_per_event_time_per_cohort": effects,
        "overall_att_unweighted": overall_att,
    }


def _write_dgp_truth(out_path: Path, *, seed: int, df_full: pd.DataFrame, n_rows: int) -> None:
    """Write the ground-truth sidecar JSON with deterministic key order."""
    payload = {
        "dgp_version": CASE_STUDY_V1_DGP_VERSION,
        "generator_function": "diff_diff.generate_staggered_data",
        "diff_diff_version": diff_diff.__version__,
        "seed": int(seed),
        "parameters": _DGP_CALL_KWARGS,
        "ground_truth": _compute_ground_truth(df_full),
        "schema": {
            "columns": list(_PERSISTED_COLUMNS),
            "dtypes": dict(_PERSISTED_DTYPES),
            "n_rows": int(n_rows),
        },
        "uncalibrated": True,
        "notes": (
            "Starter parameters; calibration loop pending. The persisted "
            "data.parquet drops the DGP's `true_effect` column to preserve "
            "eval validity (agent must estimate, not read, the effect)."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=True)
        f.write("\n")


_README_TEXT = """\
# Case study v1 — Phase 1 synthetic dataset

Generated by `harness.dgp.generate_case_study_v1`. Wraps
`diff_diff.generate_staggered_data` with locked Phase 1 starter
parameters (see `harness/dgp.py::_DGP_CALL_KWARGS`).

## Files

- `data.parquet` — the dataset the agent reads.
- `dgp_truth.json` — ground-truth sidecar (parameters + true effects).
- `README.md` — this file.

## Columns in `data.parquet`

| Column | Type | Meaning |
|---|---|---|
| `unit` | int64 | Unit identifier (0..N-1) |
| `period` | int64 | Time period (0..T-1) |
| `outcome` | float64 | Observed outcome variable |
| `first_treat` | int64 | First treatment period (0 = never treated) |
| `treated` | int64 | 1 if unit is treated at THIS period (= first_treat > 0 AND period >= first_treat) |

The DGP's ground-truth `true_effect` column is INTENTIONALLY DROPPED to
preserve eval validity (the agent must estimate, not read, the effect).
True per-cohort per-event-time effects are recorded in `dgp_truth.json`.

## Regenerating

```
python -m harness.dgp datasets/case_study_v1 --seed 42
```

Idempotent given a stable pyarrow version. If pyarrow drifts and the
regenerated bytes differ from the committed artifact, the regeneration
test in `tests/test_dgp.py` will fail; recovery is "regenerate, recommit,
note in CHANGELOG".
"""


def generate_case_study_v1(out_dir: Path, *, seed: int = 42) -> Path:
    """Materialize the Phase 1 case-study dataset into ``out_dir``.

    Produces:
        ``out_dir/data.parquet``    — the dataset; deterministic byte content
        ``out_dir/dgp_truth.json``  — the ground-truth sidecar
        ``out_dir/README.md``       — human description

    Args:
        out_dir: directory to materialize into. Created if missing.
        seed: random seed passed to ``diff_diff.generate_staggered_data``.
            Default 42 is the committed artifact's seed; bumping requires
            bumping :data:`CASE_STUDY_V1_DGP_VERSION` and regenerating.

    Returns:
        ``out_dir / "data.parquet"`` — the path the agent reads from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_full = diff_diff.generate_staggered_data(seed=seed, **_DGP_CALL_KWARGS)
    df_persist = df_full[list(_PERSISTED_COLUMNS)].copy()
    df_persist = _coerce_dgp_dtypes(df_persist)

    parquet_path = out_dir / "data.parquet"
    _deterministic_write_parquet(df_persist, parquet_path)

    _write_dgp_truth(
        out_dir / "dgp_truth.json",
        seed=seed,
        df_full=df_full,
        n_rows=len(df_persist),
    )

    (out_dir / "README.md").write_text(_README_TEXT)

    return parquet_path


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.dgp",
        description="Materialize the Phase 1 case-study dataset.",
    )
    parser.add_argument("out_dir", type=Path, help="Directory to materialize artifacts into.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42; the committed-artifact seed).",
    )
    args = parser.parse_args(argv)
    parquet_path = generate_case_study_v1(args.out_dir, seed=args.seed)
    print(f"Wrote {parquet_path} (sha256={_sha256_of_file(parquet_path)})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
