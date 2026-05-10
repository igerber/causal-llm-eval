"""Per-arm venv management for cold-start runs.

For Phase 1 (30 runs total + calibration + dry-pass), a fresh venv per run is
acceptable. For Phase 2 (~1500+ runs), per-arm venv templates are pre-built once
and cloned per run; never mutated post-clone.

The shape of this module is set in Phase 0 so Phase 2 doesn't require rewrites.

Skeleton only.
"""

from __future__ import annotations

from pathlib import Path


def build_arm_template(arm: str, library_version: str, template_dir: Path) -> Path:
    """Build a per-arm venv template that can be cloned per run.

    arm: "diff_diff" or "statsmodels"
    library_version: PyPI version string (e.g., "3.3.2" for diff-diff)
    template_dir: where to materialize the template venv

    Returns the path to the materialized template venv.

    Implementation pending.
    """
    del arm, library_version, template_dir
    raise NotImplementedError("venv_pool.build_arm_template is not yet implemented")


def clone_for_run(template_dir: Path, run_dir: Path) -> Path:
    """Clone a per-run venv from a template. Returns the new venv path.

    Implementation pending.
    """
    del template_dir, run_dir
    raise NotImplementedError("venv_pool.clone_for_run is not yet implemented")
