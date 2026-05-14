"""Per-arm venv management for cold-start runs.

For Phase 1 (30 runs total + calibration + dry-pass), a fresh venv per run is
acceptable. For Phase 2 (~1500+ runs), per-arm venv templates are pre-built once
and cloned per run; never mutated post-clone.

The shape of this module is set in Phase 0 so Phase 2 doesn't require rewrites.

PR #5 implements ``build_arm_template`` end-to-end: create venv, pip install the
arm library at the pinned version, copy the sitecustomize shim into
site-packages, install the layer-1.5 ``python`` wrapper. ``clone_for_run``
remains a stub (Phase 2; PR #6+).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

_ARM_TO_PIP_PACKAGE: dict[str, str] = {
    "diff_diff": "diff-diff",
    # "statsmodels" deferred to PR #7 (statsmodels arm instrumentation).
}

# Filenames inside ${venv}/bin/ that must point at the layer-1.5 wrapper after
# install. Each is renamed to "<name>-real" and replaced with a copy of
# python_wrapper.sh so any agent invocation hits the wrapper regardless of the
# alias chosen.
_PYTHON_INTERPRETER_NAMES: tuple[str, ...] = (
    "python",
    "python3",
    f"python3.{sys.version_info.minor}",
)

_WRAPPER_SOURCE = Path(__file__).parent / "python_wrapper.sh"
_SITECUSTOMIZE_SOURCE = Path(__file__).parent / "sitecustomize_template.py"


def build_arm_template(arm: str, library_version: str, template_dir: Path) -> Path:
    """Build a per-arm venv template that can be cloned per run.

    Steps:
        1. Create a fresh venv at ``template_dir`` (no system site packages).
        2. ``pip install`` the arm library at the pinned version.
        3. Copy ``harness/sitecustomize_template.py`` into the venv's
           ``site-packages`` as ``sitecustomize.py``. Python's site machinery
           auto-loads this file on every interpreter start.
        4. Install the layer-1.5 ``python`` wrapper: rename
           ``${venv}/bin/python`` to ``python-real`` and replace with a copy
           of ``harness/python_wrapper.sh``. Repeat for ``python3`` and
           ``python3.X`` so every interpreter alias the agent might invoke
           hits the wrapper. The wrapper records an ``exec_python`` event to
           ``_PYRUNTIME_EVENT_LOG`` then ``exec``s the real binary.
        5. Return ``template_dir``.

    Order matters: shim install happens BEFORE wrapper install so the
    sysconfig probe in step 3 hits the real ``${venv}/bin/python``, not the
    wrapper (which would either skip telemetry write or hard-exit if the
    operator's ``_PYRUNTIME_EVENT_LOG`` is set).

    Args:
        arm: ``"diff_diff"`` or ``"statsmodels"``. Statsmodels deferred to PR #7.
        library_version: PyPI version string for the arm library
            (e.g., ``"3.3.2"`` for diff-diff).
        template_dir: where to materialize the template venv. Created if missing;
            its parent directory must exist.

    Returns:
        ``template_dir`` (the path to the materialized template venv).

    Raises:
        NotImplementedError: if ``arm == "statsmodels"`` (PR #7).
        ValueError: if ``arm`` is not a recognized arm name.
        subprocess.CalledProcessError: if pip install fails (network,
            version-not-found, etc.).
    """
    if arm == "statsmodels":
        raise NotImplementedError(
            "statsmodels arm: deferred to PR #7 (statsmodels arm instrumentation)"
        )
    if arm not in _ARM_TO_PIP_PACKAGE:
        raise ValueError(
            f"unknown arm {arm!r}; expected one of {sorted(_ARM_TO_PIP_PACKAGE) + ['statsmodels']}"
        )

    template_dir = Path(template_dir)
    template_dir.parent.mkdir(parents=True, exist_ok=True)

    venv.create(str(template_dir), with_pip=True, clear=False, symlinks=True)

    pip_package = _ARM_TO_PIP_PACKAGE[arm]
    subprocess.run(
        [
            str(template_dir / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            f"{pip_package}=={library_version}",
        ],
        check=True,
        capture_output=True,
    )

    _install_shim_into_venv(template_dir)
    _install_python_wrapper(template_dir)

    return template_dir


def clone_for_run(template_dir: Path, run_dir: Path) -> Path:
    """Clone a per-run venv from a template. Returns the new venv path.

    Phase 2 work; deferred to PR #6+ when eval volume justifies the
    template-and-clone optimization. Phase 1 (PR #5) builds a fresh venv per
    run via :func:`build_arm_template`; the per-run cost is acceptable for
    the ~30-run Phase 1 budget.
    """
    del template_dir, run_dir
    raise NotImplementedError(
        "venv_pool.clone_for_run is deferred to PR #6+ (Phase 2 template clone)"
    )


def _install_shim_into_venv(venv_path: Path) -> None:
    """Copy ``harness/sitecustomize_template.py`` into the venv's site-packages
    as ``sitecustomize.py``.

    Locates ``site-packages`` via ``sysconfig.get_paths()['purelib']``
    invoked through the venv's own python. Pass a clean env to the
    subprocess so the operator's ``_PYRUNTIME_EVENT_LOG`` (if set) does not
    leak into the build-time probe (which would hard-exit if the path is
    stale).

    Called BEFORE ``_install_python_wrapper`` so the sysconfig invocation
    here hits the real ``${venv}/bin/python``, not the wrapper.
    """
    probe = subprocess.run(
        [
            str(venv_path / "bin" / "python"),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    site_packages = Path(probe.stdout.strip())
    target = site_packages / "sitecustomize.py"
    shutil.copyfile(_SITECUSTOMIZE_SOURCE, target)


def _install_python_wrapper(venv_path: Path) -> None:
    """Replace each ``${venv}/bin/python*`` interpreter with the layer-1.5
    wrapper, preserving the real interpreter at ``${venv}/bin/python-real``.

    The wrapper computes its real-interpreter target as
    ``$(dirname "$0")/python-real`` so any of the renamed interpreters can
    invoke the same canonical real binary.

    Steps:
        1. For the first existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           resolve its absolute target (venvs typically symlink ``python`` ->
           ``python3.X`` -> the real system interpreter). Place the
           canonical ``python-real`` symlink at ``${venv}/bin/python-real``
           pointing at that absolute target.
        2. For every existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           remove the existing entry and write a fresh copy of the wrapper
           script at that path (``chmod +x``).
    """
    bin_dir = venv_path / "bin"
    real_target = bin_dir / "python-real"

    # Pass 1: pick the first existing interpreter and capture its real path.
    canonical_real: Path | None = None
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if original.exists() or original.is_symlink():
            canonical_real = Path(os.path.realpath(original))
            break

    if canonical_real is None:
        raise RuntimeError(
            f"no python interpreter found in {bin_dir!r}; venv build appears incomplete"
        )

    # Place python-real symlink (or skip if it already resolves correctly).
    if real_target.exists() or real_target.is_symlink():
        real_target.unlink()
    os.symlink(canonical_real, real_target)

    # Pass 2: replace each interpreter name with the wrapper script.
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if not (original.exists() or original.is_symlink()):
            continue
        original.unlink()
        shutil.copyfile(_WRAPPER_SOURCE, original)
        original.chmod(0o755)
