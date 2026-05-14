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

    # symlinks=False: copy the interpreter rather than symlinking. Closes the
    # operator-home exposure vector where a symlinked ``python-real`` would
    # resolve to ``/opt/homebrew/Cellar/...`` / ``~/.pyenv/...`` / etc.,
    # leaking the operator's interpreter location to the agent.
    venv.create(str(template_dir), with_pip=True, clear=False, symlinks=False)

    # Scrubbed env for pip install: strip operator-set PIP_* / PYTHON*
    # / HOME so the install is reproducible regardless of operator
    # configuration. PIP_CONFIG_FILE=os.devnull disables operator pip
    # config; PIP_DISABLE_PIP_VERSION_CHECK silences the version notice.
    pip_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HOME": str(template_dir),
    }
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
        env=pip_env,
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


_REAL_INTERPRETER_DIRNAME = ".pyruntime-real"
_REAL_INTERPRETER_FILENAME = "python-real"


def _install_python_wrapper(venv_path: Path) -> None:
    """Replace each ``${venv}/bin/python*`` interpreter with the layer-1.5
    wrapper, preserving the real interpreter at
    ``${venv}/.pyruntime-real/python-real``.

    The real interpreter lives OUTSIDE ``bin/`` so it does not appear on
    the agent's PATH. An agent's ``python-real`` typed on the command
    line gets "command not found"; absolute-path invocations
    (``${venv}/.pyruntime-real/python-real``) are catchable via
    filesystem-scan defenses but are not on the easy bypass path.

    The wrapper computes its target as
    ``$(dirname "$0")/../.pyruntime-real/python-real``.

    Steps:
        1. For the first existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           materialize the real interpreter at
           ``${venv}/.pyruntime-real/python-real``.
           - If the existing entry is a symlink, the canonical real path is
             ``os.path.realpath()`` of it; symlink ``python-real`` -> that
             absolute path.
           - If the existing entry is a real file copy (``symlinks=False``
             in ``venv.create``), MOVE it to ``python-real``. We cannot
             symlink-to-self because the next step overwrites the
             original.
        2. For every existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           remove the existing entry and write a fresh copy of the wrapper
           script at that path (``chmod +x``).
    """
    bin_dir = venv_path / "bin"
    real_dir = venv_path / _REAL_INTERPRETER_DIRNAME
    real_dir.mkdir(parents=True, exist_ok=True)
    real_target = real_dir / _REAL_INTERPRETER_FILENAME

    # Pass 1: find the first existing interpreter and materialize python-real.
    materialized = False
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if not (original.exists() or original.is_symlink()):
            continue
        if real_target.exists() or real_target.is_symlink():
            real_target.unlink()
        if original.is_symlink():
            # symlinks=True venv path: original is a symlink chain to the
            # real system interpreter. Symlink python-real to the absolute
            # realpath target so it survives the next overwrite.
            canonical_real = Path(os.path.realpath(original))
            os.symlink(canonical_real, real_target)
        else:
            # symlinks=False venv path: original IS the real binary copy.
            # MOVE it (rename) to python-real so it survives the next
            # overwrite. A symlink would dangle once the original is
            # replaced with the wrapper.
            original.rename(real_target)
        materialized = True
        break

    if not materialized:
        raise RuntimeError(
            f"no python interpreter found in {bin_dir!r}; venv build appears incomplete"
        )

    # Pass 2: replace each interpreter name with the wrapper script.
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if original.exists() or original.is_symlink():
            original.unlink()
        shutil.copyfile(_WRAPPER_SOURCE, original)
        original.chmod(0o755)
