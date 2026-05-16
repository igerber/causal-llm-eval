"""Per-arm venv management for cold-start runs.

For Phase 1 (30 runs total + calibration + dry-pass), a fresh venv per run is
acceptable. For Phase 2 (~1500+ runs), per-arm venv templates are pre-built once
and cloned per run; never mutated post-clone.

The shape of this module is set in Phase 0 so Phase 2 doesn't require rewrites.

``build_arm_template`` builds a fresh venv per run end-to-end: create venv,
pip install the arm library at the pinned version, copy the sitecustomize
shim into site-packages, install the layer-1.5 ``python`` wrapper. Both
arms (diff_diff, statsmodels) are supported as of PR #7. ``clone_for_run``
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
    "statsmodels": "statsmodels",
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
           ``site-packages`` as ``_pyruntime_shim.py`` and write a
           ``_pyruntime_shim.pth`` next to it (PR #6 fix; see
           ``_install_shim_into_venv`` for the rationale on why .pth-based
           load is required on Homebrew Python). Python's site machinery
           processes the .pth file during site init; our shim loads even
           when the operator's system Python ships its own
           stdlib-level ``sitecustomize.py``.
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
        arm: ``"diff_diff"`` or ``"statsmodels"``.
        library_version: PyPI version string for the arm library
            (e.g., ``"3.3.2"`` for diff-diff; ``"0.14.6"`` for statsmodels).
        template_dir: where to materialize the template venv. Created if missing;
            its parent directory must exist.

    Returns:
        ``template_dir`` (the path to the materialized template venv).

    Raises:
        ValueError: if ``arm`` is not a recognized arm name.
        subprocess.CalledProcessError: if pip install fails (network,
            version-not-found, etc.).
    """
    if arm not in _ARM_TO_PIP_PACKAGE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_ARM_TO_PIP_PACKAGE)}")

    template_dir = Path(template_dir)
    template_dir.parent.mkdir(parents=True, exist_ok=True)

    # symlinks=False: copy the interpreter rather than symlinking. Closes the
    # operator-home exposure vector where a symlinked ``python-real`` would
    # resolve to ``/opt/homebrew/Cellar/...`` / ``~/.pyenv/...`` / etc.,
    # leaking the operator's interpreter location to the agent.
    # PR #5 R8 P2: clear=True so direct callers (not just run_one) can't
    # accidentally reuse stale venv state. run_one passes a fresh tmpdir
    # so this is a no-op there; defensive for any future caller.
    venv.create(str(template_dir), with_pip=True, clear=True, symlinks=False)

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


_SHIM_INSTALL_NAME = "_pyruntime_shim"
_SHIM_PTH_CONTENT = "import _pyruntime_shim\n"


def _install_shim_into_venv(venv_path: Path) -> None:
    """Copy ``harness/sitecustomize_template.py`` into the venv's site-packages
    as ``_pyruntime_shim.py`` and write a ``_pyruntime_shim.pth`` next to it.

    The .pth file's ``import _pyruntime_shim`` line is processed by Python's
    site machinery during initialization (BEFORE ``execsitecustomize`` runs),
    so our shim loads regardless of whether the operator's system Python has
    its own ``sitecustomize.py``. Homebrew's ``python@3.13`` (Feb 2026+)
    ships a stdlib-level ``sitecustomize.py`` that would otherwise shadow
    any ``sitecustomize.py`` we install in venv site-packages, because
    stdlib comes before site-packages in sys.path. The .pth approach is the
    canonical workaround (used by coverage.py and pytest-cov).

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
    shim_target = site_packages / f"{_SHIM_INSTALL_NAME}.py"
    pth_target = site_packages / f"{_SHIM_INSTALL_NAME}.pth"
    shutil.copyfile(_SITECUSTOMIZE_SOURCE, shim_target)
    pth_target.write_text(_SHIM_PTH_CONTENT)


_REAL_INTERPRETER_DIRNAME = ".pyruntime-real"
_REAL_INTERPRETER_FILENAME = "python-real"
# The actual CPython binary lives one layer deeper. python-real is a tiny
# shell script that strips ``-S`` (and any compact form like ``-Sc``,
# ``-IS``) from argv before execing the real binary, so sitecustomize
# always loads regardless of how python-real was invoked. PR #5 R3 P0:
# the strip-S step structurally closes the ``python-real -S script.py``
# direct-bypass class.
_ACTUAL_INTERPRETER_FILENAME = ".actual-python"

_PYTHON_REAL_STRIP_S_SCRIPT = """\
#!/bin/sh
# Layer-1.5 strip-S shim. Sits between the wrapper (or any direct
# invoker) and the real CPython binary at
# ``${VENV}/.pyruntime-real/.actual-python``. Strips ``-S`` (and compact
# forms like ``-Sc`` / ``-IS``) from the pre-script argv so sitecustomize
# always loads. POSIX shell, no bashisms.
#
# Argv preservation: rebuilds positional parameters via ``set --`` rather
# than string flattening, so quoted args containing spaces / globs / tabs
# pass through unchanged.
#
# PR #5 R9 P0: pin internal command resolution to /usr/bin:/bin so
# ``dirname`` / ``sed`` cannot be hijacked by agent files in
# ``${venv}/bin/``. Restore the agent's PATH before exec.
agent_path="${PATH-}"
PATH="/usr/bin:/bin"
export PATH
actual="$(dirname "$0")/.actual-python"
seen_script=0
orig_count=$#
i=0
while [ $i -lt $orig_count ]; do
    a="$1"
    shift
    if [ $seen_script -eq 0 ]; then
        case "$a" in
            --)
                # End of options marker; everything after is the script + args.
                seen_script=1
                set -- "$@" "$a"
                i=$((i + 1))
                continue
                ;;
            -c|-m)
                # -c CODE / -m MODULE: next argv element is the script
                # payload. After this element+arg, remaining args are
                # sys.argv to the script and must NOT be S-stripped.
                set -- "$@" "$a"
                if [ $i -lt $((orig_count - 1)) ]; then
                    next="$1"
                    shift
                    set -- "$@" "$next"
                    i=$((i + 2))
                else
                    i=$((i + 1))
                fi
                seen_script=1
                continue
                ;;
            -c*|-m*)
                # Attached form -cCODE / -mMODULE: rest of the element
                # is the script payload, no extra consumption.
                set -- "$@" "$a"
                i=$((i + 1))
                seen_script=1
                continue
                ;;
            -W|-X)
                # -W FILTER / -X OPTION (separate form): consumes next
                # argv element verbatim; flag scanning continues.
                set -- "$@" "$a"
                if [ $i -lt $((orig_count - 1)) ]; then
                    next="$1"
                    shift
                    set -- "$@" "$next"
                    i=$((i + 2))
                else
                    i=$((i + 1))
                fi
                continue
                ;;
            -W*|-X*)
                # Attached form -Werror::SomeWarning / -Xdev: payload
                # is in the same element. Pass through without scanning
                # the payload for ``S``.
                set -- "$@" "$a"
                i=$((i + 1))
                continue
                ;;
            -*S*)
                # Pre-script flag containing S in any position
                # (-S, -Sc, -IS, etc.). Strip S characters; evaluate.
                stripped=$(printf '%s' "$a" | sed 's/S//g')
                if [ "$stripped" = "-" ]; then
                    # Token was just -S (or -SSS, etc.); drop entirely.
                    i=$((i + 1))
                    continue
                fi
                set -- "$@" "$stripped"
                i=$((i + 1))
                continue
                ;;
            -*)
                # Other flag without S: pass through.
                set -- "$@" "$a"
                i=$((i + 1))
                continue
                ;;
            *)
                # First non-flag word (script path) ends pre-script region.
                seen_script=1
                ;;
        esac
    fi
    set -- "$@" "$a"
    i=$((i + 1))
done
PATH="$agent_path"
export PATH
exec "$actual" "$@"
"""


def _install_python_wrapper(venv_path: Path) -> None:
    """Replace each ``${venv}/bin/python*`` interpreter with the layer-1.5
    wrapper. Two-stage hidden-real-interpreter layout:

        ``${venv}/bin/python``           -> wrapper script (logs exec_python,
                                            execs python-real)
        ``${venv}/.pyruntime-real/python-real``
                                          -> strip-S shim script (always
                                             loads sitecustomize)
        ``${venv}/.pyruntime-real/.actual-python``
                                          -> real CPython binary

    The strip-S shim closes the ``python-real -S script.py`` direct-
    bypass class structurally: any invocation that reaches python-real
    (via wrapper or direct exec) has -S stripped before reaching the
    real interpreter, so sitecustomize always loads.

    The hidden ``.actual-python`` is the single remaining path that
    could bypass everything if invoked directly with -S; substring
    detection catches references to ``.actual-python`` in visible Bash
    commands.

    Steps:
        1. For the first existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           materialize the real interpreter at
           ``${venv}/.pyruntime-real/.actual-python``.
        2. Write the strip-S shim at
           ``${venv}/.pyruntime-real/python-real``.
        3. For every existing name in :data:`_PYTHON_INTERPRETER_NAMES`,
           remove the existing entry and write a fresh copy of the wrapper
           script at that path (``chmod +x``).
    """
    bin_dir = venv_path / "bin"
    real_dir = venv_path / _REAL_INTERPRETER_DIRNAME
    real_dir.mkdir(parents=True, exist_ok=True)
    actual_target = real_dir / _ACTUAL_INTERPRETER_FILENAME
    python_real_target = real_dir / _REAL_INTERPRETER_FILENAME

    # Pass 1: find the first existing interpreter and materialize
    # .actual-python (the real CPython binary).
    materialized = False
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if not (original.exists() or original.is_symlink()):
            continue
        if actual_target.exists() or actual_target.is_symlink():
            actual_target.unlink()
        if original.is_symlink():
            canonical_real = Path(os.path.realpath(original))
            os.symlink(canonical_real, actual_target)
        else:
            original.rename(actual_target)
        materialized = True
        break

    if not materialized:
        raise RuntimeError(
            f"no python interpreter found in {bin_dir!r}; venv build appears incomplete"
        )

    # Pass 2: install the strip-S shim at python-real. The shim drops -S
    # (and compact forms) before exec'ing .actual-python, so sitecustomize
    # always loads regardless of how python-real was invoked.
    if python_real_target.exists() or python_real_target.is_symlink():
        python_real_target.unlink()
    python_real_target.write_text(_PYTHON_REAL_STRIP_S_SCRIPT)
    python_real_target.chmod(0o755)

    # Pass 3: replace each interpreter name with the wrapper script.
    for name in _PYTHON_INTERPRETER_NAMES:
        original = bin_dir / name
        if original.exists() or original.is_symlink():
            original.unlink()
        shutil.copyfile(_WRAPPER_SOURCE, original)
        original.chmod(0o755)
