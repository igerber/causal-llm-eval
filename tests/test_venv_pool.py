"""Unit tests for ``harness/venv_pool.py``.

PR #5 implements ``build_arm_template`` (Phase 1 per-arm venv pool). These
tests assert the four installation steps fire correctly:

    1. Venv created at ``template_dir``.
    2. Arm library pip-installed at the pinned version.
    3. ``_pyruntime_shim.py`` + ``_pyruntime_shim.pth`` installed into
       ``site-packages`` (PR #6: replaces the prior ``sitecustomize.py``
       install which loses to Homebrew's stdlib-level sitecustomize on
       affected systems).
    4. Layer-1.5 ``python_wrapper.sh`` installed as ``python`` / ``python3``
       / ``python3.X``; original interpreter moved to
       ``${venv}/.pyruntime-real/python-real`` (off PATH).

The file is marked ``slow`` because each test session pays one ~30s venv
build cost (cached via the ``shared_venv`` session-scoped fixture). The 11
shared-venv tests then complete in <1s each. The 2 tests that don't need
a venv (NotImplementedError dispatch, deferred-stub semantics) are not
gated by the fixture but stay in this file for cohesion.

Run via: ``pytest -m slow tests/test_venv_pool.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness import venv_pool

pytestmark = pytest.mark.slow

_DIFF_DIFF_VERSION = "3.3.2"
_STATSMODELS_VERSION = "0.14.6"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def shared_venv(tmp_path_factory):
    """One venv built per pytest session, shared across most venv_pool tests.

    Each ``build_arm_template`` call takes ~10-30s (network bound by pip
    install). Sharing across tests cuts the file's total runtime from
    ~5.5 min (11 fresh builds) to ~36s (one build + per-test setup).

    Tests that need to assert build-from-scratch behavior use the
    per-function ``fresh_venv`` fixture instead.
    """
    template_dir = tmp_path_factory.mktemp("shared_venv")
    return venv_pool.build_arm_template("diff_diff", _DIFF_DIFF_VERSION, template_dir)


@pytest.fixture(scope="session")
def shared_statsmodels_venv(tmp_path_factory):
    """Session-scoped statsmodels-arm venv. PR #7: validates that
    ``build_arm_template`` works for ``arm="statsmodels"`` end-to-end.

    Cost: ~20-30s for one statsmodels wheel install (numpy + scipy +
    patsy are already wheels). Shared across the 3 statsmodels-arm
    tests so the per-file slow budget stays bounded.
    """
    template_dir = tmp_path_factory.mktemp("shared_statsmodels_venv")
    return venv_pool.build_arm_template("statsmodels", _STATSMODELS_VERSION, template_dir)


def _site_packages(venv_path: Path) -> Path:
    """Return the venv's site-packages directory.

    Test-only introspection: invokes ``.actual-python`` with ``-S`` to
    skip ``site.py`` (and therefore ``sitecustomize.py``) since this is
    a build-time probe, not an agent run. Without ``-S`` the sitecustomize
    in the venv would hard-exit with code 2 because
    ``_PYRUNTIME_EVENT_LOG`` is unset (correct production behavior; not
    what we want for a sysconfig query).
    """
    probe = subprocess.run(
        [
            str(venv_path / ".pyruntime-real" / ".actual-python"),
            "-S",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(probe.stdout.strip())


# ---------------------------------------------------------------------------
# Tests that share the session-scoped venv
# ---------------------------------------------------------------------------


def test_build_arm_template_creates_venv_with_python_executable(shared_venv):
    assert (shared_venv / "bin" / "python").exists()
    assert (shared_venv / ".pyruntime-real" / ".actual-python").exists()


def test_build_arm_template_installs_correct_library_version(shared_venv):
    # Test-only introspection: -S bypasses sitecustomize since this
    # build-time probe runs without _PYRUNTIME_EVENT_LOG. See
    # _site_packages() docstring above.
    probe = subprocess.run(
        [
            str(shared_venv / ".pyruntime-real" / ".actual-python"),
            "-S",
            "-c",
            "import diff_diff; print(diff_diff.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == _DIFF_DIFF_VERSION


def test_build_arm_template_installs_pyruntime_shim_into_site_packages(shared_venv):
    """PR #6: shim is installed as ``_pyruntime_shim.py`` + ``_pyruntime_shim.pth``
    rather than ``sitecustomize.py``. The .pth-based load survives Homebrew's
    stdlib-level ``sitecustomize.py`` (which would otherwise shadow our
    ``sitecustomize.py`` in venv site-packages because stdlib comes before
    site-packages in sys.path).
    """
    site_packages = _site_packages(shared_venv)
    shim = site_packages / "_pyruntime_shim.py"
    pth = site_packages / "_pyruntime_shim.pth"
    assert shim.exists(), f"_pyruntime_shim.py missing under {site_packages}"
    assert pth.exists(), f"_pyruntime_shim.pth missing under {site_packages}"
    source = Path(__file__).parent.parent / "harness" / "sitecustomize_template.py"
    assert _file_sha256(shim) == _file_sha256(source)
    assert pth.read_text() == "import _pyruntime_shim\n"
    # Defensive: the legacy sitecustomize.py path MUST NOT be installed
    # alongside the .pth path (would double-instrument and double-emit
    # session_start events).
    assert not (
        site_packages / "sitecustomize.py"
    ).exists(), "sitecustomize.py should not be installed; .pth-based load is the only path"


def test_build_arm_template_installs_wrapper_for_python_python3_python3X(shared_venv):
    """Every python alias in the venv is the wrapper script, not the real
    interpreter.
    """
    minor = sys.version_info.minor
    for name in ("python", "python3", f"python3.{minor}"):
        path = shared_venv / "bin" / name
        if not path.exists():
            # Some aliases may not exist on every platform; skip those.
            continue
        # The wrapper starts with a shebang line we control; the real python
        # binary does not.
        first_line = path.read_text(errors="replace").splitlines()[0]
        assert first_line.startswith(
            "#!/bin/sh"
        ), f"{path} does not start with the wrapper shebang; first_line={first_line!r}"


def test_build_arm_template_wrapper_execs_python_real(shared_venv, tmp_path):
    """The wrapper transparently passes argv through the strip-S shim to
    the actual Python binary and the interpreter still produces correct
    output. ``_PYRUNTIME_EVENT_LOG`` must be set so sitecustomize loads
    successfully (otherwise the shim's reachability matters but
    sitecustomize hard-exits with code 2 before user code runs).
    """
    log_path = tmp_path / "events.jsonl"
    probe = subprocess.run(
        [str(shared_venv / "bin" / "python"), "-c", "print('ok')"],
        check=True,
        env={
            "PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
        text=True,
    )
    assert probe.stdout == "ok\n"


def test_build_arm_template_wrapper_emits_exec_python_event(shared_venv, tmp_path):
    log_path = tmp_path / "events.jsonl"
    subprocess.run(
        [str(shared_venv / "bin" / "python"), "-c", "pass"],
        check=True,
        env={
            "PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
    )
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert len(exec_events) == 1, events
    evt = exec_events[0]
    # Wrapper records ``argv = [basename($0), $@]``. When invoked as
    # ``${venv}/bin/python -c "pass"``, basename($0) is "python" and $@ is
    # ["-c", "pass"], so argv = ["python", "-c", "pass"]. The argv[1:]
    # tokens are the load-bearing match key the merger uses.
    assert evt["argv"] == ["python", "-c", "pass"]
    # Wrapper records the path it execs (the strip-S shim at python-real,
    # not the .actual-python binary one layer deeper).
    assert os.path.normpath(evt["executable"]) == str(
        shared_venv / ".pyruntime-real" / "python-real"
    )
    assert isinstance(evt["pid"], int)
    assert isinstance(evt["ppid"], int)


def test_build_arm_template_wrapper_skips_when_env_unset(shared_venv, tmp_path):
    """When ``_PYRUNTIME_EVENT_LOG`` is unset, the wrapper skips its event
    write but still execs the real interpreter. (Note: sitecustomize ALSO
    fires on the exec'd interpreter, and it WILL hard-exit if the env var
    is unset, so this test asserts the wrapper-level skip, expecting the
    overall invocation to fail with exit code 2 from sitecustomize.)
    """
    log_path = tmp_path / "events.jsonl"
    # No _PYRUNTIME_EVENT_LOG in env; wrapper skips write; sitecustomize
    # hard-exits with code 2.
    result = subprocess.run(
        [str(shared_venv / "bin" / "python"), "-c", "pass"],
        env={"PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin"},
        capture_output=True,
    )
    # Wrapper skipped its write; log was never created.
    assert not log_path.exists()
    # Sitecustomize hard-exited (code 2) because env var is unset.
    assert result.returncode == 2
    assert b"_PYRUNTIME_EVENT_LOG is unset" in result.stderr


def test_build_arm_template_wrapper_hard_exits_on_unwritable_log(shared_venv, tmp_path):
    """When ``_PYRUNTIME_EVENT_LOG`` points at an unwritable path, the
    wrapper exits with code 2 BEFORE exec'ing the real interpreter.
    """
    # Create a read-only directory; pointing the log at a child of it makes
    # the append fail at write time.
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    os.chmod(readonly_dir, 0o500)
    try:
        log_path = readonly_dir / "events.jsonl"
        result = subprocess.run(
            [str(shared_venv / "bin" / "python"), "-c", "pass"],
            env={
                "PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin",
                "_PYRUNTIME_EVENT_LOG": str(log_path),
            },
            capture_output=True,
        )
        assert result.returncode == 2
        assert b"[pyruntime-wrapper] cannot append" in result.stderr
    finally:
        os.chmod(readonly_dir, 0o700)


def test_wrapper_argv_with_special_chars(shared_venv, tmp_path):
    """Argv containing quotes, backslashes, and tabs JSON-encodes cleanly."""
    log_path = tmp_path / "events.jsonl"
    weird = 'has"quote\\backslash\there'
    subprocess.run(
        [str(shared_venv / "bin" / "python"), "-c", "import sys; sys.exit(0)", weird],
        check=True,
        env={
            "PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
    )
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert len(exec_events) == 1
    # argv[0] is basename($0) == "python"; remaining elements are $@.
    assert exec_events[0]["argv"] == ["python", "-c", "import sys; sys.exit(0)", weird]


def test_wrapper_argv_with_newlines_fails_closed(shared_venv, tmp_path):
    """PR #5 R0 P2 #7: the line-oriented awk encoder cannot preserve
    embedded newlines in argv, so the wrapper fails closed (exit 2)
    rather than emit corrupted attestation. POSIX argv may legally
    contain newlines but realistic agent invocations never do.
    """
    log_path = tmp_path / "events.jsonl"
    multiline = "line1\nline2"
    result = subprocess.run(
        [str(shared_venv / "bin" / "python"), "-c", "pass", multiline],
        env={
            "PATH": f"{shared_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
    )
    assert result.returncode == 2
    assert b"argv contains embedded newline" in result.stderr


# ---------------------------------------------------------------------------
# Tests that do NOT need a built venv
# ---------------------------------------------------------------------------


def _direct_venv_site_packages(venv_path: Path) -> Path:
    """Resolve the venv's site-packages by direct path lookup.

    Bypasses the ``.actual-python -S`` ``sysconfig`` probe used by
    ``_site_packages``: on Python 3.13 ``-S`` skips ``pyvenv.cfg``
    processing entirely, so ``sys.prefix`` resolves to the base
    interpreter, not the venv. (Pre-existing bug in ``_site_packages``;
    tracked separately. The venv layout is standardized by
    ``venv.create``, so direct path lookup is robust.)
    """
    minor = sys.version_info.minor
    return venv_path / "lib" / f"python3.{minor}" / "site-packages"


def test_build_arm_template_statsmodels_installs_correct_library_version(
    shared_statsmodels_venv,
):
    """PR #7: ``arm="statsmodels"`` builds a venv with the pinned
    statsmodels version installed and importable.

    Uses the venv's wrapper ``bin/python`` rather than ``-S`` invocation
    (which doesn't see venv site-packages on Python 3.13). Sets
    ``_PYRUNTIME_EVENT_LOG`` to a tmp file so the shim doesn't hard-exit.
    """
    log_path = shared_statsmodels_venv.parent / "version_probe_events.jsonl"
    log_path.touch()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "_PYRUNTIME_EVENT_LOG": str(log_path),
    }
    result = subprocess.run(
        [
            str(shared_statsmodels_venv / "bin" / "python"),
            "-c",
            "import statsmodels; print(statsmodels.__version__)",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == _STATSMODELS_VERSION


def test_build_arm_template_statsmodels_installs_pyruntime_shim(shared_statsmodels_venv):
    """The same shim install path (``_pyruntime_shim.py`` + ``_pyruntime_shim.pth``)
    fires for the statsmodels arm as for diff_diff."""
    site_packages = _direct_venv_site_packages(shared_statsmodels_venv)
    shim_py = site_packages / "_pyruntime_shim.py"
    shim_pth = site_packages / "_pyruntime_shim.pth"
    assert shim_py.exists(), f"{shim_py} missing — shim not installed in statsmodels-arm venv"
    assert shim_pth.exists(), f"{shim_pth} missing — .pth load not installed"
    assert shim_pth.read_text() == "import _pyruntime_shim\n"


def test_build_arm_template_statsmodels_shim_records_module_import(
    shared_statsmodels_venv, tmp_path
):
    """End-to-end attestation: invoking the venv's python with the event-log
    env-var set causes the statsmodels post-import hook to record a
    ``module_import/module=statsmodels`` event when statsmodels is imported.
    """
    log_path = tmp_path / "events.jsonl"
    log_path.touch()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "_PYRUNTIME_EVENT_LOG": str(log_path),
    }
    subprocess.run(
        [
            str(shared_statsmodels_venv / "bin" / "python"),
            "-c",
            "import statsmodels",
        ],
        check=True,
        capture_output=True,
        env=env,
        cwd=str(tmp_path),
    )
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    module_imports = [
        e for e in events if e.get("event") == "module_import" and e.get("module") == "statsmodels"
    ]
    assert (
        len(module_imports) >= 1
    ), f"expected at least one module_import/statsmodels event, got {events}"


def test_build_arm_template_unknown_arm_raises(tmp_path):
    """Unknown arm name surfaces a clear ``ValueError``."""
    with pytest.raises(ValueError, match="unknown arm"):
        venv_pool.build_arm_template("nope", _DIFF_DIFF_VERSION, tmp_path / "venv")


def test_clone_for_run_not_implemented(tmp_path):
    """``clone_for_run`` is a Phase 2 stub deferred to PR #6+."""
    with pytest.raises(NotImplementedError, match="PR #6"):
        venv_pool.clone_for_run(tmp_path / "template", tmp_path / "run")
