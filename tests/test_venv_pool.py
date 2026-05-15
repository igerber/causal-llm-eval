"""Unit tests for ``harness/venv_pool.py``.

PR #5 implements ``build_arm_template`` (Phase 1 per-arm venv pool). These
tests assert the four installation steps fire correctly:

    1. Venv created at ``template_dir``.
    2. Arm library pip-installed at the pinned version.
    3. ``sitecustomize.py`` copied into ``site-packages``.
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


def _site_packages(venv_path: Path) -> Path:
    """Return the venv's site-packages directory."""
    probe = subprocess.run(
        [
            str(venv_path / ".pyruntime-real" / ".actual-python"),
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
    probe = subprocess.run(
        [
            str(shared_venv / ".pyruntime-real" / ".actual-python"),
            "-c",
            "import diff_diff; print(diff_diff.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == _DIFF_DIFF_VERSION


def test_build_arm_template_copies_sitecustomize_into_site_packages(shared_venv):
    site_packages = _site_packages(shared_venv)
    target = site_packages / "sitecustomize.py"
    assert target.exists()
    source = Path(__file__).parent.parent / "harness" / "sitecustomize_template.py"
    assert _file_sha256(target) == _file_sha256(source)


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
            "#!/usr/bin/env sh"
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


def test_build_arm_template_statsmodels_arm_not_implemented(tmp_path):
    """``arm="statsmodels"`` raises ``NotImplementedError`` with a PR #7
    pointer (no venv built).
    """
    with pytest.raises(NotImplementedError, match="PR #7"):
        venv_pool.build_arm_template("statsmodels", _DIFF_DIFF_VERSION, tmp_path / "venv")


def test_build_arm_template_unknown_arm_raises(tmp_path):
    """Unknown arm name surfaces a clear ``ValueError``."""
    with pytest.raises(ValueError, match="unknown arm"):
        venv_pool.build_arm_template("nope", _DIFF_DIFF_VERSION, tmp_path / "venv")


def test_clone_for_run_not_implemented(tmp_path):
    """``clone_for_run`` is a Phase 2 stub deferred to PR #6+."""
    with pytest.raises(NotImplementedError, match="PR #6"):
        venv_pool.clone_for_run(tmp_path / "template", tmp_path / "run")
