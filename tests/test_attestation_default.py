"""Default-CI attestation wiring test.

PR #5 R14 P2 (DT-1): the slow + live test suites cover the full
``build_arm_template`` -> sentinel -> ``run_one`` chain end-to-end, but
default ``pytest`` excludes both. A regression in the wrapper-shim wiring
(e.g. wrong site-packages target, broken strip-S shim, missing chmod,
wrong shebang) could pass default CI silently and only fire on the slow
or live job.

This file closes that gap with CHEAP default tests:

    1. Build a fresh venv via ``venv.create`` (no pip install -> no
       network, no ~30s cost).
    2. Call ``_install_shim_into_venv`` and ``_install_python_wrapper``
       to install the production shim + wrapper into the bare venv.
    3. Run ``${venv}/bin/python -c "pass"`` with ``_PYRUNTIME_EVENT_LOG``
       set, exactly as the runner's build-time sentinel does.
    4. Assert wiring invariants the runner's post-sentinel attestation
       check would catch.

**Layer-2 sitecustomize coverage caveat (HISTORICAL — see PR #6)**:
this test was originally written when the layer-2 shim was installed
as ``sitecustomize.py`` in the venv's site-packages. On systems where
the host Python ships its own stdlib-level ``sitecustomize.py`` (e.g.
Homebrew Python on macOS), that system file shadows venv-site-packages
sitecustomize because stdlib comes before site-packages in ``sys.path``;
our layer-2 events would not fire. The two layer-2 tests below
auto-skip on shadowed systems for that historical reason.

PR #6 changed the shim install to ``_pyruntime_shim.py`` +
``_pyruntime_shim.pth``, which Python's site machinery loads via
``addsitepackages()`` regardless of whether stdlib has its own
``sitecustomize.py``. The shadow-detection skip is therefore now
OVER-CAUTIOUS — the shim would actually load successfully on systems
the test currently skips. The skip is preserved here as conservative
behavior; revisit by removing the skip and asserting layer-2 fires
unconditionally (TODO; tracked separately).

Cost: ~1-2s per test (venv create, no pip install). Default-CI
compatible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from harness.venv_pool import _install_python_wrapper, _install_shim_into_venv


def _detect_shadowing_sitecustomize(venv_root: Path) -> str | None:
    """Return the path of any ``sitecustomize.py`` on the venv's
    sys.path that lives OUTSIDE ``venv_root`` (i.e., a system-level
    file that would shadow our venv-installed shim). Returns None on
    a clean system.
    """
    probe = subprocess.run(
        [
            str(venv_root / ".pyruntime-real" / ".actual-python"),
            "-c",
            "import sys, os\n"
            "for p in sys.path:\n"
            "    if not p: continue\n"
            "    fp = os.path.join(p, 'sitecustomize.py')\n"
            "    if os.path.exists(fp): print(fp); break\n",
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    found = probe.stdout.strip()
    if not found:
        return None
    venv_str = str(venv_root.resolve())
    if Path(found).resolve().is_relative_to(Path(venv_str)):
        return None  # the one we installed is the only candidate -> not shadowed
    return found


@pytest.fixture
def bare_venv(tmp_path):
    """Build a bare venv (no pip install), then install the shim +
    wrapper into it. Returns the venv root.

    Mirrors what ``build_arm_template`` does EXCEPT the pip install
    step, which is the only slow + network-dependent part.
    """
    venv_root = tmp_path / "venv"
    venv.create(str(venv_root), with_pip=False, clear=True, symlinks=False)
    _install_shim_into_venv(venv_root)
    _install_python_wrapper(venv_root)
    return venv_root


def _sentinel_invocation(venv_root: Path, log_path: Path) -> subprocess.CompletedProcess:
    """Run ``python -c "pass"`` against the venv, mirroring the
    runner's build-time sentinel call shape."""
    return subprocess.run(
        [str(venv_root / "bin" / "python"), "-c", "pass"],
        env={
            "PATH": f"{venv_root / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
        timeout=30,
    )


def _read_events(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


# -- Layer-1.5 wrapper coverage (always runs) ---------------------------------


def test_default_attestation_wrapper_emits_exec_python(bare_venv, tmp_path):
    """The wrapper installed by ``_install_python_wrapper`` must run
    against a freshly-built venv and emit a well-formed exec_python
    event. Catches: missing chmod, wrong shebang, broken strip-S
    shim, wrong python-real layout.
    """
    log_path = tmp_path / "events.jsonl"
    log_path.touch()

    result = _sentinel_invocation(bare_venv, log_path)
    assert result.returncode == 0, (
        f"sentinel exit={result.returncode}, stderr={result.stderr!r}, " f"stdout={result.stdout!r}"
    )

    events = _read_events(log_path)
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert exec_events, f"no exec_python event; events={events!r}"


def test_default_attestation_wrapper_records_runner_pid_as_ppid(bare_venv, tmp_path):
    """The exec_python event's ``ppid`` must equal the test process pid
    (which spawns the wrapper as a direct child). This is the invariant
    the runner's post-sentinel check uses to identify sentinel events.
    """
    log_path = tmp_path / "events.jsonl"
    log_path.touch()
    runner_pid = os.getpid()

    result = _sentinel_invocation(bare_venv, log_path)
    assert result.returncode == 0
    events = _read_events(log_path)
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert exec_events
    assert exec_events[0]["ppid"] == runner_pid, (
        f"exec_python ppid={exec_events[0]['ppid']!r} != runner_pid={runner_pid!r}; "
        f"the wrapper did not see the runner as its parent"
    )


def test_default_attestation_wrapper_records_real_interpreter_executable(bare_venv, tmp_path):
    """The exec_python event's ``executable`` must point at the venv's
    ``.pyruntime-real/python-real``, the merger's venv-root allowlist
    target. Catches: ``$0`` unresolved, executable pointing at the
    wrapper itself, or pointing outside the venv root.
    """
    log_path = tmp_path / "events.jsonl"
    log_path.touch()

    result = _sentinel_invocation(bare_venv, log_path)
    assert result.returncode == 0
    events = _read_events(log_path)
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    assert exec_events
    expected = os.path.normpath(str(bare_venv / ".pyruntime-real" / "python-real"))
    actual = os.path.normpath(exec_events[0]["executable"])
    assert actual == expected, (
        f"exec_python.executable={actual!r} does not match expected "
        f"venv-root-anchored real interpreter path {expected!r}"
    )


# -- Layer-2 sitecustomize coverage (skip when shadowed) ----------------------


def test_default_attestation_full_chain_emits_session_start_and_end(bare_venv, tmp_path):
    """End-to-end: bare venv + shim + wrapper -> sentinel-shaped python
    invocation -> events log contains session_start + session_end with
    the same pid as exec_python. Auto-skips on systems where a host
    ``sitecustomize.py`` shadows the one in the venv's site-packages
    (e.g. Homebrew Python on macOS); the layer-1.5 tests above still
    cover the wrapper independently, and the live + slow suites cover
    layer 2 against an arm-installed venv.
    """
    shadowing = _detect_shadowing_sitecustomize(bare_venv)
    if shadowing is not None:
        pytest.skip(
            f"host Python ships its own sitecustomize at {shadowing!r}, "
            f"which shadows our venv-installed shim. The layer-1.5 wrapper "
            f"tests above still verify wrapper wiring; layer-2 coverage "
            f"runs in tests/test_telemetry_live.py and the slow suite."
        )
    log_path = tmp_path / "events.jsonl"
    log_path.touch()

    result = _sentinel_invocation(bare_venv, log_path)
    assert result.returncode == 0, f"sentinel exit={result.returncode}, stderr={result.stderr!r}"
    events = _read_events(log_path)
    exec_events = [e for e in events if e.get("event") == "exec_python"]
    starts = [e for e in events if e.get("event") == "session_start"]
    ends = [e for e in events if e.get("event") == "session_end"]
    assert exec_events
    assert starts, f"no session_start; layer-2 sitecustomize did not load. events={events!r}"
    assert ends, f"no session_end; atexit hook did not fire. events={events!r}"

    sentinel_pid = exec_events[0]["pid"]
    assert any(e.get("pid") == sentinel_pid for e in starts), (
        f"no session_start with pid={sentinel_pid}; pid chain broken between "
        f"layer-1.5 and layer-2. starts={starts!r}"
    )
    assert any(e.get("pid") == sentinel_pid for e in ends), (
        f"no session_end with pid={sentinel_pid}; atexit ran but did not "
        f"emit a matching pid. ends={ends!r}"
    )


def test_default_attestation_sitecustomize_hard_exits_when_env_unset(bare_venv):
    """With ``_PYRUNTIME_EVENT_LOG`` unset, sitecustomize hard-exits
    (returncode 2) so an agent's python subprocess cannot run without
    telemetry. Auto-skips on shadowed systems where our sitecustomize
    is not the one loaded.
    """
    shadowing = _detect_shadowing_sitecustomize(bare_venv)
    if shadowing is not None:
        pytest.skip(
            f"host Python ships its own sitecustomize at {shadowing!r}, "
            f"shadowing the venv shim; cannot assert hard-exit behavior."
        )
    result = subprocess.run(
        [str(bare_venv / "bin" / "python"), "-c", "pass"],
        env={"PATH": f"{bare_venv / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin"},
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        f"expected non-zero exit (sitecustomize hard-exit); got {result.returncode}, "
        f"stderr={result.stderr!r}"
    )


def test_default_attestation_sitecustomize_rewrites_base_executable(bare_venv, tmp_path):
    """R15 P1 (EV-1): sitecustomize must rewrite both ``sys.executable``
    AND ``sys._base_executable`` to the venv's wrapper. Without the
    base-executable rewrite, agent code that does
    ``subprocess.Popen([sys._base_executable, ...])`` would skip the
    layer-1.5 wrapper entirely and the spawned child would run against
    the operator's base interpreter (no telemetry, no per-run isolation).

    Auto-skips on shadowed systems (Homebrew Python on macOS) because
    our sitecustomize doesn't load there to perform the rewrite. On
    clean CI both attributes get rewritten.
    """
    shadowing = _detect_shadowing_sitecustomize(bare_venv)
    if shadowing is not None:
        pytest.skip(
            f"host Python ships its own sitecustomize at {shadowing!r}, "
            f"shadowing the venv shim; the rewrite cannot run."
        )
    log_path = tmp_path / "events.jsonl"
    log_path.touch()
    expected_wrapper = str(bare_venv / "bin" / "python")
    # Run python via the wrapper, then have it print both attributes so
    # we can verify both point at the wrapper.
    probe = subprocess.run(
        [
            str(bare_venv / "bin" / "python"),
            "-c",
            "import sys; "
            "print(sys.executable); "
            "print(getattr(sys, '_base_executable', '<missing>'))",
        ],
        env={
            "PATH": f"{bare_venv / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "_PYRUNTIME_EVENT_LOG": str(log_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, f"probe failed: stderr={probe.stderr!r}, stdout={probe.stdout!r}"
    lines = probe.stdout.strip().splitlines()
    assert len(lines) == 2, f"expected 2 lines (executable + base_executable), got: {lines!r}"
    sys_executable, sys_base_executable = lines
    assert (
        sys_executable == expected_wrapper
    ), f"sys.executable={sys_executable!r} not rewritten to wrapper {expected_wrapper!r}"
    # Skip the base-executable check on Pythons that don't expose it
    # (pre-3.11). Otherwise it MUST also point at the wrapper.
    if sys_base_executable != "<missing>":
        assert sys_base_executable == expected_wrapper, (
            f"sys._base_executable={sys_base_executable!r} not rewritten to "
            f"wrapper {expected_wrapper!r} (R15 EV-1 regression: agent could "
            f"spawn [sys._base_executable, ...] and bypass the layer-1.5 wrapper)"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
