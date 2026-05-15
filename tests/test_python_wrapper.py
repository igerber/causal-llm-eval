"""Fast tests for the layer-1.5 ``python_wrapper.sh``.

PR #5 R1 P2 #4: these tests run by default (no ``slow`` marker) and exercise
the wrapper without building a real venv. They install the wrapper at a
fake ``python`` path with ``python-real`` symlinked to ``/usr/bin/true``
and invoke it to verify the JSONL event shape, fail-closed paths, and
argv encoding rules.

For real-venv coverage (build_arm_template end-to-end), see
``tests/test_venv_pool.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_WRAPPER_SOURCE = Path(__file__).parent.parent / "harness" / "python_wrapper.sh"


@pytest.fixture
def fake_venv(tmp_path):
    """Install the wrapper at ``${tmp}/bin/python`` with python-real
    symlinked to ``/usr/bin/true`` at the canonical layer-1.5 location
    ``${tmp}/.pyruntime-real/python-real``.

    ``/usr/bin/true`` ignores all args and exits 0, so this lets tests
    assert the wrapper's event-emission behavior without depending on a
    real Python interpreter.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_dir = tmp_path / ".pyruntime-real"
    real_dir.mkdir()
    real = real_dir / "python-real"
    os.symlink("/usr/bin/true", real)
    wrapper = bin_dir / "python"
    shutil.copyfile(_WRAPPER_SOURCE, wrapper)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return tmp_path


def test_wrapper_emits_exec_python_event(fake_venv, tmp_path):
    log = tmp_path / "events.jsonl"
    subprocess.run(
        [str(fake_venv / "bin" / "python"), "-c", "pass"],
        check=True,
        env={
            "PATH": f"{fake_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
    )
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(events) == 1
    evt = events[0]
    assert evt["event"] == "exec_python"
    assert evt["argv"] == ["python", "-c", "pass"]
    # The wrapper's "$real" includes the /../ segment; that's normalized
    # on the merger side.
    assert evt["executable"].endswith(".pyruntime-real/python-real")
    assert isinstance(evt["pid"], int)
    assert isinstance(evt["ppid"], int)


def test_wrapper_skips_write_when_env_unset(fake_venv, tmp_path):
    log = tmp_path / "events.jsonl"
    result = subprocess.run(
        [str(fake_venv / "bin" / "python"), "-c", "pass"],
        env={"PATH": f"{fake_venv / 'bin'}:/usr/bin:/bin"},
        capture_output=True,
    )
    # /usr/bin/true ignores args and exits 0. The wrapper skipped its
    # event write; log was never created.
    assert result.returncode == 0
    assert not log.exists()


def test_wrapper_fails_closed_on_unwritable_log(fake_venv, tmp_path):
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    os.chmod(readonly_dir, 0o500)
    try:
        log = readonly_dir / "events.jsonl"
        result = subprocess.run(
            [str(fake_venv / "bin" / "python"), "-c", "pass"],
            env={
                "PATH": f"{fake_venv / 'bin'}:/usr/bin:/bin",
                "_PYRUNTIME_EVENT_LOG": str(log),
            },
            capture_output=True,
        )
        assert result.returncode == 2
        assert b"cannot append" in result.stderr
    finally:
        os.chmod(readonly_dir, 0o700)


def test_wrapper_fails_closed_on_newline_in_argv(fake_venv, tmp_path):
    """R1 P1 #2: the awk-record-count check fails closed when any arg
    contains a newline. POSIX argv may legally contain newlines but the
    line-oriented encoder cannot preserve them without record-shape
    corruption.
    """
    log = tmp_path / "events.jsonl"
    result = subprocess.run(
        [str(fake_venv / "bin" / "python"), "-c", "pass", "line1\nline2"],
        env={
            "PATH": f"{fake_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
    )
    assert result.returncode == 2
    assert b"embedded newline" in result.stderr


def test_wrapper_argv_with_special_chars(fake_venv, tmp_path):
    """Quotes, backslashes, and tabs JSON-encode cleanly."""
    log = tmp_path / "events.jsonl"
    weird = 'has"quote\\backslash\there'
    subprocess.run(
        [str(fake_venv / "bin" / "python"), "-c", "pass", weird],
        check=True,
        env={
            "PATH": f"{fake_venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
    )
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(events) == 1
    assert events[0]["argv"] == ["python", "-c", "pass", weird]


@pytest.fixture
def fake_venv_with_strip_s_shim(tmp_path):
    """Fake venv with the strip-S shim AND a fake actual-python binary.

    Layout:
        ${tmp}/bin/python                       -> wrapper
        ${tmp}/.pyruntime-real/python-real      -> strip-S shim
        ${tmp}/.pyruntime-real/.actual-python   -> printf-based fake binary

    The fake actual-python prints its argv so tests can assert
    argv-preservation through the strip-S layer.
    """
    from harness.venv_pool import _PYTHON_REAL_STRIP_S_SCRIPT

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_dir = tmp_path / ".pyruntime-real"
    real_dir.mkdir()
    # Fake actual-python: prints "ARGV:" then each arg on its own line,
    # using printf to preserve quoting fidelity.
    actual = real_dir / ".actual-python"
    actual.write_text(
        '#!/usr/bin/env sh\nprintf "ARGV:\\n"\nfor a in "$@"; do printf "  [%s]\\n" "$a"; done\n'
    )
    actual.chmod(0o755)
    # Strip-S shim from venv_pool source.
    python_real = real_dir / "python-real"
    python_real.write_text(_PYTHON_REAL_STRIP_S_SCRIPT)
    python_real.chmod(0o755)
    # Wrapper.
    wrapper = bin_dir / "python"
    shutil.copyfile(_WRAPPER_SOURCE, wrapper)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return tmp_path


def test_strip_s_shim_preserves_quoted_args_with_spaces(fake_venv_with_strip_s_shim):
    """R4 P1 #2: ``python -c 'print(1)'`` with quoted multi-word args
    must reach .actual-python with argv exactly preserved (no field
    splitting / globbing).
    """
    venv = fake_venv_with_strip_s_shim
    log = venv / "events.jsonl"
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import sys; print('hi')", "x y z"],
        env={
            "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # The fake actual-python prints argv; verify the quoted arg with spaces
    # arrived as a single token, not split.
    assert "[-c]" in result.stdout
    assert "[import sys; print('hi')]" in result.stdout
    assert "[x y z]" in result.stdout


def test_strip_s_shim_drops_S_flag(fake_venv_with_strip_s_shim):
    """``python -S script.py`` -> .actual-python sees ``script.py``
    (no -S). Confirms strip-S shim removes the bypass flag before
    the real interpreter runs.
    """
    venv = fake_venv_with_strip_s_shim
    log = venv / "events.jsonl"
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-S", "script.py"],
        env={
            "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "[-S]" not in result.stdout
    assert "[script.py]" in result.stdout


def test_strip_s_shim_strips_S_from_compact_IS_form(fake_venv_with_strip_s_shim):
    """R4 P1 #2: ``python -IS -c 'print(1)'`` compact form. -IS contains
    the S bypass flag; strip-S shim removes the S character, keeping
    -I.
    """
    venv = fake_venv_with_strip_s_shim
    log = venv / "events.jsonl"
    result = subprocess.run(
        [str(venv / "bin" / "python"), "-IS", "-c", "print(1)"],
        env={
            "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(log),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # -IS becomes -I; original -IS is gone.
    assert "[-I]" in result.stdout
    assert "[-IS]" not in result.stdout


@pytest.fixture
def venv_layout_fake(tmp_path):
    """Wrapper at canonical layout: ``${tmp}/venv/bin/python`` with
    ``${tmp}/.pyruntime/events.jsonl`` as the canonical runner-owned log.

    Triggers the wrapper's canonical-override path so retarget attempts
    via inherited env are overridden.
    """
    runner_root = tmp_path
    venv_dir = runner_root / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    real_dir = venv_dir / ".pyruntime-real"
    real_dir.mkdir()
    real = real_dir / "python-real"
    os.symlink("/usr/bin/true", real)
    runner_log_dir = runner_root / ".pyruntime"
    runner_log_dir.mkdir()
    canonical_log = runner_log_dir / "events.jsonl"
    canonical_log.touch()
    wrapper = bin_dir / "python"
    shutil.copyfile(_WRAPPER_SOURCE, wrapper)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return runner_root, venv_dir, canonical_log


def test_wrapper_overrides_inherited_event_log_with_canonical_path(venv_layout_fake):
    """R8 P0: the wrapper computes the canonical log path from its own
    location and overrides any inherited ``_PYRUNTIME_EVENT_LOG``. An
    agent that sets ``_PYRUNTIME_EVENT_LOG=/tmp/fake`` (via shell
    script, Python subprocess env=, env wrapper, etc.) cannot route the
    wrapper's events to a non-runner log.
    """
    runner_root, venv_dir, canonical_log = venv_layout_fake
    fake_log = runner_root / "fake.jsonl"
    subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "pass"],
        check=True,
        env={
            "PATH": f"{venv_dir / 'bin'}:/usr/bin:/bin",
            # Agent attempts to retarget; wrapper must ignore.
            "_PYRUNTIME_EVENT_LOG": str(fake_log),
        },
        capture_output=True,
    )
    # Event landed in the canonical runner-owned log, not the agent's
    # retarget destination.
    canonical_events = canonical_log.read_text().strip()
    assert canonical_events, "wrapper did not write to canonical log"
    evt = json.loads(canonical_events.splitlines()[0])
    assert evt["event"] == "exec_python"
    # Fake retarget destination remains empty.
    assert not fake_log.exists() or fake_log.read_text() == ""


def test_wrapper_uses_absolute_shebang_not_path_resolved(venv_layout_fake):
    """R9 P0: wrapper shebang must be ``#!/bin/sh`` (absolute path),
    not ``#!/usr/bin/env sh`` (PATH-resolved). Otherwise an agent that
    writes ``${venv}/bin/sh`` could hijack the wrapper invocation.
    """
    runner_root, venv_dir, _ = venv_layout_fake
    wrapper = venv_dir / "bin" / "python"
    first_line = wrapper.read_text().splitlines()[0]
    assert first_line == "#!/bin/sh", (
        f"wrapper shebang must be absolute (#!/bin/sh) to prevent "
        f"agent-shadowed sh in venv/bin/; got {first_line!r}"
    )


def test_wrapper_resists_shadowed_sh_in_venv_bin(venv_layout_fake, tmp_path):
    """R9 P0: agent drops a malicious ``${venv}/bin/sh`` (e.g., ``exit 1``).
    Because the wrapper's shebang is ``#!/bin/sh`` (absolute), the
    malicious shadow is never used. Telemetry still lands.
    """
    runner_root, venv_dir, canonical_log = venv_layout_fake
    # Agent shadows venv/bin/sh.
    bad_sh = venv_dir / "bin" / "sh"
    bad_sh.write_text("#!/bin/sh\nexit 1\n")
    bad_sh.chmod(0o755)
    canonical_log.write_text("")  # reset
    subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "pass"],
        check=True,
        env={
            "PATH": f"{venv_dir / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(canonical_log),  # ignored; canonical override applies
        },
        capture_output=True,
    )
    events = canonical_log.read_text().strip()
    assert events, "wrapper failed to write event despite shadowed venv/bin/sh"
    evt = json.loads(events.splitlines()[0])
    assert evt["event"] == "exec_python"


def test_wrapper_resists_shadowed_awk_in_venv_bin(venv_layout_fake):
    """R9 P0: agent drops a malicious ``${venv}/bin/awk``. Because the
    wrapper internally pins PATH=/usr/bin:/bin, the shadow is never
    used.
    """
    runner_root, venv_dir, canonical_log = venv_layout_fake
    bad_awk = venv_dir / "bin" / "awk"
    bad_awk.write_text("#!/bin/sh\necho '[hijacked]'\nexit 0\n")
    bad_awk.chmod(0o755)
    canonical_log.write_text("")
    subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "pass"],
        check=True,
        env={
            "PATH": f"{venv_dir / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(canonical_log),
        },
        capture_output=True,
    )
    events = canonical_log.read_text().strip()
    assert events, "wrapper failed despite shadowed venv/bin/awk"
    evt = json.loads(events.splitlines()[0])
    # Real awk produced a real JSON event, not the hijacked output.
    assert evt["event"] == "exec_python"
    assert "argv" in evt


def test_strip_s_shim_resists_shadowed_sed_in_venv_bin(venv_layout_fake):
    """R10 P2 DT-1: agent drops a malicious ``${venv}/bin/sed``. The
    strip-S shim's internal PATH=/usr/bin:/bin pin prevents the
    shadow from being used; sed produces the real strip output.
    """
    runner_root, venv_dir, canonical_log = venv_layout_fake
    # Install the strip-S shim properly (replacing the symlink the
    # fake_venv fixture installed).
    from harness.venv_pool import _PYTHON_REAL_STRIP_S_SCRIPT

    real_dir = venv_dir / ".pyruntime-real"
    python_real = real_dir / "python-real"
    if python_real.exists() or python_real.is_symlink():
        python_real.unlink()
    python_real.write_text(_PYTHON_REAL_STRIP_S_SCRIPT)
    python_real.chmod(0o755)
    actual = real_dir / ".actual-python"
    if actual.exists() or actual.is_symlink():
        actual.unlink()
    actual.write_text(
        '#!/bin/sh\nprintf "ARGV:\\n"\nfor a in "$@"; do printf "  [%s]\\n" "$a"; done\n'
    )
    actual.chmod(0o755)
    # Agent shadows venv/bin/sed with a malicious version.
    bad_sed = venv_dir / "bin" / "sed"
    bad_sed.write_text("#!/bin/sh\necho '[hijacked]'\nexit 0\n")
    bad_sed.chmod(0o755)
    canonical_log.write_text("")
    result = subprocess.run(
        [str(venv_dir / "bin" / "python"), "-S", "-c", "pass"],
        check=True,
        env={
            "PATH": f"{venv_dir / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(canonical_log),
        },
        capture_output=True,
        text=True,
    )
    # Real sed worked: strip-S removed -S; actual-python ran with -c pass argv.
    # If the malicious sed had been used, "[hijacked]" would be in stdout
    # AND the strip would not have happened.
    assert "[-S]" not in result.stdout, "strip-S shim used hijacked sed"
    assert "[-c]" in result.stdout
    assert "[pass]" in result.stdout


def test_strip_s_shim_does_not_corrupt_dash_c_argument_with_S(venv_layout_fake):
    """R10 P2 CQ-1: ``python -c "-S"`` -- the literal "-S" is the python
    code, not an interpreter flag. Strip-S shim must be option-aware so
    -c's argument passes through verbatim.
    """
    runner_root, venv_dir, canonical_log = venv_layout_fake
    from harness.venv_pool import _PYTHON_REAL_STRIP_S_SCRIPT

    real_dir = venv_dir / ".pyruntime-real"
    python_real = real_dir / "python-real"
    if python_real.exists() or python_real.is_symlink():
        python_real.unlink()
    python_real.write_text(_PYTHON_REAL_STRIP_S_SCRIPT)
    python_real.chmod(0o755)
    actual = real_dir / ".actual-python"
    if actual.exists() or actual.is_symlink():
        actual.unlink()
    actual.write_text(
        '#!/bin/sh\nprintf "ARGV:\\n"\nfor a in "$@"; do printf "  [%s]\\n" "$a"; done\n'
    )
    actual.chmod(0o755)
    canonical_log.write_text("")
    result = subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "-S"],
        check=True,
        env={
            "PATH": f"{venv_dir / 'bin'}:/usr/bin:/bin",
            "_PYRUNTIME_EVENT_LOG": str(canonical_log),
        },
        capture_output=True,
        text=True,
    )
    # The "-S" reaches actual-python intact as the -c argument, NOT
    # stripped by the shim's S-detection.
    assert "[-c]" in result.stdout
    assert "[-S]" in result.stdout
