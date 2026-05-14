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
