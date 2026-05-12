"""Unit tests for harness.runner.

Tests the cold-start invocation contract, env hygiene, and run_one's control
flow without spawning live agents. Live tests are in test_runner_live.py
(marked @pytest.mark.live).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.runner import (
    _ALLOWLISTED_PASSTHROUGH_KEYS,
    RunConfig,
    _build_command,
    clean_env,
    run_one,
)

# -- clean_env tests ----------------------------------------------------------


def test_clean_env_returns_only_allowlist(tmp_path, monkeypatch):
    """clean_env returns only allowlisted passthrough + runner-set keys."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "/operator/xdg")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/operator/claude")
    monkeypatch.setenv("AWS_PROFILE", "operator")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
    monkeypatch.setenv("LC_RPATH", "/should/not/leak")
    monkeypatch.setenv("CODEX_HOME", "/operator/codex")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LANG", "en_US.UTF-8")

    env = clean_env(tmp_path, tmp_path / "events.jsonl")

    expected_max = set(_ALLOWLISTED_PASSTHROUGH_KEYS) | {"HOME", "_PYRUNTIME_EVENT_LOG"}
    leaks = set(env) - expected_max
    assert not leaks, f"clean_env leaked non-allowlisted keys: {leaks}"
    for forbidden in (
        "XDG_CONFIG_HOME",
        "CLAUDE_CONFIG_DIR",
        "AWS_PROFILE",
        "OPENAI_API_KEY",
        "LC_RPATH",
        "CODEX_HOME",
    ):
        assert forbidden not in env, f"clean_env leaked {forbidden}"


def test_clean_env_uses_tmpdir_for_home(tmp_path, monkeypatch):
    """HOME points at the per-run tmpdir, never the operator's $HOME."""
    monkeypatch.setenv("HOME", "/operator/home/should/not/leak")
    env = clean_env(tmp_path, tmp_path / "events.jsonl")
    assert env["HOME"] == str(tmp_path)


def test_clean_env_passes_through_anthropic_key_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    env = clean_env(tmp_path, tmp_path / "events.jsonl")
    assert env["ANTHROPIC_API_KEY"] == "sk-test-123"


def test_clean_env_omits_anthropic_key_when_unset(tmp_path, monkeypatch):
    """If ANTHROPIC_API_KEY is unset, it's NOT fabricated in clean_env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = clean_env(tmp_path, tmp_path / "events.jsonl")
    assert "ANTHROPIC_API_KEY" not in env


def test_clean_env_event_log_uses_new_var_name(tmp_path):
    """Regression for env-var rename: _PYRUNTIME_EVENT_LOG set; old name absent."""
    event_log = tmp_path / "events.jsonl"
    env = clean_env(tmp_path, event_log)
    assert env["_PYRUNTIME_EVENT_LOG"] == str(event_log)
    assert "CAUSAL_LLM_EVAL_EVENT_LOG" not in env


def test_clean_env_overwrites_inherited_pyruntime_event_log(tmp_path, monkeypatch):
    """Operator-set _PYRUNTIME_EVENT_LOG must not pass through; runner wins."""
    monkeypatch.setenv("_PYRUNTIME_EVENT_LOG", "/sentinel/operator/value")
    event_log = tmp_path / "events.jsonl"
    env = clean_env(tmp_path, event_log)
    assert env["_PYRUNTIME_EVENT_LOG"] == str(event_log)
    assert env["_PYRUNTIME_EVENT_LOG"] != "/sentinel/operator/value"


def test_clean_env_lc_keys_explicit_no_wildcard(tmp_path, monkeypatch):
    """LC_RPATH (not in explicit list) is dropped; LC_ALL etc. pass through."""
    monkeypatch.setenv("LC_RPATH", "/should/not/leak")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("LC_CTYPE", "en_US.UTF-8")
    env = clean_env(tmp_path, tmp_path / "events.jsonl")
    assert "LC_RPATH" not in env
    assert env["LC_ALL"] == "en_US.UTF-8"
    assert env["LC_CTYPE"] == "en_US.UTF-8"


# -- _build_command tests -----------------------------------------------------


def test_build_command_includes_all_seven_locked_flags(tmp_path):
    """The seven required cold-start flags are all present, with exact tokens."""
    cmd = _build_command("test prompt", tmp_path, "claude-opus-4-7")
    assert "--bare" in cmd
    assert "--setting-sources" in cmd
    idx = cmd.index("--setting-sources")
    assert cmd[idx + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert "--disable-slash-commands" in cmd
    assert "--print" in cmd
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"
    assert "--add-dir" in cmd
    idx = cmd.index("--add-dir")
    assert cmd[idx + 1] == str(tmp_path)
    # The prompt is the LAST argv element.
    assert cmd[-1] == "test prompt"


def test_build_command_uses_bare_first(tmp_path):
    """--bare must precede other flags (CLI semantics)."""
    cmd = _build_command("p", tmp_path, "claude-opus-4-7")
    assert cmd[0] == "claude"
    assert cmd[1] == "--bare"


def test_build_command_includes_model_flag(tmp_path):
    """--model pins the model so CLI defaults can't drift across runs."""
    cmd = _build_command("p", tmp_path, "claude-opus-4-7")
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "claude-opus-4-7"


def test_build_command_passes_through_arbitrary_model_string(tmp_path):
    """The model string is whatever RunConfig.model holds, not hardcoded."""
    cmd = _build_command("p", tmp_path, "claude-sonnet-4-6")
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "claude-sonnet-4-6"


# -- run_one tests ------------------------------------------------------------


def _config(timeout: int = 1800) -> RunConfig:
    return RunConfig(
        arm="diff_diff",
        library_version="n/a",
        dataset_path=Path("/dev/null"),
        prompt_path=Path("/dev/null"),
        prompt_version="test/v1",
        timeout_seconds=timeout,
    )


def test_run_one_creates_output_dir_and_in_process_events_stub(tmp_path):
    """run_one creates output_dir + an empty in_process_events.jsonl stub."""
    output_dir = tmp_path / "run_out"

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        result = run_one(_config(), "prompt", output_dir)

    assert output_dir.exists()
    assert (output_dir / "in_process_events.jsonl").exists()
    assert (output_dir / "transcript.jsonl").exists()
    assert (output_dir / "cli_stderr.log").exists()
    # All paths exposed on RunResult so downstream telemetry merging doesn't
    # need to rely on filename convention.
    assert result.transcript_jsonl_path == (output_dir / "transcript.jsonl").resolve()
    assert result.in_process_events_path == (output_dir / "in_process_events.jsonl").resolve()
    assert result.cli_stderr_log_path == (output_dir / "cli_stderr.log").resolve()
    assert result.exit_code == 0
    assert result.arm == "diff_diff"
    assert len(result.run_id) == 16


def test_run_one_uses_absolute_event_log_path_with_relative_output_dir(tmp_path, monkeypatch):
    """P0-1 regression: relative output_dir resolves to absolute before spawn.

    Without this, the in-process shim (running with cwd=run-tmpdir) would
    resolve _PYRUNTIME_EVENT_LOG against the spawned process's cwd, writing
    to a different file than the runner pre-touched.
    """
    monkeypatch.chdir(tmp_path)
    relative_output_dir = Path("rel_run_out")  # NOT absolute

    captured_env: dict = {}

    def fake_popen(*_args, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        proc = MagicMock()
        proc.wait.return_value = 0
        return proc

    with patch("harness.runner.subprocess.Popen", side_effect=fake_popen):
        result = run_one(_config(), "prompt", relative_output_dir)

    assert "_PYRUNTIME_EVENT_LOG" in captured_env
    event_log_path = captured_env["_PYRUNTIME_EVENT_LOG"]
    assert Path(event_log_path).is_absolute(), (
        f"_PYRUNTIME_EVENT_LOG must be absolute (got {event_log_path!r}) so the "
        f"shim cannot misinterpret it against the spawned subprocess's cwd."
    )
    # And the file the runner touched is the same file the shim will see.
    assert Path(event_log_path) == result.in_process_events_path
    assert result.in_process_events_path.exists()


def test_run_one_pre_spawn_check_raises_on_unwritable_event_log(tmp_path):
    """Pre-spawn touch fails before subprocess.Popen is called."""
    # output_dir's parent is a regular file, not a directory -> mkdir fails.
    blocker_file = tmp_path / "blocker"
    blocker_file.write_text("not a directory")
    output_dir = blocker_file / "subdir"

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        with pytest.raises((NotADirectoryError, OSError, FileNotFoundError)):
            run_one(_config(), "prompt", output_dir)
        mock_popen.assert_not_called()


def test_run_one_raises_on_preexisting_transcript(tmp_path):
    """Existing transcript.jsonl in output_dir prevents subprocess spawn."""
    output_dir = tmp_path / "run_out"
    output_dir.mkdir()
    (output_dir / "transcript.jsonl").write_text("old transcript line\n")

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        with pytest.raises(FileExistsError):
            run_one(_config(), "prompt", output_dir)
        mock_popen.assert_not_called()


def test_run_one_raises_on_preexisting_in_process_events(tmp_path):
    """Existing in_process_events.jsonl in output_dir prevents subprocess spawn."""
    output_dir = tmp_path / "run_out"
    output_dir.mkdir()
    (output_dir / "in_process_events.jsonl").write_text("old event line\n")

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        with pytest.raises(FileExistsError):
            run_one(_config(), "prompt", output_dir)
        mock_popen.assert_not_called()


def test_run_one_raises_on_preexisting_cli_stderr_log(tmp_path):
    """Existing cli_stderr.log in output_dir prevents subprocess spawn (R3 P2 fix).

    Without this, a reused output_dir could silently destroy layer-3 stderr
    telemetry. All three sinks (transcript, in_process_events, cli_stderr)
    are now in the no-overwrite guard.
    """
    output_dir = tmp_path / "run_out"
    output_dir.mkdir()
    (output_dir / "cli_stderr.log").write_text("old stderr content\n")

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        with pytest.raises(FileExistsError):
            run_one(_config(), "prompt", output_dir)
        mock_popen.assert_not_called()


def test_run_one_on_timeout_returns_negative_exit_code_with_stderr_marker(tmp_path):
    """TimeoutExpired -> killpg, marker line in stderr log, exit_code=-1, no raise."""
    output_dir = tmp_path / "run_out"
    config = _config(timeout=1)

    with (
        patch("harness.runner.subprocess.Popen") as mock_popen,
        patch("harness.runner.os.killpg") as mock_killpg,
    ):
        proc = MagicMock()
        proc.pid = 99999
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="claude", timeout=1),
            0,
        ]
        mock_popen.return_value = proc
        result = run_one(config, "prompt", output_dir)

    assert result.exit_code == -1
    # P2 R2 fix: killpg kills the whole process group, not just the parent.
    mock_killpg.assert_called_once()
    args = mock_killpg.call_args[0]
    assert args[0] == 99999, "killpg must be called with proc.pid (the session leader)"
    stderr_content = (output_dir / "cli_stderr.log").read_text()
    assert "TIMEOUT" in stderr_content
    assert "killed" in stderr_content


def test_run_one_starts_subprocess_in_new_session(tmp_path):
    """R2 P2 fix: Popen called with start_new_session=True so killpg targets the whole tree."""
    output_dir = tmp_path / "run_out"

    with patch("harness.runner.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        run_one(_config(), "prompt", output_dir)

    _, kwargs = mock_popen.call_args
    assert kwargs.get("start_new_session") is True, (
        "Popen must use start_new_session=True so the spawned process becomes "
        "the session leader; otherwise killpg on timeout cannot reach children."
    )


def test_run_one_on_timeout_handles_already_dead_process_group(tmp_path):
    """killpg can race the OS reaping the parent; ProcessLookupError must be swallowed."""
    output_dir = tmp_path / "run_out"
    config = _config(timeout=1)

    with (
        patch("harness.runner.subprocess.Popen") as mock_popen,
        patch("harness.runner.os.killpg", side_effect=ProcessLookupError),
    ):
        proc = MagicMock()
        proc.pid = 99999
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="claude", timeout=1),
            0,
        ]
        mock_popen.return_value = proc
        result = run_one(config, "prompt", output_dir)

    # Race is harmless: we still record exit_code=-1 and the timeout marker.
    assert result.exit_code == -1
    stderr_content = (output_dir / "cli_stderr.log").read_text()
    assert "TIMEOUT" in stderr_content
