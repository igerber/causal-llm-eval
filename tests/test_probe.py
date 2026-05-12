"""Unit tests for harness.probe leakage assessment.

Tests _assess_leakage against curated responses. The live probe (which
actually spawns an agent) lives in test_probe_live.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.probe import _assess_leakage, _check_structural, _extract_structural_block


def test_assess_leakage_passes_clean_response_with_affirmative_no():
    response = "I was not given any preloaded context. Nothing preloaded."
    a = _assess_leakage(response)
    assert a.passed is True
    assert a.findings == []


def test_assess_leakage_passes_clean_response_that_recites_question_terms():
    """Canonical clean cold-start response: recites probe terms in negation.

    Guards against a regression where the blacklist false-fires on
    probe-prompt-recited tokens (CLAUDE.md, MCP servers, etc.). If the
    blacklist re-acquires those tokens, this test catches it.
    """
    response = (
        "I do not have any preloaded CLAUDE.md, MCP servers, slash commands, "
        "or skills in this session. Nothing was preloaded into my context."
    )
    a = _assess_leakage(response)
    assert a.passed is True
    assert a.findings == []


def test_assess_leakage_fails_on_skill_name_mention():
    response = "I see a /ai-review-local command available. Nothing was preloaded."
    a = _assess_leakage(response)
    assert a.passed is False
    assert any("/ai-review-local" in f for f in a.findings)


def test_assess_leakage_fails_on_diff_diff_project_name_mention():
    response = "I have access to the diff-diff library context. Nothing was preloaded."
    a = _assess_leakage(response)
    assert a.passed is False
    assert any("diff-diff" in f for f in a.findings)


def test_assess_leakage_fails_on_silence_no_affirmative_statement():
    response = "I am ready to help."
    a = _assess_leakage(response)
    assert a.passed is False
    assert "no_affirmative_no_statement" in a.findings


def test_assess_leakage_findings_list_includes_each_hit():
    """A response with a blacklist hit AND no affirmative-no has both findings."""
    response = "I have feedback_lint_before_commit available."
    a = _assess_leakage(response)
    assert a.passed is False
    assert any("feedback_" in f for f in a.findings)
    assert "no_affirmative_no_statement" in a.findings


# -- structural verification (P1-3) ------------------------------------------


def _structural_block(cwd: str, home: str | None = None, env_keys: list[str] | None = None) -> str:
    """Build a --BEGIN-STRUCTURED--/--END-STRUCTURED-- block embeddable in a response."""
    payload = {
        "cwd": cwd,
        "home": home if home is not None else cwd,
        "env_keys": env_keys if env_keys is not None else ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG"],
    }
    return f"--BEGIN-STRUCTURED--\n{json.dumps(payload)}\n--END-STRUCTURED--\n"


def test_extract_structural_block_finds_json_between_markers():
    response = "Some prose.\n" + _structural_block("/tmp/causal_run_abc") + "Trailing prose."
    data = _extract_structural_block(response)
    assert data is not None
    assert data["cwd"] == "/tmp/causal_run_abc"


def test_extract_structural_block_returns_none_when_markers_absent():
    response = "No markers at all in this response."
    assert _extract_structural_block(response) is None


def test_extract_structural_block_returns_none_on_malformed_json():
    response = "--BEGIN-STRUCTURED--\n{not valid json}\n--END-STRUCTURED--\n"
    assert _extract_structural_block(response) is None


def test_check_structural_passes_on_matching_cwd_home_and_clean_env(tmp_path):
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "ANTHROPIC_API_KEY"],
    }
    assert _check_structural(data, str(tmp_path)) == []


def test_check_structural_fails_on_cwd_mismatch(tmp_path):
    data = {
        "cwd": "/some/other/path",
        "home": str(tmp_path),
        "env_keys": [],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("cwd_mismatch" in f for f in findings)


def test_check_structural_fails_on_home_mismatch(tmp_path):
    data = {
        "cwd": str(tmp_path),
        "home": "/Users/operator",
        "env_keys": [],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("home_mismatch" in f for f in findings)


def test_check_structural_fails_on_operator_env_leak(tmp_path):
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "XDG_CONFIG_HOME", "OPENAI_API_KEY"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: XDG_CONFIG_HOME" in f for f in findings)
    assert any("operator_env_leak: OPENAI_API_KEY" in f for f in findings)


def test_check_structural_fails_on_github_token_leak(tmp_path):
    """GITHUB_TOKEN / GH_TOKEN are unambiguous auth leaks (R1 P1 fix)."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "GITHUB_TOKEN"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: GITHUB_TOKEN" in f for f in findings)


def test_check_structural_fails_on_anthropic_auth_token_leak(tmp_path):
    """ANTHROPIC_AUTH_TOKEN is denylist-only even though ANTHROPIC_ prefix is allowed."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
    }
    findings = _check_structural(data, str(tmp_path))
    # ANTHROPIC_API_KEY is in the exact allowlist; ANTHROPIC_AUTH_TOKEN is in
    # the denylist and must take precedence over the ANTHROPIC_ prefix rule.
    assert any("operator_env_leak: ANTHROPIC_AUTH_TOKEN" in f for f in findings)
    assert not any("ANTHROPIC_API_KEY" in f for f in findings)


def test_check_structural_allowlist_passes_known_claude_cli_prefixed_keys(tmp_path):
    """CLI-injected CLAUDE_CODE_*/CLAUDECODE_* keys are recognized."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": [
            "PATH",
            "HOME",
            "_PYRUNTIME_EVENT_LOG",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE_TOOL_NAME",
            "ANTHROPIC_API_KEY",
        ],
    }
    findings = _check_structural(data, str(tmp_path))
    assert findings == []


# -- Fail-closed schema + required-keys + tightened deny rules (R2 fix) -------


def test_check_structural_fails_when_env_keys_missing(tmp_path):
    """Schema check: missing env_keys field -> finding (was a fail-open gap)."""
    data = {"cwd": str(tmp_path), "home": str(tmp_path)}
    findings = _check_structural(data, str(tmp_path))
    assert "missing_env_keys" in findings


def test_check_structural_fails_when_env_keys_is_empty_list(tmp_path):
    """Schema check: empty env_keys -> finding."""
    data = {"cwd": str(tmp_path), "home": str(tmp_path), "env_keys": []}
    findings = _check_structural(data, str(tmp_path))
    assert "empty_env_keys" in findings


def test_check_structural_fails_when_env_keys_is_not_a_list(tmp_path):
    """Schema check: env_keys must be a list, not a dict/string/etc."""
    data = {"cwd": str(tmp_path), "home": str(tmp_path), "env_keys": "PATH,HOME"}
    findings = _check_structural(data, str(tmp_path))
    assert any("malformed_env_keys" in f for f in findings)


def test_check_structural_fails_on_missing_required_keys(tmp_path):
    """Required keys (HOME, PATH, _PYRUNTIME_EVENT_LOG) absent -> findings."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["LANG"],  # all required keys absent
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("missing_required_env_key: PATH" in f for f in findings)
    assert any("missing_required_env_key: HOME" in f for f in findings)
    assert any("missing_required_env_key: _PYRUNTIME_EVENT_LOG" in f for f in findings)


def test_check_structural_fails_on_claude_oauth_token_via_prefix(tmp_path):
    """CLAUDE_OAUTH_TOKEN is caught by CLAUDE_OAUTH deny prefix (not allowed by CLAUDE_CODE_)."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "CLAUDE_OAUTH_TOKEN"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("sensitive_env_key: CLAUDE_OAUTH_TOKEN" in f for f in findings)


def test_check_structural_fails_on_claude_mcp_servers_via_prefix(tmp_path):
    """CLAUDE_MCP_SERVERS is caught by CLAUDE_MCP deny prefix."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "CLAUDE_MCP_SERVERS"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("sensitive_env_key: CLAUDE_MCP_SERVERS" in f for f in findings)


def test_check_structural_fails_on_anthropic_project_name_via_prefix(tmp_path):
    """ANTHROPIC_PROJECT_NAME is caught by ANTHROPIC_PROJECT_ deny prefix."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "ANTHROPIC_PROJECT_NAME"],
    }
    findings = _check_structural(data, str(tmp_path))
    # ANTHROPIC_PROJECT_NAME is in the explicit denylist; either label is acceptable.
    assert any(
        "operator_env_leak: ANTHROPIC_PROJECT_NAME" in f
        or "sensitive_env_key: ANTHROPIC_PROJECT_NAME" in f
        for f in findings
    )


def test_check_structural_fails_on_key_substring_unless_explicitly_allowed(tmp_path):
    """SOMETHING_API_KEY contains 'KEY' substring -> sensitive_env_key.

    ANTHROPIC_API_KEY also contains 'KEY' but is in the exact allowlist.
    """
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": [
            "PATH",
            "HOME",
            "_PYRUNTIME_EVENT_LOG",
            "ANTHROPIC_API_KEY",
            "SOMETHING_API_KEY",
        ],
    }
    findings = _check_structural(data, str(tmp_path))
    # Allowlist override: ANTHROPIC_API_KEY does NOT trigger a finding.
    assert not any("ANTHROPIC_API_KEY" in f for f in findings)
    # Substring deny: SOMETHING_API_KEY DOES.
    assert any("sensitive_env_key: SOMETHING_API_KEY" in f for f in findings)


def test_check_structural_fails_on_oauth_substring(tmp_path):
    """OAUTH substring catches arbitrary OAUTH-named keys."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "MY_VENDOR_OAUTH_TOKEN"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("sensitive_env_key: MY_VENDOR_OAUTH_TOKEN" in f for f in findings)


def test_check_structural_drops_broad_claude_prefix(tmp_path):
    """Operator-set CLAUDE_FOO (not CLAUDE_CODE_) is no longer auto-allowed."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "CLAUDE_FOO_BAR"],
    }
    findings = _check_structural(data, str(tmp_path))
    # CLAUDE_FOO_BAR doesn't match CLAUDE_CODE_ allow prefix, doesn't hit any
    # deny pattern -> unrecognized.
    assert any("unrecognized_env_key: CLAUDE_FOO_BAR" in f for f in findings)


# -- Python interpreter env vars (R3 P1 fix) ----------------------------------


def test_check_structural_fails_on_pythonpath(tmp_path):
    """PYTHONPATH alters import resolution -> denylist."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "PYTHONPATH"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: PYTHONPATH" in f for f in findings)


def test_check_structural_fails_on_pythonhome(tmp_path):
    """PYTHONHOME points to an alternate Python install -> denylist."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "PYTHONHOME"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: PYTHONHOME" in f for f in findings)


def test_check_structural_fails_on_pythonstartup(tmp_path):
    """PYTHONSTARTUP names a file Python runs at REPL startup -> denylist."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "PYTHONSTARTUP"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: PYTHONSTARTUP" in f for f in findings)


def test_check_structural_fails_on_pythonuserbase(tmp_path):
    """PYTHONUSERBASE redirects user site-packages -> denylist."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "PYTHONUSERBASE"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: PYTHONUSERBASE" in f for f in findings)


def test_check_structural_python_prefix_no_longer_blanket_allowed(tmp_path):
    """Generic PYTHON-prefixed key not in denylist falls through to unrecognized."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "PYTHONUNBUFFERED"],
    }
    findings = _check_structural(data, str(tmp_path))
    # PYTHONUNBUFFERED is benign behaviorally but we no longer blanket-allow
    # PYTHON-prefixed keys; it should now be unrecognized. If real probe runs
    # show the CLI sets this benignly, add it to the exact allowlist.
    assert any("unrecognized_env_key: PYTHONUNBUFFERED" in f for f in findings)


# -- run_probe CLI exit-code handling (R3 P2 fix) -----------------------------


# -- env_path_values verification (R4 P1 fix) ---------------------------------


def test_check_structural_env_path_values_passes_when_inside_tmpdir(tmp_path):
    """Path-valued env vars under tmpdir -> no finding."""
    inside = str(tmp_path / ".pyruntime" / "events.jsonl")
    (tmp_path / ".pyruntime").mkdir()
    Path(inside).touch()
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG"],
        "env_path_values": {"_PYRUNTIME_EVENT_LOG": inside, "PWD": str(tmp_path)},
    }
    findings = _check_structural(data, str(tmp_path))
    assert findings == []


def test_check_structural_env_path_values_fails_when_outside_tmpdir(tmp_path):
    """Path-valued env var pointing OUTSIDE tmpdir -> env_path_outside_tmpdir finding."""
    leaky_path = "/Users/operator/causal-llm-eval/runs/probe/abc/in_process_events.jsonl"
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG"],
        "env_path_values": {"_PYRUNTIME_EVENT_LOG": leaky_path},
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("env_path_outside_tmpdir: _PYRUNTIME_EVENT_LOG" in f for f in findings)


def test_check_structural_env_path_values_missing_is_skipped(tmp_path):
    """env_path_values absent -> no path findings (back-compat for older payloads)."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG"],
        # no env_path_values field
    }
    findings = _check_structural(data, str(tmp_path))
    # No env_path_* findings produced.
    assert not any("env_path_" in f for f in findings)


def test_check_structural_env_path_values_skips_empty_string(tmp_path):
    """An empty string env_path_values entry is treated as not-reported (no finding)."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG"],
        "env_path_values": {"CLAUDE_PROJECT_DIR": ""},
    }
    findings = _check_structural(data, str(tmp_path))
    assert not any("env_path_" in f for f in findings)


# -- Deny/allow disjointness + ordering (R4 P2 fix) ---------------------------


def test_check_structural_denylist_and_allowlist_are_disjoint():
    """Defense in depth: ensure no key is in both denylist and allowlist.

    If a future allowlist edit accidentally adds a denylist key, the deny-
    before-allow ordering still flags it, but the contract should explicitly
    keep the sets disjoint.
    """
    from harness.probe import _PROBE_ENV_ALLOWED_EXACT, _PROBE_ENV_DENYLIST

    overlap = set(_PROBE_ENV_DENYLIST) & set(_PROBE_ENV_ALLOWED_EXACT)
    assert not overlap, f"Denylist and allowlist must be disjoint; overlap: {overlap}"


def test_check_structural_denylist_wins_over_allowlist_even_if_both_set(tmp_path, monkeypatch):
    """If a future edit puts a key in both lists, denylist takes precedence."""
    # Synthesize an overlap by injecting into the allowlist tuple at runtime.
    from harness import probe

    monkeypatch.setattr(
        probe, "_PROBE_ENV_ALLOWED_EXACT", probe._PROBE_ENV_ALLOWED_EXACT + ("XDG_CONFIG_HOME",)
    )
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "_PYRUNTIME_EVENT_LOG", "XDG_CONFIG_HOME"],
    }
    findings = _check_structural(data, str(tmp_path))
    # Denylist hit must fire even though the key is now also in allowlist.
    assert any("operator_env_leak: XDG_CONFIG_HOME" in f for f in findings)


# -- run_probe CLI exit-code handling (R3 P2 fix) -----------------------------


def test_run_probe_marks_assessment_failed_on_cli_nonzero_exit(tmp_path):
    """A nonzero CLI exit invalidates the probe even with a clean final message."""
    from unittest.mock import patch

    from harness.probe import run_probe
    from harness.runner import RunResult

    fake_tmpdir = tmp_path / "tmp"
    fake_tmpdir.mkdir()

    # Build a "clean" probe response with valid structural data so the
    # assessment WOULD pass if exit_code were 0.
    clean_response = (
        "I do not have any preloaded CLAUDE.md or skills. Nothing was preloaded.\n"
        + _structural_block(str(fake_tmpdir))
    )
    fake_transcript = tmp_path / "transcript.jsonl"
    fake_transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": clean_response}]},
            }
        )
        + "\n"
    )

    fake_run_result = RunResult(
        run_id="abc1234567890def",
        arm="diff_diff",
        tmpdir=fake_tmpdir,
        transcript_jsonl_path=fake_transcript,
        in_process_events_path=tmp_path / "events.jsonl",
        cli_stderr_log_path=tmp_path / "cli_stderr.log",
        record_parquet_path=None,
        final_code_path=None,
        wall_clock_seconds=1.0,
        exit_code=1,  # <-- CLI failed
    )

    output_dir = tmp_path / "probe_out"
    output_dir.mkdir()

    with patch("harness.probe.run_one", return_value=fake_run_result):
        result = run_probe(output_dir=output_dir)

    assert result.assessment.passed is False
    assert any("cli_nonzero_exit: 1" in f for f in result.assessment.findings)


def test_check_structural_allowlist_flags_unrecognized_keys(tmp_path):
    """A key not in the exact allowlist and not matching a prefix rule -> finding."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["PATH", "HOME", "SOME_UNKNOWN_TOOL_VAR"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("unrecognized_env_key: SOME_UNKNOWN_TOOL_VAR" in f for f in findings)


def test_check_structural_denylist_does_not_double_flag_via_allowlist(tmp_path):
    """A denylist hit is reported as operator_env_leak, NOT also as unrecognized."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": ["XDG_CONFIG_HOME"],
    }
    findings = _check_structural(data, str(tmp_path))
    assert any("operator_env_leak: XDG_CONFIG_HOME" in f for f in findings)
    assert not any("unrecognized_env_key: XDG_CONFIG_HOME" in f for f in findings)


def test_assess_leakage_with_tmpdir_passes_on_clean_response_and_valid_structural(tmp_path):
    """End-to-end: self-report clean + structural matches expected tmpdir -> PASS."""
    response = (
        "I do not have any preloaded CLAUDE.md or skills. Nothing was preloaded.\n"
        + _structural_block(str(tmp_path))
    )
    a = _assess_leakage(response, expected_tmpdir=str(tmp_path))
    assert a.passed is True, f"unexpected findings: {a.findings}"
    assert a.findings == []
    assert a.structural is not None


def test_assess_leakage_with_tmpdir_fails_when_structural_block_missing(tmp_path):
    """Agent gave a clean self-report but skipped the python -c command -> FAIL."""
    response = "I do not have any preloaded context. Nothing was preloaded."
    a = _assess_leakage(response, expected_tmpdir=str(tmp_path))
    assert a.passed is False
    assert "no_structural_block" in a.findings


def test_assess_leakage_with_tmpdir_fails_on_cwd_leak(tmp_path):
    """Structural shows operator-leak cwd even though self-report is clean -> FAIL."""
    response = "Nothing was preloaded.\n" + _structural_block(
        "/Users/operator/something", home="/Users/operator/something"
    )
    a = _assess_leakage(response, expected_tmpdir=str(tmp_path))
    assert a.passed is False
    assert any("cwd_mismatch" in f for f in a.findings)


def test_assess_leakage_without_tmpdir_skips_structural_layer():
    """Backward-compat: expected_tmpdir=None runs self-report only."""
    response = "Nothing was preloaded."
    a = _assess_leakage(response)
    assert a.passed is True
    assert a.findings == []
    assert a.structural is None
