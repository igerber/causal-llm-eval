"""Unit tests for harness.probe leakage assessment.

Tests _assess_leakage against curated responses. The live probe (which
actually spawns an agent) lives in test_probe_live.py.
"""

from __future__ import annotations

import json

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
    """CLI-injected CLAUDE_*/CLAUDECODE_*/ANTHROPIC_* keys are recognized."""
    data = {
        "cwd": str(tmp_path),
        "home": str(tmp_path),
        "env_keys": [
            "PATH",
            "HOME",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE_TOOL_NAME",
            "ANTHROPIC_API_KEY",
        ],
    }
    findings = _check_structural(data, str(tmp_path))
    assert findings == []


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
