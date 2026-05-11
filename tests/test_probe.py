"""Unit tests for harness.probe leakage assessment.

Tests _assess_leakage against curated responses. The live probe (which
actually spawns an agent) lives in test_probe_live.py.
"""

from __future__ import annotations

from harness.probe import _assess_leakage


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
