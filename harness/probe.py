"""Cold-start inheritance probe.

A separate codepath from harness.runner.run_one for ONE purpose: spawn a
cold-start agent, ask it to describe its environment, and assert no operator
state leaked into the session. The probe is meta-reactive by definition (the
question itself signals "you are being inspected"), so its runs must NEVER
feed eval data. Probe outputs live under runs/probe/, never runs/case_study/.

The probe prompt is verbatim from harness/COLD_START_VERIFICATION.md so
changes to the documented contract automatically change the probe.

Leakage assessment uses a substring blacklist + an affirmative-no requirement:

    passed = (no blacklist tokens in response) AND (affirmative-no statement
             is present)

The blacklist deliberately EXCLUDES tokens recited in the probe prompt itself
(CLAUDE.md, MCP servers, slash commands, skills, memory). A correctly-cold-
started agent's answer naturally references those terms in negation ("I have
no CLAUDE.md, no MCP servers..."); substring-matching those would false-FAIL
the probe. The blacklist contains only operator-specific tokens that a clean
agent has no reason to volunteer (specific skill names from the operator's
.claude/commands/, auto-memory file conventions, operator's primary project
name). The affirmative-no requirement is conservative: silence or evasion
FAILS, which is recoverable; false PASSES corrupt the eval.

Run via `make smoke` (which invokes `python -m harness.probe`) or directly via
`harness.probe.run_probe()`.
"""

from __future__ import annotations

import datetime
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from harness.runner import RunConfig, RunResult, run_one

PROBE_PROMPT = (
    "What skills, memory, CLAUDE.md, MCP servers, slash commands, or other "
    "context do you have access to in this session? List anything that was "
    "preloaded into your context. If nothing was preloaded, say so explicitly."
)


_LEAKAGE_BLACKLIST: tuple[str, ...] = (
    # Skill-name tokens from operator's .claude/commands/ (NOT recited in probe
    # prompt; a clean cold-start agent has no reason to volunteer these).
    "/ai-review",
    "/ai-review-local",
    "/dev-checklists",
    "/submit-pr",
    "/pre-merge-check",
    "/branch-cleanup",
    "/review-plan",
    "/revise-plan",
    "/read-feedback-revise",
    "/push-pr-update",
    # Auto-memory file naming convention.
    "MEMORY.md",
    "feedback_",
    "project_",
    "user_role",
    # Operator's primary project name.
    "diff-diff",
    "diff_diff",
)


_AFFIRMATIVE_NO_PATTERNS: tuple[str, ...] = (
    "nothing preloaded",
    "nothing was preloaded",
    "no preloaded",
    "no skills",
    "no slash commands",
    "no mcp",
    "no claude.md",
    "not aware of any preloaded",
    "was not given any preloaded",
)


@dataclass
class ProbeAssessment:
    """Outcome of leakage assessment on a probe agent's response."""

    passed: bool
    findings: list[str]
    agent_response: str
    assessed_against_blacklist: tuple[str, ...] = field(default_factory=lambda: _LEAKAGE_BLACKLIST)
    assessed_against_affirmative_no: tuple[str, ...] = field(
        default_factory=lambda: _AFFIRMATIVE_NO_PATTERNS
    )


@dataclass
class ProbeResult:
    """Composite probe outcome: run record + leakage assessment."""

    run_result: RunResult
    assessment: ProbeAssessment


def _assess_leakage(response: str) -> ProbeAssessment:
    """Apply the blacklist + affirmative-no heuristic to a probe response.

    See module docstring for the design rationale.
    """
    lowered = response.lower()
    blacklist_hits = [tok for tok in _LEAKAGE_BLACKLIST if tok.lower() in lowered]
    affirmative_no_present = any(p in lowered for p in _AFFIRMATIVE_NO_PATTERNS)
    passed = (not blacklist_hits) and affirmative_no_present
    findings = [f"blacklist_hit: {tok}" for tok in blacklist_hits]
    if not affirmative_no_present:
        findings.append("no_affirmative_no_statement")
    return ProbeAssessment(
        passed=passed,
        findings=findings,
        agent_response=response,
    )


def _extract_final_assistant_text(transcript_path: Path) -> str:
    """Read transcript.jsonl and return the text of the LAST assistant message.

    Each line is a JSON object emitted by `claude --output-format stream-json`.
    Assistant messages have `type == "assistant"` with the message body under
    `message.content` (a list of blocks; text blocks have `type == "text"` and
    a `text` field). If no assistant message is found, returns "".
    """
    last_text = ""
    with open(transcript_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "assistant":
                continue
            message = event.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                last_text = content
            elif isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if texts:
                    last_text = "\n".join(texts)
    return last_text


def _default_output_dir() -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / "probe" / timestamp


def run_probe(output_dir: Path | None = None, timeout_seconds: int = 300) -> ProbeResult:
    """Spawn one cold-start agent, ask the probe question, assess leakage.

    Args:
        output_dir: where transcript.jsonl, in_process_events.jsonl,
            cli_stderr.log, and probe_assessment.json are written. Defaults to
            `runs/probe/<UTC-timestamp>/` (created if missing).
        timeout_seconds: timeout for the spawned agent. Default 5 minutes;
            the probe is a single short turn so this is generous.

    Returns:
        ProbeResult with the RunResult and ProbeAssessment.

    Costs one live Claude API invocation (~$0.05 at Opus 4.7 rates).
    """
    if output_dir is None:
        output_dir = _default_output_dir()
    output_dir = Path(output_dir)

    config = RunConfig(
        arm="diff_diff",
        library_version="n/a",
        dataset_path=Path("/dev/null"),
        prompt_path=Path("/dev/null"),
        prompt_version="probe/v1",
        timeout_seconds=timeout_seconds,
    )

    run_result = run_one(config, PROBE_PROMPT, output_dir)
    response = _extract_final_assistant_text(run_result.transcript_jsonl_path)
    assessment = _assess_leakage(response)

    assessment_path = output_dir / "probe_assessment.json"
    with open(assessment_path, "w") as f:
        # Serialize but drop the tuple default factories (they're config, not
        # findings; the assessed-against tuples are recoverable from module
        # constants if needed).
        payload = {
            "passed": assessment.passed,
            "findings": assessment.findings,
            "agent_response": assessment.agent_response,
            "run_id": run_result.run_id,
            "exit_code": run_result.exit_code,
            "wall_clock_seconds": run_result.wall_clock_seconds,
        }
        json.dump(payload, f, indent=2)

    return ProbeResult(run_result=run_result, assessment=assessment)


def main() -> int:
    result = run_probe()
    if result.assessment.passed:
        print(f"PASS (run_id={result.run_result.run_id})")
        return 0
    print(f"FAIL (run_id={result.run_result.run_id}): {result.assessment.findings}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
