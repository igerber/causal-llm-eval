"""Cold-start inheritance probe.

A separate codepath from harness.runner.run_one for ONE purpose: spawn a
cold-start agent, ask it to describe its environment, and assert no operator
state leaked into the session. The probe is meta-reactive by definition (the
question itself signals "you are being inspected"), so its runs must NEVER
feed eval data. Probe outputs live under runs/probe/, never runs/case_study/.

The probe runs two layers of verification:

1. **Self-report (substring blacklist + affirmative-no)**: parses the agent's
   prose answer for operator-state tokens (specific skill names, auto-memory
   file conventions, the operator's primary project name) and requires an
   explicit "nothing was preloaded"-style statement.

2. **Structural (pwd / HOME / env-key allowlist + denylist)**: asks the
   agent to run a `python -c` one-liner that emits `{cwd, home, env_keys}`
   between `--BEGIN-STRUCTURED--` / `--END-STRUCTURED--` markers. The
   assessment verifies cwd points at the per-run tmpdir, HOME equals cwd,
   and env keys split into two layers:

       (a) Denylist: known operator-state leak keys (auth tokens, AWS, etc.)
           — definite findings, the spawned environment must never have them.
       (b) Allowlist: keys not in the expected set + not matching a
           Claude/Python prefix rule are flagged as "unrecognized" for
           review. Catches CLI-injected vars we haven't enumerated as well
           as genuine leaks we didn't predict.

   Black-box self-report alone could pass a leaky cold-start where the agent
   doesn't notice the leak; the structural layer catches what self-report
   would miss.

The blacklist deliberately EXCLUDES tokens recited in the probe prompt itself
(CLAUDE.md, MCP servers, slash commands, skills, memory). A correctly-cold-
started agent's answer naturally references those terms in negation ("I have
no CLAUDE.md, no MCP servers..."); substring-matching those would false-FAIL
the probe. The blacklist contains only operator-specific tokens that a clean
agent has no reason to volunteer. The affirmative-no requirement is
conservative: silence or evasion FAILS, which is recoverable; false PASSES
corrupt the eval.

Run via `make smoke` (which invokes `python -m harness.probe`) or directly via
`harness.probe.run_probe()`.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from harness.runner import RunConfig, RunResult, run_one

_STRUCTURAL_BEGIN = "--BEGIN-STRUCTURED--"
_STRUCTURAL_END = "--END-STRUCTURED--"

PROBE_PROMPT = (
    "What skills, memory, CLAUDE.md, MCP servers, slash commands, or other "
    "context do you have access to in this session? List anything that was "
    "preloaded into your context. If nothing was preloaded, say so explicitly.\n"
    "\n"
    "Then run this single python command verbatim using your Bash tool and "
    "include the raw output in your reply between the markers shown:\n"
    "\n"
    "python3 -c 'import os, json, sys; "
    'sys.stdout.write("--BEGIN-STRUCTURED--\\n" + '
    'json.dumps({"cwd": os.getcwd(), "home": os.path.expanduser("~"), '
    '"env_keys": sorted(os.environ.keys())}) + '
    '"\\n--END-STRUCTURED--\\n")\'\n'
    "\n"
    "Include the full output between the markers verbatim. Do not interpret it."
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


# Env keys whose presence is an UNAMBIGUOUS operator-state leak. Definite
# findings regardless of any allowlist hit; we never want these in the
# spawned process even if they happen to match a Claude-injected prefix.
_PROBE_ENV_DENYLIST: tuple[str, ...] = (
    "XDG_CONFIG_HOME",
    "CLAUDE_CONFIG_DIR",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "OPENAI_API_KEY",
    "CODEX_HOME",
    "ANTHROPIC_PROJECT_ID",
    "ANTHROPIC_AUTH_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


# Allowlist: keys we EXPECT to see in a cold-started agent's environment.
# Sources: clean_env() in runner.py, plus shell/OS-level vars commonly set
# by the agent's Bash subprocess when it spawns python.
_PROBE_ENV_ALLOWED_EXACT: tuple[str, ...] = (
    # From clean_env()
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "ANTHROPIC_API_KEY",
    "_PYRUNTIME_EVENT_LOG",
    # Set by shells / Python startup when the agent spawns python via Bash.
    "PWD",
    "OLDPWD",
    "SHELL",
    "USER",
    "LOGNAME",
    "TERM",
    "TERMINFO",
    "TMPDIR",
    "_",
)


# Allowed prefixes: keys CLI tools may inject into their subprocesses
# (Claude Code, Anthropic SDK, Python). Catching common families without
# enumerating every variant. ANTHROPIC_AUTH_TOKEN is denylist-only and
# stays a finding regardless of the prefix rule.
_PROBE_ENV_ALLOWED_PREFIXES: tuple[str, ...] = (
    "CLAUDE_",
    "CLAUDECODE_",
    "ANTHROPIC_",
    "PYTHON",
)


def _env_key_is_recognized(key: str) -> bool:
    """Allowlist check: key is in the explicit list or matches an allowed prefix."""
    if key in _PROBE_ENV_ALLOWED_EXACT:
        return True
    return any(key.startswith(p) for p in _PROBE_ENV_ALLOWED_PREFIXES)


@dataclass
class ProbeAssessment:
    """Outcome of leakage + structural assessment on a probe agent's response."""

    passed: bool
    findings: list[str]
    agent_response: str
    structural: dict | None = None
    assessed_against_blacklist: tuple[str, ...] = field(default_factory=lambda: _LEAKAGE_BLACKLIST)
    assessed_against_affirmative_no: tuple[str, ...] = field(
        default_factory=lambda: _AFFIRMATIVE_NO_PATTERNS
    )
    assessed_against_env_denylist: tuple[str, ...] = field(
        default_factory=lambda: _PROBE_ENV_DENYLIST
    )
    assessed_against_env_allowed_exact: tuple[str, ...] = field(
        default_factory=lambda: _PROBE_ENV_ALLOWED_EXACT
    )
    assessed_against_env_allowed_prefixes: tuple[str, ...] = field(
        default_factory=lambda: _PROBE_ENV_ALLOWED_PREFIXES
    )


@dataclass
class ProbeResult:
    """Composite probe outcome: run record + assessment."""

    run_result: RunResult
    assessment: ProbeAssessment


def _extract_structural_block(response: str) -> dict | None:
    """Extract the JSON between --BEGIN-STRUCTURED-- / --END-STRUCTURED-- markers."""
    match = re.search(
        rf"{re.escape(_STRUCTURAL_BEGIN)}\s*(\{{.*?\}})\s*{re.escape(_STRUCTURAL_END)}",
        response,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _check_structural(data: dict, expected_tmpdir: str) -> list[str]:
    """Verify pwd/HOME/env-keys match the cold-start contract.

    On macOS, tempfile.mkdtemp() returns paths like /var/folders/... while
    os.getcwd() inside the subprocess returns /private/var/folders/...
    (the resolved /private prefix). Path.resolve() handles both.
    """
    findings: list[str] = []
    expected_resolved = str(Path(expected_tmpdir).resolve())

    cwd = data.get("cwd", "")
    home = data.get("home", "")
    env_keys = data.get("env_keys") or []

    cwd_resolved = str(Path(cwd).resolve()) if cwd else ""
    home_resolved = str(Path(home).resolve()) if home else ""

    if cwd_resolved != expected_resolved:
        findings.append(f"cwd_mismatch: got {cwd_resolved!r}, expected {expected_resolved!r}")
    if home_resolved != expected_resolved:
        findings.append(f"home_mismatch: got {home_resolved!r}, expected {expected_resolved!r}")

    # Two-pass env check:
    # 1. Denylist: unambiguous operator-state leaks (auth tokens, AWS, etc.).
    #    Definite findings; we never want these in the spawned environment.
    # 2. Allowlist: any key not in the expected set + not matching a Claude/
    #    Python-prefix rule is flagged as "unrecognized". May be benign (a
    #    CLI-injected var we haven't enumerated yet) or genuinely leaky;
    #    review the finding and decide whether to extend the allowlist or
    #    treat as a real leak.
    denylist_hits: set[str] = set()
    for key in env_keys:
        if key in _PROBE_ENV_DENYLIST:
            findings.append(f"operator_env_leak: {key}")
            denylist_hits.add(key)
    for key in env_keys:
        if key in denylist_hits:
            continue
        if not _env_key_is_recognized(key):
            findings.append(f"unrecognized_env_key: {key}")

    return findings


def _assess_leakage(response: str, expected_tmpdir: str | None = None) -> ProbeAssessment:
    """Apply self-report + (optional) structural assessment to a probe response.

    When `expected_tmpdir` is None, only the self-report layer runs (substring
    blacklist + affirmative-no). When provided, the structural layer also runs
    against the JSON block between --BEGIN-STRUCTURED-- / --END-STRUCTURED--.

    The split lets unit tests exercise self-report in isolation; `run_probe`
    always passes `expected_tmpdir` so live probes apply both layers.
    """
    lowered = response.lower()
    blacklist_hits = [tok for tok in _LEAKAGE_BLACKLIST if tok.lower() in lowered]
    affirmative_no_present = any(p in lowered for p in _AFFIRMATIVE_NO_PATTERNS)
    findings = [f"blacklist_hit: {tok}" for tok in blacklist_hits]
    if not affirmative_no_present:
        findings.append("no_affirmative_no_statement")

    structural = None
    if expected_tmpdir is not None:
        structural = _extract_structural_block(response)
        if structural is None:
            findings.append("no_structural_block")
        else:
            findings.extend(_check_structural(structural, expected_tmpdir))

    return ProbeAssessment(
        passed=not findings,
        findings=findings,
        agent_response=response,
        structural=structural,
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
    """Build a unique default probe output dir.

    Microsecond-resolution timestamp + short uuid suffix prevents collision
    when two run_probe() calls land within the same second.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
    suffix = uuid.uuid4().hex[:6]
    return Path("runs") / "probe" / f"{timestamp}-{suffix}"


def run_probe(output_dir: Path | None = None, timeout_seconds: int = 300) -> ProbeResult:
    """Spawn one cold-start agent, ask the probe question, assess leakage.

    Args:
        output_dir: where transcript.jsonl, in_process_events.jsonl,
            cli_stderr.log, and probe_assessment.json are written. Defaults to
            `runs/probe/<UTC-timestamp_microseconds-uuid>/` (created if missing).
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
    assessment = _assess_leakage(response, expected_tmpdir=str(run_result.tmpdir))

    assessment_path = output_dir / "probe_assessment.json"
    with open(assessment_path, "w") as f:
        payload = {
            "passed": assessment.passed,
            "findings": assessment.findings,
            "agent_response": assessment.agent_response,
            "structural": assessment.structural,
            "run_id": run_result.run_id,
            "exit_code": run_result.exit_code,
            "wall_clock_seconds": run_result.wall_clock_seconds,
            "tmpdir": str(run_result.tmpdir),
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
