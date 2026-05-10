#!/usr/bin/env python3
"""Local AI code review using OpenAI Responses API.

Compiles a review prompt from the project's review criteria template plus the
local git diff against a base ref, sends it to the OpenAI API, and writes the
review markdown to an output file.

Uses only Python stdlib - no external dependencies.

Skill/Script Contract:
    Called by the /ai-review-local skill (.claude/commands/ai-review-local.md).
    Responsibilities are divided as follows:

    Skill (caller) handles:
    - Git operations: committing changes, determining base branch
    - Secret scanning: runs canonical patterns BEFORE calling this script
    - User interaction: displaying results, offering next steps
    - Cleanup: removing temp files

    Script (this file) handles:
    - Diff capture: git diff against base ref
    - Prompt compilation: review criteria + diff + local-context framing
    - OpenAI API call: authentication, request, error handling, timeout
    - Output: writing review markdown to --output path

This is a minimal port of diff-diff's openai_review.py, dropped to ~280 lines
by removing REGISTRY.md section extraction, import-graph expansion, and
re-review state tracking. Re-add features here as the eval grows.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Git diff capture
# ---------------------------------------------------------------------------


def _git(args: "list[str]", cwd: "str | None" = None) -> str:
    """Run a git command and return stdout (decoded). Exit on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: git {' '.join(args)} failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def capture_diff(base_ref: str, repo_root: str) -> "tuple[str, str]":
    """Return (changed_files_text, unified_diff_text) for working tree vs merge-base(base_ref).

    Diffs against the merge-base of base_ref and HEAD rather than base_ref directly.
    This produces the PR-shaped diff (only this branch's changes) even when the
    branch is behind base_ref, mirroring the GitHub PR view's three-dot diff.
    Diffing directly against base_ref would mix in unrelated upstream deltas if
    the branch was created from an older base_ref.

    Captures both staged and unstaged changes, plus committed-on-branch.
    Excludes large generated artifacts (per-run records, transcripts, dataset binaries)
    matching the same patterns as the CI workflow.
    """
    merge_base = _git(["merge-base", base_ref, "HEAD"], cwd=repo_root).strip()
    if not merge_base:
        # Fall back to direct base_ref diff if merge-base is empty (e.g., disjoint history)
        merge_base = base_ref
    name_status = _git(["diff", "--name-status", merge_base], cwd=repo_root)
    diff = _git(
        [
            "diff",
            "--unified=5",
            merge_base,
            "--",
            ".",
            ":!runs/**/*.parquet",
            ":!runs/**/*.jsonl",
            ":!runs/**/transcript.txt",
            ":!datasets/*.parquet",
            ":!datasets/*.csv",
        ],
        cwd=repo_root,
    )
    return name_status, diff


# ---------------------------------------------------------------------------
# Prompt compilation
# ---------------------------------------------------------------------------


def compile_prompt(
    criteria_text: str,
    name_status: str,
    diff_text: str,
    branch: str,
    base_ref: str,
) -> str:
    """Assemble the full review prompt from criteria + local diff context."""
    parts = [
        criteria_text,
        "",
        "---",
        "",
        "Local pre-PR review context (the PR has not been opened yet; this review",
        f"is run by the developer locally on branch `{branch}` against base `{base_ref}`).",
        "",
        "Changed files:",
        name_status if name_status.strip() else "(no changes against base ref)",
        "",
        "Unified diff (context=5):",
        diff_text if diff_text.strip() else "(empty diff)",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------

ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT = 300  # non-reasoning models
REASONING_TIMEOUT = 900  # reasoning models can take 10-15 min
DEFAULT_MAX_TOKENS = 16384
REASONING_MAX_TOKENS = 32768


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "o4", "gpt-5.4", "gpt-5.5")) or "-pro" in model


def _resolve_timeout(timeout: "int | None", model: str) -> int:
    if timeout is not None:
        return timeout
    return REASONING_TIMEOUT if _is_reasoning_model(model) else DEFAULT_TIMEOUT


def _extract_response_text(result: dict) -> str:
    """Extract review text from a Responses API JSON payload."""
    text = result.get("output_text") or ""
    if text:
        return text
    for item in result.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text += block.get("text", "")
    return text


def call_openai(
    prompt: str,
    model: str,
    api_key: str,
    timeout: "int | None" = None,
) -> "tuple[str, dict]":
    """Call the OpenAI Responses API and return (content, usage)."""
    timeout = _resolve_timeout(timeout, model)
    reasoning = _is_reasoning_model(model)
    max_tokens = REASONING_MAX_TOKENS if reasoning else DEFAULT_MAX_TOKENS

    payload: dict = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if not reasoning:
        payload["temperature"] = 0

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code == 401:
            print("Error: Invalid or expired OpenAI API key.", file=sys.stderr)
            print("Set OPENAI_API_KEY in your shell environment (~/.zshrc).", file=sys.stderr)
        elif e.code == 429:
            print("Error: Rate limited by OpenAI. Wait and retry.", file=sys.stderr)
        elif e.code >= 500:
            print(f"Error: OpenAI server error (HTTP {e.code}).", file=sys.stderr)
            if body:
                print(body[:500], file=sys.stderr)
        else:
            print(f"Error: OpenAI API returned HTTP {e.code}.", file=sys.stderr)
            if body:
                print(body[:500], file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"Error: Request timed out (>{timeout}s). Try a smaller diff.", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: Network error - {e.reason}", file=sys.stderr)
        sys.exit(1)

    content = _extract_response_text(result)

    status = result.get("status")
    if content.strip() and status == "incomplete":
        detail = result.get("incomplete_details") or ""
        print(
            "Error: Review was truncated (status='incomplete'). "
            "Output may be missing findings.",
            file=sys.stderr,
        )
        if detail:
            print(f"Detail: {detail}", file=sys.stderr)
        sys.exit(1)

    if not content.strip():
        status = result.get("status", "<missing>")
        detail = result.get("incomplete_details") or result.get("error") or ""
        if status not in ("completed", "<missing>"):
            print(
                f"Error: OpenAI response status is '{status}' with no review content.",
                file=sys.stderr,
            )
        else:
            print("Error: Empty review content from OpenAI API.", file=sys.stderr)
        if detail:
            print(f"Detail: {detail}", file=sys.stderr)
        sys.exit(1)

    usage = result.get("usage", {})
    return (content, usage)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _read_file(path: str, label: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Required file not found: {path} ({label})", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local AI code review via OpenAI Responses API."
    )
    parser.add_argument(
        "--review-criteria",
        default=".github/codex/prompts/pr_review.md",
        help="Path to review criteria template (default: .github/codex/prompts/pr_review.md)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the review markdown to.",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base git ref to diff against (default: origin/main).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            f"Per-request HTTP timeout in seconds. Defaults to {REASONING_TIMEOUT}s "
            f"for reasoning models and {DEFAULT_TIMEOUT}s otherwise."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root to run git from (default: current directory).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    repo_root = args.repo_root or os.getcwd()

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).strip()
    name_status, diff_text = capture_diff(args.base, repo_root)

    if not diff_text.strip():
        print(f"No diff against {args.base}. Nothing to review.", file=sys.stderr)
        sys.exit(0)

    criteria_text = _read_file(args.review_criteria, "review criteria")
    prompt = compile_prompt(criteria_text, name_status, diff_text, branch, args.base)

    content, usage = call_openai(prompt, args.model, api_key, args.timeout)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)

    in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    print(
        f"Review written to {args.output} "
        f"({in_tok} input tokens, {out_tok} output tokens, model={args.model}).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
