"""AI judge applies a YAML rubric to a per-run transcript and final code.

Separate from the harness: this is a plain Claude API call, NOT another Claude
Code subprocess. Returns structured JSON matching the rubric schema.

Used as the second stage of two-stage extraction (deterministic from in-process
events first, judge from transcript second). Disagreements between deterministic
and judge extractions are flagged for spot-check per the judge spot-check
protocol in the plan.

Skeleton only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class JudgeResult:
    """Structured output from one judge call against one transcript."""

    rubric_version: str
    fields: dict  # rubric-driven; structure validated against the rubric schema


def judge_transcript(
    transcript_path: Path,
    final_code_path: Path | None,
    rubric_path: Path,
    model: str = "claude-opus-4-7",
) -> JudgeResult:
    """Apply the rubric at rubric_path to the transcript. Returns structured JSON.

    Implementation pending.
    """
    del transcript_path, final_code_path, rubric_path, model
    raise NotImplementedError("graders.ai_judge.judge_transcript is not yet implemented")
