"""Three-layer telemetry capture for agent runs.

Layer 1 - Stream-JSON event log from Claude Code:
    Parsed from `claude --print --output-format stream-json` output. Contains
    every user/assistant turn, every tool call (Bash, Read, Edit, Write, Grep)
    with arguments and results, and file reads with paths.

Layer 2 - In-process Python instrumentation (the discoverability ground truth):
    A `sitecustomize.py` installed in the per-run venv hooks the target library
    and writes per-event JSON records. Catches access that stream-JSON misses
    (e.g., `python -c "from diff_diff import get_llm_guide"` reads the file via
    Python, not Claude's Read tool). See `harness/sitecustomize_template.py`.

Layer 3 - Subprocess stderr capture:
    Captures Python warnings and any other stderr the agent's Python processes
    emit. Cross-checked with the in-process warning log.

Skeleton only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TelemetryRecord:
    """Merged per-run record assembled from three layers."""

    stream_json_path: Path
    in_process_events_path: Path
    stderr_path: Path
    # Resolved discoverability flags (from in-process layer)
    opened_llms_txt: bool = False
    opened_llms_practitioner: bool = False
    opened_llms_autonomous: bool = False
    opened_llms_full: bool = False
    called_get_llm_guide: bool = False
    get_llm_guide_variants: tuple[str, ...] = ()
    saw_fit_time_warning: bool = False
    diagnostic_methods_invoked: tuple[str, ...] = ()
    estimator_classes_instantiated: tuple[str, ...] = ()


def merge_layers(
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Merge the three telemetry layers into a single record.

    Implementation pending.
    """
    del stream_json_path, in_process_events_path, stderr_path
    raise NotImplementedError("telemetry.merge_layers is not yet implemented")
