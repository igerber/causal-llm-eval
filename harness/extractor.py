"""Deterministic result extraction from in-process event logs.

Two-stage extraction (per the plan):
    1. Deterministic from in-process instrumentation log: estimator class name,
       diagnostic method calls, warnings observed. Fast, exact.
    2. AI judge as fallback / cross-check (see graders/ai_judge.py).

Both run on every transcript; disagreements flagged for spot review per the
judge spot-check protocol.

Skeleton only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeterministicExtraction:
    """Fields extracted deterministically from in-process events."""

    estimator_classes: tuple[str, ...]
    diagnostic_methods: tuple[str, ...]
    fit_time_warnings: tuple[str, ...]
    get_llm_guide_calls: tuple[str, ...]


def extract_from_in_process_log(events_path: Path) -> DeterministicExtraction:
    """Parse a per-run in-process event log into structured signals.

    Implementation pending.
    """
    del events_path
    raise NotImplementedError("extractor.extract_from_in_process_log is not yet implemented")
