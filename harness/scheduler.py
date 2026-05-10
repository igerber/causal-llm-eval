"""Parallel run scheduler with budget tracking.

Phase 1 runs sequentially (30 runs is small). Phase 2 runs use this scheduler to
parallelize across N concurrent agents while tracking dollar budget so we don't
overshoot during automated re-runs.

Skeleton only - sequential single-run path is the Phase 1 default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetConfig:
    """Soft and hard budget caps in USD."""

    soft_cap_usd: float
    hard_cap_usd: float


def run_batch_sequential(configs, output_dir, budget: BudgetConfig | None = None):
    """Run a batch of agent configs sequentially. Used for Phase 1.

    Implementation pending.
    """
    del configs, output_dir, budget
    raise NotImplementedError("scheduler.run_batch_sequential is not yet implemented")


def run_batch_concurrent(configs, output_dir, max_workers: int, budget: BudgetConfig):
    """Run a batch of agent configs with concurrency. Used for Phase 2.

    Implementation pending.
    """
    del configs, output_dir, max_workers, budget
    raise NotImplementedError("scheduler.run_batch_concurrent is not yet implemented")
