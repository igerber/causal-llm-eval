"""Template for the in-process instrumentation shim.

Copied into each per-run venv as `sitecustomize.py` (Python's standard
auto-import-on-startup hook). Logs library access events to a per-run JSON
event log specified by the `CAUSAL_LLM_EVAL_EVENT_LOG` environment variable.

Tracked events for arm 1 (diff-diff):
    - Every `import diff_diff` event
    - Every `get_llm_guide(variant)` call with the variant argument
    - Every fit-time `warnings.warn(...)` from `diff_diff.*` with message + category
    - Every diagnostic method call (compute_pretrends_power, compute_honest_did,
      bacon_decomposition, in_time_placebo, placebo_test, etc.)
    - Every estimator class instantiation in `diff_diff.*`

Tracked events for arm 2 (statsmodels) - parity instrumentation:
    - Same set, watching `statsmodels.*` instead.

Skeleton only. The instrumentation hooks (warnings.showwarning override, import
hooks, monkey-patch of estimator __init__ for class instantiation tracking) are
implemented in subsequent PRs.
"""

from __future__ import annotations

import json
import os


def _get_event_log_path() -> str | None:
    """Return the path the per-run shim writes events to, or None if not set."""
    return os.environ.get("CAUSAL_LLM_EVAL_EVENT_LOG")


def _write_event(event: dict) -> None:
    """Append a single JSON event to the per-run event log."""
    path = _get_event_log_path()
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


# Phase 0 placeholder: actual instrumentation hooks land in subsequent PRs.
# See harness/COLD_START_VERIFICATION.md for the contract this shim must fulfill.
