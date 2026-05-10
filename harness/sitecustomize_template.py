"""Template for the in-process instrumentation shim.

Copied into each per-run venv as `sitecustomize.py` (Python's standard
auto-import-on-startup hook). Logs library access events to a per-run JSON
event log specified by the `CAUSAL_LLM_EVAL_EVENT_LOG` environment variable.

Tracked events for arm 1 (diff-diff) - this contract pairs 1:1 with the
discoverability fields on `TelemetryRecord` and the rubric's "Discovered LLM
guidance" column:

    - Every `import diff_diff` event
    - Every direct guide-file read for the bundled guides (`llms.txt`,
      `llms-practitioner.txt`, `llms-autonomous.txt`, `llms-full.txt`),
      whether via `builtins.open()`, `pathlib.Path.read_text/read_bytes`,
      `importlib.resources.files(...).read_text/read_bytes`, or any other
      Python-internal path that returns the file's bytes. This grounds the
      `opened_llms_*` flags on TelemetryRecord.
    - Every `get_llm_guide(variant)` call with the variant argument
    - Every fit-time `warnings.warn(...)` from `diff_diff.*` with message + category
    - Every diagnostic method call (compute_pretrends_power, compute_honest_did,
      bacon_decomposition, in_time_placebo, placebo_test, etc.)
    - Every estimator class instantiation in `diff_diff.*`

Tracked events for arm 2 (statsmodels) - parity instrumentation:
    - Same set, watching `statsmodels.*` instead. Guide-file read tracking is
      "not applicable" rather than "not discovered" for statsmodels (no bundled
      LLM guides exist); the merger encodes this as `None`/sentinel rather than
      `False` so comparator-fairness analysis can distinguish absence-of-feature
      from absence-of-discovery.

## Failure mode: fail closed, not open

If `CAUSAL_LLM_EVAL_EVENT_LOG` is unset or its path is unwritable, the shim
MUST raise rather than silently no-op. A misconfigured run that returns a
record with all-`False` discoverability flags is indistinguishable from a real
"agent never accessed any guide" outcome and would corrupt the eval. Failure
is loud and fatal; the runner catches the exception and marks the run failed.

Skeleton only. The instrumentation hooks (warnings.showwarning override, import
hooks, file-read hooks, monkey-patch of estimator __init__) land in subsequent
PRs.
"""

from __future__ import annotations

import json
import os
import sys


class TelemetryConfigError(RuntimeError):
    """Raised when the in-process shim cannot find a writable event log path.

    Failing closed prevents silent telemetry loss: a misconfigured run that
    yields all-`False` discoverability flags is indistinguishable from a real
    "no access" outcome and corrupts the eval. The runner is expected to catch
    this and mark the run failed rather than continuing.
    """


def _get_event_log_path() -> str:
    """Return the path the per-run shim writes events to, or raise.

    Fail-closed contract: missing/empty env var raises TelemetryConfigError.
    """
    path = os.environ.get("CAUSAL_LLM_EVAL_EVENT_LOG")
    if not path:
        raise TelemetryConfigError(
            "CAUSAL_LLM_EVAL_EVENT_LOG is unset; in-process telemetry cannot "
            "be written. The runner must set this env var before spawning the "
            "agent."
        )
    return path


def _write_event(event: dict) -> None:
    """Append a single JSON event to the per-run event log.

    Raises TelemetryConfigError if the event log env var is unset, or OSError
    if the path is unwritable. Both are fatal: silent no-op would invalidate
    the eval (see module docstring).
    """
    path = _get_event_log_path()
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        # Surface to stderr so the runner can capture it in layer 3 even if
        # the in-process layer is the failing one.
        print(
            f"[causal-llm-eval] FATAL: cannot write in-process event to {path}: {e}",
            file=sys.stderr,
        )
        raise


# Phase 0 placeholder: actual instrumentation hooks land in subsequent PRs.
# See harness/COLD_START_VERIFICATION.md for the contract this shim must fulfill.
