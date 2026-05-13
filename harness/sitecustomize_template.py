"""Template for the in-process instrumentation shim.

Copied into each per-run venv as `sitecustomize.py` (Python's standard
auto-import-on-startup hook). Logs library access events to a per-run JSON
event log specified by the `_PYRUNTIME_EVENT_LOG` environment variable. The
underscore prefix + Python-runtime framing is a deliberate low-reactivity
choice: an agent enumerating `os.environ` is less likely to flag the name as
eval-related.

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
      from absence-of-discovery. The statsmodels-side hooks are wired in a
      follow-up PR alongside the case-study statsmodels prompt; this PR
      establishes only the diff-diff side and the sentinel-None encoding.

## Failure mode: fail closed at module load; fail open inside hooks

If `_PYRUNTIME_EVENT_LOG` is unset or its path is unwritable AT MODULE LOAD,
the shim MUST raise rather than silently no-op. A misconfigured run that
returns a record with all-`False` discoverability flags is indistinguishable
from a real "agent never accessed any guide" outcome and would corrupt the
eval. Module-load failure is loud and fatal; the runner catches the exception
and marks the run failed.

Inside hook calls, transient OSError on event write is caught and logged to
stderr (captured by layer 3) while the original function STILL executes.
Telemetry loss on one event is recoverable; aborting the agent on telemetry
hiccup would be a worse evaluation artifact. The runner-level
`telemetry_missing` sentinel covers the case where the event log truly
disappeared post-exec.

Implements the contract above; see `harness/telemetry.py:merge_layers` for
the consumer.
"""

from __future__ import annotations

import builtins
import functools
import importlib.util
import io
import json
import os
import sys
import warnings
from datetime import datetime, timezone


class TelemetryConfigError(RuntimeError):
    """Raised when the in-process shim cannot find a writable event log path.

    Failing closed prevents silent telemetry loss: a misconfigured run that
    yields all-`False` discoverability flags is indistinguishable from a real
    "no access" outcome and corrupts the eval. The runner is expected to catch
    this and mark the run failed rather than continuing.
    """


# Canonical list of fit-bearing classes exported from `diff_diff.__all__`.
# Patched at both `__init__` and `fit` to record agent usage.
#
# Includes HonestDiD and PreTrendsPower (class-based wrappers around the
# diagnostic functions) so agents who write `HonestDiD(...).fit(...)` instead
# of `compute_honest_did(...)` are still recorded.
_ESTIMATOR_CLASS_NAMES: tuple[str, ...] = (
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SyntheticDiD",
    "CallawaySantAnna",
    "ChaisemartinDHaultfoeuille",
    "ContinuousDiD",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "TripleDifference",
    "TROP",
    "StackedDiD",
    "StaggeredTripleDifference",
    "EfficientDiD",
    "WooldridgeDiD",
    "BaconDecomposition",
    "HeterogeneousAdoptionDiD",
    "HonestDiD",
    "PreTrendsPower",
    "LinearRegression",
)


# Canonical list of module-level diagnostic functions in `diff_diff`.
# Patched to record agent invocation.
_DIAGNOSTIC_FUNCTION_NAMES: tuple[str, ...] = (
    "compute_pretrends_power",
    "compute_honest_did",
    "bacon_decompose",
    "run_placebo_test",
    "compute_power",
)


# The four bundled guide file names. The `builtins.open` hook records reads
# whose path ends in any of these (and which lie under a diff_diff guides
# directory).
_GUIDE_FILENAMES: tuple[str, ...] = (
    "llms.txt",
    "llms-practitioner.txt",
    "llms-autonomous.txt",
    "llms-full.txt",
)


def _get_event_log_path() -> str:
    """Return the path the per-run shim writes events to, or raise.

    Fail-closed contract: missing/empty env var raises TelemetryConfigError.
    """
    path = os.environ.get("_PYRUNTIME_EVENT_LOG")
    if not path:
        raise TelemetryConfigError(
            "_PYRUNTIME_EVENT_LOG is unset; in-process telemetry cannot be "
            "written. The runner must set this env var before spawning the "
            "agent."
        )
    return path


def _write_event(event: dict) -> None:
    """Append a single JSON event to the per-run event log.

    Raises TelemetryConfigError if the event log env var is unset, or OSError
    if the path is unwritable. Both are fatal at module load; inside hook
    calls, callers catch OSError and continue (see module docstring).
    """
    path = _get_event_log_path()
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        print(
            f"[pyruntime] cannot write event to {path}: {e}",
            file=sys.stderr,
        )
        raise


def _utc_iso_now() -> str:
    """Return a UTC ISO 8601 timestamp with microsecond precision.

    Stable, sortable, timezone-aware. `timezone.utc` is the conventional
    spelling; the 3.11+ `UTC` shortcut would also work (pyproject floor is
    Python 3.11).
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _safe_write(event: dict) -> None:
    """Write an event from inside a hook, catching transient write failures.

    Differs from `_write_event` in that OSError is caught here (event is
    dropped, message logged to stderr) so the wrapped function can still
    execute. Telemetry loss on one event is recoverable; aborting the agent
    mid-call would corrupt the run more than missing a single record.

    `TelemetryConfigError` (env var unset) is NOT caught here because that
    only fires if the env var was unset between module load and the call,
    which would indicate a broken contract worth surfacing.
    """
    try:
        _write_event(event)
    except OSError:
        # Already logged to stderr by _write_event; drop the event and let
        # the caller continue.
        pass


def _wrap_get_llm_guide(original):
    """Wrap `get_llm_guide(variant)` to record the variant before delegating."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(variant: str = "concise"):
        _safe_write(
            {
                "event": "guide_file_read",
                "via": "get_llm_guide",
                "variant": variant,
                "ts": _utc_iso_now(),
            }
        )
        return original(variant)

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_estimator_init(original, class_name: str):
    """Wrap an estimator class `__init__` to record instantiation."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        _safe_write(
            {
                "event": "estimator_init",
                "class": class_name,
                "ts": _utc_iso_now(),
            }
        )
        return original(self, *args, **kwargs)

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_estimator_fit(original, class_name: str):
    """Wrap an estimator class `fit` method to record the call."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        _safe_write(
            {
                "event": "estimator_fit",
                "class": class_name,
                "ts": _utc_iso_now(),
            }
        )
        return original(self, *args, **kwargs)

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_diagnostic(original, name: str):
    """Wrap a module-level diagnostic function to record invocation."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        _safe_write(
            {
                "event": "diagnostic_call",
                "name": name,
                "ts": _utc_iso_now(),
            }
        )
        return original(*args, **kwargs)

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


# Module-level state for the guide-files dir captured at attach time so the
# `_pyruntime_open` hook can decide whether a path is "ours" cheaply.
_diff_diff_guides_dir: str | None = None


def _path_is_diff_diff_guide(path) -> tuple[bool, str | None]:
    """Return (True, basename) if `path` is a bundled diff_diff guide file.

    Checks (a) the path resolves to one of the four known guide filenames,
    AND (b) the resolved path lies under the captured `_diff_diff_guides_dir`
    via normalized containment (not raw startswith). Both checks are
    necessary so non-guide opens (data files, source files) are not recorded
    and sibling directories sharing a prefix (e.g. `.../guides_extra/`)
    cannot overmatch.
    """
    try:
        path_str = os.fspath(path)
    except TypeError:
        return False, None
    if _diff_diff_guides_dir is None:
        return False, None
    try:
        from pathlib import Path as _Path

        resolved = _Path(path_str).resolve()
        guides_dir = _Path(_diff_diff_guides_dir).resolve()
        resolved.relative_to(guides_dir)
    except (ValueError, OSError):
        return False, None
    for filename in _GUIDE_FILENAMES:
        if resolved.name == filename:
            return True, filename
    return False, None


def _install_open_hook() -> None:
    """Override `builtins.open` AND `io.open` to record bundled guide-file reads.

    The override is idempotent — re-running leaves the existing wrapper in
    place, not a doubly-wrapped one.

    Both `builtins.open` and `io.open` must be patched. They share the same
    underlying C function but each name holds an independent reference, so
    patching only `builtins.open` would miss `pathlib.Path.read_text` (which
    uses `io.open` directly). Patching both catches:

    - Direct calls: `open("/path/to/llms.txt")`
    - Pathlib calls: `Path("/path/to/llms.txt").read_text()`
    - `importlib.resources.files("diff_diff.guides").joinpath("llms.txt").read_text()`
      (which goes through pathlib for installed packages)

    Low-level `os.read` on raw file descriptors is not caught; agents on
    causal-inference tasks don't reach for that.
    """
    if getattr(builtins.open, "_pyruntime_wrapped", False):
        return
    original_open = builtins.open

    @functools.wraps(original_open)
    def _pyruntime_open(file, *args, **kwargs):
        # Only record reads. Skip writes/appends/exclusive-creates, which
        # would otherwise produce false-positive guide-discovery telemetry
        # (a write to a guide path is not a read of guide content).
        mode = args[0] if args else kwargs.get("mode", "r")
        is_read = isinstance(mode, str) and "w" not in mode and "a" not in mode and "x" not in mode
        if is_read:
            matched, filename = _path_is_diff_diff_guide(file)
            if matched:
                _safe_write(
                    {
                        "event": "guide_file_read",
                        "via": "open",
                        "filename": filename,
                        "ts": _utc_iso_now(),
                    }
                )
        return original_open(file, *args, **kwargs)

    _pyruntime_open._pyruntime_wrapped = True  # type: ignore[attr-defined]
    builtins.open = _pyruntime_open  # type: ignore[assignment]
    io.open = _pyruntime_open  # type: ignore[assignment]


def _install_warning_hook() -> None:
    """Override `warnings.showwarning` to record warnings emitted from diff_diff.

    Idempotent: re-running the shim does not double-wrap. Filtering by
    substring on the filename is intentional (installed-package paths
    reliably contain the package name).
    """
    if getattr(warnings.showwarning, "_pyruntime_wrapped", False):
        return
    original_showwarning = warnings.showwarning

    def _pyruntime_showwarning(message, category, filename, lineno, file=None, line=None):
        if "diff_diff" in str(filename):
            _safe_write(
                {
                    "event": "warning_emitted",
                    "category": category.__name__,
                    "filename": str(filename),
                    "lineno": lineno,
                    "message": str(message)[:500],
                    "ts": _utc_iso_now(),
                }
            )
        return original_showwarning(message, category, filename, lineno, file, line)

    _pyruntime_showwarning._pyruntime_wrapped = True  # type: ignore[attr-defined]
    warnings.showwarning = _pyruntime_showwarning


def _attach_diff_diff_hooks(module) -> None:
    """Patch the loaded `diff_diff` module to record agent usage.

    Called from `_DiffDiffPostImportHook.exec_module` after Python finishes
    importing `diff_diff`. Wraps:

    - `module.get_llm_guide` and `module._guides_api.get_llm_guide`
      (both bindings; closes the `from diff_diff._guides_api import ...`
      bypass)
    - Each estimator class's `__init__` and `fit`
    - Each module-level diagnostic function

    Also captures the bundled guides directory for the `_pyruntime_open`
    filter, and installs the `builtins.open` hook (idempotent).
    """
    global _diff_diff_guides_dir

    # Wrap get_llm_guide at both bindings so submodule-direct imports
    # (`from diff_diff._guides_api import get_llm_guide`) also see the
    # wrapper. The meta_path hook fires on the parent-package import, which
    # always precedes submodule binding; at this point both attributes point
    # at the same original function.
    if hasattr(module, "get_llm_guide") and hasattr(module, "_guides_api"):
        wrapped = _wrap_get_llm_guide(module.get_llm_guide)
        module.get_llm_guide = wrapped
        module._guides_api.get_llm_guide = wrapped

    # Capture the guides dir for the open hook. Use importlib.resources to
    # resolve the package's data dir without depending on __file__ layout.
    try:
        from importlib.resources import files as _resources_files

        guides_traversable = _resources_files("diff_diff.guides")
        _diff_diff_guides_dir = str(guides_traversable)
    except (ImportError, ModuleNotFoundError):
        # Should not happen — diff_diff.guides exists in the wheel — but
        # fail-soft: the open hook simply won't fire if the dir is unknown.
        _diff_diff_guides_dir = None

    _install_open_hook()

    for class_name in _ESTIMATOR_CLASS_NAMES:
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        if hasattr(cls, "__init__"):
            cls.__init__ = _wrap_estimator_init(cls.__init__, class_name)
        if hasattr(cls, "fit"):
            cls.fit = _wrap_estimator_fit(cls.fit, class_name)

    for func_name in _DIAGNOSTIC_FUNCTION_NAMES:
        original = getattr(module, func_name, None)
        if original is None or not callable(original):
            continue
        setattr(module, func_name, _wrap_diagnostic(original, func_name))


class _DiffDiffPostImportHook:
    """`sys.meta_path` finder that triggers hook attachment after diff_diff loads.

    For non-`diff_diff` module names, `find_spec` returns `None` so the
    default finders win. For `diff_diff` itself, the hook:

    1. Removes itself from `sys.meta_path` to prevent recursion when it
       calls `importlib.util.find_spec("diff_diff")` below.
    2. Asks the rest of `sys.meta_path` to resolve `diff_diff`.
    3. Wraps the returned spec's `loader.exec_module` so that after Python
       finishes importing the module normally, `_attach_diff_diff_hooks` is
       invoked.
    4. Writes a `module_import` event.
    5. Returns the wrapped spec.

    Only the top-level `diff_diff` import triggers attachment; submodule
    imports (`from diff_diff._guides_api import X`) trigger parent-package
    import first per Python's guarantee, so the hook firing once on the
    parent is sufficient.
    """

    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname != "diff_diff":
            return None
        # Remove ourselves so the recursive find_spec doesn't loop.
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.util.find_spec("diff_diff")
        if spec is None or spec.loader is None:
            return spec
        original_exec_module = spec.loader.exec_module

        def _wrapped_exec_module(module):
            original_exec_module(module)
            _safe_write(
                {
                    "event": "module_import",
                    "module": "diff_diff",
                    "ts": _utc_iso_now(),
                }
            )
            _attach_diff_diff_hooks(module)

        spec.loader.exec_module = _wrapped_exec_module  # type: ignore[method-assign]
        return spec


# Module-load top-level: fail-closed write of session_start (requires
# _PYRUNTIME_EVENT_LOG to be set), then install hooks. If the env var is
# unset, _write_event raises TelemetryConfigError and the agent's Python
# startup fails — exactly the PR #1 fail-closed contract.
_write_event({"event": "session_start", "ts": _utc_iso_now()})
# Insert at FRONT of meta_path so we win against the default PathFinder for
# the diff_diff name. Appending would never fire because PathFinder already
# resolves installed packages.
sys.meta_path.insert(0, _DiffDiffPostImportHook())
_install_warning_hook()
