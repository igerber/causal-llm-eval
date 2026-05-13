"""Python runtime instrumentation. Module documentation lives outside this
file in `harness/COLD_START_VERIFICATION.md`."""

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
    """Raised when the event log path is unset or unwritable."""


# Canonical list of fit-bearing classes exported from `diff_diff.__all__`.
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
            "_PYRUNTIME_EVENT_LOG is unset; runtime event log cannot be written."
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
    """Write an event, dropping on transient OSError so the caller proceeds."""
    try:
        _write_event(event)
    except OSError:
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

    Low-level `os.read` on raw file descriptors is not caught.
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


def _caller_is_from_diff_diff(start_frame) -> tuple[bool, str, int]:
    """Walk the call stack looking for a frame whose module is diff_diff.

    Returns (matched, filename, lineno). When matched, the returned filename
    and lineno point at the actual diff_diff frame in the call stack, not
    whatever the caller's `stacklevel` argument designated for display.
    """
    frame = start_frame
    while frame is not None:
        mod = frame.f_globals.get("__name__", "") or ""
        if mod == "diff_diff" or mod.startswith("diff_diff."):
            return True, frame.f_code.co_filename, frame.f_lineno
        frame = frame.f_back
    return False, "", 0


def _install_warning_hook() -> None:
    """Override `warnings.warn` to record warnings whose call stack passes
    through diff_diff. Idempotent."""
    if getattr(warnings.warn, "_pyruntime_wrapped", False):
        return
    original_warn = warnings.warn

    @functools.wraps(original_warn)
    def _pyruntime_warn(message, category=UserWarning, stacklevel=1, source=None, **kwargs):
        # sys._getframe(1) is the immediate caller (one frame up from our
        # wrapper). Walk upward; record if a diff_diff frame is present.
        try:
            matched, frame_filename, frame_lineno = _caller_is_from_diff_diff(sys._getframe(1))
        except ValueError:
            matched, frame_filename, frame_lineno = False, "", 0
        if matched:
            try:
                category_name = category.__name__ if isinstance(category, type) else str(category)
            except Exception:
                category_name = "UserWarning"
            _safe_write(
                {
                    "event": "warning_emitted",
                    "category": category_name,
                    "filename": frame_filename,
                    "lineno": frame_lineno,
                    "message": str(message)[:500],
                    "ts": _utc_iso_now(),
                }
            )
        # Bump stacklevel by 1 to skip this wrapper frame so the caller's
        # intended display filename/lineno are preserved for the user.
        return original_warn(message, category, stacklevel + 1, source, **kwargs)

    _pyruntime_warn._pyruntime_wrapped = True  # type: ignore[attr-defined]
    warnings.warn = _pyruntime_warn  # type: ignore[assignment]


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

    # Re-resolve the guides dir from the freshly-imported module (defense
    # in depth; the top-level resolution via find_spec should already have
    # set _diff_diff_guides_dir before any user code ran).
    if _diff_diff_guides_dir is None:
        try:
            from importlib.resources import files as _resources_files

            guides_traversable = _resources_files("diff_diff.guides")
            _diff_diff_guides_dir = str(guides_traversable)
        except (ImportError, ModuleNotFoundError):
            _diff_diff_guides_dir = None

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
        wrapped = _wrap_diagnostic(original, func_name)
        setattr(module, func_name, wrapped)
        # Mirror to the defining submodule so
        # `from diff_diff.<submod> import <func>` paths reach the wrapper
        # too, matching the `get_llm_guide` dual-binding fix.
        src_module_name = getattr(original, "__module__", None)
        if src_module_name and src_module_name != "diff_diff":
            src_module = sys.modules.get(src_module_name)
            if src_module is not None and getattr(src_module, func_name, None) is original:
                setattr(src_module, func_name, wrapped)


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


# Module-load top-level: record session_start with full identity (raises
# TelemetryConfigError if `_PYRUNTIME_EVENT_LOG` is unset).
_write_event(
    {
        "event": "session_start",
        "ts": _utc_iso_now(),
        "sys_executable": sys.executable,
        "argv": list(getattr(sys, "orig_argv", sys.argv)),
        "pid": os.getpid(),
    }
)

# Resolve the bundled guides directory WITHOUT importing diff_diff. The
# open-hook below needs this to recognize Path/open reads of guide files;
# without top-level resolution, reads that happen before `import diff_diff`
# would silently miss.
try:
    _diff_diff_spec = importlib.util.find_spec("diff_diff")
    if _diff_diff_spec is not None and _diff_diff_spec.origin:
        from pathlib import Path as _Path  # noqa: F401

        _diff_diff_guides_dir = str(_Path(_diff_diff_spec.origin).resolve().parent / "guides")
except (ImportError, ValueError, AttributeError):
    pass

# Insert post-import hook at FRONT of meta_path so it beats the default
# PathFinder for `diff_diff`.
sys.meta_path.insert(0, _DiffDiffPostImportHook())
_install_warning_hook()
_install_open_hook()
