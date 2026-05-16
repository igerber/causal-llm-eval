"""Python runtime instrumentation."""

from __future__ import annotations

import atexit
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


# Statsmodels (arm 2). Statsmodels' submodule layout is not flattened the
# way diff_diff's is, so each entry is (submodule_path, class_name); the
# attach hook imports each submodule and patches the class on it.
_STATSMODELS_ESTIMATOR_CLASSES: tuple[tuple[str, str], ...] = (
    ("statsmodels.regression.linear_model", "OLS"),
    ("statsmodels.regression.linear_model", "WLS"),
    ("statsmodels.regression.linear_model", "GLS"),
    ("statsmodels.regression.linear_model", "GLSAR"),
    ("statsmodels.robust.robust_linear_model", "RLM"),
    ("statsmodels.genmod.generalized_linear_model", "GLM"),
    ("statsmodels.regression.mixed_linear_model", "MixedLM"),
    ("statsmodels.discrete.discrete_model", "Logit"),
    ("statsmodels.discrete.discrete_model", "Probit"),
)


# (submodule_path, function_name) — diagnostics a panel-data analyst plausibly
# invokes during ATT methodology selection. Excludes the long tail of
# specialty (multivariate, time-series) diagnostics that don't apply.
_STATSMODELS_DIAGNOSTIC_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("statsmodels.stats.diagnostic", "het_breuschpagan"),
    ("statsmodels.stats.diagnostic", "het_white"),
    ("statsmodels.stats.diagnostic", "linear_reset"),
    ("statsmodels.stats.diagnostic", "acorr_breusch_godfrey"),
    ("statsmodels.stats.diagnostic", "acorr_ljungbox"),
    ("statsmodels.stats.stattools", "durbin_watson"),
)


# (submodule_path, class_name, method_name) for post-fit results inspection.
# Patched on RegressionResults (the parent class); the wrapper records
# ``type(self).__name__`` at call time so OLSResults / RLMResults /
# MixedLMResults / etc. instances all surface as the correct class without
# enumerating every subclass.
_STATSMODELS_RESULTS_METHODS: tuple[tuple[str, str, str], ...] = (
    ("statsmodels.regression.linear_model", "RegressionResults", "summary"),
    ("statsmodels.regression.linear_model", "RegressionResults", "get_robustcov_results"),
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


# Captured at module load (see top-level code below); ``_write_event`` uses
# these rather than re-reading ``os.environ`` per call. Capturing the path
# once defends against an agent that mutates or deletes
# ``_PYRUNTIME_EVENT_LOG`` after the shim's session_start has fired:
# subsequent hook events would otherwise be diverted to an attacker-chosen
# file (or dropped silently with no marker), leaving the runner-owned log
# with only ``session_start`` while the merger still emits a clean-looking
# record. Holding a fd open from startup adds POSIX-level resilience: the
# fd refers to the inode, not the path, so chmod / rename / unlink of the
# original path do not break later appends.
_EVENT_LOG_PATH: str = ""
_EVENT_LOG_FD: int = -1


def _write_event(event: dict) -> None:
    """Append a single JSON event to the captured per-run event log.

    Writes via the fd held open at module load, not by re-opening the path,
    so the write survives chmod / rename / unlink of ``_EVENT_LOG_PATH``.
    Raises OSError if even the fd-level write fails (rare: out of space,
    fd-table corruption, etc.); callers in hook paths re-raise to the
    ``_safe_write`` hard-exit guard.
    """
    payload = (json.dumps(event) + "\n").encode("utf-8")
    try:
        os.write(_EVENT_LOG_FD, payload)
    except OSError as e:
        print(
            f"[pyruntime] cannot write event to {_EVENT_LOG_PATH} " f"(fd={_EVENT_LOG_FD}): {e}",
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
    """Write an event. Hard-exit on write failure.

    Pre-R22 this caught OSError and continued so a transient telemetry
    failure would not abort the agent's run. The reviewer flagged that
    behavior as silently-incomplete-telemetry: when the agent's command
    redirects stderr (e.g. ``python script.py 2>/dev/null``), the
    ``[pyruntime] cannot write event`` stderr marker disappears, the merger
    has no signal to fail closed, and a dropped ``guide_file_read`` event
    becomes a clean-looking ``opened_llms_*=False``.

    R22 architecture: hold a startup-opened fd (resilient to chmod / rename
    / unlink of the path), and on the remaining unreachable-write case
    terminate the interpreter with ``os._exit(2)`` so the agent's python
    subprocess fails visibly. The marker is still emitted to stderr first
    (caught by cli_stderr.log + Bash tool_result content scans when stderr
    is not redirected); the exit code is the second-line defense when
    stderr suppression hides the marker.
    """
    try:
        _write_event(event)
    except OSError:
        os._exit(2)


def _wrap_get_llm_guide(original):
    """Wrap `get_llm_guide(variant)` to record successful reads. The event is
    emitted AFTER the underlying call returns, so failed reads don't count.

    Only meaningful for diff_diff (statsmodels ships no LLM guides); the
    ``library`` field is fixed to ``"diff_diff"`` so the merger's
    library-attribution filter handles the event uniformly with other
    diff_diff-attributed events.
    """
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(variant: str = "concise"):
        result = original(variant)
        _safe_write(
            {
                "event": "guide_file_read",
                "via": "get_llm_guide",
                "variant": variant,
                "library": "diff_diff",
                "ts": _utc_iso_now(),
            }
        )
        return result

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_estimator_init(original, class_name: str, *, library: str = "diff_diff"):
    """Wrap an estimator class `__init__` to record SUCCESSFUL construction.

    The event is emitted only if the constructor returns; a constructor
    that raises does not produce telemetry. This matters because an agent
    that attempts `CallawaySantAnna(...)` with bad arguments should not
    have CallawaySantAnna in `estimator_classes_instantiated` — the
    attempt failed.

    ``library`` (kw-only) is attached to the event so the merger's
    record-builder filters by arm without re-walking the call stack.
    """
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _safe_write(
            {
                "event": "estimator_init",
                "class": class_name,
                "library": library,
                "ts": _utc_iso_now(),
            }
        )
        return result

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_estimator_fit(original, class_name: str, *, library: str = "diff_diff"):
    """Wrap `fit` method; record AFTER successful return so failed fits
    don't count as use. ``library`` (kw-only) is attached to the event."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _safe_write(
            {
                "event": "estimator_fit",
                "class": class_name,
                "library": library,
                "ts": _utc_iso_now(),
            }
        )
        return result

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_diagnostic(original, name: str, *, library: str = "diff_diff"):
    """Wrap a module-level diagnostic function; record AFTER successful
    return so failed diagnostic calls don't count. ``library`` (kw-only)
    is attached to the event."""
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        _safe_write(
            {
                "event": "diagnostic_call",
                "name": name,
                "library": library,
                "ts": _utc_iso_now(),
            }
        )
        return result

    wrapper._pyruntime_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_results_method(original, method_name: str, *, library: str = "statsmodels"):
    """Wrap a post-fit results method (e.g. ``OLSResults.summary``) to record
    SUCCESSFUL inspection. Records ``type(self).__name__`` at call time so
    a single patch on the parent class (e.g. ``RegressionResults``) attributes
    correctly across subclasses (``OLSResults``, ``RLMResults``,
    ``MixedLMResults``, …) without enumerating them.

    Event type ``estimator_diagnostic_method`` is distinct from
    ``estimator_init`` / ``estimator_fit`` / ``diagnostic_call`` so the
    merger can route them cleanly. Recording is AFTER successful return;
    a call that raises does not produce telemetry.
    """
    if getattr(original, "_pyruntime_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _safe_write(
            {
                "event": "estimator_diagnostic_method",
                "class": type(self).__name__,
                "method": method_name,
                "library": library,
                "ts": _utc_iso_now(),
            }
        )
        return result

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

    NOT caught (known coverage gap, tracked in TODO under "Low-level
    guide-read coverage"):

    - ``os.open`` + ``os.read`` on raw file descriptors.
    - ``pkgutil.get_data("diff_diff.guides", "llms.txt")`` (uses package
      loader's ``get_data`` rather than ``open``).
    - ``mmap.mmap`` + manual byte slicing.
    - C-extension reads via ``ctypes``.

    These vectors are uncommon in agent flows (no production library uses
    them for reading documentation files), but a determined agent could
    use them to read a bundled guide without producing a
    ``guide_file_read`` event. The merger emits ``opened_llms_*=False``
    in that case, which is misleading. Closing this gap is deferred to
    PR #5+ when shim install moves to per-arm-venv site-packages and we
    can revisit the fd-tracking architecture.
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
        # Delegate first; record only if the underlying open succeeded so
        # FileNotFoundError / PermissionError don't count as discovery.
        result = original_open(file, *args, **kwargs)
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
        return result

    _pyruntime_open._pyruntime_wrapped = True  # type: ignore[attr-defined]
    builtins.open = _pyruntime_open  # type: ignore[assignment]
    io.open = _pyruntime_open  # type: ignore[assignment]


# Library prefixes the warning hook attributes against. Order matters: the
# stack walk returns the FIRST matching prefix, so more-specific names should
# come first (none currently — diff_diff and statsmodels share no prefix).
_WARNING_LIBRARY_PREFIXES: tuple[str, ...] = ("diff_diff", "statsmodels")


def _caller_is_from_library(
    start_frame, prefixes: tuple[str, ...] = _WARNING_LIBRARY_PREFIXES
) -> tuple[bool, str, int, str]:
    """Walk the call stack looking for a frame whose module matches one of
    the library ``prefixes`` (e.g. ``("diff_diff", "statsmodels")``).

    Returns ``(matched, filename, lineno, library)``. ``library`` is the
    bare prefix string of the matched frame (e.g. ``"diff_diff"``,
    ``"statsmodels"``) so the caller can attribute the event without
    re-walking. When unmatched, returns ``(False, "", 0, "")``.

    A frame's module matches a prefix iff its ``__name__`` equals the
    prefix or starts with ``"<prefix>."``. The first match wins, so the
    nearest library frame is attributed (avoids cross-attribution when a
    diff_diff function calls a statsmodels function which calls
    ``warnings.warn``).
    """
    frame = start_frame
    while frame is not None:
        mod = frame.f_globals.get("__name__", "") or ""
        for prefix in prefixes:
            if mod == prefix or mod.startswith(prefix + "."):
                return True, frame.f_code.co_filename, frame.f_lineno, prefix
        frame = frame.f_back
    return False, "", 0, ""


def _install_warning_hook() -> None:
    """Override `warnings.warn` to record warnings whose call stack passes
    through diff_diff OR statsmodels. The event carries a ``library``
    field so the merger's record-builder routes the event to the right
    arm without re-walking the stack. Idempotent."""
    if getattr(warnings.warn, "_pyruntime_wrapped", False):
        return
    original_warn = warnings.warn

    @functools.wraps(original_warn)
    def _pyruntime_warn(message, category=UserWarning, stacklevel=1, source=None, **kwargs):
        # sys._getframe(1) is the immediate caller (one frame up from our
        # wrapper). Walk upward; record if a tracked library frame is present.
        try:
            matched, frame_filename, frame_lineno, library = _caller_is_from_library(
                sys._getframe(1)
            )
        except ValueError:
            matched, frame_filename, frame_lineno, library = False, "", 0, ""
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
                    "library": library,
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
    """Patch the loaded `diff_diff` module to record library entry-point calls.

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
            cls.__init__ = _wrap_estimator_init(cls.__init__, class_name, library="diff_diff")
        if hasattr(cls, "fit"):
            cls.fit = _wrap_estimator_fit(cls.fit, class_name, library="diff_diff")

    for func_name in _DIAGNOSTIC_FUNCTION_NAMES:
        original = getattr(module, func_name, None)
        if original is None or not callable(original):
            continue
        wrapped = _wrap_diagnostic(original, func_name, library="diff_diff")
        setattr(module, func_name, wrapped)
        # Mirror to the defining submodule so
        # `from diff_diff.<submod> import <func>` paths reach the wrapper
        # too, matching the `get_llm_guide` dual-binding fix.
        src_module_name = getattr(original, "__module__", None)
        if src_module_name and src_module_name != "diff_diff":
            src_module = sys.modules.get(src_module_name)
            if src_module is not None and getattr(src_module, func_name, None) is original:
                setattr(src_module, func_name, wrapped)


def _attach_statsmodels_hooks(module) -> None:
    """Patch the loaded ``statsmodels`` package to record library entry-point
    calls. Called from ``_StatsmodelsPostImportHook.exec_module`` after Python
    finishes importing ``statsmodels``.

    Walks :data:`_STATSMODELS_ESTIMATOR_CLASSES`, :data:`_STATSMODELS_DIAGNOSTIC_FUNCTIONS`,
    and :data:`_STATSMODELS_RESULTS_METHODS`. For each, imports the named
    submodule (``importlib.import_module``), looks up the attribute, and
    wraps it in place. Imports cascade through statsmodels' submodule
    graph; this is bounded by what the agent would import anyway (the
    agent's first ``import statsmodels.api`` already loads most of these).

    Idempotency is enforced at the per-callable level (each ``_wrap_*``
    sets a ``_pyruntime_wrapped`` flag; re-wrap is a no-op). Submodule
    import failures are tolerated silently (a future statsmodels release
    that drops a submodule should not break shim attach for the rest).
    """
    del module  # signature parity with _attach_diff_diff_hooks; we re-import below
    for submodule_path, class_name in _STATSMODELS_ESTIMATOR_CLASSES:
        try:
            submodule = importlib.import_module(submodule_path)
        except ImportError:
            continue
        cls = getattr(submodule, class_name, None)
        if cls is None:
            continue
        if hasattr(cls, "__init__"):
            cls.__init__ = _wrap_estimator_init(cls.__init__, class_name, library="statsmodels")
        if hasattr(cls, "fit"):
            cls.fit = _wrap_estimator_fit(cls.fit, class_name, library="statsmodels")

    for submodule_path, func_name in _STATSMODELS_DIAGNOSTIC_FUNCTIONS:
        try:
            submodule = importlib.import_module(submodule_path)
        except ImportError:
            continue
        original = getattr(submodule, func_name, None)
        if original is None or not callable(original):
            continue
        wrapped = _wrap_diagnostic(original, func_name, library="statsmodels")
        setattr(submodule, func_name, wrapped)

    for submodule_path, class_name, method_name in _STATSMODELS_RESULTS_METHODS:
        try:
            submodule = importlib.import_module(submodule_path)
        except ImportError:
            continue
        cls = getattr(submodule, class_name, None)
        if cls is None:
            continue
        original = getattr(cls, method_name, None)
        if original is None or not callable(original):
            continue
        setattr(
            cls, method_name, _wrap_results_method(original, method_name, library="statsmodels")
        )


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

    _in_lookup = False

    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname != "diff_diff":
            return None
        # Reentrancy guard: the inner `importlib.util.find_spec("diff_diff")`
        # walks `sys.meta_path` and would call us again. A class-level flag
        # is sufficient (Python imports are serialized by the import lock).
        # We do NOT remove ourselves from meta_path: a plain availability
        # check via `importlib.util.find_spec("diff_diff")` does not run
        # `exec_module`, so removing the hook would permanently lose
        # attachment for a later real `import diff_diff`.
        cls = type(self)
        if cls._in_lookup:
            return None
        cls._in_lookup = True
        try:
            spec = importlib.util.find_spec("diff_diff")
        finally:
            cls._in_lookup = False
        if spec is None or spec.loader is None:
            return spec
        # Idempotency: don't re-wrap if `find_spec` returned the same loader
        # whose `exec_module` we've already wrapped. Double-wrapping would
        # double-fire the `module_import` event and chain `_attach_diff_diff_hooks`
        # (which itself is idempotent, but the event would not be).
        if getattr(spec.loader.exec_module, "_pyruntime_wrapped", False):
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

        _wrapped_exec_module._pyruntime_wrapped = True  # type: ignore[attr-defined]
        spec.loader.exec_module = _wrapped_exec_module  # type: ignore[method-assign]
        return spec


class _StatsmodelsPostImportHook:
    """``sys.meta_path`` finder that triggers hook attachment after
    ``statsmodels`` loads. Architectural mirror of
    :class:`_DiffDiffPostImportHook`.

    Only the top-level ``statsmodels`` package import triggers attachment;
    Python's import machinery guarantees the parent package is imported
    before any submodule, so a single hook firing on the parent is
    sufficient to schedule the wrap-pass over
    :data:`_STATSMODELS_ESTIMATOR_CLASSES` /
    :data:`_STATSMODELS_DIAGNOSTIC_FUNCTIONS` /
    :data:`_STATSMODELS_RESULTS_METHODS`.

    Reentrancy + idempotency semantics are identical to the diff_diff hook;
    see its docstring for the rationale on the in-lookup flag and the
    exec_module wrap idempotency check.
    """

    _in_lookup = False

    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname != "statsmodels":
            return None
        cls = type(self)
        if cls._in_lookup:
            return None
        cls._in_lookup = True
        try:
            spec = importlib.util.find_spec("statsmodels")
        finally:
            cls._in_lookup = False
        if spec is None or spec.loader is None:
            return spec
        if getattr(spec.loader.exec_module, "_pyruntime_wrapped", False):
            return spec
        original_exec_module = spec.loader.exec_module

        def _wrapped_exec_module(module):
            original_exec_module(module)
            _safe_write(
                {
                    "event": "module_import",
                    "module": "statsmodels",
                    "ts": _utc_iso_now(),
                }
            )
            _attach_statsmodels_hooks(module)

        _wrapped_exec_module._pyruntime_wrapped = True  # type: ignore[attr-defined]
        spec.loader.exec_module = _wrapped_exec_module  # type: ignore[method-assign]
        return spec


# ``_EVENT_LOG_PATH`` and ``_EVENT_LOG_FD`` are declared at module top
# (lines ~78-79) with sentinel values ("" and -1). The gate below rebinds
# them when Python's site machinery loads the file as ``sitecustomize``.
# Tests / docs tooling that import the module as ``harness.sitecustomize_template``
# see the sentinel values; any call to ``_write_event()`` then OSError's
# (fd=-1 is invalid) which is the desired fail-closed behavior outside the
# gated production path.
_initial_path: str | None = None


def _write_session_end() -> None:
    """Record a ``session_end`` event paired by pid with this process's
    ``session_start``.

    Registered via ``atexit`` so it fires on normal interpreter
    shutdown, sys.exit(), and unhandled exceptions. It does NOT fire on
    ``os._exit()`` (the shim's hard-exit path on event-write failure)
    or on SIGKILL - which is exactly what the merger uses to detect
    that the run completed without the shim's full attestation. An
    agent that masks shell exit status with ``2>/dev/null || true``
    cannot synthesize a session_end because they cannot run code after
    Python has exited via os._exit.

    Best-effort: any write failure here is silent because os._exit-on-
    failure during atexit would interfere with Python's own shutdown
    cleanup, and a missing session_end is already the fail-closed
    signal the merger requires.
    """
    try:
        _write_event(
            {
                "event": "session_end",
                "ts": _utc_iso_now(),
                "pid": os.getpid(),
            }
        )
    except OSError:
        # Already in shutdown; missing session_end is the merger's
        # fail-closed signal regardless of why it's missing.
        pass


def _install_production_state() -> None:
    """Production initialization: env-var check, fd open, session_start
    emit, atexit register, hook install.

    Called exactly once when Python's site machinery processes the
    ``_pyruntime_shim.pth`` file in the per-arm venv's site-packages,
    which executes ``import _pyruntime_shim`` and triggers the
    ``__name__ == "_pyruntime_shim"`` gate below. (PR #6 changed the
    load mechanism from a stdlib-shadow-vulnerable ``sitecustomize.py``
    install to a robust ``.pth``-based load.) Also called by test
    fixtures via ``_import_shim_fresh()`` to fire the side effects on the
    importlib-imported module's namespace (so classes defined here and
    instances inserted into ``sys.meta_path`` are the same identity the
    tests see via ``shim.<attr>`` access).

    Idempotent guard: if the meta-path hook is already installed, this
    is a no-op. Used by tests that may call ``_import_shim_fresh()``
    multiple times.
    """
    global _initial_path, _EVENT_LOG_PATH, _EVENT_LOG_FD, _diff_diff_guides_dir
    # Resolve and capture the event log path FIRST so subsequent hook events
    # use this captured value, not whatever ``os.environ`` happens to hold
    # when the hook fires (an agent could mutate or unset the env var
    # mid-run; capturing here pins the file the runner owns).
    _initial_path = os.environ.get("_PYRUNTIME_EVENT_LOG")
    if not _initial_path:
        # Hard exit: Python's site machinery normally catches ordinary
        # exceptions raised from sitecustomize and continues running. That
        # would let an agent's Python subprocess start without telemetry
        # capture, silently. Print to stderr (so the runner / merger can
        # see the marker via layer-3 or tool_result scan) and terminate the
        # interpreter immediately.
        print(
            "[pyruntime] _PYRUNTIME_EVENT_LOG is unset; sitecustomize cannot "
            "initialize the runtime event log",
            file=sys.stderr,
        )
        os._exit(2)
    _EVENT_LOG_PATH = _initial_path

    # Open the event log fd ONCE at startup and hold it for the lifetime of
    # the interpreter. Resilient to chmod / rename / unlink of the path:
    # the fd refers to the inode, not the pathname, so the agent cannot
    # break later writes by mutating the filesystem entry. If even this
    # initial open fails (path unwritable / parent dir gone), hard-exit so
    # the agent subprocess fails visibly rather than running without
    # telemetry.
    try:
        _EVENT_LOG_FD = os.open(
            _EVENT_LOG_PATH,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
    except OSError as _open_err:
        print(
            f"[pyruntime] cannot open event log {_EVENT_LOG_PATH}: {_open_err}",
            file=sys.stderr,
        )
        os._exit(2)

    # Now record session_start with full identity. Use the hard-exit
    # variant: if the very first event write fails, sitecustomize's
    # ordinary exception propagation would let Python continue running
    # without hooks (Python's site machinery catches sitecustomize errors
    # and continues). The session_start failure is the same fail-closed
    # class as the hook write failures - leave no path where the agent
    # runs without instrumentation.
    _safe_write(
        {
            "event": "session_start",
            "ts": _utc_iso_now(),
            "sys_executable": sys.executable,
            "argv": list(getattr(sys, "orig_argv", sys.argv)),
            "pid": os.getpid(),
        }
    )

    atexit.register(_write_session_end)

    # Resolve the bundled guides directory WITHOUT importing diff_diff. The
    # open-hook below needs this to recognize Path/open reads of guide
    # files; without top-level resolution, reads that happen before
    # `import diff_diff` would silently miss.
    try:
        _diff_diff_spec = importlib.util.find_spec("diff_diff")
        if _diff_diff_spec is not None and _diff_diff_spec.origin:
            from pathlib import Path as _Path  # noqa: F401

            _diff_diff_guides_dir = str(_Path(_diff_diff_spec.origin).resolve().parent / "guides")
    except (ImportError, ValueError, AttributeError):
        pass

    # Insert post-import hooks at FRONT of meta_path so they beat the default
    # PathFinder for the respective parent packages. Each hook's ``find_spec``
    # returns ``None`` for non-matching names; coexistence is by-construction.
    sys.meta_path.insert(0, _DiffDiffPostImportHook())
    sys.meta_path.insert(0, _StatsmodelsPostImportHook())
    _install_warning_hook()
    _install_open_hook()

    # PR #5 R1: route ``subprocess.Popen([sys.executable, ...])`` and
    # equivalent child-spawn idioms through the layer-1.5 wrapper rather
    # than the real interpreter. After exec'ing through the wrapper +
    # strip-S shim chain, the real python initializes ``sys.executable``
    # to the path the kernel exec'd: either
    # ``${venv}/.pyruntime-real/python-real`` (R1 single-stage layout)
    # or ``${venv}/.pyruntime-real/.actual-python`` (R3 two-stage
    # strip-S layout). Either way, walk up two directory levels to find
    # the venv root, then rewrite ``sys.executable`` to the wrapper at
    # ``${venv}/bin/python``. Subsequent ``subprocess.Popen``
    # invocations re-enter the wrapper, which records exec_python and
    # exec's the chain again.
    #
    # PR #5 R15 P1 (EV-1): ``sys.executable`` is not the only Python
    # interpreter handle. ``sys._base_executable`` (CPython 3.11+) is
    # set during venv site initialization to the BASE interpreter
    # (typically the one outside the per-run venv) and is what
    # ``venv.EnvBuilder``, ``subprocess`` in some pip / pyenv idioms,
    # and any agent that does ``subprocess.run([sys._base_executable,
    # ...])`` will use. Leaving it untouched is an uninstrumented
    # escape hatch: a child python spawned through it skips the
    # wrapper, the AST parser sees only the opaque attribute access
    # (no python invocation in the visible argv), and layer-2
    # sitecustomize for that child process loads from whatever venv
    # the base interpreter is in (NOT the per-run venv with our
    # shim). Rewrite it alongside ``sys.executable`` so all
    # documented sys-attribute interpreter handles route through the
    # wrapper.
    _real_executable = sys.executable
    if _real_executable.endswith("/python-real") or _real_executable.endswith("/.actual-python"):
        _venv_root = os.path.dirname(os.path.dirname(_real_executable))
        _wrapper_path = os.path.join(_venv_root, "bin", "python")
        if os.path.exists(_wrapper_path):
            sys.executable = _wrapper_path
            if hasattr(sys, "_base_executable"):
                sys._base_executable = _wrapper_path


if __name__ == "_pyruntime_shim":
    # Production path: Python's site machinery processes ``_pyruntime_shim.pth``
    # in the per-arm venv's site-packages, which executes ``import _pyruntime_shim``,
    # which loads this file (installed by ``_install_shim_into_venv`` as
    # ``_pyruntime_shim.py``).
    #
    # The .pth-based load supersedes the previous ``sitecustomize.py``-based
    # load: a Homebrew-installed ``sitecustomize.py`` in the system stdlib
    # (introduced in homebrew python@3.13 around Feb 2026) shadows any
    # ``sitecustomize.py`` we install in venv site-packages because stdlib
    # comes before site-packages in sys.path. .pth files are processed
    # during site-init regardless of stdlib's sitecustomize, so this load
    # path is robust to operator system-Python configuration.
    #
    # Tests / docs tooling that import the module as
    # ``harness.sitecustomize_template`` skip this block and call
    # ``_install_production_state()`` explicitly via ``_import_shim_fresh()``.
    _install_production_state()
