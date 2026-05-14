"""Unit tests for `harness/sitecustomize_template.py`.

The shim's top-level code (session_start + meta_path append + warning hook)
runs at import. Tests that exercise top-level behavior use `importlib.reload`
with `_PYRUNTIME_EVENT_LOG` set. Tests that exercise wrappers in isolation
call the builders (`_wrap_*`) directly without re-running top-level.

The `restore_globals` fixture saves and restores `warnings.showwarning`,
`builtins.open`, and `sys.meta_path` around each test to prevent cross-test
pollution.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import io
import json
import sys
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _unwrap(fn):
    """Walk the __wrapped__ chain to recover the underlying original.

    Other tests in the suite (test_harness_imports) trigger the shim's
    top-level open/warn hooks without a cleanup fixture, so the "current"
    `builtins.open` / `warnings.warn` may already be wrapped when this
    fixture's save runs. Walking the chain finds the true original.
    """
    while getattr(fn, "__wrapped__", None) is not None:
        fn = fn.__wrapped__
    return fn


@pytest.fixture
def restore_globals():
    """Save and restore module globals the shim mutates."""
    original_showwarning = _unwrap(warnings.showwarning)
    original_warn = _unwrap(warnings.warn)
    original_builtins_open = _unwrap(builtins.open)
    original_io_open = _unwrap(io.open)
    original_meta_path = list(sys.meta_path)
    yield
    warnings.showwarning = original_showwarning
    warnings.warn = original_warn
    builtins.open = original_builtins_open
    io.open = original_io_open
    sys.meta_path[:] = original_meta_path
    sys.modules.pop("harness.sitecustomize_template", None)


@pytest.fixture
def event_log(tmp_path, monkeypatch, restore_globals):
    """Set `_PYRUNTIME_EVENT_LOG` to a tmp file and return its Path."""
    path = tmp_path / "events.jsonl"
    path.touch()
    monkeypatch.setenv("_PYRUNTIME_EVENT_LOG", str(path))
    return path


def _read_events(path: Path) -> list[dict]:
    """Parse the per-test event log into a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _import_shim_fresh():
    """Import `harness.sitecustomize_template` triggering its top-level code.

    Drops any cached copy first so module-load side effects (session_start
    write, hook install) re-run.
    """
    sys.modules.pop("harness.sitecustomize_template", None)
    return importlib.import_module("harness.sitecustomize_template")


# ---------------------------------------------------------------------------
# Top-level / module-load behavior
# ---------------------------------------------------------------------------


def test_session_start_event_written_on_import(event_log):
    _import_shim_fresh()
    events = _read_events(event_log)
    assert any(e.get("event") == "session_start" for e in events), events
    session_events = [e for e in events if e.get("event") == "session_start"]
    assert "ts" in session_events[0]
    # ISO 8601 microsecond timestamp; just check the rough shape
    assert "T" in session_events[0]["ts"]


def test_session_start_hard_exits_without_env_var(monkeypatch, restore_globals, capsys):
    """R21 P1: when ``_PYRUNTIME_EVENT_LOG`` is unset at shim import, the
    shim calls ``os._exit(2)`` rather than raising. Python's site machinery
    catches ordinary exceptions from sitecustomize.py and continues the
    user program; only a hard exit reliably stops the subprocess. The
    test mocks ``os._exit`` to capture the exit code instead of killing
    pytest."""
    monkeypatch.delenv("_PYRUNTIME_EVENT_LOG", raising=False)
    sys.modules.pop("harness.sitecustomize_template", None)

    captured_exit_codes: list[int] = []

    def fake_exit(code):
        captured_exit_codes.append(code)
        # Raise to abort module load (mimics process termination).
        raise SystemExit(code)

    monkeypatch.setattr("os._exit", fake_exit)
    with pytest.raises(SystemExit):
        importlib.import_module("harness.sitecustomize_template")
    assert captured_exit_codes == [2]
    captured = capsys.readouterr()
    assert "[pyruntime]" in captured.err
    assert "_PYRUNTIME_EVENT_LOG is unset" in captured.err


def test_event_log_path_captured_at_import_resists_env_mutation(event_log, monkeypatch, tmp_path):
    """R14 P0: an agent that mutates ``_PYRUNTIME_EVENT_LOG`` after the
    shim's session_start has fired must NOT be able to redirect later hook
    events. The shim captures the path at module load and uses that
    captured value for every subsequent write; agent mutations are
    ignored.

    Pre-R14 ``_write_event`` re-read ``os.environ`` per call. An agent
    could point ``_PYRUNTIME_EVENT_LOG`` at a sibling tmp file (or
    delete it entirely) before invoking a tracked surface; subsequent
    layer-2 events would land elsewhere or fail silently while the
    runner-owned log retained only ``session_start``. The merger then
    emitted a clean-looking record with all-False guide flags.

    This test imports the shim with a known path, mutates the env var to
    a different (decoy) file, fires a wrapper, and asserts that the
    event landed in the ORIGINAL captured path, not the decoy.
    """
    shim = _import_shim_fresh()
    decoy = tmp_path / "decoy_events.jsonl"
    monkeypatch.setenv("_PYRUNTIME_EVENT_LOG", str(decoy))

    # Fire a wrapper after the env-var mutation. The wrapped builder
    # exists at module level; calling it should still write to the
    # captured path, NOT the decoy.
    wrapped = shim._wrap_get_llm_guide(lambda variant="concise": variant)
    wrapped("practitioner")

    # Decoy must remain empty / non-existent.
    assert (
        not decoy.exists() or decoy.stat().st_size == 0
    ), f"event landed in decoy {decoy}; env-var mutation hijacked the shim"
    # Original captured log must contain the guide_file_read event.
    events = _read_events(event_log)
    guide_events = [e for e in events if e.get("event") == "guide_file_read"]
    assert any(e.get("variant") == "practitioner" for e in guide_events), events


# ---------------------------------------------------------------------------
# Meta-path post-import hook
# ---------------------------------------------------------------------------


def test_meta_path_hook_returns_none_for_non_diff_diff_names(event_log):
    shim = _import_shim_fresh()
    hook = shim._DiffDiffPostImportHook()
    assert hook.find_spec("numpy", None, None) is None
    assert hook.find_spec("os", None, None) is None
    assert hook.find_spec("sys", None, None) is None
    assert hook.find_spec("statsmodels.regression.linear_model", None, None) is None


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_meta_path_hook_records_module_import(event_log):
    # Drop diff_diff from sys.modules so the meta_path hook fires.
    for name in list(sys.modules):
        if name == "diff_diff" or name.startswith("diff_diff."):
            del sys.modules[name]
    _import_shim_fresh()
    import diff_diff  # noqa: F401

    events = _read_events(event_log)
    assert any(
        e.get("event") == "module_import" and e.get("module") == "diff_diff" for e in events
    ), events


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_meta_path_hook_survives_find_spec_call(event_log):
    """R12 P0: a plain ``importlib.util.find_spec("diff_diff")``
    availability check (without subsequent import / exec_module) must
    NOT consume the hook. Pre-R12 the hook removed itself from
    sys.meta_path at find_spec time and never re-added; a later real
    ``import diff_diff`` then loaded uninstrumented (silent layer-2
    loss). Post-R12 the hook uses a reentrancy guard and stays on
    meta_path."""
    for name in list(sys.modules):
        if name == "diff_diff" or name.startswith("diff_diff."):
            del sys.modules[name]
    shim = _import_shim_fresh()

    # Simulate the availability-check pattern: find_spec without import.
    spec = importlib.util.find_spec("diff_diff")
    assert spec is not None, "diff_diff not findable in test venv"

    # Hook must STILL be present on meta_path after find_spec.
    assert any(
        isinstance(finder, shim._DiffDiffPostImportHook) for finder in sys.meta_path
    ), "shim hook was consumed by find_spec; re-importing diff_diff would be uninstrumented"

    # And a subsequent real `import` must still trigger module_import + attach.
    import diff_diff  # noqa: F401

    events = _read_events(event_log)
    module_import_count = sum(
        1 for e in events if e.get("event") == "module_import" and e.get("module") == "diff_diff"
    )
    # Exactly one module_import event: idempotency guard prevents
    # double-fire when find_spec is called before the real import.
    assert (
        module_import_count == 1
    ), f"expected exactly 1 module_import event, got {module_import_count}: {events}"


# ---------------------------------------------------------------------------
# Wrapper builders (exercised without importing real diff_diff)
# ---------------------------------------------------------------------------


def test_get_llm_guide_wrapper_records_variant(event_log):
    shim = _import_shim_fresh()
    fake_original = lambda variant="concise": f"contents of {variant}"  # noqa: E731
    wrapped = shim._wrap_get_llm_guide(fake_original)
    result = wrapped("practitioner")
    assert result == "contents of practitioner"
    events = _read_events(event_log)
    guide_events = [e for e in events if e.get("event") == "guide_file_read"]
    assert len(guide_events) == 1
    assert guide_events[0]["via"] == "get_llm_guide"
    assert guide_events[0]["variant"] == "practitioner"


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_get_llm_guide_wrapper_catches_submodule_import_path(event_log):
    # Clear diff_diff from cache so the meta_path hook fires and attaches.
    for name in list(sys.modules):
        if name == "diff_diff" or name.startswith("diff_diff."):
            del sys.modules[name]
    _import_shim_fresh()

    # Trigger parent-package import first (always happens for submodule imports).
    # Then submodule-direct binding to verify the wrapper is reached.
    from diff_diff._guides_api import get_llm_guide

    _ = get_llm_guide("concise")
    events = _read_events(event_log)
    via_events = [
        e for e in events if e.get("event") == "guide_file_read" and e.get("via") == "get_llm_guide"
    ]
    assert len(via_events) >= 1, events
    assert any(e.get("variant") == "concise" for e in via_events), events


def test_estimator_init_wrapper_records_class_name(event_log):
    shim = _import_shim_fresh()

    class FakeEstimator:
        def __init__(self, x, y=None):
            self.x = x
            self.y = y

    FakeEstimator.__init__ = shim._wrap_estimator_init(FakeEstimator.__init__, "FakeEstimator")
    inst = FakeEstimator(42, y="hello")
    assert inst.x == 42
    assert inst.y == "hello"
    events = _read_events(event_log)
    init_events = [e for e in events if e.get("event") == "estimator_init"]
    assert len(init_events) == 1
    assert init_events[0]["class"] == "FakeEstimator"


def test_estimator_fit_wrapper_records_class_name(event_log):
    shim = _import_shim_fresh()

    class FakeEstimator:
        def fit(self, data):
            return data * 2

    FakeEstimator.fit = shim._wrap_estimator_fit(FakeEstimator.fit, "FakeEstimator")
    inst = FakeEstimator()
    result = inst.fit(7)
    assert result == 14
    events = _read_events(event_log)
    fit_events = [e for e in events if e.get("event") == "estimator_fit"]
    assert len(fit_events) == 1
    assert fit_events[0]["class"] == "FakeEstimator"


def test_diagnostic_wrapper_records_function_name(event_log):
    shim = _import_shim_fresh()
    fake_diag = lambda data, alpha=0.05: alpha * data  # noqa: E731
    wrapped = shim._wrap_diagnostic(fake_diag, "fake_diagnostic")
    result = wrapped(100, alpha=0.1)
    assert result == pytest.approx(10.0)
    events = _read_events(event_log)
    diag_events = [e for e in events if e.get("event") == "diagnostic_call"]
    assert len(diag_events) == 1
    assert diag_events[0]["name"] == "fake_diagnostic"


def test_warning_hook_records_diff_diff_warning(event_log):
    """Verify warning capture via stack inspection: a warning emitted from
    a frame whose module is `diff_diff.*` is recorded regardless of the
    `stacklevel` argument used."""
    shim = _import_shim_fresh()
    shim._install_warning_hook()
    # Exec the warn() call inside a globals dict whose __name__ is
    # diff_diff-like; the wrapper's stack walk finds this frame.
    diff_diff_globals = {
        "__name__": "diff_diff.estimators",
        "warnings": warnings,
    }
    exec("warnings.warn('example warning')", diff_diff_globals)
    events = _read_events(event_log)
    warn_events = [e for e in events if e.get("event") == "warning_emitted"]
    assert len(warn_events) >= 1
    assert warn_events[-1]["category"] == "UserWarning"


def test_warning_hook_ignores_non_diff_diff_warning(event_log):
    shim = _import_shim_fresh()
    shim._install_warning_hook()
    pre = len([e for e in _read_events(event_log) if e.get("event") == "warning_emitted"])
    numpy_globals = {"__name__": "numpy.core", "warnings": warnings}
    exec("warnings.warn('numpy warning')", numpy_globals)
    post = len([e for e in _read_events(event_log) if e.get("event") == "warning_emitted"])
    assert post == pre, "warning from non-diff_diff frame should not be recorded"


def test_warning_hook_records_diff_diff_warning_with_stacklevel(event_log):
    """R4 P1 regression: a diff_diff warning emitted with `stacklevel=2`
    rewrites the displayed filename to point at the user's caller, but the
    actual call stack still contains a diff_diff frame. The new warn-wrapper
    must record it via stack inspection."""
    shim = _import_shim_fresh()
    shim._install_warning_hook()
    diff_diff_globals = {
        "__name__": "diff_diff.estimators",
        "warnings": warnings,
    }
    # stacklevel=2 would point the displayed filename at the user; verify
    # the wrapper still records the warning.
    exec(
        "warnings.warn('warn with stacklevel', stacklevel=2)",
        diff_diff_globals,
    )
    events = _read_events(event_log)
    warn_events = [e for e in events if e.get("event") == "warning_emitted"]
    assert len(warn_events) >= 1
    assert warn_events[-1]["message"].startswith("warn with stacklevel")


# ---------------------------------------------------------------------------
# Fail-open + idempotency contract
# ---------------------------------------------------------------------------


def test_wrapper_propagates_telemetry_write_failure_to_stderr_but_calls_original(
    event_log, monkeypatch
):
    del event_log
    shim = _import_shim_fresh()
    called = []

    def fake_original(x):
        called.append(x)
        return x + 1

    wrapped = shim._wrap_diagnostic(fake_original, "test_diag")

    def boom(event):
        del event
        raise OSError("simulated disk full")

    monkeypatch.setattr(shim, "_write_event", boom)
    # Wrapper must STILL call original and return its value, even when
    # _write_event raises.
    result = wrapped(5)
    assert result == 6
    assert called == [5]


def test_wrap_is_idempotent_on_double_attach(event_log):
    shim = _import_shim_fresh()
    fake_original = lambda x: x * 2  # noqa: E731
    once = shim._wrap_diagnostic(fake_original, "fake_diag")
    twice = shim._wrap_diagnostic(once, "fake_diag")
    # The double-wrap must return the same object as the single-wrap,
    # so calling `twice(5)` writes exactly ONE diagnostic_call event.
    assert once is twice
    _ = twice(5)
    events = _read_events(event_log)
    diag_events = [e for e in events if e.get("event") == "diagnostic_call"]
    assert len(diag_events) == 1


# ---------------------------------------------------------------------------
# Bidirectional regression: constants must match diff_diff exports
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_estimator_class_names_bidirectional_against_diff_diff_exports(event_log):
    import diff_diff

    shim = _import_shim_fresh()
    listed = set(shim._ESTIMATOR_CLASS_NAMES)
    # Forward: every name resolves to a public class on diff_diff with .fit
    for name in listed:
        obj = getattr(diff_diff, name, None)
        assert (
            obj is not None
        ), f"_ESTIMATOR_CLASS_NAMES has {name} but diff_diff does not export it"
        assert isinstance(obj, type), f"diff_diff.{name} is not a class"
        assert hasattr(obj, "fit"), f"diff_diff.{name} has no .fit method"
    # Reverse: every canonical fit-bearing class in __all__ is in the list.
    # Canonical name detection: a class's __name__ is fixed at definition time
    # via `class X:`. Aliases (e.g., `Bacon = BaconDecomposition`) share the
    # same class object, so `Bacon.__name__ == "BaconDecomposition"`. When the
    # __all__ name differs from __name__, the entry is an alias.
    all_exports = getattr(diff_diff, "__all__", [])
    canonical_classes: set[str] = set()
    for name in all_exports:
        obj = getattr(diff_diff, name, None)
        if not isinstance(obj, type) or not hasattr(obj, "fit"):
            continue
        if name != obj.__name__:
            continue  # alias
        canonical_classes.add(name)
    missing = canonical_classes - listed
    assert not missing, (
        f"diff_diff.__all__ exports these fit-bearing classes that are "
        f"NOT in _ESTIMATOR_CLASS_NAMES: {sorted(missing)}. Update the "
        f"shim to add them."
    )


@pytest.mark.skipif(
    importlib.util.find_spec("diff_diff") is None,
    reason="diff_diff not importable in this venv",
)
def test_diagnostic_function_names_match_diff_diff_exports(event_log):
    import diff_diff

    shim = _import_shim_fresh()
    for name in shim._DIAGNOSTIC_FUNCTION_NAMES:
        obj = getattr(diff_diff, name, None)
        assert obj is not None, f"diff_diff has no export named {name}"
        assert callable(obj), f"diff_diff.{name} is not callable"


def test_message_capped_at_500_chars(event_log):
    shim = _import_shim_fresh()
    shim._install_warning_hook()
    diff_diff_globals = {"__name__": "diff_diff.estimators", "warnings": warnings}
    exec("warnings.warn('x' * 10000)", diff_diff_globals)
    events = _read_events(event_log)
    warn_events = [e for e in events if e.get("event") == "warning_emitted"]
    assert len(warn_events) >= 1
    assert len(warn_events[-1]["message"]) <= 500


# ---------------------------------------------------------------------------
# Builtins.open guide-file hook
# ---------------------------------------------------------------------------


def test_open_hook_does_not_record_non_guide_paths(event_log, tmp_path):
    shim = _import_shim_fresh()
    # Capture initial open-via reads
    pre = [
        e
        for e in _read_events(event_log)
        if e.get("event") == "guide_file_read" and e.get("via") == "open"
    ]
    # Set the guide dir to a known location; create a non-guide file and read it.
    setattr(shim, "_diff_diff_guides_dir", str(tmp_path))
    shim._install_open_hook()
    non_guide = tmp_path / "random_data.csv"
    non_guide.write_text("col1,col2\n1,2\n")
    _ = non_guide.read_text()
    post = [
        e
        for e in _read_events(event_log)
        if e.get("event") == "guide_file_read" and e.get("via") == "open"
    ]
    assert len(post) == len(pre), "non-guide file read should not be recorded"


def test_open_hook_records_guide_file_paths(event_log, tmp_path):
    shim = _import_shim_fresh()
    setattr(shim, "_diff_diff_guides_dir", str(tmp_path))
    shim._install_open_hook()
    guide_file = tmp_path / "llms-practitioner.txt"
    guide_file.write_text("guide contents")
    _ = guide_file.read_text()
    events = _read_events(event_log)
    open_events = [
        e
        for e in events
        if e.get("event") == "guide_file_read"
        and e.get("via") == "open"
        and e.get("filename") == "llms-practitioner.txt"
    ]
    assert len(open_events) >= 1, events
