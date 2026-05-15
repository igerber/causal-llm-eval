#!/bin/sh
# Layer-1.5 exec wrapper around the per-arm venv's python interpreter.
#
# Installed by harness.venv_pool._install_python_wrapper. For every invocation
# of ${venv}/bin/python (or python3, python3.X), this wrapper appends one
# JSONL ``exec_python`` event to ${_PYRUNTIME_EVENT_LOG} and then ``exec``s
# the real interpreter at ${venv}/bin/python-real. The event captures the
# argv that reached the wrapper plus pid + ppid so the merger can attribute
# the invocation across the three-layer attestation chain.
#
# See harness/COLD_START_VERIFICATION.md for the architecture.
#
# Failure semantics (matches sitecustomize's fail-closed posture):
#   - log path unset: skip the write (interpreter still runs; layer-1 AST
#     still detects the invocation; bypass detection separately fails-closed
#     on env -i / env -u that strips the var). Layer-2 sitecustomize will
#     ALSO hard-exit when the var is unset, so the spawned interpreter
#     terminates before user code runs.
#   - log path set but unwritable: exit 2 (same as sitecustomize on event
#     write failure).
#   - argv encoding failure: exit 2 (best-effort NUL byte detection).
#
# POSIX-only: no bashisms (no [[ ]], no arrays, no ${var//pat/repl}).
# Tested compatible with dash, mawk/gawk/busybox awk, macOS bash 3.2 in
# POSIX mode.
#
# PR #5 R9 P0: pin internal command resolution to a trusted system PATH
# so an agent that drops malicious files into ${venv}/bin/ (e.g.,
# venv/bin/sh, venv/bin/awk, venv/bin/dirname, venv/bin/date) cannot
# hijack the wrapper's helper utilities. We save the agent's PATH
# (which the runner prepends with ${venv}/bin) and restore it just
# before exec'ing python-real, so the agent's own python invocations
# still resolve correctly.
agent_path="${PATH-}"
PATH="/usr/bin:/bin"
export PATH

# Resolve $0 to its absolute directory so the recorded ``executable``
# path is canonical regardless of how the wrapper was invoked
# (``python script.py``, ``./venv/bin/python script.py``,
# ``${venv}/bin/python script.py``). The merger normalizes both sides
# but a relative recorded path would not match the venv-root-anchored
# allowlist.
case "$0" in
    /*) wrapper_dir="$(dirname "$0")" ;;
    *)  wrapper_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" ;;
esac
# Resolve /.. by chdir + pwd (POSIX-portable; no realpath dependency).
real_dir="$(cd "${wrapper_dir}/../.pyruntime-real" 2>/dev/null && pwd)"
real="${real_dir}/python-real"

# PR #5 R8 P0: compute the canonical runner-owned event log path from
# the wrapper's own location and OVERRIDE any inherited
# _PYRUNTIME_EVENT_LOG before exec'ing the real interpreter. The
# runner sets the env var to ``${tmpdir}/.pyruntime/events.jsonl`` and
# installs the wrapper at ``${tmpdir}/venv/bin/python``, so walking up
# two directories from wrapper_dir lands at ``${tmpdir}`` and the
# canonical log path is reachable. This closes the env-retarget bypass
# class structurally: any agent code that sets
# ``_PYRUNTIME_EVENT_LOG=/tmp/fake`` (via shell script, Python
# subprocess env=, etc.) has its mutation overridden by the wrapper
# before the child's events fire.
#
# Fallback: if walking up two directories does not land at a tree
# containing ``.pyruntime/`` (e.g., the wrapper is invoked outside a
# production venv layout, like in unit-test fixtures), fall back to
# the inherited env var. Test-only invocations still work; production
# always overrides.
canonical_root="$(cd "${wrapper_dir}/../.." 2>/dev/null && pwd)"
canonical_log="${canonical_root}/.pyruntime/events.jsonl"
if [ -d "${canonical_root}/.pyruntime" ]; then
    export _PYRUNTIME_EVENT_LOG="$canonical_log"
    log="$canonical_log"
else
    log="${_PYRUNTIME_EVENT_LOG:-}"
fi
if [ -n "$log" ]; then
    # JSON-encode argv via a single awk invocation. The recorded argv is
    # ``[basename($0), $@]`` so the merger can match on ``argv[1:]``
    # against the layer-1 AST extraction (which produces
    # ``[interpreter, *args]``) and against layer-2's ``sys.orig_argv[1:]``
    # (which the real python records post-exec).
    #
    # POSIX argv may legally contain newlines, but the line-oriented
    # printf-awk pipeline cannot preserve embedded newlines (one arg with
    # a newline splits into two records). The wrapper fails closed on
    # newline/CR via awk's NR record count: if awk processes more records
    # than (expected_args + 1), an arg contained an embedded newline.
    # The +1 accounts for the basename($0) prepended to the stream.
    #
    # Note: $(printf '\n') strips its own trailing newline and would
    # produce an empty string; case-pattern matching against an empty
    # substring matches every argv. Using awk's NR counter avoids this
    # POSIX shell-quoting trap entirely.
    arg0_basename=$(basename "$0")
    expected_records=$(($# + 1))
    args_json=$(printf '%s\n' "$arg0_basename" "$@" | awk -v expected="$expected_records" '
        BEGIN { first = 1; printf "[" }
        {
            gsub(/\\/, "\\\\")
            gsub(/"/, "\\\"")
            gsub(/\010/, "\\b")
            gsub(/\014/, "\\f")
            gsub(/\015/, "\\r")
            gsub(/\t/, "\\t")
            if (first) { first = 0 } else { printf "," }
            printf "\"%s\"", $0
        }
        END {
            printf "]"
            if (NR != expected) {
                printf "RECORD_COUNT_MISMATCH" > "/dev/stderr"
                exit 2
            }
        }
    ') || {
        printf '[pyruntime-wrapper] argv contains embedded newline; cannot attest\n' >&2
        exit 2
    }
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    line="{\"event\":\"exec_python\",\"pid\":$$,\"ppid\":${PPID:-0},\"ts\":\"${ts}\",\"executable\":\"${real}\",\"argv\":${args_json}}"
    printf '%s\n' "$line" >> "$log" || {
        printf '[pyruntime-wrapper] cannot append to %s\n' "$log" >&2
        exit 2
    }
fi

# PR #5 R9 P0: restore the agent's PATH (which the runner prepended
# with ${venv}/bin) before exec'ing python-real. Agent code that
# spawns subprocesses via PATH resolution still gets the same view it
# had on entry. Only the wrapper's INTERNAL command resolution was
# pinned to /usr/bin:/bin.
PATH="$agent_path"
export PATH

exec "$real" "$@"
