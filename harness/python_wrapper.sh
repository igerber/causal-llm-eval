#!/usr/bin/env sh
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

real="$(dirname "$0")/python-real"

log="${_PYRUNTIME_EVENT_LOG:-}"
if [ -n "$log" ]; then
    # JSON-encode argv via a single awk invocation. awk's gsub handles all
    # required JSON control-char escapes (\b \f \n \r \t \" \\) uniformly.
    # POSIX.1 awk is portable across mawk / gawk / nawk / busybox awk for
    # printf %s, gsub, and ASCII control codes.
    #
    # NUL bytes cannot legally appear in POSIX argv (execve contract), so the
    # wrapper trusts the kernel-provided argv and does not attempt awk-level
    # NUL detection. (Earlier `/\000/` checks were unreliable across awk
    # variants — macOS BSD awk treats `\000` as zero-width / always-match.)
    # Upstream argv sanitization at the merger remains the primary defense
    # for any pathological inputs.
    args_json=$(printf '%s\n' "$@" | awk '
        BEGIN { first = 1; printf "[" }
        {
            gsub(/\\/, "\\\\")
            gsub(/"/, "\\\"")
            gsub(/\010/, "\\b")
            gsub(/\014/, "\\f")
            gsub(/\n/, "\\n")
            gsub(/\r/, "\\r")
            gsub(/\t/, "\\t")
            if (first) { first = 0 } else { printf "," }
            printf "\"%s\"", $0
        }
        END { printf "]" }
    ') || {
        printf '[pyruntime-wrapper] argv encoding failed\n' >&2
        exit 2
    }
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    line="{\"event\":\"exec_python\",\"pid\":$$,\"ppid\":${PPID:-0},\"ts\":\"${ts}\",\"executable\":\"${real}\",\"argv\":${args_json}}"
    printf '%s\n' "$line" >> "$log" || {
        printf '[pyruntime-wrapper] cannot append to %s\n' "$log" >&2
        exit 2
    }
fi

exec "$real" "$@"
