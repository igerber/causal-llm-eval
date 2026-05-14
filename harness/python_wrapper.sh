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
    # Fail closed on newline/CR in any argv element BEFORE encoding. The
    # line-oriented awk encoder cannot preserve embedded newlines (one
    # element with a newline would split into two records, so the JSON
    # array no longer matches the real argv). POSIX argv may legally
    # contain newlines but realistic agent invocations never do; the
    # wrapper fails closed rather than emit corrupted attestation.
    nl=$(printf '\n')
    cr=$(printf '\r')
    for a in "$@"; do
        case "$a" in
            *"$nl"* | *"$cr"*)
                printf '[pyruntime-wrapper] argv contains newline/CR; cannot attest\n' >&2
                exit 2
                ;;
        esac
    done

    # JSON-encode argv via a single awk invocation. The recorded argv is
    # ``[basename($0), $@]`` so the merger can match on ``argv[1:]``
    # against the layer-1 AST extraction (which produces
    # ``[interpreter, *args]``) and against layer-2's ``sys.orig_argv[1:]``
    # (which the real python records post-exec). All three layers'
    # ``argv[1:]`` = the script + flags.
    arg0_basename=$(basename "$0")
    args_json=$(printf '%s\n' "$arg0_basename" "$@" | awk '
        BEGIN { first = 1; printf "[" }
        {
            gsub(/\\/, "\\\\")
            gsub(/"/, "\\\"")
            gsub(/\010/, "\\b")
            gsub(/\014/, "\\f")
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
