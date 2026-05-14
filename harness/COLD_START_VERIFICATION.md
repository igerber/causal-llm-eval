# Cold-Start Verification

The harness's central scientific claim depends on the spawned Claude Code agent being truly cold-started: no inherited CLAUDE.md, no auto-memory, no plugins, no skills, no MCP servers, no keychain reads. If any of these leak in, the agent is not "fresh" and the eval is scientifically invalid.

This document specifies how cold-start is verified.

## The locked invocation

```
claude --bare \
       --setting-sources "" \
       --strict-mcp-config \
       --disable-slash-commands \
       --print \
       --output-format stream-json \
       --add-dir <tmpdir> \
       <prompt>
```

`--bare` is load-bearing - it suppresses CLAUDE.md auto-discovery, auto-memory, plugin sync, attribution, and keychain reads. The other flags strip the remaining surfaces.

## Subprocess hygiene (equally load-bearing)

CLI flags are necessary but not sufficient. The runner also pins:

- **`cwd=<run tmpdir>`** - the spawned Claude process starts in the per-run tmpdir, not the harness's working directory. This prevents the agent from discovering files in the harness repo itself (CLAUDE.md, .claude/, etc.).
- **`env=clean_env`** - the parent's environment is NOT inherited. The runner constructs a minimal `clean_env` containing ONLY the variables required for the agent to function:

  | Variable | Why allowed |
  |---|---|
  | `PATH` | needed for the agent to find `python`, `pip`, etc. (set to the per-run venv's bin first) |
  | `HOME` | set to the per-run tmpdir, NOT the operator's `$HOME`, so any file lookup against `~` lands in the sandbox |
  | `LANG`, `LC_*` | locale; harmless |
  | `ANTHROPIC_API_KEY` | the agent needs to call the Claude API to function |
  | `LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`, `LC_NUMERIC`, `LC_TIME`, `LC_COLLATE`, `LC_MONETARY` | locale specifics (POSIX-defined keys; explicit enumeration so unrelated `LC_*` vars cannot leak) |
  | `_PYRUNTIME_EVENT_LOG` | tells the in-process shim where to write. The underscore prefix + Python-runtime framing is a deliberate low-reactivity choice (less likely to be flagged as eval-related by an agent enumerating `os.environ`). Runner-controlled — any inherited operator value is ignored. |

  Anything else (especially `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, `ANTHROPIC_PROJECT_*`, `OPENAI_*`, `AWS_*`, MCP-related vars, GitHub auth tokens, `CODEX_*`) is dropped. The runner enforces this via an explicit allowlist in the spawn site, not a denylist.

- **No symlink tricks**: the per-run tmpdir is a real directory containing the dataset, the prompt, and a hidden runner-owned `.pyruntime/` subdirectory holding the in-process telemetry event log (`.pyruntime/events.jsonl`, kept inside tmpdir so its path does not leak the harness repo location via `_PYRUNTIME_EVENT_LOG`). No symlinks back into the operator's homedir.

## Inheritance probe

The smoke test runs the agent with a two-layer probe prompt (defined in `harness/probe.py::PROBE_PROMPT`):

> What skills, memory, CLAUDE.md, MCP servers, slash commands, or other context do you have access to in this session? List anything that was preloaded into your context. If nothing was preloaded, say so explicitly.
>
> Then run this single python command verbatim using your Bash tool and include the raw output in your reply between the markers shown:
>
> `python3 -c 'import os, json, sys; _P=("_PYRUNTIME_EVENT_LOG","PWD","CLAUDE_PROJECT_DIR"); sys.stdout.write("--BEGIN-STRUCTURED--\n" + json.dumps({"cwd": os.getcwd(), "home": os.path.expanduser("~"), "env_keys": sorted(os.environ.keys()), "env_path_values": {k: os.environ.get(k, "") for k in _P if k in os.environ}}) + "\n--END-STRUCTURED--\n")'`

The probe assessment has two layers (both must pass):

1. **Self-report**: parses the prose for operator-state tokens (specific skill names, auto-memory file conventions, the operator's primary project name) and requires an explicit "nothing was preloaded"-style statement. Substring blacklist + affirmative-no requirement.
2. **Structural**: parses the JSON block between the markers and verifies cwd points at the per-run tmpdir, HOME equals cwd, and runs a fail-closed env-key check:
   - **Schema**: `env_keys` must be a non-empty list of strings. Missing, empty, malformed → finding.
   - **Required keys**: `PATH`, `HOME`, and `_PYRUNTIME_EVENT_LOG` must be present (proves `clean_env()` applied at spawn).
   - **Explicit denylist**: `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, `AWS_*`, `OPENAI_API_KEY`, `CODEX_HOME`, `ANTHROPIC_PROJECT_*`, `ANTHROPIC_AUTH_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`, `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `PYTHONUSERBASE`. Any of these → unambiguous operator-state leak (Python interpreter vars are denylisted because they alter import resolution or run operator-controlled startup code).
   - **Deny substrings**: `KEY`, `TOKEN`, `SECRET`, `OAUTH`, `PASSWORD`, `PASSWD` (overridden only by exact-allowlist entries — e.g., `ANTHROPIC_API_KEY` contains "KEY" but is explicitly allowed).
   - **Deny prefixes**: `AWS_`, `CODEX_`, `MCP_`/`MCP`, `ANTHROPIC_PROJECT_`, `ANTHROPIC_OAUTH`, `CLAUDE_OAUTH`, `CLAUDE_MCP`, `CLAUDE_CONFIG`, `GITHUB_`, `GH_`.
   - **Narrow allow prefixes**: `CLAUDE_CODE_`, `CLAUDECODE_` only (CLI-injected vars). The prior broad `CLAUDE_*` / `ANTHROPIC_*` / `PYTHON*` prefixes were dropped because they let operator-state through.
   - **Path-value check**: the agent additionally reports `env_path_values` for `_PYRUNTIME_EVENT_LOG`, `PWD`, and `CLAUDE_PROJECT_DIR`. Each reported path must resolve under the per-run tmpdir; off-tmpdir paths trigger `env_path_outside_tmpdir` findings. Catches the case where the agent could infer the harness repo path by reading `os.environ['_PYRUNTIME_EVENT_LOG']`.
   - Anything not in the exact allowlist and not matching an allow prefix and not in the deny rules → flagged as `unrecognized_env_key` for review.

   Black-box self-report alone could pass a leaky cold-start where the agent doesn't notice; the structural layer catches what self-report would miss.

## Layer-2 reactivity surfaces (accepted)

The in-process shim (`harness/sitecustomize_template.py`) monkey-patches `diff_diff` API entries, `warnings.warn` (call-stack-inspecting, so warnings emitted with `stacklevel=N` are correctly attributed to the diff_diff frame), and `builtins.open` / `io.open` (for guide-file paths). These patches are detectable by an introspecting agent:

- `inspect.getsource(diff_diff.TwoWayFixedEffects.fit)` reveals a wrapper in `sitecustomize.py`; the `__wrapped__` attribute points at the original (the shim uses `functools.wraps`).
- `sys.modules['sitecustomize']` is present (Python convention; not eval-specific).
- `warnings.warn is not <original>` reveals a custom warning handler is installed (the override wraps `warn` rather than `showwarning` so it can inspect the call stack before `stacklevel` rewrites the displayed filename).
- `builtins.open is not <original>` and `io.open is not <original>` could be detected — both are overridden so pathlib/importlib.resources reads of guide files are recorded.
- The shim's `_ESTIMATOR_CLASS_NAMES` and `_DIAGNOSTIC_FUNCTION_NAMES` constants enumerate the methodology surface the eval tracks. An agent reading `sitecustomize.py` source can see exactly which estimators and diagnostics are monitored.
- The package name `"diff_diff"` appears as a string literal in the warning filter and patch targets. Visible to introspecting agents.

The shim contains NO string literals naming the eval framework, the agent under test, the comparator arm, or the rubric. `grep -r causal_llm_eval $(python -c 'import sys; print(sys.prefix)')` finds nothing in the shim. Helper names use a neutral `_pyruntime_*` prefix matching the env-var convention (`_PYRUNTIME_EVENT_LOG`).

Determined introspection still wins. Empirically, agents on causal-inference tasks rarely introspect their interpreter; the eval accepts this surface and documents it. Revisit if early case-study runs show agents probing the shim.

Event-size discipline: every layer-2 event keeps the `message` field capped to 500 chars to bound the per-run events.jsonl size. One run should not produce more than ~1 MB of layer-2 events under normal operation.

## Layer-1 attestation architecture

The merger validates Python invocations through three independent layers:

```
Layer 1   (text):    AST walk of the agent's Bash command
Layer 1.5 (exec):    per-arm-venv python wrapper
Layer 2   (in-proc): sitecustomize event log
```

Each layer attests independently. Any inconsistency between layers raises a `RunValidityError` subclass and invalidates the run.

Layer 1 is implemented in `harness/shell_parser.py` and walks the bashlex AST of every visible Bash command. The walker visits every `CommandNode` reachable from the parsed tree: top-level commands, members of `&&` / `||` / `;` / `|` pipelines, bodies of `if` / `while` / `for` control flow, contents of brace-groups and subshells, command-substitution targets (`$(...)`, `` `...` ``), and the recursively-re-parsed payloads of `eval` / `bash -c "..."` / `sh -c "..."` (bounded at 10 nesting levels).

For each `CommandNode`, the parser identifies the command-word by skipping leading `KEY=value` env-prefix assignments and recognized wrapper basenames (`time`, `nice`, `nohup`, `timeout`, `env`, `xargs`, `exec`, `stdbuf`, `ionice`, `chrt`, `command`, `sudo`, `doas`). If the command-word's basename matches `python` / `python3` / `python3.x`, the invocation is recorded with its argv. The parser is shape-agnostic: quoted command words (`"python"`), path-qualified wrappers (`/usr/bin/time python`), modifiers after shell separators (`cd /tmp && time python`), and sudo-prefixed launches (`sudo python`) all surface naturally from the same AST walk rather than from a hand-curated list of regex patterns.

This architecture is principled rather than enumeration-based - the failure mode is "AST surfaces an invocation we then have to attribute" rather than "the regex didn't recognize this shape". Two fail-closed contracts complement the AST walk:

- **Indeterminate command-words** (`${PY} script.py`, `$(which python) script.py`): static parsing cannot prove the resolved command word is not a Python launch, so the parser raises `ShellCommandIndeterminate`. The merger does not attest a run where the visible command may have invoked a Python interpreter we cannot enumerate.
- **bashlex parse failures** (Bash forms bashlex does not model, e.g. `case` pattern lists): raised as `ShellCommandParseError`. Same fail-closed semantics.

Both subclasses inherit from `RunValidityError`. Callers catching the parent unify fail-closed handling across layer-1 (parser-side) and layer-2/3 (`TelemetryMergeError`) failure modes. The parent name is neutral so methodology surfaces do not telegraph the specific check that fired.

## Layer-1.5 attestation architecture

Layer 1.5 is implemented as a POSIX shell wrapper (`harness/python_wrapper.sh`) installed in every per-arm venv by `harness.venv_pool._install_python_wrapper`. The original interpreter is preserved at `${venv}/bin/python-real`; each of `python`, `python3`, and `python3.X` becomes a copy of the wrapper script. Any `python` invocation in the venv routes through the wrapper before reaching the real interpreter.

For every invocation, the wrapper:

1. JSON-encodes the entire argv array via a single `awk` pass (handles `\b \f \n \r \t \" \\` control-char escapes; portable across mawk / gawk / busybox awk).
2. Appends one JSONL `exec_python` event (`pid`, `ppid`, `ts`, `executable`, `argv`) to `${_PYRUNTIME_EVENT_LOG}` via `>>` (O_APPEND atomic for writes under PIPE_BUF, typically 4 KiB).
3. `exec`s `${venv}/bin/python-real` with the original argv, transferring control to the real interpreter with no further wrapper overhead.

Fail-closed semantics match the layer-2 shim: `exit 2` on append failure (unwritable log path), and the event-write step is skipped entirely when `_PYRUNTIME_EVENT_LOG` is unset (in which case layer-2 sitecustomize will independently hard-exit, so the spawned interpreter still fails closed). NUL bytes in argv cannot occur in POSIX argv by the `execve` contract; the wrapper does not implement awk-level NUL detection (BSD awk behavior is unreliable).

The merger validates layer-1.5 events against the run's venv-root-anchored allowlist: `executable` MUST equal one of `${venv}/bin/python-real` / `python3-real` / `python3.X-real` (the rename set produced by `_install_python_wrapper`). Forged executables outside the venv root fail schema validation.

## Build-time python sentinel

After `build_arm_template` returns, `run_one` invokes a direct `subprocess.run([${venv}/bin/python, "-c", "pass"])` with `_PYRUNTIME_EVENT_LOG` set, BEFORE spawning the `claude --bare` agent subprocess. This sentinel:

- Produces exactly one `exec_python` event (from the wrapper) plus one `session_start` event (from sitecustomize) in the event log, proving wrapper + shim are wired before any agent code runs.
- Has `ppid == runner_pid` (the wrapper records the immediate parent, which is the runner's `subprocess.run`).

The merger's three-layer cross-check uses `ppid == runner_pid` as a binary partition between sentinel and agent events. Sentinel events satisfy the "≥1 exec_python required" demand even for shell-only agent runs that never invoke python. Agent-spawned exec_python events (`ppid != runner_pid`) match layer-1 AST-extracted python invocations by `argv[1:]` equality (argv[0] differs by construction: wrapper records basename `python`, sitecustomize records `${venv}/bin/python-real`).

## Per-arm venv build (PR #5)

`harness.venv_pool.build_arm_template(arm, library_version, template_dir)` builds a fresh per-run venv:

1. `venv.create(template_dir, with_pip=True)` materializes the venv with pip.
2. `${venv}/bin/python -m pip install <arm-pkg>==<library_version>` installs the arm library at the pinned PyPI version (`diff-diff==3.3.2` today; `statsmodels` is deferred to PR #7).
3. `_install_shim_into_venv` copies `harness/sitecustomize_template.py` into the venv's `site-packages` as `sitecustomize.py`. Python's site machinery auto-loads it on every interpreter start.
4. `_install_python_wrapper` installs the layer-1.5 wrapper (see above).

Phase 1 builds a fresh venv per run (~10-30s per build). Phase 2 (PR #6+) replaces this with `clone_for_run`: build the template once per pytest session / per cell, then clone-per-run cuts the cost to <1s.

## CI gate

The label gate (`ready-for-ci`) blocks merge until a maintainer applies the label. The `.github/workflows/tests.yml` workflow runs `pytest` (default excludes `slow` and `live`) on labeled PRs and on push to main, catching regressions in the cold-start invocation and supporting harness contracts.

`make smoke` (the live inheritance probe) is a developer command, not a CI hook: it spawns a real agent and costs ~$0.05 per invocation. Run it locally before push when the cold-start invocation changes.

The pre-merge-check skill (Section 2.1) runs an AST-based scan for required cold-start flags on any new subprocess spawn site, providing a fast first-pass before the smoke test. This catches multiline `subprocess.run([...])` invocations that single-line `grep` would miss.

## Verification matrix (run periodically)

| Surface | How to verify suppressed |
|---|---|
| User-global CLAUDE.md | Probe asks "do you see any CLAUDE.md content"; agent reports none |
| Project CLAUDE.md | Tmpdir doesn't contain one; probe confirms |
| Auto-memory | Probe asks "what memory entries do you have"; agent reports none |
| Plugins | Probe asks "what plugins are loaded"; agent reports none |
| Skills | Probe asks "what slash commands or skills are available"; agent reports only built-in `/help` etc. |
| MCP servers | `--strict-mcp-config` plus no MCP config file; probe confirms no MCP tools |
| Keychain / auth | `--bare` suppresses keychain reads; verify no Anthropic OAuth token in env |
| Operator `$HOME` | Spawned process's `$HOME` points at the per-run tmpdir, not the operator's homedir; probe confirms `~` resolves inside the sandbox |
| Inherited env vars | `env=clean_env` allowlist enforced at spawn time; probe asks `printenv` and asserts only allowed keys are present |
| Operator cwd | Spawn site uses `cwd=<run tmpdir>`, not the harness repo; probe confirms `pwd` is the tmpdir |

This file is updated when a new surface is identified that the probe should check.
