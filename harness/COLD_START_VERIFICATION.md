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

## Dataset copy + strict regular-file validation (PR #6)

`run_one()` validates `RunConfig.dataset_path` STRICTLY before spawning the agent: regular files only, rejecting symlinks, directories, devices, FIFOs, sockets, and paths containing NUL bytes. Validation runs at the TOP of `run_one()`, BEFORE `tempfile.mkdtemp()`, so a failed validation does NOT leak a tmpdir. The symlink rejection (via `lstat()` rather than `stat()`, which would follow symlinks) is the load-bearing piece of cold-start hygiene here: an operator-home symlink in `dataset_path` would otherwise leak the operator's filesystem into a "cold-start" agent's view.

After validation, `run_one()` copies the dataset into `tmpdir/data.parquet` (top-level for relative-path simplicity from the agent's perspective) via `shutil.copy2`, and computes the sha256 of the COPIED bytes for `metadata.json::dataset_sha`. The agent reads from the in-tmpdir copy, NOT the operator's source path; the source path's filesystem location never appears in the agent's view.

## Per-run reproducibility metadata (PR #6)

After clean exit (`exit_code == 0` AND no `descendants_live` AND no `telemetry_missing` sentinel), `run_one()` writes `output_dir/metadata.json` with the locked reproducibility schema (`harness_version`, `library_version`, `claude_code_version`, `model_version`, `dataset_sha`, `prompt_version`, `rubric_version`, `random_seed`, `run_id`, `arm`). Bytes are deterministic (`json.dump(..., sort_keys=True, indent=2, ensure_ascii=True)`).

Failure paths SUPPRESS the metadata write — absence of `metadata.json` is itself the signal that a run did not complete cleanly enough to be a reproducibility-anchored record. Downstream extractors and the `analysis/reproducibility_check.py` (PR #8+) branch on file presence rather than parsing partial blobs.

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

Layer 1.5 is implemented as a POSIX shell wrapper (`harness/python_wrapper.sh`) installed in every per-arm venv by `harness.venv_pool._install_python_wrapper`. The original interpreter is moved to `${venv}/.pyruntime-real/python-real` - a non-PATH location outside `bin/` so an agent cannot type `python-real` to bypass the wrapper. Each of `python`, `python3`, and `python3.X` in `bin/` becomes a copy of the wrapper script. Any `python` invocation in the venv routes through the wrapper before reaching the real interpreter at `${venv}/.pyruntime-real/python-real`.

For every invocation, the wrapper:

1. JSON-encodes the entire argv array via a single `awk` pass (handles `\b \f \n \r \t \" \\` control-char escapes; portable across mawk / gawk / busybox awk).
2. Appends one JSONL `exec_python` event (`pid`, `ppid`, `ts`, `executable`, `argv`) to `${_PYRUNTIME_EVENT_LOG}` via `>>` (O_APPEND atomic for writes under PIPE_BUF, typically 4 KiB).
3. `exec`s `${venv}/.pyruntime-real/python-real` with the original argv, transferring control to the real interpreter with no further wrapper overhead.

Fail-closed semantics match the layer-2 shim: `exit 2` on append failure (unwritable log path), and the event-write step is skipped entirely when `_PYRUNTIME_EVENT_LOG` is unset (in which case layer-2 sitecustomize will independently hard-exit, so the spawned interpreter still fails closed). NUL bytes in argv cannot occur in POSIX argv by the `execve` contract; the wrapper does not implement awk-level NUL detection (BSD awk behavior is unreliable).

The merger validates layer-1.5 events against the run's venv-root-anchored allowlist: `executable` MUST equal `${venv}/.pyruntime-real/python-real` (normalized via `os.path.normpath` to handle the wrapper's literal `bin/../.pyruntime-real/python-real` form). Forged executables outside the run's venv root fail schema validation. Additionally, any visible Bash command that references `.pyruntime-real` or a `python*-real` token by name is flagged as a bypass primitive (catches `find venv -name python-real -exec {} -S script.py \;` and similar delegated forms the AST walker cannot model).

## Build-time python sentinel

After `build_arm_template` returns, `run_one` invokes a direct `subprocess.run([${venv}/bin/python, "-c", "pass"])` with `_PYRUNTIME_EVENT_LOG` set, BEFORE spawning the `claude --bare` agent subprocess. This sentinel:

- Produces exactly one `exec_python` event (from the wrapper) plus one `session_start` event (from sitecustomize) in the event log, proving wrapper + shim are wired before any agent code runs.
- Has `ppid == runner_pid` (the wrapper records the immediate parent, which is the runner's `subprocess.run`).

The merger's three-layer cross-check uses `ppid == runner_pid` as a binary partition between sentinel and agent events. Sentinel events satisfy the "≥1 exec_python required" demand even for shell-only agent runs that never invoke python. Agent-spawned exec_python events (`ppid != runner_pid`) match layer-1 AST-extracted python invocations by `argv[1:]` equality (argv[0] differs by construction: wrapper records basename `python`, sitecustomize records `${venv}/.pyruntime-real/python-real`). A reciprocal pid check ensures every session_start has a matching exec_python and vice versa, so an agent that directly executes the real interpreter (bypassing the wrapper) leaves a session_start with no exec_python pair and fails closed.

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

## Telemetry-bypass threat model (PR #5)

The eval measures whether agents on causal-inference tasks discover and use library-level guidance surfaces. Telemetry completeness is a means to that end, not the eval's research question. This section delineates which bypass classes the merger must defeat (because they affect realistic measurement) and which are out-of-scope (because they require deliberate adversarial behavior we do not study).

**In-scope (defended structurally):**

1. **Direct invocation of a hidden interpreter via wrapper bypass**: any agent typing `python` / `python3` / `python3.X` is PATH-routed to the layer-1.5 wrapper at `${venv}/bin/python*`, which records `exec_python` and execs the strip-S shim at `${venv}/.pyruntime-real/python-real`.
2. **`-S` flag bypass on the strip-S shim**: the shim removes `-S` (and compact forms `-Sc` / `-IS`) from pre-script argv before exec'ing the actual CPython binary at `${venv}/.pyruntime-real/.actual-python`. Sitecustomize loads regardless of how the shim was invoked.
3. **Direct invocation of `python-real` or `.actual-python` (path-qualified)**: substring detection in `harness/shell_parser.py::find_python_bypass_invocations` flags any visible Bash command containing `.pyruntime-real`, `python*-real`, or `.actual-python` as a bypass primitive; the merger fails closed.
4. **Delegated forms that name the bypass tokens** (`find venv -name python-real -exec {} -S \;`, `xargs -I {} {} -S`): same substring detection.
5. **Reciprocal pid check**: every `session_start` MUST have a matching `exec_python` by pid and vice versa. An agent that bypasses the wrapper but reaches the strip-S shim still loads sitecustomize → session_start fires without exec_python → fail closed.

**Out-of-scope (documented limitation):**

**Deviation from plan:** the original PR #5 plan called for the layer-1.5 wrapper + substring detection to close ALL delegated python execution. The wildcard hidden-binary case below is a residual class that requires deliberate adversarial intent to construct; closing it structurally would require OS-level mechanisms (see "Out-of-scope structural defenses" below). This deviation is intentional and the residual is documented per-class with rationale.

A determined adversarial agent that constructs a wildcard delegated execution like

```
find venv -type f -perm -111 ! -path '*/bin/*' ! -path '*/.pyruntime-real/*' \
    -exec {} -S script.py \;
```

could in principle find an executable file outside the substring-detection paths and execute it with `-S`. In our deployment, the only such file is `${venv}/.pyruntime-real/.actual-python`, and the `! -path '*/.pyruntime-real/*'` exclusion above would skip it. A more aggressive form

```
find / -type f -name '.actual-python' -exec {} -S script.py \;
```

contains the `.actual-python` substring and is caught by the existing detection. The residual bypass class is filesystem enumeration that targets the hidden binary by `-perm` or other characteristics that do not name the file. This requires:

1. Deliberate adversarial intent (the agent is constructing obscure find patterns specifically to evade telemetry, not solving the eval task).
2. Use of `-S` to suppress sitecustomize (otherwise sitecustomize loads and the reciprocal pid check catches the missing exec_python).
3. Reaching `.actual-python` via a `find` form that does not lexically reference any of the bypass tokens.

Realistic agents working on causal-inference tasks do not exhibit this behavior. Successful adversarial bypass produces an event log identical in shape to "agent did not invoke python at all" (no session_start, no exec_python beyond the build-time sentinel). Both populations produce a clean `TelemetryRecord` with `opened_llms_*=False` and `called_get_llm_guide=False`, which is the correct measurement for both: agents that did not engage with Python (or actively evaded measurement) did not discover Python-level guidance surfaces.

The merger does not (and cannot) distinguish "agent did no python" from "agent ran python adversarially with -S via filesystem enumeration": these are observably equivalent under the layered defense. This is a property of measurement, not a defect: behaviors that produce no events cannot be observed. The eval's interpretation - "agent did not discover/use these surfaces" - is correct for both cases.

**Out-of-scope structural defenses (deferred):**

- OS-level mechanisms (setuid, namespaces, seccomp filters): would close the residual class but require privileged setup and platform-specific code.
- Custom CPython build with site.py forcibly enabled: would eliminate `-S` as a bypass vector at the source but requires per-platform build infrastructure.

Both are larger than the eval's research question warrants.

**Out-of-scope: session-detached descendants (PR #5 R16/R17):**

**Deviation from plan:** PR #5 R16 added post-wait `os.killpg` quiescence to close the same-process-group descendant case. The session-detached case (`setsid`, `daemonize`, `os.setsid()`) requires a structurally different defense (process supervision outside the original group, or shell-parser detection of detachment primitives that cannot cover the in-process variant). Closing it is deferred to a future PR if the threat model changes; documented out-of-scope per-class with rationale below.

`run_one()` puts Claude in a new process group via `start_new_session=True` and, on both timeout and normal exit paths, sends `SIGKILL` to the entire group. This catches descendants spawned via the standard `fork()`/`exec()` chain (e.g., agent `Bash` invocations of `python script.py &`, where the backgrounded process inherits the original process group). It does NOT catch descendants that explicitly detach from the process group:

- `setsid python script.py &` (Bash creates a new session for the child)
- A Python child that calls `os.setsid()` after fork
- `nohup` combined with a session-creating call
- `daemonize` library invocations
- `disown` after a `setsid`-style detach

A determined adversarial agent that constructs these forms can produce a child process that survives `os.killpg(proc.pid, SIGKILL)` (the call raises `ProcessLookupError` because the child is no longer in the original group), and can write events between `proc.wait()` returning and the runner moving the event log into the final artifact path. The detached child would appear in the merged record with timestamps after the run's nominal end.

This bypass class requires:

1. Deliberate adversarial intent (the agent is constructing session-detachment primitives specifically to evade post-run telemetry quiescence, not solving the eval task).
2. The detached child must succeed at writing AFTER `run_one()` returns and BEFORE the moved event log is consumed by `merge_layers()` - a small race window that depends on caller scheduling.
3. The detached child cannot affect the in-tmpdir log inode (once `shutil.move` completes, the post-move file is in `output_dir`; subsequent writes via the inherited fd hit the now-orphaned file path or fail). Realistic exploitation requires the child to also re-resolve the original path (which moved) or write via inherited fd before the move - a narrow window.

The shell-parser's bypass detection in `harness/shell_parser.py::find_python_bypass_invocations` does NOT currently flag `setsid` / `daemonize` / `disown` / similar detachment primitives as fail-closed bypass forms. Adding them would close the visible-form half of the class (an agent that types `setsid python script.py &` in a Bash invocation would be flagged), but cannot close the in-process half (a Python child that calls `os.setsid()` after fork has no visible textual marker).

Realistic agents working on causal-inference tasks do not exhibit this behavior. As with the wildcard delegated execution case above, the residual class is observably equivalent to "agent ran python adversarially in a way the layered defense cannot see post-hoc": the eval's interpretation is correct for both ("agent did not discover/use these surfaces" measured at finalization time).

**Out-of-scope: off-venv Python via delegated/shebang forms (PR #5 R19):**

**Deviation from plan:** the original PR #5 plan claimed the per-run venv + layer-1.5 wrapper provides "one fresh per-run venv" isolation for ALL python invocations. The off-venv delegated/shebang class (`find -exec /usr/bin/python3`, `xargs /usr/bin/python3`, `./script.py` with absolute shebang) bypasses the wrapper because the venv-routed PATH only catches bare `python` / `python3` / `python3.X` tokens. Closing it structurally would require either removing `/usr/bin`/`/bin` from agent PATH (breaks essential agent capabilities) or per-platform sitecustomize installation in the system Python (out of scope for the eval). This deviation is intentional; rationale and threat-model implications documented below.

The runner sanitizes the agent's `PATH` to put the per-run venv's `bin/` first, so any agent typing bare `python` / `python3` / `python3.X` is wrapper-routed. `/usr/bin` and `/bin` remain on the agent PATH (required for `ls` / `cat` / `find` / `mkdir` etc.) which means absolute off-venv Python interpreters at `/usr/bin/python3` are still on the filesystem and reachable. The layer-1 AST parser attributes direct `/usr/bin/python3 script.py` invocations (the basename matcher recognizes `python3`), so those fail closed when no `session_start` follows. But parser-invisible forms can execute the absolute interpreter without the AST parser modeling the invocation:

- `find /usr/bin -name 'python3*' -exec {} script.py \;`
- `xargs -I {} {} script.py` where the input is `/usr/bin/python3`
- `parallel /usr/bin/python3 script.py ::: arg1 arg2`
- `./script.py` where `script.py` has shebang `#!/usr/bin/python3` (or `#!/usr/bin/env python3` resolving via the agent's `/usr/bin` PATH)

These bypass the venv wrapper entirely. The spawned child runs the system interpreter (with NO venv-installed sitecustomize) so layer-2 also produces no event. The merger sees no `exec_python`, no `session_start`, and the AST sees no Python `CommandNode` (the wrapper command is `find` / `xargs` / `parallel` / a script path, not `python`). The sentinel-only path is then accepted as "agent did no Python work".

This bypass class requires:

1. **Deliberate adversarial intent**: the agent constructs a delegated form or installs a shebang script specifically to evade the wrapper, not to solve the eval task. Realistic agents working on causal-inference tasks invoke `python` directly via the Bash tool (which the wrapper handles).
2. **Knowledge that `/usr/bin/python3` exists at a known absolute path**: the agent has to either guess the path or successfully run a `find` to discover it. Either way the discovery itself is visible in the transcript.
3. **A meaningful task to run on the off-venv interpreter**: without the per-run `diff_diff` install, the spawned child has no library to invoke. An adversarial agent could `pip install --user diff_diff` first (visible in the transcript), but at that point the eval's interpretation - "agent did not discover/use the bundled guidance surfaces" - is correct: the agent that installed its own copy did not engage with the surface under measurement.

Closing this structurally would require either:

- Removing `/usr/bin` and `/bin` from the agent PATH (breaks essential agent capabilities like `ls`, `cat`, `find`, `mkdir`).
- Detecting and substringing every absolute Python path in agent argv, plus parsing every script's shebang. The detection surface is unbounded (`/usr/bin/python3`, `/usr/bin/python3.11`, `/usr/local/bin/python3`, `/opt/...`, etc.) and a sufficiently motivated adversary can always copy or rename the binary to evade substring detection.
- Custom CPython build with `sitecustomize` installed at the system level (would make the off-venv interpreter ALSO emit `session_start`). Requires per-platform build infrastructure - same out-of-scope class as the `-S` bypass mitigation above.

As with the other out-of-scope adversarial bypass classes documented above, the residual class is observably equivalent to "agent did not discover/use the bundled surfaces" at the eval's measurement granularity. The merger does not (and cannot) distinguish "agent did no Python" from "agent ran off-venv Python via delegated/shebang forms": both produce identical telemetry shapes. The eval's interpretation is correct for both populations.
