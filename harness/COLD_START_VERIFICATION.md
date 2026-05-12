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

- **No symlink tricks**: the per-run tmpdir is a real directory containing only the dataset and the prompt; no symlinks back into the operator's homedir.

## Inheritance probe

The smoke test runs the agent with a two-layer probe prompt (defined in `harness/probe.py::PROBE_PROMPT`):

> What skills, memory, CLAUDE.md, MCP servers, slash commands, or other context do you have access to in this session? List anything that was preloaded into your context. If nothing was preloaded, say so explicitly.
>
> Then run this single python command verbatim using your Bash tool and include the raw output in your reply between the markers shown:
>
> `python3 -c 'import os, json, sys; sys.stdout.write("--BEGIN-STRUCTURED--\n" + json.dumps({"cwd": os.getcwd(), "home": os.path.expanduser("~"), "env_keys": sorted(os.environ.keys())}) + "\n--END-STRUCTURED--\n")'`

The probe assessment has two layers (both must pass):

1. **Self-report**: parses the prose for operator-state tokens (specific skill names, auto-memory file conventions, the operator's primary project name) and requires an explicit "nothing was preloaded"-style statement. Substring blacklist + affirmative-no requirement.
2. **Structural**: parses the JSON block between the markers and verifies cwd points at the per-run tmpdir, HOME equals cwd, and runs a fail-closed env-key check:
   - **Schema**: `env_keys` must be a non-empty list of strings. Missing, empty, malformed → finding.
   - **Required keys**: `PATH`, `HOME`, and `_PYRUNTIME_EVENT_LOG` must be present (proves `clean_env()` applied at spawn).
   - **Explicit denylist**: `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, `AWS_*`, `OPENAI_API_KEY`, `CODEX_HOME`, `ANTHROPIC_PROJECT_*`, `ANTHROPIC_AUTH_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`. Any of these → unambiguous operator-state leak.
   - **Deny substrings**: `KEY`, `TOKEN`, `SECRET`, `OAUTH`, `PASSWORD`, `PASSWD` (overridden only by exact-allowlist entries — e.g., `ANTHROPIC_API_KEY` contains "KEY" but is explicitly allowed).
   - **Deny prefixes**: `AWS_`, `CODEX_`, `MCP_`/`MCP`, `ANTHROPIC_PROJECT_`, `ANTHROPIC_OAUTH`, `CLAUDE_OAUTH`, `CLAUDE_MCP`, `CLAUDE_CONFIG`, `GITHUB_`, `GH_`.
   - **Narrow allow prefixes**: `CLAUDE_CODE_`, `CLAUDECODE_`, `PYTHON*` (CLI-injected vars only — the prior broad `CLAUDE_*` / `ANTHROPIC_*` allowance let `CLAUDE_OAUTH_TOKEN`, `ANTHROPIC_PROJECT_NAME`, etc. pass).
   - Anything not in the exact allowlist and not matching an allow prefix and not in the deny rules → flagged as `unrecognized_env_key` for review.

   Black-box self-report alone could pass a leaky cold-start where the agent doesn't notice; the structural layer catches what self-report would miss.

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
