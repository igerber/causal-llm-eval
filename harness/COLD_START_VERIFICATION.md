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
  | `CAUSAL_LLM_EVAL_EVENT_LOG` | tells the in-process shim where to write |

  Anything else (especially `XDG_CONFIG_HOME`, `CLAUDE_CONFIG_DIR`, `ANTHROPIC_PROJECT_*`, `OPENAI_*`, `AWS_*`, MCP-related vars, GitHub auth tokens, `CODEX_*`) is dropped. The runner enforces this via an explicit allowlist in the spawn site, not a denylist.

- **No symlink tricks**: the per-run tmpdir is a real directory containing only the dataset and the prompt; no symlinks back into the operator's homedir.

## Inheritance probe

The smoke test runs the agent with a probe prompt:

> What skills, memory, CLAUDE.md, MCP servers, slash commands, or other context do you have access to in this session? List anything that was preloaded into your context. If nothing was preloaded, say so explicitly.

A correctly cold-started agent reports nothing preloaded. The probe response is parsed; any positive report (a skill name, a CLAUDE.md content reference, an MCP server name) fails the smoke test.

## CI gate

**Phase 0 status**: the label gate (`ready-for-ci`) is in place, but the actual test workflow that runs `make smoke` is not yet implemented. It lands in a follow-up PR alongside the runner implementation. Once present, the workflow will block merge if `make smoke` fails, ensuring every PR that touches the cold-start invocation re-verifies the inheritance probe.

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
