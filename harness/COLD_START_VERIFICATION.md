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

## Inheritance probe

The smoke test runs the agent with a probe prompt:

> What skills, memory, CLAUDE.md, MCP servers, slash commands, or other context do you have access to in this session? List anything that was preloaded into your context. If nothing was preloaded, say so explicitly.

A correctly cold-started agent reports nothing preloaded. The probe response is parsed; any positive report (a skill name, a CLAUDE.md content reference, an MCP server name) fails the smoke test.

## CI gate

`make smoke` runs the inheritance probe on every PR. The CI workflow (in `.github/workflows/`) blocks merge if `make smoke` fails. Adding a new code path that touches the cold-start invocation requires the smoke test to still pass.

The pre-merge-check skill (Section 2.1) runs grep-based pattern checks for missing `--bare` in any new subprocess spawn site, providing a fast first-pass before the smoke test.

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

This file is updated when a new surface is identified that the probe should check.
