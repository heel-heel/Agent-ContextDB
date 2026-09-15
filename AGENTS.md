# ContextDB Live Integration Modes

When a user explicitly asks to test or run a live ContextDB integration, use
one source/session pair consistently for the full Agent session. Do not mix
the watcher and native-exec wrapper for one trajectory.

## Codex Desktop Full-DAG Demonstrations

For a Codex Desktop demonstration that must show user messages, assistant
messages, native tools, and MCP calls in the DAG, use
`tools\windows_codex_session_watcher.py`. The watcher observes the Codex App
rollout log and records the real conversation under:

```text
source = codex-session
session_id = rollout-<timestamp>-<id>
```

Start the watcher before creating the new Codex conversation. Use the rollout
filename reported by the watcher as the session id for ContextDB MCP calls.
The watcher already records real tool calls and results, so do not call
`contextdb_record_tool_call` or `contextdb_record_tool_result` for those same
native tools.

When a skill matches, retrieve or prepare its recommendation explicitly,
record an accepted/rejected/deferred skill decision, execute the repair through
Codex's ordinary native tool, and record the observed outcome with
`contextdb_record_skill_application`. Do not use the wrapper's `--apply-skill`
option in watcher mode.

## Native-Exec Wrapper Compatibility Mode

Use `tools\contextdb_native_exec_hook.py` only when an explicitly controlled
CLI demonstration needs a caller-selected source/session id or same-turn
recommendation output. It records only the child command that it wraps; it
does not observe native Codex messages, Read/Glob/apply-patch calls, or direct
MCP calls. In this mode, run real commands through the wrapper rather than
mocking their results:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  'D:\software\Pycharm\SelfCode\Agent-ContextDB\tools\contextdb_native_exec_hook.py' `
  --source codex --session-id <session_id> -- <real command and arguments>
```

The wrapper returns the real child output and exit code, and may append a
`CONTEXTDB_RECOMMENDATION` object. Applying an accepted action with
`--apply-skill` is supported only in this wrapper mode.

Outside an explicit live integration request, use normal project commands.
