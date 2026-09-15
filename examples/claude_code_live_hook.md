# Claude Code Live Hook

ContextDB uses Claude Code project hooks to record user prompts and tool
lifecycle events in real time. The hooks call the agent-neutral
`contextdb.agent_hook.v1` HTTP protocol already used by Codex.

## Start ContextDB

```powershell
cd D:\software\Pycharm\SelfCode\Agent-ContextDB
$env:CONTEXTDB_BASE_URL = 'http://127.0.0.1:8765'
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' scripts\serve_dashboard.py
```

Keep this terminal open. In a second terminal, export the same base URL and
start Claude Code from this project directory:

```powershell
cd D:\software\Pycharm\SelfCode\Agent-ContextDB
$env:CONTEXTDB_BASE_URL = 'http://127.0.0.1:8765'
claude
```

Claude Code loads the project-local `.claude/settings.json`. Run `/hooks` to
confirm the six ContextDB hooks. The hook input `session_id` becomes the
ContextDB session id, so each Claude Code session creates one trajectory.

`SessionStart` initializes a cursor, while `PreToolUse` and `Stop` wake
`tools/claude_code_hook.py` to read only the new JSONL records in Claude's
supplied `transcript_path`. Visible assistant text is recorded as
`assistant_message`; thinking, tool-use blocks, and tool results are excluded.
The hook persists its per-session cursor and UUID deduplication state under
`data/claude-transcript-sync`, so it is a hook-triggered incremental
transcript reader rather than a resident watcher.

## What is shared with Codex

The Claude adapter uses the same HTTP endpoints, SQLite event store, vector
skill index, learned skills, and `skill_recommendation` event schema as Codex.
After a failed Bash tool call, the `PostToolUseFailure` hook writes a Claude
Code `additionalContext` payload. Claude receives it before its next decision.

The hook only observes and records. It does not execute a recommended command
or claim that Claude accepted it. A later explicit decision/application bridge
can use the existing `/api/v1/hooks/skill_decision` and
`/api/v1/hooks/skill_application` endpoints.

The failed-tool hook requests only the recommendation, not a full prompt
digest. This keeps `PostToolUseFailure` bounded while still persisting the
`skill_match` and `skill_recommendation` audit events.

## Verify

Use a failing Bash command in Claude Code, then run:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  -m contextdb.cli --root data hook-status claude-code <Claude-session-id>
```

The resulting trajectory contains `user_message`, `assistant_message`,
`tool_call`, `tool_result`, `skill_match`, and `skill_recommendation` events.
When a historical Codex skill matches, the Claude hook's `additionalContext`
contains its selected action.

## Version-Control Fixture

Keep the Skill-reuse failure separate from the failures used to demonstrate
snapshots, repair branches, and rollback. In particular, do not reuse a Git
failure when the trajectory already contains a Git repair Skill: vector
retrieval can correctly identify the common tool, but that makes the version
control demonstration noisy.

For the two version-control failures, use these real Node.js write attempts
from the project root. Their parent directories do not exist, so each command
fails without creating workspace files. Writing a file through Node.js is
state-changing and therefore triggers the ContextDB pre-action snapshot policy.

```powershell
node -e "require('fs').writeFileSync('contextdb-version-first-parent-absent/note.txt', 'first note')"
```

```powershell
node -e "require('fs').writeFileSync('contextdb-version-second-parent-absent/note.txt', 'second note')"
```

Use the first failure to reject its repair-branch suggestion, then use the
second failure to accept its new suggestion and create the repair branch. The
most recent snapshot is the rollback target after the branch inspection.
