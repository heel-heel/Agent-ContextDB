# Claude Code Live Hook

ContextDB uses Claude Code project hooks to record user prompts and Bash tool
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
confirm the five ContextDB hooks. The hook input `session_id` becomes the
ContextDB session id, so each Claude Code session creates one trajectory.

## What is shared with Codex

The Claude adapter uses the same HTTP endpoints, SQLite event store, vector
skill index, learned skills, and `skill_recommendation` event schema as Codex.
After a failed Bash tool call, the `PostToolUseFailure` hook writes a Claude
Code `additionalContext` payload. Claude receives it before its next decision.

The hook only observes and records. It does not execute a recommended command
or claim that Claude accepted it. A later explicit decision/application bridge
can use the existing `/api/v1/hooks/skill_decision` and
`/api/v1/hooks/skill_application` endpoints.

## Verify

Use a failing Bash command in Claude Code, then run:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  -m contextdb.cli --root data hook-status claude-code <Claude-session-id>
```

The resulting trajectory contains `user_message`, `tool_call`, `tool_result`,
`skill_match`, and `skill_recommendation` events. When a historical Codex skill
matches, the Claude hook's `additionalContext` contains its selected action.

## Version-Control Fixture

Keep the Skill-reuse failure separate from the failures used to demonstrate
snapshots, repair branches, and rollback. In particular, do not reuse a Git
failure when the trajectory already contains a Git repair Skill: vector
retrieval can correctly identify the common tool, but that makes the version
control demonstration noisy.

For the two version-control failures, use these real PowerShell write attempts
from the project root. Their parent directories do not exist, so each command
fails without creating workspace files. `Set-Content` is state-changing and
therefore triggers the ContextDB pre-action snapshot policy.

```powershell
powershell -NoProfile -Command '[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US"); [System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US"); $ErrorActionPreference = "Stop"; Set-Content -LiteralPath ".\_contextdb_claude_version_first_missing_\first-note.txt" -Value "first note"'
```

```powershell
powershell -NoProfile -Command '[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US"); [System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::GetCultureInfo("en-US"); $ErrorActionPreference = "Stop"; Set-Content -LiteralPath ".\_contextdb_claude_version_second_missing_\second-note.txt" -Value "second note"'
```

Use the first failure to reject its repair-branch suggestion, then use the
second failure to accept its new suggestion and create the repair branch. The
most recent snapshot is the rollback target after the branch inspection.
