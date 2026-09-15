# Codex Desktop Watcher Live Demo

Use this flow when the Codex DAG must contain the same classes of evidence as
the Claude Code DAG: user messages, assistant messages, native tool calls,
native tool results, and ContextDB MCP calls.

This is not a wrapper demo. Do not use `contextdb_native_exec_hook.py`,
`--apply-skill`, `contextdb_record_tool_call`, or
`contextdb_record_tool_result` for native Codex tools in this flow.

## Start the services

In one PowerShell window, start the dashboard from the project root:

```powershell
$env:CONTEXTDB_BASE_URL = 'http://127.0.0.1:8765'
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' scripts\serve_dashboard.py
```

In a second PowerShell window, start the watcher before opening a new Codex
conversation:

```powershell
cd D:\software\Pycharm\SelfCode\Agent-ContextDB
$env:CONTEXTDB_BASE_URL = 'http://127.0.0.1:8765'
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  tools\windows_codex_session_watcher.py --base-url $env:CONTEXTDB_BASE_URL `
  --state-file .\data\watcher-codex-1-watcher-state.json
```

After the first Codex message or tool call, the watcher reports a filename such
as `rollout-2026-09-14T10-30-00-example.jsonl`. The filename without `.jsonl`
is the session id for that conversation. The watcher also mirrors that value
to `data\codex-watcher-current-session.json` after it forwards an event. Do
not discover it by scanning `data\objects\hook_sessions`: that directory is
populated asynchronously and may select another Codex task.

To let Codex retain the current rollout id automatically, send this as the
first separate message of the new conversation. It waits briefly for the
watcher to forward this first message, then reads the watcher-owned workspace
pointer. This is necessary because Codex's native terminal sandbox cannot read
the protected raw session-log directory directly:

```text
Before any failure occurs, determine the ContextDB session id for this current
Codex conversation. Use exactly one ordinary native terminal tool call from
the project root to run:

powershell -NoProfile -Command '$pointer = ".\data\codex-watcher-current-session.json"; $deadline = (Get-Date).AddSeconds(10); do { if (Test-Path -LiteralPath $pointer) { $session = Get-Content -Raw -LiteralPath $pointer | ConvertFrom-Json; $field = "session" + [char]95 + "id"; $id = $session.PSObject.Properties[$field].Value; if ($session.source -eq "codex-session" -and $id) { Write-Output $id; exit 0 } }; Start-Sleep -Milliseconds 250 } while ((Get-Date) -lt $deadline); [Console]::Error.WriteLine("ContextDB watcher session was not available within 10 seconds."); exit 1'

Do not use contextdb_native_exec_hook.py. Retain the returned rollout id and
use it as the session id for every later ContextDB MCP call in this
conversation. Report the session id in English.
```

Every later ContextDB MCP call uses these values:

```text
source: codex-session
session_id: rollout-2026-09-14T10-30-00-example
```

Do not substitute a wrapper-style id such as `codex-path-repair-001`, and do
not reuse a retained id after creating a new Codex conversation.

## Controlled skill reuse sequence

Send every quoted block below as a separate user message to Codex. Keep all
responses and tool output in English.

1. Establish the session and materialize pre-turn context using the retained
   rollout id.

   ```text
   Use ContextDB MCP tool contextdb_prepare_context with source codex-session
   and the session id retained earlier. Use English-only text and responses. Report the
   returned trajectory id, then continue with the next requested action.
   ```

2. Produce a real failure with Codex's ordinary native terminal tool. This is
   deliberately not wrapped.

   ```text
   From the project root, run this real command with your ordinary native tool
   and report the complete raw output in English:

   powershell -NoProfile -Command '$env:LC_ALL="C"; $env:LANG="C"; git show HEAD:assets/settings.json'
   ```

3. Make the real successful repair through the same ordinary native tool.

   ```text
   From the project root, run this real command with your ordinary native tool
   and report the complete raw output in English:

   powershell -NoProfile -Command '$env:LC_ALL="C"; $env:LANG="C"; git show HEAD:examples/live-path-demo/assets/settings.json'
   ```

4. Repeat the exact failing command from step 2 through the ordinary native
   tool. The watcher records this tool result and ContextDB performs retrieval.

5. Fetch and inspect a recommendation without duplicating the native tool
   events:

   ```text
   Use ContextDB MCP tool contextdb_get_recommendation with source codex-session
   and the session id retained earlier. If a matched action is present, report its
   skill match event id, skill id, action id, and proposed action in English.
   ```

6. Record an explicit acceptance before running the repair:

   ```text
   Use ContextDB MCP tool contextdb_record_skill_decision with source
   codex-session and the session id retained earlier. If the reported recommendation
   has a matched action, record an accepted decision using its skill match event
   id, skill id, and action id. Explain that the proposed Git file path will be
   used for the repair.
   ```

7. Run the proposed repair with Codex's ordinary native tool. Then record the
   observed outcome as the skill application:

   ```text
   Execute the accepted Git action with your ordinary native tool and report
   its complete raw output in English. Then use ContextDB MCP tool
   contextdb_record_skill_application with source codex-session and the session
   id retained earlier. Record the actual tool name, command, observed status, raw
   output preview, exit code, and the recommendation's skill match event id,
   skill id, and action id. ContextDB automatically reuses the most recent
   watcher-recorded native call with the same tool name and command; do not
   create a generic tool record.
   ```

At this point, Application Trace should show an explicit matched,
recommendation-delivered, accepted, and applied sequence. The real terminal
activity and the MCP calls both appear in the watcher-collected DAG.

## Version-control sequence

After the skill sequence, keep using the same `codex-session` source and
watcher-derived session id.

1. Use a native Codex terminal tool to run a first real failing state-changing
   command. The watcher records it and ContextDB creates a snapshot and
   repair-branch recommendation under the existing hook policy.

   ```text
   From the project root, run this real command with your ordinary native tool
   and report the complete raw output in English:

   powershell -NoProfile -Command '$env:LC_ALL="C"; $env:LANG="C"; git checkout --detach contextdb-demo-first-001'
   ```

2. Inspect the current repair-branch suggestion, then reject it in a separate
   message using `contextdb_get_version_status` followed by
   `contextdb_record_version_decision` with `action=create_repair_branch` and
   `decision=rejected`. Use the reason: `Keep this first invalid Git checkout
   on main for diagnosis.`

3. Run a normal native inspection on `main`, then use a second distinct failed
   checkout:

   ```text
   From the project root, run this real command with your ordinary native tool
   and report the complete raw output in English:

   powershell -NoProfile -Command '$env:LC_ALL="C"; $env:LANG="C"; git checkout --detach contextdb-demo-second-001'
   ```

4. In separate messages, inspect the newest suggestion with
   `contextdb_get_version_status`, then accept it with
   `contextdb_create_repair_branch`. Inspect using native Codex tools on the
   resulting repair branch.

5. In one message, use `contextdb_get_version_status` to report the newest
   snapshot. In a second message, call `contextdb_rollback_context` for that
   snapshot. State that this ContextDB rollback creates a new branch and does
   not restore workspace files or Git state.

## Verification

Open the watcher-derived trajectory in the Dashboard. Global Overview and the
DAG should include actual Codex `user_message`, `assistant_message`, native
tool, and MCP activity. Version Control should show the first rejected repair
suggestion, the later accepted repair branch, and the rollback.
