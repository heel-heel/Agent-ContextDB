# Agent ContextDB Demo

ContextDB is an independent Agent Context Database prototype. It models agent state as durable, queryable, versioned data instead of framework-private transcript files.

The ICDE demo angle is **Git + Database for agent context**: messages, tool calls, tool results, memory updates, summaries, artifacts, and skills can be represented as events in a branchable trajectory DAG. Views such as memory, failures, summaries, prompt windows, and RL rows are derived database views over the same trajectory, not separate ad-hoc files.

## Relation to OpenViking

OpenViking focuses on a context database and virtual filesystem for organizing and retrieving memory, resources, and skills. ContextDB is positioned one layer below/alongside that idea: it stores the full agent execution trajectory as versioned database state, then exposes retrieval-ready views that systems such as OpenViking-style context loaders could consume.

## What is implemented

- Core objects: Agent, Source, Trajectory, ContextEvent, Branch, Snapshot, View, Artifact.
- Trajectory DAG: branch-local events now include reachable ancestor events through `parent_event_ids`.
- Versioning: snapshot, branch, rollback, branch-aware log, and semantic diff.
- Query/View/Index: filters by trajectory, branch, type, actor, status, timestamp, and text containment.
- Materialized views: memory, summary, failures, current_prompt, rl_dataset, failure_patterns, success_patterns, repair_strategies, semantic_repair_judgments, and learned_skills.
- Streamed context loading: `current_prompt` returns summary + recent events + memory plus estimated token savings.
- Graph API: `/api/v1/graph` returns nodes, parent edges, branches, snapshots, and materialized views.
- Diff API: `/api/v1/diff` compares branches by event set, failure count, and summary view changes.
- HTTP server and a dependency-free ICDE dashboard at `/demo`.

## Run in VMware Ubuntu

```bash
source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
cd /home/benjamin/windows/contextdb
contextdb demo  # replays examples/demo_trace.jsonl
contextdb serve --host 0.0.0.0 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Open the dashboard:

```text
http://127.0.0.1:8765/demo
```

If you ran `contextdb demo`, paste the printed `trajectory_id` into the dashboard, or open:

```text
http://127.0.0.1:8765/demo\?trajectory_id\=\<trajectory_id\>
```


## Real Agent integration

ContextDB supports two integration paths for real agents.

### Offline trace import

Use `import-trace` for real Codex/Claude/Cursor/LangGraph style logs after converting them to a framework-neutral JSONL format:

```bash
contextdb import-trace examples/agent_trace.jsonl --source generic-jsonl --agent-id codex-like
```

Each JSONL line can be compact:

```json
{"type":"tool_call","tool":"shell","command":"pytest tests/"}
{"type":"tool_result","status":"failed","exit_code":1,"output":"AssertionError..."}
```

or ContextDB-native:

```json
{"event_type":"assistant_message","actor":"agent","payload":{"text":"I will inspect the error."}}
```

The adapter normalizes these records into `ContextEvent` objects and creates a trajectory with materialized `summary`, `failures`, and `current_prompt` views.

### Experience mining views

`failure_patterns` extracts failed/error/timeout tool results with the preceding assistant action, preceding tool call, error signature, inferred likely cause, and nearby repair events.

`success_patterns` extracts successful tool results from every branch with the command and preceding strategy.

`repair_strategies` links failures to successes using deterministic structural rules such as same-branch-first-success-after-failure, branch-from-failure-first-success, branch-from-failure-ancestor-first-success, rollback-then-repair-first-success, and same-tool-command-variant. The output includes explicit evidence instead of an opaque similarity score; command variants are supporting evidence and do not create repair links by themselves.

The experience-mining views use the stable `experience_mining.v1` schema for the UI layer. `failure_patterns` includes `normalized_signature`, `source_event_ids`, `evidence_event_ids`, and core `highlight_event_ids`. `success_patterns` includes `is_first_success_on_branch` and highlight metadata. `repair_strategies` keeps the old `repairs` field for compatibility and also exposes `repair_candidates`, `repair_status`, `candidate_count`, `excluded_successes`, `why_linked`, `highlight_event_ids`, `highlight_branch_ids`, and a placeholder `semantic_judge` object for optional future LLM judging.

The LLM selectors in Context Digest and Query Explorer use the Bailian OpenAI-compatible API. Configure `DASHSCOPE_API_KEY` and `CONTEXTDB_LLM_BASE_URL`; the selectable profiles are defined in `config/llm_profiles.json` and default to `Deepseek-v4.1-flash`. Background tasks without a selector continue to use their separate `CONTEXTDB_BACKGROUND_LLM_*` configuration. For offline UI testing, set `CONTEXTDB_LLM_PROVIDER=mock`.

`learned_skills` materializes reusable agent skills from repair candidates that pass the semantic judgment layer. Each skill contains a trigger, recommended actions, avoid/validation-only actions, confidence metadata, and evidence references back to the trajectory.

### Live agent client

Agents can also write events while they run through the HTTP API using the standard-library client:

```python
from contextdb.client import ContextDBClient

ctx = ContextDBClient("http://127.0.0.1:8765")
traj = ctx.create_trajectory("Live task", agent_id="my-agent", source_id="runtime")
tid = traj["trajectory_id"]
ctx.log_user(tid, "Fix failing tests")
ctx.log_tool_call(tid, "shell", "pytest -q")
ctx.log_tool_result(tid, "failed", "AssertionError...")
```

For a quick end-to-end check, start the server and run:

```bash
contextdb serve --host 127.0.0.1 --port 8765
contextdb client-demo --base-url http://127.0.0.1:8765
```


### Real-time Codex Hook

`codex exec --json` emits JSONL while a Codex run is in progress. ContextDB consumes that stream through a stable Hook protocol, so the integration is live rather than an after-the-fact import. The Hook protocol is agent-neutral: a future adapter only needs to submit a `contextdb.agent_hook.v1` envelope with `source`, `session_id`, and an event object.

Start ContextDB in one terminal:

```bash
source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
cd /home/benjamin/windows/contextdb
contextdb serve --host 127.0.0.1 --port 8765
```

In a second terminal, run Codex through its public JSONL stream. Keep only Codex stdout in the pipe; ContextDB prints Hook diagnostics and skill suggestions to stderr.

```bash
source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
cd /path/to/your/coding/repository
codex exec --json "Run the test suite and repair the failing issue." | \
  contextdb hook-stream --base-url http://127.0.0.1:8765 \
    --source codex --title "Codex live repair"
```

The initial `thread.started` record determines the Hook `session_id`; it is also printed in Codex JSONL. Each Hook session creates exactly one ContextDB trajectory. Codex command start/completion records become `tool_call`/`tool_result`, and a non-zero exit code immediately creates a `skill_match` event through the persistent vector skill index. The Hook response chooses the highest-confidence recommended action (ties retain stored action order) and returns it as an `agent_context` suggestion. It never auto-executes that command: Codex or another Agent must choose to apply it.

Verify a session using its Codex thread id:

```bash
contextdb hook-status codex <thread_id>
contextdb hook-context codex <thread_id> --format prompt
contextdb query-view <trajectory_id> skill_application_trace
```

For a follow-up Codex turn, feed the `hook-context --format prompt` result into `codex exec resume <thread_id> ...` and pipe that resumed JSONL stream through `hook-stream` again. This is the explicit application step: the first run records and retrieves, while the resumed Agent turn decides whether to execute the recommended action.

Successful real-time integration has three observable facts: `hook-status` shows the persistent source/session/trajectory mapping; the trajectory graph contains events with `metadata.integration = "agent-hook.v1"`; and, when a tool fails, it contains a `skill_match` event whose `metadata.operation` is `online_skill_retrieval`. A non-empty `agent_context.selected_action` in the Hook diagnostic proves that a stored skill was actually matched and ranked. An eventual Codex tool call is only an *applied* skill when its event refs explicitly carry the returned `skill_match_event_id` and `skill_id`.

For an adapter other than Codex, POST the following envelope to `/api/v1/hooks/events` (or feed one JSON object per line to `contextdb hook-stream --source generic --session-id <id>`):

```json
{
  "protocol_version": "contextdb.agent_hook.v1",
  "source": "future-agent",
  "session_id": "session_123",
  "agent_id": "future-agent",
  "event": {
    "event_type": "tool_result",
    "actor": "tool",
    "payload": {
      "tool_name": "shell",
      "command": "pytest -q",
      "status": "failed",
      "preview": "AssertionError"
    }
  }
}
```

### Bidirectional MCP Skill Injection

The Windows session watcher is intentionally observation-only: it can record a
Codex App tool failure after it happens, but it cannot alter the context of an
already-running model turn. For actual context injection, ContextDB now exposes
a dependency-free **stdio MCP server**:

```powershell
cd D:\software\Pycharm\SelfCode\Agent-ContextDB
D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe -m contextdb.cli --root data mcp-serve
```

For the Windows project, configure an MCP client to launch the same command, or
use `scripts\run_mcp_server.py` as the Python script target. Keep stdout clean:
it is reserved for MCP JSON-RPC. The Agent-side policy template is in
`examples\agent_contextdb_mcp_instructions.md`.

The server offers an Agent-neutral protocol suitable for Codex, Claude Code,
or another MCP-capable harness:

- `contextdb_prepare_context`: materializes compact turn context and records a
  `skill_recommendation` delivery event.
- `contextdb_record_tool_result`: records a real tool result; a failure invokes
  vector skill retrieval and returns the recommendation in the same tool result.
- `contextdb_get_recommendation`: redelivers a pending recommendation after a
  resumed turn.
- `contextdb_record_skill_decision`: records `accepted`, `rejected`, or
  `deferred` rather than assuming a match was used.
- `contextdb_record_skill_application`: records the outcome of an action the
  Agent actually executed through its normal tool and approval mechanism. It
  never executes the command itself.
- `contextdb_record_tool_call`: records the tool start. Potentially
  state-changing calls receive an automatic **ContextDB snapshot**
  before the call runs.
- `contextdb_create_snapshot`, `contextdb_create_repair_branch`, and
  `contextdb_rollback_context`: let an Agent explicitly checkpoint, branch, or
  recover the trajectory state. They never modify the Agent workspace or run
  Git commands.
- `contextdb_record_version_decision` and `contextdb_get_version_status`:
  preserve the Agent's choice to continue, repair on a new branch, or roll
  back, together with the active branch and pending repair advice.

The resulting application trace is explicit:

```text
tool failure -> skill_match -> skill_recommendation delivered
             -> skill_decision -> agent tool_call -> tool_result
```

This distinction is intentional: a matched skill is evidence retrieval; a
delivered skill is context injection; only an Agent-recorded normal tool call
is an application. The ContextDB dashboard's **Application Trace** view shows
these states separately.

### Version Control for Live Agents

The live Hook follows a conservative hybrid policy:

```text
potentially state-changing tool call -> automatic snapshot
failure -> skill retrieval + repair-branch suggestion
Agent decision -> continue | create repair branch | rollback
Agent workspace action -> only after the Agent explicitly chooses it
```

The snapshot contains ContextDB trajectory state, not a copy of workspace
files. A rollback creates a new DAG branch at the snapshot event; it
does not call `git reset`, check out files, or change the local filesystem.
Open **Version Control** in the Dashboard to inspect the branch registry,
snapshots, decision timeline, and the workspace-restore boundary. The page also
offers manual Snapshot, Create Repair Branch, and Rollback controls for
demonstration purposes.

### Codex Desktop: Full-DAG Watcher Mode

For a Codex Desktop demonstration that should look like the Claude Code DAG,
prefer the Windows session watcher. It records the actual `user_message`,
`assistant_message`, native tool-call/result, and MCP-call records written by
the Codex App, under one `codex-session` trajectory. Start it before creating
a new Codex conversation:

```powershell
cd D:\software\Pycharm\SelfCode\Agent-ContextDB
$env:CONTEXTDB_BASE_URL = 'http://127.0.0.1:8765'
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  tools\windows_codex_session_watcher.py --base-url $env:CONTEXTDB_BASE_URL
```

The watcher prints the rollout filename after it forwards events. Use that
filename, without `.jsonl`, as the MCP session id and use `codex-session` as
the source. It begins at the end of existing rollout files, so create the
conversation after the watcher is ready. Do not run the native-exec wrapper for
the same conversation.

The watcher already records native Codex tools. For an accepted skill, the
Agent must make the audit trail explicit:

```text
real tool failure (recorded by watcher)
-> contextdb_prepare_context or contextdb_get_recommendation
-> contextdb_record_skill_decision
-> real repair through a native Codex tool (recorded by watcher)
-> contextdb_record_skill_application with the observed outcome
```

Do not call `contextdb_record_tool_call` or `contextdb_record_tool_result` for
native calls that the watcher has already observed; doing so duplicates the
DAG. The watcher is observation-only: it delivers no automatic in-turn
recommendation and never executes a repair.

See `examples\codex_watcher_live_demo.md` for the controlled test flow.

### Native Exec Hook: Controlled CLI Compatibility Mode

For a controlled CLI-only test with a caller-selected source/session id,
ContextDB also provides `tools\contextdb_native_exec_hook.py`. Codex invokes
this wrapper through its normal native `exec`, and the wrapper executes the
real child command.

```text
Codex native exec -> ContextDB wrapper -> real child command
  -> tool_result hook -> skill retrieval -> same exec output contains recommendation
```

Start the HTTP server as usual, then have Codex use one stable session id for
the live task. A native PowerShell execution looks like this:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  'D:\software\Pycharm\SelfCode\Agent-ContextDB\tools\contextdb_native_exec_hook.py' `
  --source codex --session-id contextdb-live-001 -- `
  powershell -NoProfile -Command "python --version"
```

If the real child command fails and a skill matches, its regular command output
is followed by one machine-readable line:

```text
CONTEXTDB_RECOMMENDATION { ... }
```

That line is visible to Codex as part of the same native tool result, so it is
available for the next repair decision without replaying a trace. To record an
accepted, real skill-guided action, use the same wrapper with `--apply-skill`:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  'D:\software\Pycharm\SelfCode\Agent-ContextDB\tools\contextdb_native_exec_hook.py' `
  --source codex --session-id contextdb-live-001 --apply-skill -- `
  powershell -NoProfile -Command "CC=clang cargo build --release"
```

Do not run the Windows session watcher for the same live task: the wrapper is
the source of truth for tool calls and results in this compatibility mode,
while MCP remains available for pre-turn context retrieval and other Agent
frameworks. The wrapper cannot capture the full native Codex conversation; use
watcher mode above when that is required.

### Unified Agent Runtime Contract

Every real-time adapter uses the same `contextdb.agent_hook.v1` envelope and
the pair `(source, session_id)`. An Agent can integrate through either the HTTP
hook response, the native-exec wrapper, or MCP, while ContextDB stores one
trajectory and one application trace for that pair. Future Agent-specific
adapters only need to map their tool start/result callbacks to this contract;
they do not need a separate memory or skill storage implementation.

### Claude Code Live Hook

The project includes `.claude/settings.json` and
`tools\claude_code_hook.py` for a project-local Claude Code integration. Its
`PreToolUse`, `PostToolUse`, and `PostToolUseFailure` hooks map real Claude
Code tool callbacks, including Bash and Read, to the same Agent Hook protocol
used by Codex. On a failed tool call, the adapter retrieves a ContextDB recommendation and returns it to Claude
as Claude Code `additionalContext` before the next model decision. See
`examples\claude_code_live_hook.md` for setup and verification.

## Useful CLI commands

```bash
contextdb demo
contextdb demo --trace examples/demo_trace.jsonl
contextdb graph <trajectory_id>
contextdb query-view <trajectory_id> current_prompt --branch clang-attempt
contextdb query-view <trajectory_id> failure_patterns --branch main
contextdb query-view <trajectory_id> success_patterns
contextdb query-view <trajectory_id> repair_strategies
contextdb query-view <trajectory_id> semantic_repair_judgments
contextdb query-view <trajectory_id> learned_skills
contextdb diff <trajectory_id> docker-attempt clang-attempt
contextdb snapshot <trajectory_id> --branch main --message "before risky action"
```

## HTTP API additions

```bash
curl "http://127.0.0.1:8765/api/v1/graph?trajectory_id=$TRAJ"
curl -s -X POST http://127.0.0.1:8765/api/v1/diff \
  -d "{\"trajectory_id\":\"$TRAJ\",\"left_branch\":\"docker-attempt\",\"right_branch\":\"clang-attempt\"}"
```

## ICDE demo script

1. Run `contextdb demo` to replay `examples/demo_trace.jsonl` into a realistic agent setup trajectory.
2. Show `/demo` graph: `main` fails on native build, `rollback-clean` time-travels back to the clean snapshot, and `docker-attempt` / `clang-attempt` branch from that rollback point.
3. Switch views: `current_prompt` shows context loading and token savings, `failure_patterns` extracts failed tool calls with preceding agent actions, `success_patterns` extracts successful strategies across branches, and `repair_strategies` links failures to later successful repairs with deterministic structural evidence. The bundled demo includes same-branch and repair-branch cases where only the first successful tool result after a failure is linked as the direct repair; later smoke-test successes are intentionally not linked.
4. Run `contextdb diff` to compare successful repair branches, then show `rl_dataset` as training-data export for SFT/RL/distillation.
5. Explain the key claim: context is no longer only prompt text; it is a first-class, persistent, queryable, versioned database object.

## Windows Codex App Live Hook

ContextDB can record real Windows Codex App sessions without requiring the Codex CLI.
The watcher tails the App's local `~/.codex/sessions/**/rollout-*.jsonl` files and
forwards only appended records. Each rollout file becomes one `codex-session`
trajectory; failed tool results use the normal ContextDB skill retrieval path.

1. On Ubuntu, run ContextDB in `contextdb_env`:

   ```bash
   source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
   cd /home/benjamin/windows/contextdb
   contextdb serve --host 127.0.0.1 --port 8765
   ```

2. In a separate Windows PowerShell window, create and keep an SSH tunnel open:

   ```powershell
   ssh -N -L 8765:127.0.0.1:8765 benjamin@192.168.253.128
   ```

3. Copy and start the watcher on Windows:

   ```powershell
   scp benjamin@192.168.253.128:/home/benjamin/windows/contextdb/tools/windows_codex_session_watcher.py $env:USERPROFILE\bin\
   python $env:USERPROFILE\bin\windows_codex_session_watcher.py --base-url http://127.0.0.1:8765
   ```

4. Use the Codex App normally. Watcher output `CONTEXTDB_SKILL_HOOK` after a failed
   tool result proves a live skill retrieval. In Ubuntu, inspect the corresponding
   session and graph with:

   ```bash
   contextdb hook-status codex-session rollout-YYYY-MM-DDTHH-MM-SS-<id>
   ```

On its first normal run the watcher starts at the end of every log that already exists,
including files with stale entries in a prior state file, so it does not import old
conversations. Rollout files created after that initial scan are read from their
beginning, including the session-start and first user-message events. Start the watcher
before creating the Codex conversation. Use `--replay-existing --once` to import
existing logs once.
The hook is observation-only:
it records what Codex actually did and retrieves a recommendation, but it never forces
Codex to execute a recommendation.
