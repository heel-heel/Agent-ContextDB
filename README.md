# Agent ContextDB Demo

ContextDB is an independent Agent Context Database prototype. It models agent state as durable, queryable, versioned data instead of framework-private transcript files.

The ICDE demo angle is **Git + Database for agent context**: messages, tool calls, tool results, memory updates, summaries, artifacts, and skills can be represented as events in a branchable trajectory DAG. Views such as memory, failures, summaries, prompt windows, and RL rows are derived database views over the same trajectory, not separate ad-hoc files.

## Relation to OpenViking

OpenViking focuses on a context database and virtual filesystem for organizing and retrieving memory, resources, and skills. ContextDB is positioned one layer below/alongside that idea: it stores the full agent execution trajectory as versioned database state, then exposes retrieval-ready views that systems such as OpenViking-style context loaders could consume.

## What is implemented

- Logical objects: Agent, Source, Trajectory, ContextEvent, Branch, Snapshot, View, Artifact.
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

`semantic_repair_judgments` optionally calls a real LLM to judge whether each structural repair candidate is semantically a likely repair, partial repair, validation-only action, unrelated success, or insufficient-context case. By default it is disabled for reproducible offline demos. To use Alibaba Cloud Bailian/Qwen through the OpenAI-compatible API, set `CONTEXTDB_LLM_PROVIDER=qwen`, `DASHSCOPE_API_KEY=<key>`, `CONTEXTDB_LLM_MODEL=qwen3.7-max`, and `CONTEXTDB_LLM_BASE_URL=https://ws-5jkepnkdq4vt4m5c.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`. For offline UI testing, set `CONTEXTDB_LLM_PROVIDER=mock`.

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
