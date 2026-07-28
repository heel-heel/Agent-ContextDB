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
- Materialized views: memory, summary, failures, current_prompt, and rl_dataset.
- Streamed context loading: `current_prompt` returns summary + recent events + memory plus estimated token savings.
- Graph API: `/api/v1/graph` returns nodes, parent edges, branches, snapshots, and materialized views.
- Diff API: `/api/v1/diff` compares branches by event set, failure count, and summary view changes.
- HTTP server and a dependency-free ICDE dashboard at `/demo`.

## Run in VMware Ubuntu

```bash
source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
cd /home/benjamin/windows/contextdb
contextdb demo
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

## Useful CLI commands

```bash
contextdb demo
contextdb graph <trajectory_id>
contextdb query-view <trajectory_id> current_prompt --branch clang-attempt
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

1. Run `contextdb demo` to seed a realistic agent setup trajectory.
2. Show `/demo` graph: the main branch fails on native build, while `docker-attempt` and `clang-attempt` branch from earlier states.
3. Switch views: `current_prompt` shows streamed loading and token savings, `failures` mines failed tool results, `rl_dataset` turns assistant actions into training rows.
4. Run `contextdb diff` to compare successful repair strategies.
5. Explain the key claim: context is no longer only prompt text; it is a first-class, persistent, queryable, versioned database object.
