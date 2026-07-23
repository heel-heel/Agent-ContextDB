# Agent ContextDB Demo

ContextDB is an independent Agent Context Database prototype. It models agent state as durable, queryable, versioned data instead of framework-private transcript files.

## What is implemented

- Logical objects: Agent, Source, Trajectory, ContextEvent, Branch, Snapshot, View, Artifact.
- Agent adapter shape: `import-transcript` converts simple transcript JSON into ContextEvents.
- ContextDB API: create trajectory, append event, get event, query, query_view, branch, snapshot, rollback, stream_context, export_rl_dataset.
- Query/View/Index: event indexes by trajectory, branch, type, timestamp; materialized views for memory, summary, failures, current_prompt, and rl_dataset.
- Versioning: Git-like snapshot, branch, rollback, diff, log over trajectory DAG state.
- Storage abstraction: ContextStore interface with a SQLiteStore backend and object payload persistence under `data/objects`.
- HTTP server: standard-library server, no third-party runtime dependencies.

## Run

```bash
source /home/benjamin/miniconda3/envs/contextdb_env/bin/activate
cd /home/benjamin/windows/contextdb
contextdb demo
contextdb serve --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```
