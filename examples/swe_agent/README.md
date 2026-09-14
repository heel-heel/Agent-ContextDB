# SWE-agent / SWE-bench trajectory sample

This directory contains a small official SWE-agent `.traj` sample used to test ContextDB import compatibility with software-engineering agent trajectories.

Source file:

- `marshmallow-code__marshmallow-1867.traj`
- Official URL: https://raw.githubusercontent.com/SWE-agent/SWE-agent/main/trajectories/demonstrations/replay__marshmallow-code__marshmallow-1867__default__t-0.20__p-0.95__c-2.00__install-1___install_from_source/marshmallow-code__marshmallow-1867.traj

Why this source is used:

- SWE-bench provides real GitHub issue repair tasks for software-engineering agents.
- SWE-agent produces step-level `.traj` files on those tasks, with `thought`, `response`, `action`, `observation`, and final submitted patch metadata.
- ContextDB imports this trajectory through the `swe-agent-traj` adapter without changing the original `examples/demo_trace.jsonl` demo.

Commands:

```bash
contextdb import-trace examples/swe_agent/marshmallow-code__marshmallow-1867.traj --source swe-agent-traj --agent-id swe-agent
contextdb swe-demo
```


## LLM-assisted annotation

The default adapter is deterministic and reproducible:

```bash
contextdb import-trace examples/swe_agent/marshmallow-code__marshmallow-1867.traj --source swe-agent-traj --agent-id swe-agent
contextdb swe-demo --no-llm-annotate
```

By default, `contextdb swe-demo` enables the LLM semantic annotation pass for ambiguous `tool_result` status and error signatures. It loads the dedicated background configuration from `_API/background_llm.env`; the bundled configuration uses `CONTEXTDB_BACKGROUND_LLM_PROVIDER=openai-compatible` and `CONTEXTDB_BACKGROUND_LLM_MODEL=deepseek-flash`. If those credentials are unavailable, the annotation result records that the provider is disabled or unavailable.

```bash
export CONTEXTDB_LLM_PROVIDER=mock   # or qwen/openai-compatible with API credentials
contextdb swe-demo
contextdb import-trace examples/swe_agent/marshmallow-code__marshmallow-1867.traj --source swe-agent-traj-llm --agent-id swe-agent
```

The normalized event keeps the deterministic `adapter_status` in metadata and stores `llm_trace_annotation` separately. The final `payload.status` is resolved from the LLM annotation only when the annotation is confident enough.


## Pre-normalized ContextDB JSONL

You can materialize the normalized/annotated trace once and reuse it through the generic JSONL path:

```bash
export CONTEXTDB_LLM_PROVIDER=mock   # or qwen with API credentials
contextdb normalize-trace examples/swe_agent/marshmallow-code__marshmallow-1867.traj \
  --source swe-agent-traj-llm \
  --agent-id swe-agent \
  --out examples/swe_agent/marshmallow-code__marshmallow-1867.contextdb.jsonl

contextdb import-trace examples/swe_agent/marshmallow-code__marshmallow-1867.contextdb.jsonl \
  --source generic-jsonl \
  --agent-id swe-agent
```

This preserves the official raw `.traj` file and stores the processed ContextDB event stream separately.


## SWE demo JSONL-first flow

`contextdb swe-demo` now writes a processed ContextDB JSONL file first, then replays/imports that JSONL through the same generic JSONL path used by the hand-authored demo trace.

Default output:

```text
examples/swe_agent/marshmallow-code__marshmallow-1867.contextdb.jsonl
```

Override it with:

```bash
contextdb swe-demo --out examples/swe_agent/custom.contextdb.jsonl
```
