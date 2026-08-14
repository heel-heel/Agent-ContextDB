from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .adapters import CodexJSONLAdapter, GenericJSONLAdapter, SWEAgentLLMAnnotatedTrajectoryAdapter, SWEAgentTrajectoryAdapter, TraceAdapter
from .service import ContextDB


ADAPTERS = {
    "codex-jsonl": CodexJSONLAdapter,
    "codex": CodexJSONLAdapter,
    "generic-jsonl": GenericJSONLAdapter,
    "jsonl": GenericJSONLAdapter,
    "generic": GenericJSONLAdapter,
    "swe-agent-traj": SWEAgentTrajectoryAdapter,
    "swe-agent": SWEAgentTrajectoryAdapter,
    "swe": SWEAgentTrajectoryAdapter,
    "swe-agent-traj-llm": SWEAgentLLMAnnotatedTrajectoryAdapter,
    "swe-agent-llm": SWEAgentLLMAnnotatedTrajectoryAdapter,
    "swe-llm": SWEAgentLLMAnnotatedTrajectoryAdapter,
}


def get_adapter(source: str) -> TraceAdapter:
    try:
        return ADAPTERS[source]()
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"unknown trace source '{source}'. available: {available}") from exc


def normalize_trace_to_jsonl(path: str | Path, out_path: str | Path, source: str = "generic-jsonl", branch_id: str = "main", agent_id: str = "external-agent") -> Dict[str, Any]:
    """Normalize a framework trace into ContextDB generic JSONL without importing it.

    The output can be consumed by GenericJSONLAdapter, so downstream commands can
    use `import-trace --source generic-jsonl` or `demo --trace` without touching
    the original framework-specific format again.
    """
    adapter = get_adapter(source)
    raw_trace = adapter.load(path)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    first_type = None
    last_type = None
    with output.open("w", encoding="utf-8") as fh:
        for index, event in enumerate(adapter.iter_events(raw_trace), start=1):
            record = {
                "event_type": event.event_type,
                "payload": event.payload,
                "actor": event.actor,
                "branch_id": event.branch_id if event.branch_id != "main" or branch_id == "main" else branch_id,
                "refs": event.refs,
                "metadata": {
                    **event.metadata,
                    "normalized_from": str(path),
                    "normalizer_source": adapter.source_name,
                    "normalizer_event_index": index,
                    "agent_id": agent_id,
                },
            }
            if event.parent_event_ids:
                record["parent_event_ids"] = event.parent_event_ids
            if event.timestamp:
                record["timestamp"] = event.timestamp
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            first_type = first_type or event.event_type
            last_type = event.event_type
    return {
        "source_path": str(path),
        "out_path": str(output),
        "source": adapter.source_name,
        "output_source": "generic-jsonl",
        "agent_id": agent_id,
        "branch_id": branch_id,
        "event_count": count,
        "first_event_type": first_type,
        "last_event_type": last_type,
    }


def import_trace(path: str | Path, root: str = "data", source: str = "generic-jsonl", title: str | None = None, agent_id: str = "external-agent", branch_id: str = "main") -> Dict[str, Any]:
    adapter = get_adapter(source)
    raw_trace = adapter.load(path)
    db = ContextDB(root)
    trace_title = title or f"Imported {adapter.source_name} trace: {Path(path).name}"
    traj = db.create_trajectory(
        trace_title,
        agent_id=agent_id,
        source_id=adapter.source_name,
        metadata={"import_path": str(path), "adapter": adapter.source_name},
    )
    tid = traj["trajectory_id"]
    if branch_id != "main":
        db.create_branch(tid, branch_id, from_branch="main")
    imported = []
    for normalized in adapter.iter_events(raw_trace):
        event = db.append_event(
            tid,
            normalized.event_type,
            normalized.payload,
            branch_id=normalized.branch_id if normalized.branch_id != "main" or branch_id == "main" else branch_id,
            actor=normalized.actor,
            parent_event_ids=normalized.parent_event_ids,
            refs=normalized.refs,
            metadata={**normalized.metadata, "imported_timestamp": normalized.timestamp} if normalized.timestamp else normalized.metadata,
        )
        imported.append(event)
    db.query_view(tid, "summary", branch_id)
    #db.query_view(tid, "failures", branch_id)
    db.query_view(tid, "current_prompt", branch_id)
    return {
        "trajectory_id": tid,
        "title": trace_title,
        "source": adapter.source_name,
        "agent_id": agent_id,
        "event_count": len(imported),
        "branch_id": branch_id,
        "first_event_id": imported[0]["event_id"] if imported else None,
        "head_event_id": imported[-1]["event_id"] if imported else None,
    }


def replay_trace(path: str | Path, root: str = "data", source: str = "generic-jsonl", title: str | None = None, agent_id: str = "external-agent", branch_id: str = "main") -> Dict[str, Any]:
    """Replay an agent trace that may include ContextDB operations.

    Supported operation records:
      {"op":"event", ...}
      {"op":"snapshot", "as":"clean", "message":"..."}
      {"op":"rollback", "snapshot":"clean", "branch_id":"rollback-clean"}
      {"op":"branch", "branch_id":"try-fix", "from_branch":"rollback-clean"}

    Plain JSONL event records without `op` are treated as `op=event`.
    """
    adapter = get_adapter(source)
    records = adapter.load(path)
    db = ContextDB(root)
    trace_title = title or f"Replay {adapter.source_name} trace: {Path(path).name}"
    traj = db.create_trajectory(
        trace_title,
        agent_id=agent_id,
        source_id=adapter.source_name,
        metadata={"import_path": str(path), "adapter": adapter.source_name, "replay": True},
    )
    tid = traj["trajectory_id"]
    snapshots: Dict[str, Dict[str, Any]] = {}
    last_event_by_branch: Dict[str, str] = {}
    imported = []
    operations = []

    for record in records:
        op = record.get("op", "event")
        if op == "event":
            normalized = adapter._normalize(record)  # type: ignore[attr-defined]
            target_branch = normalized.branch_id if normalized.branch_id != "main" or branch_id == "main" else branch_id
            event = db.append_event(
                tid,
                normalized.event_type,
                normalized.payload,
                branch_id=target_branch,
                actor=normalized.actor,
                parent_event_ids=normalized.parent_event_ids,
                refs=normalized.refs,
                metadata={**normalized.metadata, "imported_timestamp": normalized.timestamp} if normalized.timestamp else normalized.metadata,
            )
            imported.append(event)
            last_event_by_branch[target_branch] = event["event_id"]
        elif op == "snapshot":
            name = record.get("as") or record.get("name") or record.get("snapshot")
            snap = db.snapshot(tid, branch_id=record.get("branch_id", branch_id), message=record.get("message", ""))
            if name:
                snapshots[name] = snap
            operations.append({"op": "snapshot", "snapshot_id": snap["snapshot_id"], "name": name})
        elif op == "rollback":
            snap_ref = record.get("snapshot") or record.get("snapshot_id")
            snap = snapshots.get(snap_ref, {"snapshot_id": snap_ref})
            rollback = db.rollback(tid, snap["snapshot_id"], target_branch_id=record.get("branch_id") or record.get("target_branch_id", "rollback"))
            operations.append({"op": "rollback", "branch_id": rollback["branch_id"], "snapshot_id": snap["snapshot_id"]})
        elif op == "branch":
            new_branch = record["branch_id"]
            from_branch = record.get("from_branch", branch_id)
            base_event_id = record.get("base_event_id") or last_event_by_branch.get(from_branch)
            branch = db.create_branch(tid, new_branch, base_event_id=base_event_id, from_branch=from_branch)
            operations.append({"op": "branch", "branch_id": branch["branch_id"], "from_branch": from_branch})
        elif op == "materialize_skills":
            view_branch = record.get("branch_id", branch_id)
            view = db.query_view(tid, "learned_skills", view_branch)
            operations.append({"op": "materialize_skills", "branch_id": view_branch, "skill_count": len(view.get("content", []))})
        elif op == "apply_skill":
            failure = record.get("failure") or {}
            target_branch = record.get("branch_id", branch_id)
            result = db.apply_skill(
                tid,
                failure,
                branch_id=target_branch,
                top_k=int(record.get("top_k", 1)),
                outcome_status=record.get("outcome_status", "ok"),
            )
            operations.append({"op": "apply_skill", "branch_id": target_branch, "match_event_id": result.get("match_event_id"), "result_event_id": result.get("result_event_id")})
        else:
            raise ValueError(f"unknown trace operation: {op}")

    default_view_branch = record.get("branch_id", branch_id) if records else branch_id
    for view_name in ("summary", "failures", "current_prompt"):
        try:
            db.query_view(tid, view_name, default_view_branch)
        except Exception:
            pass
    graph = db.graph(tid)
    return {
        "trajectory_id": tid,
        "title": trace_title,
        "source": adapter.source_name,
        "agent_id": agent_id,
        "event_count": len(imported),
        "operation_count": len(operations),
        "branches": [b["branch_id"] for b in graph["branches"]],
        "snapshots": [s["snapshot_id"] for s in graph["snapshots"]],
        "head_event_id": imported[-1]["event_id"] if imported else None,
    }
