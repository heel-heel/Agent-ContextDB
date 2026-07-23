from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Branch, ContextEvent, Snapshot, Trajectory, View, new_id, to_dict, utc_now
from .store import SQLiteStore
from . import uris


class ContextDB:
    def __init__(self, root: str | Path = "data"):
        self.store = SQLiteStore(root)

    def create_trajectory(self, title: str, agent_id: str = "unknown-agent", source_id: str = "unknown-source", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tid = new_id("traj")
        traj = Trajectory(trajectory_id=tid, title=title, agent_id=agent_id, source_id=source_id, metadata=metadata or {})
        self.store.create_namespace(f"trajectories/{tid}/events")
        self.store.create_namespace(f"trajectories/{tid}/branches")
        self.store.create_namespace(f"trajectories/{tid}/views")
        self.store.put_object(uris.trajectory_meta(tid), to_dict(traj))
        main = Branch(branch_id="main", trajectory_id=tid)
        self.store.put_object(uris.branch_key(tid, "main"), to_dict(main))
        return to_dict(traj)

    def get_trajectory(self, trajectory_id: str) -> Dict[str, Any]:
        obj = self.store.get_object(uris.trajectory_meta(trajectory_id))
        if obj is None:
            raise KeyError(f"trajectory not found: {trajectory_id}")
        return obj

    def append_event(self, trajectory_id: str, event_type: str, payload: Dict[str, Any], branch_id: str = "main", actor: str = "agent", parent_event_ids: Optional[List[str]] = None, refs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        branch = self.get_branch(trajectory_id, branch_id)
        parents = parent_event_ids if parent_event_ids is not None else ([branch["head_event_id"]] if branch.get("head_event_id") else [])
        event = ContextEvent(
            event_id=new_id("evt"), trajectory_id=trajectory_id, branch_id=branch_id,
            parent_event_ids=parents, event_type=event_type, actor=actor,
            payload=payload, refs=refs or {}, metadata=metadata or {},
        )
        event_dict = to_dict(event)
        self.store.put_object(uris.event_key(trajectory_id, event.event_id), event_dict)
        branch["head_event_id"] = event.event_id
        self.store.put_object(uris.branch_key(trajectory_id, branch_id), branch)
        traj = self.get_trajectory(trajectory_id)
        if branch_id == traj.get("default_branch", "main"):
            traj["head_event_id"] = event.event_id
        traj["updated_at"] = utc_now()
        self.store.put_object(uris.trajectory_meta(trajectory_id), traj)
        return event_dict

    def get_event(self, trajectory_id: str, event_id: str) -> Dict[str, Any]:
        obj = self.store.get_object(uris.event_key(trajectory_id, event_id))
        if obj is None:
            raise KeyError(f"event not found: {event_id}")
        return obj

    def list_events(self, trajectory_id: str, branch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/events"))
        if branch_id:
            events = [e for e in events if e.get("branch_id") == branch_id]
        return sorted(events, key=lambda e: e.get("timestamp", ""))

    def get_branch(self, trajectory_id: str, branch_id: str) -> Dict[str, Any]:
        obj = self.store.get_object(uris.branch_key(trajectory_id, branch_id))
        if obj is None:
            raise KeyError(f"branch not found: {branch_id}")
        return obj

    def create_branch(self, trajectory_id: str, branch_id: str, base_event_id: Optional[str] = None, from_branch: str = "main") -> Dict[str, Any]:
        if self.store.get_object(uris.branch_key(trajectory_id, branch_id)) is not None:
            raise ValueError(f"branch already exists: {branch_id}")
        base = base_event_id or self.get_branch(trajectory_id, from_branch).get("head_event_id")
        branch = Branch(branch_id=branch_id, trajectory_id=trajectory_id, base_event_id=base, head_event_id=base)
        self.store.put_object(uris.branch_key(trajectory_id, branch_id), to_dict(branch))
        return to_dict(branch)

    def query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        trajectory_id = filters.get("trajectory_id")
        if not trajectory_id:
            raise ValueError("query requires trajectory_id")
        events = self.list_events(trajectory_id, filters.get("branch_id"))
        event_type = filters.get("event_type")
        if isinstance(event_type, str):
            events = [e for e in events if e.get("event_type") == event_type]
        elif isinstance(event_type, list):
            events = [e for e in events if e.get("event_type") in set(event_type)]
        contains = filters.get("contains")
        if contains:
            needle = str(contains).lower()
            events = [e for e in events if needle in json.dumps(e, ensure_ascii=False).lower()]
        limit = int(filters.get("limit", len(events)))
        return events[-limit:]

    def query_view(self, trajectory_id: str, view_name: str, branch_id: str = "main", token_budget: int = 4000) -> Dict[str, Any]:
        events = self.list_events(trajectory_id, branch_id)
        if view_name == "memory":
            content = [e["payload"] for e in events if e.get("event_type") == "memory_update"]
        elif view_name == "summary":
            facts = [self._event_line(e) for e in events[-12:]]
            content = "\n".join(f"- {x}" for x in facts)
        elif view_name == "failures":
            content = [e for e in events if e.get("event_type") == "tool_result" and e.get("payload", {}).get("status") in {"failed", "error"}]
        elif view_name == "current_prompt":
            recent = events[-8:]
            memory = [e["payload"] for e in events if e.get("event_type") == "memory_update"][-5:]
            content = {"recent_events": recent, "relevant_memory": memory, "token_budget": token_budget}
        elif view_name == "rl_dataset":
            content = self.export_rl_dataset(trajectory_id, branch_id)
        else:
            content = events
        view = View(view_name=view_name, trajectory_id=trajectory_id, branch_id=branch_id, content=content, source_events=[e["event_id"] for e in events])
        view_dict = to_dict(view)
        self.store.put_object(uris.view_key(trajectory_id, branch_id, view_name), view_dict)
        return view_dict

    def snapshot(self, trajectory_id: str, branch_id: str = "main", message: str = "") -> Dict[str, Any]:
        keys = self.store.list_objects(f"trajectories/{trajectory_id}")
        branch = self.get_branch(trajectory_id, branch_id)
        snapshot = Snapshot(snapshot_id=new_id("snap"), trajectory_id=trajectory_id, branch_id=branch_id, event_id=branch.get("head_event_id"), message=message, object_keys=keys)
        snap_dict = to_dict(snapshot)
        self.store.snapshot_objects(snapshot.snapshot_id, keys)
        self.store.put_object(uris.snapshot_key(trajectory_id, snapshot.snapshot_id), snap_dict)
        branch["snapshot_id"] = snapshot.snapshot_id
        self.store.put_object(uris.branch_key(trajectory_id, branch_id), branch)
        return snap_dict

    def rollback(self, trajectory_id: str, snapshot_id: str, target_branch_id: str = "rollback") -> Dict[str, Any]:
        key_prefix = f"trajectories/{trajectory_id}"
        self.store.restore_snapshot_files(snapshot_id, key_prefix)
        snap = self.store.get_object(uris.snapshot_key(trajectory_id, snapshot_id)) or {"event_id": None}
        branch = Branch(branch_id=target_branch_id, trajectory_id=trajectory_id, base_event_id=snap.get("event_id"), head_event_id=snap.get("event_id"), snapshot_id=snapshot_id, metadata={"rollback_from": snapshot_id})
        self.store.put_object(uris.branch_key(trajectory_id, target_branch_id), to_dict(branch))
        return to_dict(branch)

    def diff(self, trajectory_id: str, left_branch: str, right_branch: str) -> Dict[str, Any]:
        left = {e["event_id"]: e for e in self.list_events(trajectory_id, left_branch)}
        right = {e["event_id"]: e for e in self.list_events(trajectory_id, right_branch)}
        return {"only_left": list(left.keys() - right.keys()), "only_right": list(right.keys() - left.keys()), "shared": list(left.keys() & right.keys())}

    def stream_context(self, trajectory_id: str, branch_id: str = "main", token_budget: int = 4000) -> Dict[str, Any]:
        return self.query_view(trajectory_id, "current_prompt", branch_id, token_budget)

    def export_rl_dataset(self, trajectory_id: str, branch_id: str = "main") -> List[Dict[str, Any]]:
        events = self.list_events(trajectory_id, branch_id)
        rows = []
        for i, e in enumerate(events):
            if e.get("event_type") == "assistant_message":
                prev = events[max(0, i - 3):i]
                rows.append({"state_events": prev, "action": e.get("payload"), "event_id": e.get("event_id")})
        return rows

    def log(self, trajectory_id: str) -> Dict[str, Any]:
        branches = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/branches"))
        snapshots = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/snapshots"))
        events = self.list_events(trajectory_id)
        return {"trajectory": self.get_trajectory(trajectory_id), "branches": branches, "snapshots": snapshots, "event_count": len(events)}

    def _event_line(self, event: Dict[str, Any]) -> str:
        payload = event.get("payload", {})
        text = payload.get("text") or payload.get("command") or payload.get("summary") or str(payload)[:120]
        return f"{event.get('event_type')}:{text}"
