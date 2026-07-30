from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Branch, ContextEvent, Snapshot, Trajectory, View, new_id, to_dict, utc_now
from .store import SQLiteStore
from . import uris


FAILURE_STATUSES = {"failed", "error", "timeout"}


class ContextDB:
    def __init__(self, root: str | Path = "data"):
        self.store = SQLiteStore(root)

    def create_trajectory(self, title: str, agent_id: str = "unknown-agent", source_id: str = "unknown-source", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tid = new_id("traj")
        traj = Trajectory(trajectory_id=tid, title=title, agent_id=agent_id, source_id=source_id, metadata=metadata or {})
        self.store.create_namespace(f"trajectories/{tid}/events")
        self.store.create_namespace(f"trajectories/{tid}/branches")
        self.store.create_namespace(f"trajectories/{tid}/views")
        self.store.create_namespace(f"trajectories/{tid}/snapshots")
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

    def _all_events(self, trajectory_id: str) -> List[Dict[str, Any]]:
        return sorted(list(self.store.scan_prefix(f"trajectories/{trajectory_id}/events")), key=lambda e: e.get("timestamp", ""))

    def _event_map(self, trajectory_id: str) -> Dict[str, Dict[str, Any]]:
        return {e["event_id"]: e for e in self._all_events(trajectory_id)}

    def _reachable_event_ids(self, trajectory_id: str, head_event_id: Optional[str]) -> Set[str]:
        events = self._event_map(trajectory_id)
        reachable: Set[str] = set()
        stack = [head_event_id] if head_event_id else []
        while stack:
            eid = stack.pop()
            if not eid or eid in reachable:
                continue
            event = events.get(eid)
            if event is None:
                continue
            reachable.add(eid)
            stack.extend(event.get("parent_event_ids") or [])
        return reachable

    def list_events(self, trajectory_id: str, branch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = self._all_events(trajectory_id)
        if not branch_id:
            return events
        branch = self.get_branch(trajectory_id, branch_id)
        reachable = self._reachable_event_ids(trajectory_id, branch.get("head_event_id"))
        return [e for e in events if e.get("event_id") in reachable]

    def get_branch(self, trajectory_id: str, branch_id: str) -> Dict[str, Any]:
        obj = self.store.get_object(uris.branch_key(trajectory_id, branch_id))
        if obj is None:
            raise KeyError(f"branch not found: {branch_id}")
        return obj

    def create_branch(self, trajectory_id: str, branch_id: str, base_event_id: Optional[str] = None, from_branch: str = "main") -> Dict[str, Any]:
        if self.store.get_object(uris.branch_key(trajectory_id, branch_id)) is not None:
            raise ValueError(f"branch already exists: {branch_id}")
        base = base_event_id or self.get_branch(trajectory_id, from_branch).get("head_event_id")
        branch = Branch(branch_id=branch_id, trajectory_id=trajectory_id, base_event_id=base, head_event_id=base, metadata={"from_branch": from_branch})
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
        actor = filters.get("actor")
        if actor:
            events = [e for e in events if e.get("actor") == actor]
        status = filters.get("status")
        if status:
            statuses = {status} if isinstance(status, str) else set(status)
            events = [e for e in events if e.get("payload", {}).get("status") in statuses]
        since = filters.get("since")
        if since:
            events = [e for e in events if e.get("timestamp", "") >= since]
        until = filters.get("until")
        if until:
            events = [e for e in events if e.get("timestamp", "") <= until]
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
            content = [e for e in events if e.get("event_type") == "tool_result" and e.get("payload", {}).get("status") in FAILURE_STATUSES]
        elif view_name == "failure_patterns":
            content = self.failure_patterns(trajectory_id, branch_id)
        elif view_name == "success_patterns":
            content = self.success_patterns(trajectory_id)
        elif view_name == "repair_strategies":
            content = self.repair_strategies(trajectory_id)
        elif view_name == "current_prompt":
            recent = events[-8:]
            memory = [e["payload"] for e in events if e.get("event_type") == "memory_update"][-5:]
            summary = self._compact_summary(events[:-8]) if len(events) > 8 else ""
            full_tokens = self._estimate_tokens(events)
            loaded_tokens = self._estimate_tokens(recent) + self._estimate_tokens(memory) + self._estimate_tokens(summary)
            content = {
                "summary": summary,
                "recent_events": recent,
                "relevant_memory": memory,
                "token_budget": token_budget,
                "estimated_full_history_tokens": full_tokens,
                "estimated_loaded_tokens": loaded_tokens,
                "estimated_saved_tokens": max(0, full_tokens - loaded_tokens),
                "loading_policy": "summary + recent_events + relevant_memory",
            }
        elif view_name == "rl_dataset":
            content = self.export_rl_dataset(trajectory_id, branch_id)
        else:
            content = events
        view = View(view_name=view_name, trajectory_id=trajectory_id, branch_id=branch_id, content=content, source_events=[e["event_id"] for e in events], metadata={"event_count": len(events)})
        view_dict = to_dict(view)
        self.store.put_object(uris.view_key(trajectory_id, branch_id, view_name), view_dict)
        return view_dict

    def snapshot(self, trajectory_id: str, branch_id: str = "main", message: str = "") -> Dict[str, Any]:
        branch = self.get_branch(trajectory_id, branch_id)
        keys = self.store.list_objects(f"trajectories/{trajectory_id}")
        snapshot = Snapshot(snapshot_id=new_id("snap"), trajectory_id=trajectory_id, branch_id=branch_id, event_id=branch.get("head_event_id"), message=message, object_keys=keys)
        snap_dict = to_dict(snapshot)
        self.store.put_object(uris.snapshot_key(trajectory_id, snapshot.snapshot_id), snap_dict)
        snap_dict["object_keys"] = self.store.list_objects(f"trajectories/{trajectory_id}")
        self.store.put_object(uris.snapshot_key(trajectory_id, snapshot.snapshot_id), snap_dict)
        self.store.snapshot_objects(snapshot.snapshot_id, snap_dict["object_keys"])
        branch["snapshot_id"] = snapshot.snapshot_id
        self.store.put_object(uris.branch_key(trajectory_id, branch_id), branch)
        return snap_dict

    def rollback(self, trajectory_id: str, snapshot_id: str, target_branch_id: str = "rollback") -> Dict[str, Any]:
        snap = self.store.get_object(uris.snapshot_key(trajectory_id, snapshot_id))
        if snap is None:
            raise KeyError(f"snapshot not found: {snapshot_id}")
        if self.store.get_object(uris.branch_key(trajectory_id, target_branch_id)) is not None:
            raise ValueError(f"branch already exists: {target_branch_id}")
        # Preserve the failed branch as analyzable history, and create a new branch whose
        # head points at the snapshot event. This matches database time-travel semantics.
        branch = Branch(
            branch_id=target_branch_id,
            trajectory_id=trajectory_id,
            base_event_id=snap.get("event_id"),
            head_event_id=snap.get("event_id"),
            snapshot_id=snapshot_id,
            metadata={"rollback_from": snapshot_id, "rollback_mode": "branch_from_snapshot"},
        )
        self.store.put_object(uris.branch_key(trajectory_id, target_branch_id), to_dict(branch))
        return to_dict(branch)

    def diff(self, trajectory_id: str, left_branch: str, right_branch: str) -> Dict[str, Any]:
        left_events = self.list_events(trajectory_id, left_branch)
        right_events = self.list_events(trajectory_id, right_branch)
        left = {e["event_id"]: e for e in left_events}
        right = {e["event_id"]: e for e in right_events}
        only_left_ids = sorted(left.keys() - right.keys(), key=lambda eid: left[eid].get("timestamp", ""))
        only_right_ids = sorted(right.keys() - left.keys(), key=lambda eid: right[eid].get("timestamp", ""))
        left_summary = self.query_view(trajectory_id, "summary", left_branch)["content"]
        right_summary = self.query_view(trajectory_id, "summary", right_branch)["content"]
        return {
            "trajectory_id": trajectory_id,
            "left_branch": left_branch,
            "right_branch": right_branch,
            "only_left": [left[eid] for eid in only_left_ids],
            "only_right": [right[eid] for eid in only_right_ids],
            "shared_event_count": len(left.keys() & right.keys()),
            "left_failure_count": len([e for e in left_events if e.get("payload", {}).get("status") in FAILURE_STATUSES]),
            "right_failure_count": len([e for e in right_events if e.get("payload", {}).get("status") in FAILURE_STATUSES]),
            "view_delta": {"summary_changed": left_summary != right_summary, "left_summary": left_summary, "right_summary": right_summary},
        }

    def graph(self, trajectory_id: str) -> Dict[str, Any]:
        events = self._all_events(trajectory_id)
        branches = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/branches"))
        snapshots = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/snapshots"))
        views = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/views"))
        branch_heads: Dict[str, List[str]] = {}
        for b in branches:
            if b.get("head_event_id"):
                branch_heads.setdefault(b["head_event_id"], []).append(b.get("branch_id"))
        snapshot_events: Dict[str, List[str]] = {}
        for s in snapshots:
            snapshot_events.setdefault(s.get("event_id"), []).append(s.get("snapshot_id"))
        nodes = []
        edges = []
        for e in events:
            eid = e["event_id"]
            nodes.append({
                "id": eid,
                "type": "event",
                "event_type": e.get("event_type"),
                "branch_id": e.get("branch_id"),
                "actor": e.get("actor"),
                "timestamp": e.get("timestamp"),
                "label": self._event_line(e),
                "is_branch_head": eid in branch_heads,
                "branch_heads": branch_heads.get(eid, []),
                "branch_head": ",".join(branch_heads.get(eid, [])),
                "snapshots": snapshot_events.get(eid, []),
            })
            for parent in e.get("parent_event_ids") or []:
                edges.append({"source": parent, "target": eid, "kind": "parent"})
        for b in branches:
            if b.get("base_event_id") and b.get("head_event_id") and b.get("base_event_id") != b.get("head_event_id"):
                edges.append({"source": b["base_event_id"], "target": b["head_event_id"], "kind": "branch", "branch_id": b.get("branch_id")})
        return {"trajectory": self.get_trajectory(trajectory_id), "nodes": nodes, "edges": edges, "branches": branches, "snapshots": snapshots, "views": views}

    def failure_patterns(self, trajectory_id: str, branch_id: str = "main") -> List[Dict[str, Any]]:
        events = self.list_events(trajectory_id, branch_id)
        patterns = []
        for i, event in enumerate(events):
            if not self._is_failed_tool_result(event):
                continue
            tool_call = self._previous_event(events, i, "tool_call")
            assistant = self._previous_event(events, i, "assistant_message")
            repair_events = self._following_repair_events(events, i)
            signature = self._result_signature(event)
            patterns.append({
                "pattern_id": f"fp_{event['event_id'].split('_')[-1]}",
                "failure_event_id": event["event_id"],
                "branch_id": event.get("branch_id"),
                "failed_tool": self._tool_name(tool_call),
                "failed_command": self._command_text(tool_call),
                "error_signature": signature,
                "preceding_action": self._payload_text(assistant),
                "likely_cause": self._infer_likely_cause(signature),
                "repair_events_after_failure": repair_events,
            })
        return patterns

    def success_patterns(self, trajectory_id: str) -> List[Dict[str, Any]]:
        events_by_branch = self._events_by_branch(trajectory_id)
        patterns = []
        for branch_id, events in events_by_branch.items():
            for i, event in enumerate(events):
                if not self._is_success_tool_result(event):
                    continue
                tool_call = self._previous_event(events, i, "tool_call")
                assistant = self._previous_event(events, i, "assistant_message")
                patterns.append({
                    "pattern_id": f"sp_{event['event_id'].split('_')[-1]}",
                    "success_event_id": event["event_id"],
                    "branch_id": branch_id,
                    "successful_tool": self._tool_name(tool_call),
                    "successful_command": self._command_text(tool_call),
                    "strategy": self._payload_text(assistant),
                    "outcome": event.get("payload", {}).get("status"),
                    "result_preview": event.get("payload", {}).get("preview", ""),
                })
        return patterns

    def repair_strategies(self, trajectory_id: str) -> List[Dict[str, Any]]:
        failures = []
        for branch in self._branches(trajectory_id):
            failures.extend(self.failure_patterns(trajectory_id, branch["branch_id"]))
        successes = self.success_patterns(trajectory_id)
        events = self._event_map(trajectory_id)
        branches = {b["branch_id"]: b for b in self._branches(trajectory_id)}
        snapshots = {s["snapshot_id"]: s for s in self._snapshots(trajectory_id)}
        strategies = []
        seen = set()
        for failure in failures:
            failure_event = events.get(failure["failure_event_id"])
            if not failure_event:
                continue
            repairs = []
            for success in successes:
                success_event = events.get(success["success_event_id"])
                if not success_event:
                    continue
                evidence = self._repair_link_evidence(failure_event, success_event, branches, snapshots, trajectory_id)
                structural_evidence = [item for item in evidence if item.get("strength") == "structural"]
                if not structural_evidence:
                    continue
                key = (failure["failure_event_id"], success["success_event_id"])
                if key in seen:
                    continue
                seen.add(key)
                repairs.append({
                    "branch_id": success["branch_id"],
                    "success_event_id": success["success_event_id"],
                    "strategy": success.get("strategy"),
                    "tool": success.get("successful_tool"),
                    "command": success.get("successful_command"),
                    "outcome": success.get("outcome"),
                    "link_type": evidence[0]["rule"],
                    "evidence": evidence,
                })
            if repairs:
                strategies.append({
                    "failure": {
                        "failure_event_id": failure["failure_event_id"],
                        "branch_id": failure["branch_id"],
                        "command": failure.get("failed_command"),
                        "tool": failure.get("failed_tool"),
                        "error_signature": failure.get("error_signature"),
                        "preceding_action": failure.get("preceding_action"),
                        "likely_cause": failure.get("likely_cause"),
                    },
                    "repairs": repairs,
                })
        return strategies

    def _repair_link_evidence(self, failure_event: Dict[str, Any], success_event: Dict[str, Any], branches: Dict[str, Dict[str, Any]], snapshots: Dict[str, Dict[str, Any]], trajectory_id: str) -> List[Dict[str, Any]]:
        evidence = []
        failure_id = failure_event["event_id"]
        success_id = success_event["event_id"]
        failure_branch = failure_event.get("branch_id")
        success_branch = success_event.get("branch_id")
        if failure_branch == success_branch:
            branch_events = self.list_events(trajectory_id, failure_branch)
            failure_index = self._index_in_branch(trajectory_id, failure_branch, failure_id)
            success_index = self._index_in_branch(trajectory_id, success_branch, success_id)
            first_success_after_failure = None
            if failure_index >= 0 and success_index > failure_index:
                for event in branch_events[failure_index + 1:]:
                    if self._is_success_tool_result(event):
                        first_success_after_failure = event
                        break
            if first_success_after_failure and first_success_after_failure.get("event_id") == success_id:
                evidence.append({
                    "rule": "same_branch_first_success_after_failure",
                    "strength": "structural",
                    "branch_id": failure_branch,
                    "failure_event_id": failure_id,
                    "success_event_id": success_id,
                })
        success_branch_obj = branches.get(success_branch, {})
        success_is_first_on_branch = self._is_first_success_on_branch(trajectory_id, success_branch, success_id) if success_branch else False
        if success_branch_obj.get("base_event_id") == failure_id and success_is_first_on_branch:
            evidence.append({"rule": "branch_from_failure_first_success", "strength": "structural", "repair_branch": success_branch, "base_event_id": failure_id, "success_event_id": success_id})
        failure_ancestors = self._reachable_event_ids(trajectory_id, failure_id)
        base_event_id = success_branch_obj.get("base_event_id")
        if base_event_id and base_event_id in failure_ancestors and base_event_id != failure_id and success_is_first_on_branch:
            evidence.append({"rule": "branch_from_failure_ancestor_first_success", "strength": "structural", "repair_branch": success_branch, "base_event_id": base_event_id, "failure_event_id": failure_id, "success_event_id": success_id})
        from_branch = success_branch_obj.get("metadata", {}).get("from_branch")
        rollback_branch = branches.get(from_branch or "", {})
        rollback_snapshot_id = rollback_branch.get("metadata", {}).get("rollback_from")
        rollback_snapshot = snapshots.get(rollback_snapshot_id or "")
        if from_branch and rollback_snapshot_id and rollback_snapshot and rollback_snapshot.get("event_id") in failure_ancestors and success_is_first_on_branch:
            evidence.append({
                "rule": "rollback_then_repair_first_success",
                "strength": "structural",
                "rollback_branch": from_branch,
                "repair_branch": success_branch,
                "snapshot_id": rollback_snapshot_id,
                "snapshot_event_id": rollback_snapshot.get("event_id"),
                "failure_event_id": failure_id,
            })
        failed_call = self._previous_event(self.list_events(trajectory_id, failure_branch), self._index_in_branch(trajectory_id, failure_branch, failure_id), "tool_call") if failure_branch else None
        success_call = self._previous_event(self.list_events(trajectory_id, success_branch), self._index_in_branch(trajectory_id, success_branch, success_id), "tool_call") if success_branch else None
        if failed_call and success_call and self._tool_name(failed_call) == self._tool_name(success_call) and self._commands_overlap(self._command_text(failed_call), self._command_text(success_call)):
            evidence.append({"rule": "same_tool_command_variant", "strength": "supporting", "failed_command": self._command_text(failed_call), "successful_command": self._command_text(success_call)})
        return evidence

    def _events_by_branch(self, trajectory_id: str) -> Dict[str, List[Dict[str, Any]]]:
        return {b["branch_id"]: self.list_events(trajectory_id, b["branch_id"]) for b in self._branches(trajectory_id)}

    def _branches(self, trajectory_id: str) -> List[Dict[str, Any]]:
        return list(self.store.scan_prefix(f"trajectories/{trajectory_id}/branches"))

    def _snapshots(self, trajectory_id: str) -> List[Dict[str, Any]]:
        return list(self.store.scan_prefix(f"trajectories/{trajectory_id}/snapshots"))

    def _previous_event(self, events: List[Dict[str, Any]], index: int, event_type: str) -> Optional[Dict[str, Any]]:
        for event in reversed(events[:index]):
            if event.get("event_type") == event_type:
                return event
        return None

    def _following_repair_events(self, events: List[Dict[str, Any]], index: int, limit: int = 4) -> List[Dict[str, Any]]:
        repairs = []
        for event in events[index + 1:]:
            if event.get("event_type") in {"assistant_message", "tool_call", "tool_result", "file_edit"}:
                repairs.append({"event_id": event["event_id"], "event_type": event.get("event_type"), "summary": self._event_line(event)})
            if len(repairs) >= limit:
                break
        return repairs

    def _is_first_success_on_branch(self, trajectory_id: str, branch_id: str, success_event_id: str) -> bool:
        for event in self.list_events(trajectory_id, branch_id):
            if event.get("branch_id") != branch_id:
                continue
            if self._is_success_tool_result(event):
                return event.get("event_id") == success_event_id
        return False

    def _index_in_branch(self, trajectory_id: str, branch_id: str, event_id: str) -> int:
        for i, event in enumerate(self.list_events(trajectory_id, branch_id)):
            if event.get("event_id") == event_id:
                return i
        return -1

    def _is_failed_tool_result(self, event: Dict[str, Any]) -> bool:
        return event.get("event_type") == "tool_result" and event.get("payload", {}).get("status") in FAILURE_STATUSES

    def _is_success_tool_result(self, event: Dict[str, Any]) -> bool:
        return event.get("event_type") == "tool_result" and event.get("payload", {}).get("status") == "ok"

    def _payload_text(self, event: Optional[Dict[str, Any]]) -> str:
        if not event:
            return ""
        payload = event.get("payload", {})
        return str(payload.get("text") or payload.get("summary") or payload.get("preview") or payload.get("command") or "")

    def _tool_name(self, event: Optional[Dict[str, Any]]) -> str:
        return str((event or {}).get("payload", {}).get("tool_name") or "")

    def _command_text(self, event: Optional[Dict[str, Any]]) -> str:
        return str((event or {}).get("payload", {}).get("command") or "")

    def _result_signature(self, event: Dict[str, Any]) -> str:
        payload = event.get("payload", {})
        return str(payload.get("preview") or payload.get("stderr") or payload.get("output") or payload.get("message") or "")[:500]

    def _infer_likely_cause(self, signature: str) -> str:
        text = signature.lower()
        if "gcc" in text or "compiler" in text or "clang" in text:
            return "compiler or native dependency incompatibility"
        if "timeout" in text:
            return "timeout or long-running tool execution"
        if "permission" in text or "denied" in text:
            return "permission or sandbox restriction"
        if "assert" in text or "test" in text:
            return "test assertion or behavior mismatch"
        return "unknown; inspect error signature and preceding tool call"

    def _commands_overlap(self, left: str, right: str) -> bool:
        left_tokens = {x for x in left.replace("=", " ").replace("/", " ").split() if len(x) > 2}
        right_tokens = {x for x in right.replace("=", " ").replace("/", " ").split() if len(x) > 2}
        return bool(left_tokens & right_tokens)

    def stream_context(self, trajectory_id: str, branch_id: str = "main", token_budget: int = 4000) -> Dict[str, Any]:
        return self.query_view(trajectory_id, "current_prompt", branch_id, token_budget)

    def export_rl_dataset(self, trajectory_id: str, branch_id: str = "main") -> List[Dict[str, Any]]:
        events = self.list_events(trajectory_id, branch_id)
        rows = []
        for i, e in enumerate(events):
            if e.get("event_type") == "assistant_message":
                prev = events[max(0, i - 3):i]
                future = events[i + 1:min(len(events), i + 4)]
                reward = self._infer_reward(future)
                rows.append({"state_events": prev, "action": e.get("payload"), "event_id": e.get("event_id"), "branch_id": branch_id, "reward_hint": reward})
        return rows

    def log(self, trajectory_id: str) -> Dict[str, Any]:
        branches = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/branches"))
        snapshots = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/snapshots"))
        events = self.list_events(trajectory_id)
        return {"trajectory": self.get_trajectory(trajectory_id), "branches": branches, "snapshots": snapshots, "event_count": len(events)}

    def _event_line(self, event: Dict[str, Any]) -> str:
        payload = event.get("payload", {})
        text = payload.get("text") or payload.get("command") or payload.get("summary") or payload.get("preview") or str(payload)[:120]
        return f"{event.get('event_type')}:{text}"

    def _compact_summary(self, events: List[Dict[str, Any]]) -> str:
        if not events:
            return ""
        lines = [self._event_line(e) for e in events[-6:]]
        return "Earlier trajectory context:\n" + "\n".join(f"- {line}" for line in lines)

    def _estimate_tokens(self, value: Any) -> int:
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        return max(1, len(text) // 4)

    def _infer_reward(self, future_events: List[Dict[str, Any]]) -> float:
        for event in future_events:
            if event.get("event_type") == "tool_result":
                status = event.get("payload", {}).get("status")
                if status == "ok":
                    return 1.0
                if status in FAILURE_STATUSES:
                    return -1.0
        return 0.0
