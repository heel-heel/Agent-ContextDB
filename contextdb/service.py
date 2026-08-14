from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Branch, ContextEvent, Snapshot, Trajectory, View, new_id, to_dict, utc_now
from .store import SQLiteStore
from . import uris
from .llm_judge import SemanticRepairJudge
from .vector_index import SQLiteVectorIndex
from .contextql import ContextQLExecutor


FAILURE_STATUSES = {"failed", "error", "timeout"}


class ContextDB:
    def __init__(self, root: str | Path = "data"):
        self.store = SQLiteStore(root)
        self.vector_index = SQLiteVectorIndex(self.store.db_path)

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
        elif view_name == "semantic_repair_judgments":
            content = self.semantic_repair_judgments(trajectory_id)
        elif view_name == "learned_skills":
            content = self.learned_skills(trajectory_id)
        elif view_name == "skill_library":
            content = self.skill_library(trajectory_id)
        elif view_name == "skill_application_trace":
            content = self.skill_application_trace(trajectory_id)
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

    def query_sql(self, trajectory_id: str, sql: str, branch_id: str = "main") -> Dict[str, Any]:
        """Execute one read-only ContextQL statement over logical trajectory relations."""
        return ContextQLExecutor(self).execute(trajectory_id, sql, branch_id)

    def translate_natural_language_sql(self, trajectory_id: str, question: str, branch_id: str = "main", provider: str = "qwen", model: Optional[str] = None) -> Dict[str, Any]:
        """Translate one question into ContextQL SQL without executing it."""
        if not str(question or "").strip():
            raise ValueError("natural-language question is empty")
        # Validate the trajectory and branch before asking the LLM for a scoped query.
        self.get_trajectory(trajectory_id)
        self.get_branch(trajectory_id, branch_id)
        relations = [
            {"name": "trajectories", "columns": ["trajectory_id", "title", "agent_id", "source_id", "default_branch", "head_event_id", "metadata_json"]},
            {"name": "events", "columns": ["event_id", "trajectory_id", "branch_id", "event_type", "actor", "timestamp", "status", "tool_name", "command", "preview", "error_signature", "text", "payload_json", "refs_json", "metadata_json"]},
            {"name": "event_edges", "columns": ["trajectory_id", "parent_event_id", "child_event_id"]},
            {"name": "branches", "columns": ["trajectory_id", "branch_id", "base_event_id", "head_event_id", "snapshot_id", "metadata_json"]},
            {"name": "snapshots", "columns": ["snapshot_id", "trajectory_id", "branch_id", "event_id", "message", "created_at"]},
            {"name": "skills", "columns": ["skill_id", "trajectory_id", "name", "status", "trigger_json", "confidence_json", "highlight_event_ids"]},
            {"name": "skill_evidence", "columns": ["skill_id", "trajectory_id", "failure_event_id", "success_event_id", "action_id", "primary_rule", "judgment_label"]},
            {"name": "failure_patterns", "columns": ["trajectory_id", "failure_event_id", "branch_id", "failed_tool", "failed_command", "likely_cause", "error_signature", "normalized_signature", "preceding_action", "source_event_ids", "highlight_event_ids"]},
            {"name": "repair_strategies", "columns": ["trajectory_id", "failure_event_id", "success_event_id", "failure_branch", "repair_branch", "tool", "command", "strategy", "outcome", "primary_rule", "why_linked", "evidence_json", "highlight_event_ids"]},
            {"name": "learned_skills", "columns": ["skill_id", "trajectory_id", "name", "status", "trigger_json", "recommended_actions_json", "confidence_json", "highlight_event_ids"]},
            {"name": "skill_application_trace", "columns": ["trajectory_id", "event_id", "branch_id", "event_type", "actor", "timestamp", "payload_json", "refs_json"]},
        ]
        translation = SemanticRepairJudge(provider=provider, model=model).translate_contextql(question, relations)
        sql = str(translation.get("sql") or "").strip()
        if not translation.get("enabled") or not sql:
            raise ValueError(translation.get("error") or translation.get("reason") or "LLM did not return SQL")
        # Apply the same read-only grammar gate now, while deferring execution to /api/v1/sql.
        validated_sql = ContextQLExecutor(self)._validate(sql)
        translation["sql"] = validated_sql
        return {"schema_version": "contextql_nl_translation.v1", "trajectory_id": trajectory_id, "branch_id": branch_id, "question": question, "translation": translation}

    def natural_language_query(self, trajectory_id: str, question: str, branch_id: str = "main", provider: str = "qwen", model: Optional[str] = None) -> Dict[str, Any]:
        """Compatibility helper: translate then execute a natural-language ContextQL request."""
        translated = self.translate_natural_language_sql(trajectory_id, question, branch_id, provider, model)
        result = self.query_sql(trajectory_id, translated["translation"]["sql"], branch_id)
        return {"schema_version": "contextql_nl_result.v1", "question": question, "translation": translated["translation"], "result": result}

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
        # no usage now.
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
            normalized_signature = self._normalize_signature(signature)
            cause_judgment = self._llm_likely_cause(
                trajectory_id,
                event,
                {
                    "failed_tool": self._tool_name(tool_call),
                    "failed_command": self._command_text(tool_call),
                    "error_signature": signature,
                    "normalized_signature": normalized_signature,
                    "preceding_action": self._payload_text(assistant),
                },
            )
            pattern = {
                "schema_version": "experience_mining.v2",
                "pattern_id": f"fp_{event['event_id'].split('_')[-1]}",
                "failure_event_id": event["event_id"],
                "branch_id": event.get("branch_id"),
                "failed_tool": self._tool_name(tool_call),
                "failed_command": self._command_text(tool_call),
                "error_signature": signature,
                "normalized_signature": normalized_signature,
                "preceding_action": self._payload_text(assistant),
                "likely_cause": cause_judgment.get("likely_cause"),
                "likely_cause_judgment": cause_judgment,
                "repair_status": "not_evaluated",
                "repair_events_after_failure": repair_events,
                "source_event_ids": self._compact_ids([assistant, tool_call, event]),
                "evidence_event_ids": self._compact_ids([assistant, tool_call, event]),
                "highlight_event_ids": self._compact_ids([event]),
            }
            patterns.append(pattern)
        return patterns

    def success_patterns(self, trajectory_id: str) -> List[Dict[str, Any]]:
        events_by_branch = self._events_by_branch(trajectory_id)
        patterns = []
        for branch_id, events in events_by_branch.items():
            for i, event in enumerate(events):
                if event.get("branch_id") != branch_id:
                    continue
                if not self._is_success_tool_result(event):
                    continue
                tool_call = self._previous_event(events, i, "tool_call")
                assistant = self._previous_event(events, i, "assistant_message")
                is_first = self._is_first_success_on_branch(trajectory_id, branch_id, event["event_id"])
                patterns.append({
                    "schema_version": "experience_mining.v1",
                    "pattern_id": f"sp_{event['event_id'].split('_')[-1]}",
                    "success_event_id": event["event_id"],
                    "branch_id": branch_id,
                    "successful_tool": self._tool_name(tool_call),
                    "successful_command": self._command_text(tool_call),
                    "strategy": self._payload_text(assistant),
                    "outcome": event.get("payload", {}).get("status"),
                    "result_preview": event.get("payload", {}).get("preview", ""),
                    "is_first_success_on_branch": is_first,
                    "source_event_ids": self._compact_ids([assistant, tool_call, event]),
                    "evidence_event_ids": self._compact_ids([assistant, tool_call, event]),
                    "highlight_event_ids": self._compact_ids([event]),
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
        seen_failures = set()
        for failure in failures:
            if failure["failure_event_id"] in seen_failures:
                continue
            seen_failures.add(failure["failure_event_id"])
            failure_event = events.get(failure["failure_event_id"])
            if not failure_event:
                continue
            repair_candidates = []
            excluded_successes = []
            for success in successes:
                success_event = events.get(success["success_event_id"])
                if not success_event:
                    continue
                evidence = self._repair_link_evidence(failure_event, success_event, branches, snapshots, trajectory_id)
                primary_evidence = self._primary_repair_evidence(evidence)
                key = (failure["failure_event_id"], success["success_event_id"])
                if primary_evidence and key not in seen:
                    seen.add(key)
                    candidate = {
                        "schema_version": "experience_mining.v1",
                        "branch_id": success["branch_id"],
                        "success_event_id": success["success_event_id"],
                        "strategy": success.get("strategy"),
                        "tool": success.get("successful_tool"),
                        "command": success.get("successful_command"),
                        "outcome": success.get("outcome"),
                        "primary_rule": primary_evidence.get("rule"),
                        "link_type": primary_evidence.get("rule"),
                        "strength": primary_evidence.get("strength"),
                        "why_linked": self._why_linked(primary_evidence),
                        "evidence": evidence,
                        "evidence_event_ids": self._highlight_event_ids(failure_event, success_event, evidence),
                        "highlight_event_ids": self._compact_ids([failure_event, success_event]),
                        "highlight_branch_ids": [],
                    }
                    repair_candidates.append(candidate)
                    continue
                exclusion = self._excluded_success(failure_event, success_event, success, evidence, trajectory_id)
                if exclusion:
                    excluded_successes.append(exclusion)
            failure_summary = {
                "failure_event_id": failure["failure_event_id"],
                "branch_id": failure["branch_id"],
                "command": failure.get("failed_command"),
                "tool": failure.get("failed_tool"),
                "error_signature": failure.get("error_signature"),
                "normalized_signature": failure.get("normalized_signature"),
                "preceding_action": failure.get("preceding_action"),
                "likely_cause": failure.get("likely_cause"),
                "evidence_event_ids": failure.get("evidence_event_ids", failure.get("source_event_ids", [])),
                "highlight_event_ids": failure.get("highlight_event_ids", []),
            }
            status = "resolved_candidate" if repair_candidates else "unresolved"
            strategies.append({
                "schema_version": "experience_mining.v1",
                "failure": failure_summary,
                "repair_status": status,
                "candidate_count": len(repair_candidates),
                "repair_candidates": repair_candidates,
                "repairs": repair_candidates,
                "excluded_successes": excluded_successes[:20],
                "highlight_event_ids": self._dedupe_ids(
                    failure_summary.get("highlight_event_ids", [])
                    + [event_id for candidate in repair_candidates for event_id in candidate.get("highlight_event_ids", [])]
                ),
                "highlight_branch_ids": self._dedupe_ids(
                    [failure.get("branch_id")]
                    + [branch_id for candidate in repair_candidates for branch_id in candidate.get("highlight_branch_ids", [])]
                ),
                "semantic_judge": {"enabled": False, "label": "not_evaluated"},
            })
        return strategies

    def semantic_repair_judgments(self, trajectory_id: str) -> List[Dict[str, Any]]:
        judge = SemanticRepairJudge()
        rows = []
        for group in self.repair_strategies(trajectory_id):
            failure = group.get("failure", {})
            for candidate in group.get("repair_candidates", group.get("repairs", [])):
                judgment = judge.judge(failure, candidate)
                rows.append({
                    "schema_version": "semantic_repair_judgment.v1",
                    "trajectory_id": trajectory_id,
                    "failure_event_id": failure.get("failure_event_id"),
                    "success_event_id": candidate.get("success_event_id"),
                    "failure": failure,
                    "repair_candidate": candidate,
                    "structural_rule": candidate.get("primary_rule") or candidate.get("link_type"),
                    "judgment": judgment,
                    "highlight_event_ids": self._dedupe_ids([failure.get("failure_event_id"), candidate.get("success_event_id")]),
                })
        return rows

    def learned_skills(self, trajectory_id: str) -> List[Dict[str, Any]]:
        # no usage now.
        judgments = self.semantic_repair_judgments(trajectory_id)
        recommended_rows = []
        failures_by_id: Dict[str, Dict[str, Any]] = {}
        for row in judgments:
            failure = row.get("failure", {})
            judgment = row.get("judgment", {})
            if not self._judgment_recommends_skill(judgment):
                continue
            failure_id = failure.get("failure_event_id")
            if failure_id:
                failures_by_id[failure_id] = failure
            recommended_rows.append(row)

        judge = SemanticRepairJudge()
        failure_grouping = judge.group_failure_patterns(list(failures_by_id.values()))
        failure_groups = list(failure_grouping.get("groups", []))
        assigned_failure_ids = {fid for group in failure_groups for fid in group.get("failure_event_ids", [])}
        for failure_id, failure in failures_by_id.items():
            if failure_id not in assigned_failure_ids:
                failure_groups.append(self._single_failure_group(failure))

        grouped: Dict[str, Dict[str, Any]] = {}
        rows_by_failure: Dict[str, List[Dict[str, Any]]] = {}
        for row in recommended_rows:
            rows_by_failure.setdefault(row.get("failure_event_id"), []).append(row)

        for group in failure_groups:
            failure_ids = [fid for fid in group.get("failure_event_ids", []) if fid in failures_by_id]
            if not failure_ids:
                continue
            first_failure = failures_by_id[failure_ids[0]]
            key = group.get("skill_id") or self._skill_key(first_failure)
            skill = grouped.setdefault(key, {
                "schema_version": "learned_skill.v1",
                "skill_id": key,
                "name": group.get("name") or self._skill_name(first_failure),
                "trigger": {
                    "failed_tool": first_failure.get("tool"),
                    "failed_command_pattern": first_failure.get("command"),
                    "normalized_signature": first_failure.get("normalized_signature"),
                    "error_signature": first_failure.get("error_signature"),
                    "likely_cause": first_failure.get("likely_cause"),
                },
                "failure_grouping": {
                    "enabled": failure_grouping.get("enabled"),
                    "provider": failure_grouping.get("provider"),
                    "model": failure_grouping.get("model"),
                    "confidence": group.get("confidence"),
                    "reason": group.get("reason") or failure_grouping.get("reason"),
                    "failure_event_ids": failure_ids,
                },
                "recommended_actions": [],
                "avoid_actions": [],
                "evidence_refs": [],
                "confidence": {"support_count": 0, "max_llm_confidence": 0.0, "evidence_levels": []},
                "status": "candidate",
                "highlight_event_ids": [],
                "action_grouping": {},
                "_support_events_pending": [],
            })
            for failure_id in failure_ids:
                failure = failures_by_id[failure_id]
                for row in rows_by_failure.get(failure_id, []):
                    candidate = row.get("repair_candidate", {})
                    judgment = row.get("judgment", {})
                    action = {
                        "trajectory_id": trajectory_id,
                        "failure_event_id": row.get("failure_event_id"),
                        "success_event_id": row.get("success_event_id"),
                        "branch_id": candidate.get("branch_id"),
                        "tool": candidate.get("tool"),
                        "command": candidate.get("command"),
                        "strategy": candidate.get("strategy"),
                        "outcome": candidate.get("outcome"),
                        "primary_rule": candidate.get("primary_rule") or candidate.get("link_type"),
                        "judgment_label": judgment.get("label"),
                        "confidence": judgment.get("confidence", 0.0),
                        "reason": judgment.get("reason"),
                    }
                    skill["_support_events_pending"].append(action)
                    skill["evidence_refs"].append({
                        "trajectory_id": trajectory_id,
                        "failure_event_id": row.get("failure_event_id"),
                        "success_event_id": row.get("success_event_id"),
                        "primary_rule": candidate.get("primary_rule") or candidate.get("link_type"),
                        "judgment_label": judgment.get("label"),
                    })
                    skill["confidence"]["support_count"] += 1
                    skill["confidence"]["max_llm_confidence"] = max(skill["confidence"].get("max_llm_confidence", 0.0), float(judgment.get("confidence", 0.0) or 0.0))
                    level = candidate.get("strength") or "unknown"
                    if level not in skill["confidence"]["evidence_levels"]:
                        skill["confidence"]["evidence_levels"].append(level)
                    skill["highlight_event_ids"] = self._dedupe_ids(skill["highlight_event_ids"] + row.get("highlight_event_ids", []))
                for excluded in self._excluded_for_failure(trajectory_id, failure.get("failure_event_id")):
                    avoid = {"command": excluded.get("command"), "tool": excluded.get("tool"), "reason": excluded.get("reason")}
                    if avoid.get("command") and not self._has_action(skill["avoid_actions"], avoid):
                        skill["avoid_actions"].append(avoid)

        for skill in grouped.values():
            support_events = skill.pop("_support_events_pending", [])
            grouping = judge.group_recommended_actions(skill.get("trigger", {}), support_events)
            skill["action_grouping"] = {
                "enabled": grouping.get("enabled"),
                "provider": grouping.get("provider"),
                "model": grouping.get("model"),
                "reason": grouping.get("reason"),
                "error": grouping.get("error"),
                "unassigned_support_event_ids": grouping.get("unassigned_support_event_ids", []),
            }
            skill["recommended_actions"] = self._materialize_grouped_actions(grouping, support_events)
        return list(grouped.values())

    def skill_library(self, trajectory_id: str) -> Dict[str, Any]:
        skills, cached_views = self._cached_learned_skills(trajectory_id)
        return {
            "schema_version": "skill_library.v1",
            "trajectory_id": trajectory_id,
            "materialized": bool(cached_views),
            "skill_count": len(skills),
            "skills": skills,
            "status_counts": self._skill_status_counts(skills),
            "source_views": cached_views,
        }

    def match_skill(self, trajectory_id: str, failure: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        # no usage now.
        skills, cached_views = self._cached_learned_skills(trajectory_id)
        normalized_query = self._normalize_signature(str(failure.get("error_signature") or failure.get("normalized_signature") or ""))
        query_command = str(failure.get("command") or "")
        query_tool = str(failure.get("tool") or "")
        judge = SemanticRepairJudge()
        llm_match = judge.match_skill({"tool": query_tool, "command": query_command, "error_signature": failure.get("error_signature"), "normalized_signature": normalized_query}, skills, top_k=top_k)
        skills_by_id = {skill.get("skill_id"): skill for skill in skills}
        rows = []
        for item in llm_match.get("matches", []):
            skill = skills_by_id.get(item.get("skill_id"))
            if not skill:
                continue
            rows.append({
                "schema_version": "skill_match.v1",
                "matched_skill_id": skill.get("skill_id"),
                "skill_name": skill.get("name"),
                "score": item.get("score", 0.0),
                "match_reason": item.get("reason", ""),
                "matched_trigger": skill.get("trigger", {}),
                "recommended_actions": skill.get("recommended_actions", []),
                "skill_status": skill.get("status", "candidate"),
                "highlight_event_ids": skill.get("highlight_event_ids", []),
            })
        rows.sort(key=lambda item: item.get("score", 0), reverse=True)
        return {
            "schema_version": "skill_match_result.v1",
            "trajectory_id": trajectory_id,
            "input_failure": {
                "tool": query_tool,
                "command": query_command,
                "error_signature": failure.get("error_signature"),
                "normalized_signature": normalized_query,
            },
            "match_judge": {
                "enabled": llm_match.get("enabled"),
                "provider": llm_match.get("provider"),
                "model": llm_match.get("model"),
                "reason": llm_match.get("reason"),
                "error": llm_match.get("error"),
            },
            "matches": rows[:top_k],
            "materialized": bool(cached_views),
            "source_views": cached_views,
        }

    def apply_skill(self, trajectory_id: str, failure: Dict[str, Any], branch_id: str = "skill-application", top_k: int = 1, outcome_status: str = "ok") -> Dict[str, Any]:
        if self.store.get_object(uris.branch_key(trajectory_id, branch_id)) is None:
            self.create_branch(trajectory_id, branch_id, from_branch="main")
        match = self.match_skill(trajectory_id, failure, top_k=top_k)
        match_event = self.append_event(
            trajectory_id,
            "skill_match",
            match,
            branch_id=branch_id,
            actor="contextdb",
            metadata={"operation": "skill_retrieval", "schema_version": "skill_match_result.v1"},
        )
        selected = match.get("matches", [None])[0] if match.get("matches") else None
        selected_action = None
        selected_action_score = 0.0
        selected_action_index = None
        selected_action_policy = "highest max_llm_confidence; tie keeps original order"
        if selected:
            actions = selected.get("recommended_actions", [])
            selected_action, selected_action_score, selected_action_index = self._select_recommended_action(actions)
        if selected_action:
            selection_refs = {
                "skill_id": selected.get("matched_skill_id"),
                "skill_match_event_id": match_event.get("event_id"),
                "selected_action_id": selected_action.get("action_id"),
                "selected_action_index": selected_action_index,
                "selected_action_score": selected_action_score,
                "selected_action_policy": selected_action_policy,
            }
            self.append_event(
                trajectory_id,
                "assistant_message",
                {"text": "Retrieved skill %s and selected action by %s: %s (score %.3f)" % (selected.get("matched_skill_id"), selected_action_policy, selected_action.get("name") or selected_action.get("strategy"), selected_action_score)},
                branch_id=branch_id,
                actor="agent",
                refs=selection_refs,
            )
            tool_call = self.append_event(
                trajectory_id,
                "tool_call",
                {"tool_name": selected_action.get("canonical_tool") or selected_action.get("tool"), "command": selected_action.get("canonical_command_template") or selected_action.get("command_template")},
                branch_id=branch_id,
                actor="agent",
                refs=selection_refs,
            )
            result = self.append_event(
                trajectory_id,
                "tool_result",
                {"status": outcome_status, "preview": "Skill-guided action succeeded: %s" % (selected_action.get("name") or selected_action.get("canonical_command_template"))},
                branch_id=branch_id,
                actor="tool",
                refs={**selection_refs, "tool_call_event_id": tool_call.get("event_id")},
            )
        else:
            result = self.append_event(
                trajectory_id,
                "tool_result",
                {"status": "failed", "preview": "No matching skill was found."},
                branch_id=branch_id,
                actor="tool",
                refs={"skill_match_event_id": match_event.get("event_id")},
            )
        return {"trajectory_id": trajectory_id, "branch_id": branch_id, "match": match, "match_event_id": match_event.get("event_id"), "result_event_id": result.get("event_id")}

    def _select_recommended_action(self, actions: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float, Optional[int]]:
        best_action = None
        best_score = -1.0
        best_index: Optional[int] = None
        for index, action in enumerate(actions):
            score = self._recommended_action_max_score(action)
            if score > best_score:
                best_action = action
                best_score = score
                best_index = index
        return best_action, max(best_score, 0.0), best_index

    def _recommended_action_max_score(self, action: Dict[str, Any]) -> float:
        confidence = action.get("confidence", {}) or {}
        try:
            return float(confidence.get("max_llm_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def skill_application_trace(self, trajectory_id: str) -> List[Dict[str, Any]]:
        rows = []
        for event in self._all_events(trajectory_id):
            refs = event.get("refs", {}) or {}
            metadata = event.get("metadata", {}) or {}
            if event.get("event_type") in {"skill_match", "tool_call", "tool_result", "assistant_message"} and (
                metadata.get("operation") == "skill_retrieval"
                or refs.get("skill_match_event_id")
                or refs.get("skill_id")
            ):
                rows.append(event)
        return rows

    def _cached_learned_skills(self, trajectory_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        skills_by_id: Dict[str, Dict[str, Any]] = {}
        cached_views = []
        prefix = f"trajectories/{trajectory_id}/views"
        for key in self.store.list_objects(prefix):
            if not key.endswith("/learned_skills"):
                continue
            view = self.store.get_object(key)
            if not view:
                continue
            cached_views.append({
                "key": key,
                "branch_id": view.get("branch_id"),
                "created_at": view.get("created_at"),
                "skill_count": len(view.get("content", []) if isinstance(view.get("content"), list) else []),
            })
            content = view.get("content", [])
            if not isinstance(content, list):
                continue
            for skill in content:
                if isinstance(skill, dict) and skill.get("skill_id"):
                    skills_by_id[skill["skill_id"]] = skill
        cached_views.sort(key=lambda item: (item.get("created_at") or "", item.get("key") or ""))
        return list(skills_by_id.values()), cached_views

    def _skill_status_counts(self, skills: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for skill in skills:
            status = skill.get("status", "candidate")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _judgment_recommends_skill(self, judgment: Dict[str, Any]) -> bool:
        # A missing/failed LLM judgment is not evidence for a reusable skill.
        return bool(judgment.get("enabled")) and not judgment.get("error") and (
            judgment.get("label") in {"likely_repair", "partial_repair"}
            and bool(judgment.get("recommended_for_skill"))
        )

    def _skill_key(self, failure: Dict[str, Any]) -> str:
        base = " ".join(str(x or "") for x in [failure.get("likely_cause"), failure.get("normalized_signature"), failure.get("command")]).lower()
        tokens = [part.strip("-_") for part in base.replace("/", " ").replace("=", " ").split() if len(part.strip("-_")) > 2]
        return "skill_" + "_".join(tokens[:8] or ["agent_repair"])

    def _skill_name(self, failure: Dict[str, Any]) -> str:
        cause = failure.get("likely_cause") or "agent task failure"
        return "Repair " + str(cause)

    def _single_failure_group(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "skill_id": self._skill_key(failure),
            "name": self._skill_name(failure),
            "failure_event_ids": [failure.get("failure_event_id")],
            "confidence": 0.0,
            "reason": "Fallback single-failure group because LLM failure grouping did not assign this failure.",
        }

    def _materialize_grouped_actions(self, grouping: Dict[str, Any], support_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # no usage now.
        by_id = {event.get("success_event_id"): event for event in support_events if event.get("success_event_id")}
        actions = []
        for item in grouping.get("actions", []):
            support_ids = item.get("support_event_ids", [])
            events = [by_id[event_id] for event_id in support_ids if event_id in by_id]
            if not events:
                continue
            variants = []
            max_confidence = 0.0
            labels = []
            rules = []
            for event in events:
                command = event.get("command")
                if command and command not in variants:
                    variants.append(command)
                try:
                    max_confidence = max(max_confidence, float(event.get("confidence", 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass
                for field, target in [("judgment_label", labels), ("primary_rule", rules)]:
                    value = event.get(field)
                    if value and value not in target:
                        target.append(value)
            group_confidence = item.get("confidence", 0.0)
            try:
                group_confidence = float(group_confidence or 0.0)
            except (TypeError, ValueError):
                group_confidence = 0.0
            actions.append({
                "schema_version": "recommended_action.v2",
                "action_id": "act_" + str(item.get("action_id") or f"group_{len(actions) + 1}"),
                "name": item.get("name"),
                "tool": item.get("canonical_tool") or events[0].get("tool"),
                "canonical_tool": item.get("canonical_tool") or events[0].get("tool"),
                "command_template": item.get("canonical_command_template"),
                "canonical_command_template": item.get("canonical_command_template"),
                "strategy": item.get("strategy"),
                "dedupe_method": "llm:semantic_action_grouping",
                "variants": variants,
                "support_events": events,
                "judgment_label": ",".join(labels),
                "reason": item.get("reason"),
                "confidence": {
                    "support_count": len(events),
                    "max_llm_confidence": max_confidence,
                    "grouping_confidence": group_confidence,
                    "judgment_labels": labels,
                    "evidence_rules": rules,
                },
            })
        assigned = {event_id for action in actions for event_id in [event.get("success_event_id") for event in action.get("support_events", [])]}
        for event in support_events:
            if event.get("success_event_id") in assigned:
                continue
            actions.append(self._event_level_action(event))
        return actions

    def _event_level_action(self, event: Dict[str, Any]) -> Dict[str, Any]:
        # no usage now.
        return {
            "schema_version": "recommended_action.v2",
            "action_id": "act_unassigned_" + str(event.get("success_event_id") or len(str(event))),
            "name": "Unassigned successful action",
            "tool": event.get("tool"),
            "canonical_tool": event.get("tool"),
            "command_template": event.get("command"),
            "canonical_command_template": event.get("command"),
            "strategy": event.get("strategy"),
            "dedupe_method": "llm:unassigned_support_event",
            "variants": [event.get("command")] if event.get("command") else [],
            "support_events": [event],
            "judgment_label": event.get("judgment_label"),
            "reason": event.get("reason"),
            "confidence": {
                "support_count": 1,
                "max_llm_confidence": event.get("confidence", 0.0),
                "grouping_confidence": 0.0,
                "judgment_labels": [event.get("judgment_label")] if event.get("judgment_label") else [],
                "evidence_rules": [event.get("primary_rule")] if event.get("primary_rule") else [],
            },
        }

    def _merge_recommended_action(self, actions: List[Dict[str, Any]], support_event: Dict[str, Any]) -> None:
        # no usage now.
        key = self._action_dedupe_key(support_event)
        existing = next((action for action in actions if action.get("action_key") == key), None)
        if not existing:
            command_template = self._canonical_command_template(support_event.get("command"))
            existing = {
                "schema_version": "recommended_action.v1",
                "action_id": "act_" + key,
                "action_key": key,
                "tool": support_event.get("tool"),
                "canonical_tool": support_event.get("tool"),
                "command_template": command_template,
                "canonical_command_template": command_template,
                "strategy": support_event.get("strategy"),
                "dedupe_method": "rule:tool+canonical_command",
                "variants": [],
                "support_events": [],
                "confidence": {
                    "support_count": 0,
                    "max_llm_confidence": 0.0,
                    "judgment_labels": [],
                    "evidence_rules": [],
                },
            }
            actions.append(existing)
        self._append_variant(existing, support_event.get("command"))
        if not self._has_support_event(existing["support_events"], support_event):
            existing["support_events"].append(support_event)
            existing["confidence"]["support_count"] += 1
        try:
            confidence = float(support_event.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        existing["confidence"]["max_llm_confidence"] = max(existing["confidence"].get("max_llm_confidence", 0.0), confidence)
        for field, target in [("judgment_label", "judgment_labels"), ("primary_rule", "evidence_rules")]:
            value = support_event.get(field)
            if value and value not in existing["confidence"][target]:
                existing["confidence"][target].append(value)
        if confidence >= float(existing.get("confidence_value_for_strategy", -1.0)):
            existing["strategy"] = support_event.get("strategy") or existing.get("strategy")
            existing["judgment_label"] = support_event.get("judgment_label")
            existing["reason"] = support_event.get("reason")
            existing["outcome"] = support_event.get("outcome")
            existing["primary_rule"] = support_event.get("primary_rule")
            existing["confidence_value_for_strategy"] = confidence

    def _action_dedupe_key(self, action: Dict[str, Any]) -> str:
        # no usage now.
        tool = str(action.get("tool") or "unknown_tool").lower()
        command = self._canonical_command_template(action.get("command"))
        raw = f"{tool} {command}".lower()
        tokens = [part.strip("-_") for part in re.split(r"[^a-z0-9_+-]+", raw) if len(part.strip("-_")) > 1]
        return "_".join(tokens[:10] or ["agent_action"])

    def _canonical_command_template(self, command: Any) -> str:
        # no usage now.
        if command is None:
            return ""
        if not isinstance(command, str):
            return json.dumps(command, ensure_ascii=False, sort_keys=True)
        parts = command.split()
        if not parts:
            return ""
        canonical: List[str] = []
        for part in parts:
            if part in {"--locked", "--frozen", "--offline"}:
                continue
            canonical.append(part)
        return " ".join(canonical)

    def _append_variant(self, action: Dict[str, Any], command: Any) -> None:
        # no usage now.
        if command is None:
            return
        variant = command if isinstance(command, str) else json.dumps(command, ensure_ascii=False, sort_keys=True)
        if variant not in action["variants"]:
            action["variants"].append(variant)

    def _has_support_event(self, support_events: List[Dict[str, Any]], support_event: Dict[str, Any]) -> bool:
        success_event_id = support_event.get("success_event_id")
        if success_event_id:
            return any(item.get("success_event_id") == success_event_id for item in support_events)
        return any(
            item.get("tool") == support_event.get("tool")
            and item.get("command") == support_event.get("command")
            and item.get("branch_id") == support_event.get("branch_id")
            for item in support_events
        )

    def _has_action(self, actions: List[Dict[str, Any]], action: Dict[str, Any]) -> bool:
        action_command = action.get("command_template") or action.get("command")
        action_tool = action.get("tool")
        return any(
            existing.get("tool") == action_tool
            and (existing.get("command_template") or existing.get("command")) == action_command
            for existing in actions
        )

    def _excluded_for_failure(self, trajectory_id: str, failure_event_id: Optional[str]) -> List[Dict[str, Any]]:
        # no usage now.
        if not failure_event_id:
            return []
        for group in self.repair_strategies(trajectory_id):
            if group.get("failure", {}).get("failure_event_id") == failure_event_id:
                return group.get("excluded_successes", [])
        return []

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
                    "evidence_level": "primary_structural",
                    "branch_id": failure_branch,
                    "failure_event_id": failure_id,
                    "success_event_id": success_id,
                })
        success_branch_obj = branches.get(success_branch, {})
        success_is_first_on_branch = self._is_first_success_on_branch(trajectory_id, success_branch, success_id) if success_branch else False
        if success_branch_obj.get("base_event_id") == failure_id and success_is_first_on_branch:
            evidence.append({
                "rule": "branch_from_failure_first_success",
                "strength": "structural",
                "evidence_level": "primary_structural",
                "repair_branch": success_branch,
                "base_event_id": failure_id,
                "success_event_id": success_id,
            })
        failure_ancestors = self._reachable_event_ids(trajectory_id, failure_id)
        base_event_id = success_branch_obj.get("base_event_id")
        if base_event_id and base_event_id in failure_ancestors and base_event_id != failure_id and success_is_first_on_branch:
            evidence.append({
                "rule": "branch_from_failure_ancestor_first_success",
                "strength": "structural",
                "evidence_level": "weak_structural",
                "repair_branch": success_branch,
                "base_event_id": base_event_id,
                "failure_event_id": failure_id,
                "success_event_id": success_id,
            })
        from_branch = success_branch_obj.get("metadata", {}).get("from_branch")
        rollback_branch = branches.get(from_branch or "", {})
        rollback_snapshot_id = rollback_branch.get("metadata", {}).get("rollback_from")
        rollback_snapshot = snapshots.get(rollback_snapshot_id or "")
        if from_branch and rollback_snapshot_id and rollback_snapshot and rollback_snapshot.get("event_id") in failure_ancestors and success_is_first_on_branch:
            evidence.append({
                "rule": "rollback_then_repair_first_success",
                "strength": "structural",
                "evidence_level": "primary_structural",
                "rollback_branch": from_branch,
                "repair_branch": success_branch,
                "snapshot_id": rollback_snapshot_id,
                "snapshot_event_id": rollback_snapshot.get("event_id"),
                "failure_event_id": failure_id,
                "success_event_id": success_id,
            })
        failed_call = self._previous_event(self.list_events(trajectory_id, failure_branch), self._index_in_branch(trajectory_id, failure_branch, failure_id), "tool_call") if failure_branch else None
        success_call = self._previous_event(self.list_events(trajectory_id, success_branch), self._index_in_branch(trajectory_id, success_branch, success_id), "tool_call") if success_branch else None
        if failed_call and success_call and self._tool_name(failed_call) == self._tool_name(success_call) and self._commands_overlap(self._command_text(failed_call), self._command_text(success_call)):
            evidence.append({
                "rule": "same_tool_command_variant",
                "strength": "supporting",
                "evidence_level": "supporting",
                "failed_command": self._command_text(failed_call),
                "successful_command": self._command_text(success_call),
            })
        return evidence

    def _primary_repair_evidence(self, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        priority = {
            "rollback_then_repair_first_success": 0,
            "branch_from_failure_first_success": 1,
            "same_branch_first_success_after_failure": 2,
            "branch_from_failure_ancestor_first_success": 3,
        }
        structural = [item for item in evidence if item.get("strength") == "structural"]
        if not structural:
            return None
        return sorted(structural, key=lambda item: priority.get(item.get("rule", ""), 99))[0]

    def _why_linked(self, evidence: Dict[str, Any]) -> str:
        rule = evidence.get("rule")
        if rule == "same_branch_first_success_after_failure":
            return "The success is the first successful tool result after the failure on the same branch."
        if rule == "branch_from_failure_first_success":
            return "The repair branch was created directly from the failed event, and this is the first success on that branch."
        if rule == "branch_from_failure_ancestor_first_success":
            return "The repair branch was created from an ancestor of the failed event, and this is the first success on that branch."
        if rule == "rollback_then_repair_first_success":
            return "The repair branch comes from a rollback branch whose snapshot is on the failed event ancestry, and this is the first success on the repair branch."
        return "The candidate has deterministic trajectory evidence."

    def _excluded_success(self, failure_event: Dict[str, Any], success_event: Dict[str, Any], success: Dict[str, Any], evidence: List[Dict[str, Any]], trajectory_id: str) -> Optional[Dict[str, Any]]:
        if success_event.get("timestamp", "") <= failure_event.get("timestamp", ""):
            return None
        branch_id = success.get("branch_id")
        success_event_id = success.get("success_event_id")
        is_first = self._is_first_success_on_branch(trajectory_id, branch_id, success_event_id) if branch_id and success_event_id else False
        if any(item.get("strength") == "supporting" for item in evidence):
            reason = "supporting_only"
        elif not is_first:
            reason = "not_first_success_on_branch"
        else:
            reason = "no_structural_evidence"
        if reason == "no_structural_evidence" and not evidence:
            return None
        return {
            "success_event_id": success_event_id,
            "branch_id": branch_id,
            "command": success.get("successful_command"),
            "tool": success.get("successful_tool"),
            "outcome": success.get("outcome"),
            "reason": reason,
            "evidence": evidence,
        }

    def _highlight_event_ids(self, failure_event: Dict[str, Any], success_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> List[str]:
        ids = [failure_event.get("event_id"), success_event.get("event_id")]
        for item in evidence:
            ids.extend([
                item.get("failure_event_id"),
                item.get("success_event_id"),
                item.get("base_event_id"),
                item.get("snapshot_event_id"),
            ])
        return self._dedupe_ids(ids)

    def _highlight_branch_ids(self, failure_event: Dict[str, Any], success_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> List[str]:
        ids = [failure_event.get("branch_id"), success_event.get("branch_id")]
        for item in evidence:
            ids.extend([item.get("branch_id"), item.get("rollback_branch"), item.get("repair_branch")])
        return self._dedupe_ids(ids)

    def _compact_ids(self, events: List[Optional[Dict[str, Any]]]) -> List[str]:
        return self._dedupe_ids([event.get("event_id") for event in events if event])

    def _dedupe_ids(self, values: List[Optional[str]]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _normalize_signature(self, signature: str) -> str:
        lowered = signature.lower()
        cleaned = []
        for token in lowered.replace("/", " ").replace("\\", " ").replace(":", " ").split():
            if any(ch.isdigit() for ch in token) and not token.startswith("gcc"):
                continue
            cleaned.append(token.strip(".,;()[]{}"))
        return " ".join(cleaned[:24])

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

    def _llm_likely_cause(self, trajectory_id: str, event: Dict[str, Any], failure: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a failure cause and cache only a successful result for this LLM configuration."""
        metadata = event.setdefault("metadata", {})
        cached = metadata.get("likely_cause_judgment")
        fingerprint = {
            "failed_tool": failure.get("failed_tool"),
            "failed_command": failure.get("failed_command"),
            "error_signature": failure.get("error_signature"),
            "preceding_action": failure.get("preceding_action"),
        }
        judge = SemanticRepairJudge()
        if self._is_reusable_likely_cause_cache(cached, fingerprint, judge):
            return cached
        result = judge.classify_likely_cause(fingerprint)
        result["input"] = fingerprint
        # Keep diagnostics, but allow a later configured/healthy LLM to retry unavailable or failed calls.
        metadata["likely_cause_judgment"] = result
        self.store.put_object(uris.event_key(trajectory_id, event["event_id"]), event)
        return result

    @staticmethod
    def _is_reusable_likely_cause_cache(cached: Any, fingerprint: Dict[str, Any], judge: SemanticRepairJudge) -> bool:
        if not isinstance(cached, dict) or cached.get("input") != fingerprint:
            return False
        if cached.get("error") or not cached.get("enabled"):
            return False
        if cached.get("provider") != judge.provider or cached.get("model") != judge.model:
            return False
        cause = str(cached.get("likely_cause") or "").strip().lower()
        return bool(cause) and cause != "llm cause classification unavailable"

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



def _vector_text(parts):
    return " ".join(str(part or "") for part in parts if part)


def _vector_cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def _vector_cluster(index, items, threshold):
    groups = []
    for item in items:
        vector = index.embedder.embed(item["vector_text"])
        best = max(groups, key=lambda group: _vector_cosine(vector, group["centroid"]), default=None)
        score = _vector_cosine(vector, best["centroid"]) if best else -1.0
        if best and score >= threshold:
            best["items"].append(item)
            count = len(best["items"])
            center = [(value * (count - 1) + current) / count for value, current in zip(best["centroid"], vector)]
            norm = sum(value * value for value in center) ** .5
            best["centroid"] = [value / norm for value in center] if norm else center
        else:
            groups.append({"items": [item], "centroid": vector})
    return [group["items"] for group in groups]


def _vector_learned_skills(self, trajectory_id):
    judgments = self.semantic_repair_judgments(trajectory_id)
    failures, support = {}, {}
    for row in judgments:
        failure, judgment = row.get("failure", {}), row.get("judgment", {})
        fid = failure.get("failure_event_id")
        if not fid or not self._judgment_recommends_skill(judgment):
            continue
        failures[fid] = failure
        candidate = row.get("repair_candidate", {})
        support.setdefault(fid, []).append({
            "trajectory_id": trajectory_id, "failure_event_id": fid,
            "success_event_id": row.get("success_event_id"), "branch_id": candidate.get("branch_id"),
            "tool": candidate.get("tool"), "command": candidate.get("command"),
            "strategy": candidate.get("strategy"), "outcome": candidate.get("outcome"),
            "primary_rule": candidate.get("primary_rule") or candidate.get("link_type"),
            "judgment_label": judgment.get("label"), "confidence": float(judgment.get("confidence", 0) or 0),
            "reason": judgment.get("reason"),
        })
    failure_rows = [{"failure_id": fid, "failure": failure, "vector_text": _vector_text([failure.get("tool"), failure.get("command"), failure.get("normalized_signature"), failure.get("error_signature"), failure.get("likely_cause")])} for fid, failure in failures.items()]
    skills = []
    for number, group in enumerate(_vector_cluster(self.vector_index, failure_rows, .74), 1):
        first = group[0]["failure"]; ids = [row["failure_id"] for row in group]
        skill_id = self._skill_key(first)
        if any(skill["skill_id"] == skill_id for skill in skills):
            skill_id = f"{skill_id}_{number}"
        action_rows = []
        for fid in ids:
            for event in support.get(fid, []):
                action_rows.append({**event, "vector_text": _vector_text([event.get("tool"), event.get("command"), event.get("strategy")])})
        actions = []
        for action_number, action_group in enumerate(_vector_cluster(self.vector_index, action_rows, .78), 1):
            rep = max(enumerate(action_group), key=lambda item: (item[1].get("confidence", 0), -item[0]))[1]
            support_events = []
            for event in action_group:
                if not self._has_support_event(support_events, event):
                    support_events.append(event)
            scores = [float(event.get("confidence", 0) or 0) for event in support_events]
            name = str(rep.get("strategy") or ("Apply " + str(rep.get("command") or "successful repair")))[:160]
            actions.append({
                "action_id": f"act_{skill_id.split('skill_', 1)[-1]}_{action_number}",
                "name": name, "canonical_tool": rep.get("tool"), "canonical_command_template": rep.get("command"),
                "strategy": rep.get("strategy") or name,
                "support_event_ids": [event.get("success_event_id") for event in support_events if event.get("success_event_id")],
                "support_events": support_events,
                "confidence": {"max_llm_confidence": max(scores) if scores else 0, "support_count": len(support_events)},
                "dedupe": {"method": "vector_cosine", "embedding": "hash-384", "threshold": .78, "group_size": len(support_events)},
            })
        all_support = [event for fid in ids for event in support.get(fid, [])]
        refs = [{"trajectory_id": trajectory_id, "failure_event_id": event.get("failure_event_id"), "success_event_id": event.get("success_event_id"), "primary_rule": event.get("primary_rule"), "judgment_label": event.get("judgment_label")} for event in all_support]
        skill = {
            "schema_version": "learned_skill.v2", "skill_id": skill_id, "name": self._skill_name(first),
            "trigger": {"failed_tool": first.get("tool"), "failed_command_pattern": first.get("command"), "normalized_signature": first.get("normalized_signature"), "error_signature": first.get("error_signature"), "likely_cause": first.get("likely_cause")},
            "failure_grouping": {"method": "vector_cosine", "embedding": "hash-384", "threshold": .74, "failure_event_ids": ids, "group_size": len(ids)},
            "recommended_actions": actions, "avoid_actions": [], "evidence_refs": refs,
            "confidence": {"support_count": len(all_support), "max_llm_confidence": max([float(event.get("confidence", 0) or 0) for event in all_support] or [0]), "evidence_levels": ["structural"]},
            "status": "candidate", "highlight_event_ids": self._dedupe_ids(ids + [event.get("success_event_id") for event in all_support]),
            "action_grouping": {"method": "vector_cosine", "embedding": "hash-384", "threshold": .78, "group_count": len(actions)},
        }
        skills.append(skill)
    entries = []
    for skill in skills:
        document = _vector_text([skill["name"], *skill["trigger"].values(), *[action["name"] for action in skill["recommended_actions"]]])
        entries.append({"entry_id": f"{trajectory_id}:{skill['skill_id']}", "document": document, "metadata": {"skill": skill, "trajectory_id": trajectory_id}})
    self.vector_index.replace_owner("skills", trajectory_id, entries)
    return skills


def _vector_match_skill(self, trajectory_id, failure, top_k=3):
    query = _vector_text([failure.get("tool"), failure.get("command"), failure.get("error_signature"), failure.get("normalized_signature")])
    threshold = .32
    raw = self.vector_index.search("skills", query, top_k=max(top_k * 3, top_k), min_score=threshold)
    rows, seen = [], set()
    for item in raw:
        skill = item.get("metadata", {}).get("skill", {}); sid = skill.get("skill_id")
        if not sid or sid in seen: continue
        seen.add(sid)
        rows.append({"schema_version": "skill_match.v2", "matched_skill_id": sid, "skill_name": skill.get("name"), "score": item["score"], "match_reason": "vector cosine similarity over failure and learned trigger", "matched_trigger": skill.get("trigger", {}), "recommended_actions": skill.get("recommended_actions", []), "skill_status": skill.get("status", "candidate"), "highlight_event_ids": skill.get("highlight_event_ids", []), "source_trajectory_id": item.get("metadata", {}).get("trajectory_id")})
    return {
        "schema_version": "skill_match_result.v2", "trajectory_id": trajectory_id,
        "input_failure": {"tool": failure.get("tool"), "command": failure.get("command"), "error_signature": failure.get("error_signature"), "normalized_signature": self._normalize_signature(str(failure.get("error_signature") or failure.get("normalized_signature") or ""))},
        "match_engine": {"type": "sqlite-vector-index", "embedding": "hash-384", "similarity": "cosine", "threshold": threshold, "indexed_skill_count": self.vector_index.count("skills")},
        "match_judge": {"enabled": False, "provider": "replaced_by_vector_index", "reason": "Skill matching uses vector retrieval."},
        "matches": rows[:top_k], "materialized": bool(self.vector_index.count("skills")), "source_views": self._cached_learned_skills(trajectory_id)[1],
    }


def _retrieve_for_failure(self, trajectory_id, failure, branch_id="main", source_event_id=None):
    match = self.match_skill(trajectory_id, failure, top_k=3)
    event = self.append_event(trajectory_id, "skill_match", match, branch_id=branch_id, actor="contextdb", refs={"failure_event_id": source_event_id} if source_event_id else {}, metadata={"operation": "online_skill_retrieval", "match_engine": "sqlite-vector-index"})
    return {"match_event_id": event.get("event_id"), "match": match}


ContextDB.learned_skills = _vector_learned_skills
ContextDB.match_skill = _vector_match_skill
ContextDB.retrieve_for_failure = _retrieve_for_failure
