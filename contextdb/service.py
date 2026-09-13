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
VERSION_EVENT_OPERATIONS = {
    "version_snapshot_created",
    "version_repair_branch_suggested",
    "version_branch_created",
    "version_rollback_created",
    "version_decision",
}
RECENT_CAUSAL_EVENT_LIMIT = 8
MEMORY_RETRIEVAL_LIMIT = 5
CONTEXT_SEGMENT_ORDER = (
    "current_task",
    "recent_causal_trace",
    "branch_version_state",
    "retrieved_memory",
    "matched_skills",
    "earlier_trajectory_digest",
)


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

    def global_overview(self) -> Dict[str, Any]:
        """Aggregate trajectory metadata, tool calls, and globally indexed skills."""
        trajectories = []
        for key in self.store.list_objects("trajectories"):
            if not key.endswith("/meta"):
                continue
            trajectory = self.store.get_object(key)
            if trajectory and trajectory.get("trajectory_id"):
                trajectories.append(trajectory)
        trajectories.sort(key=lambda item: (item.get("updated_at") or "", item.get("trajectory_id") or ""), reverse=True)

        skills = []
        skills_by_key: Dict[str, Dict[str, Any]] = {}
        skills_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self.vector_index.list_entries("skills"):
            metadata = entry.get("metadata", {}) or {}
            skill = metadata.get("skill", {}) or {}
            skill_id = skill.get("skill_id")
            source_trajectory_id = metadata.get("trajectory_id") or entry.get("owner_id")
            if not skill_id or not source_trajectory_id:
                continue
            key = f"skill:{source_trajectory_id}:{skill_id}"
            row = {
                "node_id": key,
                "skill_id": skill_id,
                "name": skill.get("name") or skill_id,
                "source_trajectory_id": source_trajectory_id,
                "trigger_tool": (skill.get("trigger", {}) or {}).get("failed_tool"),
                "support_count": (skill.get("confidence", {}) or {}).get("support_count", 0),
                "status": skill.get("status", "candidate"),
            }
            skills.append(row)
            skills_by_key[key] = row
            skills_by_id.setdefault(skill_id, []).append(row)

        trajectory_rows = []
        edges: Dict[Tuple[str, str, str], int] = {}
        all_tools: Dict[str, int] = {}
        for trajectory in trajectories:
            trajectory_id = trajectory["trajectory_id"]
            events = self._all_events(trajectory_id)
            match_sources: Dict[str, Tuple[str, str]] = {}
            for event in events:
                if event.get("event_type") != "skill_match":
                    continue
                match = (event.get("payload", {}) or {}).get("matches", [])
                first = match[0] if isinstance(match, list) and match else {}
                skill_id = first.get("matched_skill_id")
                source_id = first.get("source_trajectory_id")
                if skill_id and source_id:
                    match_sources[event.get("event_id", "")] = (source_id, skill_id)

            tools: Dict[str, int] = {}
            associated_skills: Dict[str, int] = {}
            for event in events:
                if event.get("event_type") != "tool_call":
                    continue
                payload = event.get("payload", {}) or {}
                tool_name = str(payload.get("tool_name") or payload.get("tool") or "unknown tool")
                tools[tool_name] = tools.get(tool_name, 0) + 1
                all_tools[tool_name] = all_tools.get(tool_name, 0) + 1
                trajectory_node = f"trajectory:{trajectory_id}"
                tool_node = f"tool:{tool_name}"
                edges[(trajectory_node, tool_node, "calls")] = edges.get((trajectory_node, tool_node, "calls"), 0) + 1

                refs = event.get("refs", {}) or {}
                skill_id = refs.get("skill_id")
                source_id = refs.get("skill_source_trajectory_id")
                if not source_id and refs.get("skill_match_event_id") in match_sources:
                    source_id, matched_skill_id = match_sources[refs["skill_match_event_id"]]
                    skill_id = skill_id or matched_skill_id
                candidates = skills_by_id.get(skill_id, []) if skill_id else []
                if not source_id and len(candidates) == 1:
                    source_id = candidates[0]["source_trajectory_id"]
                skill_node = f"skill:{source_id}:{skill_id}" if source_id and skill_id else ""
                if skill_node not in skills_by_key:
                    continue
                associated_skills[skill_node] = associated_skills.get(skill_node, 0) + 1
                edges[(trajectory_node, skill_node, "uses_skill")] = edges.get((trajectory_node, skill_node, "uses_skill"), 0) + 1
                edges[(tool_node, skill_node, "skill_action")] = edges.get((tool_node, skill_node, "skill_action"), 0) + 1

            trajectory_rows.append({
                **trajectory,
                "event_count": len(events),
                "tool_call_count": sum(tools.values()),
                "tools": [{"tool_name": name, "call_count": count} for name, count in sorted(tools.items())],
                "associated_skills": [
                    {**skills_by_key[key], "application_count": count}
                    for key, count in sorted(associated_skills.items())
                ],
            })

        nodes = (
            [{"id": f"trajectory:{row['trajectory_id']}", "type": "trajectory", "label": row.get("title") or row["trajectory_id"], "trajectory_id": row["trajectory_id"], "detail": row.get("agent_id") or "unknown-agent"} for row in trajectory_rows]
            + [{"id": f"tool:{name}", "type": "tool", "label": name, "detail": f"{count} calls"} for name, count in sorted(all_tools.items())]
            + [{"id": skill["node_id"], "type": "skill", "label": skill["name"], "detail": skill["source_trajectory_id"]} for skill in skills]
        )
        return {
            "schema_version": "global_overview.v1",
            "trajectory_count": len(trajectory_rows),
            "tool_count": len(all_tools),
            "skill_count": len(skills),
            "trajectories": trajectory_rows,
            "skills": skills,
            "graph": {
                "nodes": nodes,
                "edges": [
                    {"source": source, "target": target, "kind": kind, "count": count}
                    for (source, target, kind), count in sorted(edges.items())
                ],
            },
        }

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

    def create_branch(self, trajectory_id: str, branch_id: str, base_event_id: Optional[str] = None, from_branch: str = "main", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.store.get_object(uris.branch_key(trajectory_id, branch_id)) is not None:
            raise ValueError(f"branch already exists: {branch_id}")
        base = base_event_id or self.get_branch(trajectory_id, from_branch).get("head_event_id")
        branch = Branch(branch_id=branch_id, trajectory_id=trajectory_id, base_event_id=base, head_event_id=base, metadata={"from_branch": from_branch, **(metadata or {})})
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

    def query_view(
        self,
        trajectory_id: str,
        view_name: str,
        branch_id: str = "main",
        token_budget: int = 4000,
        profile_id: Optional[str] = None,
        refresh_summary: bool = False,
    ) -> Dict[str, Any]:
        events = self.list_events(trajectory_id, branch_id)
        if view_name == "memory":
            content = [e["payload"] for e in events if e.get("event_type") == "memory_update"]
        elif view_name == "summary":
            content = self._semantic_summary(
                trajectory_id, branch_id, events, profile_id=profile_id, force_refresh=refresh_summary,
            )
        elif view_name == "failures":
            content = [e for e in events if e.get("event_type") == "tool_result" and e.get("payload", {}).get("status") in FAILURE_STATUSES]
        elif view_name == "failure_patterns":
            content = self.failure_patterns(trajectory_id)
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
            content = self._context_assembly(
                trajectory_id, branch_id, events, token_budget, profile_id=profile_id, refresh_summary=refresh_summary,
            )
        elif view_name == "context_budget":
            content = self._context_assembly(
                trajectory_id, branch_id, events, token_budget, profile_id=profile_id, refresh_summary=refresh_summary,
            )
        elif view_name == "rl_dataset":
            content = self.export_rl_dataset(trajectory_id, branch_id)
        else:
            content = events
        source_events = [e["event_id"] for e in events]
        if view_name == "failure_patterns":
            # A failure-pattern source is the action that failed, rather than
            # the failed result which is already represented by failure_event_id.
            source_events = self._dedupe_ids([
                event_id
                for pattern in content
                for event_id in pattern.get("source_event_ids", [])
            ])
        view = View(view_name=view_name, trajectory_id=trajectory_id, branch_id=branch_id, content=content, source_events=source_events, metadata={"event_count": len(events)})
        view_dict = to_dict(view)
        # _semantic_summary owns profile-scoped summary materialization.
        if view_name != "summary":
            self.store.put_object(uris.view_key(trajectory_id, branch_id, view_name), view_dict)
        return view_dict

    def _semantic_summary(
        self,
        trajectory_id: str,
        branch_id: str,
        events: List[Dict[str, Any]],
        profile_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Materialize an incremental LLM digest for the prefix outside short-term context."""
        judge = SemanticRepairJudge(profile_id=profile_id)
        profile_identity = judge.profile_id or f"{judge.provider}:{judge.model or 'default'}"
        source_events = events[:-RECENT_CAUSAL_EVENT_LIMIT] if len(events) > RECENT_CAUSAL_EVENT_LIMIT else []
        source_ids = [event["event_id"] for event in source_events]
        raw_tokens = self._estimate_tokens(source_events)
        if not source_events:
            return {
                "schema_version": "semantic_summary.v1",
                "trajectory_id": trajectory_id,
                "branch_id": branch_id,
                "summary": "",
                "key_facts": [],
                "open_items": [],
                "status": "not_needed",
                "method": "incremental_llm_semantic_summary",
                "scope": {"start_event_id": None, "end_event_id": None, "source_event_count": 0},
                "source_event_ids": [],
                "source_event_timeline": [],
                "raw_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 0.0,
                "materialization": {"cache_hit": True, "generation_mode": "not_needed", "previous_covered_event_count": 0, "profile_id": judge.profile_id or None},
                "llm": {"enabled": False, "profile_id": judge.profile_id or None, "provider": None, "model": None, "reason": "The complete trajectory fits in the recent causal window; no earlier prefix needs compression.", "error": None, "execution": {}},
            }
        summary_key = uris.summary_view_key(trajectory_id, branch_id, profile_identity)
        cached_view = self.store.get_object(summary_key) or {}
        cached = cached_view.get("content") if isinstance(cached_view, dict) else None
        if not force_refresh and isinstance(cached, dict) and cached.get("schema_version") == "semantic_summary.v1" and cached.get("source_event_ids") == source_ids:
            content = dict(cached)
            content["materialization"] = {**(content.get("materialization") or {}), "cache_hit": True}
            return content

        previous_summary = ""
        delta_events = source_events
        generation_mode = "full"
        if isinstance(cached, dict):
            cached_ids = cached.get("source_event_ids") or []
            cached_text = str(cached.get("summary") or "")
            if cached_text and source_ids[:len(cached_ids)] == cached_ids and len(source_ids) > len(cached_ids):
                previous_summary = cached_text
                delta_events = source_events[len(cached_ids):]
                generation_mode = "incremental"

        result = judge.summarize_trajectory_context(delta_events, previous_summary)
        summary_text = str(result.get("summary") or "")
        status = "ready" if result.get("enabled") and summary_text and not result.get("error") else "unavailable"
        content = {
            "schema_version": "semantic_summary.v1",
            "trajectory_id": trajectory_id,
            "branch_id": branch_id,
            "summary": summary_text,
            "key_facts": result.get("key_facts") or [],
            "open_items": result.get("open_items") or [],
            "status": status,
            "method": "incremental_llm_semantic_summary",
            "scope": {
                "start_event_id": source_ids[0] if source_ids else None,
                "end_event_id": source_ids[-1] if source_ids else None,
                "source_event_count": len(source_events),
            },
            "source_event_ids": source_ids,
            "source_event_timeline": [
                {"event_id": event["event_id"], "event_type": event.get("event_type"), "branch_id": event.get("branch_id"), "text": self._event_line(event, 260)}
                for event in source_events
            ],
            "raw_tokens": raw_tokens,
            "compressed_tokens": self._estimate_tokens(summary_text) if summary_text else 0,
            "compression_ratio": round((1 - (self._estimate_tokens(summary_text) / raw_tokens)) * 100, 1) if summary_text and raw_tokens else 0.0,
            "materialization": {
                "cache_hit": False,
                "generation_mode": generation_mode,
                "previous_covered_event_count": len(source_events) - len(delta_events),
                "profile_id": judge.profile_id or None,
            },
            "llm": {
                "enabled": bool(result.get("enabled")),
                "profile_id": judge.profile_id or None,
                "provider": result.get("provider"),
                "model": result.get("model"),
                "reason": result.get("reason"),
                "error": result.get("error"),
                "execution": result.get("execution") or result.get("_contextdb_execution") or {},
            },
        }
        view = View(
            view_name="summary", trajectory_id=trajectory_id, branch_id=branch_id,
            content=content, source_events=source_ids, metadata={"event_count": len(source_events), "semantic": True},
        )
        self.store.put_object(summary_key, to_dict(view))
        return content

    def _context_assembly(
        self,
        trajectory_id: str,
        branch_id: str,
        events: List[Dict[str, Any]],
        token_budget: int,
        profile_id: Optional[str] = None,
        refresh_summary: bool = False,
    ) -> Dict[str, Any]:
        """Pack short-term, retrieved, and compressed context with provenance."""
        budget = max(1, int(token_budget or 2000))
        recent_events = events[-RECENT_CAUSAL_EVENT_LIMIT:]
        summary = self._semantic_summary(
            trajectory_id, branch_id, events, profile_id=profile_id, force_refresh=refresh_summary,
        )
        current_task_event = next((event for event in reversed(events) if event.get("event_type") == "user_message"), None)
        current_task = self._payload_text(current_task_event) or "No user request has been recorded on this branch."

        task_ids = [current_task_event["event_id"]] if current_task_event else []
        recent_text = "\n".join(f"- {self._event_line(event, 420)}" for event in recent_events)
        branch_state = self._context_branch_state(trajectory_id, branch_id)
        memories = self._retrieve_context_memories(trajectory_id, events, current_task)
        matched_skills = self._context_matched_skills(events)
        memory_text = "\n".join(f"- {item['fact']}" for item in memories)
        skill_text = "\n".join(f"- {item['instruction']}" for item in matched_skills)
        digest_text = str(summary.get("summary") or "")

        segments = [
            self._context_segment("current_task", "P0", "Current Task", current_task, task_ids, "Latest user request on the selected branch."),
            self._context_segment("recent_causal_trace", "P1", "Recent Causal Trace", recent_text, [event["event_id"] for event in recent_events], "Short-term causal window: recent user, assistant, tool-call, and tool-result events."),
            self._context_segment("branch_version_state", "P2", "Branch / Version State", branch_state["text"], branch_state["source_event_ids"], "Active branch head and the latest snapshot/version transition."),
            self._context_segment("retrieved_memory", "P3", "Retrieved Memory", memory_text, [item["event_id"] for item in memories], "Top memory_update records ranked by vector relevance to the current task, with recency as a tie-break."),
            self._context_segment("matched_skills", "P4", "Matched Skills", skill_text, [item["event_id"] for item in matched_skills], "Latest ContextDB skill recommendation events on the selected branch."),
            self._context_segment("earlier_trajectory_digest", "P5", "Earlier Trajectory Digest", digest_text, summary.get("source_event_ids") or [], "LLM-generated incremental semantic digest of the earlier completed trajectory prefix."),
        ]
        remaining = budget
        rendered_sections = []
        for segment in segments:
            if not segment["text"]:
                segment.update({"included": False, "loaded_token_cost": 0, "selection_status": "not_available", "omission_reason": "No eligible context was available for this segment."})
                continue
            cost = segment["token_cost"]
            if cost <= remaining:
                segment.update({"included": True, "loaded_text": segment["text"], "loaded_token_cost": cost, "selection_status": "included", "omission_reason": ""})
                remaining -= cost
            elif remaining > 0:
                truncated = self._truncate_to_tokens(segment["text"], remaining)
                loaded_cost = self._estimate_tokens(truncated)
                segment.update({"included": True, "loaded_text": truncated, "loaded_token_cost": loaded_cost, "selection_status": "truncated", "omission_reason": "Truncated to fit the remaining context budget."})
                remaining = max(0, remaining - loaded_cost)
            else:
                segment.update({"included": False, "loaded_text": "", "loaded_token_cost": 0, "selection_status": "omitted", "omission_reason": "Omitted after the budget was exhausted."})
            if segment.get("included"):
                rendered_sections.append(f"## {segment['title']}\n{segment['loaded_text']}")

        selected_tokens = sum(segment.get("loaded_token_cost", 0) for segment in segments)
        max_selectable_tokens = sum(segment.get("token_cost", 0) for segment in segments if segment.get("text"))
        full_tokens = self._estimate_tokens(events)
        return {
            "schema_version": "context_assembly.v1",
            "trajectory_id": trajectory_id,
            "branch_id": branch_id,
            "head_event_id": self.get_branch(trajectory_id, branch_id).get("head_event_id"),
            "token_budget": budget,
            "selected_tokens": selected_tokens,
            "max_selectable_tokens": max_selectable_tokens,
            "remaining_tokens": max(0, budget - selected_tokens),
            "estimated_full_history_tokens": full_tokens,
            "estimated_loaded_tokens": selected_tokens,
            "estimated_saved_tokens": max(0, full_tokens - selected_tokens),
            "estimated_saved_percent": round((max(0, full_tokens - selected_tokens) / full_tokens) * 100, 1) if full_tokens else 0.0,
            "loading_policy": "priority budget packing over current task, recent causal trace, version state, retrieved memory, matched skills, and semantic digest",
            "segments": segments,
            "rendered_agent_context": "\n\n".join(rendered_sections),
            "summary": summary,
            # Compatibility fields for existing consumers of current_prompt.
            "recent_events": recent_events,
            "relevant_memory": memories,
        }

    def _context_segment(self, segment_id: str, priority: str, title: str, text: str, source_event_ids: List[str], selection_reason: str) -> Dict[str, Any]:
        text = str(text or "").strip()
        return {
            "segment_id": segment_id,
            "priority": priority,
            "title": title,
            "text": text,
            "token_cost": self._estimate_tokens(text) if text else 0,
            "source_event_ids": self._dedupe_ids(source_event_ids),
            "selection_reason": selection_reason,
        }

    def _context_branch_state(self, trajectory_id: str, branch_id: str) -> Dict[str, Any]:
        branch = self.get_branch(trajectory_id, branch_id)
        snapshots = [snapshot for snapshot in self._snapshots(trajectory_id) if snapshot.get("branch_id") == branch_id]
        latest_snapshot = max(snapshots, key=lambda item: item.get("created_at", ""), default=None)
        version_events = [
            event for event in self.list_events(trajectory_id, branch_id)
            if event.get("event_type") == "version_decision" or (event.get("metadata", {}) or {}).get("operation") in VERSION_EVENT_OPERATIONS
        ][-3:]
        parts = [f"Active branch: {branch_id}", f"Head: {branch.get('head_event_id') or 'empty'}"]
        if latest_snapshot:
            parts.append(f"Latest snapshot: {latest_snapshot.get('snapshot_id')} ({latest_snapshot.get('message') or 'no message'})")
        if version_events:
            parts.append("Recent version state: " + "; ".join(self._event_line(event, 180) for event in version_events))
        ids = [branch.get("head_event_id"), latest_snapshot.get("event_id") if latest_snapshot else None] + [event["event_id"] for event in version_events]
        return {"text": "\n".join(parts), "source_event_ids": self._dedupe_ids(ids)}

    def _retrieve_context_memories(self, trajectory_id: str, events: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        memories = [event for event in events if event.get("event_type") == "memory_update"]
        if not memories:
            return []
        entries = []
        for event in memories:
            payload = event.get("payload", {}) or {}
            fact = str(payload.get("fact") or payload.get("text") or payload.get("summary") or "").strip()
            if fact:
                importance = payload.get("importance", (event.get("metadata", {}) or {}).get("importance", 0))
                try:
                    importance = float(importance or 0)
                except (TypeError, ValueError):
                    importance = 0.0
                entries.append({"entry_id": f"{trajectory_id}:{event['event_id']}", "document": fact, "metadata": {"event_id": event["event_id"], "fact": fact, "timestamp": event.get("timestamp", ""), "importance": importance}})
        if not entries:
            return []
        collection = f"context_memories:{trajectory_id}"
        self.vector_index.replace_owner(collection, trajectory_id, entries)
        matches = self.vector_index.search(collection, query, top_k=MEMORY_RETRIEVAL_LIMIT, min_score=-1.0)
        by_id = {event["event_id"]: event for event in memories}
        selected = []
        for match in matches:
            metadata = match.get("metadata") or {}
            event = by_id.get(metadata.get("event_id"))
            if not event:
                continue
            selected.append({
                "event_id": event["event_id"],
                "fact": metadata.get("fact") or self._payload_text(event),
                "score": match.get("score"),
                "importance": metadata.get("importance", 0),
                "timestamp": metadata.get("timestamp", ""),
            })
        selected.sort(key=lambda item: (float(item.get("score", 0.0) or 0.0), float(item.get("importance", 0.0) or 0.0), str(item.get("timestamp", ""))), reverse=True)
        for item in selected:
            item["selection_reason"] = (
                f"Vector relevance to current task (score {float(item.get('score', 0.0)):.3f}); "
                f"importance ({float(item.get('importance', 0.0)):.2f}) and recency break ties."
            )
        return selected

    def _context_matched_skills(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected = []
        for event in reversed(events):
            if event.get("event_type") != "skill_recommendation":
                continue
            payload = event.get("payload", {}) or {}
            if not payload.get("matched") and not payload.get("skill_id"):
                continue
            selected.append({
                "event_id": event["event_id"],
                "skill_id": payload.get("skill_id") or payload.get("matched_skill_id"),
                "confidence": payload.get("score") or payload.get("confidence"),
                "evidence_count": payload.get("evidence_count"),
                "instruction": str(payload.get("instruction") or payload.get("recommended_action") or "ContextDB supplied a skill recommendation."),
            })
            if len(selected) >= 3:
                break
        return list(reversed(selected))

    def _truncate_to_tokens(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        max_chars = max(1, token_budget * 4)
        if len(text) <= max_chars:
            return text
        return text[:max(1, max_chars - 3)].rstrip() + "..."

    def query_sql(self, trajectory_id: str, sql: str, branch_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute one read-only ContextQL statement over logical trajectory relations."""
        resolved_branch_id = branch_id or self.get_trajectory(trajectory_id).get("default_branch") or "main"
        return ContextQLExecutor(self).execute(trajectory_id, sql, resolved_branch_id)

    def translate_natural_language_sql(self, trajectory_id: str, question: str, branch_id: Optional[str] = None, profile_id: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Translate one question into ContextQL SQL without executing it."""
        if not str(question or "").strip():
            raise ValueError("natural-language question is empty")
        # ContextQL is trajectory-scoped. The default branch is only used for
        # internal materialization of branch-addressable relations.
        trajectory = self.get_trajectory(trajectory_id)
        resolved_branch_id = branch_id or trajectory.get("default_branch") or "main"
        self.get_branch(trajectory_id, resolved_branch_id)
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
        translation = SemanticRepairJudge(provider=provider, model=model, profile_id=profile_id).translate_contextql(question, relations)
        sql = str(translation.get("sql") or "").strip()
        if not translation.get("enabled") or not sql:
            raise ValueError(translation.get("error") or translation.get("reason") or "LLM did not return SQL")
        # Apply the same read-only grammar gate now, while deferring execution to /api/v1/sql.
        validated_sql = ContextQLExecutor(self)._validate(sql)
        translation["sql"] = validated_sql
        return {"schema_version": "contextql_nl_translation.v1", "trajectory_id": trajectory_id, "branch_id": resolved_branch_id, "question": question, "translation": translation}

    def natural_language_query(self, trajectory_id: str, question: str, branch_id: Optional[str] = None, profile_id: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Compatibility helper: translate then execute a natural-language ContextQL request."""
        translated = self.translate_natural_language_sql(trajectory_id, question, branch_id, profile_id, provider, model)
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

    def create_version_snapshot(
        self,
        trajectory_id: str,
        branch_id: str = "main",
        message: str = "",
        origin: str = "manual",
        reason: str = "",
        actor: str = "contextdb",
        refs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an auditable logical snapshot at the current branch head."""
        event = self.append_event(
            trajectory_id,
            "system_event",
            {
                "text": "ContextDB created a logical snapshot before a versioned action.",
                "message": message,
                "reason": reason,
                "origin": origin,
                "logical_context_only": True,
            },
            branch_id=branch_id,
            actor=actor,
            refs=refs,
            metadata={"operation": "version_snapshot_created", "origin": origin, "logical_context_only": True},
        )
        snapshot = self.snapshot(trajectory_id, branch_id=branch_id, message=message)
        event["payload"]["snapshot_id"] = snapshot["snapshot_id"]
        event["refs"] = {**event.get("refs", {}), "snapshot_id": snapshot["snapshot_id"]}
        self.store.put_object(uris.event_key(trajectory_id, event["event_id"]), event)
        return {"trajectory_id": trajectory_id, "branch_id": branch_id, "snapshot": snapshot, "event": event}

    def create_version_branch(
        self,
        trajectory_id: str,
        branch_id: str,
        from_branch: str = "main",
        base_event_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        reason: str = "",
        origin: str = "manual",
        refs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a branch and persist a branch-created event on the new path."""
        snapshot = None
        if snapshot_id:
            snapshot = self.store.get_object(uris.snapshot_key(trajectory_id, snapshot_id))
            if snapshot is None:
                raise KeyError(f"snapshot not found: {snapshot_id}")
            base_event_id = snapshot.get("event_id")
            from_branch = snapshot.get("branch_id") or from_branch
        branch = self.create_branch(
            trajectory_id,
            branch_id,
            base_event_id=base_event_id,
            from_branch=from_branch,
            metadata={
                "version_origin": origin,
                "created_from_snapshot": snapshot_id,
                "reason": reason,
            },
        )
        event = self.append_event(
            trajectory_id,
            "system_event",
            {
                "text": f"ContextDB created repair branch {branch_id}.",
                "branch_id": branch_id,
                "from_branch": from_branch,
                "snapshot_id": snapshot_id,
                "reason": reason,
                "origin": origin,
                "logical_context_only": True,
            },
            branch_id=branch_id,
            actor="contextdb",
            refs={**(refs or {}), **({"snapshot_id": snapshot_id} if snapshot_id else {})},
            metadata={"operation": "version_branch_created", "origin": origin, "logical_context_only": True},
        )
        return {"trajectory_id": trajectory_id, "branch": self.get_branch(trajectory_id, branch_id), "event": event, "snapshot": snapshot}

    def create_version_rollback(
        self,
        trajectory_id: str,
        snapshot_id: str,
        target_branch_id: str,
        reason: str = "",
        origin: str = "manual",
        refs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a logical rollback branch without touching an Agent workspace."""
        branch = self.rollback(trajectory_id, snapshot_id, target_branch_id)
        branch["metadata"] = {**branch.get("metadata", {}), "version_origin": origin, "reason": reason, "logical_context_only": True}
        self.store.put_object(uris.branch_key(trajectory_id, target_branch_id), branch)
        event = self.append_event(
            trajectory_id,
            "system_event",
            {
                "text": f"ContextDB created logical rollback branch {target_branch_id}.",
                "snapshot_id": snapshot_id,
                "branch_id": target_branch_id,
                "reason": reason,
                "origin": origin,
                "logical_context_only": True,
                "workspace_restored": False,
            },
            branch_id=target_branch_id,
            actor="contextdb",
            refs={**(refs or {}), "snapshot_id": snapshot_id},
            metadata={"operation": "version_rollback_created", "origin": origin, "logical_context_only": True},
        )
        return {"trajectory_id": trajectory_id, "branch": self.get_branch(trajectory_id, target_branch_id), "event": event}

    def version_control_status(self, trajectory_id: str) -> Dict[str, Any]:
        """Return the version-control state and its auditable lifecycle events."""
        branches = sorted(self._branches(trajectory_id), key=lambda item: item.get("branch_id", ""))
        snapshots = sorted(self._snapshots(trajectory_id), key=lambda item: item.get("created_at", ""))
        events = [
            event for event in self._all_events(trajectory_id)
            if event.get("event_type") == "version_decision"
            or (event.get("metadata", {}) or {}).get("operation") in VERSION_EVENT_OPERATIONS
        ]
        return {
            "schema_version": "contextdb.version_control.v1",
            "trajectory": self.get_trajectory(trajectory_id),
            "branches": branches,
            "snapshots": snapshots,
            "version_events": events,
            "workspace_restore": {
                "supported": False,
                "status": "not_connected",
                "message": "ContextDB rollback creates a logical trajectory branch only; it never restores local workspace files automatically.",
            },
        }

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
                "payload": e.get("payload", {}),
                "is_branch_head": eid in branch_heads,
                "branch_heads": branch_heads.get(eid, []),
                "branch_head": ",".join(branch_heads.get(eid, [])),
                "snapshots": snapshot_events.get(eid, []),
                "version_operation": (e.get("metadata", {}) or {}).get("operation"),
            })
            for parent in e.get("parent_event_ids") or []:
                edges.append({"source": parent, "target": eid, "kind": "parent"})
        for b in branches:
            if b.get("base_event_id") and b.get("head_event_id") and b.get("base_event_id") != b.get("head_event_id"):
                edges.append({"source": b["base_event_id"], "target": b["head_event_id"], "kind": "branch", "branch_id": b.get("branch_id")})
        return {"trajectory": self.get_trajectory(trajectory_id), "nodes": nodes, "edges": edges, "branches": branches, "snapshots": snapshots, "views": views}

    def event_detail(self, trajectory_id: str, event_id: str) -> Dict[str, Any]:
        """Return a complete event together with the graph metadata used by Playback."""
        event = self.get_event(trajectory_id, event_id)
        branches = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/branches"))
        snapshots = list(self.store.scan_prefix(f"trajectories/{trajectory_id}/snapshots"))
        branch_heads: Dict[str, List[str]] = {}
        for branch in branches:
            if branch.get("head_event_id"):
                branch_heads.setdefault(branch["head_event_id"], []).append(branch.get("branch_id"))
        snapshot_ids = [snapshot.get("snapshot_id") for snapshot in snapshots if snapshot.get("event_id") == event_id]

        detail = dict(event)
        detail.update({
            "id": event_id,
            "type": "event",
            "label": self._event_line(event, max_length=None),
            "is_branch_head": event_id in branch_heads,
            "branch_heads": branch_heads.get(event_id, []),
            "branch_head": ",".join(branch_heads.get(event_id, [])),
            "snapshots": snapshot_ids,
            "version_operation": (event.get("metadata", {}) or {}).get("operation"),
        })
        return detail

    def failure_patterns(self, trajectory_id: str, branch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract failures from every branch, mirroring ``success_patterns``.

        A branch-local filter prevents an inherited ancestor event from being
        emitted repeatedly for every descendant branch.
        """
        events_by_branch = (
            {branch_id: self.list_events(trajectory_id, branch_id)}
            if branch_id is not None
            else self._events_by_branch(trajectory_id)
        )
        patterns = []
        for current_branch_id, events in events_by_branch.items():
            for i, event in enumerate(events):
                if event.get("branch_id") != current_branch_id or not self._is_failed_tool_result(event):
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
                patterns.append({
                    "schema_version": "experience_mining.v2",
                    "pattern_id": f"fp_{event['event_id'].split('_')[-1]}",
                    "failure_event_id": event["event_id"],
                    "branch_id": current_branch_id,
                    "failed_tool": self._tool_name(tool_call),
                    "failed_command": self._command_text(tool_call),
                    "error_signature": signature,
                    "normalized_signature": normalized_signature,
                    "preceding_action": self._payload_text(assistant),
                    "likely_cause": cause_judgment.get("likely_cause"),
                    "likely_cause_judgment": cause_judgment,
                    "repair_status": "not_evaluated",
                    "repair_events_after_failure": repair_events,
                    # The visible source of a failure pattern is its direct
                    # non-failure action; the full causal chain remains below.
                    "source_event_ids": self._compact_ids([tool_call]),
                    "evidence_event_ids": self._compact_ids([assistant, tool_call, event]),
                    "highlight_event_ids": self._compact_ids([event]),
                })
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
                    "source_event_ids": self._compact_ids([tool_call]),
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
                        "tool_call_event_id": (success.get("source_event_ids") or [None])[0],
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
                "tool_call_event_id": (failure.get("source_event_ids") or [None])[0],
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
        source_events = self.list_events(trajectory_id)
        source_event_ids = [event["event_id"] for event in source_events]
        profile_identity = judge.profile_id or f"{judge.provider}:{judge.model or 'default'}"
        cache_key = uris.semantic_judgments_view_key(trajectory_id, profile_identity)
        cached_view = self.store.get_object(cache_key) or {}
        cached_content = cached_view.get("content") if isinstance(cached_view, dict) else None
        if (
            cached_view.get("metadata", {}).get("schema_version") == "semantic_repair_judgment_cache.v1"
            and cached_view.get("source_events") == source_event_ids
            and isinstance(cached_content, list)
        ):
            return cached_content

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

        # Do not preserve transient transport failures: a later request can
        # retry the semantic judgment batch after the LLM becomes available.
        cacheable = all(
            judgment.get("enabled") and not judgment.get("error")
            for row in rows
            for judgment in [row.get("judgment", {})]
        )
        if cacheable:
            view = View(
                view_name="semantic_repair_judgments",
                trajectory_id=trajectory_id,
                branch_id="all_branches",
                content=rows,
                source_events=source_event_ids,
                metadata={
                    "schema_version": "semantic_repair_judgment_cache.v1",
                    "event_count": len(source_events),
                    "profile_id": judge.profile_id or None,
                    "provider": judge.provider,
                    "model": judge.model,
                },
            )
            self.store.put_object(cache_key, to_dict(view))
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
                "skill_source_trajectory_id": selected.get("source_trajectory_id"),
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
        events = self._all_events(trajectory_id)
        linked_event_ids: Set[str] = set()
        for event in events:
            refs = event.get("refs", {}) or {}
            metadata = event.get("metadata", {}) or {}
            if event.get("event_type") == "skill_match" and metadata.get("operation") in {"skill_retrieval", "online_skill_retrieval"}:
                failure_id = refs.get("failure_event_id")
                if failure_id:
                    linked_event_ids.add(failure_id)
        event_by_id = {event.get("event_id"): event for event in events}
        for event_id in list(linked_event_ids):
            linked_call_id = (event_by_id.get(event_id, {}).get("refs", {}) or {}).get("tool_call_event_id")
            if linked_call_id:
                linked_event_ids.add(linked_call_id)
        rows = []
        for event in events:
            refs = event.get("refs", {}) or {}
            metadata = event.get("metadata", {}) or {}
            if event.get("event_type") in {"skill_match", "skill_recommendation", "skill_decision", "tool_call", "tool_result", "assistant_message"} and (
                metadata.get("operation") in {
                    "skill_retrieval", "online_skill_retrieval", "skill_recommendation_delivery",
                    "skill_decision", "skill_application",
                }
                or refs.get("skill_match_event_id")
                or refs.get("skill_id")
                or event.get("event_id") in linked_event_ids
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
        """Return a stable semantic identity for a learned skill.

        Error output is intentionally excluded: it varies across operating systems,
        encodings, and individual runs, and must not create a new skill identity.
        """
        tool = self._skill_id_token(failure.get("tool") or failure.get("failed_tool"), "agent")
        cause = self._canonical_failure_kind(failure)
        command = self._command_family(failure.get("command") or failure.get("failed_command_pattern"))
        return f"skill_{tool}_{cause}_{command}"

    def _skill_trigger(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Build a readable trigger without carrying forward broken console text."""
        likely_cause = str(failure.get("likely_cause") or "")
        fallback_signature = self._canonical_failure_kind(failure).replace("_", " ")
        normalized_signature = self._clean_skill_text(failure.get("normalized_signature"))
        error_signature = self._clean_skill_text(failure.get("error_signature"))
        return {
            "failed_tool": failure.get("tool") or failure.get("failed_tool"),
            "failed_command_pattern": failure.get("command") or failure.get("failed_command_pattern"),
            "normalized_signature": normalized_signature or fallback_signature,
            "error_signature": error_signature,
            "likely_cause": likely_cause or fallback_signature,
        }

    @staticmethod
    def _clean_skill_text(value: Any) -> str:
        text = str(value or "")
        # These markers are emitted when Windows console output was decoded
        # through the wrong code page. Keep source events intact, but never
        # propagate that noise into a reusable learned-skill artifact.
        return "" if "\ufffd" in text or "锟" in text else text

    def _canonical_failure_kind(self, failure: Dict[str, Any]) -> str:
        text = " ".join(str(failure.get(key) or "") for key in (
            "likely_cause", "normalized_signature", "error_signature",
        )).lower()
        if any(marker in text for marker in (
            "directorynotfound", "directory not found", "missing parent directory",
            "missing target directory", "parent path", "getcontentwriterdirectorynotfounderror",
        )):
            return "missing_parent_directory"
        if any(marker in text for marker in (
            "pathnotfound", "itemnotfound", "objectnotfound", "file not found",
            "missing file", "missing path", "specified path",
        )):
            return "file_not_found"
        return self._skill_id_token(failure.get("likely_cause"), "agent_failure")

    def _command_family(self, command: Any) -> str:
        text = str(command or "").lower()
        known_commands = (
            ("get-content", "get_content"), ("set-content", "set_content"),
            ("add-content", "add_content"), ("apply_patch", "apply_patch"),
            ("git ", "git"), ("pip ", "pip"), ("npm ", "npm"),
        )
        for marker, family in known_commands:
            if marker in text:
                return family
        return self._skill_id_token(text, "agent_action")

    @staticmethod
    def _skill_id_token(value: Any, fallback: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
        return "_".join(tokens[:4]) or fallback

    def _skill_key_from_trigger(self, trigger: Dict[str, Any]) -> str:
        return self._skill_key({
            "tool": trigger.get("failed_tool") or trigger.get("tool"),
            "command": trigger.get("failed_command_pattern") or trigger.get("command"),
            "likely_cause": trigger.get("likely_cause"),
            "normalized_signature": trigger.get("normalized_signature"),
            "error_signature": trigger.get("error_signature"),
        })

    def _skill_id_aliases(self, trajectory_id: str, current_skills: List[Dict[str, Any]]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}

        def add(skill_id: Any, trigger: Any) -> None:
            if not isinstance(trigger, dict) or not skill_id:
                return
            canonical = self._skill_key_from_trigger(trigger)
            if str(skill_id) != canonical:
                aliases[str(skill_id)] = canonical

        for key in self.store.list_objects(f"trajectories/{trajectory_id}/views"):
            view = self.store.get_object(key) or {}
            content = view.get("content")
            if isinstance(content, list):
                candidates = content
            elif isinstance(content, dict):
                candidates = content.get("skills", [])
            else:
                candidates = []
            for skill in candidates if isinstance(candidates, list) else []:
                if isinstance(skill, dict):
                    add(skill.get("skill_id"), skill.get("trigger"))

        for event in self._all_events(trajectory_id):
            payload = event.get("payload") or {}
            for match in payload.get("matches", []) if isinstance(payload.get("matches"), list) else []:
                if isinstance(match, dict):
                    add(match.get("matched_skill_id") or match.get("skill_id"), match.get("matched_trigger") or match.get("trigger"))

        for skill in current_skills:
            add(skill.get("skill_id"), skill.get("trigger"))
        return aliases

    def _rewrite_skill_ids(self, value: Any, aliases: Dict[str, str]) -> bool:
        changed = False
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"skill_id", "matched_skill_id"} and isinstance(child, str) and child in aliases:
                    value[key] = aliases[child]
                    changed = True
                else:
                    changed = self._rewrite_skill_ids(child, aliases) or changed
        elif isinstance(value, list):
            for child in value:
                changed = self._rewrite_skill_ids(child, aliases) or changed
        return changed

    def _migrate_skill_references(self, trajectory_id: str, current_skills: List[Dict[str, Any]]) -> Dict[str, str]:
        """Align mutable trace records with stable IDs without rewriting snapshots."""
        aliases = self._skill_id_aliases(trajectory_id, current_skills)
        if not aliases:
            return aliases

        for event in self._all_events(trajectory_id):
            if self._rewrite_skill_ids(event, aliases):
                self.store.put_object(uris.event_key(trajectory_id, event["event_id"]), event)

        view_prefix = f"trajectories/{trajectory_id}/views"
        for key in self.store.list_objects(view_prefix):
            view = self.store.get_object(key)
            if view and self._rewrite_skill_ids(view, aliases):
                self.store.put_object(key, view)

        for key in self.store.list_objects("hook_sessions"):
            session = self.store.get_object(key)
            if session and session.get("trajectory_id") == trajectory_id and self._rewrite_skill_ids(session, aliases):
                self.store.put_object(key, session)
        return aliases

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

    def _event_line(self, event: Dict[str, Any], max_length: Optional[int] = 120) -> str:
        payload = event.get("payload", {})
        text = str(payload.get("text") or payload.get("command") or payload.get("summary") or payload.get("preview") or payload)
        if max_length is not None and len(text) > max_length:
            text = text[:max(0, max_length - 3)].rstrip() + "..."
        return f"{event.get('event_type')}:{text}"

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


def _tool_call_ids_for_results(self, trajectory_id, result_event_ids):
    target_ids = {event_id for event_id in result_event_ids if event_id}
    tool_call_ids = {}
    if not target_ids:
        return tool_call_ids
    for events in self._events_by_branch(trajectory_id).values():
        for index, event in enumerate(events):
            result_id = event.get("event_id")
            if result_id not in target_ids:
                continue
            tool_call = self._previous_event(events, index, "tool_call")
            if tool_call and tool_call.get("event_id"):
                tool_call_ids[result_id] = tool_call["event_id"]
    return tool_call_ids


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
            "failure_tool_call_event_id": failure.get("tool_call_event_id"),
            "success_tool_call_event_id": candidate.get("tool_call_event_id"),
            "tool": candidate.get("tool"), "command": candidate.get("command"),
            "strategy": candidate.get("strategy"), "outcome": candidate.get("outcome"),
            "primary_rule": candidate.get("primary_rule") or candidate.get("link_type"),
            "judgment_label": judgment.get("label"), "confidence": float(judgment.get("confidence", 0) or 0),
            "reason": judgment.get("reason"),
        })
    failure_rows = [{"failure_id": fid, "failure": failure, "vector_text": _vector_text([failure.get("tool"), failure.get("command"), failure.get("normalized_signature"), failure.get("error_signature"), failure.get("likely_cause")])} for fid, failure in failures.items()]
    result_event_ids = list(failures) + [
        event.get("success_event_id")
        for events in support.values()
        for event in events
    ]
    inferred_tool_calls = _tool_call_ids_for_results(self, trajectory_id, result_event_ids)
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
            "trigger": self._skill_trigger(first),
            "failure_grouping": {"method": "vector_cosine", "embedding": "hash-384", "threshold": .74, "failure_event_ids": ids, "group_size": len(ids)},
            "recommended_actions": actions, "avoid_actions": [], "evidence_refs": refs,
            "confidence": {"support_count": len(all_support), "max_llm_confidence": max([float(event.get("confidence", 0) or 0) for event in all_support] or [0]), "evidence_levels": ["structural"]},
            "status": "candidate", "highlight_event_ids": self._dedupe_ids(
                ids
                + [inferred_tool_calls.get(event_id) for event_id in ids]
                + [event.get("failure_tool_call_event_id") for event in all_support]
                + [event.get("success_event_id") for event in all_support]
                + [
                    event.get("success_tool_call_event_id")
                    or inferred_tool_calls.get(event.get("success_event_id"))
                    for event in all_support
                ]
            ),
            "action_grouping": {"method": "vector_cosine", "embedding": "hash-384", "threshold": .78, "group_count": len(actions)},
        }
        skills.append(skill)
    self._migrate_skill_references(trajectory_id, skills)
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
