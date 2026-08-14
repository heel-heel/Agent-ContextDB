from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple


READ_ONLY_PREFIXES = ("select", "with", "explain")


class ContextQLExecutor:
    """Read-only SQL facade over ContextDB's logical relational model."""

    def __init__(self, contextdb: Any):
        self.db = contextdb

    def execute(self, trajectory_id: str, sql: str, branch_id: str = "main") -> Dict[str, Any]:
        statement = self._validate(sql)
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        self._create_schema(connection)
        self._load_trajectory(connection, trajectory_id, branch_id)
        cursor = connection.execute(statement, {"trajectory_id": trajectory_id, "branch_id": branch_id})
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [column[0] for column in cursor.description] if cursor.description else []
        return {
            "schema_version": "contextql_result.v1",
            "trajectory_id": trajectory_id,
            "branch_id": branch_id,
            "sql": statement,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "highlight_event_ids": self._collect_event_ids(rows),
            "relations": ["trajectories", "events", "event_edges", "branches", "snapshots", "skills", "skill_evidence", "materialized_views", "failure_patterns", "repair_strategies", "learned_skills", "skill_application_trace"],
        }

    def _validate(self, sql: str) -> str:
        statement = str(sql or "").strip()
        if not statement:
            raise ValueError("SQL query is empty")
        statement = statement[:-1].strip() if statement.endswith(";") else statement
        if ";" in statement:
            raise ValueError("ContextQL accepts one statement only")
        compact = re.sub(r"\s+", " ", statement).strip().lower()
        if not compact.startswith(READ_ONLY_PREFIXES):
            raise ValueError("ContextQL is read-only: only SELECT, WITH, and EXPLAIN are allowed")
        forbidden = r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex|load_extension)\b"
        if re.search(forbidden, compact):
            raise ValueError("ContextQL rejected a non-read-only SQL keyword")
        return statement

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        statements = [
            "CREATE TABLE trajectories (trajectory_id TEXT, title TEXT, agent_id TEXT, source_id TEXT, default_branch TEXT, head_event_id TEXT, metadata_json TEXT)",
            "CREATE TABLE events (event_id TEXT, trajectory_id TEXT, branch_id TEXT, event_type TEXT, actor TEXT, timestamp TEXT, payload_json TEXT, refs_json TEXT, metadata_json TEXT, status TEXT, tool_name TEXT, command TEXT, preview TEXT, error_signature TEXT, text TEXT, reserved TEXT)",
            "CREATE TABLE event_edges (trajectory_id TEXT, parent_event_id TEXT, child_event_id TEXT)",
            "CREATE TABLE branches (trajectory_id TEXT, branch_id TEXT, base_event_id TEXT, head_event_id TEXT, snapshot_id TEXT, metadata_json TEXT)",
            "CREATE TABLE snapshots (snapshot_id TEXT, trajectory_id TEXT, branch_id TEXT, event_id TEXT, message TEXT, created_at TEXT)",
            "CREATE TABLE skills (skill_id TEXT, trajectory_id TEXT, name TEXT, status TEXT, trigger_json TEXT, confidence_json TEXT, failure_grouping_json TEXT, action_grouping_json TEXT, highlight_event_ids TEXT)",
            "CREATE TABLE skill_evidence (skill_id TEXT, trajectory_id TEXT, failure_event_id TEXT, success_event_id TEXT, action_id TEXT, primary_rule TEXT, judgment_label TEXT)",
            "CREATE TABLE materialized_views (trajectory_id TEXT, branch_id TEXT, view_name TEXT, created_at TEXT, event_count INTEGER, object_key TEXT)",
            "CREATE TABLE failure_patterns (trajectory_id TEXT, failure_event_id TEXT, branch_id TEXT, failed_tool TEXT, failed_command TEXT, likely_cause TEXT, error_signature TEXT, normalized_signature TEXT, preceding_action TEXT, source_event_ids TEXT, highlight_event_ids TEXT)",
            "CREATE TABLE repair_strategies (trajectory_id TEXT, failure_event_id TEXT, success_event_id TEXT, failure_branch TEXT, repair_branch TEXT, tool TEXT, command TEXT, strategy TEXT, outcome TEXT, primary_rule TEXT, why_linked TEXT, evidence_json TEXT, highlight_event_ids TEXT)",
            "CREATE TABLE learned_skills (skill_id TEXT, trajectory_id TEXT, name TEXT, status TEXT, trigger_json TEXT, recommended_actions_json TEXT, confidence_json TEXT, highlight_event_ids TEXT)",
            "CREATE TABLE skill_application_trace (trajectory_id TEXT, event_id TEXT, branch_id TEXT, event_type TEXT, actor TEXT, timestamp TEXT, payload_json TEXT, refs_json TEXT)",
        ]
        for statement in statements:
            conn.execute(statement)

    def _load_trajectory(self, conn: sqlite3.Connection, trajectory_id: str, branch_id: str) -> None:
        db = self.db
        trajectory = db.get_trajectory(trajectory_id)
        conn.execute("INSERT INTO trajectories VALUES (?,?,?,?,?,?,?)", (
            trajectory_id, trajectory.get("title"), trajectory.get("agent_id"), trajectory.get("source_id"),
            trajectory.get("default_branch"), trajectory.get("head_event_id"), self._json(trajectory.get("metadata")),
        ))
        all_events = db._all_events(trajectory_id)
        for event in all_events:
            payload = event.get("payload", {}) or {}
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                event.get("event_id"), trajectory_id, event.get("branch_id"), event.get("event_type"), event.get("actor"),
                event.get("timestamp"), self._json(payload), self._json(event.get("refs")), self._json(event.get("metadata")),
                payload.get("status"), payload.get("tool_name"), self._text(payload.get("command")),
                self._text(payload.get("preview")), self._text(payload.get("error_signature")), self._text(payload.get("text")), None,
            ))
            for parent_id in event.get("parent_event_ids") or []:
                conn.execute("INSERT INTO event_edges VALUES (?,?,?)", (trajectory_id, parent_id, event.get("event_id")))
        for branch in db._branches(trajectory_id):
            conn.execute("INSERT INTO branches VALUES (?,?,?,?,?,?)", (
                trajectory_id, branch.get("branch_id"), branch.get("base_event_id"), branch.get("head_event_id"),
                branch.get("snapshot_id"), self._json(branch.get("metadata")),
            ))
        for snapshot in db._snapshots(trajectory_id):
            conn.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?)", (
                snapshot.get("snapshot_id"), trajectory_id, snapshot.get("branch_id"), snapshot.get("event_id"),
                snapshot.get("message"), snapshot.get("created_at"),
            ))
        self._load_materialized_views(conn, trajectory_id)
        self._load_failure_patterns(conn, trajectory_id, branch_id)
        self._load_repair_strategies(conn, trajectory_id)
        self._load_skills(conn, trajectory_id, branch_id)
        self._load_application_trace(conn, trajectory_id)

    def _load_materialized_views(self, conn: sqlite3.Connection, trajectory_id: str) -> None:
        prefix = f"trajectories/{trajectory_id}/views"
        for key in self.db.store.list_objects(prefix):
            view = self.db.store.get_object(key)
            if not view:
                continue
            conn.execute("INSERT INTO materialized_views VALUES (?,?,?,?,?,?)", (
                trajectory_id, view.get("branch_id"), view.get("view_name"), view.get("created_at"),
                (view.get("metadata") or {}).get("event_count"), key,
            ))

    def _load_failure_patterns(self, conn: sqlite3.Connection, trajectory_id: str, branch_id: str) -> None:
        for row in self.db.failure_patterns(trajectory_id, branch_id):
            conn.execute("INSERT INTO failure_patterns VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                trajectory_id, row.get("failure_event_id"), row.get("branch_id"), row.get("failed_tool"),
                row.get("failed_command"), row.get("likely_cause"), row.get("error_signature"),
                row.get("normalized_signature"), row.get("preceding_action"), self._json(row.get("source_event_ids")),
                self._json(row.get("highlight_event_ids")),
            ))

    def _load_repair_strategies(self, conn: sqlite3.Connection, trajectory_id: str) -> None:
        for group in self.db.repair_strategies(trajectory_id):
            failure = group.get("failure", {}) or {}
            for candidate in group.get("repair_candidates", group.get("repairs", [])):
                conn.execute("INSERT INTO repair_strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    trajectory_id, failure.get("failure_event_id"), candidate.get("success_event_id"),
                    failure.get("branch_id"), candidate.get("branch_id"), candidate.get("tool"), candidate.get("command"),
                    candidate.get("strategy"), candidate.get("outcome"), candidate.get("primary_rule") or candidate.get("link_type"),
                    candidate.get("why_linked"), self._json(candidate.get("evidence")), self._json(candidate.get("highlight_event_ids")),
                ))

    def _load_skills(self, conn: sqlite3.Connection, trajectory_id: str, branch_id: str) -> None:
        cached, _ = self.db._cached_learned_skills(trajectory_id)
        if not cached or self._skills_need_refresh(cached):
            # Materialize a fresh view so SQL no longer exposes stale disabled-LLM candidates.
            cached = self.db.query_view(trajectory_id, "learned_skills", branch_id).get("content", [])
        for skill in cached:
            conn.execute("INSERT INTO skills VALUES (?,?,?,?,?,?,?,?,?)", (
                skill.get("skill_id"), trajectory_id, skill.get("name"), skill.get("status"),
                self._json(skill.get("trigger")), self._json(skill.get("confidence")),
                self._json(skill.get("failure_grouping")), self._json(skill.get("action_grouping")),
                self._json(skill.get("highlight_event_ids")),
            ))
            conn.execute("INSERT INTO learned_skills VALUES (?,?,?,?,?,?,?,?)", (
                skill.get("skill_id"), trajectory_id, skill.get("name"), skill.get("status"),
                self._json(skill.get("trigger")), self._json(skill.get("recommended_actions")),
                self._json(skill.get("confidence")), self._json(skill.get("highlight_event_ids")),
            ))
            for action in skill.get("recommended_actions", []):
                support = action.get("support_events", [])
                if not support:
                    support = [{"success_event_id": event_id} for event_id in action.get("support_event_ids", [])]
                for event in support:
                    conn.execute("INSERT INTO skill_evidence VALUES (?,?,?,?,?,?,?)", (
                        skill.get("skill_id"), trajectory_id, event.get("failure_event_id"), event.get("success_event_id"),
                        action.get("action_id"), event.get("primary_rule"), event.get("judgment_label"),
                    ))
            for ref in skill.get("evidence_refs", []):
                conn.execute("INSERT INTO skill_evidence VALUES (?,?,?,?,?,?,?)", (
                    skill.get("skill_id"), trajectory_id, ref.get("failure_event_id"), ref.get("success_event_id"),
                    None, ref.get("primary_rule"), ref.get("judgment_label"),
                ))

    @staticmethod
    def _skills_need_refresh(skills: List[Dict[str, Any]]) -> bool:
        for skill in skills:
            trigger = skill.get("trigger", {}) or {}
            confidence = skill.get("confidence", {}) or {}
            cause = str(trigger.get("likely_cause") or "").strip().lower()
            if cause == "llm cause classification unavailable":
                return True
            if not skill.get("recommended_actions") and float(confidence.get("max_llm_confidence", 0.0) or 0.0) == 0.0:
                return True
        return False

    def _load_application_trace(self, conn: sqlite3.Connection, trajectory_id: str) -> None:
        for event in self.db.skill_application_trace(trajectory_id):
            conn.execute("INSERT INTO skill_application_trace VALUES (?,?,?,?,?,?,?,?)", (
                trajectory_id, event.get("event_id"), event.get("branch_id"), event.get("event_type"), event.get("actor"),
                event.get("timestamp"), self._json(event.get("payload")), self._json(event.get("refs")),
            ))

    def _collect_event_ids(self, rows: Iterable[Dict[str, Any]]) -> List[str]:
        keys = {"event_id", "failure_event_id", "success_event_id", "parent_event_id", "child_event_id", "base_event_id", "head_event_id", "snapshot_event_id"}
        ids = []
        for row in rows:
            for key, value in row.items():
                if key in keys and isinstance(value, str) and value.startswith("evt_"):
                    ids.append(value)
                if key in {"highlight_event_ids", "source_event_ids"}:
                    ids.extend(self._event_ids_from_json(value))
        return list(dict.fromkeys(ids))

    def _event_ids_from_json(self, value: Any) -> List[str]:
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, str) and item.startswith("evt_")]
        return []

    def _json(self, value: Any) -> str:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)

    def _text(self, value: Any) -> str:
        return str(value or "")
