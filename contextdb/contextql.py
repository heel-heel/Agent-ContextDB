from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, Iterable, List


READ_ONLY_PREFIXES = ("select", "with", "explain")


class ContextQLExecutor:
    """Read-only SQL facade over ContextDB's relational model."""

    def __init__(self, contextdb: Any):
        self.db = contextdb

    def execute(self, trajectory_id: str, sql: str, branch_id: str = "main") -> Dict[str, Any]:
        statement = self._validate(sql)
        # Query Explorer only reads persisted entities and materialized views.
        # It never creates an analytical schema or recomputes a dashboard view.
        self.db.get_trajectory(trajectory_id)
        self.db.get_branch(trajectory_id, branch_id)
        connection = sqlite3.connect(self.db.store.db_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
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
                "relations": ["trajectories", "events", "event_edges", "branches", "snapshots", "materialized_views"],
            }
        finally:
            connection.close()

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
        if not re.search(r":trajectory_id\b", compact):
            raise ValueError(
                "ContextQL is trajectory-scoped: filter the relevant relation with trajectory_id = :trajectory_id"
            )
        return statement

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
