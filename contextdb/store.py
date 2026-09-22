from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


class ContextStore(Protocol):
    def put_object(self, key: str, value: Dict[str, Any]) -> None: ...
    def get_object(self, key: str) -> Optional[Dict[str, Any]]: ...
    def list_objects(self, prefix: str) -> List[str]: ...
    def delete_object(self, key: str) -> None: ...
    def create_namespace(self, prefix: str) -> None: ...
    def scan_prefix(self, prefix: str) -> Iterable[Dict[str, Any]]: ...


class SQLiteStore:
    """Relational ContextDB storage with the existing object-key API.

    Entity identity, ownership, time, and relationships live in SQLite tables.
    JSON columns are only for flexible event payloads and metadata. The old
    object-file index remains solely to import existing ``objects/*.json`` data.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_dir = self.root / "objects"  # Legacy import location.
        self.snapshots_dir = self.root / "snapshots"  # Legacy import location.
        self.db_path = self.root / "contextdb.sqlite3"
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        self._migrate_legacy_objects()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
              trajectory_id TEXT PRIMARY KEY, title TEXT NOT NULL,
              agent_id TEXT NOT NULL, source_id TEXT NOT NULL,
              default_branch TEXT NOT NULL, head_event_id TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              branch_id TEXT NOT NULL, event_type TEXT NOT NULL, actor TEXT NOT NULL,
              timestamp TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
              parent_event_ids_json TEXT NOT NULL DEFAULT '[]', refs_json TEXT NOT NULL DEFAULT '{}',
              metadata_json TEXT NOT NULL DEFAULT '{}', status TEXT,
              tool_name TEXT, command TEXT, preview TEXT, error_signature TEXT, text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_trajectory_time ON events(trajectory_id, timestamp, event_id);
            CREATE INDEX IF NOT EXISTS idx_events_branch_time ON events(trajectory_id, branch_id, timestamp, event_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(trajectory_id, event_type);
            CREATE TABLE IF NOT EXISTS event_edges (
              trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              parent_event_id TEXT NOT NULL, child_event_id TEXT NOT NULL REFERENCES events(event_id),
              PRIMARY KEY (trajectory_id, parent_event_id, child_event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_event_edges_child ON event_edges(trajectory_id, child_event_id);
            CREATE TABLE IF NOT EXISTS branches (
              trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id), branch_id TEXT NOT NULL,
              base_event_id TEXT, head_event_id TEXT, snapshot_id TEXT, created_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (trajectory_id, branch_id)
            );
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_id TEXT PRIMARY KEY, trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              branch_id TEXT NOT NULL, event_id TEXT, message TEXT NOT NULL, created_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_trajectory_time ON snapshots(trajectory_id, created_at);
            CREATE TABLE IF NOT EXISTS snapshot_records (
              snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), object_key TEXT NOT NULL,
              value_json TEXT NOT NULL, PRIMARY KEY (snapshot_id, object_key)
            );
            CREATE TABLE IF NOT EXISTS materialized_views (
              object_key TEXT PRIMARY KEY, trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              branch_id TEXT NOT NULL, view_name TEXT NOT NULL, content_json TEXT NOT NULL,
              source_events_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}', event_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_views_trajectory_branch ON materialized_views(trajectory_id, branch_id, view_name);
            CREATE TABLE IF NOT EXISTS artifacts (
              artifact_id TEXT PRIMARY KEY, trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
              kind TEXT NOT NULL, content_ref TEXT NOT NULL, preview TEXT NOT NULL, created_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS hook_sessions (
              source TEXT NOT NULL, session_id TEXT NOT NULL, value_json TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (source, session_id)
            );
            CREATE TABLE IF NOT EXISTS documents (
              object_key TEXT PRIMARY KEY, value_json TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS namespaces (prefix TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);

            -- Legacy inventory, kept only so existing JSON objects can migrate once.
            CREATE TABLE IF NOT EXISTS objects (
              key TEXT PRIMARY KEY, type TEXT NOT NULL, trajectory_id TEXT, branch_id TEXT,
              event_type TEXT, timestamp TEXT, path TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._ensure_column("events", "status", "TEXT")
        self._ensure_column("events", "tool_name", "TEXT")
        self._ensure_column("events", "command", "TEXT")
        self._ensure_column("events", "preview", "TEXT")
        self._ensure_column("events", "error_signature", "TEXT")
        self._ensure_column("events", "text", "TEXT")
        self._ensure_column("materialized_views", "event_count", "INTEGER")
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _json(value: Any, default: Any) -> str:
        return json.dumps(default if value is None else value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _parts(key: str) -> List[str]:
        return [part for part in str(key).strip("/").split("/") if part]

    def _kind_for_key(self, key: str) -> Tuple[str, Tuple[str, ...]]:
        parts = self._parts(key)
        if len(parts) == 3 and parts[0] == "trajectories" and parts[2] == "meta": return "trajectory", (parts[1],)
        if len(parts) == 4 and parts[0] == "trajectories" and parts[2] == "events": return "event", (parts[1], parts[3])
        if len(parts) == 4 and parts[0] == "trajectories" and parts[2] == "branches": return "branch", (parts[1], parts[3])
        if len(parts) == 4 and parts[0] == "trajectories" and parts[2] == "snapshots": return "snapshot", (parts[1], parts[3])
        if len(parts) >= 5 and parts[0] == "trajectories" and parts[2] == "views": return "view", (parts[1], parts[3])
        if len(parts) == 4 and parts[0] == "trajectories" and parts[2] == "artifacts": return "artifact", (parts[1], parts[3])
        if len(parts) == 3 and parts[0] == "hook_sessions": return "hook_session", (parts[1], parts[2])
        return "document", ()

    def create_namespace(self, prefix: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR IGNORE INTO namespaces(prefix) VALUES(?)", (prefix.strip("/"),))
            self.conn.commit()

    def put_object(self, key: str, value: Dict[str, Any]) -> None:
        key = key.strip("/")
        kind, ids = self._kind_for_key(key)
        with self._lock:
            if kind == "trajectory":
                self.conn.execute(
                    """INSERT INTO trajectories VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(trajectory_id) DO UPDATE SET title=excluded.title,agent_id=excluded.agent_id,
                    source_id=excluded.source_id,default_branch=excluded.default_branch,head_event_id=excluded.head_event_id,
                    created_at=excluded.created_at,updated_at=excluded.updated_at,metadata_json=excluded.metadata_json""",
                    (ids[0], value.get("title", ""), value.get("agent_id", "unknown-agent"), value.get("source_id", "unknown-source"),
                     value.get("default_branch", "main"), value.get("head_event_id"), value.get("created_at", ""),
                     value.get("updated_at", ""), self._json(value.get("metadata"), {})),
                )
            elif kind == "event":
                parents = list(value.get("parent_event_ids") or [])
                payload = value.get("payload") or {}
                self.conn.execute(
                    """INSERT INTO events(event_id,trajectory_id,branch_id,event_type,actor,timestamp,payload_json,parent_event_ids_json,refs_json,metadata_json,status,tool_name,command,preview,error_signature,text)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id) DO UPDATE SET trajectory_id=excluded.trajectory_id,branch_id=excluded.branch_id,
                    event_type=excluded.event_type,actor=excluded.actor,timestamp=excluded.timestamp,payload_json=excluded.payload_json,
                    parent_event_ids_json=excluded.parent_event_ids_json,refs_json=excluded.refs_json,metadata_json=excluded.metadata_json,
                    status=excluded.status,tool_name=excluded.tool_name,command=excluded.command,preview=excluded.preview,
                    error_signature=excluded.error_signature,text=excluded.text""",
                    (ids[1], ids[0], value.get("branch_id", "main"), value.get("event_type", "unknown"), value.get("actor", "agent"),
                     value.get("timestamp", ""), self._json(payload, {}), self._json(parents, []),
                     self._json(value.get("refs"), {}), self._json(value.get("metadata"), {}), payload.get("status"),
                     payload.get("tool_name") or payload.get("result_tool_name") or payload.get("invoked_tool"),
                     payload.get("command"), payload.get("preview"), payload.get("error_signature"), payload.get("text")),
                )
                self.conn.execute("DELETE FROM event_edges WHERE trajectory_id=? AND child_event_id=?", (ids[0], ids[1]))
                self.conn.executemany("INSERT OR IGNORE INTO event_edges VALUES(?,?,?)", [(ids[0], parent, ids[1]) for parent in parents if parent])
            elif kind == "branch":
                self.conn.execute(
                    """INSERT INTO branches VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(trajectory_id,branch_id) DO UPDATE SET base_event_id=excluded.base_event_id,
                    head_event_id=excluded.head_event_id,snapshot_id=excluded.snapshot_id,created_at=excluded.created_at,
                    metadata_json=excluded.metadata_json""",
                    (ids[0], ids[1], value.get("base_event_id"), value.get("head_event_id"), value.get("snapshot_id"),
                     value.get("created_at", ""), self._json(value.get("metadata"), {})),
                )
            elif kind == "snapshot":
                self.conn.execute(
                    """INSERT INTO snapshots VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(snapshot_id) DO UPDATE SET trajectory_id=excluded.trajectory_id,branch_id=excluded.branch_id,
                    event_id=excluded.event_id,message=excluded.message,created_at=excluded.created_at,metadata_json=excluded.metadata_json""",
                    (ids[1], ids[0], value.get("branch_id", "main"), value.get("event_id"), value.get("message", ""),
                     value.get("created_at", ""), self._json(value.get("metadata"), {})),
                )
            elif kind == "view":
                self.conn.execute(
                    """INSERT INTO materialized_views(object_key,trajectory_id,branch_id,view_name,content_json,source_events_json,created_at,metadata_json,event_count)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(object_key) DO UPDATE SET trajectory_id=excluded.trajectory_id,branch_id=excluded.branch_id,
                    view_name=excluded.view_name,content_json=excluded.content_json,source_events_json=excluded.source_events_json,
                    created_at=excluded.created_at,metadata_json=excluded.metadata_json,event_count=excluded.event_count""",
                    (key, ids[0], value.get("branch_id", ids[1]), value.get("view_name", self._parts(key)[-1]),
                     self._json(value.get("content"), {}), self._json(value.get("source_events"), []), value.get("created_at", ""),
                     self._json(value.get("metadata"), {}), (value.get("metadata") or {}).get("event_count")),
                )
            elif kind == "artifact":
                self.conn.execute(
                    """INSERT INTO artifacts VALUES(?,?,?,?,?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET
                    trajectory_id=excluded.trajectory_id,kind=excluded.kind,content_ref=excluded.content_ref,preview=excluded.preview,
                    created_at=excluded.created_at,metadata_json=excluded.metadata_json""",
                    (ids[1], ids[0], value.get("kind", ""), value.get("content_ref", ""), value.get("preview", ""),
                     value.get("created_at", ""), self._json(value.get("metadata"), {})),
                )
            elif kind == "hook_session":
                self.conn.execute("""INSERT INTO hook_sessions(source,session_id,value_json,updated_at) VALUES(?,?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(source,session_id) DO UPDATE SET value_json=excluded.value_json,updated_at=CURRENT_TIMESTAMP""", (ids[0], ids[1], self._json(value, {})))
            else:
                self.conn.execute("""INSERT INTO documents(object_key,value_json,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(object_key) DO UPDATE SET value_json=excluded.value_json,updated_at=CURRENT_TIMESTAMP""", (key, self._json(value, {})))
            self.conn.commit()

    def get_object(self, key: str) -> Optional[Dict[str, Any]]:
        key = key.strip("/")
        kind, ids = self._kind_for_key(key)
        if kind == "trajectory": return self._trajectory(self.conn.execute("SELECT * FROM trajectories WHERE trajectory_id=?", ids).fetchone())
        if kind == "event": return self._event(self.conn.execute("SELECT * FROM events WHERE trajectory_id=? AND event_id=?", ids).fetchone())
        if kind == "branch": return self._branch(self.conn.execute("SELECT * FROM branches WHERE trajectory_id=? AND branch_id=?", ids).fetchone())
        if kind == "snapshot": return self._snapshot(self.conn.execute("SELECT * FROM snapshots WHERE trajectory_id=? AND snapshot_id=?", ids).fetchone())
        if kind == "view": return self._view(self.conn.execute("SELECT * FROM materialized_views WHERE object_key=?", (key,)).fetchone())
        if kind == "artifact": return self._artifact(self.conn.execute("SELECT * FROM artifacts WHERE trajectory_id=? AND artifact_id=?", ids).fetchone())
        if kind == "hook_session":
            row = self.conn.execute("SELECT value_json FROM hook_sessions WHERE source=? AND session_id=?", ids).fetchone()
            return self._decode(row["value_json"], {}) if row else None
        row = self.conn.execute("SELECT value_json FROM documents WHERE object_key=?", (key,)).fetchone()
        return self._decode(row["value_json"], {}) if row else None

    def list_objects(self, prefix: str) -> List[str]:
        prefix = prefix.strip("/")
        keys: List[str] = []
        keys.extend(row[0] for row in self.conn.execute("SELECT object_key FROM materialized_views"))
        keys.extend(row[0] for row in self.conn.execute("SELECT object_key FROM documents"))
        keys.extend(f"trajectories/{r['trajectory_id']}/meta" for r in self.conn.execute("SELECT trajectory_id FROM trajectories"))
        keys.extend(f"trajectories/{r['trajectory_id']}/events/{r['event_id']}" for r in self.conn.execute("SELECT trajectory_id,event_id FROM events"))
        keys.extend(f"trajectories/{r['trajectory_id']}/branches/{r['branch_id']}" for r in self.conn.execute("SELECT trajectory_id,branch_id FROM branches"))
        keys.extend(f"trajectories/{r['trajectory_id']}/snapshots/{r['snapshot_id']}" for r in self.conn.execute("SELECT trajectory_id,snapshot_id FROM snapshots"))
        keys.extend(f"trajectories/{r['trajectory_id']}/artifacts/{r['artifact_id']}" for r in self.conn.execute("SELECT trajectory_id,artifact_id FROM artifacts"))
        keys.extend(f"hook_sessions/{r['source']}/{r['session_id']}" for r in self.conn.execute("SELECT source,session_id FROM hook_sessions"))
        return sorted(key for key in set(keys) if key == prefix or key.startswith(prefix.rstrip("/") + "/"))

    def scan_prefix(self, prefix: str) -> Iterable[Dict[str, Any]]:
        for key in self.list_objects(prefix):
            value = self.get_object(key)
            if value is not None: yield value

    def delete_object(self, key: str) -> None:
        key = key.strip("/")
        kind, ids = self._kind_for_key(key)
        with self._lock:
            if kind == "trajectory": self.conn.execute("DELETE FROM trajectories WHERE trajectory_id=?", ids)
            elif kind == "event": self.conn.execute("DELETE FROM events WHERE trajectory_id=? AND event_id=?", ids)
            elif kind == "branch": self.conn.execute("DELETE FROM branches WHERE trajectory_id=? AND branch_id=?", ids)
            elif kind == "snapshot": self.conn.execute("DELETE FROM snapshots WHERE trajectory_id=? AND snapshot_id=?", ids)
            elif kind == "view": self.conn.execute("DELETE FROM materialized_views WHERE object_key=?", (key,))
            elif kind == "artifact": self.conn.execute("DELETE FROM artifacts WHERE trajectory_id=? AND artifact_id=?", ids)
            elif kind == "hook_session": self.conn.execute("DELETE FROM hook_sessions WHERE source=? AND session_id=?", ids)
            else: self.conn.execute("DELETE FROM documents WHERE object_key=?", (key,))
            self.conn.commit()

    def snapshot_objects(self, snapshot_id: str, keys: List[str]) -> Path:
        """Store snapshot state in a table rather than copying entity JSON files."""
        with self._lock:
            self.conn.execute("DELETE FROM snapshot_records WHERE snapshot_id=?", (snapshot_id,))
            rows = [(snapshot_id, key, self._json(value, {})) for key in keys if (value := self.get_object(key)) is not None]
            self.conn.executemany("INSERT INTO snapshot_records VALUES(?,?,?)", rows)
            self.conn.commit()
        return self.db_path

    def restore_snapshot_files(self, snapshot_id: str, key_prefix: str) -> None:
        """Compatibility restore using relational snapshot records; no files are read."""
        prefix = key_prefix.strip("/")
        rows = self.conn.execute("SELECT object_key,value_json FROM snapshot_records WHERE snapshot_id=? AND object_key LIKE ? ORDER BY object_key", (snapshot_id, prefix.rstrip("/") + "/%")).fetchall()
        if not rows: raise KeyError(f"snapshot records not found: {snapshot_id}")
        for row in rows: self.put_object(row["object_key"], self._decode(row["value_json"], {}))

    def _migrate_legacy_objects(self) -> None:
        """One-time import of old object files. New reads/writes are SQLite-only."""
        legacy_snapshots: List[Tuple[str, List[str]]] = []
        for row in self.conn.execute("SELECT key,path FROM objects ORDER BY key").fetchall():
            key, path = row["key"], Path(row["path"])
            if self.get_object(key) is not None or not path.is_file(): continue
            try: value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): continue
            if not isinstance(value, dict): continue
            if self._kind_for_key(key)[0] == "snapshot":
                legacy_snapshots.append((str(value.get("snapshot_id") or ""), list(value.get("object_keys") or [])))
            self.put_object(key, value)
        for snapshot_id, keys in legacy_snapshots:
            if snapshot_id and keys:
                existing = self.conn.execute("SELECT 1 FROM snapshot_records WHERE snapshot_id=? LIMIT 1", (snapshot_id,)).fetchone()
                if not existing:
                    self.snapshot_objects(snapshot_id, keys)

    def _trajectory(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        return {"trajectory_id":row["trajectory_id"],"title":row["title"],"agent_id":row["agent_id"],"source_id":row["source_id"],"default_branch":row["default_branch"],"head_event_id":row["head_event_id"],"created_at":row["created_at"],"updated_at":row["updated_at"],"metadata":self._decode(row["metadata_json"],{})}

    def _event(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        return {"event_id":row["event_id"],"trajectory_id":row["trajectory_id"],"branch_id":row["branch_id"],"event_type":row["event_type"],"actor":row["actor"],"timestamp":row["timestamp"],"payload":self._decode(row["payload_json"],{}),"parent_event_ids":self._decode(row["parent_event_ids_json"],[]),"refs":self._decode(row["refs_json"],{}),"metadata":self._decode(row["metadata_json"],{})}

    def _branch(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        return {"branch_id":row["branch_id"],"trajectory_id":row["trajectory_id"],"base_event_id":row["base_event_id"],"head_event_id":row["head_event_id"],"snapshot_id":row["snapshot_id"],"created_at":row["created_at"],"metadata":self._decode(row["metadata_json"],{})}

    def _snapshot(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        keys=[x["object_key"] for x in self.conn.execute("SELECT object_key FROM snapshot_records WHERE snapshot_id=? ORDER BY object_key",(row["snapshot_id"],))]
        return {"snapshot_id":row["snapshot_id"],"trajectory_id":row["trajectory_id"],"branch_id":row["branch_id"],"event_id":row["event_id"],"message":row["message"],"object_keys":keys,"created_at":row["created_at"],"metadata":self._decode(row["metadata_json"],{})}

    def _view(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        return {"view_name":row["view_name"],"trajectory_id":row["trajectory_id"],"branch_id":row["branch_id"],"content":self._decode(row["content_json"],{}),"source_events":self._decode(row["source_events_json"],[]),"created_at":row["created_at"],"metadata":self._decode(row["metadata_json"],{})}

    def _artifact(self, row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if not row: return None
        return {"artifact_id":row["artifact_id"],"trajectory_id":row["trajectory_id"],"kind":row["kind"],"content_ref":row["content_ref"],"preview":row["preview"],"created_at":row["created_at"],"metadata":self._decode(row["metadata_json"],{})}
