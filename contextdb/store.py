from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol
import threading


class ContextStore(Protocol):
    def put_object(self, key: str, value: Dict[str, Any]) -> None: ...
    def get_object(self, key: str) -> Optional[Dict[str, Any]]: ...
    def list_objects(self, prefix: str) -> List[str]: ...
    def delete_object(self, key: str) -> None: ...
    def create_namespace(self, prefix: str) -> None: ...
    def scan_prefix(self, prefix: str) -> Iterable[Dict[str, Any]]: ...


class SQLiteStore:
    """A small ContextStore backend: SQLite metadata + JSON object files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_dir = self.root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "contextdb.sqlite3"
        #self.conn = sqlite3.connect(self.db_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Dashboard, MCP, and a hook watcher can be separate local processes.
        # WAL plus a bounded wait avoids transient write-lock failures in that
        # normal live-integration setup.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects (
              key TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              trajectory_id TEXT,
              branch_id TEXT,
              event_type TEXT,
              timestamp TEXT,
              path TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_objects_prefix ON objects(key);
            CREATE INDEX IF NOT EXISTS idx_objects_traj ON objects(trajectory_id);
            CREATE INDEX IF NOT EXISTS idx_objects_branch ON objects(branch_id);
            CREATE INDEX IF NOT EXISTS idx_objects_event_type ON objects(event_type);
            CREATE INDEX IF NOT EXISTS idx_objects_timestamp ON objects(timestamp);
            """
        )
        self.conn.commit()

    def _path_for_key(self, key: str) -> Path:
        return self.objects_dir / (key.strip("/") + ".json")

    def create_namespace(self, prefix: str) -> None:
        (self.objects_dir / prefix.strip("/")).mkdir(parents=True, exist_ok=True)

    def put_object(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        obj_type = key.strip("/").split("/")[-2] if "/" in key.strip("/") else "object"
        self.conn.execute(
            """
            INSERT INTO objects(key,type,trajectory_id,branch_id,event_type,timestamp,path)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
              type=excluded.type, trajectory_id=excluded.trajectory_id,
              branch_id=excluded.branch_id, event_type=excluded.event_type,
              timestamp=excluded.timestamp, path=excluded.path
            """,
            (
                key,
                obj_type,
                value.get("trajectory_id"),
                value.get("branch_id"),
                value.get("event_type"),
                value.get("timestamp") or value.get("created_at"),
                str(path),
            ),
        )
        self.conn.commit()

    def get_object(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_objects(self, prefix: str) -> List[str]:
        like = prefix.rstrip("/") + "/%"
        rows = self.conn.execute("SELECT key FROM objects WHERE key LIKE ? ORDER BY key", (like,)).fetchall()
        return [r["key"] for r in rows]

    def scan_prefix(self, prefix: str) -> Iterable[Dict[str, Any]]:
        for key in self.list_objects(prefix):
            obj = self.get_object(key)
            if obj is not None:
                yield obj

    def delete_object(self, key: str) -> None:
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()
        self.conn.execute("DELETE FROM objects WHERE key=?", (key,))
        self.conn.commit()

    def snapshot_objects(self, snapshot_id: str, keys: List[str]) -> Path:
        target = self.snapshots_dir / snapshot_id
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for key in keys:
            src = self._path_for_key(key)
            if src.exists():
                dst = target / (key.strip("/") + ".json")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        return target

    def restore_snapshot_files(self, snapshot_id: str, key_prefix: str) -> None:
        snap = self.snapshots_dir / snapshot_id
        if not snap.exists():
            raise KeyError(f"snapshot files not found: {snapshot_id}")
        prefix_path = self.objects_dir / key_prefix.strip("/")
        if prefix_path.exists():
            shutil.rmtree(prefix_path)
        source_prefix = snap / key_prefix.strip("/")
        if source_prefix.exists():
            shutil.copytree(source_prefix, prefix_path)
