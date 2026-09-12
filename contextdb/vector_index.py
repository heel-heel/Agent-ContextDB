from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class HashEmbedding:
    """Deterministic local embedding used by the built-in vector store."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w.+/-]+", " ".join(str(text or "").lower().split()), flags=re.UNICODE)
        features = list(tokens)
        compact = " ".join(tokens)
        features.extend(compact[index:index + 3] for index in range(max(0, len(compact) - 2)))
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SQLiteVectorIndex:
    """Persistent vector index colocated with ContextDB's SQLite metadata."""

    def __init__(self, db_path: str | Path, dimensions: int = 384):
        self.db_path = Path(db_path)
        self.embedder = HashEmbedding(dimensions)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS vector_entries (
              collection_name TEXT NOT NULL, entry_id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL, document TEXT NOT NULL,
              vector_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_entries_collection ON vector_entries(collection_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_entries_owner ON vector_entries(owner_id)")
        self.conn.commit()

    @property
    def dimensions(self) -> int:
        return self.embedder.dimensions

    def replace_owner(self, collection: str, owner_id: str, entries: Iterable[Dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM vector_entries WHERE collection_name=? AND owner_id=?", (collection, owner_id))
        for entry in entries:
            document = str(entry.get("document") or "")
            self.conn.execute(
                "INSERT OR REPLACE INTO vector_entries(collection_name, entry_id, owner_id, document, vector_json, metadata_json) VALUES(?,?,?,?,?,?)",
                (collection, str(entry["entry_id"]), owner_id, document,
                 json.dumps(self.embedder.embed(document)),
                 json.dumps(entry.get("metadata") or {}, ensure_ascii=False)),
            )
        self.conn.commit()

    def search(self, collection: str, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed(query)
        rows = self.conn.execute(
            "SELECT entry_id, owner_id, document, vector_json, metadata_json FROM vector_entries WHERE collection_name=?",
            (collection,),
        ).fetchall()
        matches = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = sum(left * right for left, right in zip(query_vector, vector))
            if score >= min_score:
                matches.append({
                    "entry_id": row["entry_id"], "owner_id": row["owner_id"],
                    "document": row["document"], "score": round(float(score), 6),
                    "metadata": json.loads(row["metadata_json"]),
                })
        matches.sort(key=lambda item: (-item["score"], item["entry_id"]))
        return matches[:max(0, top_k)]

    def list_entries(self, collection: str) -> List[Dict[str, Any]]:
        """Return persisted entries for a collection without running a search."""
        rows = self.conn.execute(
            "SELECT entry_id, owner_id, document, metadata_json FROM vector_entries WHERE collection_name=? ORDER BY entry_id",
            (collection,),
        ).fetchall()
        return [
            {
                "entry_id": row["entry_id"],
                "owner_id": row["owner_id"],
                "document": row["document"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def count(self, collection: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS count FROM vector_entries" + (" WHERE collection_name=?" if collection else "")
        row = self.conn.execute(query, (collection,) if collection else ()).fetchone()
        return int(row["count"] if row else 0)
