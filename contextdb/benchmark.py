from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .agent_import import replay_trace
from .service import ContextDB, FAILURE_STATUSES


class BenchmarkRunner:
    """Leakage-aware offline-history / held-out-online evaluation runner."""

    def __init__(self, root: str):
        self.root, self.db = root, ContextDB(root)

    def run(self, manifest_path: str | Path) -> Dict[str, Any]:
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        base = manifest_file.parent
        histories = [self._replay(base, item, "history") for item in manifest.get("history", [])]
        materialized = []
        for row in histories:
            view = self.db.query_view(row["trajectory_id"], "learned_skills", "main")
            materialized.append({"trajectory_id": row["trajectory_id"], "skill_count": len(view.get("content", []))})
        evaluations = [self._replay(base, item, "evaluation") for item in manifest.get("evaluation", [])]
        queries = []
        for row in evaluations:
            for event in self.db.list_events(row["trajectory_id"], "main"):
                if event.get("event_type") != "tool_result" or event.get("payload", {}).get("status") not in FAILURE_STATUSES:
                    continue
                previous = self._previous_tool_call(row["trajectory_id"], event["event_id"])
                failure = {
                    "tool": (previous or {}).get("payload", {}).get("tool_name", "shell"),
                    "command": (previous or {}).get("payload", {}).get("command", ""),
                    "error_signature": event.get("payload", {}).get("error_signature") or event.get("payload", {}).get("preview", ""),
                }
                match = self.db.match_skill(row["trajectory_id"], failure, top_k=1)
                expected = (event.get("metadata", {}) or {}).get("benchmark_expected_skill")
                actual = (match.get("matches") or [{}])[0].get("matched_skill_id")
                queries.append({"trajectory_id": row["trajectory_id"], "failure_event_id": event["event_id"], "expected_skill": expected, "matched_skill": actual, "score": (match.get("matches") or [{}])[0].get("score", 0.0), "hit": bool(actual) and (not expected or expected in actual)})
        hits = sum(1 for query in queries if query["hit"])
        return {
            "schema_version": "contextdb_benchmark.v1", "name": manifest.get("name", manifest_file.stem),
            "history": histories, "materialized": materialized, "evaluation": evaluations, "queries": queries,
            "metrics": {"history_trajectories": len(histories), "evaluation_trajectories": len(evaluations),
                        "materialized_skills": sum(item["skill_count"] for item in materialized),
                        "failure_queries": len(queries), "top1_retrieval_hits": hits,
                        "top1_retrieval_accuracy": round(hits / len(queries), 4) if queries else 0.0,
                        "vector_entries": self.db.vector_index.count("skills")},
            "protocol": "Build skills from history only; hold evaluation trajectories out of skill materialization; retrieve before any evaluation-side skill extraction.",
        }

    def _replay(self, base: Path, item: Dict[str, Any], split: str) -> Dict[str, Any]:
        path = base / item["path"]
        result = replay_trace(path, root=self.root, source=item.get("source", "generic-jsonl"), title=item.get("title") or f"{split}: {path.name}", agent_id=item.get("agent_id", "benchmark-agent"))
        return {"split": split, "path": str(path), "source": item.get("source", "generic-jsonl"), "trajectory_id": result["trajectory_id"], "event_count": result["event_count"]}

    def _previous_tool_call(self, trajectory_id: str, event_id: str) -> Dict[str, Any] | None:
        events = self.db.list_events(trajectory_id, "main")
        for index, event in enumerate(events):
            if event.get("event_id") == event_id:
                return next((candidate for candidate in reversed(events[:index]) if candidate.get("event_type") == "tool_call"), None)
        return None
