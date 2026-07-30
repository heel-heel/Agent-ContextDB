from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base import NormalizedEvent


TYPE_MAP = {
    "user": "user_message",
    "assistant": "assistant_message",
    "message": "assistant_message",
    "tool": "tool_call",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "result": "tool_result",
    "memory": "memory_update",
    "memory_update": "memory_update",
    "summary": "summary_update",
    "summary_update": "summary_update",
    "file_read": "file_read",
    "file_edit": "file_edit",
    "artifact": "artifact",
    "error": "error",
    "system": "system_event",
    "system_event": "system_event",
}


class GenericJSONLAdapter:
    """Adapter for framework-neutral agent traces stored as JSON Lines.

    Each line can either use ContextDB-native fields (`event_type`, `payload`,
    `actor`, ...), or a compact trace format such as:
      {"type":"tool_call", "tool":"shell", "command":"pytest"}
      {"type":"tool_result", "status":"failed", "output":"..."}
    """

    source_name = "generic-jsonl"

    def load(self, path: str | Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {line_no} must be a JSON object")
            item.setdefault("_line_no", line_no)
            records.append(item)
        return records

    def iter_events(self, raw_trace: List[Dict[str, Any]]) -> Iterable[NormalizedEvent]:
        for item in raw_trace:
            yield self._normalize(item)

    def _normalize(self, item: Dict[str, Any]) -> NormalizedEvent:
        raw_type = item.get("event_type") or item.get("type") or item.get("role") or "message"
        event_type = TYPE_MAP.get(str(raw_type), str(raw_type))
        actor = item.get("actor") or self._infer_actor(event_type, item)
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else self._payload_for(event_type, item)
        refs = dict(item.get("refs") or {})
        refs.setdefault("source_line", item.get("_line_no"))
        if item.get("raw_event_id"):
            refs.setdefault("raw_event_id", item["raw_event_id"])
        metadata = dict(item.get("metadata") or {})
        for key in ("source", "session_id", "conversation_id", "model", "cwd"):
            if key in item:
                metadata.setdefault(key, item[key])
        return NormalizedEvent(
            event_type=event_type,
            payload=payload,
            actor=actor,
            branch_id=item.get("branch_id", "main"),
            parent_event_ids=item.get("parent_event_ids"),
            refs=refs,
            metadata=metadata,
            timestamp=item.get("timestamp"),
        )

    def _infer_actor(self, event_type: str, item: Dict[str, Any]) -> str:
        if event_type == "user_message":
            return "user"
        if event_type == "tool_result":
            return "tool"
        if item.get("role") in {"user", "assistant", "system", "tool"}:
            return item["role"]
        return "agent"

    def _payload_for(self, event_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
        if event_type in {"user_message", "assistant_message", "summary_update", "system_event"}:
            return {"text": item.get("text") or item.get("content") or item.get("message", "")}
        if event_type == "tool_call":
            return {
                "tool_name": item.get("tool_name") or item.get("tool") or item.get("name", "unknown-tool"),
                "command": item.get("command") or item.get("input") or item.get("args") or "",
            }
        if event_type == "tool_result":
            output = item.get("preview") or item.get("output") or item.get("stderr") or item.get("stdout") or ""
            return {
                "status": item.get("status") or self._status_from_exit_code(item.get("exit_code")),
                "exit_code": item.get("exit_code"),
                "preview": str(output)[:500],
            }
        if event_type == "memory_update":
            return {"fact": item.get("fact") or item.get("text") or item.get("content", "")}
        if event_type in {"file_read", "file_edit", "artifact"}:
            return {
                "path": item.get("path"),
                "summary": item.get("summary") or item.get("text") or item.get("content", ""),
                "diff": item.get("diff"),
            }
        if event_type == "error":
            return {"message": item.get("message") or item.get("error") or item.get("text", ""), "code": item.get("code")}
        return {k: v for k, v in item.items() if not k.startswith("_") and k not in {"event_type", "type", "role", "actor", "branch_id", "parent_event_ids", "refs", "metadata", "timestamp"}}

    def _status_from_exit_code(self, exit_code: Any) -> str:
        if exit_code is None:
            return "ok"
        return "ok" if exit_code == 0 else "failed"
