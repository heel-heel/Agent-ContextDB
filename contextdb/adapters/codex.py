## This adapter is retained for offline Codex JSONL imports and is not used by the current live Codex session watcher.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base import NormalizedEvent


class CodexJSONLAdapter:
    """Normalize common Codex session JSONL envelopes into ContextDB events."""

    source_name = "codex-jsonl"

    def load(self, path: str | Path) -> List[Dict[str, Any]]:
        records = []
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    item.setdefault("_line_no", line_no)
                    records.append(item)
        return records

    def iter_events(self, raw_trace: List[Dict[str, Any]]) -> Iterable[NormalizedEvent]:
        for record in raw_trace:
            yield from self._normalize(record)

    def _normalize(self, record: Dict[str, Any]) -> Iterable[NormalizedEvent]:
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
        kind = str(payload.get("type") or record.get("type") or "").lower()
        timestamp = record.get("timestamp") or payload.get("timestamp")
        refs = {"source_line": record.get("_line_no")}
        metadata = {"codex_record_type": kind, "source": self.source_name}
        role = str(payload.get("role") or record.get("role") or "").lower()
        text = self._text(payload.get("content") or payload.get("text") or payload.get("message"))
        if role == "user" or kind in {"user_message", "input_message"}:
            yield NormalizedEvent("user_message", {"text": text}, actor="user", refs=refs, metadata=metadata, timestamp=timestamp)
        elif role == "assistant" or kind in {"assistant_message", "message", "output_text"}:
            yield NormalizedEvent("assistant_message", {"text": text}, actor="agent", refs=refs, metadata=metadata, timestamp=timestamp)
        elif kind in {"function_call", "tool_call", "command_execution", "shell_call"}:
            command = payload.get("arguments") or payload.get("input") or payload.get("command") or text
            yield NormalizedEvent("tool_call", {"tool_name": payload.get("name") or payload.get("tool_name") or "shell", "command": command}, actor="agent", refs=refs, metadata=metadata, timestamp=timestamp)
        elif kind in {"function_call_output", "tool_result", "command_execution_result", "shell_result"}:
            output = self._text(payload.get("output") or payload.get("content") or payload.get("result") or text)
            failed = bool(payload.get("is_error")) or str(payload.get("status") or "").lower() in {"failed", "error", "timeout"}
            yield NormalizedEvent("tool_result", {"status": "failed" if failed else "ok", "preview": output[:500], "exit_code": payload.get("exit_code")}, actor="tool", refs=refs, metadata=metadata, timestamp=timestamp)
        elif kind in {"file_change", "file_edit"}:
            yield NormalizedEvent("file_edit", {"path": payload.get("path"), "summary": text, "diff": payload.get("diff")}, actor="agent", refs=refs, metadata=metadata, timestamp=timestamp)
        else:
            yield NormalizedEvent("system_event", {"text": text or json.dumps(payload, ensure_ascii=False)[:1000]}, actor="system", refs=refs, metadata=metadata, timestamp=timestamp)

    def _text(self, value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(self._text(item.get("text") if isinstance(item, dict) else item) for item in value)
        if isinstance(value, dict):
            return str(value.get("text") or value.get("content") or value.get("output") or "")
        return str(value or "")
