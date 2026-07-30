from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import request


class ContextDBClient:
    """Small standard-library client for live Agent integration.

    Agents can use this client to create trajectories and append events while they run.
    It intentionally mirrors the HTTP API and has no third-party dependency.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        self.base_url = base_url.rstrip("/")

    def create_trajectory(self, title: str, agent_id: str = "external-agent", source_id: str = "live-agent", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._post("/api/v1/trajectories", {"title": title, "agent_id": agent_id, "source_id": source_id, "metadata": metadata or {}})

    def append_event(self, trajectory_id: str, event_type: str, payload: Dict[str, Any], branch_id: str = "main", actor: str = "agent", refs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._post("/api/v1/events", {"trajectory_id": trajectory_id, "event_type": event_type, "payload": payload, "branch_id": branch_id, "actor": actor, "refs": refs or {}, "metadata": metadata or {}})

    def log_user(self, trajectory_id: str, text: str, branch_id: str = "main") -> Dict[str, Any]:
        return self.append_event(trajectory_id, "user_message", {"text": text}, branch_id=branch_id, actor="user")

    def log_assistant(self, trajectory_id: str, text: str, branch_id: str = "main") -> Dict[str, Any]:
        return self.append_event(trajectory_id, "assistant_message", {"text": text}, branch_id=branch_id, actor="agent")

    def log_tool_call(self, trajectory_id: str, tool_name: str, command: Any, branch_id: str = "main") -> Dict[str, Any]:
        return self.append_event(trajectory_id, "tool_call", {"tool_name": tool_name, "command": command}, branch_id=branch_id, actor="agent")

    def log_tool_result(self, trajectory_id: str, status: str, preview: str = "", exit_code: Optional[int] = None, branch_id: str = "main") -> Dict[str, Any]:
        return self.append_event(trajectory_id, "tool_result", {"status": status, "preview": preview, "exit_code": exit_code}, branch_id=branch_id, actor="tool")

    def query_view(self, trajectory_id: str, view_name: str, branch_id: str = "main", token_budget: int = 4000) -> Dict[str, Any]:
        return self._post("/api/v1/query_view", {"trajectory_id": trajectory_id, "view_name": view_name, "branch_id": branch_id, "token_budget": token_budget})

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
