from __future__ import annotations

"""A dependency-free stdio MCP bridge for ContextDB live skill reuse.

The server deliberately separates three facts that are often conflated in
agent demos: a skill was matched, ContextDB delivered it to an Agent, and the
Agent chose and executed an action.  It never executes a recommended command.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict

from .hooks import HookSessionBridge
from .service import ContextDB


SERVER_INFO = {"name": "contextdb", "version": "0.1.0"}


def _tool(name: str, description: str, properties: Dict[str, Any], required: list[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }


TOOLS = [
    _tool(
        "contextdb_record_tool_call",
        "Record a real tool call before it runs. For potentially state-changing calls, ContextDB creates a logical snapshot automatically. This never snapshots or modifies the Agent workspace itself.",
        {
            "source": {"type": "string"}, "session_id": {"type": "string"},
            "tool_name": {"type": "string"}, "command": {"type": "string"},
            "tool_call_id": {"type": "string", "description": "Stable id reused by contextdb_record_tool_result."},
        },
        ["source", "session_id", "tool_name", "command"],
    ),
    _tool(
        "contextdb_prepare_context",
        "Call before planning a turn or retry. Returns compact trajectory context and delivers the latest ContextDB skill recommendation into this tool result. A recommendation is advisory and must not be executed automatically.",
        {
            "source": {"type": "string", "description": "Agent source, for example codex or claude-code."},
            "session_id": {"type": "string", "description": "Stable identifier for this live Agent session."},
            "token_budget": {"type": "integer", "minimum": 200, "maximum": 8000, "default": 1200},
        },
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_record_tool_result",
        "Record a tool execution that actually occurred. If the status is failed, error, or timeout, ContextDB automatically retrieves a matching skill and returns the next recommendation in this same tool response. Do not claim a command ran unless it really ran.",
        {
            "source": {"type": "string"}, "session_id": {"type": "string"},
            "tool_name": {"type": "string"}, "command": {"type": "string"},
            "status": {"type": "string", "enum": ["ok", "failed", "error", "timeout"]},
            "preview": {"type": "string", "description": "Observed command output or concise failure text."},
            "exit_code": {"type": ["integer", "null"]},
            "tool_call_id": {"type": "string", "description": "Id supplied to contextdb_record_tool_call, when available."},
        },
        ["source", "session_id", "tool_name", "command", "status"],
    ),
    _tool(
        "contextdb_get_recommendation",
        "Deliver the latest skill recommendation after a previously recorded failure. Use this if the Agent did not receive the result inline or is resuming a session.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}},
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_record_skill_decision",
        "Record whether the Agent accepted, rejected, or deferred ContextDB's delivered recommendation. Call this before an accepted skill-guided action, or explain why it was rejected.",
        {
            "source": {"type": "string"}, "session_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["accepted", "rejected", "deferred"]},
            "reason": {"type": "string"}, "skill_match_event_id": {"type": "string"},
            "skill_id": {"type": "string"}, "action_id": {"type": "string"},
        },
        ["source", "session_id", "decision"],
    ),
    _tool(
        "contextdb_record_skill_application",
        "Record the result of an accepted skill-guided action that the Agent actually executed through its normal tool system. This records evidence only; ContextDB does not execute the command itself.",
        {
            "source": {"type": "string"}, "session_id": {"type": "string"},
            "tool_name": {"type": "string"}, "command": {"type": "string"},
            "status": {"type": "string", "enum": ["ok", "failed", "error", "timeout"]},
            "preview": {"type": "string"}, "exit_code": {"type": ["integer", "null"]},
            "skill_match_event_id": {"type": "string"}, "skill_id": {"type": "string"}, "action_id": {"type": "string"},
            "tool_call_id": {"type": "string", "description": "Optional id from contextdb_record_tool_call."},
            "tool_call_event_id": {"type": "string", "description": "Optional ContextDB tool_call event id to reuse."},
        },
        ["source", "session_id", "tool_name", "command", "status"],
    ),
    _tool(
        "contextdb_create_snapshot",
        "Create an explicit logical ContextDB snapshot for the active trajectory branch. It records context state only and never restores or changes workspace files.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "message": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_create_repair_branch",
        "Accept a repair-branch suggestion or create a named logical repair branch from a ContextDB snapshot. The Agent remains responsible for any Git or workspace operation.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "branch_id": {"type": "string"}, "snapshot_id": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_rollback_context",
        "Create a logical rollback branch from a ContextDB snapshot after an explicit Agent decision. It never restores local files automatically.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "snapshot_id": {"type": "string"}, "target_branch_id": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id", "snapshot_id"],
    ),
    _tool(
        "contextdb_record_version_decision",
        "Record an Agent decision to continue, create a repair branch, or roll back logically. Use create-repair-branch or rollback-context to perform the accepted logical transition.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "action": {"type": "string", "enum": ["continue_current_branch", "create_repair_branch", "rollback_context"]}, "decision": {"type": "string", "enum": ["accepted", "rejected", "deferred"]}, "reason": {"type": "string"}, "suggestion_event_id": {"type": "string"}},
        ["source", "session_id", "action", "decision"],
    ),
    _tool(
        "contextdb_get_version_status",
        "Return active branch, snapshots, repair suggestion, and the logical-only workspace-restore boundary for a live Agent session.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}},
        ["source", "session_id"],
    ),
]


class ContextDBMCPServer:
    def __init__(self, root: str):
        self.db = ContextDB(root)
        self.bridge = HookSessionBridge(self.db)

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "contextdb_prepare_context": self._prepare_context,
            "contextdb_record_tool_call": self._record_tool_call,
            "contextdb_record_tool_result": self._record_tool_result,
            "contextdb_get_recommendation": self._get_recommendation,
            "contextdb_record_skill_decision": self._record_skill_decision,
            "contextdb_record_skill_application": self._record_skill_application,
            "contextdb_create_snapshot": self._create_snapshot,
            "contextdb_create_repair_branch": self._create_repair_branch,
            "contextdb_rollback_context": self._rollback_context,
            "contextdb_record_version_decision": self._record_version_decision,
            "contextdb_get_version_status": self._get_version_status,
        }
        if name not in handlers:
            raise ValueError("unknown ContextDB MCP tool: %s" % name)
        return handlers[name](arguments)

    def _prepare_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.prepare_context(args["source"], args["session_id"], int(args.get("token_budget", 1200)))

    def _record_tool_call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.bridge.ensure_session(args["source"], args["session_id"])
        return self.bridge.ingest({
            "source": args["source"], "session_id": args["session_id"], "adapter": "generic",
            "event": {
                "event_type": "tool_call", "actor": "agent", "event_id": args.get("tool_call_id"),
                "payload": {"tool_name": args["tool_name"], "command": args["command"]},
                "metadata": {"mcp_session": True},
            },
        })

    def _record_tool_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.bridge.ensure_session(args["source"], args["session_id"])
        session = self.bridge.status(args["source"], args["session_id"])["session"]
        response = self.bridge.ingest({
            "source": args["source"], "session_id": args["session_id"],
            "adapter": "generic",
            "event": {
                "event_type": "tool_result", "actor": "tool", "event_id": args.get("tool_call_id"),
                "payload": {
                    "tool_name": args["tool_name"], "command": args["command"], "status": args["status"],
                    "preview": args.get("preview", ""), "exit_code": args.get("exit_code"),
                },
                "metadata": {"mcp_session": True, "trajectory_id": session["trajectory_id"]},
            },
        })
        return response

    def _get_recommendation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.prepare_context(args["source"], args["session_id"], 1200)

    def _record_skill_decision(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.record_skill_decision(
            args["source"], args["session_id"], args["decision"], args.get("reason", ""),
            args.get("skill_match_event_id"), args.get("skill_id"), args.get("action_id"),
        )

    def _record_skill_application(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.record_skill_application(
            args["source"], args["session_id"], args["tool_name"], args["command"], args["status"],
            args.get("preview", ""), args.get("exit_code"), args.get("skill_match_event_id"),
            args.get("skill_id"), args.get("action_id"), args.get("tool_call_id"), args.get("tool_call_event_id"),
        )

    def _create_snapshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.create_snapshot(args["source"], args["session_id"], args.get("message", ""), args.get("reason", ""))

    def _create_repair_branch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.create_repair_branch(args["source"], args["session_id"], args.get("branch_id"), args.get("snapshot_id"), args.get("reason", ""))

    def _rollback_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.rollback_context(args["source"], args["session_id"], args["snapshot_id"], args.get("target_branch_id"), args.get("reason", ""))

    def _record_version_decision(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.record_version_decision(args["source"], args["session_id"], args["action"], args["decision"], args.get("reason", ""), args.get("suggestion_event_id"))

    def _get_version_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.version_status(args["source"], args["session_id"])


def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(payload: Dict[str, Any], is_error: bool = False) -> Dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return {"content": [{"type": "text", "text": text}], "structuredContent": payload, "isError": is_error}


def serve_stdio(root: str) -> int:
    """Serve JSON-RPC messages as newline-delimited MCP stdio transport."""
    server = ContextDBMCPServer(root)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("JSON-RPC message must be an object")
            request_id = message.get("id")
            method = message.get("method")
            params = message.get("params") or {}
            if method == "initialize":
                response = _result(request_id, {
                    "protocolVersion": params.get("protocolVersion", "2025-03-26"),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": "Call contextdb_prepare_context before planning and contextdb_record_tool_result after every real tool execution. Never execute a recommendation automatically.",
                })
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                response = _result(request_id, {"tools": TOOLS})
            elif method == "tools/call":
                try:
                    payload = server.dispatch(str(params.get("name") or ""), dict(params.get("arguments") or {}))
                    response = _result(request_id, _tool_result(payload))
                except Exception as exc:
                    response = _result(request_id, _tool_result({"error": str(exc)}, is_error=True))
            elif request_id is not None:
                response = _error(request_id, -32601, "method not found: %s" % method)
            else:
                continue
        except Exception as exc:
            response = _error(locals().get("request_id"), -32700, str(exc))
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="contextdb-mcp")
    parser.add_argument("--root", default=os.environ.get("CONTEXTDB_DATA_ROOT", "data"))
    args = parser.parse_args()
    raise SystemExit(serve_stdio(args.root))


if __name__ == "__main__":
    main()
