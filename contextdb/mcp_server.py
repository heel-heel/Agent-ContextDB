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
        "Record a real tool call before it runs. For potentially state-changing calls, ContextDB creates a snapshot automatically. This never snapshots or modifies the Agent workspace itself.",
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
        "Create an explicit ContextDB snapshot for the active trajectory branch. It records context state only and never restores or changes workspace files.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "message": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_create_repair_branch",
        "Accept a repair-branch suggestion or create a named repair branch from a ContextDB snapshot. The Agent remains responsible for any Git or workspace operation.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "branch_id": {"type": "string"}, "snapshot_id": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id"],
    ),
    _tool(
        "contextdb_rollback_context",
        "Create a rollback branch from a ContextDB snapshot after an explicit Agent decision. It never restores local files automatically.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "snapshot_id": {"type": "string"}, "target_branch_id": {"type": "string"}, "reason": {"type": "string"}},
        ["source", "session_id", "snapshot_id"],
    ),
    _tool(
        "contextdb_record_version_decision",
        "Record an Agent decision to continue, create a repair branch, or roll back. Use create-repair-branch or rollback-context to perform the accepted transition.",
        {"source": {"type": "string"}, "session_id": {"type": "string"}, "action": {"type": "string", "enum": ["continue_current_branch", "create_repair_branch", "rollback_context"]}, "decision": {"type": "string", "enum": ["accepted", "rejected", "deferred"]}, "reason": {"type": "string"}, "suggestion_event_id": {"type": "string"}},
        ["source", "session_id", "action", "decision"],
    ),
    _tool(
        "contextdb_get_version_status",
        "Return active branch, snapshots, repair suggestion, and the workspace-restore boundary for a live Agent session.",
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
        """Deliver an auditable recommendation without returning turn context.

        This read path is commonly used after a failure when an Agent only
        needs the match and selected action identifiers.  Reusing
        ``prepare_context`` verbatim also returned a rendered trajectory
        digest, which is both unnecessary here and expensive for desktop MCP
        clients that deserialize text and structured content separately.
        """
        # A recommendation lookup must not materialize the optional long-term
        # semantic digest. That LLM-backed work belongs to prepare_context and
        # can outlive an otherwise immediate skill lookup.
        prepared = self.bridge.prepare_context(
            args["source"], args["session_id"], 1200, include_prompt_context=False,
        )
        recommendation = prepared.get("agent_context") or {}
        action = recommendation.get("selected_action") or {}
        return {
            "skill_match_event_id": prepared.get("match_event_id"),
            "matched": bool(recommendation.get("matched")),
            "skill_id": recommendation.get("skill_id"),
            "action_id": action.get("action_id"),
            "proposed_action": action.get("canonical_command_template") or action.get("command_template") or action.get("command") or None,
        }

    def _record_skill_decision(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.bridge.record_skill_decision(
            args["source"], args["session_id"], args["decision"], args.get("reason", ""),
            args.get("skill_match_event_id"), args.get("skill_id"), args.get("action_id"),
        )
        return {
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": result.get("trajectory_id"),
            "decision_event_id": result.get("decision_event_id"),
            "decision": result.get("decision"),
        }

    def _record_skill_application(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.bridge.record_skill_application(
            args["source"], args["session_id"], args["tool_name"], args["command"], args["status"],
            args.get("preview", ""), args.get("exit_code"), args.get("skill_match_event_id"),
            args.get("skill_id"), args.get("action_id"), args.get("tool_call_id"), args.get("tool_call_event_id"),
        )
        retrieval = result.get("next_skill_retrieval") or {}
        return {
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": result.get("trajectory_id"),
            "tool_call_event_id": result.get("tool_call_event_id"),
            "tool_result_event_id": result.get("tool_result_event_id"),
            "status": result.get("status"),
            "next_skill_match_event_id": retrieval.get("match_event_id"),
            "repair_branch_suggestion": _repair_suggestion_summary(result.get("repair_branch_suggestion")),
            "version_context": _version_context_summary(result.get("version_context")),
        }

    def _create_snapshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.bridge.create_snapshot(args["source"], args["session_id"], args.get("message", ""), args.get("reason", ""))

    def _create_repair_branch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.bridge.create_repair_branch(
            args["source"], args["session_id"], args.get("branch_id"), args.get("snapshot_id"), args.get("reason", ""),
        )
        branch = result.get("branch") or {}
        snapshot = result.get("snapshot") or {}
        decision = result.get("decision_event") or {}
        event = result.get("event") or {}
        return {
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": result.get("trajectory_id"),
            "branch_id": branch.get("branch_id"),
            "source_snapshot_id": snapshot.get("snapshot_id"),
            "decision_event_id": decision.get("event_id"),
            "branch_created_event_id": event.get("event_id"),
            "version_context": _version_context_summary(result.get("version_context")),
        }

    def _rollback_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.bridge.rollback_context(
            args["source"], args["session_id"], args["snapshot_id"], args.get("target_branch_id"), args.get("reason", ""),
        )
        branch = result.get("branch") or {}
        decision = result.get("decision_event") or {}
        event = result.get("event") or {}
        return {
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": result.get("trajectory_id"),
            "branch_id": branch.get("branch_id"),
            "snapshot_id": args["snapshot_id"],
            "decision_event_id": decision.get("event_id"),
            "rollback_event_id": event.get("event_id"),
            "version_context": _version_context_summary(result.get("version_context")),
        }

    def _record_version_decision(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.bridge.record_version_decision(
            args["source"], args["session_id"], args["action"], args["decision"], args.get("reason", ""), args.get("suggestion_event_id"),
        )
        event = result.get("decision_event") or {}
        return {
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": result.get("trajectory_id"),
            "decision_event_id": event.get("event_id"),
            "action": args["action"],
            "decision": args["decision"],
            "version_context": _version_context_summary(result.get("version_context")),
        }

    def _get_version_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return the current version state without replaying a session.

        The bridge status is intentionally rich for the dashboard, but an MCP
        caller normally needs only the current branch and the identifiers it
        can act on.  Returning the complete session and every raw event makes
        desktop MCP clients deserialize the same large payload twice (text and
        structured content), which can make an otherwise quick status request
        appear to hang.
        """
        status = self.bridge.version_status(args["source"], args["session_id"])
        context = status.get("version_context") or {}

        def snapshot_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "snapshot_id": snapshot.get("snapshot_id"),
                "branch_id": snapshot.get("branch_id"),
                "checkpoint_event_id": snapshot.get("event_id"),
                "message": snapshot.get("message", ""),
                "created_at": snapshot.get("created_at"),
            }

        def branch_summary(branch: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "branch_id": branch.get("branch_id"),
                "base_event_id": branch.get("base_event_id"),
                "head_event_id": branch.get("head_event_id"),
                "snapshot_id": branch.get("snapshot_id"),
            }

        def event_summary(event: Dict[str, Any]) -> Dict[str, Any]:
            payload = event.get("payload") or {}
            return {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "branch_id": event.get("branch_id"),
                "timestamp": event.get("timestamp"),
                "operation": (event.get("metadata") or {}).get("operation"),
                "action": payload.get("action"),
                "decision": payload.get("decision"),
                "reason": payload.get("reason"),
                "snapshot_id": payload.get("snapshot_id") or (event.get("refs") or {}).get("snapshot_id"),
                "suggested_branch_id": payload.get("suggested_branch_id"),
            }

        trajectory = status.get("trajectory") or {}
        snapshots = status.get("snapshots") or []
        version_events = status.get("version_events") or []
        return {
            "schema_version": "contextdb.mcp.version_status.v1",
            "source": args["source"],
            "session_id": args["session_id"],
            "trajectory_id": trajectory.get("trajectory_id"),
            "active_branch_id": context.get("active_branch_id") or "main",
            "latest_snapshot": snapshot_summary(context["latest_snapshot"]) if context.get("latest_snapshot") else None,
            "repair_branch_suggestion": context.get("repair_branch_suggestion"),
            "last_version_decision": context.get("last_version_decision"),
            "branches": [branch_summary(branch) for branch in status.get("branches") or []],
            "recent_snapshots": [snapshot_summary(snapshot) for snapshot in snapshots[-5:]],
            "recent_version_events": [event_summary(event) for event in version_events[-8:]],
            "workspace_restore": status.get("workspace_restore") or context.get("workspace_restore"),
        }


def _snapshot_summary(snapshot: Any) -> Dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "branch_id": snapshot.get("branch_id"),
        "checkpoint_event_id": snapshot.get("event_id"),
        "message": snapshot.get("message", ""),
        "created_at": snapshot.get("created_at"),
    }


def _repair_suggestion_summary(suggestion: Any) -> Dict[str, Any] | None:
    if not isinstance(suggestion, dict):
        return None
    return {
        "suggestion_event_id": suggestion.get("suggestion_event_id"),
        "failure_event_id": suggestion.get("failure_event_id"),
        "snapshot_id": suggestion.get("snapshot_id"),
        "suggested_branch_id": suggestion.get("suggested_branch_id"),
        "source_branch_id": suggestion.get("source_branch_id"),
        "created_at": suggestion.get("created_at"),
    }


def _version_context_summary(context: Any) -> Dict[str, Any]:
    context = context if isinstance(context, dict) else {}
    decision = context.get("last_version_decision")
    return {
        "active_branch_id": context.get("active_branch_id") or "main",
        "latest_snapshot": _snapshot_summary(context.get("latest_snapshot")),
        "repair_branch_suggestion": _repair_suggestion_summary(context.get("repair_branch_suggestion")),
        "last_version_decision": {
            "event_id": decision.get("event_id"),
            "action": decision.get("action"),
            "decision": decision.get("decision"),
            "at": decision.get("at"),
        } if isinstance(decision, dict) else None,
        "workspace_restore": context.get("workspace_restore"),
    }


def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _mcp_response_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the agent-facing form of a ContextDB result.

    ``stream_context`` retains raw events and a full source-event timeline for
    the Dashboard and API. Returning that diagnostic material twice through a
    stdio MCP response (as text and structured content) can exceed a desktop
    client's tool-result limit. The agent already receives the selected,
    rendered context, so omit the raw trace from the MCP transport only.
    """
    prompt_context = payload.get("prompt_context")
    if not isinstance(prompt_context, dict):
        return payload

    prompt_keys = (
        "schema_version", "trajectory_id", "branch_id", "head_event_id",
        "token_budget", "selected_tokens", "max_selectable_tokens",
        "remaining_tokens", "estimated_full_history_tokens",
        "estimated_loaded_tokens", "estimated_saved_tokens",
        "estimated_saved_percent", "loading_policy", "rendered_agent_context",
        "summary", "materialization", "llm",
    )
    compact = dict(payload)
    compact["prompt_context"] = {
        key: prompt_context[key]
        for key in prompt_keys
        if key in prompt_context
    }
    summary = compact["prompt_context"].get("summary")
    if isinstance(summary, dict):
        summary_keys = (
            "schema_version", "trajectory_id", "branch_id", "summary",
            "key_facts", "open_items", "status", "method",
        )
        compact_summary = {
            key: summary[key]
            for key in summary_keys
            if key in summary
        }
        scope = summary.get("scope")
        if isinstance(scope, dict) and "source_event_count" in scope:
            compact_summary["source_event_count"] = scope["source_event_count"]
        compact["prompt_context"]["summary"] = compact_summary
    compact.pop("recent_events", None)
    return compact


def _tool_result(
    payload: Dict[str, Any],
    is_error: bool = False,
    include_structured_content: bool = True,
) -> Dict[str, Any]:
    response_payload = _mcp_response_payload(payload)
    text = json.dumps(response_payload, ensure_ascii=False, indent=2)
    result = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if include_structured_content:
        result["structuredContent"] = response_payload
    return result


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
                    name = str(params.get("name") or "")
                    payload = server.dispatch(name, dict(params.get("arguments") or {}))
                    response = _result(
                        request_id,
                        _tool_result(
                            payload,
                            include_structured_content=name != "contextdb_get_recommendation",
                        ),
                    )
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
