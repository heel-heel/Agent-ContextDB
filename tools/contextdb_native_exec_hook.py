from __future__ import annotations

"""Wrap a real command so a native Agent exec call becomes a ContextDB hook.

The wrapper is intentionally transport-only.  It runs the child command,
preserves its output and exit code, sends the actual result to ContextDB, then
prints an optional recommendation in the same tool result seen by the Agent.
It never executes a recommended action itself.
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, List
from urllib import request


HOOK_PROTOCOL_VERSION = "contextdb.agent_hook.v1"
RECOMMENDATION_MARKER = "CONTEXTDB_RECOMMENDATION "


def post_json(base_url: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _emit(text: str, stream: Any) -> None:
    if text:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.flush()


def _command_text(command: List[str]) -> str:
    return subprocess.list2cmdline(command)


def _hook_event(base_url: str, source: str, session_id: str, event_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return post_json(base_url, "/api/v1/hooks/events", {
        "protocol_version": HOOK_PROTOCOL_VERSION,
        "source": source,
        "adapter": "generic",
        "session_id": session_id,
        "agent_id": "codex" if source.startswith("codex") else source,
        "event": {"event_type": event_type, "event_id": event_id, "actor": "tool" if event_type == "tool_result" else "agent", "payload": payload},
    })


def _deliver_context(base_url: str, source: str, session_id: str) -> Dict[str, Any]:
    return post_json(base_url, "/api/v1/hooks/context", {
        "source": source,
        "session_id": session_id,
        "token_budget": 1200,
        "delivery_channel": "native-exec-hook",
    })


def _print_recommendation(context: Dict[str, Any]) -> None:
    recommendation = dict(context.get("agent_context") or {})
    recommendation["match_event_id"] = context.get("match_event_id")
    recommendation["delivery_event_id"] = context.get("delivery_event_id")
    if recommendation.get("matched"):
        _emit(RECOMMENDATION_MARKER + json.dumps(recommendation, ensure_ascii=False), sys.stdout)
    elif recommendation.get("instruction"):
        _emit(RECOMMENDATION_MARKER + json.dumps(recommendation, ensure_ascii=False), sys.stdout)


def _best_effort(action: str, callback: Any) -> Any:
    """Keep the Agent command usable when ContextDB is unavailable."""
    try:
        return callback()
    except Exception as exc:
        _emit("ContextDB hook warning (%s): %s" % (action, exc), sys.stderr)
        return None


def run_hooked_command(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("provide the real command after --")
    command_text = _command_text(command)
    call_id = "native_exec_%s" % __import__("uuid").uuid4().hex

    if args.apply_skill:
        _best_effort("record skill decision", lambda: post_json(args.base_url, "/api/v1/hooks/skill_decision", {
                "source": args.source, "session_id": args.session_id, "decision": "accepted",
                "reason": args.decision_reason or "Agent selected the ContextDB recommendation.",
                "skill_match_event_id": args.skill_match_event_id,
                "skill_id": args.skill_id,
                "action_id": args.action_id,
            }))
        # Record the real call before execution as well. This allows the common
        # Hook policy to create a logical pre-action snapshot for an accepted
        # skill that may change state, without duplicating the call later.
        _best_effort("record skill tool call", lambda: _hook_event(args.base_url, args.source, args.session_id, call_id, "tool_call", {
                "tool_name": args.tool_name, "command": command_text,
            }))
    else:
        _best_effort("record tool call", lambda: _hook_event(args.base_url, args.source, args.session_id, call_id, "tool_call", {
                "tool_name": args.tool_name, "command": command_text,
            }))

    try:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", cwd=args.cwd or None)
        stdout, stderr, exit_code = completed.stdout or "", completed.stderr or "", completed.returncode
    except FileNotFoundError as exc:
        stdout, stderr, exit_code = "", str(exc), 127
    except OSError as exc:
        stdout, stderr, exit_code = "", str(exc), 126

    _emit(stdout, sys.stdout)
    _emit(stderr, sys.stderr)
    status = "ok" if exit_code == 0 else "failed"
    preview = (stdout + ("\n" if stdout and stderr else "") + stderr)[-8000:]
    if args.apply_skill:
        result = _best_effort("record skill application", lambda: post_json(args.base_url, "/api/v1/hooks/skill_application", {
                "source": args.source, "session_id": args.session_id,
                "tool_name": args.tool_name, "command": command_text, "status": status,
                "preview": preview, "exit_code": exit_code,
                "skill_match_event_id": args.skill_match_event_id,
                "skill_id": args.skill_id,
                "action_id": args.action_id,
                "tool_call_id": call_id,
            })) or {}
        if result.get("next_skill_retrieval"):
            context = _best_effort("deliver next recommendation", lambda: _deliver_context(args.base_url, args.source, args.session_id))
            if context:
                _print_recommendation(context)
    else:
        result = _best_effort("record tool result", lambda: _hook_event(args.base_url, args.source, args.session_id, call_id, "tool_result", {
                "tool_name": args.tool_name, "command": command_text, "status": status,
                "preview": preview, "exit_code": exit_code,
            })) or {}
        if result.get("skill_retrievals"):
            context = _best_effort("deliver recommendation", lambda: _deliver_context(args.base_url, args.source, args.session_id))
            if context:
                _print_recommendation(context)
    return int(exit_code)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real command and synchronously hook its result into ContextDB.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--source", default="codex")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--tool-name", default="shell")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--apply-skill", action="store_true")
    parser.add_argument("--skill-match-event-id", default=None)
    parser.add_argument("--skill-id", default=None)
    parser.add_argument("--action-id", default=None)
    parser.add_argument("--decision-reason", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        return run_hooked_command(args)
    except Exception as exc:
        _emit("ContextDB native exec hook error: %s" % exc, sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
