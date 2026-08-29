from __future__ import annotations

"""Bridge Claude Code lifecycle hooks to the ContextDB Agent Hook protocol.

Claude Code runs this script with one hook JSON object on stdin.  The script
does not execute or alter Claude's tools: it records the lifecycle event and,
after a failed Bash call, returns a documented ``additionalContext`` payload
for Claude's next model decision.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict
from urllib import request


HOOK_PROTOCOL_VERSION = "contextdb.agent_hook.v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
SOURCE = "claude-code"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("output", "content", "text", "message", "error", "stderr", "stdout"):
            if value.get(key) is not None:
                return _text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tool_payload(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Claude's heterogeneous tools into ContextDB tool fields."""
    command = ""
    for key in ("command", "file_path", "path", "query", "url", "pattern"):
        if tool_input.get(key) is not None:
            command = _text(tool_input[key])
            break
    if not command:
        command = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)

    normalized_name = "shell" if tool_name in {"Bash", "PowerShell"} else tool_name.lower()
    return {
        "tool_name": normalized_name or "claude-tool",
        "command": command,
        "arguments": tool_input,
        "agent_tool_name": tool_name,
    }


def _post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _event(raw: Dict[str, Any], hook_event: str) -> Dict[str, Any] | None:
    session_id = str(raw.get("session_id") or "")
    tool_name = str(raw.get("tool_name") or "")
    tool_input = raw.get("tool_input") if isinstance(raw.get("tool_input"), dict) else {}
    tool_use_id = str(raw.get("tool_use_id") or "")
    metadata = {
        "adapter": "claude-code-hooks",
        "claude_hook_event": hook_event,
        "claude_tool_name": tool_name,
        "claude_cwd": raw.get("cwd"),
        "claude_transcript_path": raw.get("transcript_path"),
    }

    if hook_event == "SessionStart":
        return {
            "event_type": "system_event",
            "event_id": "claude-session-start:" + session_id,
            "actor": "system",
            "payload": {"text": "Claude Code session started."},
            "metadata": metadata,
        }
    if hook_event == "UserPromptSubmit":
        prompt = raw.get("prompt") or raw.get("user_prompt") or tool_input.get("prompt")
        if not prompt:
            return None
        return {
            "event_type": "user_message",
            "event_id": str(raw.get("prompt_id") or "claude-user-prompt:" + session_id),
            "actor": "user",
            "payload": {"text": _text(prompt)},
            "metadata": metadata,
        }
    if not tool_name or not tool_use_id:
        return None

    payload = _tool_payload(tool_name, tool_input)
    if hook_event == "PreToolUse":
        return {
            "event_type": "tool_call",
            "event_id": tool_use_id,
            "actor": "agent",
            "payload": payload,
            "metadata": metadata,
        }

    response = raw.get("tool_response")
    preview = _text(response or raw.get("error") or raw.get("message"))
    payload.update({
        "status": "failed" if hook_event == "PostToolUseFailure" else "ok",
        "preview": preview[:8000],
        "exit_code": raw.get("exit_code"),
    })
    return {
        "event_type": "tool_result",
        "event_id": tool_use_id,
        "actor": "tool",
        "payload": payload,
        "metadata": metadata,
    }


def _additional_context(recommendation: Dict[str, Any]) -> str:
    if not recommendation.get("matched"):
        return "ContextDB checked the failed tool result. No reusable skill matched; continue normal diagnosis."
    action = recommendation.get("selected_action") or {}
    lines = [
        "ContextDB matched a reusable skill for the failed tool call.",
        "skill_id: " + str(recommendation.get("skill_id") or "unknown"),
        "match_reason: " + str(recommendation.get("match_reason") or ""),
        "recommended_action: " + str(action.get("canonical_command_template") or action.get("command_template") or action.get("name") or ""),
        "Treat this as advice. Decide whether it is safe before executing it.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward Claude Code hooks to ContextDB.")
    parser.add_argument("--event", required=True, choices=("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure"))
    parser.add_argument("--base-url", default=os.environ.get("CONTEXTDB_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()

    try:
        raw = json.load(sys.stdin)
        if not isinstance(raw, dict):
            raise ValueError("Claude Code hook input must be a JSON object")
        session_id = str(raw.get("session_id") or "")
        if not session_id:
            raise ValueError("Claude Code hook input did not include session_id")
        event = _event(raw, args.event)
        if event is None:
            return 0
        response = _post(args.base_url, "/api/v1/hooks/events", {
            "protocol_version": HOOK_PROTOCOL_VERSION,
            "source": SOURCE,
            "adapter": "generic",
            "session_id": session_id,
            "agent_id": SOURCE,
            "event": event,
        })
        if args.event == "PostToolUseFailure":
            context = _post(args.base_url, "/api/v1/hooks/context", {
                "source": SOURCE,
                "session_id": session_id,
                "token_budget": 1200,
                "delivery_channel": "claude-code-post-tool-failure",
            })
            recommendation = context.get("agent_context") or {}
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": _additional_context(recommendation),
                }
            }, ensure_ascii=False))
        return 0
    except Exception as exc:
        # Claude's real tool must remain usable if the optional ContextDB service is down.
        print("ContextDB Claude hook warning: %s" % exc, file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
