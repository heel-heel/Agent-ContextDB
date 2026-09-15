from __future__ import annotations

"""Bridge Claude Code lifecycle hooks to the ContextDB Agent Hook protocol.

Claude Code runs this script with one hook JSON object on stdin. The script
does not execute or alter Claude's tools: it records lifecycle events and uses
the supplied transcript path as an incremental, hook-triggered source of
visible assistant messages. After a failed Bash call, it returns a documented
``additionalContext`` payload for Claude's next model decision.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
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

    return {
        # Preserve Claude's native tool identity. The shared hook bridge
        # canonicalizes this to the same lowercase chain format as Codex.
        "tool_name": tool_name or "claude-tool",
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


def _assistant_text(record: Dict[str, Any]) -> str:
    """Extract only visible assistant text from one Claude transcript record."""
    message = record.get("message") if isinstance(record.get("message"), dict) else {}
    if message.get("role") != "assistant":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        _text(item.get("text")).strip()
        for item in content
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
    ]
    return "\n".join(part for part in parts if part)


def _state_file(state_dir: Path, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    return state_dir / ("claude-transcript-" + digest + ".json")


def _load_state(path: Path, transcript_path: Path) -> Dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if state.get("transcript_path") != str(transcript_path):
        return {"transcript_path": str(transcript_path), "offset": 0, "seen_message_ids": []}
    seen = state.get("seen_message_ids")
    return {
        "transcript_path": str(transcript_path),
        "offset": state.get("offset") if isinstance(state.get("offset"), int) else 0,
        "seen_message_ids": seen if isinstance(seen, list) else [],
    }


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _assistant_message_event(
    record: Dict[str, Any], text: str, transcript_path: Path,
) -> tuple[str, Dict[str, Any]]:
    message_id = str(record.get("uuid") or record.get("id") or "")
    if not message_id:
        message_id = hashlib.sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    event = {
        "event_type": "assistant_message",
        "event_id": "claude-transcript-assistant:" + message_id,
        "actor": "agent",
        "payload": {"text": text},
        "metadata": {
            "adapter": "claude-code-transcript",
            "claude_transcript_path": str(transcript_path),
            "claude_transcript_message_id": message_id,
            "claude_transcript_timestamp": record.get("timestamp"),
        },
    }
    return message_id, event


def _sync_assistant_messages(
    raw: Dict[str, Any], base_url: str, state_dir: Path,
) -> int:
    """Forward transcript text appended since the prior lifecycle-hook wakeup."""
    transcript = raw.get("transcript_path")
    session_id = str(raw.get("session_id") or "")
    if not transcript or not session_id:
        return 0
    transcript_path = Path(str(transcript))
    if not transcript_path.is_file():
        return 0

    state_path = _state_file(state_dir, session_id)
    state = _load_state(state_path, transcript_path)
    size = transcript_path.stat().st_size
    offset = min(max(state["offset"], 0), size)
    seen = {str(item) for item in state["seen_message_ids"]}
    forwarded = 0

    with transcript_path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while True:
            line_start = stream.tell()
            line = stream.readline()
            if not line:
                offset = stream.tell()
                break
            if not line.endswith("\n"):
                # Claude can append while the hook runs. Revisit an unfinished line next time.
                offset = line_start
                break
            try:
                record = json.loads(line)
            except ValueError:
                offset = stream.tell()
                continue
            if not isinstance(record, dict):
                offset = stream.tell()
                continue
            text = _assistant_text(record)
            if not text:
                offset = stream.tell()
                continue
            message_id, event = _assistant_message_event(record, text, transcript_path)
            if message_id in seen:
                offset = stream.tell()
                continue
            _post(base_url, "/api/v1/hooks/events", {
                "protocol_version": HOOK_PROTOCOL_VERSION,
                "source": SOURCE,
                "adapter": "generic",
                "session_id": session_id,
                "agent_id": SOURCE,
                "event": event,
            })
            seen.add(message_id)
            forwarded += 1
            offset = stream.tell()

    state.update({
        "offset": offset,
        # This also keeps a rewritten transcript from replaying recorded assistant text.
        "seen_message_ids": sorted(seen),
    })
    _save_state(state_path, state)
    return forwarded


def _initialize_transcript_cursor(raw: Dict[str, Any], state_dir: Path) -> None:
    """Start a new Claude session from its current transcript end, without replay."""
    transcript = raw.get("transcript_path")
    session_id = str(raw.get("session_id") or "")
    if not transcript or not session_id:
        return
    transcript_path = Path(str(transcript))
    if not transcript_path.is_file():
        return
    state_path = _state_file(state_dir, session_id)
    if state_path.exists():
        return
    _save_state(state_path, {
        "transcript_path": str(transcript_path),
        "offset": transcript_path.stat().st_size,
        "seen_message_ids": [],
    })


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
    parser.add_argument("--event", required=True, choices=("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"))
    parser.add_argument("--base-url", default=os.environ.get("CONTEXTDB_BASE_URL", DEFAULT_BASE_URL))
    default_state_dir = os.environ.get("CONTEXTDB_CLAUDE_TRANSCRIPT_STATE_DIR")
    if not default_state_dir:
        default_state_dir = str(
            Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / "data" / "claude-transcript-sync"
        )
    parser.add_argument(
        "--transcript-state-dir",
        default=default_state_dir,
        help="Project-local state directory for incremental Claude transcript synchronization.",
    )
    args = parser.parse_args()

    try:
        raw = json.load(sys.stdin)
        if not isinstance(raw, dict):
            raise ValueError("Claude Code hook input must be a JSON object")
        session_id = str(raw.get("session_id") or "")
        if not session_id:
            raise ValueError("Claude Code hook input did not include session_id")
        if args.event == "SessionStart":
            _initialize_transcript_cursor(raw, Path(args.transcript_state_dir))
        elif args.event in {"PreToolUse", "Stop"}:
            _sync_assistant_messages(raw, args.base_url, Path(args.transcript_state_dir))
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
                # The hook only needs the matched action for additionalContext.
                # Skip prompt packing and any optional semantic-digest work.
                "include_prompt_context": False,
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
