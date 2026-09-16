from __future__ import annotations

"""Online Agent Hook protocol and adapters.

The transport is deliberately small: every agent emits a JSON object to
``POST /api/v1/hooks/events``.  Codex is adapted from ``codex exec --json``;
other agents can emit the framework-neutral event envelope directly.
"""

from dataclasses import dataclass, field
import json
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib import request
from urllib.parse import quote

from .agent_runtime import AgentContextBridge, CodexContextBridge
from .models import utc_now
from .service import ContextDB


HOOK_PROTOCOL_VERSION = 'contextdb.agent_hook.v1'

# These are deliberately conservative. A snapshot is a ContextDB
# checkpoint, not a filesystem backup, so read-only shell commands should not
# clutter an online trajectory with version events.
_MUTATING_TOOL_NAMES = {'write', 'edit', 'file_edit', 'apply_patch'}
_COMMAND_TOOL_NAMES = {
    'shell', 'exec', 'powershell', 'bash', 'terminal', 'command', 'git',
    'python', 'functions.exec', 'functions.exec_command', 'exec_command',
    'ordinary native terminal', 'native terminal',
}
_MUTATING_COMMAND = re.compile(
    r"(?:^|\s|[\"'])(?:set-content|add-content|new-item|remove-item|move-item|copy-item|"
    r"mkdir|rmdir|rm|mv|cp|touch|git\s+(?:checkout|reset|clean|apply|commit)|"
    r"pip\s+(?:install|uninstall)|npm\s+(?:install|uninstall|update)|"
    r"cargo\s+(?:add|update))(?:\s|$)|"
    r"\.(?:write_text|write_bytes)\s*\(|"
    r"\.(?:writefilesync|appendfilesync|copyfilesync|renamesync)\s*\(|"
    r"\bopen\s*\([^,\n]+,\s*[\"'](?:w|a|x)[\"']|"
    r"(?<![<>=])>(?!>)",
    re.IGNORECASE,
)
_NESTED_EXEC_TOOL = re.compile(
    r"\btools\.(?:"
    r"(web__run)"
    r"|mcp__contextdb__(contextdb_[A-Za-z0-9_]+)"
    r")\s*\(",
)
_DIRECT_CONTEXTDB_TOOL = re.compile(
    r"^(?:mcp__contextdb__)?(contextdb_[A-Za-z0-9_]+)$",
    re.IGNORECASE,
)
_POWERSHELL_HOST = re.compile(r"^\s*(?:&\s*)?(?:powershell|pwsh)(?:\.exe)?\b", re.IGNORECASE)
_COMMAND_LEAF = re.compile(
    r"(?:^|[;|&]\s*|-command\s+)(?:['\"]\s*)?(?:&\s*)?"
    r"(git|python(?:\.exe)?|pip(?:\.exe)?|npm(?:\.cmd)?|node(?:\.exe)?|"
    r"pytest(?:\.exe)?|rg(?:\.exe)?|get-content|set-content|add-content|"
    r"get-childitem|new-item|remove-item|copy-item|move-item)\b",
    re.IGNORECASE,
)
_POWERSHELL_PARSE_FAILURE = re.compile(
    r"(?:parsererror|unexpected token|missing (?:expression|terminator)|at line:\d+ char:\d+)",
    re.IGNORECASE,
)


def _is_contextdb_tool_call(payload: Dict[str, Any]) -> bool:
    """Return whether a call invokes a ContextDB MCP operation.

    ContextDB MCP operations can change the audit model, but never the Agent
    workspace. They must not create an automatic pre-action snapshot or a
    repair branch may incorrectly start from its own control-plane call.
    """
    transport = _transport_tool_name(payload).lower()
    if transport.startswith("contextdb_"):
        return True
    command = _text(payload.get("command") or payload.get("cmd") or "")
    return any(contextdb_tool for _, contextdb_tool in _NESTED_EXEC_TOOL.findall(command))


def _canonical_tool_name(tool_name: Any) -> str:
    """Return the stable display identity for a native or MCP tool.

    Claude Code exposes ContextDB operations directly as
    ``mcp__contextdb__contextdb_*`` while Codex invokes the same operation
    inside ``exec``. The audit model uses the common ``contextdb_*`` name.
    """
    raw = str(tool_name or "tool").strip() or "tool"
    direct_contextdb = _DIRECT_CONTEXTDB_TOOL.match(raw)
    if direct_contextdb:
        return direct_contextdb.group(1).lower()
    normalized = raw.lower()
    if normalized in {"pwsh", "powershell.exe", "pwsh.exe"}:
        return "powershell"
    if normalized.endswith(".exe"):
        return normalized[:-4]
    if normalized.endswith(".cmd"):
        return normalized[:-4]
    return normalized


def _should_auto_snapshot(payload: Dict[str, Any]) -> bool:
    """Return whether a Hook tool call merits a pre-action snapshot.

    Python and Node.js one-liners that open a file for writing are included
    because they are common in live-agent runs and mutate the workspace just
    as a shell ``Set-Content`` call would.
    """
    if _is_contextdb_tool_call(payload):
        return False
    explicit = payload.get('contextdb_snapshot')
    if explicit is not None:
        return bool(explicit)
    tool_name = _transport_tool_name(payload).lower()
    if tool_name in _MUTATING_TOOL_NAMES:
        return True
    if tool_name not in _COMMAND_TOOL_NAMES:
        return False
    command = _text(payload.get('command') or payload.get('cmd') or '')
    return bool(_MUTATING_COMMAND.search(command))


def _command_tool_chain(command: Any) -> List[str]:
    """Return the observable process/cmdlet chain inside a terminal command."""
    text = _text(command).strip()
    if not text:
        return []
    if _POWERSHELL_HOST.match(text):
        chain = ['powershell']
        leaves = [match.group(1).lower().removesuffix('.exe').removesuffix('.cmd') for match in _COMMAND_LEAF.finditer(text)]
        if leaves:
            chain.append(leaves[-1])
        return chain
    leaves = [match.group(1).lower().removesuffix('.exe').removesuffix('.cmd') for match in _COMMAND_LEAF.finditer(text)]
    return [leaves[-1]] if leaves else []


def _tool_chain(tool_name: Any, command: Any) -> List[str]:
    """Keep the transport tool and every inner tool identifiable from its input."""
    raw_tool = _canonical_tool_name(tool_name)
    nested_names: List[str] = []
    for web_tool, contextdb_tool in _NESTED_EXEC_TOOL.findall(_text(command)):
        nested_name = web_tool or contextdb_tool
        if nested_name and nested_name not in nested_names:
            nested_names.append(nested_name)
    chain = [raw_tool]
    if nested_names:
        chain.extend(nested_names)
    else:
        chain.extend(_command_tool_chain(command))
    return [name for index, name in enumerate(chain) if index == 0 or name != chain[index - 1]]


def _transport_tool_name(payload: Dict[str, Any]) -> str:
    """Recover the raw entry point without duplicating it in tool-call data."""
    chain = payload.get('tool_chain') or []
    return _canonical_tool_name(
        payload.get('transport_tool_name')
        or payload.get('tool_name')
        or payload.get('tool')
        or (chain[0] if chain else 'tool')
    )


def _invoked_tool_name(tool_name: Any, command: Any) -> Optional[str]:
    """Backward-compatible shorthand for the observable innermost tool."""
    chain = _tool_chain(tool_name, command)
    return chain[-1] if len(chain) > 1 else None


def _failure_tool_name(tool_chain: List[str], preview: Any) -> str:
    """Attribute a failed execution to the observed failing layer when possible."""
    if not tool_chain:
        return 'tool'
    output = _text(preview)
    if 'powershell' in tool_chain and _POWERSHELL_PARSE_FAILURE.search(output):
        return 'powershell'
    # A child tool normally owns a propagated non-zero exit code.  The final
    # chain member is therefore the best truthful attribution unless the
    # PowerShell parser itself failed above.
    return tool_chain[-1]


def _result_tool_payload(
    transport_tool_name: Any,
    command: Any,
    status: Any,
    preview: Any = '',
    exit_code: Any = None,
    tool_chain: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a result payload with its semantic tool identity and full chain."""
    raw_tool = _canonical_tool_name(transport_tool_name)
    chain = list(tool_chain or _tool_chain(raw_tool, command)) or [raw_tool]
    failed = str(status or '').lower() in {'failed', 'error', 'timeout'}
    result_tool = _failure_tool_name(chain, preview) if failed else chain[-1]
    payload = {
        'tool_name': result_tool,
        'transport_tool_name': raw_tool,
        'tool_chain': chain,
        'command': command,
        'status': status,
        'preview': preview,
        'exit_code': exit_code,
    }
    if len(chain) > 1:
        payload['invoked_tool'] = chain[-1]
    if failed:
        payload['failure_tool_name'] = result_tool
    return payload


def _tool_payload(tool_name: Any, command: Any, *, include_transport: bool = False, **extra: Any) -> Dict[str, Any]:
    """Build call data without duplicating the transport already in its chain."""
    raw_tool = _canonical_tool_name(tool_name)
    chain = _tool_chain(raw_tool, command)
    payload = {
        'tool_chain': chain,
        'command': command,
        **extra,
    }
    if include_transport:
        payload['tool_name'] = raw_tool
        payload['transport_tool_name'] = raw_tool
    else:
        payload['status'] = 'pending'
    return payload


def _normalize_tool_call_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make direct Agent hooks use the same compact call shape as Codex."""
    command = payload.get('command') or payload.get('cmd') or ''
    ignored = {
        'tool_name', 'tool', 'transport_tool_name', 'tool_chain', 'command',
        'cmd', 'status', 'preview', 'output', 'message', 'exit_code',
        'invoked_tool', 'failure_tool_name', 'result_tool_name',
        'result_event_id',
    }
    extra = {key: value for key, value in payload.items() if key not in ignored}
    return _tool_payload(_transport_tool_name(payload), command, **extra)


@dataclass
class HookEvent:
    event_type: str
    payload: Dict[str, Any]
    actor: str
    external_id: Optional[str] = None
    refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentHookAdapter(Protocol):
    """Adapter boundary for future Claude Code, Cursor, or SDK hook sources."""

    source_name: str

    def infer_session_id(self, raw_event: Dict[str, Any]) -> Optional[str]: ...

    def normalize(self, raw_event: Dict[str, Any]) -> Iterable[HookEvent]: ...


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(_text(item.get('text') if isinstance(item, dict) else item) for item in value)
    if isinstance(value, dict):
        for key in ('text', 'content', 'output', 'message', 'aggregated_output'):
            if value.get(key) is not None:
                return _text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _command(item: Dict[str, Any]) -> str:
    for key in ('command', 'cmd', 'input', 'arguments'):
        value = item.get(key)
        if value not in (None, ''):
            return _text(value)
    return ''


def _tool_output(item: Dict[str, Any]) -> str:
    for key in ('aggregated_output', 'output', 'result', 'content', 'message'):
        value = item.get(key)
        if value not in (None, ''):
            return _text(value)
    return ''


def _tool_status(item: Dict[str, Any]) -> str:
    status = str(item.get('status') or '').lower()
    if status in {'failed', 'error', 'timeout', 'cancelled'}:
        return 'failed'
    exit_code = item.get('exit_code')
    try:
        if exit_code is not None and int(exit_code) != 0:
            return 'failed'
    except (TypeError, ValueError):
        pass
    return 'ok'


class GenericHookAdapter:
    """Adapter for the documented ContextDB hook envelope used by future agents."""

    source_name = 'generic'

    def infer_session_id(self, raw_event: Dict[str, Any]) -> Optional[str]:
        return str(raw_event.get('session_id') or raw_event.get('thread_id') or '') or None

    def normalize(self, raw_event: Dict[str, Any]) -> Iterable[HookEvent]:
        name = str(raw_event.get('event_type') or raw_event.get('name') or raw_event.get('type') or 'system_event')
        payload = raw_event.get('payload') if isinstance(raw_event.get('payload'), dict) else {}
        if not payload:
            payload = {key: value for key, value in raw_event.items() if key not in {'event_type', 'name', 'type', 'event_id', 'session_id', 'thread_id', 'actor', 'metadata', 'refs'}}
        if name in {'session_start', 'session_started'}:
            name = 'system_event'
            payload = {'text': payload.get('text') or 'Agent hook session started.'}
        yield HookEvent(
            event_type=name,
            payload=payload,
            actor=str(raw_event.get('actor') or ('tool' if name == 'tool_result' else 'agent')),
            external_id=str(raw_event.get('event_id') or raw_event.get('id') or '') or None,
            refs=dict(raw_event.get('refs') or {}),
            metadata={'adapter': 'generic-hook', **dict(raw_event.get('metadata') or {})},
        )


class CodexExecJSONLHookAdapter:
    """Maps the public JSONL events from ``codex exec --json`` into ContextDB."""

    source_name = 'codex'

    def infer_session_id(self, raw_event: Dict[str, Any]) -> Optional[str]:
        thread_id = raw_event.get('thread_id')
        if thread_id:
            return str(thread_id)
        thread = raw_event.get('thread')
        return str(thread.get('id')) if isinstance(thread, dict) and thread.get('id') else None

    def normalize(self, raw_event: Dict[str, Any]) -> Iterable[HookEvent]:
        event_type = str(raw_event.get('type') or '').lower()
        base_metadata = {
            'adapter': 'codex-exec-jsonl',
            'codex_event_type': event_type,
        }
        if event_type == 'thread.started':
            yield HookEvent('system_event', {'text': 'Codex thread started.', 'thread_id': raw_event.get('thread_id')}, 'system', metadata=base_metadata)
            return
        if event_type in {'turn.started', 'turn.completed'}:
            payload = {'text': 'Codex ' + event_type.replace('.', ' ') + '.'}
            if raw_event.get('usage') is not None:
                payload['usage'] = raw_event.get('usage')
            yield HookEvent('system_event', payload, 'system', metadata=base_metadata)
            return
        if event_type in {'turn.failed', 'error'}:
            yield HookEvent('system_event', {'text': _text(raw_event.get('error') or raw_event.get('message') or 'Codex run failed.'), 'status': 'failed'}, 'system', metadata=base_metadata)
            return
        if not event_type.startswith('item.'):
            yield HookEvent('system_event', {'text': json.dumps(raw_event, ensure_ascii=False)[:1000]}, 'system', metadata=base_metadata)
            return

        item = raw_event.get('item') if isinstance(raw_event.get('item'), dict) else {}
        item_type = str(item.get('type') or '').lower()
        item_id = str(item.get('id') or '') or None
        metadata = {**base_metadata, 'codex_item_type': item_type, 'codex_item_id': item_id}
        if item_type in {'agent_message', 'assistant_message', 'message'} and event_type == 'item.completed':
            yield HookEvent('assistant_message', {'text': _text(item.get('text') or item.get('content') or item.get('message'))}, 'agent', item_id, metadata=metadata)
        elif item_type in {'command_execution', 'shell_call', 'tool_call'}:
            command = _command(item)
            if event_type == 'item.started':
                yield HookEvent('tool_call', _tool_payload(item.get('tool_name') or 'shell', command), 'agent', item_id, metadata=metadata)
            elif event_type == 'item.completed':
                yield HookEvent('tool_result', _tool_payload(
                    item.get('tool_name') or 'shell', command,
                    status=_tool_status(item), preview=_tool_output(item)[:4000],
                    exit_code=item.get('exit_code'), include_transport=True,
                ), 'tool', item_id, metadata=metadata)
        elif item_type in {'file_change', 'file_edit'} and event_type == 'item.completed':
            yield HookEvent('file_edit', {'path': item.get('path'), 'summary': _text(item.get('summary') or item.get('text')), 'diff': item.get('diff')}, 'agent', item_id, metadata=metadata)
        elif item_type == 'error':
            yield HookEvent('system_event', {'text': _text(item.get('message') or item.get('error')), 'status': 'failed'}, 'system', item_id, metadata=metadata)


class CodexDesktopSessionHookAdapter:
    """Normalize records written by the Windows Codex App session logger."""

    source_name = "codex-session"

    def infer_session_id(self, raw_event: Dict[str, Any]) -> Optional[str]:
        payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
        value = raw_event.get("session_id") or payload.get("session_id")
        return str(value) if value else None

    def normalize(self, raw_event: Dict[str, Any]) -> Iterable[HookEvent]:
        record_type = str(raw_event.get("type") or "").lower()
        payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
        metadata = {
            "adapter": "codex-desktop-session-jsonl",
            "codex_session_record_type": record_type,
            "codex_session_timestamp": raw_event.get("timestamp"),
        }
        if record_type == "session_meta":
            yield HookEvent(
                "system_event",
                {"text": "Codex session started.", "cwd": payload.get("cwd")},
                "system",
                str(payload.get("id") or payload.get("session_id") or "") or None,
                metadata=metadata,
            )
            return
        if record_type != "response_item":
            return

        item_type = str(payload.get("type") or "").lower()
        item_id = str(payload.get("id") or "") or None
        metadata = {**metadata, "codex_item_type": item_type, "codex_item_id": item_id}
        if item_type == "message":
            role = str(payload.get("role") or "").lower()
            if role == "user":
                yield HookEvent("user_message", {"text": _text(payload.get("content"))}, "user", item_id, metadata=metadata)
            elif role == "assistant":
                yield HookEvent("assistant_message", {"text": _text(payload.get("content"))}, "agent", item_id, metadata=metadata)
            return
        if item_type in {"function_call", "custom_tool_call"}:
            arguments = payload.get("arguments")
            command = _desktop_command(arguments)
            if item_type == "custom_tool_call":
                command = _custom_tool_command(payload.get("input"))
            call_id = str(payload.get("call_id") or item_id or "") or None
            yield HookEvent(
                "tool_call",
                _tool_payload(payload.get("name") or "tool", command, arguments=arguments),
                "agent",
                call_id,
                metadata=metadata,
            )
            return
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            result = _desktop_output(_text(payload.get("output")))
            if item_type == "custom_tool_call_output":
                result = _custom_tool_result(payload)
            call_id = str(payload.get("call_id") or item_id or "") or None
            yield HookEvent(
                "tool_result",
                _tool_payload(
                    result.get("tool_name") or "tool", result.get("command") or "",
                    status=result["status"], preview=result["preview"],
                    exit_code=result.get("exit_code"), include_transport=True,
                ),
                "tool",
                call_id,
                metadata=metadata,
            )


def _desktop_command(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return _command(arguments)
    if not isinstance(arguments, str):
        return _text(arguments)
    try:
        decoded = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    return _command(decoded) if isinstance(decoded, dict) else _text(decoded)


def _desktop_output(output: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    try:
        decoded = json.loads(output)
        if isinstance(decoded, dict):
            parsed = decoded
    except (TypeError, ValueError):
        pass
    exit_code = parsed.get("exit_code")
    preview = _text(parsed.get("output") or parsed.get("content") or output)
    status = _tool_status({"status": parsed.get("status"), "exit_code": exit_code})
    if status == "ok" and any(token in preview.lower() for token in ("error:", "failed", "command not found", "traceback")):
        status = "failed"
    return {
        "tool_name": parsed.get("tool_name"),
        "command": parsed.get("command") or parsed.get("cmd"),
        "status": status,
        "preview": preview[:4000],
        "exit_code": exit_code,
    }


def _custom_tool_command(value: Any) -> str:
    """Extract a shell command from Codex App's serialized custom tool input."""
    raw = _text(value)
    match = re.search(r'"cmd"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
    if not match:
        return raw
    try:
        return json.loads('"' + match.group(1) + '"')
    except (TypeError, ValueError):
        return match.group(1)


def _custom_tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Interpret the output wrapper emitted by the Windows Codex App."""
    preview = _text(payload.get("output"))
    codes = [int(value) for value in re.findall(r'"exit_code"\s*:\s*(-?\d+)', preview)]
    exit_code = next((value for value in reversed(codes) if value != 0), codes[-1] if codes else payload.get("exit_code"))
    status = _tool_status({"status": payload.get("status"), "exit_code": exit_code})
    if status == "ok" and re.search(r"(?:script|command) failed|traceback|modulenotfounderror", preview, re.IGNORECASE):
        status = "failed"
    return {
        "tool_name": payload.get("tool_name"),
        "command": payload.get("command") or payload.get("cmd"),
        "status": status,
        "preview": preview[:4000],
        "exit_code": exit_code,
    }


class HookSessionBridge:
    """Persists hook sessions and records their normalized events into a trajectory."""

    def __init__(self, db: ContextDB):
        self.db = db
        self.adapters: Dict[str, AgentHookAdapter] = {
            'codex': CodexExecJSONLHookAdapter(),
            'codex-session': CodexDesktopSessionHookAdapter(),
            'generic': GenericHookAdapter(),
        }

    def ingest(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        if envelope.get('protocol_version', HOOK_PROTOCOL_VERSION) != HOOK_PROTOCOL_VERSION:
            raise ValueError('unsupported hook protocol version')
        source = str(envelope.get('source') or 'generic').lower()
        adapter_name = str(envelope.get('adapter') or source).lower()
        adapter = self.adapters.get(adapter_name, self.adapters['generic'])
        raw_event = envelope.get('event') if isinstance(envelope.get('event'), dict) else envelope.get('raw_event')
        if not isinstance(raw_event, dict):
            raise ValueError('hook event must be a JSON object')
        session_id = str(envelope.get('session_id') or adapter.infer_session_id(raw_event) or '')
        if not session_id:
            raise ValueError('session_id is required until the source emits one')
        session = self._ensure_session(source, session_id, envelope)
        trajectory_id = session['trajectory_id']
        bridge = CodexContextBridge(self.db, source_id=f'hook:{source}') if source == 'codex' else AgentContextBridge(self.db, agent_id=session['agent_id'], source_id=f'hook:{source}')
        emitted, retrievals = [], []
        for item in adapter.normalize(raw_event):
            active_branch_id = session.get('active_branch_id') or 'main'
            refs = {
                'hook_protocol': HOOK_PROTOCOL_VERSION,
                'hook_source': source,
                'hook_adapter': adapter_name,
                'hook_session_id': session_id,
                **item.refs,
            }
            metadata = {
                'online': True,
                'integration': 'agent-hook.v1',
                'hook_received_at': utc_now(),
                **item.metadata,
            }
            if item.event_type == 'tool_call':
                call_payload = _normalize_tool_call_payload(item.payload)
                if _should_auto_snapshot(call_payload):
                    version = self._auto_snapshot_before_tool(session, call_payload, refs)
                    emitted.append(version['event'])
                event = bridge.record(
                    trajectory_id, 'tool_call', call_payload, branch_id=active_branch_id,
                    actor='agent', refs=refs, metadata=metadata,
                )
                if item.external_id:
                    session['pending_tools'][item.external_id] = {
                        'event_id': event['event_id'],
                        'tool_name': _transport_tool_name(call_payload),
                        'command': call_payload.get('command') or '',
                        'tool_chain': call_payload.get('tool_chain') or [],
                        'invoked_tool': call_payload.get('invoked_tool'),
                        'branch_id': active_branch_id,
                    }
                emitted.append(event)
                continue
            if item.event_type == 'tool_result':
                pending = session['pending_tools'].pop(item.external_id, None) if item.external_id else None
                reported_tool_name = item.payload.get('tool_name')
                # Codex Desktop function-call outputs often contain only the
                # call id and output. Preserve the native call's name rather
                # than splitting a single execution into e.g. `functions.exec`
                # and a generic `tool` result.
                transport_tool_name = (
                    (pending or {}).get('tool_name')
                    if reported_tool_name in {None, '', 'tool'} and pending
                    else reported_tool_name or _transport_tool_name(item.payload)
                )
                command = item.payload.get('command') or (pending or {}).get('command') or ''
                status = item.payload.get('status', 'ok')
                tool_chain = list((pending or {}).get('tool_chain') or _tool_chain(transport_tool_name, command))
                result_payload = _result_tool_payload(
                    transport_tool_name, command, status,
                    preview=item.payload.get('preview', ''), exit_code=item.payload.get('exit_code'),
                    tool_chain=tool_chain,
                )
                result_branch_id = (pending or {}).get('branch_id') or active_branch_id
                if pending:
                    refs['tool_call_event_id'] = pending['event_id']
                else:
                    implicit_payload = _tool_payload(transport_tool_name, command)
                    implicit = bridge.record(
                        trajectory_id, 'tool_call', implicit_payload,
                        branch_id=result_branch_id, actor='agent',
                        refs={**refs, 'implicit_from_hook_result': True}, metadata=metadata,
                    )
                    refs['tool_call_event_id'] = implicit['event_id']
                    emitted.append(implicit)
                tool_call_event_id = refs['tool_call_event_id']
                recorded = bridge.record_tool_result(
                    trajectory_id, result_payload['tool_name'], command, status,
                    preview=result_payload['preview'], exit_code=result_payload['exit_code'],
                    branch_id=result_branch_id, refs=refs, metadata=metadata,
                    extra_payload={
                        key: value for key, value in result_payload.items()
                        if key not in {'tool_name', 'command', 'status', 'preview', 'exit_code'}
                    },
                )
                emitted.append(recorded['tool_result'])
                self.db.update_tool_call_outcome(
                    trajectory_id, tool_call_event_id, status=status,
                    result_tool_name=result_payload['tool_name'],
                    tool_chain=result_payload['tool_chain'],
                    result_event_id=recorded['tool_result']['event_id'],
                )
                if status in {'failed', 'error', 'timeout'}:
                    suggestion = self._suggest_repair_branch(session, recorded['tool_result'], result_branch_id, refs)
                    if suggestion:
                        emitted.append(suggestion['event'])
                if recorded.get('skill_retrieval'):
                    retrieval = self._skill_recommendation(recorded['skill_retrieval'])
                    retrieval['agent_context']['version_context'] = self._version_context(session)
                    retrievals.append(retrieval)
                    session['last_skill_retrieval'] = retrieval
                continue
            event = bridge.record(
                trajectory_id, item.event_type, item.payload, branch_id=active_branch_id,
                actor=item.actor, refs=refs, metadata=metadata,
            )
            emitted.append(event)
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + len(emitted)
        self._put_session(source, session_id, session)
        return {
            'protocol_version': HOOK_PROTOCOL_VERSION,
            'trajectory_id': trajectory_id,
            'source': source,
            'session_id': session_id,
            'emitted_event_ids': [event['event_id'] for event in emitted],
            'skill_retrievals': retrievals,
            'agent_context': [entry['agent_context'] for entry in retrievals if entry.get('agent_context')],
            'version_context': self._version_context(session),
        }

    def create_snapshot(self, source: str, session_id: str, message: str = '', reason: str = '') -> Dict[str, Any]:
        """Explicitly create a ContextDB snapshot for a live Agent."""
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        version = self.db.create_version_snapshot(
            session['trajectory_id'], session.get('active_branch_id') or 'main', message=message,
            reason=reason or 'Agent requested a checkpoint.', origin='agent', actor='agent',
            refs={'hook_source': source, 'hook_session_id': session_id},
        )
        session['last_version_snapshot'] = self._snapshot_context(version['snapshot'])
        self._touch_session(source, session_id, session, 1)
        return {**version, 'version_context': self._version_context(session)}

    def create_repair_branch(
        self,
        source: str,
        session_id: str,
        branch_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        reason: str = '',
    ) -> Dict[str, Any]:
        """Accept a branch suggestion or create a named repair branch."""
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        suggestion = session.get('last_repair_suggestion') or {}
        snapshot_id = snapshot_id or suggestion.get('snapshot_id') or (session.get('last_version_snapshot') or {}).get('snapshot_id')
        if not snapshot_id:
            raise ValueError('create a snapshot before creating a repair branch')
        candidate = branch_id or suggestion.get('suggested_branch_id') or 'repair'
        candidate = self._available_branch_id(session['trajectory_id'], candidate)
        active_branch_id = session.get('active_branch_id') or 'main'
        decision = self._record_version_decision(
            session, 'create_repair_branch', 'accepted', reason or 'Agent selected a repair branch.',
            {'suggestion_event_id': suggestion.get('suggestion_event_id'), 'snapshot_id': snapshot_id, 'target_branch_id': candidate},
        )
        version = self.db.create_version_branch(
            session['trajectory_id'], candidate, from_branch=active_branch_id, snapshot_id=snapshot_id,
            reason=reason or 'Agent accepted repair branch.', origin='agent',
            refs={'version_decision_event_id': decision['event_id'], 'suggestion_event_id': suggestion.get('suggestion_event_id')},
        )
        session['active_branch_id'] = candidate
        session['last_repair_suggestion'] = None
        self._touch_session(source, session_id, session, 2)
        return {**version, 'decision_event': decision, 'version_context': self._version_context(session)}

    def rollback_context(
        self,
        source: str,
        session_id: str,
        snapshot_id: str,
        target_branch_id: Optional[str] = None,
        reason: str = '',
    ) -> Dict[str, Any]:
        """Create a rollback branch; workspace restoration stays external."""
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        target = self._available_branch_id(session['trajectory_id'], target_branch_id or f'rollback-{snapshot_id.split("_")[-1][:6]}')
        decision = self._record_version_decision(
            session, 'rollback_context', 'accepted', reason or 'Agent selected rollback.',
            {'snapshot_id': snapshot_id, 'target_branch_id': target},
        )
        version = self.db.create_version_rollback(
            session['trajectory_id'], snapshot_id, target, reason=reason or 'Agent requested rollback.',
            origin='agent', refs={'version_decision_event_id': decision['event_id']},
        )
        session['active_branch_id'] = target
        session['last_repair_suggestion'] = None
        self._touch_session(source, session_id, session, 2)
        return {**version, 'decision_event': decision, 'version_context': self._version_context(session)}

    def record_version_decision(
        self,
        source: str,
        session_id: str,
        action: str,
        decision: str,
        reason: str = '',
        suggestion_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if decision not in {'accepted', 'rejected', 'deferred'}:
            raise ValueError('decision must be accepted, rejected, or deferred')
        if action not in {'continue_current_branch', 'create_repair_branch', 'rollback_context'}:
            raise ValueError('unsupported version decision action')
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        event = self._record_version_decision(session, action, decision, reason, {'suggestion_event_id': suggestion_event_id})
        self._touch_session(source, session_id, session, 1)
        return {'trajectory_id': session['trajectory_id'], 'decision_event': event, 'version_context': self._version_context(session)}

    def version_status(self, source: str, session_id: str) -> Dict[str, Any]:
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        return {
            **self.db.version_control_status(session['trajectory_id']),
            'session': session,
            'version_context': self._version_context(session),
        }

    def status(self, source: str, session_id: str) -> Dict[str, Any]:
        session = self.db.store.get_object(self._session_key(source, session_id))
        if not session:
            raise KeyError(f'hook session not found: {source}/{session_id}')
        return {
            'protocol_version': HOOK_PROTOCOL_VERSION,
            'session': session,
            'trajectory': self.db.get_trajectory(session['trajectory_id']),
            'skill_application_trace': self.db.skill_application_trace(session['trajectory_id']),
        }

    def ensure_session(self, source: str, session_id: str, title: Optional[str] = None, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a live session on first MCP turn, or return its persisted state."""
        return self._ensure_session(source, session_id, {'title': title, 'agent_id': agent_id})

    def agent_context(self, source: str, session_id: str) -> Dict[str, Any]:
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        retrieval = session.get('last_skill_retrieval') or {}
        return {
            'protocol_version': HOOK_PROTOCOL_VERSION,
            'source': source,
            'session_id': session_id,
            'trajectory_id': session.get('trajectory_id'),
            'agent_context': retrieval.get('agent_context') or {
                'type': 'contextdb_skill_recommendation',
                'matched': False,
                'instruction': 'No live skill recommendation is available for this session yet.',
                'version_context': self._version_context(session),
            },
        }

    def prepare_context(
        self,
        source: str,
        session_id: str,
        token_budget: int = 1200,
        delivery_channel: str = 'mcp',
        include_prompt_context: bool = True,
    ) -> Dict[str, Any]:
        """Return compact turn context and persist an auditable delivery event.

        An MCP-aware Agent calls this before planning a turn.  Persisting the
        delivery separately from matching keeps ``matched`` and ``seen by the
        Agent`` distinct in the application trace.
        """
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        trajectory_id = session['trajectory_id']
        recommendation = self.agent_context(source, session_id)['agent_context']
        retrieval = session.get('last_skill_retrieval') or {}

        # Before any tool failure, prepare_context has no recommendation to
        # deliver. Returning that empty state is useful to the Agent, but
        # recording it as a skill_recommendation makes a DAG begin with a
        # misleading recommendation node.
        if not retrieval:
            response = {
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': source,
                'session_id': session_id,
                'trajectory_id': trajectory_id,
                'delivery_event_id': None,
                'match_event_id': None,
                'agent_context': recommendation,
                'version_context': self._version_context(session),
            }
            if include_prompt_context:
                response['prompt_context'] = self.db.stream_context(
                    trajectory_id, session.get('active_branch_id') or 'main', token_budget=token_budget,
                ).get('content', {})
            return response
        selected_action = recommendation.get('selected_action') or {}
        refs = {
            'skill_match_event_id': retrieval.get('match_event_id'),
            'skill_id': recommendation.get('skill_id'),
            'selected_action_id': selected_action.get('action_id'),
        }
        refs = {key: value for key, value in refs.items() if value}
        delivery = self.db.append_event(
            trajectory_id,
            'skill_recommendation',
            recommendation,
            branch_id=session.get('active_branch_id') or 'main',
            actor='contextdb',
            refs=refs,
            metadata={
                'operation': 'skill_recommendation_delivery',
                'delivery_channel': delivery_channel,
                'online': True,
            },
        )
        session['last_skill_delivery'] = {
            'delivery_event_id': delivery['event_id'],
            'match_event_id': refs.get('skill_match_event_id'),
            'matched': bool(recommendation.get('matched')),
            'delivered_at': utc_now(),
        }
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + 1
        self._put_session(source, session_id, session)
        response = {
            'protocol_version': HOOK_PROTOCOL_VERSION,
            'source': source,
            'session_id': session_id,
            'trajectory_id': trajectory_id,
            'delivery_event_id': delivery['event_id'],
            'match_event_id': refs.get('skill_match_event_id'),
            'agent_context': recommendation,
            'version_context': self._version_context(session),
        }
        if include_prompt_context:
            response['prompt_context'] = self.db.stream_context(
                trajectory_id, session.get('active_branch_id') or 'main', token_budget=token_budget,
            ).get('content', {})
        return response

    def record_skill_decision(
        self,
        source: str,
        session_id: str,
        decision: str,
        reason: str = '',
        skill_match_event_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record that an Agent accepted, rejected, or deferred a delivered skill."""
        if decision not in {'accepted', 'rejected', 'deferred'}:
            raise ValueError('decision must be accepted, rejected, or deferred')
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        recommendation = self.agent_context(source, session_id)['agent_context']
        selected_action = recommendation.get('selected_action') or {}
        last_retrieval = session.get('last_skill_retrieval') or {}
        refs = {
            'skill_match_event_id': skill_match_event_id or last_retrieval.get('match_event_id'),
            'skill_id': skill_id or recommendation.get('skill_id'),
            'selected_action_id': action_id or selected_action.get('action_id'),
        }
        refs = {key: value for key, value in refs.items() if value}
        event = self.db.append_event(
            session['trajectory_id'],
            'skill_decision',
            {
                'decision': decision,
                'reason': reason,
                'skill_id': refs.get('skill_id'),
                'action_id': refs.get('selected_action_id'),
            },
            branch_id=session.get('active_branch_id') or 'main', actor='agent',
            refs=refs,
            metadata={'operation': 'skill_decision', 'online': True, 'integration': 'agent-hook.v1'},
        )
        session['last_skill_decision'] = {'event_id': event['event_id'], 'decision': decision, 'at': utc_now()}
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + 1
        self._put_session(source, session_id, session)
        return {'trajectory_id': session['trajectory_id'], 'decision_event_id': event['event_id'], 'decision': decision}

    def record_skill_application(
        self,
        source: str,
        session_id: str,
        tool_name: str,
        command: str,
        status: str,
        preview: str = '',
        exit_code: Optional[int] = None,
        skill_match_event_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        action_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_call_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a real Agent-executed skill action; this method never executes it."""
        self.ensure_session(source, session_id)
        session = self.status(source, session_id)['session']
        recommendation = self.agent_context(source, session_id)['agent_context']
        selected_action = recommendation.get('selected_action') or {}
        last_retrieval = session.get('last_skill_retrieval') or {}
        refs = {
            'skill_match_event_id': skill_match_event_id or last_retrieval.get('match_event_id'),
            'skill_id': skill_id or recommendation.get('skill_id'),
            'selected_action_id': action_id or selected_action.get('action_id'),
        }
        refs = {key: value for key, value in refs.items() if value}
        metadata = {'operation': 'skill_application', 'online': True, 'integration': 'agent-hook.v1'}
        bridge = AgentContextBridge(self.db, agent_id=session['agent_id'], source_id=f'hook:{source}')
        transport_tool_name = tool_name
        pending = session['pending_tools'].pop(tool_call_id, None) if tool_call_id else None
        existing_call_id = tool_call_event_id or (pending or {}).get('event_id')
        if not existing_call_id and source == 'codex-session':
            existing_call_id = self._recent_matching_tool_call(
                session['trajectory_id'], session.get('active_branch_id') or 'main', tool_name, command,
            )
        if existing_call_id:
            call = self.db.get_event(session['trajectory_id'], existing_call_id)
            if call.get('event_type') != 'tool_call':
                raise ValueError('tool_call_event_id must reference a tool_call event')
            previous_status = str((call.get('payload') or {}).get('status') or 'pending')
            # Never reinterpret an already-completed failed call as the later
            # successful skill application.  The latter needs its own call.
            if previous_status != 'pending' and previous_status != status:
                existing_call_id = None
        if existing_call_id:
            call = self.db.get_event(session['trajectory_id'], existing_call_id)
            application_branch_id = call.get('branch_id') or session.get('active_branch_id') or 'main'
            transport_tool_name = _transport_tool_name(call.get('payload') or {})
            tool_chain = list((call.get('payload') or {}).get('tool_chain') or _tool_chain(transport_tool_name, command))
        else:
            application_branch_id = session.get('active_branch_id') or 'main'
            call = bridge.record(
                session['trajectory_id'], 'tool_call', _tool_payload(tool_name, command),
                branch_id=application_branch_id, actor='agent', refs=refs, metadata=metadata,
            )
            tool_chain = list((call.get('payload') or {}).get('tool_chain') or _tool_chain(tool_name, command))
        result_payload = _result_tool_payload(
            transport_tool_name, command, status, preview=preview,
            exit_code=exit_code, tool_chain=tool_chain,
        )
        result = bridge.record_tool_result(
            session['trajectory_id'], result_payload['tool_name'], command, status, preview=result_payload['preview'],
            branch_id=application_branch_id, exit_code=result_payload['exit_code'],
            refs={**refs, 'tool_call_event_id': call['event_id']}, metadata=metadata,
            extra_payload={
                key: value for key, value in result_payload.items()
                if key not in {'tool_name', 'command', 'status', 'preview', 'exit_code'}
            },
        )
        self.db.update_tool_call_outcome(
            session['trajectory_id'], call['event_id'], status=status,
            result_tool_name=result_payload['tool_name'], tool_chain=result_payload['tool_chain'],
            result_event_id=result['tool_result']['event_id'],
        )
        suggestion = None
        if status in {'failed', 'error', 'timeout'}:
            suggestion = self._suggest_repair_branch(session, result['tool_result'], application_branch_id, refs)
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + (1 if existing_call_id else 2) + (1 if suggestion else 0)
        self._put_session(source, session_id, session)
        retrieval = result.get('skill_retrieval')
        if retrieval:
            recommendation_result = self._skill_recommendation(retrieval)
            recommendation_result['agent_context']['version_context'] = self._version_context(session)
            session['last_skill_retrieval'] = recommendation_result
            self._put_session(source, session_id, session)
        return {
            'trajectory_id': session['trajectory_id'],
            'tool_call_event_id': call['event_id'],
            'tool_result_event_id': result['tool_result']['event_id'],
            'status': status,
            'next_skill_retrieval': retrieval,
            'repair_branch_suggestion': suggestion.get('suggestion') if suggestion else None,
            'version_context': self._version_context(session),
        }

    def _recent_matching_tool_call(
        self,
        trajectory_id: str,
        branch_id: str,
        tool_name: str,
        command: str,
    ) -> Optional[str]:
        """Find the native watcher call that immediately preceded an application."""
        for event in reversed(self.db.list_events(trajectory_id, branch_id)):
            if event.get('event_type') != 'tool_call':
                continue
            payload = event.get('payload') or {}
            chain = payload.get('tool_chain') or []
            if (
                payload.get('command') == command
                and tool_name in {payload.get('tool_name'), payload.get('result_tool_name'), *(chain or [])}
            ):
                return event.get('event_id')
        return None

    def _ensure_session(self, source: str, session_id: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
        key = self._session_key(source, session_id)
        session = self.db.store.get_object(key)
        if session:
            changed = False
            for name, value in {
                'active_branch_id': 'main',
                'last_version_snapshot': None,
                'last_repair_suggestion': None,
                'last_version_decision': None,
            }.items():
                if name not in session:
                    session[name] = value
                    changed = True
            if session.get('schema_version') != 'contextdb.hook_session.v2':
                session['schema_version'] = 'contextdb.hook_session.v2'
                changed = True
            if changed:
                self._put_session(source, session_id, session)
            return session
        agent_id = str(envelope.get('agent_id') or ('codex' if source in {'codex', 'codex-session'} else source))
        title = str(envelope.get('title') or f'{agent_id} live hook session {session_id}')
        trajectory = self.db.create_trajectory(title, agent_id=agent_id, source_id=f'hook:{source}', metadata={
            'integration': 'agent-hook.v1',
            'hook_protocol': HOOK_PROTOCOL_VERSION,
            'hook_source': source,
            'hook_session_id': session_id,
        })
        session = {
            'schema_version': 'contextdb.hook_session.v2',
            'source': source,
            'session_id': session_id,
            'agent_id': agent_id,
            'trajectory_id': trajectory['trajectory_id'],
            'created_at': utc_now(),
            'updated_at': utc_now(),
            'event_count': 0,
            'pending_tools': {},
            'last_skill_retrieval': None,
            'last_skill_delivery': None,
            'last_skill_decision': None,
            'active_branch_id': 'main',
            'last_version_snapshot': None,
            'last_repair_suggestion': None,
            'last_version_decision': None,
        }
        self._put_session(source, session_id, session)
        return session

    def _put_session(self, source: str, session_id: str, value: Dict[str, Any]) -> None:
        self.db.store.create_namespace('hook_sessions')
        self.db.store.put_object(self._session_key(source, session_id), value)

    def _session_key(self, source: str, session_id: str) -> str:
        return 'hook_sessions/%s/%s' % (quote(source, safe='-_.').lower(), quote(session_id, safe='-_ .').replace(' ', '_'))

    def _touch_session(self, source: str, session_id: str, session: Dict[str, Any], count: int) -> None:
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + count
        self._put_session(source, session_id, session)

    @staticmethod
    def _snapshot_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'snapshot_id': snapshot.get('snapshot_id'),
            'branch_id': snapshot.get('branch_id'),
            'event_id': snapshot.get('event_id'),
            'message': snapshot.get('message', ''),
            'created_at': snapshot.get('created_at'),
        }

    def _version_context(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'active_branch_id': session.get('active_branch_id') or 'main',
            'latest_snapshot': session.get('last_version_snapshot'),
            'repair_branch_suggestion': session.get('last_repair_suggestion'),
            'last_version_decision': session.get('last_version_decision'),
            'workspace_restore': {
                'supported': False,
                'instruction': 'ContextDB records branch and rollback state only. Restore files through the Agent workspace after an explicit decision.',
            },
        }

    def _auto_snapshot_before_tool(self, session: Dict[str, Any], payload: Dict[str, Any], refs: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = _transport_tool_name(payload)
        command = _text(payload.get('command') or payload.get('cmd') or '')
        version = self.db.create_version_snapshot(
            session['trajectory_id'], session.get('active_branch_id') or 'main',
            message=f'Before state-changing {tool_name}',
            reason=f'Hook policy detected a potentially state-changing action: {command[:240]}',
            origin='hook_policy', actor='contextdb',
            refs={**refs, 'planned_tool_name': tool_name, 'planned_command': command[:1000]},
        )
        session['last_version_snapshot'] = self._snapshot_context(version['snapshot'])
        return version

    def _suggest_repair_branch(
        self,
        session: Dict[str, Any],
        failure_event: Dict[str, Any],
        branch_id: str,
        refs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        snapshot = session.get('last_version_snapshot') or {}
        if snapshot.get('branch_id') != branch_id or not snapshot.get('snapshot_id'):
            return None
        suffix = str(failure_event.get('event_id') or '').split('_')[-1][:8] or 'failure'
        suggested_branch_id = self._available_branch_id(session['trajectory_id'], f'repair-{suffix}')
        event = self.db.append_event(
            session['trajectory_id'], 'system_event',
            {
                'text': f'ContextDB suggests repair branch {suggested_branch_id} from snapshot {snapshot["snapshot_id"]}.',
                'failure_event_id': failure_event.get('event_id'),
                'snapshot_id': snapshot['snapshot_id'],
                'suggested_branch_id': suggested_branch_id,
                'source_branch_id': branch_id,
                'logical_context_only': True,
            },
            branch_id=branch_id, actor='contextdb',
            refs={**refs, 'failure_event_id': failure_event.get('event_id'), 'snapshot_id': snapshot['snapshot_id']},
            metadata={'operation': 'version_repair_branch_suggested', 'origin': 'hook_policy', 'logical_context_only': True},
        )
        session['last_repair_suggestion'] = {
            'suggestion_event_id': event['event_id'],
            'failure_event_id': failure_event.get('event_id'),
            'snapshot_id': snapshot['snapshot_id'],
            'suggested_branch_id': suggested_branch_id,
            'source_branch_id': branch_id,
            'created_at': utc_now(),
        }
        return {'event': event, 'suggestion': session['last_repair_suggestion']}

    def _record_version_decision(
        self,
        session: Dict[str, Any],
        action: str,
        decision: str,
        reason: str,
        refs: Dict[str, Any],
    ) -> Dict[str, Any]:
        event = self.db.append_event(
            session['trajectory_id'], 'version_decision',
            {'action': action, 'decision': decision, 'reason': reason, 'logical_context_only': True},
            branch_id=session.get('active_branch_id') or 'main', actor='agent', refs={key: value for key, value in refs.items() if value},
            metadata={'operation': 'version_decision', 'origin': 'agent', 'logical_context_only': True, 'online': True},
        )
        session['last_version_decision'] = {'event_id': event['event_id'], 'action': action, 'decision': decision, 'at': utc_now()}
        return event

    def _available_branch_id(self, trajectory_id: str, preferred: str) -> str:
        base = re.sub(r'[^a-zA-Z0-9._-]+', '-', preferred.strip()).strip('-') or 'repair'
        candidate, number = base, 2
        while True:
            try:
                self.db.get_branch(trajectory_id, candidate)
            except KeyError:
                return candidate
            candidate = f'{base}-{number}'
            number += 1

    def _skill_recommendation(self, retrieval: Dict[str, Any]) -> Dict[str, Any]:
        match = retrieval.get('match') or {}
        selected = (match.get('matches') or [None])[0]
        if not selected:
            return {
                'match_event_id': retrieval.get('match_event_id'),
                'matched': False,
                'agent_context': {
                    'type': 'contextdb_skill_recommendation',
                    'matched': False,
                    'instruction': 'No reusable ContextDB skill matched this failure. Continue with normal diagnosis.',
                },
            }
        actions = list(selected.get('recommended_actions') or [])
        best_action, best_score, best_index = None, -1.0, None
        for index, action in enumerate(actions):
            try:
                score = float((action.get('confidence') or {}).get('max_llm_confidence', 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score > best_score:
                best_action, best_score, best_index = action, score, index
        context = {
            'type': 'contextdb_skill_recommendation',
            'matched': True,
            'skill_id': selected.get('matched_skill_id'),
            'skill_name': selected.get('skill_name'),
            'match_score': selected.get('score'),
            'match_reason': selected.get('match_reason'),
            'selected_action': best_action,
            'selected_action_index': best_index,
            'selected_action_score': max(best_score, 0.0),
            'selection_policy': 'highest max_llm_confidence; ties keep original action order',
            'instruction': 'Use this as a suggestion. The Agent must decide whether it is safe and appropriate before executing it.',
        }
        return {'match_event_id': retrieval.get('match_event_id'), 'matched': True, 'agent_context': context}


def post_hook_event(base_url: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(envelope, ensure_ascii=False).encode('utf-8')
    req = request.Request(base_url.rstrip('/') + '/api/v1/hooks/events', data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))


def stream_hook_events(base_url: str, source: str = 'codex', session_id: Optional[str] = None, title: Optional[str] = None, agent_id: Optional[str] = None, echo: bool = True) -> int:
    """Forward newline-delimited live agent events from stdin to ContextDB."""
    adapters: Dict[str, AgentHookAdapter] = {
        'codex': CodexExecJSONLHookAdapter(),
        'codex-session': CodexDesktopSessionHookAdapter(),
        'generic': GenericHookAdapter(),
    }
    adapter = adapters.get(source, adapters['generic'])
    active_session_id = session_id
    failures = 0
    for line_no, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        if echo:
            print(line.rstrip('\n'), flush=True)
        try:
            raw_event = json.loads(line)
            if not isinstance(raw_event, dict):
                raise ValueError('JSONL record is not an object')
            active_session_id = active_session_id or adapter.infer_session_id(raw_event)
            if not active_session_id:
                raise ValueError('use --session-id when the source does not emit an initial session id')
            response = post_hook_event(base_url, {
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': source,
                'session_id': active_session_id,
                'title': title,
                'agent_id': agent_id,
                'event': raw_event,
            })
            for retrieval in response.get('skill_retrievals', []):
                print('CONTEXTDB_SKILL_HOOK ' + json.dumps(retrieval, ensure_ascii=False), file=sys.stderr, flush=True)
        except Exception as exc:
            failures += 1
            print(f'ContextDB hook error at JSONL line {line_no}: {exc}', file=sys.stderr, flush=True)
    return 1 if failures else 0
