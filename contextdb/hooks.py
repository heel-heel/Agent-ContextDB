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
                yield HookEvent('tool_call', {'tool_name': item.get('tool_name') or 'shell', 'command': command}, 'agent', item_id, metadata=metadata)
            elif event_type == 'item.completed':
                yield HookEvent('tool_result', {
                    'tool_name': item.get('tool_name') or 'shell',
                    'command': command,
                    'status': _tool_status(item),
                    'preview': _tool_output(item)[:4000],
                    'exit_code': item.get('exit_code'),
                }, 'tool', item_id, metadata=metadata)
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
                {"text": "Codex App session started.", "cwd": payload.get("cwd")},
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
                {"tool_name": str(payload.get("name") or "tool"), "command": command, "arguments": arguments},
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
                {
                    "tool_name": result.get("tool_name") or "tool",
                    "command": result.get("command") or "",
                    "status": result["status"],
                    "preview": result["preview"],
                    "exit_code": result.get("exit_code"),
                },
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
                event = bridge.record(trajectory_id, 'tool_call', item.payload, actor='agent', refs=refs, metadata=metadata)
                if item.external_id:
                    session['pending_tools'][item.external_id] = {
                        'event_id': event['event_id'],
                        'tool_name': item.payload.get('tool_name') or 'shell',
                        'command': item.payload.get('command') or '',
                    }
                emitted.append(event)
                continue
            if item.event_type == 'tool_result':
                pending = session['pending_tools'].pop(item.external_id, None) if item.external_id else None
                tool_name = item.payload.get('tool_name') or (pending or {}).get('tool_name') or 'shell'
                command = item.payload.get('command') or (pending or {}).get('command') or ''
                if pending:
                    refs['tool_call_event_id'] = pending['event_id']
                else:
                    implicit = bridge.record(trajectory_id, 'tool_call', {'tool_name': tool_name, 'command': command}, actor='agent', refs={**refs, 'implicit_from_hook_result': True}, metadata=metadata)
                    refs['tool_call_event_id'] = implicit['event_id']
                    emitted.append(implicit)
                recorded = bridge.record_tool_result(
                    trajectory_id, tool_name, command, item.payload.get('status', 'ok'),
                    preview=item.payload.get('preview', ''), exit_code=item.payload.get('exit_code'),
                    refs=refs, metadata=metadata,
                )
                emitted.append(recorded['tool_result'])
                if recorded.get('skill_retrieval'):
                    retrieval = self._skill_recommendation(recorded['skill_retrieval'])
                    retrievals.append(retrieval)
                    session['last_skill_retrieval'] = retrieval
                continue
            event = bridge.record(trajectory_id, item.event_type, item.payload, actor=item.actor, refs=refs, metadata=metadata)
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
            },
        }

    def prepare_context(
        self,
        source: str,
        session_id: str,
        token_budget: int = 1200,
        delivery_channel: str = 'mcp',
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
        selected_action = recommendation.get('selected_action') or {}
        refs = {
            'skill_match_event_id': (session.get('last_skill_retrieval') or {}).get('match_event_id'),
            'skill_id': recommendation.get('skill_id'),
            'selected_action_id': selected_action.get('action_id'),
        }
        refs = {key: value for key, value in refs.items() if value}
        delivery = self.db.append_event(
            trajectory_id,
            'skill_recommendation',
            recommendation,
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
        return {
            'protocol_version': HOOK_PROTOCOL_VERSION,
            'source': source,
            'session_id': session_id,
            'trajectory_id': trajectory_id,
            'delivery_event_id': delivery['event_id'],
            'match_event_id': refs.get('skill_match_event_id'),
            'agent_context': recommendation,
            'prompt_context': self.db.stream_context(trajectory_id, token_budget=token_budget).get('content', {}),
        }

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
            actor='agent',
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
        call = bridge.record(
            session['trajectory_id'], 'tool_call', {'tool_name': tool_name, 'command': command},
            actor='agent', refs=refs, metadata=metadata,
        )
        result = bridge.record_tool_result(
            session['trajectory_id'], tool_name, command, status, preview=preview,
            exit_code=exit_code, refs={**refs, 'tool_call_event_id': call['event_id']}, metadata=metadata,
        )
        session['updated_at'] = utc_now()
        session['event_count'] = int(session.get('event_count', 0)) + 2
        self._put_session(source, session_id, session)
        retrieval = result.get('skill_retrieval')
        if retrieval:
            recommendation_result = self._skill_recommendation(retrieval)
            session['last_skill_retrieval'] = recommendation_result
            self._put_session(source, session_id, session)
        return {
            'trajectory_id': session['trajectory_id'],
            'tool_call_event_id': call['event_id'],
            'tool_result_event_id': result['tool_result']['event_id'],
            'status': status,
            'next_skill_retrieval': retrieval,
        }

    def _ensure_session(self, source: str, session_id: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
        key = self._session_key(source, session_id)
        session = self.db.store.get_object(key)
        if session:
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
            'schema_version': 'contextdb.hook_session.v1',
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
        }
        self._put_session(source, session_id, session)
        return session

    def _put_session(self, source: str, session_id: str, value: Dict[str, Any]) -> None:
        self.db.store.create_namespace('hook_sessions')
        self.db.store.put_object(self._session_key(source, session_id), value)

    def _session_key(self, source: str, session_id: str) -> str:
        return 'hook_sessions/%s/%s' % (quote(source, safe='-_.').lower(), quote(session_id, safe='-_ .').replace(' ', '_'))

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
