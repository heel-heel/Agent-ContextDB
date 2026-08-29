from __future__ import annotations

from typing import Any, Dict, Optional

from .service import ContextDB


class AgentContextBridge:
    """Framework-neutral online bridge used by hooks and future agent adapters."""

    def __init__(self, db: ContextDB, agent_id: str, source_id: str = 'live-agent'):
        self.db, self.agent_id, self.source_id = db, agent_id, source_id

    def start(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.db.create_trajectory(title, agent_id=self.agent_id, source_id=self.source_id, metadata={'integration': 'agent-context-bridge.v1', **(metadata or {})})

    def record(self, trajectory_id: str, event_type: str, payload: Dict[str, Any], branch_id: str = 'main', actor: str = 'agent', refs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.db.append_event(
            trajectory_id, event_type, payload, branch_id=branch_id, actor=actor,
            refs=refs, metadata={'agent_id': self.agent_id, 'online': True, **(metadata or {})},
        )

    def record_tool_result(self, trajectory_id: str, tool: str, command: Any, status: str, preview: str = '', branch_id: str = 'main', exit_code: Optional[int] = None, refs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = self.record(
            trajectory_id, 'tool_result', {'status': status, 'preview': preview, 'exit_code': exit_code},
            branch_id, 'tool', refs=refs, metadata=metadata,
        )
        retrieval = None
        if status in {'failed', 'error', 'timeout'}:
            retrieval = self.db.retrieve_for_failure(
                trajectory_id,
                {'tool': tool, 'command': command, 'error_signature': preview},
                branch_id=branch_id,
                source_event_id=result['event_id'],
            )
        return {'tool_result': result, 'skill_retrieval': retrieval}


class CodexContextBridge(AgentContextBridge):
    def __init__(self, db: ContextDB, source_id: str = 'codex-live'):
        super().__init__(db, agent_id='codex', source_id=source_id)
