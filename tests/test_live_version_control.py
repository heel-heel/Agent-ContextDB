from __future__ import annotations

from tempfile import TemporaryDirectory

from contextdb.hooks import HOOK_PROTOCOL_VERSION, HookSessionBridge
from contextdb.service import ContextDB


def _event(event_type: str, payload: dict, event_id: str | None = None) -> dict:
    return {
        'protocol_version': HOOK_PROTOCOL_VERSION,
        'source': 'generic',
        'session_id': 'version-policy-session',
        'adapter': 'generic',
        'event': {'event_type': event_type, 'event_id': event_id, 'actor': 'tool' if event_type == 'tool_result' else 'agent', 'payload': payload},
    }


def test_live_hook_creates_checkpoint_and_agent_controlled_repair_path():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        bridge = HookSessionBridge(db)

        # The hook detects this as a possible state-changing command and creates
        # a ContextDB-only checkpoint before recording the real tool call.
        call = bridge.ingest(_event('tool_call', {'tool_name': 'powershell', 'command': "Set-Content -Path '.\\config.json' -Value '{}'"}, 'write-1'))
        session = bridge.status('generic', 'version-policy-session')['session']
        snapshot = session['last_version_snapshot']
        assert snapshot and snapshot['snapshot_id']
        assert call['version_context']['active_branch_id'] == 'main'

        failed = bridge.ingest(_event('tool_result', {
            'tool_name': 'powershell', 'command': "Set-Content -Path '.\\config.json' -Value '{}'",
            'status': 'failed', 'preview': 'Access is denied', 'exit_code': 1,
        }, 'write-1'))
        session = bridge.status('generic', 'version-policy-session')['session']
        suggestion = session['last_repair_suggestion']
        assert suggestion and suggestion['snapshot_id'] == snapshot['snapshot_id']
        assert failed['version_context']['repair_branch_suggestion']['suggested_branch_id'] == suggestion['suggested_branch_id']

        repair = bridge.create_repair_branch('generic', 'version-policy-session', reason='Try a safe repair without losing the original failure path.')
        repair_branch = repair['branch']['branch_id']
        assert repair_branch == suggestion['suggested_branch_id']
        assert repair['version_context']['active_branch_id'] == repair_branch

        retry = bridge.ingest(_event('tool_call', {'tool_name': 'powershell', 'command': "Get-Content -LiteralPath '.\\config.json'"}, 'read-2'))
        assert retry['version_context']['active_branch_id'] == repair_branch
        bridge.ingest(_event('tool_result', {
            'tool_name': 'powershell', 'command': "Get-Content -LiteralPath '.\\config.json'",
            'status': 'ok', 'preview': '{}', 'exit_code': 0,
        }, 'read-2'))

        rolled_back = bridge.rollback_context('generic', 'version-policy-session', snapshot['snapshot_id'], reason='Preserve a separate recovery path.')
        assert rolled_back['branch']['snapshot_id'] == snapshot['snapshot_id']
        assert rolled_back['version_context']['active_branch_id'] == rolled_back['branch']['branch_id']

        graph = db.graph(session['trajectory_id'])
        repair_edge = next(edge for edge in graph['edges'] if edge.get('kind') == 'branch' and edge.get('branch_id') == repair_branch)
        assert repair_edge == {
            'source': repair['branch']['base_event_id'],
            'target': repair['event']['event_id'],
            'kind': 'branch',
            'branch_id': repair_branch,
        }

        status = bridge.version_status('generic', 'version-policy-session')
        operations = {(event.get('metadata') or {}).get('operation') for event in status['version_events']}
        assert {'version_snapshot_created', 'version_repair_branch_suggested', 'version_branch_created', 'version_rollback_created', 'version_decision'} <= operations
        assert status['workspace_restore']['supported'] is False
        db.vector_index.conn.close()
        db.store.conn.close()
