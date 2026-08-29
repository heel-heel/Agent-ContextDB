from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from contextdb.hooks import CodexDesktopSessionHookAdapter, CodexExecJSONLHookAdapter, HookSessionBridge, HOOK_PROTOCOL_VERSION
from contextdb.service import ContextDB


def test_codex_jsonl_hook_persists_events_and_retrieves_skill():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        skill = {
            'skill_id': 'skill_compiler_repair',
            'name': 'Repair compiler incompatibility',
            'trigger': {'failed_tool': 'shell', 'failed_command_pattern': 'cargo build --release', 'normalized_signature': 'gcc rejected aws-lc-sys generated memcmp code'},
            'recommended_actions': [{
                'action_id': 'act_use_clang',
                'name': 'Build with Clang',
                'canonical_tool': 'shell',
                'canonical_command_template': 'CC=clang cargo build --release',
                'confidence': {'max_llm_confidence': 0.91},
            }],
            'status': 'validated',
            'highlight_event_ids': [],
        }
        db.vector_index.replace_owner('skills', 'history', [{
            'entry_id': 'history:skill_compiler_repair',
            'document': 'Repair compiler incompatibility cargo build release gcc rejected aws-lc-sys generated memcmp code',
            'metadata': {'skill': skill, 'trajectory_id': 'history'},
        }])
        adapter = CodexExecJSONLHookAdapter()
        bridge = HookSessionBridge(db)
        fixture = Path(__file__).resolve().parents[1] / 'examples' / 'codex_hook_fixture.jsonl'
        results = []
        for line in fixture.read_text(encoding='utf-8').splitlines():
            raw = json.loads(line)
            results.append(bridge.ingest({
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': 'codex',
                'session_id': adapter.infer_session_id(raw) or 'thread_contextdb_fixture',
                'title': 'Codex hook test',
                'event': raw,
            }))
        failure_result = results[-2]
        assert failure_result['skill_retrievals']
        recommendation = failure_result['skill_retrievals'][0]['agent_context']
        assert recommendation['matched'] is True
        assert recommendation['selected_action']['action_id'] == 'act_use_clang'
        status = bridge.status('codex', 'thread_contextdb_fixture')
        event_types = [event['event_type'] for event in db.list_events(status['session']['trajectory_id'])]
        assert 'tool_call' in event_types
        assert 'tool_result' in event_types
        assert 'skill_match' in event_types
        assert any(event['event_type'] == 'skill_match' for event in status['skill_application_trace'])
        db.vector_index.conn.close()
        db.store.conn.close()


def test_codex_desktop_session_hook_persists_native_app_records():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        skill = {
            'skill_id': 'skill_compiler_repair',
            'name': 'Repair compiler incompatibility',
            'trigger': {
                'failed_tool': 'functions.exec_command',
                'failed_command_pattern': 'cargo build --release',
                'normalized_signature': 'gcc rejected aws-lc-sys generated memcmp code',
            },
            'recommended_actions': [{'action_id': 'act_use_clang', 'confidence': {'max_llm_confidence': 0.91}}],
            'status': 'validated',
            'highlight_event_ids': [],
        }
        db.vector_index.replace_owner('skills', 'history', [{
            'entry_id': 'history:skill_compiler_repair',
            'document': 'Repair compiler incompatibility cargo build release gcc rejected aws-lc-sys generated memcmp code',
            'metadata': {'skill': skill, 'trajectory_id': 'history'},
        }])
        adapter = CodexDesktopSessionHookAdapter()
        bridge = HookSessionBridge(db)
        fixture = Path(__file__).resolve().parents[1] / 'examples' / 'codex_desktop_session_fixture.jsonl'
        results = []
        for line in fixture.read_text(encoding='utf-8').splitlines():
            raw = json.loads(line)
            results.append(bridge.ingest({
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': 'codex-session',
                'session_id': adapter.infer_session_id(raw) or 'desktop_contextdb_fixture',
                'title': 'Codex App hook test',
                'event': raw,
            }))
        assert results[-1]['skill_retrievals']
        recommendation = results[-1]['skill_retrievals'][0]['agent_context']
        assert recommendation['matched'] is True
        assert recommendation['selected_action']['action_id'] == 'act_use_clang'
        status = bridge.status('codex-session', 'desktop_contextdb_fixture')
        events = db.list_events(status['session']['trajectory_id'])
        assert [event['event_type'] for event in events].count('tool_call') == 1
        assert [event['event_type'] for event in events].count('tool_result') == 1
        assert any(event['event_type'] == 'skill_match' for event in status['skill_application_trace'])
        db.vector_index.conn.close()
        db.store.conn.close()


def test_codex_desktop_session_hook_persists_custom_tool_events():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        bridge = HookSessionBridge(db)
        records = [
            {"type": "session_meta", "payload": {"session_id": "custom_tool_fixture", "cwd": "C:\\work"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "id": "custom_call_item",
                    "call_id": "custom_call",
                    "name": "exec",
                    "input": 'const result = await tools.exec_command({"cmd":"python -c \\"import missing_probe\\""});',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "id": "custom_output_item",
                    "call_id": "custom_call",
                    "output": [{"type": "input_text", "text": '[{"exit_code":1,"output":"ModuleNotFoundError: missing_probe"}]'}],
                },
            },
        ]
        for raw in records:
            bridge.ingest({
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': 'codex-session',
                'session_id': 'custom_tool_fixture',
                'event': raw,
            })
        status = bridge.status('codex-session', 'custom_tool_fixture')
        events = db.list_events(status['session']['trajectory_id'])
        call = next(event for event in events if event['event_type'] == 'tool_call')
        result = next(event for event in events if event['event_type'] == 'tool_result')
        assert call['payload']['tool_name'] == 'exec'
        assert call['payload']['command'] == 'python -c "import missing_probe"'
        assert result['payload']['status'] == 'failed'
        assert result['payload']['exit_code'] == 1
        assert any(event['event_type'] == 'skill_match' for event in events)
        db.vector_index.conn.close()
        db.store.conn.close()
