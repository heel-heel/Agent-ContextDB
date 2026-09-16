from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from contextdb.hooks import CodexDesktopSessionHookAdapter, CodexExecJSONLHookAdapter, HookSessionBridge, HOOK_PROTOCOL_VERSION, _invoked_tool_name, _result_tool_payload, _should_auto_snapshot, _tool_chain
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
        result = next(event for event in events if event['event_type'] == 'tool_result')
        assert result['payload']['tool_name'] == 'functions.exec_command'
        assert result['payload']['command'] == 'cargo build --release'
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
        assert 'tool_name' not in call['payload']
        assert 'transport_tool_name' not in call['payload']
        assert call['payload']['command'] == 'python -c "import missing_probe"'
        assert call['payload']['tool_chain'] == ['exec', 'python']
        assert call['payload']['status'] == 'failed'
        assert call['payload']['failure_tool_name'] == 'python'
        assert result['payload']['tool_name'] == 'python'
        assert result['payload']['tool_chain'] == ['exec', 'python']
        assert result['payload']['failure_tool_name'] == 'python'
        assert result['payload']['command'] == 'python -c "import missing_probe"'
        assert result['payload']['status'] == 'failed'
        assert result['payload']['exit_code'] == 1
        assert any(event['event_type'] == 'skill_match' for event in events)
        application = bridge.record_skill_application(
            'codex-session', 'custom_tool_fixture', 'exec',
            'python -c "import missing_probe"', 'ok', preview='repair completed', exit_code=0,
        )
        assert application['tool_call_event_id'] != call['event_id']
        events = db.list_events(status['session']['trajectory_id'])
        assert [event['event_type'] for event in events].count('tool_call') == 2
        original_call = next(event for event in events if event['event_id'] == call['event_id'])
        assert original_call['payload']['status'] == 'failed'
        db.vector_index.conn.close()
        db.store.conn.close()


def test_codex_desktop_exec_calls_follow_the_version_snapshot_policy():
    assert _should_auto_snapshot({
        'tool_name': 'functions.exec_command',
        'command': 'git checkout --detach contextdb-demo-second-001',
    })
    assert _should_auto_snapshot({
        'tool_name': 'exec',
        'command': 'powershell -NoProfile -Command "Set-Content -LiteralPath note.txt -Value test"',
    })
    assert _should_auto_snapshot({
        'tool_name': 'bash',
        'command': 'python -c "open(\'contextdb-version-first-parent-absent/note.txt\', \'w\').write(\'first note\')"',
    })
    assert _should_auto_snapshot({
        'tool_name': 'bash',
        'command': 'node -e "require(\'fs\').writeFileSync(\'contextdb-version-first-parent-absent/note.txt\', \'first note\')"',
    })
    assert _should_auto_snapshot({
        'tool_name': 'ordinary native terminal',
        'command': 'node -e "require(\'fs\').writeFileSync(\'contextdb-version-first-parent-absent/note.txt\', \'first note\')"',
    })
    assert _should_auto_snapshot({
        'tool_name': 'bash',
        'command': 'node -e "require(\'fs\').copyFileSync(\'contextdb-version-first-missing-source.txt\', \'contextdb-version-first-copy.txt\')"',
    })
    assert not _should_auto_snapshot({
        'tool_name': 'functions.exec',
        'command': 'git status --short',
    })
    assert not _should_auto_snapshot({
        'tool_name': 'exec',
        'command': 'const r = await tools.mcp__contextdb__contextdb_record_version_decision({});',
    })
    assert not _should_auto_snapshot({
        'tool_name': 'exec',
        'command': 'const r = await tools.mcp__contextdb__contextdb_create_repair_branch({});',
        'contextdb_snapshot': True,
    })
    assert not _should_auto_snapshot({
        'tool_name': 'exec',
        'command': (
            'const created = await tools.mcp__contextdb__contextdb_create_repair_branch({}); '
            'const status = await tools.mcp__contextdb__contextdb_get_version_status({});'
        ),
    })


def test_contextdb_mcp_calls_never_create_automatic_snapshots():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        bridge = HookSessionBridge(db)
        try:
            for index, operation in enumerate((
                'contextdb_record_version_decision',
                'contextdb_create_repair_branch',
                'contextdb_rollback_context',
            )):
                bridge.ingest({
                    'protocol_version': HOOK_PROTOCOL_VERSION,
                    'source': 'codex-session',
                    'session_id': 'contextdb-control-plane-session',
                    'event': {
                        'event_type': 'tool_call',
                        'event_id': f'control-{index}',
                        'actor': 'agent',
                        'payload': {
                            'tool_name': 'exec',
                            'command': f'const r = await tools.mcp__contextdb__{operation}({{}});',
                        },
                    },
                })

            status = bridge.status('codex-session', 'contextdb-control-plane-session')['session']
            assert status.get('last_version_snapshot') is None
            events = db.list_events(status['trajectory_id'])
            assert not any((event.get('metadata') or {}).get('operation') == 'version_snapshot_created' for event in events)
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()


def test_codex_desktop_nested_exec_tools_preserve_multiple_contextdb_operations():
    assert _tool_chain(
        'exec',
        (
            'const created = await tools.mcp__contextdb__contextdb_create_repair_branch({}); '
            'const status = await tools.mcp__contextdb__contextdb_get_version_status({});'
        ),
    ) == ['exec', 'contextdb_create_repair_branch', 'contextdb_get_version_status']


def test_codex_desktop_nested_exec_tools_keep_transport_and_record_invoked_tool():
    assert _invoked_tool_name('exec', 'const r = await tools.web__run({search_query: []});') == 'web__run'
    assert _invoked_tool_name(
        'exec',
        'const r = await tools.mcp__contextdb__contextdb_get_version_status({});',
    ) == 'contextdb_get_version_status'

    with TemporaryDirectory() as root:
        db = ContextDB(root)
        bridge = HookSessionBridge(db)
        records = [
            {"type": "session_meta", "payload": {"session_id": "nested-tool-fixture"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call", "id": "nested-call-item", "call_id": "nested-call",
                    "name": "exec", "input": 'const r = await tools.web__run({"search_query":[]});',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output", "id": "nested-output-item", "call_id": "nested-call",
                    "output": [{"type": "input_text", "text": '[{"exit_code":0,"output":"ok"}]'}],
                },
            },
        ]
        for raw in records:
            bridge.ingest({
                'protocol_version': HOOK_PROTOCOL_VERSION,
                'source': 'codex-session', 'session_id': 'nested-tool-fixture', 'event': raw,
            })

        session = bridge.status('codex-session', 'nested-tool-fixture')['session']
        events = db.list_events(session['trajectory_id'])
        call = next(event for event in events if event['event_type'] == 'tool_call')
        result = next(event for event in events if event['event_type'] == 'tool_result')
        assert 'tool_name' not in call['payload']
        assert 'transport_tool_name' not in call['payload']
        assert call['payload']['tool_chain'] == ['exec', 'web__run']
        assert 'invoked_tool' not in call['payload']
        assert result['payload']['tool_name'] == 'web__run'
        assert result['payload']['tool_chain'] == ['exec', 'web__run']
        assert result['payload']['invoked_tool'] == 'web__run'
        db.vector_index.conn.close()
        db.store.conn.close()


def test_codex_desktop_terminal_chain_attributes_git_failure_and_powershell_parse_failure():
    command = (
        "powershell -NoProfile -Command '[Environment]::SetEnvironmentVariable((\"LC\" + [char]95 + \"ALL\"), "
        "\"C\", \"Process\"); $env:LANG=\"C\"; git show HEAD:assets/settings.json; exit $LASTEXITCODE'"
    )
    assert _tool_chain('exec', command) == ['exec', 'powershell', 'git']
    failed_git = _result_tool_payload('exec', command, 'failed', 'fatal: path does not exist in HEAD', 128)
    assert failed_git['tool_name'] == 'git'
    assert failed_git['failure_tool_name'] == 'git'
    assert failed_git['tool_chain'] == ['exec', 'powershell', 'git']

    failed_powershell = _result_tool_payload(
        'exec', command, 'failed', 'At line:1 char:8 Unexpected token "_ALL" in expression.', 1,
    )
    assert failed_powershell['tool_name'] == 'powershell'
    assert failed_powershell['failure_tool_name'] == 'powershell'


def test_codex_desktop_persists_git_result_identity_and_full_call_chain():
    command = "powershell -NoProfile -Command 'git show HEAD:assets/settings.json; exit $LASTEXITCODE'"
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        bridge = HookSessionBridge(db)
        try:
            records = [
                {"type": "session_meta", "payload": {"session_id": "git-chain-fixture"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call", "id": "git-chain-call", "call_id": "git-chain-call",
                        "name": "exec", "input": json.dumps({"cmd": command}),
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output", "id": "git-chain-output", "call_id": "git-chain-call",
                        "output": [{"type": "input_text", "text": '[{"exit_code":128,"output":"fatal: path does not exist in HEAD"}]'}],
                    },
                },
            ]
            for raw in records:
                bridge.ingest({
                    'protocol_version': HOOK_PROTOCOL_VERSION,
                    'source': 'codex-session', 'session_id': 'git-chain-fixture', 'event': raw,
                })
            trajectory_id = bridge.status('codex-session', 'git-chain-fixture')['session']['trajectory_id']
            call, result = [
                next(event for event in db.list_events(trajectory_id) if event['event_type'] == kind)
                for kind in ('tool_call', 'tool_result')
            ]
            assert call['payload']['tool_chain'] == ['exec', 'powershell', 'git']
            assert call['payload']['status'] == 'failed'
            assert call['payload']['failure_tool_name'] == 'git'
            assert result['payload']['tool_name'] == 'git'
            assert result['payload']['tool_chain'] == ['exec', 'powershell', 'git']
            assert db.failure_patterns(trajectory_id)[0]['failed_tool'] == 'git'
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
