from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from contextdb.mcp_server import ContextDBMCPServer, _tool_result
from contextdb.server import make_handler


def _seed_skill(server: ContextDBMCPServer) -> None:
    skill = {
        "skill_id": "skill_compiler_repair",
        "name": "Repair compiler incompatibility",
        "trigger": {
            "failed_tool": "shell",
            "failed_command_pattern": "cargo build --release",
            "normalized_signature": "gcc rejected aws-lc-sys generated memcmp code",
        },
        "recommended_actions": [{
            "action_id": "act_use_clang",
            "name": "Retry with Clang",
            "canonical_tool": "shell",
            "canonical_command_template": "CC=clang cargo build --release",
            "confidence": {"max_llm_confidence": 0.91},
        }],
        "status": "validated",
        "highlight_event_ids": [],
    }
    server.db.vector_index.replace_owner("skills", "history", [{
        "entry_id": "history:skill_compiler_repair",
        "document": "compiler incompatibility cargo build release gcc rejected aws lc sys generated memcmp code",
        "metadata": {"skill": skill, "trajectory_id": "history"},
    }])


def test_mcp_skill_delivery_decision_and_application_trace():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)
        _seed_skill(server)
        source, session_id = "codex", "mcp-live-session"

        initial = server.dispatch("contextdb_prepare_context", {"source": source, "session_id": session_id})
        assert initial["agent_context"]["matched"] is False

        failed = server.dispatch("contextdb_record_tool_result", {
            "source": source, "session_id": session_id, "tool_name": "shell",
            "command": "cargo build --release", "status": "failed",
            "preview": "gcc rejected aws-lc-sys generated memcmp code", "exit_code": 1,
        })
        assert failed["skill_retrievals"]
        recommendation = failed["agent_context"][0]
        assert recommendation["matched"] is True
        assert recommendation["selected_action"]["action_id"] == "act_use_clang"

        delivered = server.dispatch("contextdb_get_recommendation", {"source": source, "session_id": session_id})
        assert delivered["matched"] is True
        assert delivered["skill_match_event_id"]
        assert delivered["skill_id"] == "skill_compiler_repair"
        assert delivered["action_id"] == "act_use_clang"
        assert delivered["proposed_action"] == "CC=clang cargo build --release"

        decision = server.dispatch("contextdb_record_skill_decision", {
            "source": source, "session_id": session_id, "decision": "accepted",
            "reason": "The compiler switch addresses the observed GCC error.",
        })
        assert decision["decision"] == "accepted"

        applied = server.dispatch("contextdb_record_skill_application", {
            "source": source, "session_id": session_id, "tool_name": "shell",
            "command": "CC=clang cargo build --release", "status": "ok",
            "preview": "build completed", "exit_code": 0,
        })
        assert applied["status"] == "ok"

        status = server.bridge.status(source, session_id)
        trace = status["skill_application_trace"]
        event_types = [event["event_type"] for event in trace]
        assert "skill_match" in event_types
        assert "skill_recommendation" in event_types
        assert "skill_decision" in event_types
        assert event_types.count("tool_call") >= 2
        assert event_types.count("tool_result") >= 2
        app_result = [event for event in trace if event["event_type"] == "tool_result"][-1]
        assert app_result["refs"]["skill_id"] == "skill_compiler_repair"
        server.db.vector_index.conn.close()
        server.db.store.conn.close()


def test_prepare_context_does_not_record_an_empty_recommendation():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)

        prepared = server.dispatch("contextdb_prepare_context", {
            "source": "codex", "session_id": "empty-context-session",
        })

        assert prepared["agent_context"]["matched"] is False
        assert prepared["delivery_event_id"] is None
        assert server.db.list_events(prepared["trajectory_id"]) == []
        server.db.vector_index.conn.close()
        server.db.store.conn.close()


def test_mcp_tool_result_compacts_diagnostic_context_for_stdio_clients():
    result = _tool_result({
        "trajectory_id": "traj_compact",
        "agent_context": {"matched": False},
        "recent_events": [{"event_id": "evt_recent", "payload": {"text": "raw"}}],
        "prompt_context": {
            "trajectory_id": "traj_compact",
            "rendered_agent_context": "Selected context only.",
            "summary": {
                "status": "ready",
                "summary": "A compact summary.",
                "source_event_timeline": [{"event_id": "evt_old", "text": "raw history"}],
                "scope": {"source_event_count": 42},
            },
        },
    })

    payload = result["structuredContent"]
    assert "recent_events" not in payload
    assert "source_event_timeline" not in payload["prompt_context"]["summary"]
    assert payload["prompt_context"]["summary"]["source_event_count"] == 42
    assert "raw history" not in result["content"][0]["text"]


def test_mcp_recommendation_response_excludes_rendered_turn_context():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)
        try:
            _seed_skill(server)
            source, session_id = "codex", "compact-recommendation-session"
            server.dispatch("contextdb_record_tool_result", {
                "source": source, "session_id": session_id, "tool_name": "shell",
                "command": "cargo build --release", "status": "failed",
                "preview": "gcc rejected aws-lc-sys generated memcmp code", "exit_code": 1,
            })
            recommendation = server.dispatch("contextdb_get_recommendation", {
                "source": source, "session_id": session_id,
            })
            assert recommendation["matched"] is True
            assert "prompt_context" not in recommendation
            assert "agent_context" not in recommendation
            assert recommendation["proposed_action"] == "CC=clang cargo build --release"
        finally:
            server.db.vector_index.conn.close()
            server.db.store.conn.close()


def test_mcp_recommendation_skips_semantic_context_materialization():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)
        try:
            _seed_skill(server)
            source, session_id = "codex", "recommendation-without-summary"
            server.dispatch("contextdb_record_tool_result", {
                "source": source, "session_id": session_id, "tool_name": "shell",
                "command": "cargo build --release", "status": "failed",
                "preview": "gcc rejected aws-lc-sys generated memcmp code", "exit_code": 1,
            })

            original = server.db.stream_context
            server.db.stream_context = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("summary should not run"))
            recommendation = server.dispatch("contextdb_get_recommendation", {
                "source": source, "session_id": session_id,
            })
            assert recommendation["matched"] is True
            server.db.stream_context = original
        finally:
            server.db.vector_index.conn.close()
            server.db.store.conn.close()


def test_mcp_recommendation_stdio_response_uses_compact_text_only_payload():
    result = _tool_result(
        {
            "matched": True,
            "skill_match_event_id": "evt_match",
            "skill_id": "skill_example",
            "action_id": "act_example",
            "proposed_action": "git show HEAD:examples/settings.json",
        },
        include_structured_content=False,
    )
    assert "structuredContent" not in result
    assert "git show HEAD:examples/settings.json" in result["content"][0]["text"]


def test_mcp_records_pre_action_snapshot_and_agent_controlled_branch():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)
        source, session_id = 'codex', 'mcp-version-session'
        recorded = server.dispatch('contextdb_record_tool_call', {
            'source': source, 'session_id': session_id, 'tool_name': 'powershell',
            'command': "Set-Content -LiteralPath '.\\scratch.txt' -Value 'x'", 'tool_call_id': 'write-1',
        })
        assert recorded['version_context']['latest_snapshot']['snapshot_id']
        failed = server.dispatch('contextdb_record_tool_result', {
            'source': source, 'session_id': session_id, 'tool_name': 'powershell',
            'command': "Set-Content -LiteralPath '.\\scratch.txt' -Value 'x'", 'tool_call_id': 'write-1',
            'status': 'failed', 'preview': 'Access is denied', 'exit_code': 1,
        })
        suggestion = failed['version_context']['repair_branch_suggestion']
        assert suggestion and suggestion['suggested_branch_id']
        repair = server.dispatch('contextdb_create_repair_branch', {
            'source': source, 'session_id': session_id, 'reason': 'Keep the original failure branch for analysis.',
        })
        assert repair['version_context']['active_branch_id'] == repair['branch_id']
        assert repair['decision_event_id']
        assert repair['branch_created_event_id']
        assert 'event' not in repair
        assert 'snapshot' not in repair
        status = server.dispatch('contextdb_get_version_status', {'source': source, 'session_id': session_id})
        assert status['workspace_restore']['supported'] is False
        assert status['active_branch_id'] == repair['branch_id']
        assert status['trajectory_id'] == repair['trajectory_id']
        assert 'session' not in status
        assert 'trajectory' not in status
        server.db.vector_index.conn.close()
        server.db.store.conn.close()


def test_mcp_stdio_lists_contextdb_tools():
    with TemporaryDirectory() as root:
        message = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
        completed = subprocess.run(
            [sys.executable, "-m", "contextdb.mcp_server", "--root", root],
            input=message, capture_output=True, text=True, check=True,
        )
        response = json.loads(completed.stdout.strip())
        names = {tool["name"] for tool in response["result"]["tools"]}
        assert "contextdb_prepare_context" in names
        assert "contextdb_record_skill_application" in names
        assert "contextdb_create_repair_branch" in names


def test_native_exec_hook_records_and_injects_recommendation():
    with TemporaryDirectory() as root:
        server = ContextDBMCPServer(root)
        _seed_skill(server)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server.db))
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        hook = Path(__file__).resolve().parents[1] / "tools" / "contextdb_native_exec_hook.py"
        try:
            completed = subprocess.run(
                [
                    sys.executable, str(hook), "--base-url", "http://127.0.0.1:%s" % httpd.server_port,
                    "--source", "codex", "--session-id", "native-hook-session", "--",
                    sys.executable, "-c", "import sys; print('gcc rejected aws-lc-sys generated memcmp code'); sys.exit(1)",
                ],
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 1
            assert "gcc rejected aws-lc-sys" in completed.stdout
            assert "CONTEXTDB_RECOMMENDATION" in completed.stdout
            status = server.bridge.status("codex", "native-hook-session")
            event_types = [event["event_type"] for event in status["skill_application_trace"]]
            assert "tool_call" in event_types
            assert "tool_result" in event_types
            assert "skill_match" in event_types
            assert "skill_recommendation" in event_types
            applied = subprocess.run(
                [
                    sys.executable, str(hook), "--base-url", "http://127.0.0.1:%s" % httpd.server_port,
                    "--source", "codex", "--session-id", "native-hook-session", "--apply-skill",
                    "--skill-match-event-id", "match-test", "--skill-id", "skill_compiler_repair", "--action-id", "act_use_clang", "--",
                    sys.executable, "-c", "print('real skill-guided retry ran')",
                ],
                capture_output=True, text=True,
            )
            assert applied.returncode == 0
            assert "real skill-guided retry ran" in applied.stdout
            events = server.db.list_events(status["session"]["trajectory_id"])
            assert [event["event_type"] for event in events].count("tool_call") == 2
            assert [event["event_type"] for event in events].count("tool_result") == 2
            assert events[-1]["refs"]["skill_id"] == "skill_compiler_repair"
        finally:
            httpd.shutdown()
            httpd.server_close()
            server.db.vector_index.conn.close()
            server.db.store.conn.close()


def test_native_exec_hook_fails_open_when_contextdb_is_unavailable():
    hook = Path(__file__).resolve().parents[1] / "tools" / "contextdb_native_exec_hook.py"
    completed = subprocess.run(
        [
            sys.executable, str(hook), "--base-url", "http://127.0.0.1:9",
            "--session-id", "unavailable-hook-session", "--",
            sys.executable, "-c", "print('real command still ran')",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "real command still ran" in completed.stdout
    assert "ContextDB hook warning" in completed.stderr
