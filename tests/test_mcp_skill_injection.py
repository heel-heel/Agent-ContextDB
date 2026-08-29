from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from contextdb.mcp_server import ContextDBMCPServer
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
        assert delivered["agent_context"]["matched"] is True
        assert delivered["delivery_event_id"]

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
