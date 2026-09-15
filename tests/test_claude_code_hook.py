from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from contextdb.server import make_handler
from contextdb.service import ContextDB
from contextdb.hooks import HookSessionBridge


def _seed_skill(db: ContextDB) -> None:
    skill = {
        "skill_id": "skill_claude_compiler_repair",
        "name": "Repair compiler incompatibility",
        "trigger": {"failed_tool": "shell", "failed_command_pattern": "cargo build --release"},
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
    db.vector_index.replace_owner("skills", "history", [{
        "entry_id": "history:skill_claude_compiler_repair",
        "document": "compiler incompatibility cargo build release gcc rejected aws lc sys generated memcmp code",
        "metadata": {"skill": skill, "trajectory_id": "history"},
    }])


def _run_hook(hook: Path, base_url: str, event: str, payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(hook), "--base-url", base_url, "--event", event],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def test_claude_code_hook_records_failure_and_injects_recommendation():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        _seed_skill(db)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        hook = Path(__file__).resolve().parents[1] / "tools" / "claude_code_hook.py"
        base_url = "http://127.0.0.1:%s" % httpd.server_port
        raw = {
            "session_id": "claude-live-session",
            "tool_name": "Bash",
            "tool_use_id": "toolu_compiler_failure",
            "tool_input": {"command": "cargo build --release"},
            "tool_response": "gcc rejected aws-lc-sys generated memcmp code",
        }
        try:
            pre = _run_hook(hook, base_url, "PreToolUse", raw)
            failure = _run_hook(hook, base_url, "PostToolUseFailure", raw)
            assert pre.returncode == 0
            assert failure.returncode == 0
            injected = json.loads(failure.stdout)
            context = injected["hookSpecificOutput"]["additionalContext"]
            assert "skill_claude_compiler_repair" in context
            status = HookSessionBridge(db).status("claude-code", "claude-live-session")
            event_types = [event["event_type"] for event in status["skill_application_trace"]]
            assert "tool_call" in event_types
            assert "tool_result" in event_types
            assert "skill_match" in event_types
            assert "skill_recommendation" in event_types
        finally:
            httpd.shutdown()
            httpd.server_close()
            db.vector_index.conn.close()
            db.store.conn.close()


def test_claude_code_hook_records_read_tool_events():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        hook = Path(__file__).resolve().parents[1] / "tools" / "claude_code_hook.py"
        base_url = "http://127.0.0.1:%s" % httpd.server_port
        raw = {
            "session_id": "claude-read-session",
            "tool_name": "Read",
            "tool_use_id": "toolu_missing_settings",
            "tool_input": {"file_path": r".\assets\settings.json"},
            "tool_response": "File does not exist",
        }
        try:
            pre = _run_hook(hook, base_url, "PreToolUse", raw)
            failure = _run_hook(hook, base_url, "PostToolUseFailure", raw)
            assert pre.returncode == 0
            assert failure.returncode == 0
            status = HookSessionBridge(db).status("claude-code", "claude-read-session")
            tool_call = next(event for event in status["skill_application_trace"] if event["event_type"] == "tool_call")
            tool_result = next(event for event in status["skill_application_trace"] if event["event_type"] == "tool_result")
            assert "tool_name" not in tool_call["payload"]
            assert tool_call["payload"]["tool_chain"] == ["read"]
            assert tool_call["payload"]["agent_tool_name"] == "Read"
            assert tool_call["payload"]["command"] == r".\assets\settings.json"
            assert tool_result["payload"]["tool_name"] == "read"
            assert tool_result["payload"]["tool_chain"] == ["read"]
            assert tool_result["payload"]["status"] == "failed"
        finally:
            httpd.shutdown()
            httpd.server_close()
            db.vector_index.conn.close()
            db.store.conn.close()


def test_claude_code_hook_uses_codex_style_chains_and_canonical_contextdb_names():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        hook = Path(__file__).resolve().parents[1] / "tools" / "claude_code_hook.py"
        base_url = "http://127.0.0.1:%s" % httpd.server_port
        shell = {
            "session_id": "claude-chain-session",
            "tool_name": "Bash",
            "tool_use_id": "toolu_git_show",
            "tool_input": {"command": "powershell -NoProfile -Command 'git show HEAD:README.md'"},
            "tool_response": "# Agent ContextDB",
            "exit_code": 0,
        }
        contextdb = {
            "session_id": "claude-chain-session",
            "tool_name": "mcp__contextdb__contextdb_get_version_status",
            "tool_use_id": "toolu_contextdb_status",
            "tool_input": {"source": "claude-code", "session_id": "claude-chain-session"},
            "tool_response": '{"active_branch_id": "main"}',
        }
        try:
            assert _run_hook(hook, base_url, "PreToolUse", shell).returncode == 0
            assert _run_hook(hook, base_url, "PostToolUse", shell).returncode == 0
            assert _run_hook(hook, base_url, "PreToolUse", contextdb).returncode == 0
            assert _run_hook(hook, base_url, "PostToolUse", contextdb).returncode == 0

            status = HookSessionBridge(db).status("claude-code", "claude-chain-session")
            events = db.list_events(status["trajectory"]["trajectory_id"])
            shell_call = next(event for event in events if event["event_type"] == "tool_call" and event["payload"].get("command", "").startswith("powershell"))
            shell_result = next(event for event in events if event["event_type"] == "tool_result" and event["payload"].get("command", "").startswith("powershell"))
            contextdb_call = next(event for event in events if event["event_type"] == "tool_call" and event["payload"].get("agent_tool_name") == "mcp__contextdb__contextdb_get_version_status")
            contextdb_result = next(event for event in events if event["event_type"] == "tool_result" and event["payload"].get("tool_name") == "contextdb_get_version_status")

            assert shell_call["payload"]["tool_chain"] == ["bash", "powershell", "git"]
            assert shell_result["payload"]["tool_name"] == "git"
            assert shell_result["payload"]["tool_chain"] == ["bash", "powershell", "git"]
            assert contextdb_call["payload"]["tool_chain"] == ["contextdb_get_version_status"]
            assert contextdb_result["payload"]["tool_chain"] == ["contextdb_get_version_status"]
            assert not any(event["event_type"] == "snapshot" for event in events)
        finally:
            httpd.shutdown()
            httpd.server_close()
            db.vector_index.conn.close()
            db.store.conn.close()
