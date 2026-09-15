from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def _watcher_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "windows_codex_session_watcher.py"
    spec = importlib.util.spec_from_file_location("codex_watcher_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watcher_retries_a_partial_jsonl_record_without_losing_it():
    watcher = _watcher_module()
    recorded = []
    original_post = watcher.post
    watcher.post = lambda _url, envelope: recorded.append(envelope) or {"skill_retrievals": []}
    try:
        with TemporaryDirectory() as root:
            log = Path(root) / "rollout-new-session.jsonl"
            pointer = Path(root) / "current-session.json"
            log.write_text('{"type":"response_item"', encoding="utf-8")

            offset, sent = watcher.forward_file(log, 0, "http://example.test", "test", pointer)
            assert offset == 0
            assert sent == 0

            with log.open("a", encoding="utf-8") as handle:
                handle.write(',"payload":{"type":"message","role":"user"}}\n')

            offset, sent = watcher.forward_file(log, offset, "http://example.test", "test", pointer)
            assert offset == log.stat().st_size
            assert sent == 1
            assert recorded[0]["session_id"] == "rollout-new-session"
            assert json.loads(pointer.read_text(encoding="utf-8"))["session_id"] == "rollout-new-session"
    finally:
        watcher.post = original_post


def test_watcher_does_not_rewrite_an_unchanged_session_pointer():
    watcher = _watcher_module()
    with TemporaryDirectory() as root:
        pointer = Path(root) / "current-session.json"
        rollout = Path(root) / "rollout-current.jsonl"
        watcher.save_session_pointer(pointer, "rollout-current", rollout)
        first = pointer.read_text(encoding="utf-8")

        watcher.save_session_pointer(pointer, "rollout-current", rollout)

        assert pointer.read_text(encoding="utf-8") == first


def test_watcher_skips_existing_logs_even_with_a_stale_state_offset():
    watcher = _watcher_module()
    original_argv = sys.argv
    try:
        with TemporaryDirectory() as root:
            sessions = Path(root) / "sessions"
            sessions.mkdir()
            log = sessions / "rollout-existing.jsonl"
            log.write_text('{"type":"response_item"}\n', encoding="utf-8")
            state_file = Path(root) / "watcher-state.json"
            pointer = Path(root) / "current-session.json"
            state_file.write_text(json.dumps({"files": {str(log.resolve()): {"offset": 0}}}), encoding="utf-8")

            sys.argv = [
                "watcher", "--sessions-dir", str(sessions), "--state-file", str(state_file),
                "--session-pointer", str(pointer), "--base-url", "http://example.test", "--once",
            ]
            assert watcher.main() == 0

            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert state["files"][str(log.resolve())]["offset"] == log.stat().st_size
    finally:
        sys.argv = original_argv
