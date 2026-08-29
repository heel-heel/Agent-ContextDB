from __future__ import annotations

"""Tail Windows Codex App session logs and forward appended records to ContextDB."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict
from urllib import request


PROTOCOL_VERSION = "contextdb.agent_hook.v1"


def post(base_url: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    req = request.Request(base_url.rstrip("/") + "/api/v1/hooks/events", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {"files": {}}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def forward_file(path: Path, offset: int, base_url: str, title_prefix: str) -> tuple[int, int]:
    # A rollout file is exactly one Codex App conversation. Its filename remains
    # available on every record, unlike session metadata which appears only once.
    session_id = path.stem
    sent = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        while True:
            line = handle.readline()
            if not line:
                break
            offset = handle.tell()
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                response = post(base_url, {
                    "protocol_version": PROTOCOL_VERSION,
                    "source": "codex-session",
                    "session_id": session_id,
                    "title": f"{title_prefix}: {session_id}",
                    "agent_id": "codex",
                    "event": record,
                })
                sent += 1
                for retrieval in response.get("skill_retrievals", []):
                    print("CONTEXTDB_SKILL_HOOK " + json.dumps(retrieval, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(f"ContextDB watcher error for {path.name}: {exc}", file=sys.stderr, flush=True)
    return offset, sent


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward Codex App session JSONL to ContextDB.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--sessions-dir", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--state-file", default=str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ContextDB" / "codex-watcher-state.json"))
    parser.add_argument("--title-prefix", default="Windows Codex App live session")
    parser.add_argument("--interval", type=float, default=0.7)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--replay-existing", action="store_true", help="Import existing records on the first scan instead of starting at each file's end.")
    args = parser.parse_args()

    sessions = Path(args.sessions_dir).expanduser()
    state_path = Path(args.state_file).expanduser()
    state = load_state(state_path)
    state.setdefault("files", {})
    print(f"Watching Codex App sessions under {sessions}", flush=True)
    while True:
        for log_path in sorted(sessions.rglob("rollout-*.jsonl")):
            key = str(log_path.resolve())
            prior = state["files"].get(key, {})
            size = log_path.stat().st_size
            if not prior and not args.replay_existing:
                state["files"][key] = {"offset": size, "updated_at": time.time()}
                continue
            offset = int(prior.get("offset", 0))
            if offset > size:
                offset = 0
            offset, sent = forward_file(log_path, offset, args.base_url, args.title_prefix)
            state["files"][key] = {"offset": offset, "updated_at": time.time()}
            if sent:
                print(f"Forwarded {sent} event(s) from {log_path.name}", flush=True)
        save_state(state_path, state)
        if args.once:
            return 0
        time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    raise SystemExit(main())
