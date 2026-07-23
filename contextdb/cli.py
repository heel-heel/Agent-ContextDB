from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import ContextDB
from .server import serve


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def run_demo(root: str):
    db = ContextDB(root)
    traj = db.create_trajectory("Configure OpenViking", agent_id="codex", source_id="vmware-ubuntu")
    tid = traj["trajectory_id"]
    db.append_event(tid, "user_message", {"text": "Configure OpenViking in an isolated environment."}, actor="user")
    db.append_event(tid, "tool_call", {"tool_name": "ssh", "command": "check python and conda"})
    db.append_event(tid, "tool_result", {"status": "ok", "preview": "openviking_env exists"})
    db.append_event(tid, "memory_update", {"fact": "Prefer existing conda envs before creating new environments."})
    before = db.snapshot(tid, message="before native binding strategy")
    db.append_event(tid, "tool_result", {"status": "failed", "preview": "gcc 9.4 rejected by aws-lc-sys memcmp bug"})
    db.create_branch(tid, "docker-attempt")
    db.append_event(tid, "assistant_message", {"text": "Use clang in native conda branch."})
    db.append_event(tid, "tool_result", {"status": "ok", "preview": "server healthy on port 1933"})
    summary = db.query_view(tid, "summary")
    failures = db.query_view(tid, "failures")
    prompt = db.stream_context(tid, token_budget=2000)
    rl = db.export_rl_dataset(tid)
    emit({"trajectory_id": tid, "snapshot": before, "summary_view": summary, "failure_count": len(failures["content"]), "current_prompt_events": len(prompt["content"]["recent_events"]), "rl_rows": len(rl)})


def import_transcript(path: str, root: str):
    db = ContextDB(root)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    traj = db.create_trajectory(data.get("title", "Imported trajectory"), agent_id=data.get("agent_id", "imported-agent"), source_id=data.get("source_id", "import"))
    tid = traj["trajectory_id"]
    for item in data.get("events", []):
        db.append_event(tid, item.get("event_type", "message"), item.get("payload", {}), branch_id=item.get("branch_id", "main"), actor=item.get("actor", "agent"), refs=item.get("refs"), metadata=item.get("metadata"))
    emit(db.log(tid))


def main():
    parser = argparse.ArgumentParser(prog="contextdb")
    parser.add_argument("--root", default="data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_create = sub.add_parser("create-trajectory")
    p_create.add_argument("title")
    p_event = sub.add_parser("append-event")
    p_event.add_argument("trajectory_id")
    p_event.add_argument("event_type")
    p_event.add_argument("payload_json")
    p_query = sub.add_parser("query")
    p_query.add_argument("filters_json")
    p_view = sub.add_parser("query-view")
    p_view.add_argument("trajectory_id")
    p_view.add_argument("view_name")
    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("trajectory_id")
    p_snapshot.add_argument("--message", default="")
    p_import = sub.add_parser("import-transcript")
    p_import.add_argument("path")
    args = parser.parse_args()

    if args.cmd == "serve":
        serve(args.root, args.host, args.port)
        return
    if args.cmd == "demo":
        run_demo(args.root)
        return
    if args.cmd == "import-transcript":
        import_transcript(args.path, args.root)
        return

    db = ContextDB(args.root)
    if args.cmd == "create-trajectory":
        emit(db.create_trajectory(args.title))
    elif args.cmd == "append-event":
        emit(db.append_event(args.trajectory_id, args.event_type, json.loads(args.payload_json)))
    elif args.cmd == "query":
        emit(db.query(json.loads(args.filters_json)))
    elif args.cmd == "query-view":
        emit(db.query_view(args.trajectory_id, args.view_name))
    elif args.cmd == "snapshot":
        emit(db.snapshot(args.trajectory_id, message=args.message))


if __name__ == "__main__":
    main()
