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
    traj = db.create_trajectory(
        "Agent ContextDB trajectory demo",
        agent_id="codex",
        source_id="user-demo",
        metadata={"demo": "agent trajectory as versioned database state"},
    )
    tid = traj["trajectory_id"]
    db.append_event(tid, "user_message", {"text": "Set up an agent context service and expose a stable ContextDB API."}, actor="user")
    db.append_event(tid, "tool_call", {"tool_name": "shell", "command": "python --version && cargo --version"})
    db.append_event(tid, "tool_result", {"status": "ok", "preview": "Python and Rust toolchains found"}, actor="tool")
    db.append_event(tid, "memory_update", {"fact": "Prefer existing conda environments before creating new ones."})
    snap = db.snapshot(tid, message="clean environment inspected")

    db.append_event(tid, "assistant_message", {"text": "Try the native build path first."})
    db.append_event(tid, "tool_call", {"tool_name": "shell", "command": "cargo build --release"})
    db.append_event(tid, "tool_result", {"status": "failed", "preview": "gcc 9.4 rejected aws-lc-sys generated memcmp code"}, actor="tool")
    native_head = db.get_branch(tid, "main")["head_event_id"]

    db.create_branch(tid, "docker-attempt", base_event_id=snap["event_id"])
    db.append_event(tid, "assistant_message", {"text": "Use Docker to isolate compiler and dependency versions."}, branch_id="docker-attempt")
    db.append_event(tid, "tool_call", {"tool_name": "docker", "command": "docker compose up context-service"}, branch_id="docker-attempt")
    db.append_event(tid, "tool_result", {"status": "ok", "preview": "Context service healthy on port 1933"}, branch_id="docker-attempt", actor="tool")
    db.append_event(tid, "memory_update", {"fact": "Docker branch avoids host compiler drift for native dependency builds."}, branch_id="docker-attempt")

    db.create_branch(tid, "clang-attempt", base_event_id=native_head)
    db.append_event(tid, "assistant_message", {"text": "Retry native build with CC=clang to bypass gcc 9.4."}, branch_id="clang-attempt")
    db.append_event(tid, "tool_call", {"tool_name": "shell", "command": "CC=clang cargo build --release"}, branch_id="clang-attempt")
    db.append_event(tid, "tool_result", {"status": "ok", "preview": "Native server healthy on port 1933"}, branch_id="clang-attempt", actor="tool")

    summary = db.query_view(tid, "summary", "clang-attempt")
    failures = db.query_view(tid, "failures", "clang-attempt")
    prompt = db.stream_context(tid, "clang-attempt", token_budget=2000)
    diff = db.diff(tid, "docker-attempt", "clang-attempt")
    graph = db.graph(tid)
    rl = db.export_rl_dataset(tid, "clang-attempt")
    emit({
        "trajectory_id": tid,
        "snapshot_id": snap["snapshot_id"],
        "branches": [b["branch_id"] for b in graph["branches"]],
        "event_nodes": len(graph["nodes"]),
        "summary_view": summary["content"],
        "failure_count": len(failures["content"]),
        "estimated_saved_tokens": prompt["content"]["estimated_saved_tokens"],
        "diff_only_right": len(diff["only_right"]),
        "rl_rows": len(rl),
        "dashboard": "run: contextdb serve --host 0.0.0.0 --port 8765, then open /demo?trajectory_id=" + tid,
    })


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
    p_view.add_argument("--branch", default="main")
    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("trajectory_id")
    p_snapshot.add_argument("--branch", default="main")
    p_snapshot.add_argument("--message", default="")
    p_graph = sub.add_parser("graph")
    p_graph.add_argument("trajectory_id")
    p_diff = sub.add_parser("diff")
    p_diff.add_argument("trajectory_id")
    p_diff.add_argument("left_branch")
    p_diff.add_argument("right_branch")
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
        emit(db.query_view(args.trajectory_id, args.view_name, args.branch))
    elif args.cmd == "snapshot":
        emit(db.snapshot(args.trajectory_id, branch_id=args.branch, message=args.message))
    elif args.cmd == "graph":
        emit(db.graph(args.trajectory_id))
    elif args.cmd == "diff":
        emit(db.diff(args.trajectory_id, args.left_branch, args.right_branch))


if __name__ == "__main__":
    main()
