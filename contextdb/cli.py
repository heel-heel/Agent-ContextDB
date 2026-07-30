from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent_import import import_trace, replay_trace
from .client import ContextDBClient
from .service import ContextDB
from .server import serve


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def run_demo(root: str, trace_path: str = "examples/demo_trace.jsonl"):
    result = replay_trace(
        trace_path,
        root=root,
        source="generic-jsonl",
        title="Agent ContextDB trajectory demo",
        agent_id="codex",
        branch_id="main",
    )
    db = ContextDB(root)
    tid = result["trajectory_id"]
    summary = db.query_view(tid, "summary", "clang-attempt")
    failures = db.query_view(tid, "failures", "main")
    prompt = db.stream_context(tid, "clang-attempt", token_budget=2000)
    diff = db.diff(tid, "docker-attempt", "clang-attempt")
    graph = db.graph(tid)
    rl = db.export_rl_dataset(tid, "clang-attempt")
    rollback_branch = "rollback-clean" if "rollback-clean" in result["branches"] else None
    emit({
        "trajectory_id": tid,
        "trace_path": trace_path,
        "snapshot_id": result["snapshots"][0] if result["snapshots"] else None,
        "rollback_branch": rollback_branch,
        "branches": [b["branch_id"] for b in graph["branches"]],
        "event_nodes": len(graph["nodes"]),
        "summary_view": summary["content"],
        "main_failure_count": len(failures["content"]),
        "estimated_saved_tokens": prompt["content"]["estimated_saved_tokens"],
        "diff_only_right": len(diff["only_right"]),
        "rl_rows": len(rl),
        "demo_flow": "demo is replayed from examples/demo_trace.jsonl: main fails -> rollback-clean restores snapshot context -> docker-attempt/clang-attempt branch from rollback-clean",
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


def run_client_demo(base_url: str):
    client = ContextDBClient(base_url)
    traj = client.create_trajectory("Live Agent client demo", agent_id="sample-live-agent", source_id="contextdb-client")
    tid = traj["trajectory_id"]
    client.log_user(tid, "Run a quick health check and report the result.")
    client.log_assistant(tid, "I will call the shell health check tool.")
    client.log_tool_call(tid, "shell", "curl -s http://127.0.0.1:8765/health")
    client.log_tool_result(tid, "ok", "{\"status\":\"ok\",\"service\":\"contextdb\"}")
    view = client.query_view(tid, "current_prompt")
    emit({"trajectory_id": tid, "events_written": 4, "current_prompt_events": len(view["content"]["recent_events"])})


def main():
    parser = argparse.ArgumentParser(prog="contextdb")
    parser.add_argument("--root", default="data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--trace", default="examples/demo_trace.jsonl")
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
    p_trace = sub.add_parser("import-trace")
    p_trace.add_argument("path")
    p_trace.add_argument("--source", default="generic-jsonl", choices=["generic-jsonl", "jsonl", "generic"])
    p_trace.add_argument("--title", default=None)
    p_trace.add_argument("--agent-id", default="external-agent")
    p_trace.add_argument("--branch", default="main")
    p_client = sub.add_parser("client-demo")
    p_client.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    if args.cmd == "serve":
        serve(args.root, args.host, args.port)
        return
    if args.cmd == "demo":
        run_demo(args.root, args.trace)
        return
    if args.cmd == "import-transcript":
        import_transcript(args.path, args.root)
        return
    if args.cmd == "import-trace":
        emit(import_trace(args.path, root=args.root, source=args.source, title=args.title, agent_id=args.agent_id, branch_id=args.branch))
        return
    if args.cmd == "client-demo":
        run_client_demo(args.base_url)
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
