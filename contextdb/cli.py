from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from .agent_import import import_trace, normalize_trace_to_jsonl, replay_trace
from .client import ContextDBClient
from .benchmark import BenchmarkRunner
from .agent_runtime import CodexContextBridge
from .hooks import HookSessionBridge, stream_hook_events
from .mcp_server import serve_stdio as serve_mcp_stdio
from .service import ContextDB
from .server import serve


def _load_dashboard_environment() -> None:
    """Load dashboard and private background-LLM credentials for serving."""
    api_dir = Path(__file__).resolve().parent.parent / "_API"
    for env_path in (api_dir / "dashboard.env", api_dir / "background_llm.env"):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                os.environ[key] = value.strip()


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
    graph = db.graph(tid)
    branch_ids = [b["branch_id"] for b in graph["branches"]]
    view_branch = "clang-attempt" if "clang-attempt" in branch_ids else (branch_ids[0] if branch_ids else "main")
    diff_left = "docker-attempt" if "docker-attempt" in branch_ids else view_branch
    summary = db.query_view(tid, "summary", view_branch)
    failures = db.query_view(tid, "failures", "main" if "main" in branch_ids else view_branch)
    prompt = db.stream_context(tid, view_branch, token_budget=2000)
    diff = db.diff(tid, diff_left, view_branch) if diff_left != view_branch else {"only_right": []}
    rl = db.export_rl_dataset(tid, view_branch)
    rollback_branch = "rollback-clean" if "rollback-clean" in result["branches"] else None
    emit({
        "trajectory_id": tid,
        "trace_path": trace_path,
        "snapshot_id": result["snapshots"][0] if result["snapshots"] else None,
        "rollback_branch": rollback_branch,
        "branches": branch_ids,
        "event_nodes": len(graph["nodes"]),
        "summary_view": summary["content"],
        "main_failure_count": len(failures["content"]),
        "estimated_saved_tokens": prompt["content"]["estimated_saved_tokens"],
        "diff_only_right": len(diff["only_right"]),
        "rl_rows": len(rl),
        "demo_flow": "generic JSONL replay; branch-aware demo trace uses rollback/docker/clang branches when present, linear traces use main",
        "dashboard": "run: contextdb serve --host 0.0.0.0 --port 8765, then open /demo?trajectory_id=" + tid,
    })


def run_swe_demo(root: str, trace_path: str = "examples/swe_agent/marshmallow-code__marshmallow-1867.traj", llm_annotate: bool = True, out_path: str | None = None):
    source = "swe-agent-traj-llm" if llm_annotate else "swe-agent-traj"
    processed_path = out_path or str(Path(trace_path).with_suffix(".contextdb.jsonl"))
    normalized = normalize_trace_to_jsonl(
        trace_path,
        processed_path,
        source=source,
        agent_id="swe-agent",
        branch_id="main",
    )
    result = replay_trace(
        processed_path,
        root=root,
        source="generic-jsonl",
        title="SWE-agent trajectory demo on SWE-bench task marshmallow-code__marshmallow-1867",
        agent_id="swe-agent",
        branch_id="main",
    )
    db = ContextDB(root)
    tid = result["trajectory_id"]
    graph = db.graph(tid)
    summary = db.query_view(tid, "summary", "main")
    failures = db.query_view(tid, "failures", "main")
    prompt = db.stream_context(tid, "main", token_budget=2000)
    emit({
        "trajectory_id": tid,
        "trace_path": trace_path,
        "processed_trace_path": processed_path,
        "normalization_source": normalized["source"],
        "source": result["source"],
        "agent_id": "swe-agent",
        "event_nodes": len(graph["nodes"]),
        "summary_view": summary["content"],
        "failure_count": len(failures["content"]),
        "estimated_saved_tokens": prompt["content"]["estimated_saved_tokens"],
        "benchmark": "SWE-bench",
        "trajectory_source": "SWE-agent official GitHub demonstration .traj",
        "llm_trace_annotation": llm_annotate,
        "paper_context": "SWE-bench provides real GitHub issue repair tasks; SWE-agent records step-level agent trajectories on those tasks.",
        "demo_flow": "raw SWE-agent .traj -> processed ContextDB JSONL -> generic JSONL replay/import path",
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



def run_codex_online_demo(root: str, failure: dict):
    db = ContextDB(root)
    bridge = CodexContextBridge(db)
    trajectory = bridge.start("Codex online ContextDB integration", {"mode": "online", "agent_family": "codex"})
    tid = trajectory["trajectory_id"]
    bridge.record(tid, "user_message", {"text": "Continue the coding task and use prior ContextDB skills when a tool fails."}, actor="user")
    bridge.record(tid, "tool_call", {"tool_name": failure.get("tool", "shell"), "command": failure.get("command", "")})
    result = bridge.record_tool_result(tid, failure.get("tool", "shell"), failure.get("command", ""), "failed", failure.get("error_signature", "unknown failure"))
    emit({"trajectory_id": tid, "agent_id": "codex", "mode": "online", "tool_result_event_id": result["tool_result"]["event_id"], "skill_retrieval": result["skill_retrieval"], "dashboard": "run: contextdb serve --host 0.0.0.0 --port 8765, then open /demo?trajectory_id=" + tid})

def main():
    parser = argparse.ArgumentParser(prog="contextdb")
    parser.add_argument("--root", default="data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--trace", default="examples/demo_trace.jsonl")
    p_swe_demo = sub.add_parser("swe-demo")
    p_swe_demo.add_argument("--trace", default="examples/swe_agent/marshmallow-code__marshmallow-1867.traj")
    p_swe_demo.add_argument("--llm-annotate", dest="llm_annotate", action="store_true", default=True)
    p_swe_demo.add_argument("--no-llm-annotate", dest="llm_annotate", action="store_false")
    p_swe_demo.add_argument("--out", default=None)
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
    p_trace.add_argument("--source", default="generic-jsonl", choices=["generic-jsonl", "jsonl", "generic", "swe-agent-traj", "swe-agent", "swe", "swe-agent-traj-llm", "swe-agent-llm", "swe-llm", "codex-jsonl", "codex"])
    p_trace.add_argument("--title", default=None)
    p_trace.add_argument("--agent-id", default="external-agent")
    p_trace.add_argument("--branch", default="main")
    p_norm = sub.add_parser("normalize-trace")
    p_norm.add_argument("path")
    p_norm.add_argument("--source", default="swe-agent-traj-llm", choices=["generic-jsonl", "jsonl", "generic", "swe-agent-traj", "swe-agent", "swe", "swe-agent-traj-llm", "swe-agent-llm", "swe-llm", "codex-jsonl", "codex"])
    p_norm.add_argument("--out", default=None)
    p_norm.add_argument("--agent-id", default="external-agent")
    p_norm.add_argument("--branch", default="main")
    p_norm.add_argument("--no-llm-annotate", dest="no_llm_annotate", action="store_true")
    p_match = sub.add_parser("match-skill")
    p_match.add_argument("trajectory_id")
    p_match.add_argument("failure_json")
    p_match.add_argument("--top-k", type=int, default=3)
    p_apply = sub.add_parser("apply-skill")
    p_apply.add_argument("trajectory_id")
    p_apply.add_argument("failure_json")
    p_apply.add_argument("--branch", default="skill-application")
    p_client = sub.add_parser("client-demo")
    p_client.add_argument("--base-url", default="http://127.0.0.1:8765")
    p_benchmark = sub.add_parser("benchmark")
    p_benchmark.add_argument("manifest")
    p_codex_online = sub.add_parser("codex-online-demo")
    p_codex_online.add_argument("failure_json")
    p_hook_stream = sub.add_parser("hook-stream", help="Forward live JSONL Agent events to the ContextDB Hook API.")
    p_hook_stream.add_argument("--base-url", default="http://127.0.0.1:8765")
    p_hook_stream.add_argument("--source", default="codex", choices=["codex", "codex-session", "generic"])
    p_hook_stream.add_argument("--session-id", default=None)
    p_hook_stream.add_argument("--title", default=None)
    p_hook_stream.add_argument("--agent-id", default=None)
    p_hook_stream.add_argument("--no-echo", action="store_true", help="Do not re-emit source JSONL on stdout.")
    p_hook_status = sub.add_parser("hook-status", help="Show a persisted Hook session and its skill trace.")
    p_hook_status.add_argument("source")
    p_hook_status.add_argument("session_id")
    p_hook_context = sub.add_parser("hook-context", help="Return the latest live skill recommendation for an Agent resume turn.")
    p_hook_context.add_argument("source")
    p_hook_context.add_argument("session_id")
    p_hook_context.add_argument("--format", choices=["json", "prompt"], default="json")
    p_mcp = sub.add_parser("mcp-serve", help="Run the ContextDB stdio MCP server for live skill injection.")
    args = parser.parse_args()

    if args.cmd == "serve":
        _load_dashboard_environment()
        serve(args.root, args.host, args.port)
        return
    if args.cmd == "mcp-serve":
        raise SystemExit(serve_mcp_stdio(args.root))
    if args.cmd == "demo":
        run_demo(args.root, args.trace)
        return
    if args.cmd == "swe-demo":
        run_swe_demo(args.root, args.trace, args.llm_annotate, args.out)
        return
    if args.cmd == "import-transcript":
        import_transcript(args.path, args.root)
        return
    if args.cmd == "import-trace":
        emit(import_trace(args.path, root=args.root, source=args.source, title=args.title, agent_id=args.agent_id, branch_id=args.branch))
        return
    if args.cmd == "normalize-trace":
        source = args.source
        if args.no_llm_annotate and source in {"swe-agent-traj-llm", "swe-agent-llm", "swe-llm"}:
            source = "swe-agent-traj"
        out = args.out or str(Path(args.path).with_suffix(".contextdb.jsonl"))
        emit(normalize_trace_to_jsonl(args.path, out, source=source, agent_id=args.agent_id, branch_id=args.branch))
        return
    if args.cmd == "benchmark":
        emit(BenchmarkRunner(args.root).run(args.manifest))
        return
    if args.cmd == "codex-online-demo":
        run_codex_online_demo(args.root, json.loads(args.failure_json))
        return
    if args.cmd == "client-demo":
        run_client_demo(args.base_url)
        return
    if args.cmd == "hook-stream":
        raise SystemExit(stream_hook_events(args.base_url, source=args.source, session_id=args.session_id, title=args.title, agent_id=args.agent_id, echo=not args.no_echo))

    db = ContextDB(args.root)
    if args.cmd == "hook-status":
        emit(HookSessionBridge(db).status(args.source, args.session_id))
        return
    if args.cmd == "hook-context":
        context = HookSessionBridge(db).agent_context(args.source, args.session_id)
        if args.format == "prompt":
            recommendation = context["agent_context"]
            print("ContextDB live skill recommendation: " + json.dumps(recommendation, ensure_ascii=False))
        else:
            emit(context)
        return
    if args.cmd == "match-skill":
        emit(db.match_skill(args.trajectory_id, json.loads(args.failure_json), top_k=args.top_k))
        return
    if args.cmd == "apply-skill":
        emit(db.apply_skill(args.trajectory_id, json.loads(args.failure_json), branch_id=args.branch))
        return
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
