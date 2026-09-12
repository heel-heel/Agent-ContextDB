from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .service import ContextDB
from .llm_profiles import public_profiles
from .hooks import HookSessionBridge


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def make_handler(db: ContextDB):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload, content_type: str = "application/json; charset=utf-8"):
            if isinstance(payload, (dict, list)):
                data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            elif isinstance(payload, str):
                data = payload.encode("utf-8")
            else:
                data = payload
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            q = parse_qs(parsed.query)
            try:
                if parsed.path == "/" or parsed.path == "/demo":
                    html = (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")
                    self._send(200, html, "text/html; charset=utf-8")
                elif parsed.path == "/health":
                    self._send(200, {"status": "ok", "service": "contextdb"})
                elif parsed.path == "/api/v1/trajectory":
                    self._send(200, db.get_trajectory(q["trajectory_id"][0]))
                elif parsed.path == "/api/v1/event":
                    self._send(200, db.event_detail(q["trajectory_id"][0], q["event_id"][0]))
                elif parsed.path == "/api/v1/log":
                    self._send(200, db.log(q["trajectory_id"][0]))
                elif parsed.path == "/api/v1/graph":
                    self._send(200, db.graph(q["trajectory_id"][0]))
                elif parsed.path == "/api/v1/global_overview":
                    self._send(200, db.global_overview())
                elif parsed.path == "/api/v1/version_status":
                    self._send(200, db.version_control_status(q["trajectory_id"][0]))
                elif parsed.path == "/api/v1/llm_profiles":
                    self._send(200, public_profiles())
                elif parsed.path == "/api/v1/hook_session":
                    self._send(200, HookSessionBridge(db).status(q["source"][0], q["session_id"][0]))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

        def do_POST(self):
            parsed = urlparse(self.path)
            body = self._body()
            try:
                if parsed.path == "/api/v1/trajectories":
                    self._send(200, db.create_trajectory(**body))
                elif parsed.path == "/api/v1/events":
                    self._send(200, db.append_event(**body))
                elif parsed.path == "/api/v1/query":
                    self._send(200, db.query(body))
                elif parsed.path == "/api/v1/query_view":
                    self._send(200, db.query_view(**body))
                elif parsed.path == "/api/v1/sql":
                    self._send(200, db.query_sql(**body))
                elif parsed.path == "/api/v1/nl_sql":
                    self._send(200, db.translate_natural_language_sql(**body))
                elif parsed.path == "/api/v1/branches":
                    self._send(200, db.create_branch(**body))
                elif parsed.path == "/api/v1/snapshots":
                    self._send(200, db.snapshot(**body))
                elif parsed.path == "/api/v1/rollback":
                    self._send(200, db.rollback(**body))
                elif parsed.path == "/api/v1/version/snapshot":
                    self._send(200, db.create_version_snapshot(**body))
                elif parsed.path == "/api/v1/version/branch":
                    self._send(200, db.create_version_branch(**body))
                elif parsed.path == "/api/v1/version/rollback":
                    self._send(200, db.create_version_rollback(**body))
                elif parsed.path == "/api/v1/diff":
                    self._send(200, db.diff(**body))
                elif parsed.path == "/api/v1/stream_context":
                    self._send(200, db.stream_context(**body))
                elif parsed.path == "/api/v1/export_rl_dataset":
                    self._send(200, db.export_rl_dataset(**body))
                elif parsed.path == "/api/v1/match_skill":
                    self._send(200, db.match_skill(**body))
                elif parsed.path == "/api/v1/apply_skill":
                    self._send(200, db.apply_skill(**body))
                elif parsed.path == "/api/v1/retrieve_for_failure":
                    self._send(200, db.retrieve_for_failure(**body))
                elif parsed.path == "/api/v1/hooks/events":
                    self._send(200, HookSessionBridge(db).ingest(body))
                elif parsed.path == "/api/v1/hooks/context":
                    self._send(200, HookSessionBridge(db).prepare_context(
                        body["source"], body["session_id"],
                        int(body.get("token_budget", 1200)),
                        str(body.get("delivery_channel", "http-hook")),
                    ))
                elif parsed.path == "/api/v1/hooks/skill_decision":
                    self._send(200, HookSessionBridge(db).record_skill_decision(
                        body["source"], body["session_id"], body["decision"], body.get("reason", ""),
                        body.get("skill_match_event_id"), body.get("skill_id"), body.get("action_id"),
                    ))
                elif parsed.path == "/api/v1/hooks/skill_application":
                    self._send(200, HookSessionBridge(db).record_skill_application(
                        body["source"], body["session_id"], body["tool_name"], body["command"], body["status"],
                        body.get("preview", ""), body.get("exit_code"), body.get("skill_match_event_id"),
                        body.get("skill_id"), body.get("action_id"), body.get("tool_call_id"), body.get("tool_call_event_id"),
                    ))
                elif parsed.path == "/api/v1/hooks/version_snapshot":
                    self._send(200, HookSessionBridge(db).create_snapshot(
                        body["source"], body["session_id"], body.get("message", ""), body.get("reason", ""),
                    ))
                elif parsed.path == "/api/v1/hooks/repair_branch":
                    self._send(200, HookSessionBridge(db).create_repair_branch(
                        body["source"], body["session_id"], body.get("branch_id"), body.get("snapshot_id"), body.get("reason", ""),
                    ))
                elif parsed.path == "/api/v1/hooks/rollback":
                    self._send(200, HookSessionBridge(db).rollback_context(
                        body["source"], body["session_id"], body["snapshot_id"], body.get("target_branch_id"), body.get("reason", ""),
                    ))
                elif parsed.path == "/api/v1/hooks/version_decision":
                    self._send(200, HookSessionBridge(db).record_version_decision(
                        body["source"], body["session_id"], body["action"], body["decision"],
                        body.get("reason", ""), body.get("suggestion_event_id"),
                    ))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def serve(root="data", host="127.0.0.1", port=8765):
    db = ContextDB(root)
    httpd = ThreadingHTTPServer((host, port), make_handler(db))
    print(f"ContextDB HTTP server running on http://{host}:{port}")
    print(f"Demo dashboard: http://{host}:{port}/demo")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.root, args.host, args.port)
