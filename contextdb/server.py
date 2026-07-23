from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .service import ContextDB


def make_handler(db: ContextDB):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload):
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
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
                if parsed.path == "/health":
                    self._send(200, {"status": "ok", "service": "contextdb"})
                elif parsed.path == "/api/v1/trajectory":
                    self._send(200, db.get_trajectory(q["trajectory_id"][0]))
                elif parsed.path == "/api/v1/event":
                    self._send(200, db.get_event(q["trajectory_id"][0], q["event_id"][0]))
                elif parsed.path == "/api/v1/log":
                    self._send(200, db.log(q["trajectory_id"][0]))
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
                elif parsed.path == "/api/v1/branches":
                    self._send(200, db.create_branch(**body))
                elif parsed.path == "/api/v1/snapshots":
                    self._send(200, db.snapshot(**body))
                elif parsed.path == "/api/v1/rollback":
                    self._send(200, db.rollback(**body))
                elif parsed.path == "/api/v1/stream_context":
                    self._send(200, db.stream_context(**body))
                elif parsed.path == "/api/v1/export_rl_dataset":
                    self._send(200, db.export_rl_dataset(**body))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def serve(root="data", host="127.0.0.1", port=8765):
    db = ContextDB(root)
    httpd = ThreadingHTTPServer((host, port), make_handler(db))
    print(f"ContextDB HTTP server running on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.root, args.host, args.port)
