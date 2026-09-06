from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from tempfile import TemporaryDirectory
from threading import Thread
from urllib import request

from contextdb.server import make_handler
from contextdb.service import ContextDB


def _post(base_url: str, path: str, payload: dict) -> dict:
    req = request.Request(
        base_url + path, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    with request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))


def _get(base_url: str, path: str) -> dict:
    with request.urlopen(base_url + path) as response:
        return json.loads(response.read().decode('utf-8'))


def test_version_control_http_endpoints_are_logical_only():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(db))
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = 'http://127.0.0.1:%s' % httpd.server_port
        try:
            trajectory = _post(base_url, '/api/v1/trajectories', {'title': 'HTTP version control', 'agent_id': 'test', 'source_id': 'test'})
            tid = trajectory['trajectory_id']
            snap = _post(base_url, '/api/v1/version/snapshot', {
                'trajectory_id': tid, 'message': 'Before an edit', 'origin': 'manual_ui', 'reason': 'test', 'actor': 'user',
            })
            branch = _post(base_url, '/api/v1/version/branch', {
                'trajectory_id': tid, 'branch_id': 'repair-http', 'snapshot_id': snap['snapshot']['snapshot_id'], 'reason': 'test', 'origin': 'manual_ui',
            })
            rollback = _post(base_url, '/api/v1/version/rollback', {
                'trajectory_id': tid, 'snapshot_id': snap['snapshot']['snapshot_id'], 'target_branch_id': 'rollback-http', 'reason': 'test', 'origin': 'manual_ui',
            })
            status = _get(base_url, '/api/v1/version_status?trajectory_id=' + tid)
            assert branch['branch']['branch_id'] == 'repair-http'
            assert rollback['branch']['branch_id'] == 'rollback-http'
            assert status['workspace_restore']['supported'] is False
            assert len(status['snapshots']) == 1
        finally:
            httpd.shutdown()
            httpd.server_close()
            db.vector_index.conn.close()
            db.store.conn.close()
