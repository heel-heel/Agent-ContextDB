import json
import sqlite3
import tempfile
from pathlib import Path

from contextdb.service import ContextDB


def test_entities_are_persisted_in_relational_tables_not_object_files():
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        db = ContextDB(root_path)
        trajectory = db.create_trajectory("Relational storage")
        event = db.append_event(trajectory["trajectory_id"], "user_message", {"text": "hello"})
        snapshot = db.snapshot(trajectory["trajectory_id"], message="checkpoint")

        connection = sqlite3.connect(root_path / "contextdb.sqlite3")
        assert connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM snapshot_records WHERE snapshot_id=?", (snapshot["snapshot_id"],)
        ).fetchone()[0] >= 3
        assert not list((root_path / "objects").rglob("*.json"))
        assert db.get_event(trajectory["trajectory_id"], event["event_id"])["payload"]["text"] == "hello"
        connection.close()
        db.close()


def test_contextql_reads_persistent_event_relation_and_view_exposes_plan():
    with tempfile.TemporaryDirectory() as root:
        db = ContextDB(root)
        trajectory = db.create_trajectory("SQL query")
        db.append_event(
            trajectory["trajectory_id"],
            "tool_result",
            {"status": "failed", "tool_name": "git", "command": "git show HEAD:missing"},
        )

        result = db.query_sql(
            trajectory["trajectory_id"],
            "SELECT tool_name, status FROM events WHERE trajectory_id=:trajectory_id",
        )
        assert result["rows"] == [{"tool_name": "git", "status": "failed"}]

        view = db.query_view(trajectory["trajectory_id"], "current_prompt")
        operators = view["execution_plan"]["operators"]
        assert operators[0]["kind"] == "SQL"
        assert any(operator["kind"] == "SEMANTIC" for operator in operators)
        assert {operator["operator"] for operator in ContextDB._view_execution_plan("learned_skills")["operators"] if operator["kind"] == "SEMANTIC"} == {"SemClusterBy"}
        assert next(operator for operator in ContextDB._view_execution_plan("semantic_repair_judgments")["operators"] if operator["kind"] == "SEMANTIC")["operator"] == "SemJoin"
        db.close()


def test_contextql_rejects_queries_without_trajectory_scope():
    with tempfile.TemporaryDirectory() as root:
        db = ContextDB(root)
        trajectory = db.create_trajectory("Scoped SQL")
        try:
            db.query_sql(trajectory["trajectory_id"], "SELECT event_id FROM events")
        except ValueError as exc:
            assert str(exc) == (
                "ContextQL is trajectory-scoped: filter the relevant relation with trajectory_id = :trajectory_id"
            )
        else:
            raise AssertionError("ContextQL accepted an unscoped query")
        db.close()


def test_contextql_reads_materialized_views_without_creating_derived_tables():
    with tempfile.TemporaryDirectory() as root:
        db = ContextDB(root)
        trajectory_id = db.create_trajectory("ContextQL materialized views")["trajectory_id"]
        db.query_view(trajectory_id, "failure_patterns")
        success = db.append_event(
            trajectory_id,
            "tool_result",
            {"status": "ok", "tool_name": "git", "command": "git status", "preview": "clean"},
        )
        db.query_view(trajectory_id, "success_patterns")

        result = db.query_sql(
            trajectory_id,
            "SELECT view_name, event_count FROM materialized_views "
            "WHERE trajectory_id=:trajectory_id AND branch_id=:branch_id AND view_name='failure_patterns'",
        )
        connection = sqlite3.connect(Path(root) / "contextdb.sqlite3")
        derived_tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('skills', 'skill_evidence', 'failure_patterns', 'repair_strategies', 'learned_skills', 'skill_application_trace')"
        ).fetchall()
        connection.close()

        assert result["rows"] == [{"view_name": "failure_patterns", "event_count": 0}]
        assert derived_tables == []
        success_result = db.query_sql(
            trajectory_id,
            "SELECT json_extract(pattern.value, '$.success_event_id') AS success_event_id "
            "FROM materialized_views AS view JOIN json_each(view.content_json) AS pattern "
            "WHERE view.trajectory_id=:trajectory_id AND view.branch_id=:branch_id "
            "AND view.view_name='success_patterns'",
        )
        assert success_result["rows"] == [{"success_event_id": success["event_id"]}]
        db.close()


def test_existing_object_json_is_imported_into_the_relational_store_once():
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        db = ContextDB(root_path)
        db.close()
        trajectory_id = "traj_legacy"
        key = f"trajectories/{trajectory_id}/meta"
        path = root_path / "objects" / "trajectories" / trajectory_id / "meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "trajectory_id": trajectory_id, "title": "Legacy", "agent_id": "agent",
            "source_id": "source", "default_branch": "main", "head_event_id": None,
            "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00", "metadata": {},
        }), encoding="utf-8")
        connection = sqlite3.connect(root_path / "contextdb.sqlite3")
        connection.execute(
            "INSERT INTO objects(key,type,trajectory_id,path) VALUES(?,?,?,?)",
            (key, "trajectories", trajectory_id, str(path)),
        )
        connection.commit()
        connection.close()

        migrated = ContextDB(root_path)
        assert migrated.get_trajectory(trajectory_id)["title"] == "Legacy"
        migrated.close()
