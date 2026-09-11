import os
from tempfile import TemporaryDirectory

from contextdb.service import ContextDB


def test_failure_patterns_cover_all_branches_and_contextql_relations():
    previous_provider = os.environ.get("CONTEXTDB_LLM_PROVIDER")
    os.environ["CONTEXTDB_LLM_PROVIDER"] = "mock"
    try:
        with TemporaryDirectory() as root:
            db = ContextDB(root)
            try:
                trajectory_id = db.create_trajectory("trajectory-wide failures")["trajectory_id"]
                main_call = db.append_event(trajectory_id, "tool_call", {"tool_name": "shell", "command": "build-main"})
                main_failure = db.append_event(
                    trajectory_id,
                    "tool_result",
                    {"status": "failed", "preview": "main build failed"},
                    actor="tool",
                )
                db.create_branch(trajectory_id, "repair", base_event_id=main_failure["event_id"])
                repair_call = db.append_event(trajectory_id, "tool_call", {"tool_name": "shell", "command": "build-repair"}, branch_id="repair")
                db.append_event(
                    trajectory_id,
                    "tool_result",
                    {"status": "failed", "preview": "repair build failed"},
                    branch_id="repair",
                    actor="tool",
                )

                view = db.query_view(trajectory_id, "failure_patterns", branch_id="main")
                assert {row["branch_id"] for row in view["content"]} == {"main", "repair"}
                assert {tuple(row["source_event_ids"]) for row in view["content"]} == {
                    (main_call["event_id"],),
                    (repair_call["event_id"],),
                }
                assert set(view["source_events"]) == {main_call["event_id"], repair_call["event_id"]}

                sql = db.query_sql(
                    trajectory_id,
                    "SELECT branch_id FROM failure_patterns ORDER BY branch_id",
                )
                assert [row["branch_id"] for row in sql["rows"]] == ["main", "repair"]
            finally:
                db.vector_index.conn.close()
                db.store.conn.close()
    finally:
        if previous_provider is None:
            os.environ.pop("CONTEXTDB_LLM_PROVIDER", None)
        else:
            os.environ["CONTEXTDB_LLM_PROVIDER"] = previous_provider
