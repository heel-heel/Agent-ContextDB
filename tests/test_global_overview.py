from tempfile import TemporaryDirectory

from contextdb.service import ContextDB


def test_global_overview_aggregates_trajectories_tools_and_shared_skills():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        try:
            source = db.create_trajectory("Source trajectory", agent_id="source-agent")
            consumer = db.create_trajectory("Consumer trajectory", agent_id="consumer-agent")
            skill = {
                "skill_id": "skill_fix_settings",
                "name": "Repair missing settings",
                "status": "candidate",
                "trigger": {"failed_tool": "shell"},
                "confidence": {"support_count": 2},
            }
            db.vector_index.replace_owner("skills", source["trajectory_id"], [{
                "entry_id": source["trajectory_id"] + ":skill_fix_settings",
                "document": "shell settings repair",
                "metadata": {"trajectory_id": source["trajectory_id"], "skill": skill},
            }])
            db.append_event(consumer["trajectory_id"], "tool_call", {
                "tool_name": "shell", "command": "read repaired settings",
            }, refs={
                "skill_id": "skill_fix_settings",
                "skill_source_trajectory_id": source["trajectory_id"],
            })

            overview = db.global_overview()

            assert overview["trajectory_count"] == 2
            assert overview["skill_count"] == 1
            consumer_row = next(row for row in overview["trajectories"] if row["trajectory_id"] == consumer["trajectory_id"])
            assert consumer_row["tools"] == [{"tool_name": "shell", "call_count": 1}]
            assert consumer_row["associated_skills"][0]["source_trajectory_id"] == source["trajectory_id"]
            assert any(edge["kind"] == "uses_skill" for edge in overview["graph"]["edges"])
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
