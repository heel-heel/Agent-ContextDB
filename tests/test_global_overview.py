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
                "tool_name": "exec", "tool_chain": ["exec", "powershell", "get-content"],
                "command": "powershell -Command Get-Content repaired-settings.json",
            }, refs={
                "skill_id": "skill_fix_settings",
                "skill_source_trajectory_id": source["trajectory_id"],
            })
            db.append_event(consumer["trajectory_id"], "tool_result", {
                "tool_name": "get-content", "tool_chain": ["exec", "powershell", "get-content"],
                "status": "ok", "command": "powershell -Command Get-Content repaired-settings.json",
            }, refs={
                "skill_id": "skill_fix_settings",
                "skill_source_trajectory_id": source["trajectory_id"],
            })

            overview = db.global_overview()

            assert overview["trajectory_count"] == 2
            assert overview["skill_count"] == 1
            consumer_row = next(row for row in overview["trajectories"] if row["trajectory_id"] == consumer["trajectory_id"])
            assert consumer_row["tools"] == [{"tool_name": "get-content", "call_count": 1}]
            assert consumer_row["associated_skills"][0]["source_trajectory_id"] == source["trajectory_id"]
            assert any(edge["kind"] == "uses_skill" for edge in overview["graph"]["edges"])
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()


def test_global_overview_uses_only_tool_result_identity():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        try:
            trajectory = db.create_trajectory("Nested tool trajectory", agent_id="codex")
            db.append_event(trajectory["trajectory_id"], "tool_call", {
                "tool_name": "exec",
                "command": "const r = await tools.web__run({search_query: []});",
            })
            db.append_event(trajectory["trajectory_id"], "tool_call", {
                "tool_name": "exec",
                "command": "rg --files examples",
            })
            db.append_event(trajectory["trajectory_id"], "tool_result", {
                "tool_name": "web__run",
                "tool_chain": ["exec", "web__run"],
                "status": "ok",
                "command": "const r = await tools.web__run({search_query: []});",
            })
            db.append_event(trajectory["trajectory_id"], "tool_result", {
                "tool_name": "rg",
                "tool_chain": ["exec", "rg"],
                "status": "ok",
                "command": "rg --files examples",
            })

            overview = db.global_overview()
            row = overview["trajectories"][0]
            assert row["tools"] == [
                {"tool_name": "rg", "call_count": 1},
                {"tool_name": "web__run", "call_count": 1},
            ]
            tool_nodes = {node["label"]: node for node in overview["graph"]["nodes"] if node["type"] == "tool"}
            assert tool_nodes["web__run"]["detail"] == "1 results"
            assert "tool:exec" not in {edge["target"] for edge in overview["graph"]["edges"]}
            assert "tool:web__run" in {edge["target"] for edge in overview["graph"]["edges"]}
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
