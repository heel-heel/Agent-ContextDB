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
            assert overview["skills"][0]["detail"] == skill
            assert {
                "source": f"skill:{source['trajectory_id']}:skill_fix_settings",
                "target": f"trajectory:{consumer['trajectory_id']}",
                "kind": "skill_usage",
                "count": 1,
            } in overview["graph"]["edges"]
            consumer_row = next(row for row in overview["trajectories"] if row["trajectory_id"] == consumer["trajectory_id"])
            assert consumer_row["tools"] == [{"tool_name": "get-content", "call_count": 1}]
            assert consumer_row["branch_count"] == 1
            assert consumer_row["associated_skills"][0]["source_trajectory_id"] == source["trajectory_id"]
            assert not any(edge["kind"] == "uses_skill" for edge in overview["graph"]["edges"])
            assert any(edge["kind"] == "skill_action" for edge in overview["graph"]["edges"])
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()


def test_global_overview_falls_back_to_parent_tool_call_and_shows_produced_skills():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        try:
            trajectory = db.create_trajectory("Imported trajectory", agent_id="swe-agent")
            skill = {
                "skill_id": "skill_imported_repair",
                "name": "Repair imported failure",
                "status": "candidate",
                "trigger": {"failed_tool": "swe-agent-editor"},
                "confidence": {"support_count": 1},
            }
            db.vector_index.replace_owner("skills", trajectory["trajectory_id"], [{
                "entry_id": trajectory["trajectory_id"] + ":skill_imported_repair",
                "document": "imported repair",
                "metadata": {"trajectory_id": trajectory["trajectory_id"], "skill": skill},
            }])
            call = db.append_event(trajectory["trajectory_id"], "tool_call", {
                "tool_name": "swe-agent-editor", "command": "open setup.py",
            })
            db.append_event(trajectory["trajectory_id"], "tool_result", {
                "status": "ok", "preview": "opened setup.py",
            }, parent_event_ids=[call["event_id"]])

            overview = db.global_overview()
            row = overview["trajectories"][0]
            assert row["tools"] == [{"tool_name": "swe-agent-editor", "call_count": 1}]
            assert {
                "source": f"skill:{trajectory['trajectory_id']}:skill_imported_repair",
                "target": "tool:swe-agent-editor",
                "kind": "skill_trigger",
                "count": 1,
            } in overview["graph"]["edges"]
            assert {
                "source": f"skill:{trajectory['trajectory_id']}:skill_imported_repair",
                "target": f"trajectory:{trajectory['trajectory_id']}",
                "kind": "skill_usage",
                "count": 1,
            } in overview["graph"]["edges"]
            assert row["associated_skills"] == [{
                "node_id": f"skill:{trajectory['trajectory_id']}:skill_imported_repair",
                "skill_id": "skill_imported_repair",
                "name": "Repair imported failure",
                "source_trajectory_id": trajectory["trajectory_id"],
                "trigger_tool": "swe-agent-editor",
                "support_count": 1,
                "status": "candidate",
                "relationship": "learned",
            }]
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


def test_global_overview_lists_native_tools_before_contextdb_tools():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        try:
            trajectory = db.create_trajectory("Tool ordering")
            db.append_event(trajectory["trajectory_id"], "tool_result", {
                "tool_name": "contextdb_get_version_status", "status": "ok",
            })
            db.append_event(trajectory["trajectory_id"], "tool_result", {
                "tool_name": "git", "status": "ok",
            })

            overview = db.global_overview()
            assert overview["trajectories"][0]["tools"] == [
                {"tool_name": "git", "call_count": 1},
                {"tool_name": "contextdb_get_version_status", "call_count": 1},
            ]
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
