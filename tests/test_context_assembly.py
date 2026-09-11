import os
from tempfile import TemporaryDirectory

from contextdb.service import ContextDB


def test_context_assembly_materializes_incremental_semantic_summary():
    old_provider = os.environ.get("CONTEXTDB_LLM_PROVIDER")
    old_profile = os.environ.get("CONTEXTDB_LLM_PROFILE")
    os.environ["CONTEXTDB_LLM_PROVIDER"] = "mock"
    os.environ.pop("CONTEXTDB_LLM_PROFILE", None)
    try:
        with TemporaryDirectory() as root:
            db = ContextDB(root)
            try:
                trajectory_id = db.create_trajectory("context assembly test")["trajectory_id"]
                db.append_event(trajectory_id, "user_message", {"text": "Investigate a build failure and keep the environment stable."}, actor="user")
                db.append_event(trajectory_id, "memory_update", {"fact": "Prefer the existing compiler toolchain before changing dependencies."})
                for index in range(8):
                    db.append_event(
                        trajectory_id,
                        "tool_result" if index % 2 else "assistant_message",
                        {"status": "ok", "preview": f"step {index} completed"} if index % 2 else {"text": f"Continue diagnostic step {index}."},
                        actor="tool" if index % 2 else "agent",
                    )

                assembly = db.query_view(trajectory_id, "current_prompt", token_budget=500)["content"]
                assert assembly["schema_version"] == "context_assembly.v1"
                assert assembly["selected_tokens"] <= 500
                assert assembly["max_selectable_tokens"] >= assembly["selected_tokens"]
                assert [segment["priority"] for segment in assembly["segments"]] == ["P0", "P1", "P2", "P3", "P4", "P5"]
                assert any(segment["segment_id"] == "retrieved_memory" for segment in assembly["segments"])

                summary = db.query_view(trajectory_id, "summary")["content"]
                assert summary["schema_version"] == "semantic_summary.v1"
                assert summary["status"] == "ready"
                assert summary["method"] == "incremental_llm_semantic_summary"
                assert summary["scope"]["source_event_count"] == 2
                cached = db.query_view(trajectory_id, "summary")["content"]
                assert cached["materialization"]["cache_hit"] is True
            finally:
                db.vector_index.conn.close()
                db.store.conn.close()
    finally:
        if old_provider is None:
            os.environ.pop("CONTEXTDB_LLM_PROVIDER", None)
        else:
            os.environ["CONTEXTDB_LLM_PROVIDER"] = old_provider
        if old_profile is None:
            os.environ.pop("CONTEXTDB_LLM_PROFILE", None)
        else:
            os.environ["CONTEXTDB_LLM_PROFILE"] = old_profile
