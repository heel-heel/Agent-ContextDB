import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

from contextdb.llm_judge import SemanticRepairJudge
from contextdb.service import ContextDB
from contextdb import uris


def test_learned_skills_reuses_cached_semantic_judgments():
    previous_provider = os.environ.get("CONTEXTDB_LLM_PROVIDER")
    os.environ["CONTEXTDB_LLM_PROVIDER"] = "mock"
    try:
        with TemporaryDirectory() as root:
            db = ContextDB(root)
            try:
                trajectory_id = db.create_trajectory("semantic judgment cache")["trajectory_id"]
                db.append_event(trajectory_id, "tool_call", {"tool_name": "shell", "command": "read missing-settings.json"})
                db.append_event(
                    trajectory_id,
                    "tool_result",
                    {"status": "failed", "preview": "file not found"},
                    actor="tool",
                )
                db.append_event(trajectory_id, "tool_call", {"tool_name": "shell", "command": "read project-settings.json"})
                db.append_event(
                    trajectory_id,
                    "tool_result",
                    {"status": "ok", "preview": "settings loaded"},
                    actor="tool",
                )

                original_judge = SemanticRepairJudge.judge
                calls = 0

                def counted_judge(self, failure, candidate):
                    nonlocal calls
                    calls += 1
                    return original_judge(self, failure, candidate)

                with patch.object(SemanticRepairJudge, "judge", counted_judge):
                    first = db.semantic_repair_judgments(trajectory_id)
                    assert first
                    first_call_count = calls
                    assert first_call_count == len(first)

                    second = db.semantic_repair_judgments(trajectory_id)
                    assert second == first
                    assert calls == first_call_count

                    db.learned_skills(trajectory_id)
                    assert calls == first_call_count

                judge = SemanticRepairJudge()
                profile_identity = judge.profile_id or f"{judge.provider}:{judge.model or 'default'}"
                assert db.store.get_object(uris.semantic_judgments_view_key(trajectory_id, profile_identity))
            finally:
                db.vector_index.conn.close()
                db.store.conn.close()
    finally:
        if previous_provider is None:
            os.environ.pop("CONTEXTDB_LLM_PROVIDER", None)
        else:
            os.environ["CONTEXTDB_LLM_PROVIDER"] = previous_provider
