import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from contextdb.service import ContextDB


def test_semantic_summaries_are_materialized_per_llm_profile():
    previous_path = os.environ.get("CONTEXTDB_LLM_PROFILES_PATH")
    previous_provider = os.environ.get("CONTEXTDB_LLM_PROVIDER")
    with TemporaryDirectory() as root:
        config_path = Path(root) / "profiles.json"
        config_path.write_text(json.dumps({
            "default_profile": "mock-one",
            "profiles": {
                "mock-one": {"provider": "mock", "model": "mock-one"},
                "mock-two": {"provider": "mock", "model": "mock-two"},
            },
        }), encoding="utf-8")
        os.environ["CONTEXTDB_LLM_PROFILES_PATH"] = str(config_path)
        os.environ.pop("CONTEXTDB_LLM_PROVIDER", None)
        db = ContextDB(Path(root) / "data")
        try:
            trajectory_id = db.create_trajectory("profile-scoped summary")['trajectory_id']
            for index in range(10):
                db.append_event(
                    trajectory_id,
                    "assistant_message",
                    {"text": f"completed diagnostic step {index}"},
                )

            first = db.query_view(trajectory_id, "summary", profile_id="mock-one")["content"]
            second = db.query_view(trajectory_id, "summary", profile_id="mock-two")["content"]
            cached_first = db.query_view(trajectory_id, "summary", profile_id="mock-one")["content"]

            assert first["llm"]["profile_id"] == "mock-one"
            assert second["llm"]["profile_id"] == "mock-two"
            assert first["materialization"]["cache_hit"] is False
            assert second["materialization"]["cache_hit"] is False
            assert cached_first["materialization"]["cache_hit"] is True
            assert len(db.store.list_objects(f"trajectories/{trajectory_id}/views/main/summary_profiles")) == 2
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
    if previous_path is None:
        os.environ.pop("CONTEXTDB_LLM_PROFILES_PATH", None)
    else:
        os.environ["CONTEXTDB_LLM_PROFILES_PATH"] = previous_path
    if previous_provider is None:
        os.environ.pop("CONTEXTDB_LLM_PROVIDER", None)
    else:
        os.environ["CONTEXTDB_LLM_PROVIDER"] = previous_provider
