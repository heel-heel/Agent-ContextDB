from http.client import RemoteDisconnected
from unittest.mock import patch
from urllib import error

from contextdb.llm_judge import SemanticRepairJudge, _parse_json_object


def test_background_llm_configuration_does_not_replace_explicit_profile(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_API_KEY", "background-test-key")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_SUPPORTS_JSON_RESPONSE_FORMAT", "false")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashboard-test-key")

    background = SemanticRepairJudge()
    selected = SemanticRepairJudge(profile_id="qwen3.8-max")

    assert background.profile_id == "background-deepseek-v4-pro"
    assert background.provider == "bailian"
    assert background.model == "deepseek-v4-pro"
    assert background._api_key == "background-test-key"
    assert background._base_url == "https://example.invalid/v1"
    assert background.supports_json_response_format is False
    assert background.timeout == 60.0
    assert background._retry_window_seconds() == 60.0
    assert selected.profile_id == "qwen3.8-max"
    assert selected.model == "qwen3.8-max"
    assert selected._api_key == "dashboard-test-key"


def test_background_llm_transport_policy_can_be_configured_independently(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_API_KEY", "background-test-key")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_TIMEOUT", "45")
    monkeypatch.setenv("CONTEXTDB_BACKGROUND_LLM_RETRY_WINDOW_SECONDS", "75")

    judge = SemanticRepairJudge()

    assert judge.timeout == 45.0
    assert judge._retry_window_seconds() == 75.0


def test_remote_disconnect_is_reported_as_llm_diagnostic(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CONTEXTDB_LLM_RETRY_WINDOW_SECONDS", "0")
    judge = SemanticRepairJudge()

    with patch("contextdb.llm_judge.request.urlopen", side_effect=RemoteDisconnected("closed")):
        result = judge._openai_json_call("system", "prompt", {"enabled": False}, "Semantic judge failed.")

    assert result["enabled"] is True
    assert "remote LLM endpoint closed or reset" in result["error"]
    assert "CONTEXTDB_LLM_BASE_URL" in result["error"]


def test_incomplete_default_profile_reports_missing_endpoint(monkeypatch):
    monkeypatch.delenv("CONTEXTDB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CONTEXTDB_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("CONTEXTDB_LLM_BASE_URL", raising=False)
    judge = SemanticRepairJudge()

    result = judge._openai_json_call("system", "prompt", {"enabled": False}, "Semantic judge failed.")

    assert result["enabled"] is False
    assert result["error"] == "LLM profile 'deepseek-v4.1-flash' is not ready: CONTEXTDB_LLM_BASE_URL is not configured."


def test_refused_connection_has_network_diagnostic(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CONTEXTDB_LLM_RETRY_WINDOW_SECONDS", "0")
    judge = SemanticRepairJudge()

    with patch("contextdb.llm_judge.request.urlopen", side_effect=error.URLError(ConnectionRefusedError(10061, "refused"))):
        result = judge._openai_json_call("system", "prompt", {"enabled": False}, "Semantic judge failed.")

    assert "could not be reached after 3 attempts" in result["error"]


def test_http_authentication_errors_are_not_retried(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    judge = SemanticRepairJudge()

    response = error.HTTPError("https://example.invalid/v1/chat/completions", 401, "Unauthorized", None, None)
    with patch("contextdb.llm_judge.request.urlopen", side_effect=response) as call:
        result = judge._openai_json_call("system", "prompt", {"enabled": False}, "Semantic judge failed.")

    assert call.call_count == 1
    assert result["error"] == "HTTP 401: authentication failed. Check the API key configured for the selected LLM profile."


def test_unbounded_digest_wait_stops_after_three_connection_failures(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    judge = SemanticRepairJudge()

    failure = error.URLError(ConnectionRefusedError(10061, "refused"))
    with patch("contextdb.llm_judge.request.urlopen", side_effect=failure) as call, patch("contextdb.llm_judge.time.sleep"):
        result = judge._openai_json_call("system", "prompt", {"enabled": False}, "Semantic judge failed.", wait_indefinitely=True)

    assert call.call_count == 3
    assert "could not be reached after 3 attempts" in result["error"]


def test_parser_recovers_an_invalid_backslash_escape_in_model_json():
    parsed = _parse_json_object(r'{"summary":"Use \q flag", "key_facts":[], "open_items":[]}')

    assert parsed["summary"] == r"Use \q flag"


def test_default_transport_policy_uses_one_minute_retry_window(monkeypatch):
    monkeypatch.setenv("CONTEXTDB_LLM_PROVIDER", "bailian")
    monkeypatch.setenv("CONTEXTDB_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CONTEXTDB_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("CONTEXTDB_LLM_RETRY_WINDOW_SECONDS", raising=False)
    judge = SemanticRepairJudge()

    assert judge._retry_window_seconds() == 60.0
