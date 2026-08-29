from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_profiles.json"


def _profile_path() -> Path:
    configured = os.environ.get("CONTEXTDB_LLM_PROFILES_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_PROFILE_PATH


def load_profiles() -> Dict[str, Any]:
    path = _profile_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"LLM profile configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LLM profile configuration: {exc}") from exc
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("LLM profile configuration must contain a non-empty profiles object")
    return value


def resolve_profile(profile_id: str | None = None) -> Dict[str, Any]:
    config = load_profiles()
    selected = str(profile_id or os.environ.get("CONTEXTDB_LLM_PROFILE") or config.get("default_profile") or "").strip()
    profile = config["profiles"].get(selected)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown LLM profile: {selected}")
    api_key_env = str(profile.get("api_key_env") or "DASHSCOPE_API_KEY")
    base_url_env = str(profile.get("base_url_env") or "CONTEXTDB_LLM_BASE_URL")
    api_key = os.environ.get(api_key_env, "").strip()
    base_url = (os.environ.get(base_url_env) or profile.get("base_url") or "").strip().rstrip("/")
    return {
        "profile_id": selected,
        "label": str(profile.get("label") or selected),
        "provider": str(profile.get("provider") or "openai-compatible").strip().lower(),
        "model": str(profile.get("model") or "").strip(),
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
        "api_key": api_key,
        "base_url": base_url,
        "supports_json_response_format": bool(profile.get("supports_json_response_format", True)),
    }


def public_profiles() -> Dict[str, Any]:
    config = load_profiles()
    rows: List[Dict[str, Any]] = []
    for profile_id, profile in config["profiles"].items():
        resolved = resolve_profile(profile_id)
        rows.append({
            "profile_id": profile_id,
            "label": resolved["label"],
            "provider": resolved["provider"],
            "model": resolved["model"],
            "api_key_env": resolved["api_key_env"],
            "base_url_env": resolved["base_url_env"],
            "ready": bool(resolved["api_key"] and resolved["base_url"]),
        })
    return {"schema_version": "contextdb.llm_profiles.v1", "default_profile": config.get("default_profile"), "profiles": rows}
