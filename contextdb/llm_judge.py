from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List
from urllib import error, request


LABELS = {"likely_repair", "partial_repair", "validation_only", "unrelated_success", "insufficient_context", "not_evaluated"}


class SemanticRepairJudge:
    """Optional semantic judge for failure -> repair candidates.

    Default mode is disabled so ContextDB demos remain deterministic and offline.
    Set CONTEXTDB_LLM_PROVIDER=qwen and DASHSCOPE_API_KEY to call Alibaba Cloud Bailian/Qwen through its OpenAI-compatible API.
    Set CONTEXTDB_LLM_PROVIDER=mock for an offline deterministic approximation.
    """

    def __init__(self) -> None:
        self.provider = os.environ.get("CONTEXTDB_LLM_PROVIDER", "disabled").strip().lower()
        self.model = os.environ.get("CONTEXTDB_LLM_MODEL", "qwen3.7-max").strip()
        self.timeout = float(os.environ.get("CONTEXTDB_LLM_TIMEOUT", "20"))

    def enabled(self) -> bool:
        return self.provider in {"qwen", "bailian", "openai-compatible", "mock"}

    def judge(self, failure: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_result()
        if self.provider == "mock":
            return self._mock_result(failure, candidate)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            return self._openai_compatible_result(failure, candidate)
        result = self._disabled_result()
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result

    def _disabled_result(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider or "disabled",
            "model": self.model,
            "label": "not_evaluated",
            "confidence": 0.0,
            "reason": "LLM judge is disabled. Set CONTEXTDB_LLM_PROVIDER=qwen and DASHSCOPE_API_KEY to enable Bailian/Qwen semantic judging.",
            "risk": "No semantic judgment was performed.",
            "recommended_for_skill": False,
        }

    def _mock_result(self, failure: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        failed = str(failure.get("command") or "")
        command = str(candidate.get("command") or "")
        same_tokens = set(_tokens(failed)) & set(_tokens(command))
        rule = candidate.get("primary_rule") or candidate.get("link_type") or ""
        if same_tokens and candidate.get("outcome") == "ok":
            label, confidence = "likely_repair", 0.78
            reason = "Mock judge: the candidate succeeded and shares command tokens with the failed action."
        elif candidate.get("outcome") == "ok" and "repair" in str(candidate.get("why_linked", "")).lower():
            label, confidence = "partial_repair", 0.62
            reason = "Mock judge: structural evidence suggests a repair, but command similarity is weak."
        else:
            label, confidence = "insufficient_context", 0.35
            reason = "Mock judge: there is not enough semantic evidence beyond the structural rule."
        return {
            "enabled": True,
            "provider": "mock",
            "model": "deterministic-mock",
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "risk": f"Primary structural rule: {rule}.",
            "recommended_for_skill": label in {"likely_repair", "partial_repair"},
        }

    def _openai_compatible_result(self, failure: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        api_key = (
            os.environ.get("CONTEXTDB_LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("ALIYUN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            result = self._disabled_result()
            result.update({"provider": self.provider, "error": "CONTEXTDB_LLM_API_KEY or DASHSCOPE_API_KEY is not set"})
            return result
        base_url = os.environ.get("CONTEXTDB_LLM_BASE_URL", "https://ws-5jkepnkdq4vt4m5c.cn-beijing.maas.aliyuncs.com/compatible-mode/v1").rstrip("/")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You judge whether a successful agent action semantically repaired a previous failure. Return strict JSON only."},
                {"role": "user", "content": self._prompt(failure, candidate)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_object(text)
            return self._normalize_result(parsed, provider=self.provider)
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            result = self._disabled_result()
            result.update({"enabled": True, "provider": self.provider, "error": str(exc), "reason": "OpenAI-compatible LLM API call failed."})
            return result

    def _prompt(self, failure: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        compact = {
            "task": "Judge whether the successful agent action likely repaired the previous failure.",
            "instructions": [
                "Use only the provided JSON.",
                "Do not assume external facts.",
                "Return strict JSON only.",
                "Labels must be one of: likely_repair, partial_repair, validation_only, unrelated_success, insufficient_context.",
            ],
            "output_schema": {
                "label": "string",
                "confidence": "number between 0 and 1",
                "reason": "short explanation",
                "risk": "short caveat",
                "recommended_for_skill": "boolean",
            },
            "failure": failure,
            "repair_candidate": candidate,
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _normalize_result(self, value: Dict[str, Any], provider: str) -> Dict[str, Any]:
        label = str(value.get("label", "insufficient_context"))
        if label not in LABELS:
            label = "insufficient_context"
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        recommended = bool(value.get("recommended_for_skill", label in {"likely_repair", "partial_repair"}))
        return {
            "enabled": True,
            "provider": provider,
            "model": self.model,
            "label": label,
            "confidence": confidence,
            "reason": str(value.get("reason", ""))[:800],
            "risk": str(value.get("risk", ""))[:500],
            "recommended_for_skill": recommended,
        }


def _tokens(text: str) -> List[str]:
    return [part for part in re.split(r"[^A-Za-z0-9_+-]+", text.lower()) if len(part) > 2]


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)
