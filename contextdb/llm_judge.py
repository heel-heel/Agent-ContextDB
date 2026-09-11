from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List
from urllib import error, request

from .llm_profiles import resolve_profile


LABELS = {"likely_repair", "partial_repair", "validation_only", "unrelated_success", "insufficient_context", "not_evaluated"}
STRUCTURAL_REPAIR_RULES = {
    "branch_from_failure_first_success",
    "branch_from_failure_ancestor_first_success",
    "rollback_then_repair_first_success",
    "same_branch_first_success_after_failure",
}

class SemanticRepairJudge:
    """Optional semantic judge for failure -> repair candidates.

    Default mode is disabled so ContextDB demos remain deterministic and offline.
    Set CONTEXTDB_LLM_PROVIDER=qwen and DASHSCOPE_API_KEY to call Alibaba Cloud Bailian/Qwen through its OpenAI-compatible API.
    Set CONTEXTDB_LLM_PROVIDER=mock for an offline deterministic approximation.
    """

    def __init__(self, provider: str | None = None, model: str | None = None, profile_id: str | None = None) -> None:
        self.profile_id = profile_id or os.environ.get("CONTEXTDB_LLM_PROFILE") or ""
        self._api_key = ""
        self._base_url = ""
        self._profile_error = ""
        self.supports_json_response_format = True
        # Reuse the configured default profile for every LLM task.  Keeping the
        # profile identity even when its endpoint is incomplete lets the caller
        # report the exact missing configuration instead of falling back to an
        # unrelated public endpoint.
        if not self.profile_id and provider is None and model is None and not os.environ.get("CONTEXTDB_LLM_PROVIDER"):
            try:
                default_profile = resolve_profile()
                if default_profile["api_key"]:
                    self.profile_id = default_profile["profile_id"]
            except ValueError:
                pass
        if self.profile_id:
            try:
                profile = resolve_profile(self.profile_id)
                self.profile_id = profile["profile_id"]
                self.provider = profile["provider"]
                self.model = (model or profile["model"]).strip()
                self._api_key = profile["api_key"]
                self._base_url = profile["base_url"]
                self.supports_json_response_format = profile["supports_json_response_format"]
            except ValueError as exc:
                self.provider = "disabled"
                self.model = model or ""
                self._profile_error = str(exc)
        else:
            configured_provider = provider if provider is not None else os.environ.get("CONTEXTDB_LLM_PROVIDER")
            has_api_key = bool(
                os.environ.get("CONTEXTDB_LLM_API_KEY")
                or os.environ.get("DASHSCOPE_API_KEY")
                or os.environ.get("ALIYUN_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            self.provider = (configured_provider if configured_provider is not None else ("qwen" if has_api_key else "disabled")).strip().lower()
            self.model = (model or os.environ.get("CONTEXTDB_LLM_MODEL", "qwen3.7-max")).strip()
        self.timeout = float(os.environ.get("CONTEXTDB_LLM_TIMEOUT", "60"))

    def enabled(self) -> bool:
        return self.provider in {"qwen", "bailian", "openai-compatible", "mock"}

    def _retry_window_seconds(self) -> float:
        try:
            return max(0.0, min(float(os.environ.get("CONTEXTDB_LLM_RETRY_WINDOW_SECONDS", "60")), 300.0))
        except ValueError:
            return 60.0

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

    def group_recommended_actions(self, trigger: Dict[str, Any], support_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_group_result(support_events)
        if self.provider == "mock":
            return self._mock_group_result(trigger, support_events)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            return self._openai_compatible_group_result(trigger, support_events)
        result = self._disabled_group_result(support_events)
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result

    def group_failure_patterns(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_failure_group_result(failures)
        if self.provider == "mock":
            return self._mock_failure_group_result(failures)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            return self._openai_compatible_failure_group_result(failures)
        result = self._disabled_failure_group_result(failures)
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result

    def match_skill(self, failure: Dict[str, Any], skills: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_skill_match_result(failure)
        if self.provider == "mock":
            return self._mock_skill_match_result(failure, skills, top_k)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            return self._openai_compatible_skill_match_result(failure, skills, top_k)
        result = self._disabled_skill_match_result(failure)
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result


    def translate_contextql(self, question: str, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Translate a natural-language exploration request into one read-only ContextQL statement."""
        if self.provider in {"", "disabled", "off", "none"}:
            result = {"enabled": False, "provider": self.provider or "disabled", "model": self.model, "sql": "", "reason": "LLM SQL translation is disabled."}
            return result
        if self.provider == "mock":
            return {"enabled": True, "provider": "mock", "model": "deterministic-mock", "profile_id": self.profile_id or None, "sql": "SELECT event_id, branch_id, event_type, status, preview FROM events WHERE trajectory_id = :trajectory_id ORDER BY timestamp;", "reason": "Mock translator returned the default event query.", "execution": {"profile_id": self.profile_id or None, "provider": "mock", "requested_model": self.model, "response_model": "deterministic-mock", "response_id": None, "verified": self.model == "deterministic-mock"}}
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            result = self._openai_json_call(
                "You translate natural-language questions into one safe read-only ContextQL SQL statement. Return strict JSON only.",
                self._contextql_translation_prompt(question, relations),
                {"enabled": False, "provider": self.provider, "model": self.model, "sql": "", "reason": "LLM SQL translation failed."},
                "OpenAI-compatible LLM ContextQL translation failed.",
            )
            if result.get("error"):
                return result
            sql = str(result.get("sql") or "").strip()
            execution = result.get("_contextdb_execution", {})
            return {"enabled": True, "provider": self.provider, "model": self.model, "profile_id": self.profile_id or None, "sql": sql, "reason": str(result.get("reason") or "")[:800], "execution": execution}
        return {"enabled": False, "provider": self.provider, "model": self.model, "sql": "", "reason": "Unsupported LLM provider.", "error": f"unsupported provider: {self.provider}"}

    def classify_likely_cause(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a reusable failure cause with the configured LLM, not keyword rules."""
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_likely_cause_result()
        if self.provider == "mock":
            return self._mock_likely_cause_result()
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            result = self._openai_json_call(
                "You classify the root cause of one agent failure for a reusable experience database. Return strict JSON only.",
                self._likely_cause_prompt(failure),
                self._disabled_likely_cause_result(),
                "OpenAI-compatible LLM likely-cause classification failed.",
            )
            if result.get("error"):
                return result
            return self._normalize_likely_cause_result(result)
        result = self._disabled_likely_cause_result()
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result

    def summarize_trajectory_context(
        self,
        events: List[Dict[str, Any]],
        previous_summary: str = "",
    ) -> Dict[str, Any]:
        """Create a factual semantic digest for a completed trajectory prefix.

        The caller owns incremental state and caching.  This method deliberately
        has no rule-based fallback: an unavailable model is surfaced as such so
        a UI never labels a line-concatenation result as an LLM summary.
        """
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_context_summary_result()
        if self.provider == "mock":
            return self._mock_context_summary_result(events, previous_summary)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            result = self._openai_json_call(
                "You produce factual, compact semantic summaries of completed agent trajectories. Return strict JSON only.",
                self._context_summary_prompt(events, previous_summary),
                self._disabled_context_summary_result(),
                "OpenAI-compatible LLM semantic context summarization failed.",
            )
            if result.get("error"):
                return result
            normalized = self._normalize_context_summary_result(result)
            normalized["execution"] = result.get("_contextdb_execution", {})
            return normalized
        result = self._disabled_context_summary_result()
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result

    def _disabled_context_summary_result(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider or "disabled",
            "model": self.model,
            "summary": "",
            "key_facts": [],
            "open_items": [],
            "reason": "Configure a ContextDB LLM profile to generate a semantic trajectory summary.",
            "error": "llm_semantic_summary_disabled",
        }

    def _mock_context_summary_result(self, events: List[Dict[str, Any]], previous_summary: str) -> Dict[str, Any]:
        # Kept solely for deterministic tests. Production semantic summaries use
        # an explicitly configured LLM profile.
        lines = []
        if previous_summary:
            lines.append(previous_summary)
        for event in events[-4:]:
            payload = event.get("payload", {})
            text = payload.get("text") or payload.get("command") or payload.get("preview") or ""
            if text:
                lines.append(f"{event.get('event_type')}: {str(text)[:180]}")
        return {
            "enabled": True,
            "provider": "mock",
            "model": "deterministic-mock",
            "summary": "\n".join(f"- {line}" for line in lines)[:3000],
            "key_facts": [],
            "open_items": [],
            "reason": "Deterministic mock summary for offline tests.",
        }

    def _disabled_likely_cause_result(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider or "disabled",
            "model": self.model,
            "likely_cause": "LLM cause classification unavailable",
            "confidence": 0.0,
            "reason": "Configure CONTEXTDB_LLM_PROVIDER and an API key to classify failure causes.",
        }

    def _mock_likely_cause_result(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "provider": "mock",
            "model": "deterministic-mock",
            "likely_cause": "mock LLM cause classification",
            "confidence": 0.0,
            "reason": "Mock mode does not use keyword rules to infer a failure cause.",
        }

    def annotate_tool_result(self, action: str, observation: str, adapter_status: str = "unknown", tool_name: str = "") -> Dict[str, Any]:
        if self.provider in {"", "disabled", "off", "none"}:
            return self._disabled_tool_result_annotation(adapter_status)
        if self.provider == "mock":
            return self._mock_tool_result_annotation(action, observation, adapter_status, tool_name)
        if self.provider in {"qwen", "bailian", "openai-compatible"}:
            return self._openai_compatible_tool_result_annotation(action, observation, adapter_status, tool_name)
        result = self._disabled_tool_result_annotation(adapter_status)
        result.update({"error": f"unsupported provider: {self.provider}"})
        return result


    def _disabled_tool_result_annotation(self, adapter_status: str) -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider or "disabled",
            "model": self.model,
            "status": adapter_status or "unknown",
            "confidence": 0.0,
            "error_signature": "",
            "reason": "LLM trace annotation is disabled.",
            "evidence": "",
        }

    def _mock_tool_result_annotation(self, action: str, observation: str, adapter_status: str, tool_name: str = "") -> Dict[str, Any]:
        text = str(observation or "")
        lower = text.lower()
        action_lower = str(action or "").lower().strip()
        success_markers = [
            "successfully installed", "finished with status 'done'", "finished with status \"done\"",
            "requirement already satisfied", "file updated", "found 1 matches", "[file:", "collected", "passed",
        ]
        failure_markers = [
            "traceback", "subprocess-exited-with-error", "error:", "command not found",
            "no such file", "permission denied", "assertionerror", "syntaxerror", "importerror:",
            "modulenotfounderror", "indentationerror", "introduced new syntax error", "could not build wheels", "no matching distribution found",
        ]
        if action_lower == "submit":
            status, confidence, reason = "ok", 0.95, "Mock annotator: submit action produced a patch artifact."
        elif "introduced new syntax error" in lower or "errors:" in lower:
            status, confidence, reason = "failed", 0.9, "Mock annotator: editor reported an explicit failed edit."
        elif any(marker in lower for marker in success_markers):
            status, confidence, reason = "ok", 0.9, "Mock annotator: observation contains an explicit success marker."
        elif any(marker in lower for marker in failure_markers):
            status, confidence, reason = "failed", 0.88, "Mock annotator: observation contains a strong failure marker."
        elif adapter_status in {"ok", "failed", "timeout"}:
            status, confidence, reason = adapter_status, 0.62, "Mock annotator: no strong semantic marker; keeping adapter status."
        else:
            status, confidence, reason = "unknown", 0.4, "Mock annotator: insufficient evidence to resolve status."
        signature = ""
        if status == "failed":
            signature = self._extract_error_signature(text)
        return {
            "enabled": True,
            "provider": "mock",
            "model": "deterministic-mock",
            "status": status,
            "confidence": confidence,
            "error_signature": signature,
            "reason": reason,
            "evidence": text[:240],
        }

    def _openai_compatible_tool_result_annotation(self, action: str, observation: str, adapter_status: str, tool_name: str = "") -> Dict[str, Any]:
        result = self._openai_json_call(
            "You annotate an agent tool result during trace normalization. Decide whether the observation shows success, failure, timeout, warning, or unknown. Return strict JSON only.",
            self._tool_result_annotation_prompt(action, observation, adapter_status, tool_name),
            self._disabled_tool_result_annotation(adapter_status),
            "OpenAI-compatible LLM trace annotation failed.",
        )
        if result.get("error"):
            return result
        return self._normalize_tool_result_annotation(result)

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
        elif candidate.get("outcome") == "ok" and rule in STRUCTURAL_REPAIR_RULES:
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

    def _disabled_group_result(self, support_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider or "disabled",
            "model": self.model,
            "actions": [],
            "reason": "LLM action grouping is disabled. Set CONTEXTDB_LLM_PROVIDER=qwen and DASHSCOPE_API_KEY to enable semantic action grouping.",
            "error": "llm_grouping_disabled" if support_events else "no_support_events",
        }

    def _mock_group_result(self, trigger: Dict[str, Any], support_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, Dict[str, Any]] = {}
        for event in support_events:
            command = str(event.get("command") or "")
            lower = command.lower()
            if "cc=clang" in lower or " clang" in lower:
                key = "use_clang_for_native_build"
                name = "Use clang for native dependency builds"
                template = "CC=clang cargo build --release"
                strategy = "Switch the native build to clang when gcc fails on generated native dependency code."
            elif "docker" in lower:
                key = "use_docker_isolated_build"
                name = "Use Docker for isolated service build"
                template = command
                strategy = "Use Docker to isolate compiler and dependency versions."
            elif "pip install" in lower and ("prefer-binary" in lower or "pyarrow" in lower):
                key = "use_binary_wheels_for_python_dependencies"
                name = "Install Python dependencies from binary wheels"
                template = "pip install --prefer-binary <packages>"
                strategy = "Prefer prebuilt wheels or pinned wheel versions when source builds time out in constrained agent environments."
            else:
                key = "apply_successful_action_" + str(event.get("success_event_id") or len(groups))
                name = "Apply successful repair action"
                template = command
                strategy = str(event.get("strategy") or "Repeat the successful repair action.")
            group = groups.setdefault(key, {
                "action_id": "act_" + key,
                "name": name,
                "canonical_tool": event.get("tool"),
                "canonical_command_template": template,
                "strategy": strategy,
                "support_event_ids": [],
                "confidence": 0.72,
                "reason": "Mock action grouping clusters semantically equivalent repair actions for offline demos.",
            })
            if event.get("success_event_id") not in group["support_event_ids"]:
                group["support_event_ids"].append(event.get("success_event_id"))
        return {
            "enabled": True,
            "provider": "mock",
            "model": "deterministic-mock",
            "actions": list(groups.values()),
            "reason": "Mock LLM grouping completed.",
        }

    def _disabled_failure_group_result(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"enabled": False, "provider": self.provider or "disabled", "model": self.model, "groups": [], "reason": "LLM failure grouping is disabled.", "error": "llm_failure_grouping_disabled" if failures else "no_failures"}

    def _mock_failure_group_result(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, Dict[str, Any]] = {}
        for failure in failures:
            text = " ".join(str(failure.get(k) or "") for k in ("likely_cause", "normalized_signature", "command", "error_signature")).lower()
            if "gcc" in text or "compiler" in text or "clang" in text or "cargo" in text:
                key = "compiler_native_dependency_incompatibility"
                name = "Repair compiler or native dependency incompatibility"
            elif "pip" in text or "pyarrow" in text or "wheel" in text:
                key = "python_dependency_binary_wheel_source_build_incompatibility"
                name = "Repair python dependency binary wheel or source-build incompatibility"
            else:
                key = _slug(text or "agent_failure")
                name = "Repair " + str(failure.get("likely_cause") or "agent task failure")
            group = groups.setdefault(key, {"skill_id": "skill_" + key, "name": name, "failure_event_ids": [], "confidence": 0.82, "reason": "Mock failure grouping clusters equivalent failure modes for offline demos."})
            if failure.get("failure_event_id") not in group["failure_event_ids"]:
                group["failure_event_ids"].append(failure.get("failure_event_id"))
        return {"enabled": True, "provider": "mock", "model": "deterministic-mock", "groups": list(groups.values()), "reason": "Mock LLM failure grouping completed."}

    def _disabled_skill_match_result(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        return {"enabled": False, "provider": self.provider or "disabled", "model": self.model, "matches": [], "reason": "LLM skill matching is disabled.", "error": "llm_skill_matching_disabled"}

    def _mock_skill_match_result(self, failure: Dict[str, Any], skills: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        q = " ".join(str(failure.get(k) or "") for k in ("tool", "command", "error_signature", "normalized_signature")).lower()
        rows = []
        for skill in skills:
            trigger = skill.get("trigger", {})
            t = " ".join(str(trigger.get(k) or "") for k in ("failed_tool", "failed_command_pattern", "normalized_signature", "error_signature", "likely_cause")).lower()
            score = 0.0
            evidence = []
            if "gcc" in q and "gcc" in t:
                score += 0.55; evidence.append("same compiler failure mode")
            if "pip" in q and "pip" in t:
                score += 0.55; evidence.append("same python dependency failure mode")
            if str(failure.get("tool") or "") == str(trigger.get("failed_tool") or ""):
                score += 0.2; evidence.append("same failed tool")
            if str(failure.get("command") or "") == str(trigger.get("failed_command_pattern") or ""):
                score += 0.2; evidence.append("same failed command")
            if score > 0:
                rows.append({"skill_id": skill.get("skill_id"), "score": round(min(score, 1.0), 3), "reason": "; ".join(evidence)})
        rows.sort(key=lambda item: item.get("score", 0), reverse=True)
        return {"enabled": True, "provider": "mock", "model": "deterministic-mock", "matches": rows[:top_k], "reason": "Mock LLM skill matching completed."}

    def _openai_compatible_result(self, failure: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        result = self._openai_json_call(
            "You judge whether a successful agent action semantically repaired a previous failure. Return strict JSON only.",
            self._prompt(failure, candidate),
            self._disabled_result(),
            "OpenAI-compatible LLM API call failed.",
        )
        if result.get("error"):
            return result
        normalized = self._normalize_result(result, provider=self.provider)
        normalized["execution"] = result.get("_contextdb_execution", {})
        return normalized

    def _openai_compatible_group_result(self, trigger: Dict[str, Any], support_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = self._openai_json_call(
            "You cluster successful agent repair events into reusable recommended actions. Return strict JSON only.",
            self._group_prompt(trigger, support_events),
            self._disabled_group_result(support_events),
            "OpenAI-compatible LLM action grouping failed.",
        )
        if result.get("error"):
            return result
        normalized = self._normalize_group_result(result, support_events, provider=self.provider)
        normalized["execution"] = result.get("_contextdb_execution", {})
        return normalized

    def _openai_compatible_failure_group_result(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = self._openai_json_call("You group agent failure patterns into reusable skill triggers. Return strict JSON only.", self._failure_group_prompt(failures), self._disabled_failure_group_result(failures), "OpenAI-compatible LLM failure grouping failed.")
        if result.get("error"):
            return result
        return self._normalize_failure_group_result(result, failures)

    def _openai_compatible_skill_match_result(self, failure: Dict[str, Any], skills: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        result = self._openai_json_call("You match a new agent failure against a materialized skill library. Return strict JSON only.", self._skill_match_prompt(failure, skills, top_k), self._disabled_skill_match_result(failure), "OpenAI-compatible LLM skill matching failed.")
        if result.get("error"):
            return result
        return self._normalize_skill_match_result(result, skills, top_k)

    def _openai_json_call(self, system: str, prompt: str, disabled_result: Dict[str, Any], failure_reason: str) -> Dict[str, Any]:
        if self._profile_error:
            result = dict(disabled_result)
            result.update({"enabled": False, "provider": self.provider, "model": self.model, "error": self._profile_error})
            return result
        api_key = self._api_key or (os.environ.get("CONTEXTDB_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIYUN_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            result = dict(disabled_result)
            result.update({"provider": self.provider, "model": self.model, "error": "No API key is configured for the selected LLM profile"})
            return result
        if self.profile_id and not self._base_url:
            result = dict(disabled_result)
            result.update({
                "enabled": False,
                "provider": self.provider,
                "model": self.model,
                "error": (
                    f"LLM profile '{self.profile_id}' is not ready: "
                    "CONTEXTDB_LLM_BASE_URL is not configured."
                ),
                "reason": "Configure the Bailian workspace endpoint, then restart the ContextDB dashboard service.",
            })
            return result
        base_url = (self._base_url or os.environ.get("CONTEXTDB_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0}
        if self.supports_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        req = request.Request(base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
        retry_window = self._retry_window_seconds()
        deadline = time.monotonic() + retry_window
        attempt = 0
        retry_delay = 0.5

        while True:
            attempt += 1
            try:
                remaining = deadline - time.monotonic()
                request_timeout = min(self.timeout, max(1.0, remaining))
                with request.urlopen(req, timeout=request_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parsed = _parse_json_object(data["choices"][0]["message"]["content"])
                response_model = str(data.get("model") or "").strip()
                parsed["_contextdb_execution"] = {
                    "profile_id": self.profile_id or None,
                    "provider": self.provider,
                    "requested_model": self.model,
                    "response_model": response_model or None,
                    "response_id": str(data.get("id") or "") or None,
                    "verified": bool(response_model and response_model == self.model),
                }
                return parsed
            except (error.HTTPError, error.URLError, TimeoutError, OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
                retryable = isinstance(exc, (error.URLError, TimeoutError, OSError)) or (
                    isinstance(exc, error.HTTPError) and exc.code >= 500
                )
                remaining = deadline - time.monotonic()
                if retryable and remaining > 0:
                    time.sleep(min(retry_delay, remaining))
                    retry_delay = min(retry_delay * 2, 5.0)
                    continue

                detail = str(exc) or exc.__class__.__name__
                connection_refused = (
                    isinstance(exc, ConnectionRefusedError)
                    or getattr(exc, "winerror", None) == 10061
                    or getattr(getattr(exc, "reason", None), "winerror", None) == 10061
                    or isinstance(getattr(exc, "reason", None), ConnectionRefusedError)
                )
                if connection_refused:
                    detail = (
                        f"The connection to the configured LLM endpoint was refused after {attempt} attempt(s) within {retry_window:g} seconds: {detail}. "
                        "Check VPN, firewall or network egress rules, and any required HTTPS proxy."
                    )
                elif isinstance(exc, OSError):
                    detail = (
                        f"The remote LLM endpoint closed or reset the connection after {attempt} attempt(s) within {retry_window:g} seconds: {detail}. "
                        "Verify CONTEXTDB_LLM_BASE_URL for the selected Bailian workspace and any proxy/firewall settings."
                    )
                result = dict(disabled_result)
                result.update({"enabled": True, "provider": self.provider, "model": self.model, "error": detail, "reason": failure_reason, "_contextdb_execution": {"profile_id": self.profile_id or None, "provider": self.provider, "requested_model": self.model, "response_model": None, "response_id": None, "verified": False}})
                return result

    def _contextql_translation_prompt(self, question: str, relations: List[Dict[str, Any]]) -> str:
        compact = {
            "task": "Translate the user question into exactly one read-only ContextQL statement.",
            "instructions": [
                "Use only the listed relations and columns.",
                "Return exactly one SELECT, WITH ... SELECT, or EXPLAIN SELECT statement.",
                "Never use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, ATTACH, or multiple statements.",
                "Use :trajectory_id to scope trajectory data unless querying only the trajectories relation.",
                "For events.status use only: ok, failed, error, timeout, warning, unknown.",
                "Return strict JSON only.",
            ],
            "output_schema": {"sql": "one valid ContextQL statement", "reason": "brief explanation"},
            "question": str(question or "")[:4000],
            "relations": relations,
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _likely_cause_prompt(self, failure: Dict[str, Any]) -> str:
        compact = {
            "task": "Classify the root failure cause into one concise, reusable lower-case phrase.",
            "instructions": [
                "Use only the provided failure context.",
                "Do not use the literal category unknown.",
                "Prefer a concrete technical cause such as source-code syntax or indentation error, dependency resolution failure, test behavior mismatch, or network authentication failure.",
                "Return strict JSON only.",
            ],
            "output_schema": {
                "likely_cause": "concise reusable root-cause phrase, maximum 12 words",
                "confidence": "number between 0 and 1",
                "reason": "short evidence-based explanation",
            },
            "failure": failure,
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _context_summary_prompt(self, events: List[Dict[str, Any]], previous_summary: str) -> str:
        compact_events = []
        for event in events:
            payload = event.get("payload", {}) or {}
            text = (
                payload.get("text")
                or payload.get("command")
                or payload.get("preview")
                or payload.get("summary")
                or payload.get("error_signature")
                or ""
            )
            compact_events.append({
                "event_id": event.get("event_id"),
                "branch_id": event.get("branch_id"),
                "event_type": event.get("event_type"),
                "actor": event.get("actor"),
                "status": payload.get("status"),
                "text": str(text)[:1200],
            })
        compact = {
            "task": "Create an incremental semantic digest of a completed prefix of an agent trajectory.",
            "instructions": [
                "Use only the supplied previous summary and source events.",
                "Preserve concrete task intent, completed actions, tool outcomes, failures, repairs, version transitions, and unresolved work.",
                "Do not invent facts, commands, or future plans.",
                "Do not repeat the recent live context; these events are an earlier completed prefix.",
                "Keep the summary concise enough for an agent context window.",
                "Return strict JSON only.",
            ],
            "output_schema": {
                "summary": "factual compact prose or bullets",
                "key_facts": ["important durable facts"],
                "open_items": ["unresolved items, if any"],
            },
            "previous_summary": str(previous_summary or "")[:5000],
            "new_source_events": compact_events,
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _normalize_context_summary_result(self, value: Dict[str, Any]) -> Dict[str, Any]:
        summary = re.sub(r"\s+", " ", str(value.get("summary") or "").strip())[:6000]
        key_facts = [re.sub(r"\s+", " ", str(item).strip())[:500] for item in value.get("key_facts", []) if str(item).strip()][:16]
        open_items = [re.sub(r"\s+", " ", str(item).strip())[:500] for item in value.get("open_items", []) if str(item).strip()][:16]
        return {
            "enabled": True,
            "provider": self.provider,
            "model": self.model,
            "summary": summary,
            "key_facts": key_facts,
            "open_items": open_items,
            "reason": "LLM generated a factual semantic trajectory digest.",
        }

    def _normalize_likely_cause_result(self, value: Dict[str, Any]) -> Dict[str, Any]:
        try:
            confidence = float(value.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        cause = re.sub(r"\s+", " ", str(value.get("likely_cause") or "").strip())[:240]
        if not cause:
            cause = "LLM cause classification unavailable"
        return {
            "enabled": True,
            "provider": self.provider,
            "model": self.model,
            "likely_cause": cause,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(value.get("reason") or "")[:800],
        }

    def _tool_result_annotation_prompt(self, action: str, observation: str, adapter_status: str, tool_name: str = "") -> str:
        compact = {
            "task": "Annotate one tool result produced while normalizing an agent trajectory.",
            "instructions": [
                "Use only the provided action and observation.",
                "Do not mark a result failed merely because a dependency/package name contains words like exception or error.",
                "Prefer ok when the log says installation/build/check finished successfully or requirements were already satisfied.",
                "Use failed only for explicit runtime errors, tracebacks, failing tests, missing commands/files, permission errors, build failures, or nonzero-exit evidence.",
                "Return strict JSON only.",
            ],
            "output_schema": {
                "status": "one of ok, failed, timeout, warning, unknown",
                "confidence": "number between 0 and 1",
                "error_signature": "short failure signature, empty if not failed",
                "reason": "short explanation",
                "evidence": "short quoted/paraphrased evidence from observation",
            },
            "tool_name": tool_name,
            "action": action,
            "adapter_status": adapter_status,
            "observation_preview": str(observation or "")[:4000],
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _normalize_tool_result_annotation(self, value: Dict[str, Any]) -> Dict[str, Any]:
        status = str(value.get("status") or "unknown").lower().strip()
        if status not in {"ok", "failed", "timeout", "warning", "unknown"}:
            status = "unknown"
        try:
            confidence = float(value.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "enabled": True,
            "provider": self.provider,
            "model": self.model,
            "status": status,
            "confidence": max(0.0, min(1.0, confidence)),
            "error_signature": str(value.get("error_signature") or "")[:500],
            "reason": str(value.get("reason") or "")[:800],
            "evidence": str(value.get("evidence") or "")[:800],
        }

    def _extract_error_signature(self, observation: str) -> str:
        for line in str(observation or "").splitlines():
            lower = line.lower()
            if any(marker in lower for marker in ("traceback", "error:", "exception:", "failed", "failure", "assertionerror", "syntaxerror", "importerror:", "modulenotfounderror")):
                return line.strip()[:500]
        return str(observation or "").strip().splitlines()[-1][:500] if str(observation or "").strip() else ""

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

    def _group_prompt(self, trigger: Dict[str, Any], support_events: List[Dict[str, Any]]) -> str:
        compact = {
            "task": "Cluster support events into deduplicated reusable recommended actions for one learned agent skill.",
            "instructions": [
                "Use only the provided JSON.",
                "Group events together when they express the same repair strategy, even if commands differ slightly.",
                "Do not group events that solve the failure through clearly different strategies.",
                "Every support_event_id should appear in exactly one action unless it is genuinely not a reusable recommendation.",
                "Name each action and write a concise canonical command template when applicable.",
                "Return strict JSON only.",
            ],
            "output_schema": {
                "actions": [
                    {
                        "action_id": "short stable snake_case id without act_ prefix",
                        "name": "human-readable recommended action name",
                        "canonical_tool": "tool name",
                        "canonical_command_template": "canonical command or empty string",
                        "strategy": "general reusable strategy",
                        "support_event_ids": ["success event ids assigned to this action"],
                        "confidence": "number between 0 and 1",
                        "reason": "why these support events belong together",
                    }
                ]
            },
            "trigger": trigger,
            "support_events": support_events,
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _failure_group_prompt(self, failures: List[Dict[str, Any]]) -> str:
        compact = {"task": "Deduplicate failure patterns into reusable skill triggers.", "instructions": ["Group failures that share the same root failure mode and can share repair skills.", "Do not merge failures merely because they have the same tool if root causes differ.", "Every failure_event_id should appear in exactly one group.", "Return strict JSON only."], "output_schema": {"groups": [{"skill_id": "stable snake_case id without skill_ prefix", "name": "human-readable skill name", "failure_event_ids": ["ids"], "confidence": "0..1", "reason": "why grouped"}]}, "failures": failures}
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _skill_match_prompt(self, failure: Dict[str, Any], skills: List[Dict[str, Any]], top_k: int) -> str:
        compact_skills = [{"skill_id": s.get("skill_id"), "name": s.get("name"), "trigger": s.get("trigger"), "recommended_actions": [{"action_id": a.get("action_id"), "name": a.get("name"), "strategy": a.get("strategy"), "score": (a.get("confidence") or {}).get("max_llm_confidence")} for a in s.get("recommended_actions", [])]} for s in skills]
        compact = {"task": "Match a new failure to the most relevant learned skills.", "instructions": ["Use only the provided JSON.", "Return no match when no skill trigger really fits the failure.", "Rank matches by semantic fit to the failure trigger.", "Return strict JSON only."], "top_k": top_k, "output_schema": {"matches": [{"skill_id": "existing skill_id", "score": "0..1", "reason": "why this skill matches"}]}, "failure": failure, "skills": compact_skills}
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _normalize_failure_group_result(self, value: Dict[str, Any], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_ids = {str(f.get("failure_event_id")) for f in failures if f.get("failure_event_id")}
        seen = set(); groups = []
        for i, item in enumerate(value.get("groups", []) if isinstance(value.get("groups", []), list) else []):
            if not isinstance(item, dict):
                continue
            ids = []
            for fid in item.get("failure_event_ids", []):
                fid = str(fid)
                if fid in valid_ids and fid not in seen:
                    ids.append(fid); seen.add(fid)
            if not ids:
                continue
            try: confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError): confidence = 0.0
            raw_id = str(item.get("skill_id") or item.get("name") or f"failure_group_{i+1}")
            skill_id = raw_id if raw_id.startswith("skill_") else "skill_" + _slug(raw_id)
            groups.append({"skill_id": skill_id, "name": str(item.get("name") or skill_id.replace("_", " "))[:180], "failure_event_ids": ids, "confidence": max(0.0, min(1.0, confidence)), "reason": str(item.get("reason") or "")[:800]})
        return {"enabled": True, "provider": self.provider, "model": self.model, "groups": groups, "unassigned_failure_event_ids": sorted(valid_ids - seen), "reason": str(value.get("reason", "LLM failure grouping completed."))[:800]}

    def _normalize_skill_match_result(self, value: Dict[str, Any], skills: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        valid_ids = {s.get("skill_id") for s in skills if s.get("skill_id")}
        matches = []
        for item in value.get("matches", []) if isinstance(value.get("matches", []), list) else []:
            if not isinstance(item, dict):
                continue
            sid = item.get("skill_id")
            if sid not in valid_ids:
                continue
            try: score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError): score = 0.0
            if score <= 0:
                continue
            matches.append({"skill_id": sid, "score": round(max(0.0, min(1.0, score)), 3), "reason": str(item.get("reason") or "")[:800]})
        matches.sort(key=lambda item: item.get("score", 0), reverse=True)
        return {"enabled": True, "provider": self.provider, "model": self.model, "matches": matches[:top_k], "reason": str(value.get("reason", "LLM skill matching completed."))[:800]}

    def _normalize_group_result(self, value: Dict[str, Any], support_events: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
        valid_ids = {str(event.get("success_event_id")) for event in support_events if event.get("success_event_id")}
        actions = []
        seen_ids = set()
        raw_actions = value.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                continue
            support_ids = []
            for event_id in item.get("support_event_ids", []):
                event_id = str(event_id)
                if event_id in valid_ids and event_id not in seen_ids:
                    support_ids.append(event_id)
                    seen_ids.add(event_id)
            if not support_ids:
                continue
            action_id = _slug(str(item.get("action_id") or item.get("name") or f"action_{index + 1}"))
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            actions.append({
                "action_id": action_id,
                "name": str(item.get("name") or action_id.replace("_", " "))[:160],
                "canonical_tool": str(item.get("canonical_tool") or "")[:80],
                "canonical_command_template": str(item.get("canonical_command_template") or "")[:300],
                "strategy": str(item.get("strategy") or "")[:800],
                "support_event_ids": support_ids,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(item.get("reason") or "")[:800],
            })
        missing_ids = sorted(valid_ids - seen_ids)
        return {
            "enabled": True,
            "provider": provider,
            "model": self.model,
            "actions": actions,
            "unassigned_support_event_ids": missing_ids,
            "reason": str(value.get("reason", "LLM action grouping completed."))[:800],
        }

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


def _slug(text: str) -> str:
    tokens = [part.strip("-_") for part in re.split(r"[^A-Za-z0-9_+-]+", text.lower()) if len(part.strip("-_")) > 1]
    return "_".join(tokens[:10] or ["agent_action"])
