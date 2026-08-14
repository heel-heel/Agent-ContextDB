from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from .base import NormalizedEvent


class SWEAgentTrajectoryAdapter:
    """Adapter for official SWE-agent .traj files.

    The adapter performs deterministic schema normalization. When llm_annotate
    is enabled, it adds an LLM-assisted semantic annotation pass for tool_result
    status and error signatures while preserving adapter/raw evidence metadata.
    """

    source_name = "swe-agent-traj"

    def __init__(self, llm_annotate: bool | None = None, source_name: str | None = None) -> None:
        if llm_annotate is None:
            llm_annotate = os.environ.get("CONTEXTDB_TRACE_LLM_ANNOTATE", "").strip().lower() in {"1", "true", "yes", "on"}
        self.llm_annotate = bool(llm_annotate)
        if source_name:
            self.source_name = source_name

    def load(self, path: str | Path) -> Dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("SWE-agent .traj must be a JSON object")
        trajectory = data.get("trajectory")
        if not isinstance(trajectory, list):
            raise ValueError("SWE-agent .traj must contain a trajectory list")
        return data

    def iter_events(self, raw_trace: Dict[str, Any]) -> Iterable[NormalizedEvent]:
        yield from self._initial_context(raw_trace)
        annotator = self._annotator() if self.llm_annotate else None
        for step_index, step in enumerate(raw_trace.get("trajectory", []), start=1):
            if not isinstance(step, dict):
                continue
            refs = {"swe_step": step_index}
            state = self._state(step.get("state"))
            response = self._text(step.get("response") or step.get("thought"))
            thought = self._text(step.get("thought"))
            if response:
                payload = {"text": response}
                if thought and thought != response:
                    payload["thought"] = thought
                yield NormalizedEvent(
                    "assistant_message",
                    payload,
                    actor="agent",
                    refs=refs,
                    metadata={"source_step_field": "response", **state},
                )
            action = self._text(step.get("action"))
            tool_name = self._tool_name(action)
            if action:
                yield NormalizedEvent(
                    "tool_call",
                    {"tool_name": tool_name, "command": action},
                    actor="agent",
                    refs=refs,
                    metadata={"action_kind": self._action_kind(action), **state},
                )
            observation = self._text(step.get("observation"))
            if observation or action:
                adapter_status = self._status_from_observation(action, observation)
                annotation = annotator.annotate_tool_result(action, observation, adapter_status=adapter_status, tool_name=tool_name) if annotator else None
                resolved_status = self._resolve_status(adapter_status, annotation)
                payload = {
                    "status": resolved_status,
                    "exit_code": None,
                    "preview": observation[:500],
                    "raw_preview_truncated": len(observation) > 500,
                }
                if annotation and annotation.get("error_signature"):
                    payload["error_signature"] = annotation.get("error_signature")
                metadata = {"observation_chars": len(observation), "adapter_status": adapter_status, **state}
                if annotation:
                    metadata["llm_trace_annotation"] = annotation
                    metadata["status_resolution"] = "llm_annotation" if resolved_status != adapter_status else "adapter_confirmed_or_low_confidence"
                else:
                    metadata["status_resolution"] = "deterministic_adapter"
                yield NormalizedEvent("tool_result", payload, actor="tool", refs=refs, metadata=metadata)
        info = raw_trace.get("info") if isinstance(raw_trace.get("info"), dict) else {}
        submission = self._text(info.get("submission"))
        if submission:
            yield NormalizedEvent(
                "artifact",
                {"path": "submission.patch", "summary": "SWE-agent submitted patch", "diff": submission},
                actor="agent",
                refs={"swe_info": "submission"},
                metadata={"exit_status": info.get("exit_status")},
            )
        if info:
            yield NormalizedEvent(
                "summary_update",
                {
                    "text": f"SWE-agent trajectory finished with exit_status={info.get('exit_status', 'unknown')}",
                    "model_stats": info.get("model_stats", {}),
                },
                actor="agent",
                refs={"swe_info": "exit_status"},
                metadata={"exit_status": info.get("exit_status")},
            )

    def _annotator(self):
        from ..llm_judge import SemanticRepairJudge
        return SemanticRepairJudge()

    def _resolve_status(self, adapter_status: str, annotation: Dict[str, Any] | None) -> str:
        if not annotation or not annotation.get("enabled"):
            return adapter_status
        try:
            confidence = float(annotation.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        llm_status = str(annotation.get("status") or "unknown").lower()
        if llm_status in {"ok", "failed", "timeout", "warning"} and confidence >= 0.7:
            return llm_status
        return adapter_status

    def _initial_context(self, raw_trace: Dict[str, Any]) -> Iterable[NormalizedEvent]:
        metadata = {
            "environment": raw_trace.get("environment"),
            "benchmark": "SWE-bench",
            "trace_format": "SWE-agent .traj",
            "llm_trace_annotation": self.llm_annotate,
        }
        yield NormalizedEvent(
            "system_event",
            {
                "text": "Imported official SWE-agent trajectory for a SWE-bench software-engineering task.",
                "source": "SWE-agent official GitHub trajectory demonstration",
            },
            actor="system",
            refs={"source_format": self.source_name},
            metadata=metadata,
        )
        history = raw_trace.get("history") if isinstance(raw_trace.get("history"), list) else []
        first_user = next((item for item in history if isinstance(item, dict) and item.get("role") == "user" and item.get("content")), None)
        if first_user:
            yield NormalizedEvent(
                "user_message",
                {"text": self._text(first_user.get("content"))[:2000]},
                actor="user",
                refs={"swe_history": "first_user"},
                metadata=metadata,
            )

    def _tool_name(self, action: str) -> str:
        head = action.strip().split(maxsplit=1)[0] if action.strip() else "unknown"
        if head in {"open", "goto", "scroll_down", "scroll_up", "find_file", "search_dir", "search_file"}:
            return "swe-agent-editor"
        if head in {"edit", "create"}:
            return "swe-agent-file-edit"
        if head in {"submit"}:
            return "swe-agent-submit"
        return "shell"

    def _action_kind(self, action: str) -> str:
        head = action.strip().split(maxsplit=1)[0] if action.strip() else "unknown"
        if head in {"edit", "create"}:
            return "file_edit"
        if head in {"open", "goto", "scroll_down", "scroll_up", "find_file", "search_dir", "search_file"}:
            return "navigation"
        if head == "submit":
            return "submit"
        return "shell"

    def _status_from_observation(self, action: str, observation: str) -> str:
        text = observation.lower()
        action_text = action.lower().strip()
        if action_text == "submit":
            return "ok"
        if "introduced new syntax error" in text or "errors:" in text:
            return "failed"
        success_markers = ["successfully installed", "finished with status 'done'", "requirement already satisfied", "file updated", "found 1 matches", "[file:"]
        if any(marker in text for marker in success_markers):
            return "ok"
        failure_markers = [
            "traceback", "subprocess-exited-with-error", "error:", "command not found",
            "no such file", "permission denied", "assertionerror", "syntaxerror", "importerror:",
            "modulenotfounderror", "indentationerror", "could not build wheels", "no matching distribution found",
        ]
        if any(marker in text for marker in failure_markers):
            return "failed"
        return "ok"

    def _state(self, value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return {"swe_state": value}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"swe_state_raw": value[:500]}
            if isinstance(parsed, dict):
                return {"swe_state": parsed}
        return {"swe_state_raw": str(value)[:500]}

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip("\n")


class SWEAgentLLMAnnotatedTrajectoryAdapter(SWEAgentTrajectoryAdapter):
    source_name = "swe-agent-traj-llm"

    def __init__(self) -> None:
        super().__init__(llm_annotate=True, source_name=self.source_name)
