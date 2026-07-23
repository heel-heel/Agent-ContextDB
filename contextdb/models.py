from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class Agent:
    agent_id: str
    framework: str
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass
class Source:
    source_id: str
    kind: str
    location: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass
class Trajectory:
    trajectory_id: str
    title: str
    agent_id: str = "unknown-agent"
    source_id: str = "unknown-source"
    default_branch: str = "main"
    head_event_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEvent:
    event_id: str
    trajectory_id: str
    branch_id: str
    event_type: str
    payload: Dict[str, Any]
    parent_event_ids: List[str] = field(default_factory=list)
    actor: str = "agent"
    timestamp: str = field(default_factory=utc_now)
    refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    branch_id: str
    trajectory_id: str
    base_event_id: Optional[str] = None
    head_event_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Snapshot:
    snapshot_id: str
    trajectory_id: str
    branch_id: str
    event_id: Optional[str]
    message: str
    object_keys: List[str]
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class View:
    view_name: str
    trajectory_id: str
    branch_id: str
    content: Any
    source_events: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    artifact_id: str
    trajectory_id: str
    kind: str
    content_ref: str
    preview: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
