from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass
class NormalizedEvent:
    event_type: str
    payload: Dict[str, Any]
    actor: str = "agent"
    branch_id: str = "main"
    parent_event_ids: Optional[List[str]] = None
    refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


class TraceAdapter(Protocol):
    source_name: str

    def load(self, path: str | Path) -> Any:
        ...

    def iter_events(self, raw_trace: Any) -> Iterable[NormalizedEvent]:
        ...
