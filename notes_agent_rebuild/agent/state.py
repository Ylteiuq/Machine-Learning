from dataclasses import dataclass, field
from enum import Enum


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    LIMIT = "limit"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class Session:
    id: str
    title: str
    messages: list[dict] = field(default_factory=list)
    kind: str = "main"
    summary: str = ""
    summarized_until: int = 0
    active_run_id: str | None = None
    workspace: str = ""


@dataclass
class RunState:
    id: str
    session_id: str
    parent_run_id: str | None = None
    call_id: str | None = None
    phase: str = "model"
    status: RunStatus = RunStatus.RUNNING
    model_calls: int = 0
    max_steps: int = 6
    allowed_tools: list[str] | None = None
    pending_tools: list[dict] = field(default_factory=list)
    error: str = ""
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    session_id: str
    call_id: str
