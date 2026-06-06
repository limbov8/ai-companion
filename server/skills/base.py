from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from server.agent.decision import ToolCall


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    when_to_use: str
    required_slots: list[str] = field(default_factory=list)


@dataclass
class SkillState:
    skill_name: str
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "active"


@dataclass
class SkillStepResult:
    status: str
    question: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    updated_state: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None


@dataclass(frozen=True)
class SkillContext:
    user_text: str
    history: list[dict[str, str]]
    memories: list[object]
    active_task: object | None = None


class Skill(Protocol):
    @property
    def spec(self) -> SkillSpec:
        ...

    async def start(self, ctx: SkillContext) -> SkillStepResult:
        ...

    async def step(self, ctx: SkillContext, state: SkillState, user_text: str) -> SkillStepResult:
        ...
