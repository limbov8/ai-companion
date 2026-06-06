from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentMode(StrEnum):
    ANSWER = "answer"
    ASK_CLARIFYING = "ask_clarifying"
    USE_TOOL = "use_tool"
    START_SKILL = "start_skill"
    CONTINUE_SKILL = "continue_skill"


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class AgentDecision:
    mode: AgentMode
    intent: str
    confidence: float
    required_slots: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    skill_name: str | None = None
    response_style: str | None = None
    reason: str | None = None

    @classmethod
    def answer(cls, *, intent: str = "conversation", reason: str | None = None) -> AgentDecision:
        return cls(mode=AgentMode.ANSWER, intent=intent, confidence=0.75, reason=reason)
