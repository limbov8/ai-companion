from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from server.agent.task_state import ActiveTask


@dataclass
class ConversationTurn:
    role: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConversationSession:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    speaking_generation: int = 0
    active_task: ActiveTask | None = None

    def add(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))

    def messages(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    def replace_messages(self, messages: list[dict[str, str]]) -> None:
        self.turns = [
            ConversationTurn(role=message["role"], content=message["content"])
            for message in messages
            if message.get("role") and message.get("content")
        ]

    def barge_in(self) -> int:
        self.speaking_generation += 1
        return self.speaking_generation
