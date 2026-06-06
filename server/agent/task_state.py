from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ActiveTask:
    id: str
    skill_name: str
    state: dict[str, Any]
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(cls, skill_name: str, state: dict[str, Any] | None = None) -> ActiveTask:
        return cls(id=str(uuid.uuid4()), skill_name=skill_name, state=state or {})

    def update(self, state: dict[str, Any], *, status: str | None = None) -> None:
        self.state = state
        if status:
            self.status = status
        self.updated_at = datetime.now(UTC)
